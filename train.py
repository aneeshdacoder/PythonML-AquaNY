"""Stage 3 - train one model set per (parameter, horizon).

Per target/horizon:
  * Regressor predicts the ANOMALY (value - REGION seasonal normal) on the model scale
    (log1p for skewed parameters). Candidates are compared with time-series CV.
  * Split-conformal prediction interval (nominal INTERVAL_COVERAGE) from a calibration slice.
  * For parameters with a screening threshold: a calibrated classifier giving
    P(exceedance) at the horizon.
  * Honest evaluation on two holdouts, both against persistence AND climatology baselines:
      known   stations seen in training, future period
      new     stations NEVER seen in training, future period (spatial generalisation)
  * Deployment model is then refit on all labelled data from all stations.
"""
import argparse
import json
import logging
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

import config
import modeling
import utils
from climatology import Climatology

log = logging.getLogger("train")


def _subsample(df, n, seed=config.SEED):
    return df if len(df) <= n else df.sample(n, random_state=seed).sort_values("date")


def _fit_climatology(daily, target, groups, mask):
    d = daily.loc[mask, ["station_id", "date", target]].dropna()
    d = d.assign(y=utils.fwd(d[target].to_numpy(), target))
    return Climatology().fit(d[["station_id", "date", "y"]], groups)


def _fit_regressor(name, X, y, what=""):
    """Fit the chosen candidate; if it cannot be fitted on this data, fall back to persistence
    rather than crashing the whole run (and say so)."""
    try:
        return modeling.candidate_models()[name].fit(X, y), name
    except Exception as exc:
        log.warning("%s: %r could not be fitted (%s); using persistence instead", what, name, str(exc)[:80])
        return modeling.BaselineModel("persistence").fit(X, y), "persistence"


def _fit_classifier(X, flags):
    n_pos, n_neg = int(flags.sum()), int((~flags).sum())
    if min(n_pos, n_neg) < config.MIN_CLASS_ROWS:
        return None
    try:   # calibrated probabilities (time-ordered folds; rows are date-sorted)
        clf = CalibratedClassifierCV(modeling.new_classifier(), method="sigmoid",
                                     cv=TimeSeriesSplit(n_splits=3))
        return clf.fit(X, flags.astype(int))
    except Exception:
        try:
            return modeling.new_classifier().fit(X, flags.astype(int))
        except Exception as exc:
            log.warning("exceedance classifier could not be fitted (%s); skipping it", str(exc)[:80])
            return None


def train_one(df, daily, groups, target, horizon, holdout_sites):
    col = f"target_{target}_h{horizon}"
    labelled = df[col].notna()
    if labelled.sum() < config.MIN_TRAIN_ROWS + config.MIN_TEST_ROWS:
        return {"status": "skipped", "n_train": int(labelled.sum()), "n_test": 0,
                "reason": "too few labelled rows"}, None, None
    train_mask, test_mask, cutoff = utils.split_masks(
        df, config.TEST_FRACTION, horizon, ref_dates=df.loc[labelled, "date"])
    ho = df["station_id"].isin(holdout_sites)

    tr = _subsample(df[train_mask & labelled & ~ho].sort_values("date"), config.MAX_TRAIN_ROWS)
    tests = {"known": df[test_mask & labelled & ~ho], "new": df[test_mask & labelled & ho]}
    if len(tr) < config.MIN_TRAIN_ROWS or sum(map(len, tests.values())) < config.MIN_TEST_ROWS:
        return {"status": "skipped", "n_train": len(tr),
                "n_test": sum(map(len, tests.values()))}, None, None

    # drop empty AND constant features (a single distinct value breaks histogram binning on small folds)
    base_cols = [c for c in utils.feature_columns(df) if tr[c].nunique(dropna=True) > 1]
    features = base_cols + modeling.ANOM_COLS
    spec = config.THRESHOLDS.get(target)

    # --- evaluation fit (climatology from the training period of training stations only)
    clim = _fit_climatology(daily, target, groups,
                            (daily["date"] < cutoff) & ~daily["station_id"].isin(holdout_sites))
    X_tr, clim_t, clim_site_tr = modeling.make_X(tr, target, horizon, clim, base_cols)
    y_tr = utils.fwd(tr[col].to_numpy(), target) - clim_t
    ok = np.isfinite(y_tr)
    X_tr, y_tr, tr = X_tr[ok].copy(), y_tr[ok], tr[ok]
    off_tr = modeling.oof_site_offset(tr, target, horizon, daily, groups,
                                      (daily["date"] < cutoff) & ~daily["station_id"].isin(holdout_sites))
    X_tr["clim_site_offset"] = off_tr                  # out-of-year: no leakage into training
    clim_site_tr = clim_t[ok] + off_tr             # station normal, out-of-year
    X_tr = modeling.drop_site_offsets(X_tr, tr['station_id'].to_numpy())

    cv = TimeSeriesSplit(n_splits=min(config.CV_SPLITS, max(2, len(tr) // 50)))
    cv_mae = {}
    for name, model in modeling.candidate_models().items():
        with warnings.catch_warnings():              # failed folds are summarised below, not dumped
            warnings.simplefilter("ignore")
            sc = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="neg_mean_absolute_error",
                                 error_score=np.nan)
        if np.isnan(sc).any():
            log.warning("%s_h%d: candidate %r failed on %d of %d CV folds (too little data); "
                        "it is only used if it scores on the rest", target, horizon, name,
                        int(np.isnan(sc).sum()), len(sc))
        cv_mae[name] = float(-np.nanmean(sc)) if np.isfinite(sc).any() else float("inf")
    best = min(cv_mae, key=cv_mae.get)

    # --- conformal interval from out-of-fold residuals across ALL CV folds (so every season
    # --- in the training period contributes, not just the latest slice)
    res = []
    for fit_i, val_i in cv.split(X_tr):
        try:
            m = modeling.candidate_models()[best].fit(X_tr.iloc[fit_i], y_tr[fit_i])
            res.append(np.abs(y_tr[val_i] - m.predict(X_tr.iloc[val_i])))
        except Exception:                      # a fold too small/degenerate to fit: skip it
            continue
    if res:
        res = np.concatenate(res)
    else:                                      # no fold usable: use spread around the regional normal
        log.warning("%s_h%d: interval calibration fell back to a conservative width", target, horizon)
        res = np.abs(y_tr)
    calibrated = len(res) >= 20
    level = min(1.0, config.INTERVAL_COVERAGE * (1 + 1 / max(len(res), 1)))
    q = float(np.quantile(res, level))
    # wider interval for rows that must fall back to the station's seasonal normal
    q_fb = float(np.quantile(np.abs(utils.fwd(tr[col].to_numpy(), target) - clim_site_tr), level))
    q_group = float(np.quantile(np.abs(y_tr), level))  # vs the regional normal only

    model, best = _fit_regressor(best, X_tr, y_tr, f"{target}_h{horizon}")
    flags_tr = utils.exceeds(tr[col].to_numpy(), spec).to_numpy() if spec else None
    clf = _fit_classifier(X_tr, flags_tr) if spec else None

    eval_bundle = {"target": target, "horizon_days": horizon, "model": model, "features": features,
                   "base_cols": base_cols, "climatology": clim, "interval_q": q,
                   "interval_q_fallback": q_fb, "interval_q_group": q_group,
                   "classifier": clf}

    report = {"status": "trained", "best_model": best, "cv_mae_anomaly": cv_mae,
              "n_train": int(len(tr)), "n_features": len(features),
              "interval_calibrated": bool(calibrated), "classifier_fitted": clf is not None,
              "trained_at": datetime.utcnow().isoformat(), "cutoff": str(cutoff.date())}
    holdout_rows = []
    for split, te in tests.items():
        if len(te) < 3:
            continue
        pr = modeling.predict_bundle(eval_bundle, te)
        real = ~pr["fallback"]                      # rows with a recent reading = real forecasts
        if real.sum() < 3:
            continue
        y = te[col].to_numpy()
        persist = te[utils.current_col(target)].to_numpy()
        report[split] = utils.regression_report(
            y[real], pr["point"][real], persist=persist[real], clim=pr["clim"][real],
            lo=pr["lower"][real], hi=pr["upper"][real])
        report[split]["fallback_share"] = float(1 - real.mean())
        if spec and clf is not None:
            report[f"{split}_classifier"] = utils.classification_report(
                utils.exceeds(y[real], spec).to_numpy(), pr["p_exceed"][real])
        holdout_rows.append(pd.DataFrame({
            "station_id": te["station_id"].to_numpy(), "date": te["date"].to_numpy(),
            "parameter": target, "horizon": horizon, "split": split, "y_true": y,
            "y_pred": pr["point"], "lower": pr["lower"], "upper": pr["upper"],
            "p_exceed": pr["p_exceed"], "y_persist": persist, "y_clim": pr["clim"],
            "fallback": pr["fallback"]}))

    # --- an exceedance classifier that does not beat chance on the holdout must not be deployed
    auc = report.get("known_classifier", {}).get("auc")
    if clf is not None and auc is not None and auc < config.MIN_CLASSIFIER_AUC:
        report["classifier_dropped"] = f"holdout AUC {auc:.2f} < {config.MIN_CLASSIFIER_AUC}"
        report["classifier_fitted"] = False
        log.warning("%s_h%d: exceedance classifier dropped (%s)", target, horizon,
                    report["classifier_dropped"])
        clf = None
        eval_bundle["classifier"] = None

    # --- deployment fit on everything labelled (all stations, all dates)
    full = _subsample(df[labelled].sort_values("date"), config.MAX_TRAIN_ROWS)
    clim_f = _fit_climatology(daily, target, groups, pd.Series(True, index=daily.index))
    X_f, clim_tf, _ = modeling.make_X(full, target, horizon, clim_f, base_cols)
    y_f = utils.fwd(full[col].to_numpy(), target) - clim_tf
    okf = np.isfinite(y_f)
    X_f, y_f, full = X_f[okf].copy(), y_f[okf], full[okf]
    X_f["clim_site_offset"] = modeling.oof_site_offset(full, target, horizon, daily, groups,
                                                       pd.Series(True, index=daily.index))
    X_f = modeling.drop_site_offsets(X_f, full['station_id'].to_numpy())
    final, best = _fit_regressor(best, X_f, y_f, f"{target}_h{horizon}")
    clf_f = None
    if spec and clf is not None:
        clf_f = _fit_classifier(X_f, utils.exceeds(full[col].to_numpy(), spec).to_numpy())
    bundle = {**eval_bundle, "model": final, "climatology": clim_f, "classifier": clf_f,
              "model_name": best, "trained_at": report["trained_at"], "n_rows": int(len(full)),
              "holdout": {k: report[k] for k in report if k in
                          ("known", "new", "known_classifier", "new_classifier")}}
    return report, pd.concat(holdout_rows) if holdout_rows else None, bundle


def run(horizons=None, targets=None):
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = utils.load_features()
    daily = utils.load_daily()
    built = utils.load_meta()["horizons"]
    horizons = [h for h in (horizons or built) if h in built]
    groups = df.drop_duplicates("station_id").set_index("station_id")["group"]
    holdout = utils.choose_holdout_sites(df["station_id"].unique(), config.HOLDOUT_SITE_FRACTION,
                                         config.MIN_SITES_FOR_SPATIAL, config.SEED)
    log.info("%d stations; %d held out for the new-station test", df["station_id"].nunique(), len(holdout))

    summary, holdouts = {}, []
    for target in (targets or config.TARGETS):
        for h in horizons:
            key = f"{target}_h{h}"
            report, hp, bundle = train_one(df, daily, groups, target, h, holdout)
            summary[key] = report
            if report["status"] != "trained":
                log.warning("%-26s skipped (train=%d, test=%d)", key, report["n_train"], report["n_test"])
                continue
            k, n = report.get("known", {}), report.get("new", {})
            log.info("%-26s %-12s known MAE=%.3f skill(pers)=%+.2f skill(clim)=%+.2f | new-station MAE=%s",
                     key, report["best_model"], k.get("mae", np.nan),
                     k.get("skill_vs_persistence", np.nan), k.get("skill_vs_climatology", np.nan),
                     f"{n['mae']:.3f}" if "mae" in n else "n/a")
            joblib.dump(bundle, config.MODEL_DIR / f"{key}.joblib")
            if hp is not None:
                holdouts.append(hp)

    with open(config.REPORT_DIR / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    if holdouts:
        pd.concat(holdouts).to_csv(config.REPORT_DIR / "holdout_predictions.csv", index=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--horizons", type=int, nargs="+")
    ap.add_argument("--targets", nargs="+")
    a = ap.parse_args()
    run(a.horizons, a.targets)
