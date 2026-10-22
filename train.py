"""Stage 3 - train one forecasting model per water-quality target.

For every target: compare Ridge, Random Forest and HistGradientBoosting with
time-series CV on the training period, hold out the most recent period for an
honest test, compare against a persistence baseline, then refit the winner on all
labelled data for deployment.
"""
import argparse
import json
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
import utils

log = logging.getLogger("train")


def candidate_models(log_target: bool) -> dict:
    def wrap(reg):  # skewed concentrations are modelled on a log1p scale
        if not log_target:
            return reg
        return TransformedTargetRegressor(regressor=reg, func=np.log1p, inverse_func=np.expm1)

    return {
        "ridge": wrap(make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0))),
        "random_forest": wrap(make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=config.SEED))),
        "hist_gbm": wrap(HistGradientBoostingRegressor(  # handles NaN natively
            max_iter=300, learning_rate=0.05, l2_regularization=1.0, random_state=config.SEED)),
    }


def run():
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = utils.load_features()
    horizon = utils.load_meta()["horizon_days"]
    train_mask, test_mask, cutoff = utils.split_masks(df, config.TEST_FRACTION, horizon)
    log.info("Holdout starts %s (horizon %d days)", cutoff.date(), horizon)

    all_feats = utils.feature_columns(df)
    summary, holdout_rows = {}, []

    for target in config.TARGETS:
        col = f"target_{target}"
        labelled = df[col].notna()
        tr = df[train_mask & labelled].sort_values("date")
        te = df[test_mask & labelled]
        if len(tr) < config.MIN_TRAIN_ROWS or len(te) < config.MIN_TEST_ROWS:
            log.warning("%-20s skipped (train=%d, test=%d rows)", target, len(tr), len(te))
            summary[target] = {"status": "skipped", "n_train": len(tr), "n_test": len(te)}
            continue

        feats = [c for c in all_feats if tr[c].notna().any()]
        X_tr, y_tr, X_te, y_te = tr[feats], tr[col], te[feats], te[col]

        cv = TimeSeriesSplit(n_splits=min(config.CV_SPLITS, max(2, len(tr) // 50)))
        cv_mae = {}
        for name, model in candidate_models(target in config.LOG_TARGETS).items():
            scores = cross_val_score(model, X_tr, y_tr, cv=cv,
                                     scoring="neg_mean_absolute_error", error_score=np.nan)
            cv_mae[name] = float(-np.nanmean(scores)) if np.isfinite(scores).any() else float("inf")
        best = min(cv_mae, key=cv_mae.get)

        model = candidate_models(target in config.LOG_TARGETS)[best].fit(X_tr, y_tr)
        pred = model.predict(X_te)
        persist = te[target].to_numpy()          # "no change" baseline
        report = utils.regression_report(y_te, pred, persist)
        report.update({"status": "trained", "best_model": best, "cv_mae": cv_mae,
                       "n_train": len(tr), "n_features": len(feats)})
        summary[target] = report
        log.info("%-20s best=%-13s MAE=%.3f  persistence=%.3f  skill=%+.2f", target, best,
                 report["mae"], report.get("persistence_mae", float("nan")),
                 report.get("skill_vs_persistence", float("nan")))

        holdout_rows.append(pd.DataFrame({
            "site_no": te["site_no"].values, "date": te["date"].values, "target": target,
            "y_true": y_te.values, "y_pred": pred, "y_persist": persist}))

        # Refit on every labelled row for the deployed model.
        full = df[labelled]
        final = clone(model).fit(full[feats], full[col])
        joblib.dump({"model": final, "features": feats, "target": target, "model_name": best,
                     "horizon_days": horizon, "holdout_metrics": report},
                    config.MODEL_DIR / f"{target}.joblib")

    with open(config.REPORT_DIR / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    if holdout_rows:
        pd.concat(holdout_rows).to_csv(config.REPORT_DIR / "holdout_predictions.csv", index=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
