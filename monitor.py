"""Stage 6 - monitoring: have the logged forecasts been right since they were issued?

Joins forecast_log.csv (written by predict.py) with the actual observations that have since
arrived, then reports:
  * live error vs the model's holdout error, per parameter/horizon and per station
  * live interval coverage vs nominal
  * stations that stopped reporting
  * model age
and recommends retraining when drift, poor coverage or old models are found.
Schedule it daily, and re-run ingest -> features -> train when it says retrain_recommended.
"""
import argparse
import json
import logging

import numpy as np
import pandas as pd

import config
import utils

log = logging.getLogger("monitor")


def run(as_of=None, window_days=config.MONITOR_WINDOW_DAYS):
    as_of = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.today().normalize()
    daily = utils.load_daily()
    with open(config.REPORT_DIR / "metrics.json") as f:
        metrics = json.load(f)

    report = {"as_of": str(as_of.date()), "retrain_reasons": [], "drift": [], "coverage_alerts": [],
              "stale_stations": [], "old_models": []}

    # ---- data health
    wq = [p for p, s in config.PARAMETERS.items() if s["kind"] != "driver"]   # rain is always "fresh"
    last_obs = daily.dropna(subset=wq, how="all").groupby("station_id")["date"].max()
    stale = last_obs[(as_of - last_obs).dt.days > config.STATION_STALE_DAYS]
    report["stale_stations"] = [{"station_id": s, "last_observation": str(d.date())}
                                for s, d in stale.items()]
    report["n_stations"] = int(len(last_obs))

    # ---- per-parameter coverage: how many stations can actually be forecast right now?
    cov = {}
    for p in config.TARGETS:
        last = daily.dropna(subset=[p]).groupby("station_id")["date"].max()
        limit = config.DISCRETE_FFILL_DAYS if config.PARAMETERS[p]["kind"] == "lab" else config.SENSOR_MAX_AGE_DAYS
        cov[p] = {"stations_ever_measured": int(len(last)),
                  "stations_reporting_now": int(((as_of - last).dt.days <= limit).sum()),
                  "of_stations": report["n_stations"]}
    report["parameter_coverage"] = cov
    report["coverage_warnings"] = [f"{p}: only {c['stations_reporting_now']} of {c['of_stations']} stations report it now"
                                   for p, c in cov.items() if c["stations_reporting_now"] < 0.5 * c["of_stations"]]

    # ---- model age
    for key, r in metrics.items():
        if r.get("status") == "trained":
            age = (as_of - pd.Timestamp(r["trained_at"]).normalize()).days
            if age > config.MODEL_MAX_AGE_DAYS:
                report["old_models"].append({"model": key, "age_days": int(age)})
    if report["old_models"]:
        report["retrain_reasons"].append(f"{len(report['old_models'])} models older than "
                                         f"{config.MODEL_MAX_AGE_DAYS} days")

    # ---- live accuracy
    log_path = config.REPORT_DIR / "forecast_log.csv"
    station_rows = pd.DataFrame()
    if log_path.exists():
        fl = pd.read_csv(log_path, dtype={"station_id": str}, parse_dates=["issued_date", "forecast_date"])
        if "basis" not in fl:
            fl["basis"] = "recent_data"
        fl = fl.drop_duplicates(["station_id", "parameter", "horizon", "issued_date"], keep="last")
        act = (daily[["station_id", "date"] + config.ALL_PARAMS]
               .melt(id_vars=["station_id", "date"], var_name="parameter", value_name="actual").dropna())
        m = fl.merge(act, left_on=["station_id", "parameter", "forecast_date"],
                     right_on=["station_id", "parameter", "date"])
        m = m[(m["forecast_date"] <= as_of) & (m["forecast_date"] >= as_of - pd.Timedelta(days=window_days))]
        m["abs_err"] = (m["prediction"] - m["actual"]).abs()
        m["covered"] = (m["actual"] >= m["lower"]) & (m["actual"] <= m["upper"])
        report["n_matured_fallback_forecasts"] = int((m["basis"] != "recent_data").sum())
        m = m[m["basis"] == "recent_data"]           # fallbacks are not model forecasts
        report["n_matured_forecasts"] = int(len(m))

        def ref_mae(param, h):
            return metrics.get(f"{param}_h{h}", {}).get("known", {}).get("mae")

        for (p, h), g in m.groupby(["parameter", "horizon"]):
            if len(g) < config.MONITOR_MIN_MATURED:
                continue
            ref = ref_mae(p, h)
            live = float(g["abs_err"].mean())
            ratio = live / ref if ref else np.nan
            cover = float(g["covered"].mean())
            if np.isfinite(ratio) and ratio > config.ERROR_RATIO_ALERT:
                report["drift"].append({"parameter": p, "horizon": int(h), "n": int(len(g)),
                                        "live_mae": live, "holdout_mae": ref, "ratio": float(ratio)})
            if cover < config.INTERVAL_COVERAGE - 0.2:
                report["coverage_alerts"].append({"parameter": p, "horizon": int(h),
                                                  "n": int(len(g)), "coverage": cover})
        if report["drift"]:
            report["retrain_reasons"].append(f"error drift in {len(report['drift'])} parameter/horizon models")
        if report["coverage_alerts"]:
            report["retrain_reasons"].append("prediction intervals under-covering in "
                                             f"{len(report['coverage_alerts'])} models")

        station_rows = (m.groupby(["station_id", "parameter", "horizon"])
                        .agg(n=("abs_err", "size"), live_mae=("abs_err", "mean"),
                             coverage=("covered", "mean")).reset_index())
        station_rows["holdout_mae"] = [ref_mae(p, h) for p, h in
                                       zip(station_rows["parameter"], station_rows["horizon"])]
        station_rows["error_ratio"] = station_rows["live_mae"] / station_rows["holdout_mae"]
        station_rows["flag"] = ((station_rows["n"] >= config.MONITOR_MIN_MATURED)
                                & (station_rows["error_ratio"] > 2 * config.ERROR_RATIO_ALERT))
        station_rows.to_csv(config.REPORT_DIR / "monitor_stations.csv", index=False)
        report["flagged_station_series"] = int(station_rows["flag"].sum())
    else:
        report["note"] = "No forecast log yet; run predict.py on a schedule to build one."

    report["retrain_recommended"] = bool(report["retrain_reasons"])
    with open(config.REPORT_DIR / "monitor_report.json", "w") as f:
        json.dump(report, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    print(f"Monitoring as of {report['as_of']}: matured forecasts={report.get('n_matured_forecasts', 0)}, "
          f"stale stations={len(report['stale_stations'])}, drifting models={len(report['drift'])}")
    for w in report["coverage_warnings"]:
        print("  COVERAGE:", w)
    print("RETRAIN RECOMMENDED:" if report["retrain_recommended"] else "No retraining needed.",
          "; ".join(report["retrain_reasons"]))
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--as-of")
    ap.add_argument("--window-days", type=int, default=config.MONITOR_WINDOW_DAYS)
    a = ap.parse_args()
    run(a.as_of, a.window_days)
