"""Stage 5 - forecast every horizon for each station's latest data, and LOG the forecasts so
monitor.py can later compare them with what actually happened."""
import argparse
import logging

import joblib
import numpy as np
import pandas as pd

import config
import modeling
import utils

log = logging.getLogger("predict")
LOG_PATH = config.REPORT_DIR / "forecast_log.csv"


def load_bundles() -> dict:
    bundles = {}
    for path in sorted(config.MODEL_DIR.glob("*.joblib")):
        b = joblib.load(path)
        bundles[(b["target"], b["horizon_days"])] = b
    if not bundles:
        raise SystemExit("No trained models in models/. Run train.py first.")
    return bundles


def forecast_frame(rows: pd.DataFrame, bundles: dict, horizons=None) -> pd.DataFrame:
    frames = []
    for (target, h), b in sorted(bundles.items()):
        if horizons and h not in horizons:
            continue
        out = modeling.predict_bundle(b, rows)
        spec = config.THRESHOLDS.get(target)
        frames.append(pd.DataFrame({
            "issued_date": rows["date"].to_numpy(), "station_id": rows["station_id"].to_numpy(),
            "parameter": target, "horizon": h,
            "forecast_date": rows["date"].to_numpy() + pd.Timedelta(days=h),
            "prediction": out["point"], "lower": out["lower"], "upper": out["upper"],
            "p_exceed": out["p_exceed"],
            "basis": modeling.basis_labels(out),
            "data_age_days": out["age"],
            "exceeds_screening_threshold": utils.exceeds(out["point"], spec).to_numpy() if spec else pd.NA,
            "model": b["model_name"], "model_trained_at": b["trained_at"]}))
    return pd.concat(frames, ignore_index=True)


def run(as_of=None, horizons=None, log_forecasts=True):
    df = utils.load_features()
    if as_of is not None:
        df = df[df["date"] <= pd.Timestamp(as_of)]
    latest = df.sort_values("date").groupby("station_id").tail(1)

    age = utils.station_age_days(latest)
    dead = latest.loc[age.isna() | (age > config.STATION_STALE_DAYS), "station_id"].tolist()
    if dead:
        log.warning("%d station(s) have reported no water-quality data in %d+ days (rain does not "
                    "count): %s", len(dead), config.STATION_STALE_DAYS, ", ".join(dead[:8]) +
                    (" ..." if len(dead) > 8 else ""))

    result = forecast_frame(latest, load_bundles(), horizons)
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(config.REPORT_DIR / "forecast.csv", index=False)
    if log_forecasts:
        result.to_csv(LOG_PATH, mode="a", header=not LOG_PATH.exists(), index=False)

    real = result[result["basis"] == "recent_data"]           # fallbacks are NOT forecasts
    n_st = result["station_id"].nunique()
    print(f"\nStations with a REAL forecast, by parameter (of {n_st} stations):")
    cov = (real.groupby("parameter")["station_id"].nunique().reindex(sorted(result["parameter"].unique()))
           .fillna(0).astype(int))
    never = (result[result["basis"] == "never_measured_fallback"].groupby("parameter")["station_id"].nunique())
    stale = (result[result["basis"] == "stale_data_fallback"].groupby("parameter")["station_id"].nunique())
    print(pd.DataFrame({"real_forecast": cov, "never_measured": never.reindex(cov.index).fillna(0).astype(int),
                        "stale_sensor": stale.reindex(cov.index).fillna(0).astype(int)}).to_string())
    n_fb = int((result["basis"] != "recent_data").sum())
    if n_fb:
        print(f"\nNOTE: {n_fb} of {len(result)} rows ({n_fb / len(result):.0%}) are seasonal-normal fallbacks, not "
              "forecasts (see 'basis' in reports/forecast.csv). They are excluded below.")

    if len(real):
        hs = sorted(real["horizon"].unique())
        h0 = hs[len(hs) // 2]
        print(f"\nReal forecasts at horizon {h0} days (blank = no recent reading):")
        print(real[real["horizon"] == h0].pivot(index="station_id", columns="parameter",
                                                values="prediction").round(2).to_string())
    flagged = real[(real["p_exceed"] > 0.5) | (real["exceeds_screening_threshold"] == True)]  # noqa: E712
    if len(flagged):
        print(f"\nLikely screening-threshold exceedances ({len(flagged)} total; top 20 by probability):")
        print(flagged.sort_values("p_exceed", ascending=False)
              [["station_id", "parameter", "horizon", "prediction", "p_exceed"]]
              .head(20).round(2).to_string(index=False))
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--as-of", help="forecast from the latest data on/before this date")
    ap.add_argument("--horizons", type=int, nargs="+")
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args()
    run(a.as_of, a.horizons, not a.no_log)
