"""Stage 5 - forecast water quality `horizon` days past each site's latest data."""
import argparse
import logging

import joblib
import pandas as pd

import config
import utils

log = logging.getLogger("predict")


def run(max_age_days=30):
    df = utils.load_features()
    latest = df.sort_values("date").groupby("site_no").tail(1)

    age = (pd.Timestamp.today().normalize() - latest["date"]).dt.days
    if (age > max_age_days).any():
        log.warning("%d site(s) have latest data older than %d days; forecasts are stale. "
                    "Re-run collect_data.py first.", int((age > max_age_days).sum()), max_age_days)

    frames = []
    for path in sorted(config.MODEL_DIR.glob("*.joblib")):
        b = joblib.load(path)
        pred = b["model"].predict(latest[b["features"]])
        out = pd.DataFrame({
            "site_no": latest["site_no"].values,
            "data_date": latest["date"].values,
            "forecast_date": latest["date"].values + pd.Timedelta(days=b["horizon_days"]),
            "parameter": b["target"], "prediction": pred, "model": b["model_name"]})
        spec = config.THRESHOLDS.get(b["target"])
        out["exceeds_screening_threshold"] = utils.exceeds(pred, spec).values if spec else pd.NA
        frames.append(out)
    if not frames:
        raise SystemExit("No trained models found in models/. Run train.py first.")

    result = pd.concat(frames, ignore_index=True)
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(config.REPORT_DIR / "forecast.csv", index=False)
    print(result.pivot(index="site_no", columns="parameter", values="prediction").round(2).to_string())
    flagged = result[result["exceeds_screening_threshold"] == True]  # noqa: E712
    if len(flagged):
        print("\nPredicted screening-threshold exceedances:")
        print(flagged[["site_no", "forecast_date", "parameter", "prediction"]].round(2).to_string(index=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-age-days", type=int, default=30)
    run(ap.parse_args().max_age_days)
