"""Data-availability diagnostics: which parameters actually have data, and where does it end?

  python diagnose.py                       # coverage table + per-target training readiness
  python diagnose.py --wqp-names hardness  # search Water Quality Portal's valid characteristic names
                                           # (needs internet; untested against the live service)
"""
import argparse

import pandas as pd
import requests

import config
import utils


def coverage():
    obs = utils.read_table(config.RAW_DIR, "observations")
    g = obs.groupby("parameter")["timestamp"].agg(["count", "min", "max"])
    g["stations"] = obs.groupby("parameter")["station_id"].nunique()
    g["rows_last_365d"] = obs[obs["timestamp"] >= obs["timestamp"].max() - pd.Timedelta(days=365)] \
        .groupby("parameter")["timestamp"].count()
    g = g.reindex(config.ALL_PARAMS)
    g[["min", "max"]] = g[["min", "max"]].apply(lambda s: s.dt.date)
    print("\n=== Observations by parameter ===")
    print(g.fillna("-").to_string())
    missing = [p for p in config.ALL_PARAMS if p not in set(obs["parameter"])]
    if missing:
        print("\nNo data at all for:", ", ".join(missing))

    if utils.table_exists(config.PROCESSED_DIR, "features"):
        f = utils.load_features()
        rows = []
        for p in config.TARGETS:
            for h in utils.load_meta()["horizons"]:
                lab = f.loc[f[f"target_{p}_h{h}"].notna()]
                rows.append({"target": f"{p}_h{h}", "labelled_rows": len(lab), "stations": lab["station_id"].nunique(),
                             "first": lab["date"].min().date() if len(lab) else None,
                             "last": lab["date"].max().date() if len(lab) else None})
        t = pd.DataFrame(rows)
        t["trainable"] = t["labelled_rows"] >= config.MIN_TRAIN_ROWS + config.MIN_TEST_ROWS
        print("\n=== Training readiness (labelled rows per target) ===")
        print(t.to_string(index=False))


def find_wqp_names(text):
    r = requests.get("https://www.waterqualitydata.us/Codes/characteristicname",
                     params={"mimeType": "json"}, timeout=120)
    r.raise_for_status()
    data = r.json()
    codes = data.get("codes", data) if isinstance(data, dict) else data
    hits = sorted({c.get("value", "") for c in codes if text.lower() in str(c.get("value", "")).lower()})
    print(f"\nWQP characteristic names containing {text!r} ({len(hits)}):")
    for h in hits[:60]:
        print("  ", h)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--wqp-names")
    a = ap.parse_args()
    if a.wqp_names:
        find_wqp_names(a.wqp_names)
    else:
        coverage()
