"""Stage 2 - clean raw data and engineer features / forecast targets.

Output: data/processed/features.csv (one row per site-day) and meta.json.
Targets are columns named target_<param>: the value observed `horizon` days later.
Lab parameters are sparse, so a target exists only on days when a sample was taken
exactly `horizon` days ahead; training drops the rest per target.
"""
import argparse
import json
import logging

import numpy as np
import pandas as pd

import config

log = logging.getLogger("features")


# ----------------------------------------------------------------- loading ---
def _unit_factor(unit, param):
    """Multiplier converting a reported unit to the parameter's canonical unit."""
    u = str(unit).lower().replace(" ", "").replace("\u00b5", "u")
    canon = config.DISCRETE_PARAMS[param]["unit"]
    if canon == "MPN/100mL":
        return 1.0 if "/100ml" in u else np.nan
    if u.startswith("mg/l"):
        f = 1.0
    elif u.startswith("ug/l"):
        f = 1e-3
    elif u.startswith("ng/l"):
        f = 1e-6
    else:
        return np.nan
    if param == "nitrate" and "no3" in u:   # "as NO3" -> "as N"
        f *= 14.007 / 62.004
    if canon == "ug/L":
        f *= 1000.0
    return f


def clean_discrete(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["site_no", "date", "parameter", "value"])
    df = df.copy()
    df["parameter"] = df["characteristic"].map(config.CHAR_TO_PARAM)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")  # non-detects/blank -> NaN, dropped
    df = df.dropna(subset=["parameter", "value"])
    pairs = df[["unit", "parameter"]].drop_duplicates()
    pairs["factor"] = [_unit_factor(u, p) for u, p in zip(pairs["unit"], pairs["parameter"])]
    df = df.merge(pairs, on=["unit", "parameter"], how="left")
    df["value"] = df["value"] * df["factor"]
    df = df.dropna(subset=["value"])
    return df[["site_no", "date", "parameter", "value"]]


def load_raw():
    sensor = pd.read_csv(config.RAW_DIR / "sensor_daily.csv", dtype={"site_no": str}, parse_dates=["date"])
    dpath = config.RAW_DIR / "discrete_samples.csv"
    discrete = pd.read_csv(dpath, dtype={"site_no": str}) if dpath.exists() else pd.DataFrame()
    discrete = clean_discrete(discrete) if not discrete.empty else clean_discrete(pd.DataFrame())
    discrete["date"] = pd.to_datetime(discrete["date"], errors="coerce")
    return sensor, discrete.dropna(subset=["date"])


# ------------------------------------------------------------- daily table ---
def daily_table(sensor: pd.DataFrame, discrete: pd.DataFrame) -> pd.DataFrame:
    """Site-by-day table with one column per parameter (NaN = not observed)."""
    keys = ["site_no", "date", "parameter"]
    raw = sensor.groupby(keys)["value"].mean().unstack("parameter")
    if not discrete.empty:
        lab = discrete.groupby(keys)["value"].mean().unstack("parameter").add_suffix("_lab")
        raw = raw.join(lab, how="outer")
    for p in config.ALL_PARAMS:
        cur, lab = raw.get(p), raw.get(f"{p}_lab")
        if cur is None:
            raw[p] = lab if lab is not None else np.nan
        elif lab is not None:
            raw[p] = cur.combine_first(lab)     # sensor first, lab fills gaps
    raw = raw[config.ALL_PARAMS]

    for p in config.ALL_PARAMS:                 # QC: drop physically impossible values
        lo, hi = config.BOUNDS.get(p, (0, np.inf))
        raw[p] = raw[p].where(raw[p].between(lo, hi))

    frames = []
    for site, g in raw.reset_index().groupby("site_no"):
        g = g.drop(columns="site_no").set_index("date").sort_index().asfreq("D")
        g.insert(0, "site_no", site)
        frames.append(g)
    return pd.concat(frames)


# ---------------------------------------------------------------- features ---
def site_features(g: pd.DataFrame, horizon: int) -> pd.DataFrame:
    idx = g.index
    idx_s = pd.Series(idx, index=idx)
    out = {}
    for p in config.ALL_PARAMS:
        s = g[p]
        if p in config.DISCRETE_PARAMS:
            # Sparse lab-style series: carry last value forward (past only) + its age.
            filled = s.ffill(limit=config.DISCRETE_FFILL_DAYS)
            last_obs = idx_s.where(s.notna()).ffill()
            out[p] = filled
            out[f"{p}_days_since"] = (idx_s - last_obs).dt.days.where(filled.notna())
        else:
            out[p] = s
            for k in config.LAGS:
                out[f"{p}_lag{k}"] = s.shift(k)
            for w in config.ROLL_WINDOWS:
                out[f"{p}_mean{w}"] = s.rolling(w, min_periods=max(2, w // 4)).mean()
            out[f"{p}_std7"] = s.rolling(7, min_periods=3).std()
            out[f"{p}_delta7"] = s - s.shift(7)

    doy = idx.dayofyear.values
    out["doy_sin"] = pd.Series(np.sin(2 * np.pi * doy / 365.25), index=idx)
    out["doy_cos"] = pd.Series(np.cos(2 * np.pi * doy / 365.25), index=idx)

    for p in config.TARGETS:                    # raw (un-filled) future value
        out[f"target_{p}"] = g[p].shift(-horizon)
    return pd.DataFrame(out, index=idx)


def run(horizon=config.HORIZON_DAYS):
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    sensor, discrete = load_raw()
    daily = daily_table(sensor, discrete)

    parts = []
    for site, g in daily.groupby("site_no"):
        f = site_features(g.drop(columns="site_no"), horizon)
        f.insert(0, "site_no", site)
        parts.append(f)
    feats = pd.concat(parts)
    feats.index.name = "date"
    feats = feats.dropna(subset=config.ALL_PARAMS, how="all").reset_index()

    sites_path = config.RAW_DIR / "sites.csv"
    if sites_path.exists():
        s = pd.read_csv(sites_path, dtype={"site_no": str})
        s = s.rename(columns={"dec_lat_va": "lat", "dec_long_va": "lon"})
        keep = [c for c in ("site_no", "lat", "lon") if c in s]
        if len(keep) > 1:
            feats = feats.merge(s[keep], on="site_no", how="left")
            for c in ("lat", "lon"):
                if c in feats:
                    feats[c] = pd.to_numeric(feats[c], errors="coerce")

    feats.to_csv(config.PROCESSED_DIR / "features.csv", index=False)
    with open(config.PROCESSED_DIR / "meta.json", "w") as f:
        json.dump({"horizon_days": horizon, "targets": config.TARGETS}, f, indent=2)

    log.info("Feature table: %d rows, %d sites, %s to %s", len(feats),
             feats["site_no"].nunique(), feats["date"].min().date(), feats["date"].max().date())
    for p in config.TARGETS:
        log.info("  target_%-20s %6d labelled rows", p, feats[f"target_{p}"].notna().sum())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horizon", type=int, default=config.HORIZON_DAYS)
    run(ap.parse_args().horizon)
