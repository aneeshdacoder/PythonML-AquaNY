"""Stage 2 - turn standard observations into model-ready tables.

Outputs (data/processed/):
  daily     one row per station-day of OBSERVED values (used for climatology + monitoring)
  features  one row per station-day: dynamic features, static station features, and a target
            column per (parameter, horizon): target_<param>_h<days>
  meta.json horizons, targets

Sub-daily data is aggregated to daily (mean, or sum for rainfall; min/max for registry
"extras"). Sparse lab series are carried forward (past values only) with an age feature, and
their targets use the nearest sample within +/- LAB_TARGET_TOL_DAYS of the target date.
"""
import argparse
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd

import config
import schema
import utils

log = logging.getLogger("features")


def daily_columns() -> list:
    cols = []
    for p, spec in config.PARAMETERS.items():
        cols.append(p)
        cols += [f"{p}_{e}" for e in spec["extras"]]
    return cols


DAILY_COLS = daily_columns()


# ---------------------------------------------------------------- daily table ---
def daily_observations(obs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to station-day. Returns a frame indexed by (station_id, date)."""
    obs = obs.assign(date=obs["timestamp"].dt.floor("D"))
    g = obs.groupby(["station_id", "date", "parameter"])["value"].agg(
        ["mean", "sum", "min", "max"]).reset_index()
    cols = {}
    for p, spec in config.PARAMETERS.items():
        gp = g[g["parameter"] == p].set_index(["station_id", "date"])
        if gp.empty:
            continue
        cols[p] = gp[spec["daily"]]
        for e in spec["extras"]:
            cols[f"{p}_{e}"] = gp[e]
    daily = pd.DataFrame(cols)
    for c in DAILY_COLS:
        if c not in daily:
            daily[c] = np.nan
    return daily[DAILY_COLS].sort_index()


# ------------------------------------------------------------- static features ---
def static_frame(stations) -> pd.DataFrame:
    """Per-station static features + a `group` (HUC2 region) used for climatology pooling."""
    if stations is None or len(stations) == 0:
        return pd.DataFrame(columns=["group"])
    s = stations.set_index("station_id")
    out = pd.DataFrame(index=s.index)
    out["lat"], out["lon"] = s["latitude"], s["longitude"]
    out["elevation_m"] = s["elevation_m"]
    out["log_drainage_km2"] = np.log1p(s["drainage_area_km2"].clip(lower=0))
    huc2 = s["huc"].astype(str).str[:2].where(s["huc"].notna())
    out["huc2_num"] = pd.to_numeric(huc2, errors="coerce")
    for t in config.STATION_TYPES:
        out[f"type_{t}"] = (s["site_type"] == t).astype(float)
    for c in [c for c in s.columns if c.startswith("static_")]:
        out[c] = s[c]
    out["group"] = huc2.fillna("unk")
    return out


# ------------------------------------------------------------------- features ---
def _future(s: pd.Series, h: int, tol: int) -> pd.Series:
    """Value h days ahead; with tol>0 fall back to the nearest sample within +/- tol days
    (never earlier than the day after the forecast origin)."""
    out = s.shift(-h)
    for d in range(1, min(tol, max(0, h - 1)) + 1):
        out = out.combine_first(s.shift(-(h - d))).combine_first(s.shift(-(h + d)))
    return out


def site_features(g: pd.DataFrame, horizons) -> pd.DataFrame:
    """g: one station's calendar-complete daily frame (columns DAILY_COLS)."""
    idx = g.index
    idx_s = pd.Series(idx, index=idx)
    out = {}
    for p, spec in config.PARAMETERS.items():
        s, kind = g[p], spec["kind"]
        if kind == "lab":
            filled = s.ffill(limit=config.DISCRETE_FFILL_DAYS)
            last_obs = idx_s.where(s.notna()).ffill()
            out[p] = filled
            out[f"{p}_days_since"] = (idx_s - last_obs).dt.days.where(filled.notna())
        elif kind == "driver":
            out[p] = s
            for w in config.RAIN_WINDOWS:
                out[f"{p}_sum{w}"] = s.rolling(w, min_periods=max(1, w // 2)).sum()
            last_wet = idx_s.where(s >= 1.0).ffill()
            out[f"{p}_days_since_wet"] = (idx_s - last_wet).dt.days
        else:
            out[p] = s
            last = s.ffill(limit=config.SENSOR_MAX_AGE_DAYS)
            last_obs = idx_s.where(s.notna()).ffill()
            out[f"{p}_last"] = last                       # latest usable reading
            out[f"{p}_days_since"] = (idx_s - last_obs).dt.days.where(last.notna())
            for k in config.LAGS:
                out[f"{p}_lag{k}"] = s.shift(k)
            for w in config.ROLL_WINDOWS:
                out[f"{p}_mean{w}"] = s.rolling(w, min_periods=max(2, w // 4)).mean()
            out[f"{p}_std7"] = s.rolling(7, min_periods=3).std()
            out[f"{p}_delta7"] = s - s.shift(7)
            if spec["storm"]:                      # storm response (high-flow indicators)
                out[f"{p}_ratio30"] = s / s.rolling(30, min_periods=7).mean()
                out[f"{p}_max7"] = s.rolling(7, min_periods=2).max()
                out[f"{p}_rise3"] = s - s.shift(3)
        for e in spec["extras"]:
            out[f"{p}_{e}"] = g[f"{p}_{e}"]

    doy = idx.dayofyear.values
    out["doy_sin"] = pd.Series(np.sin(2 * np.pi * doy / 365.25), index=idx)
    out["doy_cos"] = pd.Series(np.cos(2 * np.pi * doy / 365.25), index=idx)

    for p in config.TARGETS:
        tol = config.LAB_TARGET_TOL_DAYS if config.PARAMETERS[p]["kind"] == "lab" else 0
        for h in horizons:
            out[f"target_{p}_h{h}"] = _future(g[p], h, tol)
    return pd.DataFrame(out, index=idx)


def build_feature_table(daily: pd.DataFrame, stations, horizons, drop_empty=True) -> pd.DataFrame:
    parts = []
    for sid, g in daily.groupby(level="station_id"):
        g = g.droplevel("station_id").sort_index().asfreq("D")
        f = site_features(g, horizons)
        f.insert(0, "station_id", sid)
        parts.append(f)
    feats = pd.concat(parts)
    feats.index.name = "date"
    feats = feats.reset_index()
    if drop_empty:
        feats = feats.dropna(subset=config.ALL_PARAMS, how="all")

    static = static_frame(stations)
    if len(static):
        feats = feats.merge(static, left_on="station_id", right_index=True, how="left")
    else:
        feats["group"] = "unk"
    feats["group"] = feats["group"].fillna("unk")
    num = [c for c in feats.columns if c not in utils.META_COLS]
    feats[num] = feats[num].astype("float32")
    return feats.reset_index(drop=True)


def features_for_observations(obs, stations, horizons=None, as_of=None) -> pd.DataFrame:
    """Latest feature row per station from raw observations. Used for brand-new stations
    (a few recent readings, or none at all, plus static metadata)."""
    horizons = horizons or config.HORIZONS
    if obs is None or len(obs) == 0:
        day = pd.Timestamp(as_of or pd.Timestamp.today()).normalize()
        sid = stations["station_id"].iloc[0]
        daily = pd.DataFrame(np.nan, columns=DAILY_COLS,
                             index=pd.MultiIndex.from_tuples([(sid, day)], names=["station_id", "date"]))
    else:
        daily = daily_observations(obs)
    feats = build_feature_table(daily, stations, horizons, drop_empty=False)
    return feats.sort_values("date").groupby("station_id").tail(1).reset_index(drop=True)


# ------------------------------------------------------------------------ run ---
def run(horizons=None):
    horizons = tuple(horizons or config.HORIZONS)
    obs = utils.read_table(config.RAW_DIR, "observations")
    stations = (utils.read_table(config.RAW_DIR, "stations")
                if utils.table_exists(config.RAW_DIR, "stations") else None)
    obs = schema.normalize_observations(obs)           # re-validate (cheap, idempotent)
    daily = daily_observations(obs)
    utils.write_table(daily.reset_index(), config.PROCESSED_DIR, "daily")

    feats = build_feature_table(daily, stations, horizons)
    utils.write_table(feats, config.PROCESSED_DIR, "features")
    with open(config.PROCESSED_DIR / "meta.json", "w") as f:
        json.dump({"horizons": list(horizons), "targets": config.TARGETS,
                   "built_at": datetime.utcnow().isoformat(),
                   "n_stations": int(feats["station_id"].nunique())}, f, indent=2)
    log.info("Features: %d rows, %d stations, %d columns, %s to %s", len(feats),
             feats["station_id"].nunique(), feats.shape[1],
             feats["date"].min().date(), feats["date"].max().date())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horizons", type=int, nargs="+", default=list(config.HORIZONS))
    run(ap.parse_args().horizons)
