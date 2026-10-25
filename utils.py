"""Shared helpers: table I/O (Parquet if available, else CSV), splits, metrics, transforms."""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score)

import config

META_COLS = {"station_id", "date", "group"}
_STR_COLS = ("station_id", "group", "huc", "name", "site_type")
_DATE_COLS = ("date", "timestamp")


# ------------------------------------------------------------------ table I/O ---
def write_table(df, directory, stem):
    directory.mkdir(parents=True, exist_ok=True)
    pq, csv = directory / f"{stem}.parquet", directory / f"{stem}.csv"
    try:
        df.to_parquet(pq, index=False)
        csv.unlink(missing_ok=True)
        return pq
    except (ImportError, ValueError):          # pyarrow not installed -> CSV
        df.to_csv(csv, index=False)
        pq.unlink(missing_ok=True)
        return csv


def table_exists(directory, stem) -> bool:
    return (directory / f"{stem}.parquet").exists() or (directory / f"{stem}.csv").exists()


def read_table(directory, stem) -> pd.DataFrame:
    pq, csv = directory / f"{stem}.parquet", directory / f"{stem}.csv"
    if pq.exists():
        return pd.read_parquet(pq)
    if not csv.exists():
        raise FileNotFoundError(f"{stem} not found in {directory}. Run the earlier pipeline stage.")
    header = pd.read_csv(csv, nrows=0).columns
    return pd.read_csv(csv, dtype={c: str for c in _STR_COLS if c in header},
                       parse_dates=[c for c in _DATE_COLS if c in header], low_memory=False)


def load_features():
    return read_table(config.PROCESSED_DIR, "features")


def load_daily():
    return read_table(config.PROCESSED_DIR, "daily")


def load_meta() -> dict:
    with open(config.PROCESSED_DIR / "meta.json") as f:
        return json.load(f)


def feature_columns(df) -> list:
    return [c for c in df.columns if c not in META_COLS and not c.startswith("target_")]


def current_col(param) -> str:
    """Column holding the latest known value of a parameter (carried forward a limited time)."""
    return param if config.PARAMETERS[param]["kind"] == "lab" else f"{param}_last"


def station_age_days(df) -> pd.Series:
    """Days since each row's station last reported ANY water-quality parameter. Drivers such
    as rainfall are excluded: they are current for every station and would mask dead sensors."""
    cols = [f"{p}_days_since" for p, s in config.PARAMETERS.items()
            if s["kind"] != "driver" and f"{p}_days_since" in df]
    if not cols:
        return pd.Series(np.nan, index=df.index)
    return df[cols].min(axis=1)


# ------------------------------------------------------------------ transforms ---
def fwd(x, param):
    """Model scale: log1p for right-skewed parameters, identity otherwise."""
    x = np.asarray(x, dtype=float)
    return np.log1p(np.clip(x, 0, None)) if config.PARAMETERS[param]["log"] else x


def inv(x, param):
    spec = config.PARAMETERS[param]
    x = np.asarray(x, dtype=float)
    with np.errstate(over="ignore"):
        y = np.expm1(x) if spec["log"] else x
    return np.clip(y, spec["lo"], spec["hi"])


# ---------------------------------------------------------------------- splits ---
def split_masks(df, test_fraction, horizon, ref_dates=None):
    """Chronological split with a purge: training rows whose target date lands in the test
    period are dropped, so no label information leaks across the cutoff.
    ref_dates: dates used to place the cutoff (pass the LABELLED rows' dates so each target
    gets its own timeline; sparse lab data can end long before sensor data does)."""
    ref = df["date"] if ref_dates is None else ref_dates
    dates = np.sort(ref.unique())
    cutoff = pd.Timestamp(dates[int(len(dates) * (1 - test_fraction))])
    train = (df["date"] + pd.Timedelta(days=horizon)) < cutoff
    test = df["date"] >= cutoff
    return train, test, cutoff


def choose_holdout_sites(station_ids, fraction, min_sites, seed):
    """Stations withheld from training entirely, to measure performance at NEW stations."""
    ids = np.array(sorted(set(station_ids)))
    if len(ids) < min_sites:
        return set()
    rng = np.random.default_rng(seed)
    n = max(1, int(round(len(ids) * fraction)))
    return set(rng.choice(ids, size=n, replace=False))


# --------------------------------------------------------------------- metrics ---
def regression_report(y, pred, persist=None, clim=None, lo=None, hi=None) -> dict:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    m = np.isfinite(y) & np.isfinite(pred)
    if m.sum() == 0:
        return {"n_test": 0}
    out = {"n_test": int(m.sum()),
           "mae": float(mean_absolute_error(y[m], pred[m])),
           "rmse": float(np.sqrt(mean_squared_error(y[m], pred[m]))),
           "r2": float(r2_score(y[m], pred[m])) if m.sum() > 1 else float("nan")}
    for name, base in (("persistence", persist), ("climatology", clim)):
        if base is None:
            continue
        base = np.asarray(base, float)
        mm = m & np.isfinite(base)
        if mm.any():
            b = mean_absolute_error(y[mm], base[mm])
            out[f"{name}_mae"] = float(b)
            out[f"skill_vs_{name}"] = (float(1 - mean_absolute_error(y[mm], pred[mm]) / b)
                                       if b > 0 else float("nan"))
    if lo is not None and hi is not None:
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        out["interval_coverage"] = float(np.mean((y[m] >= lo[m]) & (y[m] <= hi[m])))
    return out


def classification_report(y_true, p) -> dict:
    y_true, p = np.asarray(y_true, bool), np.asarray(p, float)
    ok = np.isfinite(p)                      # blank probability = no usable forecast
    y_true, p = y_true[ok], p[ok]
    out = {"n_test": int(len(y_true)), "base_rate": float(y_true.mean()) if len(y_true) else float("nan")}
    if len(y_true) and y_true.any() and not y_true.all():
        out["auc"] = float(roc_auc_score(y_true, p))
        out["avg_precision"] = float(average_precision_score(y_true, p))
        out["brier"] = float(brier_score_loss(y_true, p))
    return out


def exceeds(values, spec) -> pd.Series:
    v = pd.Series(np.asarray(values, dtype=float))
    flag = pd.Series(False, index=v.index)
    if "max" in spec:
        flag |= v > spec["max"]
    if "min" in spec:
        flag |= v < spec["min"]
    return flag
