"""Shared helpers: I/O, time-based splitting, metrics, threshold logic."""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import config

META_COLS = {"site_no", "date"}


def load_features() -> pd.DataFrame:
    path = config.PROCESSED_DIR / "features.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run build_features.py first.")
    return pd.read_csv(path, dtype={"site_no": str}, parse_dates=["date"])


def load_meta() -> dict:
    with open(config.PROCESSED_DIR / "meta.json") as f:
        return json.load(f)


def feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in META_COLS and not c.startswith("target_")]


def split_masks(df: pd.DataFrame, test_fraction: float, horizon: int):
    """Chronological split. Training rows whose target date would land inside the
    test period are purged, so no label information leaks across the cutoff."""
    dates = np.sort(df["date"].unique())
    cutoff = pd.Timestamp(dates[int(len(dates) * (1 - test_fraction))])
    train = (df["date"] + pd.Timedelta(days=horizon)) < cutoff
    test = df["date"] >= cutoff
    return train, test, cutoff


def regression_report(y, pred, persist) -> dict:
    """Model metrics plus skill versus a persistence baseline ("it stays the same")."""
    y, pred, persist = (np.asarray(a, dtype=float) for a in (y, pred, persist))
    out = {
        "n_test": int(len(y)),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)) if len(y) > 1 else float("nan"),
    }
    m = np.isfinite(persist)
    if m.any():
        base = mean_absolute_error(y[m], persist[m])
        model_same = mean_absolute_error(y[m], pred[m])
        out["persistence_mae"] = float(base)
        out["skill_vs_persistence"] = float(1 - model_same / base) if base > 0 else float("nan")
    return out


def exceeds(values, spec: dict) -> pd.Series:
    v = pd.Series(np.asarray(values, dtype=float))
    flag = pd.Series(False, index=v.index)
    if "max" in spec:
        flag |= v > spec["max"]
    if "min" in spec:
        flag |= v < spec["min"]
    return flag
