"""Model definitions plus the shared feature/prediction logic used by train, predict and API."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
import utils
from climatology import Climatology

ANOM_COLS = ["clim_origin", "clim_target", "clim_site_offset", "anom_now"]


class DropConstant(BaseEstimator, TransformerMixin):
    """Drop columns with fewer than 2 distinct observed values, decided on whatever data this
    particular fit sees. Histogram gradient boosting can raise on such columns, and small early
    time-series folds routinely contain them even when the full training set does not."""

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.keep_ = [c for c in X.columns if X[c].nunique(dropna=True) > 1]
        if not self.keep_:
            raise ValueError("no informative features in this training sample")
        return self

    def transform(self, X):
        return pd.DataFrame(X)[self.keep_]


class ClipToRange(BaseEstimator, TransformerMixin):
    """Winsorise inputs to the range seen in training (0.5th-99.5th pct). Linear models extrapolate
    wildly on out-of-range inputs (an unusually flashy stream, an extreme elevation), which tree
    models do not; this keeps a new station from producing absurd forecasts."""

    def fit(self, X, y=None):
        import warnings
        A = np.asarray(X, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lo, hi = np.nanquantile(A, 0.005, axis=0), np.nanquantile(A, 0.995, axis=0)
        self.lo_ = np.where(np.isnan(lo), -np.inf, lo)
        self.hi_ = np.where(np.isnan(hi), np.inf, hi)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


class BaselineModel(BaseEstimator, RegressorMixin):
    """Naive forecasts that compete with the learned models in model selection.
    persistence: the value stays what it is now.  climatology: it returns to the regional
    seasonal normal.  site_climatology: it returns to THIS station's seasonal normal.
    (All expressed in anomaly space, like every model here.)"""

    def __init__(self, kind="persistence"):
        self.kind = kind

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        if self.kind == "climatology":              # regional seasonal normal
            return np.zeros(len(X))
        if self.kind == "site_climatology":         # this station's own seasonal normal
            return np.nan_to_num(X["clim_site_offset"].to_numpy(float), nan=0.0)
        a = (X["anom_now"].to_numpy(float) + X["clim_origin"].to_numpy(float)
             - X["clim_target"].to_numpy(float))
        return np.nan_to_num(a, nan=0.0)


def candidate_models() -> dict:
    zoo = {
        "ridge": lambda: make_pipeline(ClipToRange(),
                                       SimpleImputer(strategy="median", keep_empty_features=True),
                                       StandardScaler(), Ridge(alpha=10.0)),
        "hist_gbm": lambda: make_pipeline(DropConstant(), HistGradientBoostingRegressor(
            max_iter=config.HGB_MAX_ITER, learning_rate=0.05, l2_regularization=1.0,
            random_state=config.SEED)),
        "random_forest": lambda: make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            RandomForestRegressor(n_estimators=200, min_samples_leaf=3, n_jobs=-1,
                                  random_state=config.SEED)),
    }
    zoo["persistence"] = lambda: BaselineModel("persistence")
    zoo["climatology"] = lambda: BaselineModel("climatology")
    zoo["site_climatology"] = lambda: BaselineModel("site_climatology")
    return {name: zoo[name]() for name in (*config.CANDIDATES, *config.BASELINE_CANDIDATES)}


def new_classifier():
    return make_pipeline(DropConstant(), HistGradientBoostingClassifier(
        max_iter=min(200, config.HGB_MAX_ITER), learning_rate=0.05, random_state=config.SEED))


def make_X(df, target, horizon, clim, base_cols):
    """Build model inputs: base features + climatology-aware anomaly features.
    Returns (X, region_clim_at_target, station_clim_at_target); climatologies are on the
    model scale. The model's target is (value - region_clim_at_target)."""
    sid = df["station_id"].to_numpy()
    grp = df["group"].to_numpy()
    month_o = df["date"].dt.month.to_numpy()
    month_t = (df["date"] + pd.Timedelta(days=int(horizon))).dt.month.to_numpy()
    clim_o = clim.lookup(sid, month_o, grp, level="group")
    clim_t = clim.lookup(sid, month_t, grp, level="group")
    clim_t_site = clim.lookup(sid, month_t, grp, level="site")
    X = df.reindex(columns=base_cols).copy()
    X["clim_origin"] = clim_o
    X["clim_target"] = clim_t
    X["clim_site_offset"] = clim_t_site - clim_t      # 0 for stations with no history
    X["anom_now"] = utils.fwd(df[utils.current_col(target)].to_numpy(), target) - clim_o
    return X, clim_t, clim_t_site


def oof_site_offset(rows, target, horizon, daily, groups, daily_mask):
    """Out-of-YEAR station offsets for training rows: each row's offset (station normal minus
    regional normal, at the target month) is computed from every OTHER calendar year. An
    in-sample offset leaks the answer into training (looks great in CV, fails on the future).
    Returns (offset, group_clim_at_target) on the model scale."""
    d = daily.loc[daily_mask, ["station_id", "date", target]].dropna()
    d = d.assign(y=utils.fwd(d[target].to_numpy(), target))
    years = rows["date"].dt.year.to_numpy()
    month_t = (rows["date"] + pd.Timedelta(days=int(horizon))).dt.month.to_numpy()
    sid, grp = rows["station_id"].to_numpy(), rows["group"].to_numpy()
    offset = np.zeros(len(rows))
    for yr in np.unique(years):
        c = Climatology().fit(d.loc[d["date"].dt.year != yr, ["station_id", "date", "y"]], groups)
        sel = years == yr
        offset[sel] = (c.lookup(sid[sel], month_t[sel], grp[sel], level="site")
                       - c.lookup(sid[sel], month_t[sel], grp[sel], level="group"))
    return np.nan_to_num(offset, nan=0.0)


def drop_site_offsets(X, station_ids, frac=None, seed=config.SEED):
    """Hide the station-level offset for a random subset of TRAINING stations (set to 0, which
    is exactly what a brand-new station gets). Without this the model leans on the offset and
    fails on new stations, where it is unavailable and anom_now must carry the station level."""
    frac = config.SITE_OFFSET_DROPOUT if frac is None else frac
    ids = np.unique(station_ids)
    if len(ids) < 3 or frac <= 0:
        return X
    rng = np.random.default_rng(seed)
    chosen = rng.choice(ids, size=max(1, int(len(ids) * frac)), replace=False)
    X = X.copy()
    X.loc[np.isin(station_ids, chosen), "clim_site_offset"] = 0.0
    return X


def basis_labels(out) -> np.ndarray:
    return np.where(~out["fallback"], "recent_data",
                    np.where(out["never_measured"], "never_measured_fallback", "stale_data_fallback"))


def predict_bundle(bundle, df) -> dict:
    """Point forecast, interval and exceedance probability for each row of df.

    Rows whose station has no recent reading of the target parameter cannot be forecast from
    current conditions. They fall back to the station's seasonal normal with a wider interval
    and are flagged (out["fallback"]) so they are never mistaken for a real forecast."""
    target, h = bundle["target"], bundle["horizon_days"]
    X, clim_t, clim_site = make_X(df, target, h, bundle["climatology"], bundle["base_cols"])
    X = X[bundle["features"]]
    center = bundle["model"].predict(X) + clim_t        # model scale

    age_col = f"{target}_days_since"
    age = df[age_col].to_numpy(float) if age_col in df else np.full(len(df), np.nan)
    limit = (config.DISCRETE_FFILL_DAYS if config.PARAMETERS[target]["kind"] == "lab"
             else config.SENSOR_MAX_AGE_DAYS)
    fallback = ~(age <= limit)                           # NaN age counts as fallback
    never = fallback & ~bundle["climatology"].has_site(df["station_id"].to_numpy())
    q_stale = bundle.get("interval_q_fallback", bundle["interval_q"])
    q_never = bundle.get("interval_q_group", q_stale)    # wider: only the regional normal is known
    q = np.where(never, q_never, np.where(fallback, q_stale, bundle["interval_q"]))
    center = np.where(fallback, clim_site, center)

    out = {"point": utils.inv(center, target),
           "lower": utils.inv(center - q, target),
           "upper": utils.inv(center + q, target),
           "clim": utils.inv(clim_site, target),
           "fallback": fallback, "never_measured": never, "age": age,
           "p_exceed": np.full(len(df), np.nan)}
    if bundle.get("classifier") is not None:
        p = bundle["classifier"].predict_proba(X)[:, 1]
        out["p_exceed"] = np.where(fallback, np.nan, p)
    return out
