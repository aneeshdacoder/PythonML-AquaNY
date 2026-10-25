"""Hierarchical seasonal climatology (the "what is normal here, this month" baseline).

Level 1  global month means
Level 2  group (HUC2 region) month means, shrunk toward global
Level 3  station month means, shrunk toward their group
Shrinkage means a station with few samples leans on its region; a brand-new station with no
history falls back to its region, then to the global value.

Models predict the anomaly relative to the REGION-level normal (level="group"): it is built
from thousands of samples (so no overfitting on sparse lab data) and exists for every station,
including brand-new ones. A station's own offset is carried by its recent readings
(anom_now), exactly as it will be for a new station at prediction time. The station-level
normal is kept as a stronger baseline for skill reporting.
"""
import numpy as np
import pandas as pd

import config

_GNAMES = ["group", "month"]
_SNAMES = ["station_id", "month"]


def _empty(names):
    return pd.Series(dtype=float, index=pd.MultiIndex.from_arrays([[], []], names=names))


class Climatology:
    def __init__(self, k_site=None, k_group=None):
        self.k_site = k_site if k_site is not None else config.CLIM_K_SITE
        self.k_group = k_group if k_group is not None else config.CLIM_K_GROUP
        self.global_mean = np.nan
        self.month_global = pd.Series(np.nan, index=range(1, 13))
        self.group_val = _empty(_GNAMES)
        self.site_val = _empty(_SNAMES)

    @staticmethod
    def _shrink(mean, n, prior, k):
        return (n * mean + k * prior) / (n + k)

    def fit(self, obs: pd.DataFrame, groups: pd.Series):
        """obs: columns station_id, date, y (model-scale values). groups: station_id -> group."""
        d = obs.dropna(subset=["y"]).copy()
        if d.empty:
            return self
        d["month"] = d["date"].dt.month
        d["group"] = d["station_id"].map(groups).fillna("unk")
        self.global_mean = float(d["y"].mean())

        m = d.groupby("month")["y"].agg(["mean", "count"])
        mg = self._shrink(m["mean"], m["count"], self.global_mean, self.k_group)
        self.month_global = mg.reindex(range(1, 13)).fillna(self.global_mean)

        g = d.groupby(["group", "month"])["y"].agg(["mean", "count"])
        prior = self.month_global.reindex(g.index.get_level_values("month")).to_numpy(float)
        self.group_val = pd.Series(
            self._shrink(g["mean"].to_numpy(), g["count"].to_numpy(), prior, self.k_group),
            index=g.index)

        s = d.groupby(["station_id", "month"])["y"].agg(["mean", "count"])
        s_groups = np.asarray(s.index.get_level_values("station_id").map(groups).fillna("unk"))
        base = self._group_lookup(s_groups, np.asarray(s.index.get_level_values("month")))
        self.site_val = pd.Series(
            self._shrink(s["mean"].to_numpy(), s["count"].to_numpy(), base, self.k_site),
            index=s.index)
        return self

    def has_site(self, station_ids):
        """True where the station has its own history in this climatology (else only the
        regional normal is available for it)."""
        known = self.site_val.index.get_level_values("station_id").unique()
        return np.isin(np.asarray(station_ids), known)

    def _group_lookup(self, groups, months):
        base = self.month_global.reindex(months).to_numpy(float)
        idx = pd.MultiIndex.from_arrays([groups, months], names=_GNAMES)
        gv = self.group_val.reindex(idx).to_numpy(float)
        return np.where(np.isnan(gv), base, gv)

    def lookup(self, station_ids, months, groups, level="site"):
        """level="group": region-level normal (what the models use; valid for any station).
        level="site":  station-level normal when the station has history (evaluation baseline)."""
        months = np.asarray(months)
        groups = np.asarray(pd.Series(groups).fillna("unk"))
        base = self._group_lookup(groups, months)
        if level == "group":
            return base
        idx = pd.MultiIndex.from_arrays([np.asarray(station_ids), months], names=_SNAMES)
        sv = self.site_val.reindex(idx).to_numpy(float)
        return np.where(np.isnan(sv), base, sv)
