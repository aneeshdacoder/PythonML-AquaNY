"""Model-serving layer: loads trained models + latest features and answers forecast queries,
including for brand-new stations. Reusable from the API, a CLI or a notebook."""
import json
import logging
import math
import threading

import joblib
import pandas as pd

import build_features
import config
import modeling
import schema
import utils

log = logging.getLogger("service")

UNITS = {p: s["unit"] for p, s in config.PARAMETERS.items()}


def _clean(x):
    """JSON cannot represent NaN/NaT; convert to None."""
    if x is None:
        return None
    try:
        return None if pd.isna(x) else x
    except (TypeError, ValueError):
        return x


def _f(x):
    x = _clean(x)
    return None if x is None else float(x)


def confidence(known: dict, new: dict, new_station: bool = False):
    """Traffic-light trust label for the frontend, from holdout evidence. Returns (level, caveats).
    new_station=True judges the model by its performance on stations it never saw."""
    m = new if new_station else known
    if not m or not m.get("n_test"):
        return "unknown", ["never tested on unseen stations" if new_station else "no holdout evaluation"]
    level, caveats = "high", []
    sp, sc, cov, n = (m.get("skill_vs_persistence"), m.get("skill_vs_climatology"),
                      m.get("interval_coverage"), m["n_test"])
    if n < 100:
        level = "medium"
        caveats.append(f"small holdout sample (n={n})")
    if sp is not None and sp < -0.05:
        level = "low"
        caveats.append("worse than 'same as now' on holdout")
    if cov is not None and cov < 0.65:
        level = "low"
        caveats.append(f"interval under-covers ({cov:.0%} vs 80% target)")
    elif cov is not None and cov < 0.75 and level == "high":
        level = "medium"
        caveats.append(f"interval slightly under-covers ({cov:.0%} vs 80% target)")
    if sc is not None and sc < 0.02 and level == "high":
        level = "medium"
        caveats.append("no better than the seasonal normal")
    return level, caveats


class ForecastService:
    def __init__(self):
        self._lock = threading.Lock()
        self._mtime = None
        self.bundles, self.history, self.latest, self.stations = {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        self.load_models()
        self.refresh_data(force=True)

    # ---- loading ----------------------------------------------------------
    def load_models(self):
        bundles = {}
        for path in sorted(config.MODEL_DIR.glob("*.joblib")):
            b = joblib.load(path)
            bundles[(b["target"], b["horizon_days"])] = b
        if not bundles:
            raise RuntimeError("No trained models in models/. Run the pipeline (train.py) first.")
        self.bundles = bundles
        self.horizons = sorted({h for _, h in bundles})
        log.info("Loaded %d models (%d parameters x horizons %s)", len(bundles),
                 len({t for t, _ in bundles}), self.horizons)

    def _features_path(self):
        for ext in ("parquet", "csv"):
            p = config.PROCESSED_DIR / f"features.{ext}"
            if p.exists():
                return p
        raise RuntimeError("features table missing. Run build_features.py first.")

    def refresh_data(self, force=False):
        """Reload features only if the file changed on disk (cheap to call per request)."""
        mtime = self._features_path().stat().st_mtime
        if not force and mtime == self._mtime:
            return
        with self._lock:
            df = utils.load_features()
            self.history = df
            self.latest = (df.sort_values("date").groupby("station_id").tail(1)
                           .set_index("station_id", drop=False))
            if utils.table_exists(config.RAW_DIR, "stations"):
                self.stations = utils.read_table(config.RAW_DIR, "stations").set_index("station_id")
            self._mtime = mtime
            log.info("Data refreshed: %d stations, latest date %s", len(self.latest), df["date"].max().date())

    # ---- helpers ----------------------------------------------------------
    @staticmethod
    def _reliability(b, new_station=False):
        known, new = b["holdout"].get("known", {}), b["holdout"].get("new", {})
        cls = b["holdout"].get("known_classifier", {})
        level, caveats = confidence(known, new, new_station)
        return {"confidence": level, "caveats": caveats,
                "mae_known_stations": _f(known.get("mae")), "mae_new_stations": _f(new.get("mae")),
                "skill_vs_persistence": _f(known.get("skill_vs_persistence")),
                "skill_vs_climatology": _f(known.get("skill_vs_climatology")),
                "interval_coverage": _f(known.get("interval_coverage")), "auc": _f(cls.get("auc"))}

    def _items(self, rows, horizons=None, parameters=None, new_station=False):
        items = []
        for (target, h), b in sorted(self.bundles.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            if (horizons and h not in horizons) or (parameters and target not in parameters):
                continue
            out = modeling.predict_bundle(b, rows)
            spec = config.THRESHOLDS.get(target)
            point = float(out["point"][0])
            items.append({
                "parameter": target, "unit": UNITS.get(target), "horizon_days": h,
                "forecast_date": (rows["date"].iloc[0] + pd.Timedelta(days=h)).date().isoformat(),
                "prediction": _f(point), "lower": _f(out["lower"][0]), "upper": _f(out["upper"][0]),
                "probability_exceedance": _f(out["p_exceed"][0]),
                "basis": str(modeling.basis_labels(out)[0]),
                "data_age_days": _f(out["age"][0]),
                "exceeds_screening_threshold": bool(utils.exceeds([point], spec).iloc[0]) if spec else None,
                "threshold": spec, "model": b["model_name"],
                "reliability": self._reliability(b, new_station)})
            if out["fallback"][0]:        # not a model forecast: never borrow the model's trust label
                items[-1]["reliability"]["confidence"] = "none"
                items[-1]["reliability"]["caveats"] = ["no recent reading: this is the seasonal normal, not a forecast"]
        return items

    # ---- queries ----------------------------------------------------------
    def site_ids(self):
        return sorted(self.latest.index)

    def site_info(self, site_no):
        row = self.stations.loc[site_no] if site_no in self.stations.index else None
        get = lambda k: _clean(row[k]) if row is not None and k in row else None  # noqa: E731
        return {"site_no": site_no, "name": get("name"), "latitude": _f(get("latitude")),
                "longitude": _f(get("longitude")), "site_type": get("site_type"),
                "latest_data_date": self.latest.loc[site_no, "date"].date().isoformat()}

    def forecast(self, site_no, horizons=None, parameters=None):
        if site_no not in self.latest.index:
            raise KeyError(site_no)
        row = self.latest.loc[[site_no]]
        d = row["date"].iloc[0]
        a = utils.station_age_days(row).iloc[0]       # days since ANY water-quality reading
        age = None if pd.isna(a) else int(a)
        return {"site_no": site_no, "data_date": d.date().isoformat(), "data_age_days": age,
                "stale": age is None or age > config.STATION_STALE_DAYS,
                "forecasts": self._items(row, horizons, parameters)}

    def forecast_new_station(self, meta: dict, observations=None, horizons=None, parameters=None):
        """Forecast for a station the models have never seen. With no readings the forecast is
        driven by static metadata + regional climatology; supply recent readings (even a few
        weeks) to let the models anchor on the station's actual level."""
        sid = "new-station"
        st = schema.normalize_stations(pd.DataFrame([{**meta, "station_id": sid}]))
        obs = None
        if observations:
            o = pd.DataFrame(observations).rename(columns={"date": "timestamp"})
            o["station_id"] = sid
            obs = schema.normalize_observations(o)
            if obs.empty:
                obs = None
        rows = build_features.features_for_observations(obs, st)
        basis = "recent_readings+static+climatology" if obs is not None else "static+climatology_only"
        d = rows["date"].iloc[0]
        return {"data_basis": basis, "data_date": d.date().isoformat(),
                "forecasts": self._items(rows, horizons, parameters, new_station=True)}

    def history_series(self, site_no, parameter, days):
        g = self.history[self.history["station_id"] == site_no]
        if g.empty:
            raise KeyError(site_no)
        g = g[g["date"] >= g["date"].max() - pd.Timedelta(days=days)]
        return [{"date": d.date().isoformat(), "value": _f(v)} for d, v in zip(g["date"], g[parameter])]

    def model_report(self):
        return {f"{t}_h{h}": {"parameter": t, "horizon_days": h, "model": b["model_name"],
                              "trained_at": b["trained_at"], **self._reliability(b)}
                for (t, h), b in sorted(self.bundles.items())}

    def monitoring(self):
        path = config.REPORT_DIR / "monitor_report.json"
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)