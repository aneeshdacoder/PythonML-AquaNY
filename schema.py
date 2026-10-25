"""The standard data contract. Every source (USGS, your app's CSV, a database) is converted
into these two tables, and everything downstream only ever sees them.

observations: station_id, timestamp, parameter, value, unit, detection_flag, quality_flag
stations:     station_id, name, latitude, longitude, site_type, huc, drainage_area_km2,
              elevation_m, plus any numeric columns named static_* (e.g. static_pct_forest)
"""
import logging
import re

import numpy as np
import pandas as pd

import config

log = logging.getLogger("schema")

OBS_COLUMNS = ["station_id", "timestamp", "parameter", "value", "unit",
               "detection_flag", "quality_flag"]
STATION_COLUMNS = ["station_id", "name", "latitude", "longitude", "site_type", "huc",
                   "drainage_area_km2", "elevation_m"]
ND_FLAGS = {"ND", "<", "U", "NON-DETECT", "NOT DETECTED", "BDL"}
BAD_QUALITY = {"X", "BAD", "REJECTED", "R", "INVALID"}


def _key(s) -> str:
    return re.sub(r"[\s_\-\.]+", " ", str(s).strip().lower()).strip()


_ALIASES = {**{_key(a): p for a, p in config.PARAM_ALIASES.items()},
            **{_key(p): p for p in config.PARAMETERS}}


def resolve_parameter(name, extra=None):
    k = _key(name)
    if extra:
        extra_map = {_key(a): p for a, p in extra.items()}
        if k in extra_map:
            return extra_map[k]
    return _ALIASES.get(k)


# ---------------------------------------------------------------- units ---------
def _norm_unit(u) -> str:
    if u is None or (isinstance(u, float) and np.isnan(u)):
        return ""
    return (str(u).lower().replace("\u00b5", "u").replace("\u03bc", "u")
            .replace("\u00b0", "deg").replace(" ", "").replace("^", ""))


_UNITS = {
    "FNU": {k: (1.0, 0.0) for k in ("fnu", "ntu", "ntru", "ftu", "fnrmu", "ntmu")},
    "pH": {k: (1.0, 0.0) for k in ("ph", "stdunits", "standardunits", "units", "su", "stdunit")},
    "degC": {"degc": (1.0, 0.0), "c": (1.0, 0.0), "celsius": (1.0, 0.0),
             "degf": (5 / 9, -160 / 9), "f": (5 / 9, -160 / 9), "fahrenheit": (5 / 9, -160 / 9)},
    "uS/cm": {"us/cm": (1.0, 0.0), "umho/cm": (1.0, 0.0), "umhos/cm": (1.0, 0.0),
              "ms/cm": (1000.0, 0.0)},
    "ft3/s": {"ft3/s": (1.0, 0.0), "cfs": (1.0, 0.0), "m3/s": (35.3147, 0.0),
              "cms": (35.3147, 0.0), "l/s": (0.0353147, 0.0)},
    "mm": {"mm": (1.0, 0.0), "in": (25.4, 0.0), "inch": (25.4, 0.0),
           "inches": (25.4, 0.0), "cm": (10.0, 0.0)},
}
_MASS = (("mg/l", 1.0), ("ug/l", 1e-3), ("ng/l", 1e-6), ("g/l", 1e3), ("ppm", 1.0), ("ppb", 1e-3))


def unit_scale(unit, param):
    """(scale, offset) so canonical = value * scale + offset. NaN scale = unknown unit."""
    canon = config.PARAMETERS[param]["unit"]
    u = _norm_unit(unit)
    if u == "" or u == _norm_unit(canon):      # blank = assume already canonical
        return 1.0, 0.0
    if canon in ("mg/L", "ug/L"):
        for k, f in _MASS:
            if u.startswith(k):
                if param == "nitrate" and "no3" in u:   # "as NO3" -> "as N"
                    f *= 14.007 / 62.004
                if canon == "ug/L":
                    f *= 1000.0
                return f, 0.0
        return np.nan, 0.0
    if canon == "MPN/100mL":
        return (1.0, 0.0) if "/100ml" in u else (np.nan, 0.0)
    return _UNITS.get(canon, {}).get(u, (np.nan, 0.0))


# --------------------------------------------------------- observations ---------
def normalize_observations(df: pd.DataFrame, extra_aliases=None) -> pd.DataFrame:
    missing = {"station_id", "timestamp", "parameter", "value"} - set(df.columns)
    if missing:
        raise ValueError(f"Observations are missing required columns: {sorted(missing)}")
    df = df.copy()
    for c in ("unit", "detection_flag", "quality_flag"):
        df[c] = df[c].fillna("").astype(str) if c in df else ""

    df["station_id"] = df["station_id"].astype(str).str.strip()
    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["timestamp"] = ts.dt.tz_localize(None)
    df["parameter"] = [resolve_parameter(x, extra_aliases) for x in df["parameter"]]
    n_unknown = int(df["parameter"].isna().sum())
    if n_unknown:
        log.warning("Dropping %d rows with parameters not in the registry", n_unknown)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Non-detects are CENSORED values: store half the detection limit rather than dropping
    nd = df["detection_flag"].str.strip().str.upper().isin(ND_FLAGS)
    df.loc[nd & (df["value"] > 0), "value"] *= 0.5
    df = df[~df["quality_flag"].str.strip().str.upper().isin(BAD_QUALITY)]
    df = df.dropna(subset=["station_id", "timestamp", "parameter", "value"])
    if df.empty:
        return pd.DataFrame(columns=OBS_COLUMNS)

    pairs = df[["unit", "parameter"]].drop_duplicates().copy()
    sc = [unit_scale(u, p) for u, p in zip(pairs["unit"], pairs["parameter"])]
    pairs["scale"] = [s[0] for s in sc]
    pairs["offset"] = [s[1] for s in sc]
    df = df.merge(pairs, on=["unit", "parameter"], how="left")
    df["value"] = df["value"] * df["scale"] + df["offset"]
    df = df.dropna(subset=["value"])

    lo = df["parameter"].map(lambda p: config.PARAMETERS[p]["lo"])
    hi = df["parameter"].map(lambda p: config.PARAMETERS[p]["hi"])
    df = df[(df["value"] >= lo) & (df["value"] <= hi)]
    return df[OBS_COLUMNS].reset_index(drop=True)


# ------------------------------------------------------------- stations ---------
_STATION_RENAMES = {
    "site_no": "station_id", "id": "station_id", "station": "station_id",
    "lat": "latitude", "lon": "longitude", "lng": "longitude", "long": "longitude",
    "station_nm": "name", "station_name": "name", "type": "site_type", "site_tp_cd": "site_type",
    "huc_cd": "huc", "elevation": "elevation_m", "drainage_area": "drainage_area_km2",
}


def _site_type(x) -> str:
    s = str(x).strip().lower()
    if s in config.STATION_TYPES:
        return s
    if s.startswith(("st", "riv", "cre")):
        return "stream"
    if s.startswith(("lk", "lak", "res", "pond")):
        return "lake"
    if s.startswith("sp"):
        return "spring"
    if s.startswith(("gw", "gro", "well")):
        return "groundwater"
    if s.startswith(("es", "tid", "coast", "bay")):
        return "estuary"
    return "other"


def normalize_stations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={k: v for k, v in _STATION_RENAMES.items()
                            if k in df.columns and v not in df.columns})
    if "station_id" not in df:
        raise ValueError("Stations table needs a station_id column")
    df = df.copy()
    df["station_id"] = df["station_id"].astype(str).str.strip()
    for c in STATION_COLUMNS[1:]:
        if c not in df:
            df[c] = np.nan
    for c in ("latitude", "longitude", "drainage_area_km2", "elevation_m"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["site_type"] = df["site_type"].map(_site_type)
    df["huc"] = df["huc"].astype(str).str.replace(r"\D", "", regex=True).replace("", np.nan)
    df["name"] = df["name"].astype(object)
    extra = [c for c in df.columns if c.startswith("static_")]
    for c in extra:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[STATION_COLUMNS + extra].drop_duplicates("station_id").reset_index(drop=True)
