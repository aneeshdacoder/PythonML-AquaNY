"""Generic adapter for YOUR app's exports (CSV, long or wide). Describe the column layout in a
small JSON mapping file instead of writing code; see examples/mapping_example.json.

Mapping keys
  sep               field separator (default ",")
  timezone          IANA zone of naive timestamps, e.g. "America/New_York" (default: already UTC)
  columns           standard column -> your column, e.g. {"station_id": "StationID", ...}
  wide              true if each parameter is its own column
  parameter_columns (wide) your column -> registry parameter name
  units             (wide) your column -> unit string
  parameter_aliases extra spellings of parameter names (long format)
  station_columns   your stations-file column -> standard column
"""
import json
import logging

import pandas as pd

import schema

log = logging.getLogger("csv_adapter")


def _localize(ts: pd.Series, tz):
    out = pd.to_datetime(ts, errors="coerce")
    if tz and out.dt.tz is None:
        out = out.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    return out


def load(file, mapping=None, stations_file=None):
    m = mapping if isinstance(mapping, dict) else json.load(open(mapping)) if mapping else {}
    cols = m.get("columns", {})
    df = pd.read_csv(file, dtype=str, low_memory=False, sep=m.get("sep", ","))

    if m.get("wide"):
        pc = m["parameter_columns"]
        sid, tsc = cols.get("station_id", "station_id"), cols.get("timestamp", "timestamp")
        long = df.melt(id_vars=[sid, tsc], value_vars=list(pc), var_name="_col", value_name="value")
        long = long.rename(columns={sid: "station_id", tsc: "timestamp"})
        long["parameter"] = long["_col"].map(pc)
        long["unit"] = long["_col"].map(m.get("units", {})).fillna("")
        long = long.drop(columns="_col")
    else:
        long = df.rename(columns={src: std for std, src in cols.items() if src in df.columns})
    long["timestamp"] = _localize(long["timestamp"], m.get("timezone"))
    obs = schema.normalize_observations(long, extra_aliases=m.get("parameter_aliases"))
    log.info("Loaded %d observations for %d stations", len(obs), obs["station_id"].nunique())

    stations = None
    if stations_file:
        sdf = pd.read_csv(stations_file, dtype=str, sep=m.get("sep", ","))
        sdf = sdf.rename(columns=m.get("station_columns", {}))
        stations = schema.normalize_stations(sdf)
    return obs, stations
