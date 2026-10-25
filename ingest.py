"""Stage 1 - ingest: run a source adapter and write the standard tables
(data/raw/observations, data/raw/stations).

  python ingest.py --source usgs --state NY --start 2015-01-01 --weather
  python ingest.py --source csv --file my_export.csv --mapping examples/mapping_example.json \
                   --stations-file my_stations.csv --weather
"""
import argparse
import logging
from datetime import date

import pandas as pd

import config
import utils
from adapters import csv_adapter, usgs, weather

log = logging.getLogger("ingest")


def run(source="usgs", state="NY", start="2015-01-01", end=None, max_sites=50, min_params=1,
        lab=True, with_weather=False, file=None, mapping=None, stations_file=None, append=False,
        recent_days=30):
    end = end or date.today().isoformat()
    if source == "usgs":
        obs, stations = usgs.collect(state, start, end, max_sites=max_sites, min_params=min_params,
                                  lab=lab, recent_days=recent_days)
    elif source == "csv":
        if not file:
            raise SystemExit("--file is required for --source csv")
        obs, stations = csv_adapter.load(file, mapping, stations_file)
        start = start if start != "2015-01-01" else str(obs["timestamp"].min())[:10]
        end = str(obs["timestamp"].max())[:10] if end == date.today().isoformat() else end
    else:
        raise SystemExit(f"Unknown source {source!r}")

    if stations is None:        # no metadata supplied: stations still work, just without static features
        stations = pd.DataFrame({"station_id": sorted(obs["station_id"].unique())})
        from schema import normalize_stations
        stations = normalize_stations(stations)

    unmatched = sorted(set(obs["station_id"]) - set(stations["station_id"]))
    if unmatched and stations["latitude"].notna().any():
        log.warning("%d observation station IDs have NO match in the stations table (e.g. %s). "
                    "Check for lost leading zeros or different ID formats; those stations get no "
                    "static features.", len(unmatched), ", ".join(unmatched[:5]))

    if with_weather:
        if stations["latitude"].notna().any():
            obs = pd.concat([obs, weather.fetch_precipitation(stations, start, end)], ignore_index=True)
        else:
            log.warning("--weather skipped: stations have no coordinates")

    if append and utils.table_exists(config.RAW_DIR, "observations"):
        obs = pd.concat([utils.read_table(config.RAW_DIR, "observations"), obs], ignore_index=True)
        stations = pd.concat([utils.read_table(config.RAW_DIR, "stations"), stations]
                             ).drop_duplicates("station_id", keep="last")
    obs = obs.drop_duplicates().sort_values(["station_id", "parameter", "timestamp"])
    utils.write_table(obs, config.RAW_DIR, "observations")
    utils.write_table(stations, config.RAW_DIR, "stations")
    log.info("Wrote %d observations, %d stations", len(obs), len(stations))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--source", choices=["usgs", "csv"], default="usgs")
    p.add_argument("--state", default="NY")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--max-sites", type=int, default=50)
    p.add_argument("--min-params", type=int, default=1,
                   help="usgs: min FORECASTABLE parameters (turbidity, pH, DO, conductance, nitrate) per station")
    p.add_argument("--skip-lab", action="store_true")
    p.add_argument("--weather", action="store_true", help="add rainfall from Open-Meteo")
    p.add_argument("--file")
    p.add_argument("--mapping")
    p.add_argument("--stations-file")
    p.add_argument("--append", action="store_true", help="merge into existing tables")
    p.add_argument("--recent-days", type=int, default=30,
                   help="usgs: only stations still reporting within this many days")
    a = p.parse_args()
    run(a.source, a.state, a.start, a.end, a.max_sites, a.min_params, not a.skip_lab,
        a.weather, a.file, a.mapping, a.stations_file, a.append, a.recent_days)
