"""Rainfall adapter (Open-Meteo historical archive). Precipitation is the strongest driver of
turbidity, E. coli and nitrate spikes, and is rarely measured at the water-quality station."""
import logging
import time

import pandas as pd

import config
import schema
from adapters.usgs import make_session

log = logging.getLogger("weather")


def _one_station(session, sid, lat, lon, start, end):
    r = session.get(config.OPEN_METEO_ARCHIVE_URL, timeout=120, params={
        "latitude": lat, "longitude": lon, "start_date": str(start)[:10],
        "end_date": str(end)[:10], "daily": "precipitation_sum", "timezone": "UTC"})
    r.raise_for_status()
    daily = r.json()["daily"]
    return [(sid, d, "precipitation", v, "mm") for d, v in
            zip(daily["time"], daily["precipitation_sum"]) if v is not None]


def fetch_precipitation(stations: pd.DataFrame, start, end, pause=0.5) -> pd.DataFrame:
    """One request per station, paced to respect Open-Meteo's rate limit. Stations that hit a
    429 are retried once after a longer pause instead of being silently left without rain."""
    session = make_session()
    rows, failed = [], []
    st = stations.dropna(subset=["latitude", "longitude"])
    items = list(zip(st["station_id"], st["latitude"], st["longitude"]))
    for i, (sid, lat, lon) in enumerate(items):
        try:
            rows += _one_station(session, sid, lat, lon, start, end)
        except Exception as exc:
            failed.append((sid, lat, lon))
            log.warning("Rain download failed for %s (will retry): %s", sid, str(exc)[:90])
        if (i + 1) % 25 == 0:
            log.info("Rain: %d/%d stations", i + 1, len(items))
        time.sleep(pause)
    if failed:
        log.info("Retrying %d rain downloads after a 60 s pause (rate limit)", len(failed))
        time.sleep(60)
        for sid, lat, lon in failed:
            try:
                rows += _one_station(session, sid, lat, lon, start, end)
            except Exception as exc:
                log.warning("Rain unavailable for %s: %s", sid, str(exc)[:90])
            time.sleep(2.0)
    df = pd.DataFrame(rows, columns=["station_id", "timestamp", "parameter", "value", "unit"])
    return schema.normalize_observations(df)
