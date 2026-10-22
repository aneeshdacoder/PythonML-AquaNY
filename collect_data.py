"""Stage 1 - collect raw data from USGS.

* Site discovery + daily sensor values  -> USGS Water Services (requests)
* Discrete lab samples (E. coli, lead, sulfate, fluoride, hardness, nitrate)
  -> Water Quality Portal, which serves USGS NWIS sample results.

Outputs (data/raw/): sites.csv, sensor_daily.csv, discrete_samples.csv
"""
import argparse
import io
import logging
from datetime import date

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import config

log = logging.getLogger("collect")


def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=5, backoff_factor=1.5,
                  status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers["User-Agent"] = "water-quality-pipeline/1.0"
    return s


def parse_rdb(text: str) -> pd.DataFrame:
    """USGS RDB: '#' comments, a header row, then a column-format row to drop."""
    df = pd.read_csv(io.StringIO(text), sep="\t", comment="#", dtype=str)
    return df.iloc[1:].reset_index(drop=True)


def discover_sites(session, state, min_params=3, min_days=365, max_sites=50) -> pd.DataFrame:
    """Active stream sites with long daily records for several target parameters."""
    codes = list(config.SENSOR_PARAMS.values())
    params = {
        "format": "rdb", "stateCd": state, "parameterCd": ",".join(codes),
        "siteType": "ST", "hasDataTypeCd": "dv", "siteStatus": "active",
        "seriesCatalogOutput": "true",
    }
    r = session.get(config.USGS_SITE_URL, params=params, timeout=120)
    r.raise_for_status()
    cat = parse_rdb(r.text)
    cat = cat[(cat["data_type_cd"] == "dv") & cat["parm_cd"].isin(codes)].copy()
    cat["count_nu"] = pd.to_numeric(cat["count_nu"], errors="coerce")
    cat = cat[cat["count_nu"] >= min_days]

    per_param = cat.groupby(["site_no", "parm_cd"])["count_nu"].max().reset_index()
    score = per_param.groupby("site_no").agg(
        n_params=("parm_cd", "nunique"), total_obs=("count_nu", "sum")).reset_index()
    score = score[score["n_params"] >= min_params]
    score = score.sort_values(["n_params", "total_obs"], ascending=False).head(max_sites)

    meta_cols = [c for c in ("site_no", "station_nm", "dec_lat_va", "dec_long_va") if c in cat]
    meta = cat.drop_duplicates("site_no")[meta_cols]
    sites = score.merge(meta, on="site_no", how="left")
    log.info("Selected %d sites in %s", len(sites), state)
    return sites


def fetch_daily_values(session, site_ids, start, end, chunk=20) -> pd.DataFrame:
    code_to_name = {v: k for k, v in config.SENSOR_PARAMS.items()}
    rows = []
    for i in range(0, len(site_ids), chunk):
        part = site_ids[i:i + chunk]
        params = {
            "format": "json", "sites": ",".join(part),
            "parameterCd": ",".join(config.SENSOR_PARAMS.values()),
            "statCd": "00003,00008",  # daily mean (median is used for pH at some sites)
            "startDT": start, "endDT": end,
        }
        r = session.get(config.USGS_DV_URL, params=params, timeout=180)
        if r.status_code == 404:
            log.warning("No daily data for sites %s", part)
            continue
        r.raise_for_status()
        for ts in r.json()["value"]["timeSeries"]:
            site = ts["sourceInfo"]["siteCode"][0]["value"]
            code = ts["variable"]["variableCode"][0]["value"]
            nodata = float(ts["variable"].get("noDataValue", -999999))
            for block in ts["values"]:
                for v in block["value"]:
                    rows.append((site, code, v["dateTime"][:10], v["value"], nodata))
        log.info("Fetched daily values for sites %d-%d of %d", i + 1, i + len(part), len(site_ids))

    df = pd.DataFrame(rows, columns=["site_no", "parm_cd", "date", "value", "nodata"])
    if df.empty:
        return df.drop(columns="nodata").assign(parameter=None)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[df["value"].notna() & (df["value"] != df["nodata"])]
    df["parameter"] = df["parm_cd"].map(code_to_name)
    return df[["site_no", "date", "parameter", "value"]]


def _wqp_request(session, site_part, names, start, end, optional=True):
    """One WQP call. Returns (DataFrame | None, error_text | None)."""
    params = [("siteid", f"USGS-{s}") for s in site_part]
    params += [("characteristicName", n) for n in names]
    params += [
        ("mimeType", "csv"), ("zip", "no"), ("dataProfile", "narrowResult"),
        ("startDateLo", pd.Timestamp(start).strftime("%m-%d-%Y")),
        ("endDateHi", pd.Timestamp(end).strftime("%m-%d-%Y")),
    ]
    if optional:  # filters that are nice-to-have; dropped on retry if WQP rejects them
        params += [("providers", "NWIS"), ("sampleMedia", "Water")]
    r = session.get(config.WQP_RESULT_URL, params=params, timeout=300)
    if r.status_code == 400:
        detail = r.headers.get("Warning") or r.text[:300] or "(empty body)"
        return None, detail.strip()
    r.raise_for_status()
    if r.status_code == 204 or not r.text.strip():
        return pd.DataFrame(), None
    return pd.read_csv(io.StringIO(r.text), dtype=str, low_memory=False), None


def _wqp_resilient(session, site_part, names, start, end):
    """Try the full query; on HTTP 400, isolate and skip whatever WQP rejects."""
    for optional in (True, False):
        df, err = _wqp_request(session, site_part, names, start, end, optional)
        if err is None:
            return df
        log.warning("WQP 400 (optional filters=%s): %s", optional, err)

    frames = []
    for name in names:  # a single unrecognised characteristic name poisons the whole call
        df, err = _wqp_request(session, site_part, [name], start, end, optional=False)
        if err is not None:
            log.warning("Skipping characteristic %r (rejected by WQP: %s)", name, err)
        elif not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_discrete_samples(session, site_ids, start, end, chunk=10) -> pd.DataFrame:
    names = [n for spec in config.DISCRETE_PARAMS.values() for n in spec["names"]]
    frames = []
    for i in range(0, len(site_ids), chunk):
        part = site_ids[i:i + chunk]
        df = _wqp_resilient(session, part, names, start, end)
        if not df.empty:
            frames.append(df)
        log.info("Lab samples: sites %d-%d of %d (%d rows)", i + 1, i + len(part), len(site_ids), len(df))

    cols = ["site_no", "date", "characteristic", "value", "unit", "fraction"]
    if not frames:
        log.warning("No lab samples retrieved; lab targets (lead, E. coli, ...) will be skipped.")
        return pd.DataFrame(columns=cols)
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={
        "MonitoringLocationIdentifier": "site_id", "ActivityStartDate": "date",
        "CharacteristicName": "characteristic", "ResultMeasureValue": "value",
        "ResultMeasure/MeasureUnitCode": "unit", "ResultSampleFractionText": "fraction"})
    df["site_no"] = df["site_id"].str.replace("USGS-", "", regex=False)
    return df[cols]


def run(state="NY", start="2015-01-01", end=None, max_sites=50, min_params=3, skip_discrete=False):
    end = end or date.today().isoformat()
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    session = make_session()

    sites = discover_sites(session, state, min_params=min_params, max_sites=max_sites)
    if sites.empty:
        raise SystemExit("No qualifying sites found; reduce --min-params or change --state.")
    sites.to_csv(config.RAW_DIR / "sites.csv", index=False)
    site_ids = sites["site_no"].tolist()

    sensor = fetch_daily_values(session, site_ids, start, end)
    sensor.to_csv(config.RAW_DIR / "sensor_daily.csv", index=False)
    log.info("Saved %d daily sensor rows", len(sensor))

    if not skip_discrete:
        try:
            disc = fetch_discrete_samples(session, site_ids, start, end)
            disc.to_csv(config.RAW_DIR / "discrete_samples.csv", index=False)
            log.info("Saved %d discrete sample rows", len(disc))
        except requests.RequestException as exc:
            log.error("Lab-sample download failed (%s). Sensor data is saved; continuing without "
                      "lab targets. Re-run with --skip-discrete to silence this.", exc)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state", default="NY", help="Two-letter state code")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--max-sites", type=int, default=50)
    p.add_argument("--min-params", type=int, default=3)
    p.add_argument("--skip-discrete", action="store_true")
    a = p.parse_args()
    run(a.state, a.start, a.end, a.max_sites, a.min_params, a.skip_discrete)
