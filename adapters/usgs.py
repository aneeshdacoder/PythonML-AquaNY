"""USGS adapter: daily sensor values (Water Services) + lab samples (Water Quality Portal),
converted to the standard observations/stations schema."""
import io
import logging
import re

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import config
import schema

log = logging.getLogger("usgs")
_BAD_QUALIFIERS = {"Ice", "Eqp", "Mnt", "Dis", "Ssn"}   # value not valid for analysis


def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=5, backoff_factor=1.5,
                  status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers["User-Agent"] = "water-quality-pipeline/2.0"
    return s


def parse_rdb(text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text), sep="\t", comment="#", dtype=str)
    return df.iloc[1:].reset_index(drop=True)      # drop the column-format row


def select_sites(cat: pd.DataFrame, target_codes, min_params=1, min_days=365, max_sites=50,
                 recent_days=30, today=None) -> pd.DataFrame:
    """Pick stations that (a) measure at least `min_params` of the parameters we FORECAST and
    (b) are still reporting them. Discharge and temperature are fetched as features but do not
    count: nearly every gauge has them, and counting them selects stations we cannot forecast."""
    today = pd.Timestamp(today) if today is not None else pd.Timestamp.today().normalize()
    cat = cat[(cat["data_type_cd"] == "dv") & cat["parm_cd"].isin(target_codes)].copy()
    cat["count_nu"] = pd.to_numeric(cat["count_nu"], errors="coerce")
    cat["end"] = pd.to_datetime(cat["end_date"], errors="coerce")
    cat = cat[(cat["count_nu"] >= min_days) & (cat["end"] >= today - pd.Timedelta(days=recent_days))]
    per = cat.groupby(["site_no", "parm_cd"])["count_nu"].max().reset_index()
    score = per.groupby("site_no").agg(n_params=("parm_cd", "nunique"),
                                       total_obs=("count_nu", "sum")).reset_index()
    score = score[score["n_params"] >= min_params]
    score = score.sort_values(["n_params", "total_obs"], ascending=False).head(max_sites)
    keep = [c for c in ("site_no", "station_nm", "site_tp_cd", "dec_lat_va", "dec_long_va",
                        "alt_va", "huc_cd") if c in cat]
    return score.merge(cat.drop_duplicates("site_no")[keep], on="site_no", how="left")


def discover_sites(session, state, min_params=1, min_days=365, max_sites=50, recent_days=30) -> pd.DataFrame:
    codes = list(config.USGS_CODES.values())                   # everything we will download
    targets = [config.USGS_CODES[p] for p, s in config.PARAMETERS.items()
               if s["target"] and p in config.USGS_CODES]     # only what we forecast
    params = {"format": "rdb", "stateCd": state, "parameterCd": ",".join(codes),
              "siteType": "ST", "hasDataTypeCd": "dv", "siteStatus": "active",
              "seriesCatalogOutput": "true"}
    r = session.get(config.USGS_SITE_URL, params=params, timeout=120)
    r.raise_for_status()
    cat = parse_rdb(r.text)
    sites = select_sites(cat, targets, min_params, min_days, max_sites, recent_days)
    log.info("Selected %d sites in %s (>= %d forecastable parameters, reporting in the last %d days)",
             len(sites), state, min_params, recent_days)
    if len(sites):
        names = {v: k for k, v in config.USGS_CODES.items()}
        sel = cat[cat["site_no"].isin(sites["site_no"]) & (cat["data_type_cd"] == "dv")
                  & cat["parm_cd"].isin(targets)]
        per = sel.groupby("parm_cd")["site_no"].nunique()
        log.info("Selected stations per forecastable parameter: %s",
                 ", ".join(f"{names[c]}={n}" for c, n in per.items()))
    return sites


def build_stations(session, sites: pd.DataFrame, chunk=50) -> pd.DataFrame:
    """Static station features: coordinates, type, HUC, elevation, drainage area."""
    df = pd.DataFrame({
        "station_id": sites["site_no"], "name": sites.get("station_nm"),
        "latitude": sites.get("dec_lat_va"), "longitude": sites.get("dec_long_va"),
        "site_type": sites.get("site_tp_cd"), "huc": sites.get("huc_cd"),
        "elevation_m": pd.to_numeric(sites.get("alt_va"), errors="coerce") * 0.3048})
    area = {}
    ids = sites["site_no"].tolist()
    for i in range(0, len(ids), chunk):
        try:
            r = session.get(config.USGS_SITE_URL, timeout=120, params={
                "format": "rdb", "sites": ",".join(ids[i:i + chunk]), "siteOutput": "expanded"})
            r.raise_for_status()
            exp = parse_rdb(r.text)
            if "drain_area_va" in exp:
                for sid, a in zip(exp["site_no"], pd.to_numeric(exp["drain_area_va"], errors="coerce")):
                    area[sid] = a * 2.58999            # sq mi -> sq km
        except requests.RequestException as exc:
            log.warning("Drainage-area lookup failed (%s); continuing without it", exc)
    df["drainage_area_km2"] = df["station_id"].map(area)
    return schema.normalize_stations(df)


def fetch_daily(session, site_ids, start, end, chunk=20) -> pd.DataFrame:
    code_to_param = {v: k for k, v in config.USGS_CODES.items()}
    rows = []
    for i in range(0, len(site_ids), chunk):
        part = site_ids[i:i + chunk]
        r = session.get(config.USGS_DV_URL, timeout=180, params={
            "format": "json", "sites": ",".join(part),
            "parameterCd": ",".join(config.USGS_CODES.values()),
            "statCd": "00003,00008", "startDT": start, "endDT": end})
        if r.status_code == 404:
            log.warning("No daily data for sites %s", part)
            continue
        r.raise_for_status()
        for ts in r.json()["value"]["timeSeries"]:
            site = ts["sourceInfo"]["siteCode"][0]["value"]
            param = code_to_param.get(ts["variable"]["variableCode"][0]["value"])
            nodata = float(ts["variable"].get("noDataValue", -999999))
            for block in ts["values"]:
                for v in block["value"]:
                    quals = set(v.get("qualifiers", []))
                    if param is None or quals & _BAD_QUALIFIERS:
                        continue
                    q = "P" if "P" in quals else ("A" if "A" in quals else "")
                    rows.append((site, v["dateTime"][:10], param, v["value"], nodata, q))
        log.info("Daily values: sites %d-%d of %d", i + 1, i + len(part), len(site_ids))
    df = pd.DataFrame(rows, columns=["station_id", "timestamp", "parameter", "value",
                                     "nodata", "quality_flag"])
    if df.empty:
        return schema.normalize_observations(df.drop(columns="nodata"))
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[df["value"] != df["nodata"]].drop(columns="nodata")
    return schema.normalize_observations(df)


# ----------------------------------------------------------- Water Quality Portal ---
def _wqp_request(session, site_part, names, start, end, optional=True):
    params = [("siteid", f"USGS-{s}") for s in site_part]
    params += [("characteristicName", n) for n in names]
    params += [("mimeType", "csv"), ("zip", "no"), ("dataProfile", "narrowResult"),
               ("startDateLo", pd.Timestamp(start).strftime("%m-%d-%Y")),
               ("endDateHi", pd.Timestamp(end).strftime("%m-%d-%Y"))]
    if optional:
        params += [("providers", "NWIS"), ("sampleMedia", "Water")]
    r = session.get(config.WQP_RESULT_URL, params=params, timeout=300)
    if r.status_code == 400:
        return None, (r.headers.get("Warning") or r.text[:300] or "(empty body)").strip()
    r.raise_for_status()
    if r.status_code == 204 or not r.text.strip():
        return pd.DataFrame(), None
    return pd.read_csv(io.StringIO(r.text), dtype=str, low_memory=False), None


_REJECT_RE = re.compile(r"characteristicName=(.+?) is not in the list", re.I)
_rejected = set()       # names WQP refused; remembered so later chunks don't retry them


def _wqp_resilient(session, site_part, names, start, end):
    """Query WQP; when it rejects a characteristic name, drop just that name and retry."""
    names = [n for n in names if n not in _rejected]
    while names:
        df, err = _wqp_request(session, site_part, names, start, end)
        if err is None:
            return df
        match = _REJECT_RE.search(err)
        bad = match.group(1).strip() if match else None
        if bad in names:
            log.warning("WQP does not recognise characteristic %r; dropping it for this run", bad)
            _rejected.add(bad)
            names = [n for n in names if n != bad]
            continue
        df, err2 = _wqp_request(session, site_part, names, start, end, optional=False)
        if err2 is None:
            return df
        log.warning("WQP 400 not attributable to a characteristic name: %s", err2)
        return pd.DataFrame()
    return pd.DataFrame()


def _wqp_to_observations(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=schema.OBS_COLUMNS)
    g = lambda c: raw[c] if c in raw else pd.Series("", index=raw.index)  # noqa: E731
    value = pd.to_numeric(g("ResultMeasureValue"), errors="coerce")
    unit = g("ResultMeasure/MeasureUnitCode").fillna("")
    cond = g("ResultDetectionConditionText").fillna("").str.lower()
    nd = cond.str.contains("not detected|below|<|non-detect", regex=True)
    limit = pd.to_numeric(g("DetectionQuantitationLimitMeasure/MeasureValue"), errors="coerce")
    limit_unit = g("DetectionQuantitationLimitMeasure/MeasureUnitCode").fillna("")
    use_limit = nd & value.isna() & limit.notna()    # non-detect: use the detection limit
    value = value.where(~use_limit, limit)
    unit = unit.where(~use_limit, limit_unit)
    return pd.DataFrame({
        "station_id": g("MonitoringLocationIdentifier").str.replace("USGS-", "", regex=False),
        "timestamp": g("ActivityStartDate"),
        "parameter": g("CharacteristicName").map(config.WQP_TO_PARAM),
        "value": value, "unit": unit,
        "detection_flag": nd.map({True: "ND", False: ""}), "quality_flag": ""})


def fetch_lab(session, site_ids, start, end, chunk=10) -> pd.DataFrame:
    names = list(config.WQP_TO_PARAM)
    frames = []
    for i in range(0, len(site_ids), chunk):
        part = site_ids[i:i + chunk]
        df = _wqp_resilient(session, part, names, start, end)
        if not df.empty:
            frames.append(df)
        log.info("Lab samples: sites %d-%d of %d (%d rows)", i + 1, i + len(part), len(site_ids), len(df))
    if not frames:
        log.warning("No lab samples retrieved; lab targets will be skipped.")
        return pd.DataFrame(columns=schema.OBS_COLUMNS)
    return schema.normalize_observations(_wqp_to_observations(pd.concat(frames, ignore_index=True)))


def collect(state, start, end, max_sites=50, min_params=1, lab=True, recent_days=30):
    session = make_session()
    sites = discover_sites(session, state, min_params=min_params, max_sites=max_sites,
                           recent_days=recent_days)
    if sites.empty:
        raise SystemExit("No qualifying sites found; relax --min-params or change --state.")
    stations = build_stations(session, sites)
    ids = sites["site_no"].tolist()
    parts = [fetch_daily(session, ids, start, end)]
    if lab:
        try:
            parts.append(fetch_lab(session, ids, start, end))
        except requests.RequestException as exc:
            log.error("Lab download failed (%s); continuing with sensor data only.", exc)
    return pd.concat(parts, ignore_index=True), stations
