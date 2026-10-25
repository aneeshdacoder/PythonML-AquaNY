"""FastAPI app exposing forecasts to a website.

Run:  uvicorn api:app --reload        (interactive docs at /docs)
Env:  WQ_CORS_ORIGINS  comma-separated allowed website origins (default http://localhost:3000)
      WQ_API_KEY       secret required by /admin/refresh

Training and heavy data collection never run inside a request. Schedule them (cron):
  nightly : python run_pipeline.py --stages ingest features predict monitor --append
  weekly+ : python run_pipeline.py --stages train evaluate     (or when /monitoring says so)
"""
import logging
import os
import secrets
import threading
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Response
from pydantic import BaseModel, Field

import build_features
import config
import ingest
from service import ForecastService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("api")

state = {}
_refresh_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["svc"] = ForecastService()          # load models once at startup, not per request
    yield
    state.clear()


app = FastAPI(title="Water Quality Forecast API", version="2.0.0", lifespan=lifespan)
origins = [o.strip() for o in os.getenv("WQ_CORS_ORIGINS", "http://localhost:3000").split(",")]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"],
                   allow_headers=["*"])


def svc() -> ForecastService:
    s = state["svc"]
    s.refresh_data()                          # picks up a rebuilt features table automatically
    return s


def _csv(value: Optional[str], cast=str):
    return [cast(v) for v in value.split(",") if v.strip()] if value else None


# ------------------------------------------------------------------ schemas ---
class SiteInfo(BaseModel):
    site_no: str
    name: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    site_type: Optional[str] = None
    latest_data_date: str


class Reliability(BaseModel):
    confidence: str = "unknown"   # high / medium / low / unknown, or "none" for fallback rows (not a forecast)
    caveats: list[str] = []
    mae_known_stations: Optional[float] = None
    mae_new_stations: Optional[float] = None
    skill_vs_persistence: Optional[float] = None
    skill_vs_climatology: Optional[float] = None
    interval_coverage: Optional[float] = None
    auc: Optional[float] = None


class ForecastItem(BaseModel):
    parameter: str
    unit: Optional[str] = None
    horizon_days: int
    forecast_date: str
    prediction: Optional[float] = None
    lower: Optional[float] = None
    upper: Optional[float] = None
    probability_exceedance: Optional[float] = None
    basis: str = "recent_data"   # or stale_data_fallback / never_measured_fallback (NOT a real forecast)
    data_age_days: Optional[float] = None
    exceeds_screening_threshold: Optional[bool] = None
    threshold: Optional[dict] = None
    model: str
    reliability: Reliability


class SiteForecast(BaseModel):
    site_no: str
    data_date: str
    data_age_days: Optional[int] = None
    stale: bool
    forecasts: list[ForecastItem]


class HistoryPoint(BaseModel):
    date: str
    value: Optional[float] = None


class ObservationIn(BaseModel):
    date: str
    parameter: str
    value: float
    unit: str = ""


class NewStationIn(BaseModel):
    latitude: float
    longitude: float
    site_type: str = "stream"
    huc: Optional[str] = None
    elevation_m: Optional[float] = None
    drainage_area_km2: Optional[float] = None
    observations: list[ObservationIn] = []


class NewStationForecast(BaseModel):
    data_basis: str
    data_date: str
    forecasts: list[ForecastItem]


# ---------------------------------------------------------------- endpoints ---
@app.get("/", include_in_schema=False)
def root():
    return {"status": "ok", "docs": "/docs", "health": "/health"}

@app.get("/health")
def health():
    s = svc()
    return {"status": "ok", "models": len(s.bundles), "horizons": s.horizons, "sites": len(s.latest)}


@app.get("/sites", response_model=list[SiteInfo])
def list_sites():
    s = svc()
    return [s.site_info(x) for x in s.site_ids()]

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.get("/sites/{site_no}/forecast", response_model=SiteForecast)
def site_forecast(site_no: str,
                  horizons: Optional[str] = Query(None, description="comma list, e.g. 1,7"),
                  parameters: Optional[str] = Query(None, description="comma list, e.g. ph,lead")):
    try:
        return svc().forecast(site_no, _csv(horizons, int), _csv(parameters))
    except KeyError:
        raise HTTPException(404, f"Unknown site {site_no}")


@app.post("/forecast/new-station", response_model=NewStationForecast)
def new_station_forecast(body: NewStationIn,
                         horizons: Optional[str] = Query(None),
                         parameters: Optional[str] = Query(None)):
    """Forecast for a station that has no history in the system. Send static metadata and,
    optionally, recent readings (long format) to anchor the forecast on the station's level."""
    meta = body.model_dump(exclude={"observations"})
    try:
        return svc().forecast_new_station(meta, [o.model_dump() for o in body.observations],
                                          _csv(horizons, int), _csv(parameters))
    except ValueError as exc:
        raise HTTPException(422, str(exc))

class BatchItemIn(BaseModel):
    id: str                                   # your own station ID, echoed back
    new_station: Optional[NewStationIn] = None  # used only if `id` isn't one of the covered sites


class BatchIn(BaseModel):
    stations: list[BatchItemIn] = Field(..., min_length=1, max_length=200)


class BatchResult(BaseModel):
    id: str
    status: str                               # "ok" or "error"
    source: Optional[str] = None              # "known_station" or "new_station"
    error: Optional[str] = None
    data_date: Optional[str] = None
    data_age_days: Optional[int] = None
    stale: Optional[bool] = None
    data_basis: Optional[str] = None
    forecasts: list[ForecastItem] = []


class BatchOut(BaseModel):
    n_ok: int
    n_error: int
    results: list[BatchResult]


@app.post("/forecast/batch", response_model=BatchOut)
def forecast_batch(body: BatchIn,
                   horizons: Optional[str] = Query(None),
                   parameters: Optional[str] = Query(None)):
    """Forecast many stations in one call. Covered stations use their own history;
    others need `new_station` metadata. One bad station never fails the batch."""
    s = svc()                                  # refresh once for the whole batch
    hz, pr = _csv(horizons, int), _csv(parameters)
    known = {str(x) for x in s.site_ids()}
    results = []
    for item in body.stations:
        try:
            if item.id in known:
                r = s.forecast(item.id, hz, pr)
                results.append(BatchResult(
                    id=item.id, status="ok", source="known_station",
                    data_date=r["data_date"], data_age_days=r.get("data_age_days"),
                    stale=r.get("stale"), forecasts=r["forecasts"]))
            elif item.new_station is not None:
                ns = item.new_station
                r = s.forecast_new_station(
                    ns.model_dump(exclude={"observations"}),
                    [o.model_dump() for o in ns.observations], hz, pr)
                results.append(BatchResult(
                    id=item.id, status="ok", source="new_station",
                    data_date=r["data_date"], data_basis=r["data_basis"],
                    forecasts=r["forecasts"]))
            else:
                results.append(BatchResult(
                    id=item.id, status="error",
                    error="Not a covered station and no new_station metadata supplied"))
        except (KeyError, ValueError) as exc:
            results.append(BatchResult(id=item.id, status="error", error=str(exc)))
        except Exception:
            log.exception("Batch item %s failed", item.id)
            results.append(BatchResult(id=item.id, status="error", error="Internal error"))
    n_ok = sum(r.status == "ok" for r in results)
    return BatchOut(n_ok=n_ok, n_error=len(results) - n_ok, results=results)


@app.get("/sites/{site_no}/history", response_model=list[HistoryPoint])
def site_history(site_no: str, parameter: str = Query(...), days: int = Query(180, ge=1, le=3650)):
    if parameter not in config.ALL_PARAMS:
        raise HTTPException(422, f"parameter must be one of {config.ALL_PARAMS}")
    try:
        return svc().history_series(site_no, parameter, days)
    except KeyError:
        raise HTTPException(404, f"Unknown site {site_no}")


@app.get("/models")
def models():
    """Holdout accuracy per parameter/horizon, including accuracy at never-seen stations."""
    return svc().model_report()


@app.get("/monitoring")
def monitoring():
    """Live accuracy vs. holdout, drift, stale stations, and whether retraining is advised."""
    report = svc().monitoring()
    if report is None:
        raise HTTPException(404, "No monitoring report yet; run monitor.py")
    return report


# -------------------------------------------------------------------- admin ---
def require_key(x_api_key: str = Header(default="")):
    expected = os.getenv("WQ_API_KEY", "")
    if not expected or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(401, "Invalid or missing API key")


def _refresh_job(state_code: str, start: str):
    if not _refresh_lock.acquire(blocking=False):
        log.warning("Refresh already running; skipping")
        return
    try:
        ingest.run(source="usgs", state=state_code, start=start, append=True)
        build_features.run()
        state["svc"].refresh_data(force=True)
        log.info("Refresh complete")
    except Exception:
        log.exception("Refresh failed")
    finally:
        _refresh_lock.release()


@app.post("/admin/refresh", dependencies=[Depends(require_key)])
def refresh(background: BackgroundTasks, state_code: str = "NY", start: str = "2023-01-01"):
    """Download new USGS data and rebuild features (does NOT retrain). Prefer a cron job."""
    background.add_task(_refresh_job, state_code, start)
    return {"status": "started"}