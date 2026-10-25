# Water Quality Forecasting Pipeline v2

Forecasts any registered water-quality parameter at any station, at several lead times
(default 1, 3, 7, 30 days), with prediction intervals, exceedance probabilities, support for
brand-new stations, and built-in monitoring.

## Architecture
```
 source adapters ──► standard schema ──► features ──► train ──► predict ──► monitor
 (usgs / csv / yours)  (schema.py)      (daily agg,   (anomaly   (+ log)     (drift, stale,
                                         static, rain, model, CI,             retrain advice)
                                         flow, lags)   exceedance)
```
| File | Role |
|---|---|
| `config.py` | **Parameter registry** (units, bounds, thresholds, log-scale, kind) + all settings |
| `schema.py` | Standard `observations` / `stations` contract; unit conversion, QC, non-detect handling |
| `adapters/usgs.py` | USGS daily values + Water Quality Portal lab samples |
| `adapters/csv_adapter.py` | **Your app's data** via a JSON column-mapping (long or wide CSV) |
| `adapters/weather.py` | Rainfall (Open-Meteo) as a driver |
| `ingest.py` | Runs an adapter, writes standard tables |
| `build_features.py` | Daily aggregation (from sub-daily), lags/rolling, storm/rain features, static station features, multi-horizon targets |
| `climatology.py` | Hierarchical seasonal "normal" (station → region → global) |
| `modeling.py` | Model zoo + shared feature/prediction logic |
| `train.py` | Per parameter×horizon: anomaly regressor, conformal intervals, exceedance classifier, spatial holdout |
| `evaluate.py` | Known-vs-new-station metrics, skill vs persistence & climatology |
| `predict.py` | Forecasts for all horizons; appends to a forecast log |
| `diagnose.py` | Which parameters actually have data, where it ends, which targets are trainable |
| `monitor.py` | Live accuracy vs holdout, drift, stale stations, model age, retrain advice |
| `service.py`, `api.py` | FastAPI serving incl. `POST /forecast/new-station` and `GET /monitoring` |

## Connecting your app's data
1. **CSV export** (fastest): copy `examples/mapping_example.json`, point `columns` at your
   column names, then
   `python run_pipeline.py --source csv --file export.csv --mapping my_mapping.json --stations-file stations.csv --weather`
   Wide format (one column per parameter) is supported via `wide`/`parameter_columns`.
2. **Database / API**: write a function returning a DataFrame with the `observations`
   columns (`station_id, timestamp, parameter, value, unit, detection_flag, quality_flag`)
   and pass it through `schema.normalize_observations`. Nothing downstream changes.
3. **Stations table**: `station_id, name, latitude, longitude, site_type, huc, drainage_area_km2,
   elevation_m` plus any numeric `static_*` columns (e.g. `static_pct_forest`). More static
   context = better forecasts at new stations.

## Adding a parameter
Add one entry to `_RAW` in `config.py` (unit, bounds, kind, optional threshold). Unit
conversions for new unit families go in `schema._UNITS`.

## Design notes
- **Anomaly target.** Models predict `value − regional seasonal normal` (log scale for skewed
  parameters). The region-level normal exists for every station, including new ones; a
  station's own level comes from its recent readings.
- **New stations.** Brand-new stations are forecast from static metadata + regional normal; if
  you supply recent readings they anchor the forecast. `train.py` withholds ~15% of stations
  entirely and reports `new_*` metrics so you know how well this works.
- **Baselines compete.** Persistence ("same as now") and climatology ("normal for the season") are
  candidates in model selection, so a learned model is only used where it beats them in CV. If
  `best_model` is `persistence` or `climatology` for a target, nothing learned adds value there.
  Every metric is also reported as skill against both baselines.
- **No leakage in the station offset.** Training rows get their station offset from every OTHER
  calendar year (out-of-year), so models are not rewarded for memorising the answer.
- **Site offset dropout.** The station's own seasonal offset is a feature for known stations, but is
  hidden for ~30% of training stations so the model also works when it is unavailable (new stations).
- **Per-target timeline.** Each target's train/test cutoff comes from its own labelled dates, so
  sparse lab data that ends early is still evaluated properly.
- **Intervals.** Split-conformal intervals from out-of-fold residuals (nominal 80%). Check
  `interval_coverage` in the evaluation table; `monitor.py` flags under-coverage live.
- **Exceedance probability.** Calibrated classifier for parameters with a screening
  threshold (only trained when enough exceedances exist in the data).
- **Non-detects** are kept as half the detection limit (censored) instead of being dropped.
- **Lab targets** use the nearest sample within ±3 days of the target date.
- **Drivers.** Discharge (USGS 00060) and rainfall give storm-response features; rain is
  usually the single biggest gain for turbidity, E. coli and nitrate.

## Real forecasts vs fallbacks
A forecast needs a recent reading of that parameter (sensors: `SENSOR_MAX_AGE_DAYS`, lab: 90 days).
Otherwise the row is a seasonal-normal fallback with a wider interval and **no exceedance
probability**. `basis` says which kind:
- `recent_data` - a real forecast
- `stale_data_fallback` - the station measures it but has stopped reporting (station's own normal)
- `never_measured_fallback` - the station has no history for it (regional normal, widest interval)

Always filter on `basis == "recent_data"` when you need forecasts. `predict.py` prints how many
stations have a real forecast per parameter, and `monitor.py` reports coverage per parameter.
Station freshness ignores rainfall, which is current everywhere and would hide dead sensors.

## Choosing stations (USGS)
Models are per parameter, so a station only needs to measure ONE forecastable parameter (turbidity,
pH, DO, conductance, nitrate) and still be reporting it (`--recent-days`, default 30). `--min-params`
(default 1) raises that bar, but be careful: in NY, requiring 3 forecastable parameters left only 3
stations, too few to train or hold anything out. More stations per parameter = better models and a
real new-station test (needs >= 8 stations). Ranking prefers stations measuring more parameters, so a
larger `--max-sites` is how you widen coverage. Discharge and temperature are downloaded as features
but never qualify a station. The log prints how many selected stations measure each parameter.

## Guards against misleading results
- Targets need at least `MIN_TRAIN_ROWS` (150) and `MIN_TEST_ROWS` (30) rows or they are skipped.
- An exceedance classifier is deployed only if its holdout AUC >= `MIN_CLASSIFIER_AUC` (0.65);
  otherwise `p_exceed` is blank and `metrics.json` records `classifier_dropped`.
- Metrics on tiny samples (a few dozen rows, a few stations) are noise: check `n_test` and
  the number of stations before trusting any skill, coverage or AUC number.

## Diagnosing missing data
`python diagnose.py` shows, per parameter, row counts, stations, first/last date and rows in the
last year, plus labelled rows per target. Use it first when targets are skipped.

## Operating it
```
nightly : python run_pipeline.py --stages ingest features predict monitor --append
weekly  : python run_pipeline.py --stages train evaluate        # or when monitor says retrain_recommended
serve   : uvicorn api:app
```
`pip install -r requirements.txt` (and `requirements-api.txt` for the API).

## Limitations (read before production)
  `service.py`, which holds the logic, was exercised directly.
- Forecasts of future *rainfall* are not used; only observed rain. Adding a weather-forecast
  feed would improve 1–7 day turbidity/E. coli forecasts.
- Tables are in memory (Parquet if `pyarrow` is installed). For tens of thousands of stations,
  move `observations`/`features` to a database or partitioned Parquet and train on a sample
  (`MAX_TRAIN_ROWS`); the per-station feature builder already processes one station at a time.
- Screening thresholds in `config.py` are not regulatory determinations.
- Skill depends on the data: lead, fluoride, hardness and other slow/noisy lab parameters often
  show little skill over climatology. The evaluation table tells you which targets to trust.
