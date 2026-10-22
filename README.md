# Water Quality Forecasting Pipeline (USGS)

(hopefully) Predicts water quality `HORIZON_DAYS` (default 7) ahead at USGS monitoring sites for:
turbidity, pH, nitrate, dissolved oxygen, E. coli, lead, sulfate, fluoride, hardness,
and specific conductance (salinity proxy).

Unfortunately, there was not enough data for most of the above. I only was able to find data for turbidity, dissolved oxygen, pH, and conductivity. 

## Layout
| File | Order of Execution | Purpose |

| `config.py` | - | Endpoints, parameter codes, thresholds, hyperparameters |
| `collect_data.py` | 1 | Site discovery + daily sensor data (USGS Water Services), lab samples (Water Quality Portal) |
| `build_features.py` | 2 | Unit conversion, QC, daily calendar, lags/rolling stats, seasonality, future targets |
| `train.py` | 3 | Per-target model selection (Ridge / RandomForest / HistGBM), holdout test, refit |
| `evaluate.py` | 4 | Skill vs. persistence, exceedance precision/recall, scatter plots |
| `predict.py` | 5 | Forecast from each site's latest data, flag threshold exceedances |
| `run_pipeline.py` | - | Runs any/all stages |
| `utils.py` | - | Time split, metrics, helpers |

## Run(I would recommed stage by stage to see and understand the data a little better)
```bash
pip install -r requirements.txt
python run_pipeline.py --state NY --start 2015-01-01 --max-sites 50
# or stage by stage:
python collect_data.py --state NY
python build_features.py
python train.py
python evaluate.py
python predict.py
```

## Design
- **Two data sources.** USGS continuous sensors only measure turbidity, pH, DO,
  conductance, temperature (and nitrate at a few sites). Lead, E. coli, sulfate,
  fluoride and hardness are discrete lab samples, pulled from the Water Quality
  Portal (which serves USGS NWIS results).
- **Sparse lab data.** Lab values are carried forward (max 90 days) as features,
  with a `_days_since` age feature. Lab targets are trained only on days where a
  sample was actually taken `horizon` days ahead.
- **No leakage.** Chronological split; training rows whose target date falls in the
  test period are purged. Rolling/lag features only use past data. CV uses `TimeSeriesSplit`.
- **Honest baseline.** Every target is compared to persistence ("same as today").
  `skill_vs_persistence > 0` means the model beats it.
- **Log-scale targets** for skewed concentrations (turbidity, E. coli, lead, etc.).
- **Evaluation uses holdout predictions** saved by `train.py`; the deployed models are
  refit on all data, so they must not be scored on the test period.

## Caveats/future directions
- Parameter codes (e.g. `63680` turbidity, `99133` nitrate) are defined in `config.py`;  Sites are chosen by data availability, not geography.
- Lead/E. coli/fluoride are often sparse and non-detect results are dropped; too little to create a reliable model. Might consider treating non-detects as half the detection limit.
- Thresholds in `config.THRESHOLDS` are screening values, not regulatory determinations.
- A pooled model across sites uses lat/lon only for site context; may consider per-region models or site-level features (drainage area, land use).
