"""Central configuration for the water-quality forecasting pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

# --- Endpoints (kept here because USGS is migrating legacy NWIS services to
# --- api.waterdata.usgs.gov; if these stop working, only this block changes).
USGS_SITE_URL = "https://waterservices.usgs.gov/nwis/site/"
USGS_DV_URL = "https://waterservices.usgs.gov/nwis/dv/"
WQP_RESULT_URL = "https://www.waterqualitydata.us/data/Result/search"

# --- Continuous in-situ sensor parameters (USGS daily values service) -------
SENSOR_PARAMS = {
    "turbidity": "63680",             # FNU
    "ph": "00400",                    # standard units
    "dissolved_oxygen": "00300",      # mg/L
    "specific_conductance": "00095",  # uS/cm, our salinity proxy
    "temperature": "00010",           # deg C (feature only)
    "nitrate": "99133",               # nitrate+nitrite, mg/L as N (sparse)
}

# --- Discrete lab samples (Water Quality Portal, which serves USGS NWIS) -----
# Canonical units: lead ug/L, e_coli MPN/100mL, everything else mg/L.
DISCRETE_PARAMS = {
    "e_coli":   {"names": ["Escherichia coli"], "unit": "MPN/100mL"},
    "lead":     {"names": ["Lead"], "unit": "ug/L"},
    "sulfate":  {"names": ["Sulfate"], "unit": "mg/L"},
    "fluoride": {"names": ["Fluoride"], "unit": "mg/L"},
    "hardness": {"names": ["Hardness, Ca + Mg", "Total hardness"], "unit": "mg/L"},
    "nitrate":  {"names": ["Nitrate", "Inorganic nitrogen (nitrate and nitrite)"],
                 "unit": "mg/L"},  # lab values fill gaps in the sensor series
}
CHAR_TO_PARAM = {n: p for p, s in DISCRETE_PARAMS.items() for n in s["names"]}

ALL_PARAMS = list(dict.fromkeys([*SENSOR_PARAMS, *DISCRETE_PARAMS]))
TARGETS = [p for p in ALL_PARAMS if p != "temperature"]
LOG_TARGETS = {"turbidity", "nitrate", "e_coli", "lead", "sulfate", "fluoride"}

# Physical sanity bounds (QC). Parameters not listed must simply be >= 0.
BOUNDS = {"ph": (0, 14), "dissolved_oxygen": (0, 25), "temperature": (-5, 45)}

# Screening thresholds for exceedance flags. These are NOT regulatory
# determinations; verify against your state's standards before using them.
THRESHOLDS = {
    "lead":             {"max": 15},      # EPA action level, ug/L
    "nitrate":          {"max": 10},      # EPA MCL, mg/L as N
    "fluoride":         {"max": 4.0},     # EPA MCL, mg/L
    "sulfate":          {"max": 250},     # EPA secondary standard, mg/L
    "e_coli":           {"max": 235},     # EPA recreational single-sample, MPN/100mL
    "turbidity":        {"max": 5},       # secondary/aesthetic guideline
    "ph":               {"min": 6.5, "max": 8.5},
    "dissolved_oxygen": {"min": 5},       # common aquatic-life criterion, mg/L
}

# --- Modeling ---------------------------------------------------------------
HORIZON_DAYS = 7            # forecast lead time
LAGS = (1, 3, 7, 14)
ROLL_WINDOWS = (7, 30)
DISCRETE_FFILL_DAYS = 90    # how long a lab value is carried forward as a feature
TEST_FRACTION = 0.2         # most recent 20% of the timeline is the holdout
CV_SPLITS = 4
MIN_TRAIN_ROWS = 60
MIN_TEST_ROWS = 10
SEED = 42
