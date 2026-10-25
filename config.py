"""Central configuration: paths, endpoints, the PARAMETER REGISTRY, and modeling settings.

To support a new water-quality parameter, add one entry to _RAW below. Nothing else needs
to change: schema validation, unit conversion, features, training, thresholds and the API
all read from this registry.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

# --- Endpoints (USGS is migrating legacy NWIS services; update here if they move) ---
USGS_SITE_URL = "https://waterservices.usgs.gov/nwis/site/"
USGS_DV_URL = "https://waterservices.usgs.gov/nwis/dv/"
WQP_RESULT_URL = "https://www.waterqualitydata.us/data/Result/search"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# --- Parameter registry ------------------------------------------------------------
# kind     sensor : dense time series (lags, rolling stats)
#          lab    : sparse series (last value carried forward + age); also used for
#                   sparse sensors like nitrate
#          driver : external drivers such as rainfall (windowed sums)
# unit     canonical unit; all inputs are converted to it (see schema.py)
# usgs     USGS parameter code for the daily-values service (optional)
# wqp      Water Quality Portal characteristic names (optional)
# lo/hi    physical QC bounds        log   model on log1p scale (right-skewed)
# extras   extra daily stats (min/max) used as features when sub-daily data exists
# daily    how sub-daily values collapse to a day ("mean" or "sum")
# storm    add storm-response features (for discharge)
# threshold  screening limits -> exceedance probabilities        target  forecast it?
_DEFAULTS = dict(kind="sensor", unit="", usgs=None, wqp=[], lo=0.0, hi=float("inf"),
                 log=False, extras=[], daily="mean", storm=False, threshold=None, target=True)

_RAW = {
    # NOTE: 5 FNU is a drinking-water guideline; many natural streams exceed it routinely.
    # Set threshold=None (or a value suited to your use) to avoid flood-of-flags in the output.
    "turbidity": dict(unit="FNU", usgs="63680", hi=5000, log=True, extras=["max"],
                      threshold={"max": 5}),
    "ph": dict(unit="pH", usgs="00400", hi=14, threshold={"min": 6.5, "max": 8.5}),
    "dissolved_oxygen": dict(unit="mg/L", usgs="00300", hi=25, extras=["min"],
                             threshold={"min": 5}),
    "specific_conductance": dict(unit="uS/cm", usgs="00095", hi=100000, log=True),
    "temperature": dict(unit="degC", usgs="00010", lo=-5, hi=45, target=False),
    "discharge": dict(unit="ft3/s", usgs="00060", hi=1e7, log=True, extras=["max"],
                      storm=True, target=False),
    "precipitation": dict(kind="driver", unit="mm", hi=1000, daily="sum", target=False),
    "nitrate": dict(kind="lab", unit="mg/L", usgs="99133", hi=1000, log=True,
                    wqp=["Nitrate", "Inorganic nitrogen (nitrate and nitrite)"],
                    threshold={"max": 10}),
    "e_coli": dict(kind="lab", unit="MPN/100mL", hi=1e7, log=True,
                   wqp=["Escherichia coli"], threshold={"max": 235}),
    "lead": dict(kind="lab", unit="ug/L", hi=1e5, log=True, wqp=["Lead"],
                 threshold={"max": 15}),
    "sulfate": dict(kind="lab", unit="mg/L", hi=1e5, log=True, wqp=["Sulfate"],
                    threshold={"max": 250}),
    "fluoride": dict(kind="lab", unit="mg/L", hi=1000, log=True, wqp=["Fluoride"],
                     threshold={"max": 4.0}),
    "hardness": dict(kind="lab", unit="mg/L", hi=1e5,
                     wqp=["Total hardness"]),   # "Hardness, Ca + Mg" is rejected by WQP
}
PARAMETERS = {k: {**_DEFAULTS, **v} for k, v in _RAW.items()}

# Names your app's data might use -> registry name (matching ignores case/underscores)
PARAM_ALIASES = {
    "do": "dissolved_oxygen", "dissolved oxygen": "dissolved_oxygen", "diss oxygen": "dissolved_oxygen",
    "turb": "turbidity", "ph level": "ph",
    "conductivity": "specific_conductance", "specific conductance": "specific_conductance",
    "spc": "specific_conductance", "conductance": "specific_conductance",
    "temp": "temperature", "water temperature": "temperature", "water temp": "temperature",
    "flow": "discharge", "streamflow": "discharge", "stream flow": "discharge",
    "rain": "precipitation", "rainfall": "precipitation", "precip": "precipitation",
    "nitrate n": "nitrate", "no3": "nitrate", "no3 n": "nitrate",
    "e coli": "e_coli", "ecoli": "e_coli", "escherichia coli": "e_coli",
    "pb": "lead", "so4": "sulfate", "f": "fluoride", "total hardness": "hardness",
}

STATION_TYPES = ["stream", "lake", "spring", "groundwater", "estuary", "other"]

# --- Derived views of the registry --------------------------------------------------
ALL_PARAMS = list(PARAMETERS)
TARGETS = [p for p, s in PARAMETERS.items() if s["target"]]
THRESHOLDS = {p: s["threshold"] for p, s in PARAMETERS.items() if s["threshold"]}
USGS_CODES = {p: s["usgs"] for p, s in PARAMETERS.items() if s["usgs"]}
WQP_TO_PARAM = {n: p for p, s in PARAMETERS.items() for n in s["wqp"]}

# --- Modeling -----------------------------------------------------------------------
HORIZONS = (1, 3, 7, 30)        # forecast lead times in days (one model per target x horizon)
LAGS = (1, 3, 7, 14)
ROLL_WINDOWS = (7, 30)
RAIN_WINDOWS = (1, 3, 7, 14, 30)
DISCRETE_FFILL_DAYS = 90
SENSOR_MAX_AGE_DAYS = 7         # a sensor reading older than this is NOT a usable 'current value'
LAB_TARGET_TOL_DAYS = 3         # lab targets: nearest sample within +/- this many days
TEST_FRACTION = 0.2             # most recent 20% of the timeline = temporal holdout
HOLDOUT_SITE_FRACTION = 0.15    # stations never seen in training = spatial holdout
MIN_SITES_FOR_SPATIAL = 8
CANDIDATES = ("ridge", "hist_gbm")      # add "random_forest" for small datasets
BASELINE_CANDIDATES = ("persistence", "climatology", "site_climatology")   # naive forecasts must be beaten to be chosen
HGB_MAX_ITER = 300
CV_SPLITS = 4
INTERVAL_COVERAGE = 0.8         # conformal prediction interval nominal coverage
SITE_OFFSET_DROPOUT = 0.3       # share of training stations whose site offset is hidden (mimics new stations)
MAX_TRAIN_ROWS = 300_000        # subsample very large targets (keeps training tractable)
MIN_TRAIN_ROWS = 150            # fewer rows than this and metrics are noise, not evidence
MIN_TEST_ROWS = 30
MIN_CLASSIFIER_AUC = 0.65       # holdout AUC below this -> exceedance classifier is NOT deployed
MIN_CLASS_ROWS = 15             # min positives AND negatives to fit an exceedance classifier
CLIM_K_SITE = 10                # climatology shrinkage: site -> group pseudo-count
CLIM_K_GROUP = 30               # group -> global pseudo-count
SEED = 42

# --- Monitoring ---------------------------------------------------------------------
MONITOR_WINDOW_DAYS = 90
MONITOR_MIN_MATURED = 5
ERROR_RATIO_ALERT = 1.5         # live MAE / holdout MAE above this = drift
STATION_STALE_DAYS = 14
MODEL_MAX_AGE_DAYS = 90
