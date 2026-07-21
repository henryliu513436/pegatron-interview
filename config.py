# config.py
# Central configuration for the Smart Factory Alert Agent

# ---------- Data generation ----------
N_ROWS = 500
ANOMALY_RATIO = 0.10
MISSING_RATIO = 0.02          # Only applied to temp/pressure/vibration
INTERVAL_MINUTES = 1
START_TIME = "2024-06-03 19:00:00"
RANDOM_SEED = 42

# Normal distribution parameters for sensors: (mean, std)
# Used for np.clip(rng.normal(mean, std), normal_min, normal_max)
NORMAL_DIST_PARAMS = {
    "temp":      (47.5, 1.0),
    "pressure":  (1.025, 0.012),
    "vibration": (0.03, 0.005),
}

THRESHOLDS = {
    "temp":      {"normal_min": 45.0, "normal_max": 50.0, "abnormal_high": 52.0, "abnormal_low": 43.0},
    "pressure":  {"normal_min": 1.00, "normal_max": 1.05, "abnormal_high": 1.08, "abnormal_low": 0.97},
    "vibration": {"normal_max": 0.04, "abnormal_high": 0.07},
}

# ---------- Temporal split (Total ratio must be 1.0, split by chronological order) ----------
TRAIN_RATIO = 0.70   # Block A: fit scaler + IsolationForest (normal only)
CAL_RATIO = 0.15     # Block B: threshold calibration (normal only, must be later than Block A)
# Remaining ~0.15 is Block C (test, mixed normal+abnormal)

# ---------- General Constraints ----------
MIN_REQUIRED_SAMPLES = 10

# ---------- Generation Ranges (derived from THRESHOLDS to ensure no-gap/no-overlap) ----------
# Format: {sensor: { "normal": (min, max), "min_limit": (min, max), "max_limit": (min, max) }}
# Note: Vibration only has "normal" and "max_limit" (no "min_limit" or abnormal_low equivalent).
GEN_RANGES = {
    "temp": {
        "normal": (45.0, 50.0),
        "min_limit": 35.0,
        "max_limit": 60.0,
    },
    "pressure": {
        "normal": (1.00, 1.05),
        "min_limit": 0.80,
        "max_limit": 1.20,
    },
    "vibration": {
        "normal": (0.02, 0.04),
        "max_limit": 0.15, # Vibration only generates abnormal_high
    },
}

# ---------- Preprocessing ----------
MISSING_VALUE_STRATEGY = "ffill_then_bfill"
SCALER_TYPE = "standard"      # standard | robust | minmax

# ---------- Feature engineering ----------
ROLLING_WINDOW = 5

RAW_SENSOR_COLUMNS = ["temp", "pressure", "vibration"]
# Features produced by features.py must match these names:
# {sensor}_rolling_mean, {sensor}_rolling_std, {sensor}_diff
ML_FEATURE_COLUMNS = (
    RAW_SENSOR_COLUMNS
    + [f"{c}_rolling_mean" for c in RAW_SENSOR_COLUMNS]
    + [f"{c}_rolling_std" for c in RAW_SENSOR_COLUMNS]
    + [f"{c}_diff" for c in RAW_SENSOR_COLUMNS]
)

# ---------- ML detector (IsolationForest) ----------
IF_N_ESTIMATORS = 100
IF_CONTAMINATION = 0.01       # Internal regularization only, not the actual alert threshold
IF_RANDOM_STATE = 42
THRESHOLD_PERCENTILE = 5      # Use 5th percentile of cal set decision_function for ml threshold

# ---------- LLM advisor ----------
OLLAMA_MODEL = "gemma4:e4b"
OLLAMA_HOST = "http://localhost:11434"
LLM_TIMEOUT_SECONDS = 30
LLM_MAX_RETRIES = 1

# ---------- CLI ----------
DEFAULT_REPLAY_SPEED = 0.0    # Seconds; 0 = no simulation, >0 = interval between alerts

# ---------- Paths ----------
RAW_DATA_PATH = "data/raw_data.csv"
ALERTS_LOG_PATH = "logs/alerts.log"
CONFUSION_MATRIX_PATH = "docs/confusion_matrices.png"
AI_USAGE_LOG_PATH = "docs/ai_usage_log.md"
