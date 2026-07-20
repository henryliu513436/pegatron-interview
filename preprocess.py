import pandas as pd
import numpy as np
from typing import Any, Tuple
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from config import (
    RAW_DATA_PATH, RAW_SENSOR_COLUMNS, MISSING_VALUE_STRATEGY,
    TRAIN_RATIO, CAL_RATIO, MIN_REQUIRED_SAMPLES, SCALER_TYPE, ML_FEATURE_COLUMNS
)

def load_raw(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Reads CSV, converts timestamp to pandas.Timestamp and sorts by it.
    Raises ValueError if timestamp or label contains NaNs.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Raw data file not found at: {path}")

    if "timestamp" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain 'timestamp' and 'label' columns")

    # Validate no NaNs in critical columns
    if df["timestamp"].isna().any() or df["label"].isna().any():
        raise ValueError("Critical columns 'timestamp' or 'label' contain NaNs")

    # Convert to datetime and sort
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles missing values in RAW_SENSOR_COLUMNS using the strategy defined in config.
    Purely causal strategy: forward fill (ffill) followed by backward fill (bfill) for leading NaNs.
    Raises ValueError if NaNs persist after processing.
    """
    df_cleaned = df.copy()

    for col in RAW_SENSOR_COLUMNS:
        if col not in df_cleaned.columns:
            continue

        if MISSING_VALUE_STRATEGY == "ffill_then_bfill":
            # Causal fill: Use previous known value
            df_cleaned[col] = df_cleaned[col].ffill()
            # Handle leading NaNs (no previous value exists)
            df_cleaned[col] = df_cleaned[col].bfill()
        else:
            # Fallback
            df_cleaned[col] = df_cleaned[col].ffill().bfill()

        # Final check for this column
        if df_cleaned[col].isna().any():
            raise ValueError(f"Column {col} still contains NaNs after missing value processing")

    return df_cleaned

def temporal_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    cal_ratio: float = CAL_RATIO,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits DF into train_fit (Block A), cal (Block B), and test (Block C) chronologically.
    - Block A: [0, train_ratio), keep only label=='normal'
    - Block B: [train_ratio, train_ratio+cal_ratio), keep only label=='normal'
    - Block C: [train_ratio+cal_ratio, 1.0], keep all
    """
    n = len(df)
    split_a = int(n * train_ratio)
    split_b = int(n * (train_ratio + cal_ratio))

    df_a = df.iloc[:split_a]
    df_b = df.iloc[split_a:split_b]
    df_c = df.iloc[split_b:]

    train_fit = df_a[df_a["label"] == "normal"].copy()
    cal = df_b[df_b["label"] == "normal"].copy()
    test = df_c.copy()

    # First check for minimum sample requirements
    if len(train_fit) < MIN_REQUIRED_SAMPLES or \
       len(cal) < MIN_REQUIRED_SAMPLES or \
       len(test) < MIN_REQUIRED_SAMPLES:
        raise ValueError(f"One or more split blocks have fewer than {MIN_REQUIRED_SAMPLES} samples")

    # Then check if test block contains any abnormal rows to avoid silent 0-metric evaluations
    if not (test["label"] == "abnormal").any():
        raise ValueError("The test segment (Block C) does not contain any abnormal rows. "
                         "This will lead to 0-metrics in evaluation. Please regenerate data.")

    return train_fit, cal, test

def fit_scaler(train_fit_df: pd.DataFrame) -> Any:
    """
    Fits a scaler on the ML_FEATURE_COLUMNS of the train_fit block.
    """
    # Select only the relevant features
    X_train = train_fit_df[ML_FEATURE_COLUMNS]

    if SCALER_TYPE == "standard":
        scaler = StandardScaler()
    elif SCALER_TYPE == "robust":
        scaler = RobustScaler()
    elif SCALER_TYPE == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unsupported SCALER_TYPE: {SCALER_TYPE}")

    scaler.fit(X_train)
    return scaler

def transform_features(df: pd.DataFrame, scaler: Any) -> pd.DataFrame:
    """
    Transforms ML_FEATURE_COLUMNS using the provided scaler.
    Appends results as {col}_scaled and preserves original columns.
    """
    df_transformed = df.copy()
    X = df[ML_FEATURE_COLUMNS]

    scaled_values = scaler.transform(X)

    for i, col in enumerate(ML_FEATURE_COLUMNS):
        df_transformed[f"{col}_scaled"] = scaled_values[:, i]

    return df_transformed
