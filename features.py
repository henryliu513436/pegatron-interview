import pandas as pd
from config import ROLLING_WINDOW, RAW_SENSOR_COLUMNS

def add_rolling_features(df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """
    Calculates causal rolling features for RAW_SENSOR_COLUMNS.
    - {col}_rolling_mean: rolling mean with window, min_periods=1
    - {col}_rolling_std: rolling std with window, min_periods=1, fillna(0)
    - {col}_diff: first-order difference, fillna(0)

    Causality: Only historical data up to the current row is used.
    Must be called on the full, sorted time series before temporal splitting.
    """
    df_feat = df.copy()

    for col in RAW_SENSOR_COLUMNS:
        # Rolling Mean
        df_feat[f"{col}_rolling_mean"] = df[col].rolling(window=window, min_periods=1).mean()

        # Rolling Std
        df_feat[f"{col}_rolling_std"] = df[col].rolling(window=window, min_periods=1).std().fillna(0)

        # Difference
        df_feat[f"{col}_diff"] = df[col].diff().fillna(0)

    return df_feat
