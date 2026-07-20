import pandas as pd
from config import THRESHOLDS

def detect(df: pd.DataFrame, thresholds: dict = THRESHOLDS) -> pd.DataFrame:
    """
    Detects anomalies based on static rule thresholds.

    Inputs:
        df: DataFrame containing raw sensor values (temp, pressure, vibration).
        thresholds: Dictionary of thresholds for each sensor.

    Outputs:
        DataFrame with new columns:
        - rule_flag (bool): True if any sensor exceeds its abnormal threshold.
        - rule_reason (str): Semicolon-separated list of triggered rules (e.g., 'temp_high;vibration_high').
    """
    df_result = df.copy()

    # Initialize columns
    df_result["rule_flag"] = False
    df_result["rule_reason"] = ""

    # Iterate through sensors defined in thresholds
    for sensor, limits in thresholds.items():
        # Check High
        if "abnormal_high" in limits:
            high_thresh = limits["abnormal_high"]
            high_mask = df[sensor] > high_thresh
            df_result.loc[high_mask, "rule_flag"] = True
            # For reason, we append the sensor_high tag
            # Using a lambda or temporary series to handle row-wise concatenation
            df_result.loc[high_mask, "rule_reason"] = df_result.loc[high_mask, "rule_reason"].apply(
                lambda x: f"{x};{sensor}_high" if x else f"{sensor}_high"
            )

        # Check Low
        if "abnormal_low" in limits:
            low_thresh = limits["abnormal_low"]
            low_mask = df[sensor] < low_thresh
            df_result.loc[low_mask, "rule_flag"] = True
            df_result.loc[low_mask, "rule_reason"] = df_result.loc[low_mask, "rule_reason"].apply(
                lambda x: f"{x};{sensor}_low" if x else f"{sensor}_low"
            )

    return df_result
