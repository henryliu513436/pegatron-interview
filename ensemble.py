import pandas as pd
import numpy as np
from config import RAW_SENSOR_COLUMNS

def combine(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combines rule-based and ML-based anomaly flags into a final alert.

    Inputs:
        df: DataFrame containing rule_flag, rule_reason, ml_flag, ml_score
            and raw/scaled sensor columns.

    Outputs:
        DataFrame containing all original columns plus:
        - is_anomaly_final (bool): rule_flag OR ml_flag
        - severity (str): "CRITICAL" (both), "HIGH" (rule only), "MEDIUM" (ml only), else None
        - triggered_sensors (str): List of sensors that caused the alert.
    """
    df_res = df.copy()

    # 1. Final anomaly flag
    df_res["is_anomaly_final"] = df_res["rule_flag"] | df_res["ml_flag"]

    # 2. Severity logic
    def determine_severity(row):
        if row["rule_flag"] and row["ml_flag"]:
            return "CRITICAL"
        if row["rule_flag"]:
            return "HIGH"
        if row["ml_flag"]:
            return "MEDIUM"
        return None

    df_res["severity"] = df_res.apply(determine_severity, axis=1)

    # 3. Triggered Sensors logic
    def determine_sensors(row):
        # If rule triggered, extract sensor names from rule_reason (e.g., "temp_high;vibration_high" -> "temp;vibration")
        if row["rule_flag"]:
            reasons = row["rule_reason"].split(";")
            sensors = [r.split("_")[0] for r in reasons if r]
            return ";".join(sensors)

        # If only ML triggered (MEDIUM), find max abs scaled value among the three sensors
        if row["ml_flag"]:
            scaled_cols = [f"{s}_scaled" for s in RAW_SENSOR_COLUMNS]
            # Get values for the scaled columns
            vals = row[scaled_cols]
            # Find the sensor name corresponding to the max absolute value
            best_sensor = vals.abs().idxmax()
            # Convert "temp_scaled" -> "temp"
            return best_sensor.replace("_scaled", "")

        return ""

    df_res["triggered_sensors"] = df_res.apply(determine_sensors, axis=1)

    return df_res
