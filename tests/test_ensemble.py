import pytest
import pandas as pd
import numpy as np
from ensemble import combine

def test_combine_severity_logic():
    """
    Test the ensemble logic for all four combinations of rule and ml flags.
    """
    data = {
        "temp": [47.0, 47.0, 47.0, 47.0],
        "pressure": [1.02, 1.02, 1.02, 1.02],
        "vibration": [0.03, 0.03, 0.03, 0.03],
        "temp_scaled": [0.0, 0.0, 0.0, 0.0],
        "pressure_scaled": [0.0, 0.0, 0.0, 0.0],
        "vibration_scaled": [0.0, 0.0, 0.0, 0.0],
        "rule_flag": [True, True, False, False],
        "ml_flag": [True, False, True, False],
        "rule_reason": ["temp_high", "temp_high", "", ""],
        "ml_score": [-0.1, -0.1, -0.1, -0.1],
        "label": ["abnormal", "abnormal", "abnormal", "normal"]
    }
    df = pd.DataFrame(data)

    result = combine(df)

    # 1. Verify no filtering (input 4 rows -> output 4 rows)
    assert len(result) == len(df)

    # Expected results:
    # Row 0: Rule=T, ML=T -> CRITICAL, is_anomaly=True
    # Row 1: Rule=T, ML=F -> HIGH,     is_anomaly=True
    # Row 2: Rule=F, ML=T -> MEDIUM,   is_anomaly=True
    # Row 3: Rule=F, ML=F -> None,     is_anomaly=False

    assert result.iloc[0]["severity"] == "CRITICAL"
    assert result.iloc[0]["is_anomaly_final"] == True

    assert result.iloc[1]["severity"] == "HIGH"
    assert result.iloc[1]["is_anomaly_final"] == True

    assert result.iloc[2]["severity"] == "MEDIUM"
    assert result.iloc[2]["is_anomaly_final"] == True

    assert pd.isna(result.iloc[3]["severity"]) or result.iloc[3]["severity"] is None
    assert result.iloc[3]["is_anomaly_final"] == False

def test_combine_triggered_sensors():
    """
    Verify triggered_sensors logic:
    - Rule trigger: use rule_reason
    - ML only trigger: use max abs scaled value
    """
    data = {
        "temp": [47.0, 47.0, 47.0],
        "pressure": [1.02, 1.02, 1.02],
        "vibration": [0.03, 0.03, 0.03],
        "temp_scaled": [0.1, 2.0, 0.1],      # Row 1: max abs
        "pressure_scaled": [0.1, 0.1, 0.1],
        "vibration_scaled": [0.1, 0.1, 5.0], # Row 2: max abs
        "rule_flag": [True, False, False],
        "ml_flag": [True, True, True],
        "rule_reason": ["temp_high;vibration_high", "", ""],
        "ml_score": [-0.1, -0.1, -0.1],
        "label": ["abnormal"] * 3
    }
    df = pd.DataFrame(data)
    result = combine(df)

    # Row 0: Both trigger -> rule_reason used
    # rule_reason is "temp_high;vibration_high", triggered_sensors should be "temp;vibration"
    assert result.iloc[0]["triggered_sensors"] == "temp;vibration"

    # Row 1: Only ML trigger -> temp_scaled is max abs (2.0)
    assert result.iloc[1]["triggered_sensors"] == "temp"

    # Row 2: Only ML trigger -> vibration_scaled is max abs (5.0)
    assert result.iloc[2]["triggered_sensors"] == "vibration"

def test_combine_ignores_label():
    """
    Verify that label doesn't affect ensemble results.
    """
    row = {
        "temp": 47.0, "pressure": 1.02, "vibration": 0.03,
        "temp_scaled": 0.0, "pressure_scaled": 0.0, "vibration_scaled": 0.0,
        "rule_flag": True, "ml_flag": False, "rule_reason": "temp_high", "ml_score": 0.0
    }
    df_normal = pd.DataFrame([row | {"label": "normal"}])
    df_abnormal = pd.DataFrame([row | {"label": "abnormal"}])

    res_normal = combine(df_normal)
    res_abnormal = combine(df_abnormal)

    pd.testing.assert_series_equal(res_normal["severity"], res_abnormal["severity"])
    pd.testing.assert_series_equal(res_normal["is_anomaly_final"], res_abnormal["is_anomaly_final"])
    pd.testing.assert_series_equal(res_normal["triggered_sensors"], res_abnormal["triggered_sensors"])
