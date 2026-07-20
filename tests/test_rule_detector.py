import pytest
import pandas as pd
import numpy as np
from config import THRESHOLDS
from rule_detector import detect

def test_rule_detector_boundaries():
    """
    Test boundary conditions for all sensor thresholds.
    Verify that values exactly at the threshold are normal, and slightly over are abnormal.
    """
    # Create a base normal row
    base_row = {
        "temp": 47.5,
        "pressure": 1.02,
        "vibration": 0.03,
        "label": "normal" # Should be ignored by detector
    }

    # Define test cases: (sensor, threshold_key, boundary_val, expected_flag, expected_reason_part)
    # Note: Based on logic: flag=True if val > abnormal_high or val < abnormal_low
    test_cases = [
        # Temp High
        ("temp", "abnormal_high", 52.0, False, ""),
        ("temp", "abnormal_high", 52.1, True, "temp_high"),
        # Temp Low
        ("temp", "abnormal_low", 43.0, False, ""),
        ("temp", "abnormal_low", 42.9, True, "temp_low"),
        # Pressure High
        ("pressure", "abnormal_high", 1.08, False, ""),
        ("pressure", "abnormal_high", 1.09, True, "pressure_high"),
        # Pressure Low
        ("pressure", "abnormal_low", 0.97, False, ""),
        ("pressure", "abnormal_low", 0.96, True, "pressure_low"),
        # Vibration High
        ("vibration", "abnormal_high", 0.07, False, ""),
        ("vibration", "abnormal_high", 0.08, True, "vibration_high"),
    ]

    for sensor, t_key, val, exp_flag, exp_reason in test_cases:
        row = base_row.copy()
        row[sensor] = val
        df = pd.DataFrame([row])

        result = detect(df)

        assert result.iloc[0]["rule_flag"] == exp_flag, f"Failed {sensor} {t_key} at {val}"
        if exp_flag:
            assert exp_reason in result.iloc[0]["rule_reason"], f"Reason mismatch for {sensor} {t_key} at {val}"
        else:
            assert result.iloc[0]["rule_reason"] == "", f"Reason should be empty for normal value at {sensor} {t_key} {val}"

def test_rule_detector_multiple_triggers():
    """
    Test that multiple sensor violations are captured in the reason string.
    """
    data = {
        "temp": [53.0],      # High
        "pressure": [1.10], # High
        "vibration": [0.03], # Normal
        "label": ["abnormal"]
    }
    df = pd.DataFrame(data)
    result = detect(df)

    assert result.iloc[0]["rule_flag"] == True
    reason = result.iloc[0]["rule_reason"]
    assert "temp_high" in reason
    assert "pressure_high" in reason
    assert "vibration" not in reason
    # Check separator
    assert len(reason.split(";")) == 2

def test_rule_detector_ignores_label():
    """
    Ensure the detector does not change its output if the label is changed.
    """
    row = {"temp": 47.0, "pressure": 1.02, "vibration": 0.03}
    df_normal = pd.DataFrame([row | {"label": "normal"}])
    df_abnormal = pd.DataFrame([row | {"label": "abnormal"}])

    res_normal = detect(df_normal)
    res_abnormal = detect(df_abnormal)

    pd.testing.assert_series_equal(res_normal["rule_flag"], res_abnormal["rule_flag"])
    pd.testing.assert_series_equal(res_normal["rule_reason"], res_abnormal["rule_reason"])

def test_rule_detector_no_scaled_columns():
    """
    Verify the detector doesn't crash or rely on scaled columns if they exist.
    """
    data = {
        "temp": [47.0],
        "temp_scaled": [0.1],
        "pressure": [1.02],
        "pressure_scaled": [0.2],
        "vibration": [0.03],
        "vibration_scaled": [0.3],
        "label": ["normal"]
    }
    df = pd.DataFrame(data)
    # Should work fine and ignore _scaled columns
    result = detect(df)
    assert result.iloc[0]["rule_flag"] == False
