import pytest
import pandas as pd
import numpy as np
from config import ROLLING_WINDOW, RAW_SENSOR_COLUMNS, ML_FEATURE_COLUMNS
from features import add_rolling_features

def test_add_rolling_features_logic():
    # Create a simple dataset
    # temp: [10, 20, 30, 40, 50]
    # rolling_mean (w=5, min=1):
    # 0: 10/1 = 10
    # 1: (10+20)/2 = 15
    # 2: (10+20+30)/3 = 20
    # 3: (10+20+30+40)/4 = 25
    # 4: (10+20+30+40+50)/5 = 30
    # diff: [NaN, 10, 10, 10, 10] -> fillna(0) -> [0, 10, 10, 10, 10]

    data = {
        "temp": [10.0, 20.0, 30.0, 40.0, 50.0],
        "pressure": [1.0, 1.1, 1.2, 1.3, 1.4],
        "vibration": [0.01, 0.02, 0.03, 0.04, 0.05],
        "label": ["normal"] * 5
    }
    df = pd.DataFrame(data)

    # Use a fixed window for testing if ROLLING_WINDOW is different
    window = 5
    df_feat = add_rolling_features(df, window=window)

    # 1. Check if all ML_FEATURE_COLUMNS are present
    for col in ML_FEATURE_COLUMNS:
        assert col in df_feat.columns

    # 2. Verify rolling_mean for 'temp'
    expected_mean = [10.0, 15.0, 20.0, 25.0, 30.0]
    pd.testing.assert_series_equal(df_feat["temp_rolling_mean"], pd.Series(expected_mean, name="temp_rolling_mean"))

    # 3. Verify diff for 'temp'
    expected_diff = [0.0, 10.0, 10.0, 10.0, 10.0]
    pd.testing.assert_series_equal(df_feat["temp_diff"], pd.Series(expected_diff, name="temp_diff"))

    # 4. Verify rolling_std for 'temp'
    # idx 0: std([10]) = NaN -> 0.0
    # idx 1: std([10, 20]) = 7.071...
    assert df_feat.loc[0, "temp_rolling_std"] == 0.0
    assert df_feat.loc[1, "temp_rolling_std"] > 0

def test_add_rolling_features_causality():
    # Ensure that the value at index i only depends on 0...i
    data = {
        "temp": [10.0] * 10,
        "pressure": [1.0] * 10,
        "vibration": [0.01] * 10,
        "label": ["normal"] * 10
    }
    df = pd.DataFrame(data)

    df_feat_before = add_rolling_features(df)

    # Change a future value
    df.loc[8, "temp"] = 100.0
    df_feat_after = add_rolling_features(df)

    # Rows 0 to 7 should be identical
    pd.testing.assert_frame_equal(
        df_feat_before.iloc[:8],
        df_feat_after.iloc[:8]
    )
    # Row 8 should differ
    assert not df_feat_before.iloc[8]["temp_rolling_mean"] == df_feat_after.iloc[8]["temp_rolling_mean"]

def test_add_rolling_features_original_preserved():
    data = {
        "temp": [10.0, 20.0],
        "pressure": [1.0, 1.1],
        "vibration": [0.01, 0.02],
        "label": ["normal", "normal"]
    }
    df = pd.DataFrame(data)
    df_feat = add_rolling_features(df)

    # Original columns should be exactly the same
    for col in ["temp", "pressure", "vibration"]:
        pd.testing.assert_series_equal(df[col], df_feat[col])
