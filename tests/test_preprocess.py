import pytest
import pandas as pd
import numpy as np
import os
from config import RAW_DATA_PATH, RAW_SENSOR_COLUMNS
from preprocess import load_raw, handle_missing_values, temporal_split, fit_scaler, transform_features

def test_load_raw_success(tmp_path):
    # Create a temporary CSV
    csv_path = tmp_path / "test_raw.csv"
    data = {
        "timestamp": ["2024-06-03 19:02:00", "2024-06-03 19:00:00", "2024-06-03 19:01:00"],
        "temp": [46.0, 47.0, 48.0],
        "pressure": [1.01, 1.02, 1.03],
        "vibration": [0.03, 0.02, 0.01],
        "label": ["normal", "normal", "abnormal"]
    }
    df_orig = pd.DataFrame(data)
    df_orig.to_csv(csv_path, index=False)

    df = load_raw(str(csv_path))

    # Check columns
    assert list(df.columns) == ["timestamp", "temp", "pressure", "vibration", "label"]
    # Check timestamp type
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    # Check sorting
    assert df["timestamp"].is_monotonic_increasing
    # Check values after sorting
    assert df.iloc[0]["timestamp"] == pd.Timestamp("2024-06-03 19:00:00")
    assert df.iloc[2]["timestamp"] == pd.Timestamp("2024-06-03 19:02:00")

def test_load_raw_nan_timestamp(tmp_path):
    csv_path = tmp_path / "nan_ts.csv"
    data = {
        "timestamp": [np.nan, "2024-06-03 19:00:00"],
        "temp": [46.0, 47.0],
        "pressure": [1.01, 1.02],
        "vibration": [0.03, 0.02],
        "label": ["normal", "normal"]
    }
    pd.DataFrame(data).to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="timestamp|label"):
        load_raw(str(csv_path))

def test_load_raw_nan_label(tmp_path):
    csv_path = tmp_path / "nan_label.csv"
    data = {
        "timestamp": ["2024-06-03 19:00:00", "2024-06-03 19:01:00"],
        "temp": [46.0, 47.0],
        "pressure": [1.01, 1.02],
        "vibration": [0.03, 0.02],
        "label": ["normal", np.nan]
    }
    pd.DataFrame(data).to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="timestamp|label"):
        load_raw(str(csv_path))

def test_handle_missing_values_success():
    # Create DF with NaNs
    # Causal fill: [10, NaN, 20] -> [10, 10, 20]
    # Backfill: [NaN, 10] -> [10, 10]
    data = {
        "temp": [np.nan, 46.0, np.nan, 48.0],
        "pressure": [1.0, np.nan, 1.1, 1.2],
        "vibration": [0.02, 0.02, np.nan, 0.04],
        "label": ["normal", "normal", "abnormal", "normal"]
    }
    df = pd.DataFrame(data)

    df_cleaned = handle_missing_values(df)

    # Check no NaNs in sensor columns
    for col in RAW_SENSOR_COLUMNS:
        assert df_cleaned[col].isna().sum() == 0

    # Check causal fill for temp: [NaN, 46, NaN, 48]
    # idx 0: bfill from 46 -> 46
    # idx 2: ffill from 46 -> 46 (instead of 47 from linear)
    assert df_cleaned.loc[0, "temp"] == 46.0
    assert df_cleaned.loc[2, "temp"] == 46.0

    # Check label is untouched
    assert (df_cleaned["label"] == df["label"]).all()

def test_handle_missing_values_causality():
    # Ensure that the fill value depends only on the past (except for leading NaNs)
    data = {
        "temp": [10.0, np.nan, 20.0], # index 1 should become 10.0, NOT 15.0
        "pressure": [1.0, 1.0, 1.0],
        "vibration": [0.01, 0.01, 0.01],
        "label": ["normal"] * 3
    }
    df = pd.DataFrame(data)
    df_cleaned = handle_missing_values(df)
    assert df_cleaned.loc[1, "temp"] == 10.0, "Missing value was filled using future value (linear interpolation)"

def test_handle_missing_values_all_nan_failure():
    data = {
        "temp": [np.nan, np.nan],
        "pressure": [1.0, 1.1],
        "vibration": [0.02, 0.03],
        "label": ["normal", "normal"]
    }
    df = pd.DataFrame(data)
    with pytest.raises(ValueError, match="NaN"):
        handle_missing_values(df)

def test_temporal_split():
    # Create a dataset of 100 rows
    # Ensure first 85 rows are normal to make Block A (70) and Block B (15) full
    data = {
        "timestamp": pd.date_range("2024-01-01", periods=100, freq="1min"),
        "temp": np.random.uniform(45, 50, 100),
        "label": ["normal"] * 85 + ["normal", "abnormal"] * 7 + ["normal"] * 1 # 100 total
    }
    df = pd.DataFrame(data)

    train_fit, cal, test = temporal_split(df)

    assert len(train_fit) == 70
    assert len(cal) == 15
    assert len(test) == 15
    assert (train_fit["label"] == "normal").all()
    assert (cal["label"] == "normal").all()
    assert train_fit.index.max() < cal.index.min()
    assert cal.index.max() < test.index.min()

def test_temporal_split_insufficient_samples():
    data = {
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="1min"),
        "temp": [45.0] * 5,
        "label": ["normal"] * 5
    }
    df = pd.DataFrame(data)
    with pytest.raises(ValueError, match="fewer than"):
        temporal_split(df)

def test_scaler_workflow():
    from config import ML_FEATURE_COLUMNS
    data = {col: np.random.randn(10) for col in ML_FEATURE_COLUMNS}
    data["label"] = ["normal"] * 10
    train_df = pd.DataFrame(data)

    test_data = {col: np.random.randn(5) for col in ML_FEATURE_COLUMNS}
    test_data["label"] = ["normal"] * 5
    test_df = pd.DataFrame(test_data)

    scaler = fit_scaler(train_df)

    train_transformed = transform_features(train_df, scaler)
    assert f"{ML_FEATURE_COLUMNS[0]}_scaled" in train_transformed.columns
    assert ML_FEATURE_COLUMNS[0] in train_transformed.columns

    test_transformed = transform_features(test_df, scaler)
    assert f"{ML_FEATURE_COLUMNS[0]}_scaled" in test_transformed.columns
