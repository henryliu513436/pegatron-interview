import pytest
import pandas as pd
import numpy as np
import os
from config import (
    N_ROWS, ANOMALY_RATIO, MISSING_RATIO, START_TIME, INTERVAL_MINUTES,
    RANDOM_SEED, RAW_DATA_PATH, MIN_REQUIRED_SAMPLES, GEN_RANGES
)
from generate_data import generate_dataset

def test_generate_dataset_basic_properties():
    # Test using default config
    df = generate_dataset()

    # 1. Row count
    assert len(df) == N_ROWS

    # 2. Column names
    assert list(df.columns) == ["timestamp", "temp", "pressure", "vibration", "label"]

    # 3. Timestamp sorted and formatted
    # Check if it's strictly increasing
    assert df["timestamp"].is_monotonic_increasing
    # Check format YYYY-MM-DD HH:MM:SS
    import re
    pattern = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$"
    assert df["timestamp"].astype(str).str.match(pattern).all()

def test_generate_dataset_label_ratio():
    # Use a larger N to get better convergence for ratios
    n_rows = 1000
    df = generate_dataset(n_rows=n_rows)

    actual_ratio = (df["label"] == "abnormal").mean()
    # Allow some margin for randomness since we use a ratio to trigger events,
    # and events like drift/stuck have multi-row length.
    assert abs(actual_ratio - ANOMALY_RATIO) < 0.05

def test_generate_dataset_missing_values():
    df = generate_dataset()

    # Check missing ratio on each sensor column
    for col in ["temp", "pressure", "vibration"]:
        missing_ratio = df[col].isna().mean()
        assert abs(missing_ratio - MISSING_RATIO) < 0.05

    # Timestamp and label must NOT have NaNs
    assert not df["timestamp"].isna().any()
    assert not df["label"].isna().any()

def test_generate_dataset_value_ranges():
    df = generate_dataset()

    # We check the values that are NOT NaN
    for sensor, ranges in GEN_RANGES.items():
        # Normal points
        normal_vals = df[df["label"] == "normal"][sensor].dropna()
        n_min, n_max = ranges["normal"]
        assert (normal_vals >= n_min).all()
        assert (normal_vals <= n_max).all()

        # Abnormal points: Must have at least one sensor in abnormal range
        # Note: The specification says abnormal rows MUST have at least one abnormal value.
        # We check that abnormal labels correspond to values outside normal ranges.
        abnormal_rows = df[df["label"] == "abnormal"]
        # For each abnormal row, at least one sensor must be outside its normal range
        for _, row in abnormal_rows.iterrows():
            has_abnormal = False
            for s in ["temp", "pressure", "vibration"]:
                val = row[s]
                if pd.isna(val): continue

                s_ranges = GEN_RANGES[s]
                n_min, n_max = s_ranges["normal"]
                if val < n_min or val > n_max:
                    has_abnormal = True
                    break
            assert has_abnormal, f"Row with label abnormal has all sensors in normal range: {row}"

def test_generate_dataset_min_samples_error():
    with pytest.raises(ValueError):
        generate_dataset(n_rows=MIN_REQUIRED_SAMPLES - 1)

def test_generate_dataset_output_file():
    df = generate_dataset()
    assert os.path.exists(RAW_DATA_PATH)

    # Read back and verify
    df_read = pd.read_csv(RAW_DATA_PATH)
    # timestamp becomes object, check content
    assert len(df_read) == len(df)
    assert (df_read["label"] == df["label"].values).all()

if __name__ == "__main__":
    pytest.main([__file__])
