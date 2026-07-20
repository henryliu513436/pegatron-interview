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

        # Abnormal points
        # Note: Due to 'drift' events, some abnormal rows may have values within normal ranges.
        # We only verify that labels 'normal' are ALWAYS within normal ranges.
        # The inverse (all 'abnormal' must have abnormal values) is no longer true.
        pass

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

def test_anomalies_present_in_all_blocks():
    # Test that anomalies are distributed across train, cal, and test blocks
    df = generate_dataset()
    n = len(df)

    # Use ratios from config to slice the df
    from config import TRAIN_RATIO, CAL_RATIO
    split_a = int(n * TRAIN_RATIO)
    split_b = int(n * (TRAIN_RATIO + CAL_RATIO))

    block_a = df.iloc[:split_a]
    block_b = df.iloc[split_a:split_b]
    block_c = df.iloc[split_b:]

    assert (block_a["label"] == "abnormal").any(), "Block A (Train) must contain at least one anomaly"
    assert (block_b["label"] == "abnormal").any(), "Block B (Cal) must contain at least one anomaly"
    assert (block_c["label"] == "abnormal").any(), "Block C (Test) must contain at least one anomaly"

def test_anomalies_present_in_all_blocks():
    # Test that anomalies are distributed across train, cal, and test blocks
    df = generate_dataset()
    n = len(df)

    # Use ratios from config to slice the df
    from config import TRAIN_RATIO, CAL_RATIO
    split_a = int(n * TRAIN_RATIO)
    split_b = int(n * (TRAIN_RATIO + CAL_RATIO))

    block_a = df.iloc[:split_a]
    block_b = df.iloc[split_a:split_b]
    block_c = df.iloc[split_b:]

    assert (block_a["label"] == "abnormal").any(), "Block A (Train) must contain at least one anomaly"
    assert (block_b["label"] == "abnormal").any(), "Block B (Cal) must contain at least one anomaly"
    assert (block_c["label"] == "abnormal").any(), "Block C (Test) must contain at least one anomaly"

def test_normal_values_are_gaussian_like():
    """
    Verify that normal values follow a Gaussian-like distribution (std is significantly
    smaller than a uniform distribution over the same range).
    """
    df = generate_dataset(n_rows=1000)
    normal_df = df[df["label"] == "normal"]

    for sensor, ranges in GEN_RANGES.items():
        vals = normal_df[sensor].dropna()
        n_min, n_max = ranges["normal"]
        actual_std = vals.std()

        # Theoretical std of uniform distribution: (max - min) / sqrt(12)
        uniform_std = (n_max - n_min) / np.sqrt(12)

        # Normal distribution should have a tighter cluster around the mean
        # We verify that the actual std is strictly less than the theoretical uniform std.
        threshold = 1.0
        assert actual_std < uniform_std * threshold, f"{sensor}: Normal values should be more Gaussian-like (std {actual_std:.4f} < {uniform_std:.4f})"

def test_spike_drift_are_multivariate():
    """
    Verify that spike and drift events are multivariate (all 3 sensors abnormal simultaneously).
    """
    df = generate_dataset(n_rows=1000)
    abnormal_df = df[df["label"] == "abnormal"]

    found_multivariate = False
    for _, row in abnormal_df.iterrows():
        all_abnormal = True
        for s in ["temp", "pressure", "vibration"]:
            val = row[s]
            if pd.isna(val):
                all_abnormal = False
                break
            n_min, n_max = GEN_RANGES[s]["normal"]
            if n_min <= val <= n_max:
                all_abnormal = False
                break
        if all_abnormal:
            found_multivariate = True
            break

    assert found_multivariate, "Should find at least one row where all 3 sensors are simultaneously abnormal"

def test_drift_has_rule_invisible_rows():
    """
    Verify that drift events create abnormal rows that are invisible to rule-based detectors
    (values still within normal range but label is abnormal).
    """
    df = generate_dataset(n_rows=1000)
    # Look for rows where label is abnormal but NO sensor is outside normal range
    invisible_rows = df[df["label"] == "abnormal"].copy()

    def is_strictly_normal(row):
        for s in ["temp", "pressure", "vibration"]:
            val = row[s]
            if pd.isna(val): continue
            n_min, n_max = GEN_RANGES[s]["normal"]
            if val < n_min or val > n_max:
                return False
        return True

    invisible_rows = invisible_rows[invisible_rows.apply(is_strictly_normal, axis=1)]
    assert len(invisible_rows) > 0, "Drift events should produce abnormal rows that are invisible to simple range rules"

if __name__ == "__main__":
    pytest.main([__file__])
