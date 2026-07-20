import pytest
import config

def test_config_constants():
    assert config.N_ROWS == 300
    assert config.RANDOM_SEED == 42
    assert "temp" in config.THRESHOLDS
    assert config.MIN_REQUIRED_SAMPLES == 10
    assert len(config.ML_FEATURE_COLUMNS) == 12
    assert config.RAW_DATA_PATH == "data/raw_data.csv"

def test_gen_ranges_consistency():
    """
    Verify that GEN_RANGES strictly follow THRESHOLDS:
    1. Normal range must be within normal_min/max.
    2. Abnormal high range must be entirely > abnormal_high threshold.
    3. Abnormal low range must be entirely < abnormal_low threshold.
    """
    for sensor, ranges in config.GEN_RANGES.items():
        thresh = config.THRESHOLDS[sensor]

        # Check Normal
        n_min, n_max = ranges["normal"]
        if "normal_min" in thresh:
            assert n_min >= thresh["normal_min"]
        if "normal_max" in thresh:
            assert n_max <= thresh["normal_max"]

        # Check Abnormal High
        if "abnormal_high" in ranges:
            ah_min, ah_max = ranges["abnormal_high"]
            assert ah_min > thresh["abnormal_high"]

        # Check Abnormal Low
        if "abnormal_low" in ranges:
            al_min, al_max = ranges["abnormal_low"]
            assert al_max < thresh["abnormal_low"]

if __name__ == "__main__":
    pytest.main([__file__])
