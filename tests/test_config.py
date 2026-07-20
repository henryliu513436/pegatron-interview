import pytest
import config

def test_config_constants():
    # assert config.N_ROWS == 300
    assert config.RANDOM_SEED == 42
    assert "temp" in config.THRESHOLDS
    assert config.MIN_REQUIRED_SAMPLES == 10
    assert len(config.ML_FEATURE_COLUMNS) == 12
    assert config.RAW_DATA_PATH == "data/raw_data.csv"

def test_gen_ranges_consistency():
    """
    Verify that GEN_RANGES are consistent with THRESHOLDS and cover gray areas.
    1. Normal range must be within THRESHOLDS normal bounds.
    2. min_limit < normal_min < normal_max < max_limit.
    3. Generation range must cover the gray area (up to abnormal_high/low).
    """
    for sensor, ranges in config.GEN_RANGES.items():
        thresh = config.THRESHOLDS[sensor]
        n_min, n_max = ranges["normal"]

        # 1. Normal range consistency
        if "normal_min" in thresh:
            assert n_min >= thresh["normal_min"], f"{sensor}: GEN_RANGES normal_min should be >= THRESHOLDS normal_min"
        if "normal_max" in thresh:
            assert n_max <= thresh["normal_max"], f"{sensor}: GEN_RANGES normal_max should be <= THRESHOLDS normal_max"

        # 2. & 3. Bound consistency and Gray Area coverage
        if "max_limit" in ranges:
            # Bound consistency
            assert ranges["max_limit"] > n_max, f"{sensor}: max_limit must be > normal_max"
            # Gray area coverage: max_limit must be at least as high as abnormal_high threshold
            if "abnormal_high" in thresh:
                assert ranges["max_limit"] >= thresh["abnormal_high"], f"{sensor}: max_limit must cover gray area up to {thresh['abnormal_high']}"

        if "min_limit" in ranges:
            # Bound consistency
            assert ranges["min_limit"] < n_min, f"{sensor}: min_limit must be < normal_min"
            # Gray area coverage: min_limit must be at least as low as abnormal_low threshold
            if "abnormal_low" in thresh:
                assert ranges["min_limit"] <= thresh["abnormal_low"], f"{sensor}: min_limit must cover gray area down to {thresh['abnormal_low']}"

if __name__ == "__main__":
    pytest.main([__file__])
