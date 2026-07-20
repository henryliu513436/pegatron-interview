import pytest
import pandas as pd
import numpy as np
from agent import run_pipeline
from config import RAW_DATA_PATH, ML_FEATURE_COLUMNS

def test_run_pipeline_output_structure():
    """
    Verify run_pipeline returns a dictionary with all required components
    and that dataframes have the expected properties.
    """
    results = run_pipeline()

    # 1. Check keys
    expected_keys = {"alerts_df", "test_df", "scaler", "ml_model", "ml_threshold"}
    assert expected_keys.issubset(results.keys())

    # 2. Verify test_df
    test_df = results["test_df"]
    assert isinstance(test_df, pd.DataFrame)
    assert not test_df.empty
    # Check for ensemble columns
    assert "is_anomaly_final" in test_df.columns
    assert "severity" in test_df.columns
    assert "triggered_sensors" in test_df.columns
    # Check for detector columns
    assert "rule_flag" in test_df.columns
    assert "ml_flag" in test_df.columns

    # 3. Verify alerts_df
    alerts_df = results["alerts_df"]
    assert isinstance(alerts_df, pd.DataFrame)

    # Ensure alerts_df is a subset of test_df where is_anomaly_final == True
    expected_alerts = test_df[test_df["is_anomaly_final"] == True]

    # We expect the values to match, but alerts_df should have a reset index
    # to avoid the index mismatch bug in output.replay
    pd.testing.assert_frame_equal(
        alerts_df.reset_index(drop=True),
        expected_alerts.reset_index(drop=True)
    )

    # CRITICAL: Check that alerts_df index is 0, 1, 2...
    # If this is not true, main.py's suggestion list mapping will fail.
    assert alerts_df.index[0] == 0 if not alerts_df.empty else True
    assert list(alerts_df.index) == list(range(len(alerts_df)))

def test_run_pipeline_causal_consistency():
    """
    Verify that the pipeline is executed in the correct sequence
    and no leakage occurs (though we mostly trust the modular tests).
    """
    results = run_pipeline()
    test_df = results["test_df"]

    # The number of rows in test_df should correspond to Block C (~15% of N_ROWS)
    # We don't assert exact number because temporal_split filters Block A/B
    # but test_df is Block C (mixed).
    assert len(test_df) > 0

    # Ensure scaled columns exist for ML features
    scaled_cols = [f"{c}_scaled" for c in ML_FEATURE_COLUMNS]
    for col in scaled_cols:
        assert col in test_df.columns

def test_run_pipeline_integration_smoke():
    """
    Smoke test to ensure the whole pipeline doesn't crash from start to finish.
    """
    try:
        run_pipeline()
    except Exception as e:
        pytest.fail(f"run_pipeline crashed with error: {e}")
