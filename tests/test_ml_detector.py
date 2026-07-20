import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from config import (
    IF_N_ESTIMATORS, IF_CONTAMINATION, IF_RANDOM_STATE,
    THRESHOLD_PERCENTILE, ML_FEATURE_COLUMNS
)
from ml_detector import fit, calibrate_threshold, detect

def test_ml_fit_reproducibility():
    """
    Verify that fitting the model with a fixed seed produces identical results.
    """
    # Create synthetic scaled data
    scaled_cols = [f"{c}_scaled" for c in ML_FEATURE_COLUMNS]
    data = {col: np.random.randn(100) for col in scaled_cols}
    df = pd.DataFrame(data)

    model1 = fit(df)
    model2 = fit(df)

    # Decision functions should be identical for the same data
    scores1 = model1.decision_function(df[scaled_cols])
    scores2 = model2.decision_function(df[scaled_cols])
    np.testing.assert_array_almost_equal(scores1, scores2)

def test_calibrate_threshold_logic():
    """
    Verify threshold is the correct percentile of decision_function on cal set.
    """
    scaled_cols = [f"{c}_scaled" for c in ML_FEATURE_COLUMNS]
    # Use distinct indices to avoid overlap error
    train_df = pd.DataFrame({col: np.random.randn(100) for col in scaled_cols}, index=range(0, 100))
    cal_df = pd.DataFrame({col: np.random.randn(100) for col in scaled_cols}, index=range(100, 200))

    model = fit(train_df)

    # Manually calculate expected threshold
    scores = model.decision_function(cal_df[scaled_cols])
    expected_thresh = np.percentile(scores, THRESHOLD_PERCENTILE)

    actual_thresh = calibrate_threshold(model, cal_df, train_df)
    assert np.isclose(actual_thresh, expected_thresh)

def test_calibrate_overlap_error():
    """
    Verify that calibrate_threshold raises ValueError if cal and train_fit overlap.
    """
    scaled_cols = [f"{c}_scaled" for c in ML_FEATURE_COLUMNS]
    train_df = pd.DataFrame({col: np.random.randn(100) for col in scaled_cols}, index=range(0, 100))
    cal_df = pd.DataFrame({col: np.random.randn(100) for col in scaled_cols}, index=range(50, 150)) # Overlap 50-99

    model = fit(train_df)
    with pytest.raises(ValueError, match="Overlap detected"):
        calibrate_threshold(model, cal_df, train_df)

def test_ml_detect_anomaly():
    """
    Verify that a clearly anomalous point is flagged as abnormal.
    """
    scaled_cols = [f"{c}_scaled" for c in ML_FEATURE_COLUMNS]
    # Train on a tight cluster around 0
    train_df = pd.DataFrame({col: np.random.normal(0, 0.1, 100) for col in scaled_cols}, index=range(0, 100))
    model = fit(train_df)

    # Calibrate threshold (should be roughly around 0 or slightly negative)
    cal_df = pd.DataFrame({col: np.random.normal(0, 0.1, 100) for col in scaled_cols}, index=range(100, 200))
    thresh = calibrate_threshold(model, cal_df, train_df)

    # Test case: one very far point (anomaly) and one central point (normal)
    test_data = {col: [0.0, 10.0] for col in scaled_cols} # 0.0 is normal, 10.0 is extreme
    test_df = pd.DataFrame(test_data)

    result = detect(test_df, model, thresh)

    # Row 0 should be normal (ml_flag=False), Row 1 should be abnormal (ml_flag=True)
    assert result.iloc[0]["ml_flag"] == False
    assert result.iloc[1]["ml_flag"] == True
    assert result.iloc[1]["ml_score"] < result.iloc[0]["ml_score"]

def test_detect_column_consistency():
    """
    Ensure detect uses ONLY the specified scaled columns and ignores others.
    """
    scaled_cols = [f"{c}_scaled" for c in ML_FEATURE_COLUMNS]
    train_df = pd.DataFrame({col: np.random.randn(100) for col in scaled_cols})
    model = fit(train_df)
    thresh = -0.1

    # Add a "distractor" column that looks like a scaled feature but isn't in ML_FEATURE_COLUMNS
    test_df = pd.DataFrame({col: np.random.randn(10) for col in scaled_cols})
    test_df["fake_scaled"] = 999.0

    # This should not raise an error or affect the logic
    result = detect(test_df, model, thresh)
    assert "ml_flag" in result.columns
    assert "ml_score" in result.columns
