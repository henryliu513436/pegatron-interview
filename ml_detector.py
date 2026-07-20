import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from config import (
    IF_N_ESTIMATORS, IF_CONTAMINATION, IF_RANDOM_STATE,
    THRESHOLD_PERCENTILE, ML_FEATURE_COLUMNS
)

def _get_scaled_cols():
    """Helper to return the consistent list of scaled feature columns."""
    return [f"{c}_scaled" for c in ML_FEATURE_COLUMNS]

def fit(train_fit_scaled: pd.DataFrame) -> IsolationForest:
    """
    Fits an IsolationForest model on the scaled features of the train_fit set.

    Input:
        train_fit_scaled: DataFrame containing scaled features.
    Output:
        Fitted IsolationForest model.
    """
    X = train_fit_scaled[_get_scaled_cols()]

    model = IsolationForest(
        n_estimators=IF_N_ESTIMATORS,
        contamination=IF_CONTAMINATION,
        random_state=IF_RANDOM_STATE
    )
    model.fit(X)
    return model

def calibrate_threshold(
    model: IsolationForest,
    cal_scaled: pd.DataFrame,
    train_fit_scaled: pd.DataFrame,
    percentile: float = THRESHOLD_PERCENTILE
) -> float:
    """
    Calibrates the anomaly threshold using the decision_function distribution of the calibration set.

    Inputs:
        model: The fitted IsolationForest model.
        cal_scaled: The calibration set (Block B).
        train_fit_scaled: The train set (Block A) used for overlap verification.
        percentile: The percentile to use for the threshold.

    Output:
        The calculated threshold value.
    """
    # 1. Verify no overlap between cal and train_fit (temporal split requirement)
    common_indices = train_fit_scaled.index.intersection(cal_scaled.index)
    if len(common_indices) > 0:
        raise ValueError(f"Overlap detected between train_fit and cal sets: {len(common_indices)} common indices")

    # 2. Calculate scores for calibration set
    X_cal = cal_scaled[_get_scaled_cols()]
    scores = model.decision_function(X_cal)

    # 3. Derive threshold from percentile
    threshold = np.percentile(scores, percentile)
    return threshold

def detect(df_scaled: pd.DataFrame, model: IsolationForest, threshold: float) -> pd.DataFrame:
    """
    Detects anomalies in the test set using the fitted model and calibrated threshold.

    Inputs:
        df_scaled: DataFrame containing scaled features.
        model: The fitted IsolationForest model.
        threshold: The calibrated anomaly threshold.

    Output:
        DataFrame with new columns:
        - ml_score (float): The decision_function value.
        - ml_flag (bool): True if ml_score < threshold.
    """
    X = df_scaled[_get_scaled_cols()]

    # Calculate anomaly score
    # IsolationForest.decision_function returns values where lower = more anomalous
    scores = model.decision_function(X)

    df_result = df_scaled.copy()
    df_result["ml_score"] = scores
    df_result["ml_flag"] = scores < threshold

    return df_result
