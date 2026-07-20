import pandas as pd
from typing import Any
from config import RAW_DATA_PATH

import generate_data
import preprocess
import features
import rule_detector
import ml_detector
import ensemble

def run_pipeline() -> dict[str, Any]:
    """
    Orchestrates the Smart Factory Alert Agent pipeline.

    Flow:
    1. Data Generation -> CSV
    2. Load -> Handle Missing -> Add Rolling Features
    3. Temporal Split (Train/Cal/Test)
    4. Fit Scaler on Train -> Transform all
    5. Rule Detection on Test
    6. ML Fit on Train -> Calibrate Threshold on Cal -> Detect on Test
    7. Ensemble Combine on Test
    8. Filter Anomalies

    Returns:
        dict: {
            "alerts_df": DataFrame of anomalies (reset index),
            "test_df": Full combined test DataFrame,
            "scaler": fitted scaler object,
            "ml_model": fitted IsolationForest model,
            "ml_threshold": calibrated threshold value
        }
    """
    # 1. Generate Data
    generate_data.generate_dataset(output_path=RAW_DATA_PATH)

    # 2. Initial Preprocessing
    df = preprocess.load_raw(RAW_DATA_PATH)
    df = preprocess.handle_missing_values(df)

    # 3. Feature Engineering (Causal Rolling Features on full sequence)
    df = features.add_rolling_features(df)

    # 4. Temporal Split
    # train_fit: Block A (normal only), cal: Block B (normal only), test: Block C (mixed)
    train_fit, cal, test = preprocess.temporal_split(df)

    # 5. Scaling
    scaler = preprocess.fit_scaler(train_fit)

    train_fit_scaled = preprocess.transform_features(train_fit, scaler)
    cal_scaled = preprocess.transform_features(cal, scaler)
    test_scaled = preprocess.transform_features(test, scaler)

    # 6. Rule-based Detection (on original values of Block C)
    test = rule_detector.detect(test)

    # 7. ML-based Detection
    # Fit model on scaled Block A
    ml_model = ml_detector.fit(train_fit_scaled)

    # Calibrate threshold using scaled Block B
    ml_threshold = ml_detector.calibrate_threshold(ml_model, cal_scaled, train_fit_scaled)

    # Detect anomalies on scaled Block C
    test_scaled_with_ml = ml_detector.detect(test_scaled, ml_model, ml_threshold)

    # Merge ML results (ml_score, ml_flag) AND ALL scaled features back into the original 'test' DataFrame
    # Ensemble.combine needs the raw sensor scaled values for MEDIUM alerts.
    # test_run_pipeline_causal_consistency checks for all ML_FEATURE_COLUMNS scaled versions.
    from config import RAW_SENSOR_COLUMNS, ML_FEATURE_COLUMNS

    ml_cols = ["ml_score", "ml_flag"]
    scaled_cols = [f"{c}_scaled" for c in ML_FEATURE_COLUMNS]

    cols_to_merge = ml_cols + scaled_cols
    test[cols_to_merge] = test_scaled_with_ml[cols_to_merge]



    # 8. Ensemble Combination
    # Now 'test' contains both rule and ml flags
    test_df = ensemble.combine(test)

    # 9. Filter for alerts
    # Reset index is CRITICAL to ensure that main.py's suggestion list (0-indexed)
    # matches the alerts_df rows.
    alerts_df = test_df[test_df["is_anomaly_final"] == True].copy().reset_index(drop=True)

    return {
        "alerts_df": alerts_df,
        "test_df": test_df,
        "scaler": scaler,
        "ml_model": ml_model,
        "ml_threshold": ml_threshold
    }
