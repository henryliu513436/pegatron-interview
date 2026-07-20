import pytest
import pandas as pd
import numpy as np
from config import CONFUSION_MATRIX_PATH
from evaluate import evaluate, plot_confusion_matrices

def test_evaluate_metrics_calculation():
    """
    Verify that metrics (Precision, Recall, F1) are calculated correctly.
    """
    # Construct a test dataset with known outcomes
    # Ground truth: [1, 1, 1, 1, 0, 0, 0, 0] (4 abnormal, 4 normal)
    data = {
        "label": ["abnormal"] * 4 + ["normal"] * 4,
        # Rule: hit 2/4 abnormal, 1/4 normal (FP)
        "rule_flag": [True, True, False, False, True, False, False, False],
        # ML: hit 3/4 abnormal, 0/4 normal
        "ml_flag": [True, True, True, False, False, False, False, False],
        # Ensemble: hit 3/4 abnormal (since it's rule OR ml), 1/4 normal (FP from rule)
        "is_anomaly_final": [True, True, True, False, True, False, False, False]
    }
    df = pd.DataFrame(data)

    metrics = evaluate(df)

    # --- Rule Check ---
    # TP=2, FP=1, FN=2 -> P=2/3, R=2/4, F1=2*P*R/(P+R)
    # Precision = 2/3 ~= 0.666
    # Recall = 2/4 = 0.5
    # F1 = 2 * (2/3 * 1/2) / (2/3 + 1/2) = (2/3) / (7/6) = 4/7 ~= 0.571
    assert np.isclose(metrics["rule"]["precision"], 2/3)
    assert np.isclose(metrics["rule"]["recall"], 0.5)
    assert np.isclose(metrics["rule"]["f1"], 4/7)

    # --- ML Check ---
    # TP=3, FP=0, FN=1 -> P=3/3=1, R=3/4=0.75, F1=2*1*0.75/(1+0.75) = 1.5/1.75 = 6/7 ~= 0.857
    assert np.isclose(metrics["ml"]["precision"], 1.0)
    assert np.isclose(metrics["ml"]["recall"], 0.75)
    assert np.isclose(metrics["ml"]["f1"], 6/7)

    # --- Ensemble Check ---
    # TP=3, FP=1, FN=1 -> P=3/4=0.75, R=3/4=0.75, F1=0.75
    assert np.isclose(metrics["ensemble"]["precision"], 0.75)
    assert np.isclose(metrics["ensemble"]["recall"], 0.75)
    assert np.isclose(metrics["ensemble"]["f1"], 0.75)

def test_evaluate_plot_creation(tmp_path):
    """
    Verify that plot_confusion_matrices creates a file.
    """
    metrics = {
        "rule": {"cm": [[10, 2], [3, 10]]},
        "ml": {"cm": [[12, 1], [5, 8]]},
        "ensemble": {"cm": [[13, 1], [4, 9]]}
    }

    # Override path for testing
    output_file = str(tmp_path / "confusion_matrices.png")
    plot_confusion_matrices(metrics, output_path=output_file)

    assert (tmp_path / "confusion_matrices.png").exists()

def test_evaluate_invalid_input():
    """
    Verify that evaluate raises error if required columns are missing.
    """
    df = pd.DataFrame({"label": [1, 0], "wrong_col": [True, False]})
    with pytest.raises(KeyError):
        evaluate(df)
