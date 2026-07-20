import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from config import CONFUSION_MATRIX_PATH

def evaluate(test_df_with_predictions: pd.DataFrame, label_col: str = "label") -> dict:
    """
    Evaluates rule, ml, and ensemble detectors using the label column as ground truth.

    Inputs:
        test_df_with_predictions: DataFrame containing ground truth and flags.
        label_col: Column name for ground truth ('normal' | 'abnormal').

    Outputs:
        A dictionary containing precision, recall, f1 and confusion matrix for each detector.
    """
    # 1. Convert labels to binary (abnormal = 1, normal = 0)
    y_true = (test_df_with_predictions[label_col] == "abnormal").astype(int)

    detectors = {
        "rule": "rule_flag",
        "ml": "ml_flag",
        "ensemble": "is_anomaly_final"
    }

    results = {}

    for name, col in detectors.items():
        if col not in test_df_with_predictions.columns:
            raise KeyError(f"Missing prediction column: {col}")

        y_pred = test_df_with_predictions[col].astype(int)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred).tolist()

        results[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "cm": cm
        }

    # Trigger plot
    plot_confusion_matrices(results)

    return results

def plot_confusion_matrices(metrics: dict, output_path: str = CONFUSION_MATRIX_PATH) -> None:
    """
    Plots confusion matrices for rule, ml, and ensemble detectors as subplots.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    detectors = ["rule", "ml", "ensemble"]

    for ax, name in zip(axes, detectors):
        if name not in metrics:
            continue

        cm = np.array(metrics[name]["cm"])
        # Labels for the confusion matrix
        # Rows: Actual (Normal, Abnormal), Cols: Predicted (Normal, Abnormal)
        # Standard sklearn cm: [[TN, FP], [FN, TP]]
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f"{name.capitalize()} Detector")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        # Add tick marks
        ax.set_xticks([0, 1], minor=False)
        ax.set_yticks([0, 1], minor=False)
        ax.set_xticklabels(["Normal", "Abnormal"])
        ax.set_yticklabels(["Normal", "Abnormal"])

        # Annotate values
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > (cm.max()/2) else "black")

    plt.tight_layout()

    # Ensure output directory exists
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close(fig)
