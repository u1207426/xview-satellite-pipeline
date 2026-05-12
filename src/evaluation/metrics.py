# src/evaluation/metrics.py
from __future__ import annotations
from typing import Dict, List

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def compute_l2_metrics(
    y_true: List[int],
    y_prob: List[float],
    threshold: float = 0.5,
) -> Dict:
    """
    Compute binary classification metrics for L2 model evaluation.

    Returns dict with: accuracy, auc_roc, precision, recall, f1, confusion_matrix.
    auc_roc returns 0.0 when only one class is present in y_true.
    """
    y_pred = [1 if p >= threshold else 0 for p in y_prob]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    try:
        auc = roc_auc_score(y_true, y_prob)
        if auc != auc:  # nan check
            auc = 0.0
    except ValueError:
        auc = 0.0
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "auc_roc": auc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
