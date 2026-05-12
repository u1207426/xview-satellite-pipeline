# tests/test_metrics.py
import pytest
from src.evaluation.metrics import compute_l2_metrics


def test_perfect_classifier():
    y_true = [0, 0, 1, 1]
    y_prob = [0.1, 0.2, 0.8, 0.9]
    m = compute_l2_metrics(y_true, y_prob)
    assert m["accuracy"] == 1.0
    assert m["auc_roc"] == 1.0
    assert m["f1"] == 1.0


def test_all_wrong_classifier():
    y_true = [0, 0, 1, 1]
    y_prob = [0.9, 0.8, 0.2, 0.1]
    m = compute_l2_metrics(y_true, y_prob)
    assert m["accuracy"] == 0.0


def test_metric_keys_present():
    m = compute_l2_metrics([0, 1], [0.3, 0.7])
    for key in ("accuracy", "auc_roc", "precision", "recall", "f1", "confusion_matrix"):
        assert key in m


def test_confusion_matrix_shape():
    m = compute_l2_metrics([0, 0, 1, 1], [0.2, 0.3, 0.7, 0.8])
    cm = m["confusion_matrix"]
    assert len(cm) == 2 and len(cm[0]) == 2


def test_custom_threshold():
    y_true = [0, 1]
    y_prob = [0.6, 0.6]
    m_low = compute_l2_metrics(y_true, y_prob, threshold=0.5)
    # Both predicted positive at threshold 0.5
    assert m_low["precision"] < 1.0


def test_single_class_auc_returns_zero():
    # Only one class present → AUC undefined, should return 0.0 not raise
    m = compute_l2_metrics([0, 0, 0], [0.1, 0.2, 0.3])
    assert m["auc_roc"] == 0.0
