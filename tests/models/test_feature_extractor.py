# tests/models/test_feature_extractor.py
import numpy as np
import pytest
from src.models.feature_extractor import Detection, extract_features, FACILITY_CLASSES, NUM_CLASSES


def test_empty_detections_returns_zero_vector():
    feat = extract_features([])
    assert feat.shape == (31,)
    assert feat.sum() == 0.0


def test_single_detection_count():
    det = Detection(class_name="aircraft", confidence=0.9, bbox=[0.1, 0.2, 0.3, 0.4])
    feat = extract_features([det])
    aircraft_idx = FACILITY_CLASSES.index("aircraft")
    assert feat[aircraft_idx] == 1.0  # count


def test_single_detection_max_confidence():
    det = Detection(class_name="aircraft", confidence=0.85, bbox=[0.1, 0.2, 0.3, 0.4])
    feat = extract_features([det])
    aircraft_idx = FACILITY_CLASSES.index("aircraft")
    assert abs(feat[NUM_CLASSES + 4 + aircraft_idx] - 0.85) < 1e-5  # max_confidence slot


def test_single_detection_total_count():
    det = Detection(class_name="runway", confidence=0.7, bbox=[0.0, 0.0, 1.0, 0.1])
    feat = extract_features([det])
    assert feat[30] == 1.0  # total_count is last element


def test_spatial_stats_single_detection():
    # bbox [0.1, 0.2, 0.3, 0.4] → center (0.2, 0.3)
    det = Detection(class_name="aircraft", confidence=0.9, bbox=[0.1, 0.2, 0.3, 0.4])
    feat = extract_features([det])
    mean_x = feat[NUM_CLASSES]
    mean_y = feat[NUM_CLASSES + 1]
    assert abs(mean_x - 0.2) < 1e-5
    assert abs(mean_y - 0.3) < 1e-5


def test_multiple_class_counts():
    dets = [
        Detection("aircraft", 0.9, [0.1, 0.1, 0.2, 0.2]),
        Detection("aircraft", 0.8, [0.3, 0.3, 0.4, 0.4]),
        Detection("runway",   0.7, [0.0, 0.0, 1.0, 0.1]),
    ]
    feat = extract_features(dets)
    aircraft_idx = FACILITY_CLASSES.index("aircraft")
    runway_idx = FACILITY_CLASSES.index("runway")
    assert feat[aircraft_idx] == 2.0
    assert feat[runway_idx] == 1.0
    assert feat[30] == 3.0  # total


def test_unknown_class_ignored():
    det = Detection(class_name="unknown_facility", confidence=0.9, bbox=[0.1, 0.1, 0.2, 0.2])
    feat = extract_features([det])
    assert feat.sum() == 0.0


def test_output_dtype_float32():
    feat = extract_features([])
    assert feat.dtype == np.float32
