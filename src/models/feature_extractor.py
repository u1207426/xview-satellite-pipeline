# src/models/feature_extractor.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List

import numpy as np

FACILITY_CLASSES = [
    "aircraft", "hangar", "vehicle_lot", "runway", "radar",
    "antenna", "storage_tank", "building", "bunker",
    "training_ground", "barracks", "helipad", "vehicle",
]
NUM_CLASSES = len(FACILITY_CLASSES)  # 13
_CLASS_TO_IDX = {c: i for i, c in enumerate(FACILITY_CLASSES)}


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2] normalized to [0, 1]


def extract_features(detections: List[Detection]) -> np.ndarray:
    """
    Convert L1 detections to a 31-dim float32 feature vector.

    Layout:
        [0 : NUM_CLASSES]              facility_counts  (13 dims)
        [NUM_CLASSES : NUM_CLASSES+4]  spatial_stats: mean_x, mean_y, std_x, std_y (4 dims)
        [NUM_CLASSES+4 : 30]           max_confidence per class (13 dims)
        [30]                           total_count (1 dim)
    """
    counts = np.zeros(NUM_CLASSES, dtype=np.float32)
    max_conf = np.zeros(NUM_CLASSES, dtype=np.float32)
    centers_x: List[float] = []
    centers_y: List[float] = []

    for det in detections:
        idx = _CLASS_TO_IDX.get(det.class_name)
        if idx is None:
            continue
        counts[idx] += 1.0
        if det.confidence > max_conf[idx]:
            max_conf[idx] = det.confidence
        centers_x.append((det.bbox[0] + det.bbox[2]) / 2.0)
        centers_y.append((det.bbox[1] + det.bbox[3]) / 2.0)

    if centers_x:
        spatial = np.array(
            [np.mean(centers_x), np.mean(centers_y), np.std(centers_x), np.std(centers_y)],
            dtype=np.float32,
        )
    else:
        spatial = np.zeros(4, dtype=np.float32)

    total = np.array(
        [float(len([d for d in detections if _CLASS_TO_IDX.get(d.class_name) is not None]))],
        dtype=np.float32,
    )
    return np.concatenate([counts, spatial, max_conf, total])
