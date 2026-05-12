# src/models/l1_detector.py
from __future__ import annotations
from pathlib import Path
from typing import List, Optional

import yaml

from src.models.feature_extractor import Detection


try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover – ultralytics not installed in test env
    YOLO = None  # type: ignore[assignment,misc]


class L1Detector:
    """
    Thin wrapper around YOLOv8.
    train() sets WANDB_PROJECT so ultralytics auto-logs to W&B.
    predict() returns normalized-coordinate Detection objects.
    """

    def __init__(self, checkpoint_path: Optional[str] = None):
        model_path = checkpoint_path or "yolov8m.pt"
        if YOLO is None:
            raise ImportError("ultralytics is required for L1Detector. Install with: pip install ultralytics")
        self.model = YOLO(model_path)

    def train(self, config_path: str, data_yaml: str) -> None:
        import os
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        aug = cfg.get("augmentation", {})
        self.model.train(
            data=data_yaml,
            epochs=cfg.get("epochs", 100),
            imgsz=cfg.get("imgsz", 640),
            batch=cfg.get("batch", 16),
            optimizer=cfg.get("optimizer", "AdamW"),
            lr0=cfg.get("lr0", 0.001),
            freeze=cfg.get("freeze", 10),
            degrees=aug.get("degrees", 45.0),
            flipud=aug.get("flipud", 0.5),
            mosaic=aug.get("mosaic", 1.0),
            hsv_h=aug.get("hsv_h", 0.015),
            project="checkpoints",
            name="l1",
            save=True,
        )

    def predict(self, image_path: str, conf_threshold: float = 0.25) -> List[Detection]:
        results = self.model(image_path, conf=conf_threshold, verbose=False)
        if not results:
            return []
        result = results[0]
        detections = []
        for box in result.boxes:
            cls_idx = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxyn[0].tolist()
            detections.append(Detection(
                class_name=result.names[cls_idx],
                confidence=conf,
                bbox=[x1, y1, x2, y2],
            ))
        return detections
