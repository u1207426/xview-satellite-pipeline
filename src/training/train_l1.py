# src/training/train_l1.py
from __future__ import annotations
import argparse
import os

from src.models.l1_detector import L1Detector


def train_l1(
    config_path: str,
    data_yaml: str,
    wandb_project: str = "satellite-l1",
) -> None:
    """
    Train L1 YOLOv11 object detector.
    Ultralytics auto-integrates with W&B when wandb is installed and
    WANDB_PROJECT env var is set.

    Args:
        config_path:    Path to configs/l1_yolo11.yaml
        data_yaml:      Path to YOLO dataset YAML (data/xview_dataset.yaml)
        wandb_project:  W&B project name
    """
    os.environ.setdefault("WANDB_PROJECT", wandb_project)
    detector = L1Detector()
    detector.train(config_path=config_path, data_yaml=data_yaml)
    print("L1 training complete. Best checkpoint: checkpoints/l1/weights/best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/l1_yolo11.yaml")
    parser.add_argument("--data", default="data/xview_dataset.yaml")
    parser.add_argument("--wandb-project", default="satellite-l1")
    args = parser.parse_args()
    train_l1(args.config, args.data, args.wandb_project)
