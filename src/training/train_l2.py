# src/training/train_l2.py
from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import yaml

from src.data.dataset import LocationDataset
from src.evaluation.metrics import compute_l2_metrics
from src.models.feature_extractor import extract_features
from src.models.l1_detector import L1Detector
from src.models.l2_classifier import L2Classifier


def cache_l1_features(
    image_dir: str,
    l1_checkpoint: str,
    output_dir: str,
) -> None:
    """
    Run L1 inference on all images in image_dir and cache 31-dim feature
    vectors as {stem}.npy files in output_dir.
    Must be called before train_l2().
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detector = L1Detector(checkpoint_path=l1_checkpoint)
    images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    for img_path in images:
        detections = detector.predict(str(img_path))
        features = extract_features(detections)
        np.save(output_dir / f"{img_path.stem}.npy", features)
    print(f"Cached L1 features for {len(images)} images → {output_dir}")


def train_l2(
    config_path: str,
    train_image_dir: str,
    val_image_dir: str,
    features_dir: str,
    train_labels: str,
    val_labels: str,
    checkpoint_dir: str = "checkpoints/l2",
    wandb_project: str = "satellite-l2",
) -> None:
    """
    Train L2 location classifier.

    Prerequisites:
        1. L1 best.pt must exist (from Task 10 training run).
        2. cache_l1_features() must have been called to populate features_dir.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = L2Classifier(backbone_name=cfg.get("backbone", "facebook/vit-mae-base"))
    model.freeze_backbone()
    model = model.to(device)

    train_ds = LocationDataset(train_image_dir, features_dir, train_labels)
    val_ds = LocationDataset(val_image_dir, features_dir, val_labels)
    batch = cfg.get("batch_size", 16)
    workers = cfg.get("num_workers", 4)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=workers)

    pos_weight = torch.tensor([cfg.get("pos_weight", 3.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def make_optimizer() -> torch.optim.Optimizer:
        return torch.optim.AdamW([
            {"params": [p for p in model.backbone.parameters() if p.requires_grad],
             "lr": cfg.get("lr_backbone", 1e-5)},
            {"params": model.struct_branch.parameters(), "lr": cfg.get("lr_head", 1e-3)},
            {"params": model.fusion_head.parameters(), "lr": cfg.get("lr_head", 1e-3)},
        ])

    optimizer = make_optimizer()
    freeze_epochs = cfg.get("freeze_backbone_epochs", 5)
    unfreeze_n = cfg.get("unfreeze_last_n_blocks", 4)
    epochs = cfg.get("epochs", 50)
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    wandb.init(project=wandb_project, config=cfg)
    best_auc = 0.0

    for epoch in range(epochs):
        if epoch == freeze_epochs:
            model.unfreeze_last_n_blocks(unfreeze_n)
            optimizer = make_optimizer()

        model.train()
        train_loss = 0.0
        for pv, sf, labels in train_loader:
            pv, sf = pv.to(device), sf.to(device)
            labels = labels.unsqueeze(1).to(device)
            optimizer.zero_grad()
            loss = criterion(model(pv, sf), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        y_true, y_prob, val_loss = [], [], 0.0
        with torch.no_grad():
            for pv, sf, labels in val_loader:
                pv, sf = pv.to(device), sf.to(device)
                logits = model(pv, sf)
                val_loss += criterion(logits, labels.unsqueeze(1).to(device)).item()
                y_prob.extend(torch.sigmoid(logits).squeeze(1).cpu().tolist())
                y_true.extend(labels.int().tolist())

        metrics = compute_l2_metrics(y_true, y_prob)
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss / len(train_loader),
            "val_loss": val_loss / len(val_loader),
            **{f"val_{k}": v for k, v in metrics.items() if k != "confusion_matrix"},
        })

        if metrics["auc_roc"] > best_auc:
            best_auc = metrics["auc_roc"]
            torch.save(model.state_dict(), ckpt_dir / "best.pt")

    torch.save(model.state_dict(), ckpt_dir / "last.pt")
    wandb.finish()
    print(f"L2 training complete. Best val AUC-ROC: {best_auc:.4f}")
    print(f"Best checkpoint: {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/l2_classifier.yaml")
    parser.add_argument("--train-images", required=True)
    parser.add_argument("--val-images", required=True)
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--train-labels", required=True)
    parser.add_argument("--val-labels", required=True)
    parser.add_argument("--wandb-project", default="satellite-l2")
    args = parser.parse_args()
    train_l2(
        args.config, args.train_images, args.val_images,
        args.features_dir, args.train_labels, args.val_labels,
        wandb_project=args.wandb_project,
    )
