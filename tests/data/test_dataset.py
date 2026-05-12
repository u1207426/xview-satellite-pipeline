# tests/data/test_dataset.py
import csv
import os
import tempfile
import numpy as np
import torch
from PIL import Image
import pytest
from src.data.dataset import LocationDataset


def make_dataset_fixtures(tmp: str, n=4) -> tuple:
    img_dir = os.path.join(tmp, "images")
    feat_dir = os.path.join(tmp, "features")
    os.makedirs(img_dir); os.makedirs(feat_dir)
    labels_path = os.path.join(tmp, "labels.csv")

    with open(labels_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label"])
        writer.writeheader()
        for i in range(n):
            fname = f"loc_{i:03d}.jpg"
            Image.new("RGB", (1024, 1024), color=(i * 30, 100, 200)).save(os.path.join(img_dir, fname))
            np.save(os.path.join(feat_dir, f"loc_{i:03d}.npy"), np.zeros(31, dtype=np.float32))
            writer.writerow({"filename": fname, "label": i % 2})

    return img_dir, feat_dir, labels_path


def test_dataset_len():
    with tempfile.TemporaryDirectory() as tmp:
        img_dir, feat_dir, labels = make_dataset_fixtures(tmp, n=4)
        ds = LocationDataset(img_dir, feat_dir, labels)
        assert len(ds) == 4


def test_dataset_item_shapes():
    with tempfile.TemporaryDirectory() as tmp:
        img_dir, feat_dir, labels = make_dataset_fixtures(tmp, n=2)
        ds = LocationDataset(img_dir, feat_dir, labels)
        pixel_values, struct_features, label = ds[0]
        assert pixel_values.shape == (3, 224, 224)
        assert struct_features.shape == (31,)
        assert label.shape == ()


def test_dataset_label_values():
    with tempfile.TemporaryDirectory() as tmp:
        img_dir, feat_dir, labels = make_dataset_fixtures(tmp, n=4)
        ds = LocationDataset(img_dir, feat_dir, labels)
        for i in range(4):
            _, _, lbl = ds[i]
            assert lbl.item() == i % 2


def test_dataset_pixel_values_are_tensors():
    with tempfile.TemporaryDirectory() as tmp:
        img_dir, feat_dir, labels = make_dataset_fixtures(tmp, n=1)
        ds = LocationDataset(img_dir, feat_dir, labels)
        pixel_values, _, _ = ds[0]
        assert isinstance(pixel_values, torch.Tensor)
        assert pixel_values.dtype == torch.float32
