# src/data/dataset.py
from __future__ import annotations
import csv
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

L2_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class LocationDataset(Dataset):
    """
    PyTorch Dataset for L2 location classification.

    Expects:
        image_dir/    *.jpg tiles (typically 1024x1024)
        features_dir/ *.npy files with 31-dim feature vectors (same stem as images)
        labels_path   CSV with header: filename,label  (0=non_military, 1=military)
    """

    def __init__(
        self,
        image_dir: str,
        features_dir: str,
        labels_path: str,
        transform=None,
    ):
        self.image_dir = Path(image_dir)
        self.features_dir = Path(features_dir)
        self.transform = transform or L2_TRANSFORM
        self.samples: List[Tuple[str, int]] = []
        with open(labels_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row["filename"], int(row["label"])))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        filename, label = self.samples[idx]
        stem = Path(filename).stem
        img = Image.open(self.image_dir / filename).convert("RGB")
        pixel_values = self.transform(img)
        struct_features = torch.tensor(
            np.load(self.features_dir / f"{stem}.npy"), dtype=torch.float32
        )
        return pixel_values, struct_features, torch.tensor(label, dtype=torch.float32)
