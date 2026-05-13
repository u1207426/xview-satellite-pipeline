# xView Satellite AI Pipeline

A two-stage pipeline for military facility detection and classification in satellite imagery, built on the [xView dataset](http://xviewdataset.org).

## Architecture

```
Satellite Image
      │
      ▼
┌─────────────┐
│  L1 Detector │  YOLOv11m — detects aircraft, storage tanks, vehicle lots, buildings
└─────────────┘
      │  detections + 31-dim structural features
      ▼
┌─────────────┐
│ L2 Classifier│  SatMAE ViT-Base + MLP fusion — P(military facility)
└─────────────┘
      │
      ▼
annotated.jpg · result.json · report.html
```

**L1** fine-tunes YOLOv11m on 4 xView object classes across 640×640 tiles.  
**L2** fuses a ViT image branch with a 31-dim structural feature vector (class counts, spatial distribution, confidence statistics) to classify whether a location is a military facility.

## Project Structure

```
defence-satellite-pipeline/
├── configs/
│   ├── l1_yolo11.yaml          # L1 training hyperparameters
│   ├── l2_classifier.yaml      # L2 training hyperparameters
│   └── data_sources.yaml       # Dataset paths and class mappings
├── src/
│   ├── data/
│   │   ├── tiling.py           # Large image → 640/1024px tile slicing
│   │   ├── converters.py       # xView GeoJSON → YOLO label format
│   │   └── dataset.py          # L2 PyTorch Dataset (image + structural features)
│   ├── models/
│   │   ├── l1_detector.py      # YOLOv11m wrapper
│   │   ├── l2_classifier.py    # ViT-Base + MLP fusion classifier
│   │   └── feature_extractor.py# 31-dim structural feature extraction
│   ├── training/
│   │   ├── train_l1.py         # L1 training entry point
│   │   └── train_l2.py         # L2 training + L1 feature caching
│   ├── evaluation/
│   │   └── metrics.py          # AUC-ROC, F1, confusion matrix
│   └── inference/
│       └── predict.py          # End-to-end inference on a single image
├── notebooks/
│   └── colab_training.ipynb    # Full training pipeline for Google Colab T4
└── tests/                      # 42 pytest tests
```

## Quickstart (Local)

### Prerequisites

```bash
pip install -r requirements.txt
```

Requires Python 3.9+, PyTorch 2.0+, CUDA recommended.

### 1. Prepare Data

Download the xView dataset from [xviewdataset.org](http://xviewdataset.org) (registration required) and place files at:

```
data/raw/xview/
├── train_images/
└── xView_train.geojson
```

Convert annotations and tile images:

```bash
python - <<'EOF'
from src.data.converters import convert_xview_to_yolo
from src.data.tiling import tile_image

convert_xview_to_yolo(
    geojson_path="data/raw/xview/xView_train.geojson",
    image_dir="data/raw/xview/train_images",
    output_dir="data/annotations/yolo_format",
)
EOF
```

### 2. Train L1 (YOLOv11m)

```bash
python -m src.training.train_l1 \
  --config configs/l1_yolo11.yaml \
  --data data/xview_dataset.yaml
```

Checkpoint saved to `checkpoints/l1/weights/best.pt`.

### 3. Cache L1 Features & Train L2

```bash
python -m src.training.train_l2 \
  --config configs/l2_classifier.yaml \
  --train-images data/processed/tiles_1024/train \
  --val-images   data/processed/tiles_1024/val \
  --features-dir data/annotations/l2_features \
  --train-labels data/annotations/location_labels/train.csv \
  --val-labels   data/annotations/location_labels/val.csv
```

L2 label CSV format:

```
image_name,label
1102.tif,0
1234.tif,1
```

`1` = military facility, `0` = non-military.

### 4. Inference

```bash
python -m src.inference.predict \
  --image path/to/image.tif \
  --l1-checkpoint checkpoints/l1/weights/best.pt \
  --l2-checkpoint checkpoints/l2/best.pt
```

Outputs written to `outputs/<location_id>_<timestamp>/`:

| File | Content |
|---|---|
| `annotated.jpg` | Image with L1 bounding boxes |
| `result.json` | All detections + L2 probability |
| `report.html` | Human-readable analysis report |

**Confidence thresholds:**

| P(military) | Classification | Level |
|---|---|---|
| ≥ 0.8 | military | high |
| 0.6 – 0.8 | military | medium |
| 0.4 – 0.6 | uncertain | low |
| < 0.4 | non-military | — |

## Google Colab Training

Open `notebooks/colab_training.ipynb` in Colab with a T4 GPU runtime. The notebook handles Drive mounting, dependency installation, tiling, L1/L2 training, and session resume automatically.

Estimated time on T4:

| Step | Time |
|---|---|
| Tiling (CPU) | 30–60 min |
| L1 training (100 epochs) | 2–4 hours |
| Feature caching | 20–40 min |
| L2 training (50 epochs) | 30–60 min |

## Model Details

### L1 — YOLOv11m

| Parameter | Value |
|---|---|
| Base model | `yolo11m.pt` (COCO pretrained) |
| Classes | aircraft, storage_tank, vehicle_lot, building |
| Input size | 640 × 640 |
| Frozen layers | 10 (during warmup) |
| Augmentation | mosaic, flipud, 45° rotation, HSV jitter |

### L2 — SatMAE ViT-Base + MLP Fusion

| Parameter | Value |
|---|---|
| Image backbone | `facebook/vit-mae-base` (swap for SatMAE when available) |
| Structural input | 31-dim vector (class counts, spatial stats, confidence stats) |
| Freeze schedule | Backbone frozen for first 5 epochs, then last 4 blocks unfrozen |
| Loss | BCEWithLogitsLoss, pos_weight = 3.0 |
| Backbone LR | 1e-5 |
| Head LR | 1e-3 |

The 31-dim structural feature vector encodes:
- 13 per-class detection counts and confidence means
- 4 spatial distribution statistics (centroid, spread)
- 13 confidence bucket counts
- 1 total detection count

## Tests

```bash
pytest
```

42 tests, ~80 seconds.

## Dataset

This project uses [xView](http://xviewdataset.org) (DIUx, 2018), a satellite imagery dataset for object detection containing 1 million+ instances across 60 classes. Access requires registration and agreement to the xView Terms of Use.

## License

For research use only. xView dataset usage is subject to the [xView Terms of Use](http://xviewdataset.org).
