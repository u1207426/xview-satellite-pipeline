# src/data/converters.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict

from PIL import Image

# xView type_id → our category name.
# Verify IDs against xview_class_labels.txt included with the xView download.
XVIEW_CLASS_MAP: Dict[int, str] = {
    11: "aircraft",
    12: "aircraft",
    15: "aircraft",
    17: "building",
    23: "storage_tank",
    53: "vehicle_lot",
    72: "vehicle_lot",
}

OUR_CLASSES = ["aircraft", "storage_tank", "vehicle_lot", "building"]
CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(OUR_CLASSES)}


def convert_xview_to_yolo(
    geojson_path: str,
    image_dir: str,
    output_dir: str,
) -> None:
    """
    Convert xView GeoJSON annotations into per-image YOLO .txt label files.
    Each line: <class_idx> <cx> <cy> <w> <h>  (all normalized to [0,1])
    Images not found in image_dir are silently skipped.
    Annotations with class IDs not in XVIEW_CLASS_MAP are skipped.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(geojson_path) as f:
        data = json.load(f)

    by_image: Dict[str, list] = {}
    for feat in data.get("features", []):
        props = feat.get("properties", {})
        type_id = int(props.get("type_id", -1))
        if type_id not in XVIEW_CLASS_MAP:
            continue
        img_id = props.get("image_id", "")
        coords = feat["geometry"]["coordinates"][0]
        by_image.setdefault(img_id, []).append((type_id, coords))

    for img_id, annotations in by_image.items():
        img_path = Path(image_dir) / img_id
        if not img_path.exists():
            continue
        with Image.open(img_path) as im:
            img_w, img_h = im.size

        lines = []
        for type_id, coords in annotations:
            category = XVIEW_CLASS_MAP[type_id]
            class_idx = CLASS_TO_IDX[category]
            xs = [p[0] for p in coords]
            ys = [p[1] for p in coords]
            cx = (min(xs) + max(xs)) / 2 / img_w
            cy = (min(ys) + max(ys)) / 2 / img_h
            bw = (max(xs) - min(xs)) / img_w
            bh = (max(ys) - min(ys)) / img_h
            lines.append(f"{class_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        (output_dir / (Path(img_id).stem + ".txt")).write_text("\n".join(lines))
