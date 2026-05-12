# tests/data/test_converters.py
import json
import os
import tempfile
from pathlib import Path
from PIL import Image
import pytest
from src.data.converters import convert_xview_to_yolo, XVIEW_CLASS_MAP, CLASS_TO_IDX


def make_geojson(image_id: str, class_id: int, coords: list) -> dict:
    return {
        "features": [{
            "type": "Feature",
            "properties": {"type_id": class_id, "image_id": image_id},
            "geometry": {"type": "Polygon", "coordinates": [coords]},
        }]
    }


def test_known_class_produces_label_file():
    with tempfile.TemporaryDirectory() as tmp:
        img_path = os.path.join(tmp, "test.tif")
        Image.new("RGB", (500, 500)).save(img_path)
        coords = [[100, 100], [200, 100], [200, 150], [100, 150], [100, 100]]
        gj = make_geojson("test.tif", 11, coords)  # class 11 = aircraft
        gj_path = os.path.join(tmp, "ann.geojson")
        with open(gj_path, "w") as f:
            json.dump(gj, f)
        out_dir = os.path.join(tmp, "labels")
        convert_xview_to_yolo(gj_path, tmp, out_dir)
        label = Path(out_dir) / "test.txt"
        assert label.exists()


def test_unknown_class_skips_annotation():
    with tempfile.TemporaryDirectory() as tmp:
        img_path = os.path.join(tmp, "img.tif")
        Image.new("RGB", (500, 500)).save(img_path)
        coords = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]
        gj = make_geojson("img.tif", 999, coords)  # unknown class
        gj_path = os.path.join(tmp, "ann.geojson")
        with open(gj_path, "w") as f:
            json.dump(gj, f)
        out_dir = os.path.join(tmp, "labels")
        convert_xview_to_yolo(gj_path, tmp, out_dir)
        label = Path(out_dir) / "img.txt"
        assert not label.exists()


def test_yolo_format_values_normalized():
    with tempfile.TemporaryDirectory() as tmp:
        img_path = os.path.join(tmp, "scene.tif")
        Image.new("RGB", (1000, 1000)).save(img_path)
        # bbox from (100,200) to (300,400) on a 1000x1000 image
        coords = [[100, 200], [300, 200], [300, 400], [100, 400], [100, 200]]
        gj = make_geojson("scene.tif", 11, coords)
        gj_path = os.path.join(tmp, "ann.geojson")
        with open(gj_path, "w") as f:
            json.dump(gj, f)
        out_dir = os.path.join(tmp, "labels")
        convert_xview_to_yolo(gj_path, tmp, out_dir)
        line = (Path(out_dir) / "scene.txt").read_text().strip()
        parts = line.split()
        assert len(parts) == 5
        class_idx, cx, cy, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        assert class_idx == CLASS_TO_IDX["aircraft"]
        assert abs(cx - 0.2) < 1e-4   # (100+300)/2 / 1000
        assert abs(cy - 0.3) < 1e-4   # (200+400)/2 / 1000
        assert abs(bw - 0.2) < 1e-4   # (300-100) / 1000
        assert abs(bh - 0.2) < 1e-4   # (400-200) / 1000


def test_missing_image_skips_silently():
    with tempfile.TemporaryDirectory() as tmp:
        coords = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]
        gj = make_geojson("nonexistent.tif", 11, coords)
        gj_path = os.path.join(tmp, "ann.geojson")
        with open(gj_path, "w") as f:
            json.dump(gj, f)
        out_dir = os.path.join(tmp, "labels")
        convert_xview_to_yolo(gj_path, tmp, out_dir)
        assert not (Path(out_dir) / "nonexistent.txt").exists()
