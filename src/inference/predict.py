# src/inference/predict.py
from __future__ import annotations
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from PIL import Image, ImageDraw
from torchvision import transforms

from src.data.tiling import tile_image
from src.models.feature_extractor import Detection, extract_features
from src.models.l1_detector import L1Detector
from src.models.l2_classifier import L2Classifier

_L2_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# (min_prob, classification, confidence_level)
_CONFIDENCE_RULES = [
    (0.8, "military",     "high"),
    (0.6, "military",     "medium"),
    (0.4, "uncertain",    "low"),
    (0.0, "non-military", None),
]


def _classify(prob: float):
    for threshold, classification, level in _CONFIDENCE_RULES:
        if prob >= threshold:
            return classification, level
    return "non-military", None


def _annotate(image_path: str, detections: list) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for det in detections:
        x1, y1 = det.bbox[0] * w, det.bbox[1] * h
        x2, y2 = det.bbox[2] * w, det.bbox[3] * h
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, max(0, y1 - 14)), f"{det.class_name} {det.confidence:.2f}", fill="red")
    return img


def _html_report(result: dict, img_name: str) -> str:
    rows = "".join(
        f"<tr><td>{d['class']}</td><td>{d['confidence']:.3f}</td></tr>"
        for d in result["l1_detections"]
    )
    l2 = result["l2_result"]
    level_str = f" ({l2['confidence_level']})" if l2.get("confidence_level") else ""
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>Satellite Analysis Report</title>"
        "<style>body{font-family:sans-serif;padding:20px}"
        "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:8px}</style></head>"
        f"<body><h2>Location: {result['location_id']}</h2>"
        f"<p>Timestamp: {result['timestamp']}</p>"
        f"<img src='{img_name}' style='max-width:800px'><br><br>"
        "<h3>L1 Detections</h3>"
        f"<table><tr><th>Class</th><th>Confidence</th></tr>{rows}</table>"
        "<h3>L2 Classification</h3>"
        f"<p>Military Probability: <strong>{l2['military_probability']:.3f}</strong></p>"
        f"<p>Classification: <strong>{l2['classification']}{level_str}</strong></p>"
        "</body></html>"
    )


def predict(
    image_path: str,
    l1_checkpoint: str,
    l2_checkpoint: str,
    output_dir: str = "outputs",
    location_id: Optional[str] = None,
    backbone_name: str = "facebook/vit-mae-base",
) -> dict:
    """
    Run end-to-end inference on a single image.
    Saves annotated.jpg, result.json, and report.html to
    {output_dir}/{location_id}_{timestamp}/.
    Returns the result dict.
    """
    image_path = Path(image_path)
    location_id = location_id or image_path.stem
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = Path(output_dir) / f"{location_id}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # L1: tile to 640 and collect all detections
    with tempfile.TemporaryDirectory() as tmp:
        tiles_640 = tile_image(str(image_path), os.path.join(tmp, "t640"), tile_size=640)
        detector = L1Detector(checkpoint_path=l1_checkpoint)
        all_dets: list = []
        for tile in tiles_640:
            all_dets.extend(detector.predict(tile.tile_path))

    # L2: tile to 1024, use first tile for image branch
    with tempfile.TemporaryDirectory() as tmp:
        tiles_1024 = tile_image(str(image_path), os.path.join(tmp, "t1024"), tile_size=1024)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        classifier = L2Classifier(backbone_name=backbone_name)
        classifier.load_state_dict(torch.load(l2_checkpoint, map_location=device))
        classifier.eval().to(device)

        struct_feat = torch.tensor(
            extract_features(all_dets), dtype=torch.float32
        ).unsqueeze(0).to(device)
        src_img = tiles_1024[0].tile_path if tiles_1024 else str(image_path)
        pv = _L2_TRANSFORM(Image.open(src_img).convert("RGB")).unsqueeze(0).to(device)

        with torch.no_grad():
            prob = float(torch.sigmoid(classifier(pv, struct_feat)).item())

    classification, level = _classify(prob)
    l1_summary: dict = {}
    for d in all_dets:
        l1_summary[d.class_name] = l1_summary.get(d.class_name, 0) + 1

    result = {
        "location_id": location_id,
        "image_path": str(image_path),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "l1_detections": [
            {"class": d.class_name,
             "confidence": round(d.confidence, 4),
             "bbox": [round(v, 4) for v in d.bbox]}
            for d in all_dets
        ],
        "l1_summary": l1_summary,
        "l2_result": {
            "military_probability": round(prob, 4),
            "classification": classification,
            "confidence_level": level,
        },
    }

    _annotate(str(image_path), all_dets).save(run_dir / "annotated.jpg")
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    (run_dir / "report.html").write_text(_html_report(result, "annotated.jpg"))
    print(f"Results saved to {run_dir}")
    return result
