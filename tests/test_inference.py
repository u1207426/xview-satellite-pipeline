# tests/test_inference.py
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image
import torch
import pytest
from src.models.feature_extractor import Detection


def make_test_image(path: str, size=(1280, 1280)):
    Image.new("RGB", size, color=(80, 120, 160)).save(path)


def _setup_mocks(monkeypatch):
    """Patch L1Detector and L2Classifier to avoid loading real model weights."""
    mock_det = Detection("aircraft", 0.9, [0.1, 0.2, 0.3, 0.4])

    mock_l1_instance = MagicMock()
    mock_l1_instance.predict.return_value = [mock_det]
    monkeypatch.setattr("src.inference.predict.L1Detector", lambda **kw: mock_l1_instance)

    # L2Classifier mock: forward returns logit 2.0 → sigmoid ≈ 0.88 → military/high
    mock_l2_instance = MagicMock()
    mock_l2_instance.return_value = torch.tensor([[2.0]])
    mock_l2_instance.eval.return_value = mock_l2_instance
    mock_l2_instance.to.return_value = mock_l2_instance
    mock_l2_instance.load_state_dict = MagicMock()
    monkeypatch.setattr("src.inference.predict.L2Classifier", lambda **kw: mock_l2_instance)
    monkeypatch.setattr("src.inference.predict.torch.load", lambda *a, **kw: {})


def test_predict_creates_output_files(monkeypatch, tmp_path):
    _setup_mocks(monkeypatch)
    img_path = str(tmp_path / "test.jpg")
    make_test_image(img_path)

    from src.inference.predict import predict
    predict(
        image_path=img_path,
        l1_checkpoint="fake_l1.pt",
        l2_checkpoint="fake_l2.pt",
        output_dir=str(tmp_path / "outputs"),
        location_id="loc_test",
    )

    run_dir = next((tmp_path / "outputs").iterdir())
    assert (run_dir / "annotated.jpg").exists()
    assert (run_dir / "result.json").exists()
    assert (run_dir / "report.html").exists()


def test_predict_result_json_structure(monkeypatch, tmp_path):
    _setup_mocks(monkeypatch)
    img_path = str(tmp_path / "scene.jpg")
    make_test_image(img_path)

    from src.inference.predict import predict
    result = predict(img_path, "l1.pt", "l2.pt", str(tmp_path / "out"), location_id="loc_001")

    assert result["location_id"] == "loc_001"
    assert "l1_detections" in result
    assert "l1_summary" in result
    assert "l2_result" in result
    assert "military_probability" in result["l2_result"]
    assert "classification" in result["l2_result"]


def test_predict_high_probability_classified_military(monkeypatch, tmp_path):
    _setup_mocks(monkeypatch)  # logit=2.0 → prob≈0.88 → military/high
    img_path = str(tmp_path / "scene.jpg")
    make_test_image(img_path)

    from src.inference.predict import predict
    result = predict(img_path, "l1.pt", "l2.pt", str(tmp_path / "out"))

    assert result["l2_result"]["classification"] == "military"
    assert result["l2_result"]["confidence_level"] == "high"
