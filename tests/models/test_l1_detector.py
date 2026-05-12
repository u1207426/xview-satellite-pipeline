# tests/models/test_l1_detector.py
import os
import tempfile
from pathlib import Path
from PIL import Image
from unittest.mock import MagicMock, patch
import pytest
from src.models.feature_extractor import Detection


def make_rgb_image(path: str, size=(640, 640)):
    Image.new("RGB", size, color=(50, 100, 150)).save(path)


def test_predict_returns_list_of_detections():
    with tempfile.TemporaryDirectory() as tmp:
        img_path = os.path.join(tmp, "test.jpg")
        make_rgb_image(img_path)

        mock_box = MagicMock()
        mock_box.cls = MagicMock()
        mock_box.cls.__getitem__ = lambda self, i: MagicMock(__int__=lambda s: 0)
        mock_box.conf = MagicMock()
        mock_box.conf.__getitem__ = lambda self, i: MagicMock(__float__=lambda s: 0.9)
        mock_box.xyxyn = MagicMock()
        mock_box.xyxyn.__getitem__ = lambda self, i: MagicMock(tolist=lambda: [0.1, 0.2, 0.3, 0.4])

        mock_result = MagicMock()
        mock_result.names = {0: "aircraft"}
        mock_result.boxes = [mock_box]

        mock_yolo = MagicMock()
        mock_yolo.return_value = [mock_result]

        with patch("src.models.l1_detector.YOLO", return_value=mock_yolo):
            from src.models.l1_detector import L1Detector
            detector = L1Detector.__new__(L1Detector)
            detector.model = mock_yolo
            results = detector.predict(img_path)

        assert isinstance(results, list)
        assert all(isinstance(d, Detection) for d in results)


def test_predict_empty_results():
    with patch("src.models.l1_detector.YOLO") as MockYOLO:
        mock_model = MagicMock()
        mock_model.return_value = []
        MockYOLO.return_value = mock_model

        from src.models.l1_detector import L1Detector
        detector = L1Detector.__new__(L1Detector)
        detector.model = mock_model
        results = detector.predict("fake.jpg")
        assert results == []
