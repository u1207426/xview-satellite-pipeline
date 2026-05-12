# tests/models/test_l1_detector.py
import os
import tempfile
from unittest.mock import MagicMock, patch
from PIL import Image
import pytest
from src.models.feature_extractor import Detection


def make_rgb_image(path: str, size=(640, 640)):
    Image.new("RGB", size, color=(50, 100, 150)).save(path)


def _make_mock_yolo_class():
    """Return a mock YOLO class whose instances behave like a trained model."""
    mock_box = MagicMock()
    mock_box.cls = [MagicMock(__int__=lambda s: 0)]
    mock_box.conf = [MagicMock(__float__=lambda s: 0.9)]
    mock_box.xyxyn = [MagicMock(tolist=lambda: [0.1, 0.2, 0.3, 0.4])]

    mock_result = MagicMock()
    mock_result.names = {0: "aircraft"}
    mock_result.boxes = [mock_box]

    mock_instance = MagicMock()
    mock_instance.return_value = [mock_result]

    MockYOLO = MagicMock(return_value=mock_instance)
    return MockYOLO, mock_instance


def test_predict_returns_list_of_detections():
    with tempfile.TemporaryDirectory() as tmp:
        img_path = os.path.join(tmp, "test.jpg")
        make_rgb_image(img_path)

        MockYOLO, mock_model = _make_mock_yolo_class()
        with patch("src.models.l1_detector.YOLO", MockYOLO):
            from importlib import reload
            import src.models.l1_detector as mod
            mod.YOLO = MockYOLO
            from src.models.l1_detector import L1Detector
            detector = L1Detector()
            detector.model = mock_model
            results = detector.predict(img_path)

        assert isinstance(results, list)
        assert all(isinstance(d, Detection) for d in results)


def test_predict_empty_results():
    mock_instance = MagicMock()
    mock_instance.return_value = []
    MockYOLO = MagicMock(return_value=mock_instance)

    with patch("src.models.l1_detector.YOLO", MockYOLO):
        import src.models.l1_detector as mod
        mod.YOLO = MockYOLO
        from src.models.l1_detector import L1Detector
        detector = L1Detector()
        detector.model = mock_instance
        results = detector.predict("fake.jpg")

    assert results == []


def test_init_raises_if_yolo_unavailable():
    with patch("src.models.l1_detector.YOLO", None):
        import src.models.l1_detector as mod
        original = mod.YOLO
        mod.YOLO = None
        try:
            from src.models.l1_detector import L1Detector
            with pytest.raises(ImportError, match="ultralytics"):
                L1Detector()
        finally:
            mod.YOLO = original


def test_train_passes_config_params(tmp_path):
    import yaml
    # Write a minimal config
    cfg = {
        "epochs": 5,
        "imgsz": 320,
        "batch": 4,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "freeze": 2,
        "augmentation": {"hsv_h": 0.015, "flipud": 0.5, "mosaic": 1.0, "degrees": 45.0},
    }
    cfg_path = str(tmp_path / "l1.yaml")
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)

    mock_instance = MagicMock()
    MockYOLO = MagicMock(return_value=mock_instance)

    with patch("src.models.l1_detector.YOLO", MockYOLO):
        import src.models.l1_detector as mod
        mod.YOLO = MockYOLO
        from src.models.l1_detector import L1Detector
        detector = L1Detector()
        detector.model = mock_instance
        detector.train(cfg_path, "data/dataset.yaml")

    mock_instance.train.assert_called_once()
    call_kwargs = mock_instance.train.call_args[1]
    assert call_kwargs["epochs"] == 5
    assert call_kwargs["imgsz"] == 320
    assert call_kwargs["project"] == "checkpoints"
    assert call_kwargs["name"] == "l1"
    assert call_kwargs["save"] is True
