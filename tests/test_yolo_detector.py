"""
Unit tests for UI Element Detectors (YOLO, CV, Hybrid, OmniParser).
"""

from pathlib import Path
import pytest
import numpy as np
import cv2
from PIL import Image, ImageDraw

from screen_analyzer.config import ScreenAnalyzerConfig
from screen_analyzer.detector.yolo_detector import (
    BaseUIDetector,
    YOLOElementDetector,
    CVElementDetector,
    HybridElementDetector,
    OmniParserDetector,
    DetectorFactory,
    normalize_class_name,
)


@pytest.fixture
def sample_ui_image() -> np.ndarray:
    """Create synthetic UI image with distinct button and checkbox."""
    img = np.ones((400, 600, 3), dtype=np.uint8) * 245

    # Button: filled rounded rectangle
    cv2.rectangle(img, (50, 50), (180, 100), (59, 130, 246), -1)

    # Checkbox: small square with border
    cv2.rectangle(img, (50, 150), (74, 174), (200, 200, 200), 2)

    # Input field: wide rectangle with border
    cv2.rectangle(img, (50, 220), (350, 260), (220, 220, 220), 2)

    return img


def test_normalize_class_name():
    """Test class normalization to 8 canonical UI types."""
    assert normalize_class_name("button") == "button"
    assert normalize_class_name("btn") == "button"
    assert normalize_class_name("check_box") == "checkbox"
    assert normalize_class_name("searchbar") == "input field"
    assert normalize_class_name("textfield") == "input field"
    assert normalize_class_name("img") == "image"
    assert normalize_class_name("hyperlink") == "link"


def test_cv_element_detector(sample_ui_image: np.ndarray):
    """Test CV contour UI element detector."""
    detector = CVElementDetector()
    results = detector.detect(sample_ui_image)

    assert isinstance(results, list)
    assert len(results) >= 2

    types_found = {r["type"] for r in results}
    assert "button" in types_found or "checkbox" in types_found or "input field" in types_found

    for elem in results:
        assert elem["type"] in [
            "button", "icon", "checkbox", "textbox", "menu", "image", "link", "input field"
        ]
        assert len(elem["bbox"]) == 4
        assert len(elem["center"]) == 2
        assert 0.0 <= elem["confidence"] <= 1.0


def test_yolo_element_detector_lazy_load(tmp_path: Path):
    """Test YOLOElementDetector lazy loads model."""
    model_path = Path(__file__).resolve().parent.parent / "screen_analyzer" / "models" / "yolov8_ui.pt"
    detector = YOLOElementDetector(model_path=model_path, device="cpu")
    assert detector._is_initialized is False

    detector.load_model()
    assert detector._is_initialized is True


def test_hybrid_detector(sample_ui_image: np.ndarray):
    """Test HybridElementDetector combines detectors."""
    model_path = Path(__file__).resolve().parent.parent / "screen_analyzer" / "models" / "yolov8_ui.pt"
    yolo = YOLOElementDetector(model_path=model_path, device="cpu")
    cv_det = CVElementDetector()
    hybrid = HybridElementDetector(yolo_detector=yolo, cv_detector=cv_det)

    results = hybrid.detect(sample_ui_image)
    assert isinstance(results, list)
    assert len(results) > 0


def test_detector_factory():
    """Test DetectorFactory creates correct detector types."""
    cfg_cv = ScreenAnalyzerConfig(detector_type="cv")
    d1 = DetectorFactory.create_detector(cfg_cv)
    assert isinstance(d1, CVElementDetector)

    cfg_hybrid = ScreenAnalyzerConfig(detector_type="hybrid")
    d2 = DetectorFactory.create_detector(cfg_hybrid)
    assert isinstance(d2, HybridElementDetector)

    cfg_omniparser = ScreenAnalyzerConfig(detector_type="omniparser")
    d3 = DetectorFactory.create_detector(cfg_omniparser)
    assert isinstance(d3, OmniParserDetector)
