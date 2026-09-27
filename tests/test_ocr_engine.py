"""
Unit tests for OCR Engine components.
"""

import pytest
import numpy as np
from PIL import Image, ImageDraw

from screen_analyzer.config import ScreenAnalyzerConfig
from screen_analyzer.ocr.ocr_engine import (
    BaseOCREngine,
    HeuristicOCREngine,
    PaddleOCREngine,
    OCREngineFactory,
)


@pytest.fixture
def test_text_image() -> np.ndarray:
    """Create an RGB image with clear synthetic UI text."""
    img = Image.new("RGB", (400, 150), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((30, 30), "Settings", fill=(0, 0, 0))
    draw.text((30, 80), "Cancel", fill=(0, 0, 0))
    return np.array(img)


def test_heuristic_ocr_detection(test_text_image: np.ndarray):
    """Test text detection on synthetic screenshot using HeuristicOCREngine."""
    engine = HeuristicOCREngine(conf_threshold=0.35)
    results = engine.detect_text(test_text_image)

    assert isinstance(results, list)
    assert len(results) >= 2

    # Check structure of each element
    for elem in results:
        assert elem["type"] == "text"
        assert isinstance(elem["text"], str)
        assert len(elem["text"]) > 0
        assert len(elem["bbox"]) == 4
        assert len(elem["center"]) == 2
        assert 0.0 <= elem["confidence"] <= 1.0

    # Verify recognized words
    found_words = [r["text"] for r in results]
    assert "Settings" in found_words
    assert "Cancel" in found_words


def test_paddle_ocr_lazy_loading():
    """Test PaddleOCREngine lazy loads model without crashing."""
    engine = PaddleOCREngine(auto_fallback=True)
    assert engine._is_initialized is False
    # Explicit load
    engine.load_model()
    assert engine._is_initialized is True


def test_paddle_ocr_fallback(test_text_image: np.ndarray):
    """Test PaddleOCREngine gracefully uses fallback when offline."""
    engine = PaddleOCREngine(auto_fallback=True)
    results = engine.detect_text(test_text_image)
    assert isinstance(results, list)
    assert len(results) >= 2


def test_ocr_factory():
    """Test OCREngineFactory creates appropriate engines."""
    cfg_heuristic = ScreenAnalyzerConfig(ocr_engine="heuristic")
    eng1 = OCREngineFactory.create_engine(cfg_heuristic)
    assert isinstance(eng1, HeuristicOCREngine)

    cfg_auto = ScreenAnalyzerConfig(ocr_engine="auto")
    eng2 = OCREngineFactory.create_engine(cfg_auto)
    assert isinstance(eng2, PaddleOCREngine)
