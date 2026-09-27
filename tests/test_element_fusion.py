"""
Unit tests for Element Fusion logic.
"""

import pytest
from screen_analyzer.processing.element_fusion import ElementFusion
from screen_analyzer.detector.element_classifier import ElementClassifier


def test_button_text_fusion():
    """
    Test user example:
    If a Button exists at [200, 300, 350, 350] and OCR detects 'Save' at [220, 310, 300, 340],
    the button absorbs the text, producing a single merged element.
    """
    fusion = ElementFusion(containment_threshold=0.45)

    detected_elements = [
        {
            "type": "button",
            "bbox": [200, 300, 350, 350],
            "center": [275, 325],
            "confidence": 0.93,
            "clickable": True,
        }
    ]

    detected_texts = [
        {
            "type": "text",
            "text": "Save",
            "bbox": [220, 310, 300, 340],
            "center": [260, 325],
            "confidence": 0.98,
        }
    ]

    fused = fusion.fuse(detected_elements, detected_texts, screen_width=1920, screen_height=1080)

    assert len(fused) == 1
    elem = fused[0]
    assert elem["id"] == 1
    assert elem["type"] == "button"
    assert elem["text"] == "Save"
    assert elem["bbox"] == [200, 300, 350, 350]
    assert elem["center"] == [275, 325]
    assert elem["clickable"] is True


def test_orphan_text_becomes_textbox():
    """Test standalone OCR text not inside any container becomes a textbox element."""
    fusion = ElementFusion(include_orphan_texts=True, orphan_text_type="textbox")

    detected_elements = []  # No containers detected
    detected_texts = [
        {
            "type": "text",
            "text": "Settings",
            "bbox": [100, 50, 180, 80],
            "center": [140, 65],
            "confidence": 0.98,
        }
    ]

    fused = fusion.fuse(detected_elements, detected_texts, screen_width=1920, screen_height=1080)
    assert len(fused) == 1
    elem = fused[0]
    assert elem["id"] == 1
    assert elem["type"] == "textbox"
    assert elem["text"] == "Settings"
    assert elem["bbox"] == [100, 50, 180, 80]
    assert elem["center"] == [140, 65]
    assert elem["clickable"] is False


def test_sequential_ids_and_spatial_sorting():
    """Test elements receive sequential 1-based IDs and are sorted top-to-bottom, left-to-right."""
    fusion = ElementFusion()

    elements = [
        {"type": "button", "bbox": [100, 500, 200, 550], "center": [150, 525], "confidence": 0.9},
        {"type": "icon", "bbox": [50, 50, 80, 80], "center": [65, 65], "confidence": 0.85},
        {"type": "button", "bbox": [200, 50, 300, 90], "center": [250, 70], "confidence": 0.95},
    ]
    texts = []

    fused = fusion.fuse(elements, texts, screen_width=1000, screen_height=1000)
    assert len(fused) == 3
    assert [e["id"] for e in fused] == [1, 2, 3]

    # First row should be elements at y ~ 50 (icon at x=50, then button at x=200)
    assert fused[0]["type"] == "icon"
    assert fused[1]["type"] == "button"
    # Second row at y=500
    assert fused[2]["type"] == "button"
