"""
End-to-end integration tests for ScreenAnalysisEngine pipeline.
"""

from pathlib import Path
import pytest
from PIL import Image, ImageDraw

from screen_analyzer.config import ScreenAnalyzerConfig
from screen_analyzer.main import ScreenAnalysisEngine, generate_sample_screenshot


@pytest.fixture
def sample_screenshot_path(tmp_path: Path) -> Path:
    """Generate sample screenshot in temporary folder."""
    p = tmp_path / "test_screenshot.png"
    return generate_sample_screenshot(p)


def test_full_pipeline_execution(sample_screenshot_path: Path):
    """Test full pipeline execution: input -> detection -> ocr -> fusion -> json."""
    config = ScreenAnalyzerConfig(
        detector_type="cv",       # Fast deterministic CV for unit test
        ocr_engine="heuristic",   # Fast deterministic OCR for unit test
        max_image_dim=1280,
    )
    engine = ScreenAnalysisEngine(config)

    result = engine.analyze(sample_screenshot_path)

    # 1. Check screen dimensions
    assert "screen" in result
    assert result["screen"]["width"] == 1280
    assert result["screen"]["height"] == 720

    # 2. Check elements array
    assert "elements" in result
    elements = result["elements"]
    assert isinstance(elements, list)
    assert len(elements) > 0

    # 3. Check element structure adheres strictly to standard format
    for elem in elements:
        assert "id" in elem and isinstance(elem["id"], int) and elem["id"] >= 1
        assert "type" in elem and isinstance(elem["type"], str)
        assert elem["type"] in [
            "button", "icon", "checkbox", "textbox", "menu", "image", "link", "input field"
        ]
        assert "bbox" in elem and len(elem["bbox"]) == 4
        assert "center" in elem and len(elem["center"]) == 2
        assert "confidence" in elem and isinstance(elem["confidence"], (int, float))
        assert "clickable" in elem and isinstance(elem["clickable"], bool)

        # Coordinate integrity
        x1, y1, x2, y2 = elem["bbox"]
        cx, cy = elem["center"]
        assert 0 <= x1 < x2 <= 1280
        assert 0 <= y1 < y2 <= 720
        assert x1 <= cx <= x2
        assert y1 <= cy <= y2

    # 4. Check timing metrics reported
    metrics = engine.last_metrics
    assert "total_time_ms" in metrics
    assert "detection_time_ms" in metrics
    assert "ocr_time_ms" in metrics
    assert "fusion_time_ms" in metrics
    assert metrics["total_time_ms"] > 0


def test_visualization_generation(sample_screenshot_path: Path, tmp_path: Path):
    """Test annotated image visualization output."""
    config = ScreenAnalyzerConfig(detector_type="cv", ocr_engine="heuristic")
    engine = ScreenAnalysisEngine(config)

    result = engine.analyze(sample_screenshot_path)
    out_img = tmp_path / "annotated.png"

    engine.visualize(sample_screenshot_path, result["elements"], out_img)
    assert out_img.is_file()
    assert out_img.stat().st_size > 0
