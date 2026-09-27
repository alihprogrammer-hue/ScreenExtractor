"""
Unit tests for ScreenshotLoader and ImagePreprocessor.
"""

from pathlib import Path
import io
import pytest
import numpy as np
from PIL import Image

from screen_analyzer.processing.image_processor import (
    ScreenshotLoader,
    ImagePreprocessor,
    ImageContainer,
)


@pytest.fixture
def sample_image_file(tmp_path: Path) -> Path:
    """Create a temporary PNG image."""
    img_path = tmp_path / "test_screen.png"
    img = Image.new("RGB", (800, 600), color=(255, 255, 255))
    img.save(img_path)
    return img_path


def test_loader_from_file_path(sample_image_file: Path):
    """Test loading screenshot from a valid file path."""
    container = ScreenshotLoader.load(str(sample_image_file))
    assert isinstance(container, ImageContainer)
    assert container.width == 800
    assert container.height == 600
    assert container.rgb.shape == (600, 800, 3)
    assert container.bgr.shape == (600, 800, 3)


def test_loader_from_bytes(sample_image_file: Path):
    """Test loading screenshot from in-memory bytes."""
    with open(sample_image_file, "rb") as f:
        data = f.read()
    container = ScreenshotLoader.load(data)
    assert container.width == 800
    assert container.height == 600
    assert container.source_type == "memory_bytes"


def test_loader_from_numpy():
    """Test loading screenshot from existing numpy array."""
    arr = np.zeros((300, 400, 3), dtype=np.uint8)
    container = ScreenshotLoader.load(arr)
    assert container.width == 400
    assert container.height == 300
    assert container.source_type == "numpy_array"


def test_loader_from_pil():
    """Test loading screenshot from PIL Image object."""
    pil_img = Image.new("RGB", (640, 480), color=(100, 150, 200))
    container = ScreenshotLoader.load(pil_img)
    assert container.width == 640
    assert container.height == 480
    assert container.source_type == "pil_image"


def test_loader_file_not_found():
    """Test loader raises FileNotFoundError for non-existent path."""
    with pytest.raises(FileNotFoundError):
        ScreenshotLoader.load("/path/to/non_existent_file_9999.png")


def test_loader_empty_bytes():
    """Test loader raises ValueError on empty bytes."""
    with pytest.raises(ValueError):
        ScreenshotLoader.load(b"")


def test_preprocessor_resizing():
    """Test resizing logic preserves aspect ratio and scales for weak CPU."""
    prep = ImagePreprocessor(max_dim=1000, resize_for_detection=True)
    # 4K image 3840 x 2160
    large_img = np.zeros((2160, 3840, 3), dtype=np.uint8)
    resized, sx, sy = prep.preprocess_for_detection(large_img)

    assert max(resized.shape[:2]) == 1000
    assert round(sx, 2) == round(3840 / 1000, 2)
    assert round(sy, 2) == round(3840 / 1000, 2)


def test_preprocessor_no_resize_when_small():
    """Test preprocessor does not resize images smaller than max_dim."""
    prep = ImagePreprocessor(max_dim=1280, resize_for_detection=True)
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    resized, sx, sy = prep.preprocess_for_detection(img)

    assert resized.shape == (600, 800, 3)
    assert sx == 1.0
    assert sy == 1.0


def test_preprocessor_scale_bbox():
    """Test scaling bounding boxes back to original coordinates."""
    prep = ImagePreprocessor()
    # Scaled down by 2x: [50, 50, 100, 100] -> original [100, 100, 200, 200]
    scaled = prep.scale_bbox([50, 50, 100, 100], scale_x=2.0, scale_y=2.0, max_w=1920, max_h=1080)
    assert scaled == [100, 100, 200, 200]


def test_preprocessor_scale_bbox_clamping():
    """Test scaling clamps within image boundary."""
    prep = ImagePreprocessor()
    scaled = prep.scale_bbox([900, 500, 1100, 600], scale_x=2.0, scale_y=2.0, max_w=1920, max_h=1080)
    assert scaled[2] == 1920  # clamped to max_w
    assert scaled[3] == 1080  # clamped to max_h
