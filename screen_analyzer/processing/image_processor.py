"""
Image processing module for Screen Analysis Engine.
Handles screenshot loading from multiple sources (file, memory, bytes, PIL, numpy)
and preprocessing optimizations (resizing for low-resource CPU inference, scaling).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Union, Tuple, List, Optional
import io
import logging
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class ImageContainer:
    """Standardized representation of a loaded screenshot."""
    rgb: np.ndarray         # RGB format for PyTorch / YOLO / PIL
    bgr: np.ndarray         # BGR format for OpenCV processing
    width: int              # Original width in pixels
    height: int             # Original height in pixels
    source_type: str        # "file", "bytes", "numpy", "pil"


class ScreenshotLoader:
    """
    Loads screenshots from various sources:
    - File path (PNG, JPG, BMP, WEBP, etc.)
    - In-memory bytes (clipboard / network)
    - NumPy arrays (OpenCV format)
    - PIL.Image objects
    """

    @staticmethod
    def load(source: Union[str, Path, bytes, np.ndarray, Image.Image]) -> ImageContainer:
        """
        Load screenshot from supported source types into a standardized ImageContainer.

        Args:
            source: File path, bytes, NumPy array, or PIL Image.

        Returns:
            ImageContainer with RGB and BGR arrays and original dimensions.

        Raises:
            ValueError: If source format is unsupported or invalid.
            FileNotFoundError: If source path does not exist.
        """
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.is_file():
                raise FileNotFoundError(f"Screenshot file not found: {path.resolve()}")
            
            # Load with PIL to guarantee RGB compatibility
            try:
                pil_img = Image.open(path).convert("RGB")
                rgb = np.array(pil_img)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                source_type = f"file ({path.name})"
            except Exception as e:
                # Fallback to cv2 if PIL fails
                bgr = cv2.imread(str(path))
                if bgr is None:
                    raise ValueError(f"Failed to read image from path: {path} (error: {e})")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                source_type = f"file ({path.name})"

        elif isinstance(source, bytes):
            if len(source) == 0:
                raise ValueError("Received empty byte buffer for screenshot.")
            nparr = np.frombuffer(source, np.uint8)
            bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if bgr is None:
                # Attempt PIL decoding
                try:
                    pil_img = Image.open(io.BytesIO(source)).convert("RGB")
                    rgb = np.array(pil_img)
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                except Exception as e:
                    raise ValueError(f"Could not decode image from byte buffer: {e}")
            else:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            source_type = "memory_bytes"

        elif isinstance(source, np.ndarray):
            if source.size == 0 or source.ndim < 2:
                raise ValueError("Invalid numpy array shape for screenshot.")
            
            if source.ndim == 2:  # Grayscale
                bgr = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
                rgb = cv2.cvtColor(source, cv2.COLOR_GRAY2RGB)
            elif source.shape[2] == 4:  # RGBA / BGRA
                # Assume RGBA
                rgb = cv2.cvtColor(source, cv2.COLOR_RGBA2RGB)
                bgr = cv2.cvtColor(source, cv2.COLOR_RGBA2BGR)
            elif source.shape[2] == 3:
                # Treat incoming array as RGB by default
                rgb = source.copy()
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            else:
                raise ValueError(f"Unsupported number of image channels: {source.shape[2]}")
            source_type = "numpy_array"

        elif isinstance(source, Image.Image):
            pil_rgb = source.convert("RGB")
            rgb = np.array(pil_rgb)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            source_type = "pil_image"

        else:
            raise ValueError(f"Unsupported screenshot source type: {type(source).__name__}")

        h, w = rgb.shape[:2]
        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid image dimensions: {w}x{h}")

        logger.debug("Loaded screenshot [%s]: %dx%d px", source_type, w, h)
        return ImageContainer(
            rgb=rgb,
            bgr=bgr,
            width=w,
            height=h,
            source_type=source_type,
        )


class ImagePreprocessor:
    """
    Optimizes screenshots for weak systems (CPU, low RAM):
    - Downscales large screenshots (e.g. 4K, 1440p) to max_dim for fast inference.
    - Provides coordinate mapping back to original resolution.
    - Provides image enhancement for OCR text detection.
    """

    def __init__(self, max_dim: int = 1280, resize_for_detection: bool = True):
        self.max_dim = max_dim
        self.resize_for_detection = resize_for_detection

    def preprocess_for_detection(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, float, float]:
        """
        Downscale image if larger than max_dim to accelerate YOLO / CV detection on CPU.

        Returns:
            Tuple of (processed_image, scale_x, scale_y)
            where scale_x and scale_y are multipliers to convert detection coordinates
            back to original image coordinates.
        """
        orig_h, orig_w = image.shape[:2]
        if not self.resize_for_detection or max(orig_w, orig_h) <= self.max_dim:
            return image, 1.0, 1.0

        # Preserve aspect ratio
        if orig_w >= orig_h:
            new_w = self.max_dim
            new_h = max(1, int(round(orig_h * (self.max_dim / orig_w))))
        else:
            new_h = self.max_dim
            new_w = max(1, int(round(orig_w * (self.max_dim / orig_h))))

        # INTER_AREA is optimal for downsampling UI screens with crisp text & borders
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        scale_x = orig_w / float(new_w)
        scale_y = orig_h / float(new_h)

        logger.debug(
            "Resized screenshot for detection: %dx%d -> %dx%d (scales: sx=%.3f, sy=%.3f)",
            orig_w, orig_h, new_w, new_h, scale_x, scale_y,
        )
        return resized, scale_x, scale_y

    @staticmethod
    def scale_bbox(
        bbox: List[int],
        scale_x: float,
        scale_y: float,
        max_w: int,
        max_h: int,
    ) -> List[int]:
        """
        Scale bounding box [x1, y1, x2, y2] from detection resolution back to original.
        Clamps coordinates to [0, max_w] and [0, max_h].
        """
        if scale_x == 1.0 and scale_y == 1.0:
            x1, y1, x2, y2 = bbox
        else:
            x1 = int(round(bbox[0] * scale_x))
            y1 = int(round(bbox[1] * scale_y))
            x2 = int(round(bbox[2] * scale_x))
            y2 = int(round(bbox[3] * scale_y))

        # Clamp and validate
        x1 = max(0, min(x1, max_w - 1))
        y1 = max(0, min(y1, max_h - 1))
        x2 = max(x1 + 1, min(x2, max_w))
        y2 = max(y1 + 1, min(y2, max_h))
        return [x1, y1, x2, y2]

    @staticmethod
    def scale_center(
        center: List[int],
        scale_x: float,
        scale_y: float,
        max_w: int,
        max_h: int,
    ) -> List[int]:
        """Scale center point [cx, cy] back to original screen resolution."""
        cx = int(round(center[0] * scale_x))
        cy = int(round(center[1] * scale_y))
        return [max(0, min(cx, max_w - 1)), max(0, min(cy, max_h - 1))]

    @staticmethod
    def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for text extraction:
        - Grayscale conversion
        - Gentle contrast enhancement for low-contrast UI labels
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        return gray
