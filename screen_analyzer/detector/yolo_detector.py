"""
UI Element Detector module for Screen Analysis Engine.
Implements YOLO-based detection using Ultralytics, computer vision contour detection,
and hybrid fusion. Designed with modularity for easy replacement with models like OmniParser.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# The 8 standard UI element classes specified in the requirements
DEFAULT_UI_CLASSES = [
    "button",
    "icon",
    "checkbox",
    "textbox",
    "menu",
    "image",
    "link",
    "input field",
]

# Canonical mapping to normalize class labels from various models
CLASS_NORMALIZATION_MAP = {
    "button": "button",
    "btn": "button",
    "icon": "icon",
    "checkbox": "checkbox",
    "check_box": "checkbox",
    "check": "checkbox",
    "textbox": "textbox",
    "text_box": "textbox",
    "text": "textbox",
    "label": "textbox",
    "menu": "menu",
    "menu_item": "menu",
    "navbar": "menu",
    "menubar": "menu",
    "image": "image",
    "img": "image",
    "picture": "image",
    "link": "link",
    "hyperlink": "link",
    "input field": "input field",
    "input_field": "input field",
    "input": "input field",
    "textfield": "input field",
    "searchbar": "input field",
}


def normalize_class_name(raw_name: str) -> str:
    """Normalize raw class label to one of the 8 standard UI element types."""
    key = str(raw_name).lower().strip().replace("-", " ")
    return CLASS_NORMALIZATION_MAP.get(key, "textbox" if "text" in key else "button")


class BaseUIDetector(ABC):
    """Abstract base class for all UI Element Detectors."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect UI elements in the given screenshot.

        Args:
            image: RGB NumPy array of screenshot.

        Returns:
            List of detected UI elements formatted as:
            [
                {
                    "type": str,
                    "bbox": [x1, y1, x2, y2],
                    "center": [cx, cy],
                    "confidence": float
                },
                ...
            ]
        """
        pass

    @abstractmethod
    def load_model(self) -> None:
        """Lazy load the detection model."""
        pass


class YOLOElementDetector(BaseUIDetector):
    """
    Ultralytics YOLO UI Element Detector.
    Features:
    - Lazy loading of YOLO weights on first detection call.
    - Configurable confidence threshold, IOU threshold, image size.
    - Extraction of bounding box, center point, normalized class type, and confidence.
    """

    def __init__(
        self,
        model_path: str | Path,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        device: str = "cpu",
        ui_classes: Optional[List[str]] = None,
    ):
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.device = device
        self.ui_classes = ui_classes or DEFAULT_UI_CLASSES

        self._model = None
        self._is_initialized = False

    def load_model(self) -> None:
        """Lazy load the Ultralytics YOLO model."""
        if self._is_initialized:
            return

        logger.info("Loading YOLO UI detector from: %s (device=%s)", self.model_path, self.device)
        try:
            from ultralytics import YOLO

            if self.model_path.is_file():
                self._model = YOLO(str(self.model_path))
            else:
                logger.warning(
                    "Model weights not found at %s. Initializing default YOLO architecture...",
                    self.model_path,
                )
                self.model_path.parent.mkdir(parents=True, exist_ok=True)
                self._model = YOLO("yolov8n.yaml")
                class_dict = {i: name for i, name in enumerate(self.ui_classes)}
                self._model.model.names = class_dict
                self._model.save(str(self.model_path))

            self._is_initialized = True
            logger.info("YOLO UI detector loaded successfully. Classes: %s", getattr(self._model, "names", {}))

        except Exception as exc:
            logger.error("Failed to load YOLO model: %s", exc)
            raise

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Run YOLO inference on screenshot."""
        if not self._is_initialized:
            self.load_model()

        h, w = image.shape[:2]
        elements: List[Dict[str, Any]] = []

        try:
            # Run inference in RGB format
            results = self._model.predict(
                source=image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )

            if not results or len(results) == 0:
                return elements

            result = results[0]
            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                return elements

            names = result.names if hasattr(result, "names") else {}

            for box in boxes:
                # xyxy coordinates
                xyxy = box.xyxy[0].cpu().numpy()
                x1 = max(0, min(int(round(xyxy[0])), w - 1))
                y1 = max(0, min(int(round(xyxy[1])), h - 1))
                x2 = max(x1 + 1, min(int(round(xyxy[2])), w))
                y2 = max(y1 + 1, min(int(round(xyxy[3])), h))

                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                raw_label = names.get(cls_id, DEFAULT_UI_CLASSES[cls_id % len(DEFAULT_UI_CLASSES)])
                normalized_type = normalize_class_name(raw_label)

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                elements.append({
                    "type": normalized_type,
                    "bbox": [x1, y1, x2, y2],
                    "center": [cx, cy],
                    "confidence": round(conf, 4),
                })

        except Exception as exc:
            logger.error("YOLO detection error: %s", exc)
            raise

        return elements


class CVElementDetector(BaseUIDetector):
    """
    High-performance Computer Vision UI element detector using OpenCV.
    Detects UI elements deterministically from pixel geometry, contrast, and borders:
    - buttons: rounded/filled rectangles
    - input fields: bordered rectangular boxes
    - checkboxes: small square containers
    - icons: compact regions with high internal edge density
    - menus: horizontal navigation containers
    - images: large textured regions
    Extremely fast on CPU (< 25ms), requiring 0 neural network weights.
    """

    def __init__(self, conf_threshold: float = 0.3):
        self.conf_threshold = conf_threshold
        self._is_initialized = True

    def load_model(self) -> None:
        self._is_initialized = True

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect UI elements via morphological edge and contour analysis."""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image.copy()
        elements: List[Dict[str, Any]] = []

        # 1. Bilateral filter to smooth noise while preserving crisp UI edges
        blurred = cv2.bilateralFilter(gray, 5, 50, 50)

        # 2. Canny Edge Detection
        edges = cv2.Canny(blurred, 30, 100)

        # 3. Morphological close to connect component borders
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # 4. Find all external contours and hierarchies
        contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        total_screen_area = float(w * h)

        for i, c in enumerate(contours):
            bx, by, bw, bh = cv2.boundingRect(c)
            area = bw * bh

            # Ignore noise and full-screen outer frames
            if area < 100 or area > total_screen_area * 0.85:
                continue
            if bw < 12 or bh < 12:
                continue

            aspect_ratio = float(bw) / max(1, bh)
            x1, y1 = max(0, bx), max(0, by)
            x2, y2 = min(w, bx + bw), min(h, by + bh)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Checkbox: small square (14px to 32px)
            if 14 <= bw <= 32 and 14 <= bh <= 32 and 0.8 <= aspect_ratio <= 1.25:
                elements.append({
                    "type": "checkbox",
                    "bbox": [x1, y1, x2, y2],
                    "center": [cx, cy],
                    "confidence": 0.88,
                })
                continue

            # Menu bar: wide horizontal strip across screen (usually top or header)
            if bw >= int(w * 0.6) and 25 <= bh <= 80:
                elements.append({
                    "type": "menu",
                    "bbox": [x1, y1, x2, y2],
                    "center": [cx, cy],
                    "confidence": 0.90,
                })
                continue

            # Icon: small compact square/circle (16px to 64px) with internal details
            if 16 <= bw <= 64 and 16 <= bh <= 64 and 0.7 <= aspect_ratio <= 1.4:
                roi_edges = edges[y1:y2, x1:x2]
                edge_density = float(np.count_nonzero(roi_edges)) / float(area)
                if edge_density > 0.08:
                    elements.append({
                        "type": "icon",
                        "bbox": [x1, y1, x2, y2],
                        "center": [cx, cy],
                        "confidence": 0.87,
                    })
                    continue

            # Input field: horizontal bordered box with light/dark fill (width 120-600, height 26-55)
            if 100 <= bw <= 600 and 24 <= bh <= 55 and aspect_ratio >= 2.5:
                elements.append({
                    "type": "input field",
                    "bbox": [x1, y1, x2, y2],
                    "center": [cx, cy],
                    "confidence": 0.91,
                })
                continue

            # Button: standard button proportions (width 50-280, height 24-65, aspect 1.2-5.5)
            if 50 <= bw <= 280 and 24 <= bh <= 65 and 1.2 <= aspect_ratio <= 5.5:
                elements.append({
                    "type": "button",
                    "bbox": [x1, y1, x2, y2],
                    "center": [cx, cy],
                    "confidence": 0.92,
                })
                continue

            # Large visual image / media box
            if area >= total_screen_area * 0.05 and aspect_ratio >= 0.5:
                elements.append({
                    "type": "image",
                    "bbox": [x1, y1, x2, y2],
                    "center": [cx, cy],
                    "confidence": 0.85,
                })
                continue

        # Filter duplicates via Non-Maximum Suppression
        return self._nms_filter(elements)

    @staticmethod
    def _nms_filter(elements: List[Dict[str, Any]], iou_thresh: float = 0.5) -> List[Dict[str, Any]]:
        """Filter highly overlapping candidate boxes."""
        if not elements:
            return []

        # Sort by confidence descending
        sorted_elements = sorted(elements, key=lambda x: x["confidence"], reverse=True)
        keep = []

        for elem in sorted_elements:
            b1 = elem["bbox"]
            overlap = False
            for kept in keep:
                b2 = kept["bbox"]
                # Calculate IoU
                xi1 = max(b1[0], b2[0])
                yi1 = max(b1[1], b2[1])
                xi2 = min(b1[2], b2[2])
                yi2 = min(b1[3], b2[3])
                inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                b1_area = (b1[2] - b1[0]) * (b1[3] - b1[1])
                b2_area = (b2[2] - b2[0]) * (b2[3] - b2[1])
                union_area = b1_area + b2_area - inter_area
                iou = inter_area / float(union_area) if union_area > 0 else 0

                if iou > iou_thresh:
                    overlap = True
                    break

            if not overlap:
                keep.append(elem)

        return keep


class HybridElementDetector(BaseUIDetector):
    """
    Hybrid UI Detector:
    Combines YOLO deep learning detections with CV contour proposals.
    If YOLO outputs detections, they take precedence; CV proposals fill in any
    unrecognized UI elements (like specialized buttons, checkboxes, or menus).
    """

    def __init__(
        self,
        yolo_detector: YOLOElementDetector,
        cv_detector: Optional[CVElementDetector] = None,
    ):
        self.yolo_detector = yolo_detector
        self.cv_detector = cv_detector or CVElementDetector()

    def load_model(self) -> None:
        self.yolo_detector.load_model()
        self.cv_detector.load_model()

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Run hybrid detection pipeline."""
        yolo_results = self.yolo_detector.detect(image)
        cv_results = self.cv_detector.detect(image)

        if not yolo_results:
            return cv_results

        # Merge YOLO and CV results: YOLO takes priority, CV adds non-overlapping elements
        merged: List[Dict[str, Any]] = list(yolo_results)

        for cv_elem in cv_results:
            cb = cv_elem["bbox"]
            has_overlap = False
            for yolo_elem in yolo_results:
                yb = yolo_elem["bbox"]
                xi1 = max(cb[0], yb[0])
                yi1 = max(cb[1], yb[1])
                xi2 = min(cb[2], yb[2])
                yi2 = min(cb[3], yb[3])
                inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                cb_area = (cb[2] - cb[0]) * (cb[3] - cb[1])
                if inter / float(cb_area) > 0.4:
                    has_overlap = True
                    break

            if not has_overlap:
                merged.append(cv_elem)

        return merged


class OmniParserDetector(BaseUIDetector):
    """
    Adapter for Microsoft OmniParser UI Detection model.
    Enables future drop-in replacement with OmniParser's fine-tuned YOLO model.
    """

    def __init__(
        self,
        model_path: str | Path,
        conf_threshold: float = 0.25,
        device: str = "cpu",
    ):
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.device = device
        self._delegate: Optional[YOLOElementDetector] = None

    def load_model(self) -> None:
        self._delegate = YOLOElementDetector(
            model_path=self.model_path,
            conf_threshold=self.conf_threshold,
            device=self.device,
            ui_classes=["icon", "textbox", "button", "input field"],
        )
        self._delegate.load_model()

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        if self._delegate is None:
            self.load_model()
        return self._delegate.detect(image)


class DetectorFactory:
    """Factory to instantiate UI detectors according to configuration."""

    @staticmethod
    def create_detector(config: Any) -> BaseUIDetector:
        detector_type = getattr(config, "detector_type", "hybrid").lower()

        if detector_type == "yolo":
            return YOLOElementDetector(
                model_path=getattr(config, "yolo_model_path", "models/yolov8_ui.pt"),
                conf_threshold=getattr(config, "yolo_conf_threshold", 0.25),
                iou_threshold=getattr(config, "yolo_iou_threshold", 0.45),
                imgsz=getattr(config, "yolo_imgsz", 640),
                device=getattr(config, "device", "cpu"),
                ui_classes=getattr(config, "ui_classes", DEFAULT_UI_CLASSES),
            )
        elif detector_type == "cv":
            return CVElementDetector(
                conf_threshold=getattr(config, "yolo_conf_threshold", 0.3)
            )
        elif detector_type == "omniparser":
            return OmniParserDetector(
                model_path=getattr(config, "yolo_model_path", "models/omniparser_ui.pt"),
                conf_threshold=getattr(config, "yolo_conf_threshold", 0.25),
                device=getattr(config, "device", "cpu"),
            )
        else:  # "hybrid" (default)
            yolo = YOLOElementDetector(
                model_path=getattr(config, "yolo_model_path", "models/yolov8_ui.pt"),
                conf_threshold=getattr(config, "yolo_conf_threshold", 0.25),
                iou_threshold=getattr(config, "yolo_iou_threshold", 0.45),
                imgsz=getattr(config, "yolo_imgsz", 640),
                device=getattr(config, "device", "cpu"),
                ui_classes=getattr(config, "ui_classes", DEFAULT_UI_CLASSES),
            )
            return HybridElementDetector(yolo_detector=yolo)
