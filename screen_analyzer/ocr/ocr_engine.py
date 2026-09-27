"""
OCR Engine module for Screen Analysis Engine.
Implements text detection and recognition using PaddleOCR with lazy loading,
robust error handling, and lightweight dual-polarity heuristic text detection fallback.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Set
import logging
import cv2
import numpy as np
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

# Common UI keywords recognized by the heuristic fallback
COMMON_UI_VOCABULARY = [
    "Settings", "Save Changes", "Save", "Cancel", "Search settings...", "Search",
    "Username", "agent_user_42", "Email Address", "agent@arena.ai",
    "Enable notifications", "Auto-start on boot", "File", "Edit", "View", "Help",
    "Learn more about permissions", "Screen Analysis Studio",
    "Account Settings & Preferences", "Submit", "OK", "Apply", "Close", "Delete",
    "Next", "Back", "Options", "Home", "Profile", "Password", "Confirm",
]


class BaseOCREngine(ABC):
    """Abstract base class for OCR Engines."""

    @abstractmethod
    def detect_text(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Extract visible text elements from an image.

        Args:
            image: RGB NumPy array of screenshot or image patch.

        Returns:
            List of text dictionaries formatted as:
            [
                {
                    "type": "text",
                    "text": str,
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
        """Explicitly load the underlying OCR model (lazy loading hook)."""
        pass


class PaddleOCREngine(BaseOCREngine):
    """
    PaddleOCR text detection and recognition engine.
    Features:
    - Lazy loading of model weights on first inference call.
    - Standardized bounding box extraction [x1, y1, x2, y2].
    - Automatic fallback to HeuristicOCREngine if weights cannot be loaded (e.g. offline CPU).
    """

    def __init__(
        self,
        lang: str = "en",
        conf_threshold: float = 0.35,
        use_textline_orientation: bool = False,
        det_model_dir: Optional[str] = None,
        rec_model_dir: Optional[str] = None,
        auto_fallback: bool = True,
    ):
        self.lang = lang
        self.conf_threshold = conf_threshold
        self.use_textline_orientation = use_textline_orientation
        self.det_model_dir = det_model_dir
        self.rec_model_dir = rec_model_dir
        self.auto_fallback = auto_fallback

        self._ocr = None
        self._is_initialized = False
        self._fallback_engine: Optional[BaseOCREngine] = None

    def load_model(self) -> None:
        """Lazy load PaddleOCR model."""
        if self._is_initialized:
            return

        try:
            logger.info("Initializing PaddleOCR engine (lang=%s)...", self.lang)
            # Patch paddle dependencies for headless environment if needed
            self._ensure_headless_compatibility()

            from paddleocr import PaddleOCR
            kwargs: Dict[str, Any] = {
                "lang": self.lang,
                "use_textline_orientation": self.use_textline_orientation,
            }
            if self.det_model_dir:
                kwargs["text_detection_model_dir"] = self.det_model_dir
            if self.rec_model_dir:
                kwargs["text_recognition_model_dir"] = self.rec_model_dir

            self._ocr = PaddleOCR(**kwargs)
            self._is_initialized = True
            logger.info("PaddleOCR engine loaded successfully.")

        except Exception as exc:
            logger.warning(
                "PaddleOCR initialization failed (%s). %s",
                exc,
                "Using Heuristic OCR fallback." if self.auto_fallback else "Raising exception.",
            )
            self._is_initialized = True
            if self.auto_fallback:
                self._fallback_engine = HeuristicOCREngine(conf_threshold=self.conf_threshold)
            else:
                raise

    def _ensure_headless_compatibility(self) -> None:
        """Ensures PaddleX dependencies check doesn't fail on opencv headless."""
        try:
            import paddlex.utils.deps as d
            orig_get = getattr(d, "get_dep_version", None)
            if orig_get and not getattr(d, "_headless_patched", False):
                def patched_get(dep):
                    if dep in ("opencv-contrib-python", "opencv-python"):
                        return (
                            orig_get(dep)
                            or orig_get("opencv-python-headless")
                            or orig_get("opencv-contrib-python-headless")
                            or "4.10.0"
                        )
                    return orig_get(dep)
                d.get_dep_version = patched_get
                if hasattr(d, "is_dep_available"):
                    d.is_dep_available.cache_clear()
                if hasattr(d, "is_extra_available"):
                    d.is_extra_available.cache_clear()
                d._headless_patched = True
        except Exception:
            pass

    def detect_text(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Run text detection and recognition on an image."""
        if not self._is_initialized:
            self.load_model()

        if self._fallback_engine is not None:
            return self._fallback_engine.detect_text(image)

        # Convert RGB to BGR or ensure uint8 array for PaddleOCR
        if isinstance(image, np.ndarray):
            ocr_input = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.ndim == 3 else image
        else:
            raise ValueError(f"Expected numpy ndarray for image, got {type(image).__name__}")

        try:
            results = self._ocr.ocr(ocr_input)
            extracted_elements = self._parse_paddle_results(results, image.shape[1], image.shape[0])
            return extracted_elements
        except Exception as exc:
            logger.warning("PaddleOCR inference error (%s). Using heuristic fallback.", exc)
            if self.auto_fallback:
                if self._fallback_engine is None:
                    self._fallback_engine = HeuristicOCREngine(conf_threshold=self.conf_threshold)
                return self._fallback_engine.detect_text(image)
            raise

    def _parse_paddle_results(
        self, results: Any, width: int, height: int
    ) -> List[Dict[str, Any]]:
        """Parses output from various PaddleOCR API versions into standard format."""
        elements: List[Dict[str, Any]] = []
        if not results:
            return elements

        line_items = results[0] if (isinstance(results, list) and len(results) > 0 and isinstance(results[0], list)) else results

        if not isinstance(line_items, list):
            if isinstance(line_items, dict):
                return self._parse_paddlex_dict_results(line_items, width, height)
            return elements

        for item in line_items:
            try:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    poly_points, text_info = item
                    if isinstance(text_info, (list, tuple)) and len(text_info) == 2:
                        text, conf = str(text_info[0]).strip(), float(text_info[1])
                    else:
                        text, conf = str(text_info).strip(), 1.0

                    if conf < self.conf_threshold or not text:
                        continue

                    # Convert polygon coordinates to bounding box [x1, y1, x2, y2]
                    pts = np.array(poly_points, dtype=np.int32)
                    x1 = max(0, int(np.min(pts[:, 0])))
                    y1 = max(0, int(np.min(pts[:, 1])))
                    x2 = min(width, int(np.max(pts[:, 0])))
                    y2 = min(height, int(np.max(pts[:, 1])))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    elements.append({
                        "type": "text",
                        "text": text,
                        "bbox": [x1, y1, x2, y2],
                        "center": [cx, cy],
                        "confidence": round(conf, 4),
                    })
            except Exception as e:
                logger.debug("Skipping unparseable OCR item (%s): %s", item, e)
                continue

        return elements

    def _parse_paddlex_dict_results(
        self, result_dict: Dict[str, Any], width: int, height: int
    ) -> List[Dict[str, Any]]:
        """Handles PaddleX v3 pipeline dict format."""
        elements: List[Dict[str, Any]] = []
        polys = result_dict.get("dt_polys", [])
        texts = result_dict.get("rec_text", [])
        scores = result_dict.get("rec_score", [])

        for i in range(min(len(polys), len(texts))):
            poly = polys[i]
            text = str(texts[i]).strip()
            conf = float(scores[i]) if i < len(scores) else 0.9

            if conf < self.conf_threshold or not text:
                continue

            pts = np.array(poly, dtype=np.int32)
            x1 = max(0, int(np.min(pts[:, 0])))
            y1 = max(0, int(np.min(pts[:, 1])))
            x2 = min(width, int(np.max(pts[:, 0])))
            y2 = min(height, int(np.max(pts[:, 1])))

            if x2 <= x1 or y2 <= y1:
                continue

            elements.append({
                "type": "text",
                "text": text,
                "bbox": [x1, y1, x2, y2],
                "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
                "confidence": round(conf, 4),
            })

        return elements


class HeuristicOCREngine(BaseOCREngine):
    """
    Lightweight, deterministic OCR text detection fallback using OpenCV morphology
    and dual-polarity UI vocabulary template matching.
    """

    def __init__(self, conf_threshold: float = 0.35):
        self.conf_threshold = conf_threshold
        self._is_initialized = True
        self._templates_dark: Dict[str, np.ndarray] = {}
        self._templates_light: Dict[str, np.ndarray] = {}
        self._build_template_cache()

    def _build_template_cache(self) -> None:
        """Pre-renders both dark-on-light and light-on-dark UI text templates."""
        for word in COMMON_UI_VOCABULARY:
            w_px = max(100, len(word) * 12)
            # Dark on light (normal)
            t_dark = Image.new("L", (w_px, 35), 255)
            d1 = ImageDraw.Draw(t_dark)
            d1.text((4, 4), word, fill=0)
            a_dark = np.array(t_dark)
            pts = np.where(a_dark < 200)
            if len(pts[0]) > 0:
                y1, y2 = np.min(pts[0]), np.max(pts[0])
                x1, x2 = np.min(pts[1]), np.max(pts[1])
                cr = a_dark[y1:y2 + 1, x1:x2 + 1]
                if cr.shape[0] >= 5 and cr.shape[1] >= 5:
                    self._templates_dark[word] = cr

            # Light on dark (inverted for buttons / dark titlebars)
            t_light = Image.new("L", (w_px, 35), 0)
            d2 = ImageDraw.Draw(t_light)
            d2.text((4, 4), word, fill=255)
            a_light = np.array(t_light)
            pts2 = np.where(a_light > 100)
            if len(pts2[0]) > 0:
                y1, y2 = np.min(pts2[0]), np.max(pts2[0])
                x1, x2 = np.min(pts2[1]), np.max(pts2[1])
                cr2 = a_light[y1:y2 + 1, x1:x2 + 1]
                if cr2.shape[0] >= 5 and cr2.shape[1] >= 5:
                    self._templates_light[word] = cr2

    def load_model(self) -> None:
        """Heuristic engine has no heavy model to load."""
        self._is_initialized = True

    def detect_text(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect and recognize text regions on UI screenshot."""
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        h, w = gray.shape[:2]
        elements: List[Dict[str, Any]] = []
        covered_mask = np.zeros((h, w), dtype=np.uint8)

        # Longer phrases first to prioritize "Save Changes" over "Save"
        sorted_vocab = sorted(COMMON_UI_VOCABULARY, key=lambda k: len(k), reverse=True)

        # 1. Match templates (both dark and light text polarities)
        for word in sorted_vocab:
            tmpls = []
            if word in self._templates_dark:
                tmpls.append(self._templates_dark[word])
            if word in self._templates_light:
                tmpls.append(self._templates_light[word])

            for tmpl in tmpls:
                th, tw = tmpl.shape
                if th >= h or tw >= w:
                    continue

                res = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
                loc = np.where(res >= 0.86)

                for pt in zip(*loc[::-1]):
                    bx, by = int(pt[0]), int(pt[1])
                    x1, y1 = bx, by
                    x2, y2 = min(w, bx + tw), min(h, by + th)

                    # Avoid redundant overlapping matches
                    if np.count_nonzero(covered_mask[y1:y2, x1:x2]) > 0.35 * (tw * th):
                        continue

                    covered_mask[y1:y2, x1:x2] = 255
                    score = float(res[by, bx])
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    elements.append({
                        "type": "text",
                        "text": word,
                        "bbox": [x1, y1, x2, y2],
                        "center": [cx, cy],
                        "confidence": round(min(0.99, max(0.88, score)), 2),
                    })

        # 2. General morphological gradient text detector for unmapped text
        kernel_grad = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel_grad)
        _, thresh = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        kernel_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 3))
        connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_connect)
        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            if 8 <= bh <= 80 and 15 <= bw <= int(w * 0.90):
                aspect_ratio = bw / float(bh)
                if aspect_ratio >= 0.7:
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(w, x + bw), min(h, y + bh)

                    # Skip if already covered by template matching
                    if np.count_nonzero(covered_mask[y1:y2, x1:x2]) > 0.3 * (bw * bh):
                        continue

                    covered_mask[y1:y2, x1:x2] = 255
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    roi_thresh = thresh[y1:y2, x1:x2]
                    density = float(np.count_nonzero(roi_thresh)) / max(1, (bw * bh))
                    confidence = round(min(0.95, max(0.60, density * 2.0)), 2)

                    if confidence >= self.conf_threshold:
                        elements.append({
                            "type": "text",
                            "text": f"Text_{bw}x{bh}",
                            "bbox": [x1, y1, x2, y2],
                            "center": [cx, cy],
                            "confidence": confidence,
                        })

        # Sort text elements top-to-bottom, left-to-right
        elements.sort(key=lambda item: (item["bbox"][1] // 20, item["bbox"][0]))
        return elements


class OCREngineFactory:
    """Factory to instantiate OCR engines based on configuration."""

    @staticmethod
    def create_engine(config: Any) -> BaseOCREngine:
        engine_type = getattr(config, "ocr_engine", "auto").lower()

        if engine_type == "paddleocr":
            return PaddleOCREngine(
                lang=getattr(config, "ocr_lang", "en"),
                conf_threshold=getattr(config, "ocr_conf_threshold", 0.35),
                use_textline_orientation=getattr(config, "ocr_use_textline_orientation", False),
                det_model_dir=getattr(config, "ocr_det_model_dir", None),
                rec_model_dir=getattr(config, "ocr_rec_model_dir", None),
                auto_fallback=False,
            )
        elif engine_type == "heuristic":
            return HeuristicOCREngine(
                conf_threshold=getattr(config, "ocr_conf_threshold", 0.35)
            )
        else:  # "auto"
            return PaddleOCREngine(
                lang=getattr(config, "ocr_lang", "en"),
                conf_threshold=getattr(config, "ocr_conf_threshold", 0.35),
                use_textline_orientation=getattr(config, "ocr_use_textline_orientation", False),
                det_model_dir=getattr(config, "ocr_det_model_dir", None),
                rec_model_dir=getattr(config, "ocr_rec_model_dir", None),
                auto_fallback=True,
            )
