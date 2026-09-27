"""
Main pipeline entrypoint for Screen Analysis Engine.
Orchestrates:
Screenshot Input -> Image Preprocessing -> UI Element Detection -> OCR -> Fusion -> Standard JSON Output.
Reports execution timing for each stage, optimized for CPU-only systems.
"""

from pathlib import Path
from typing import Union, Dict, Any, Optional, List
import argparse
import json
import logging
import time
import sys
import cv2
import numpy as np
from PIL import Image

# Ensure local imports work when run as script or package
current_dir = Path(__file__).resolve().parent
if str(current_dir.parent) not in sys.path:
    sys.path.insert(0, str(current_dir.parent))

from screen_analyzer.config import ScreenAnalyzerConfig
from screen_analyzer.processing.image_processor import (
    ScreenshotLoader,
    ImagePreprocessor,
    ImageContainer,
)
from screen_analyzer.detector.yolo_detector import DetectorFactory, BaseUIDetector
from screen_analyzer.detector.element_classifier import ElementClassifier
from screen_analyzer.ocr.ocr_engine import OCREngineFactory, BaseOCREngine
from screen_analyzer.processing.element_fusion import ElementFusion

logger = logging.getLogger("ScreenAnalyzer")


class ScreenAnalysisEngine:
    """
    Screen Analysis Engine for Computer Use Agents.
    Extracts visible UI elements and text from screenshots into standard JSON format.
    """

    def __init__(self, config: Optional[ScreenAnalyzerConfig] = None):
        self.config = config or ScreenAnalyzerConfig()
        self._setup_logging()

        # Lazy loaded components
        self._preprocessor: Optional[ImagePreprocessor] = None
        self._detector: Optional[BaseUIDetector] = None
        self._ocr_engine: Optional[BaseOCREngine] = None
        self._classifier: Optional[ElementClassifier] = None
        self._fusion: Optional[ElementFusion] = None

        self.last_metrics: Dict[str, float] = {}

    def _setup_logging(self) -> None:
        """Configure logging level and format."""
        level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        logger.setLevel(level)

    # -------------------------------------------------------------
    # Lazy Load Properties
    # -------------------------------------------------------------
    @property
    def preprocessor(self) -> ImagePreprocessor:
        if self._preprocessor is None:
            self._preprocessor = ImagePreprocessor(
                max_dim=self.config.max_image_dim,
                resize_for_detection=self.config.resize_for_detection,
            )
        return self._preprocessor

    @property
    def detector(self) -> BaseUIDetector:
        if self._detector is None:
            logger.debug("Lazy loading UI detector (%s)...", self.config.detector_type)
            self._detector = DetectorFactory.create_detector(self.config)
        return self._detector

    @property
    def ocr_engine(self) -> BaseOCREngine:
        if self._ocr_engine is None:
            logger.debug("Lazy loading OCR engine (%s)...", self.config.ocr_engine)
            self._ocr_engine = OCREngineFactory.create_engine(self.config)
        return self._ocr_engine

    @property
    def classifier(self) -> ElementClassifier:
        if self._classifier is None:
            self._classifier = ElementClassifier()
        return self._classifier

    @property
    def fusion(self) -> ElementFusion:
        if self._fusion is None:
            self._fusion = ElementFusion(
                containment_threshold=self.config.fusion_containment_threshold,
                include_orphan_texts=self.config.fusion_include_orphan_texts,
                orphan_text_type=self.config.fusion_orphan_text_type,
            )
        return self._fusion

    def load_models(self) -> None:
        """Pre-warm and explicitly load all detection and OCR models."""
        logger.info("Pre-warming detection and OCR models...")
        self.detector.load_model()
        self.ocr_engine.load_model()
        logger.info("All models loaded successfully.")

    # -------------------------------------------------------------
    # Analysis Pipeline
    # -------------------------------------------------------------
    def analyze(
        self,
        source: Union[str, Path, bytes, np.ndarray, Image.Image],
    ) -> Dict[str, Any]:
        """
        Execute full Screen Analysis pipeline on a screenshot.

        Pipeline Stages:
        1. Screenshot Input
        2. Image Preprocessing (resizing if needed for weak CPU)
        3. UI Element Detection (YOLO / CV / Hybrid)
        4. OCR Text Detection (PaddleOCR / Heuristic)
        5. Bounding Box Scaling (rescale to original resolution)
        6. Element Classification & Enrichment
        7. Element Fusion (merge text into UI containers)
        8. Standard JSON Output

        Returns:
            Standard JSON dictionary:
            {
                "screen": {"width": int, "height": int},
                "elements": [...]
            }
        """
        t_total_start = time.perf_counter()

        # Step 1: Screenshot Input
        t0 = time.perf_counter()
        img_container: ImageContainer = ScreenshotLoader.load(source)
        t_load = (time.perf_counter() - t0) * 1000.0
        orig_w, orig_h = img_container.width, img_container.height

        # Step 2: Image Preprocessing (Optimization for weak systems)
        t0 = time.perf_counter()
        processed_img, scale_x, scale_y = self.preprocessor.preprocess_for_detection(
            img_container.rgb
        )
        t_preprocess = (time.perf_counter() - t0) * 1000.0

        # Step 3: UI Element Detection
        t0 = time.perf_counter()
        raw_elements = self.detector.detect(processed_img)
        t_detection = (time.perf_counter() - t0) * 1000.0

        # Step 4: OCR Text Detection
        t0 = time.perf_counter()
        raw_texts = self.ocr_engine.detect_text(processed_img)
        t_ocr = (time.perf_counter() - t0) * 1000.0

        # Step 5: Bounding Box Coordinate Rescaling
        t0 = time.perf_counter()
        scaled_elements = []
        for elem in raw_elements:
            scaled_bbox = self.preprocessor.scale_bbox(
                elem["bbox"], scale_x, scale_y, orig_w, orig_h
            )
            scaled_center = self.preprocessor.scale_center(
                elem["center"], scale_x, scale_y, orig_w, orig_h
            )
            scaled_elem = dict(elem)
            scaled_elem["bbox"] = scaled_bbox
            scaled_elem["center"] = scaled_center
            # Step 6: Element Classification & Enrichment
            classified_elem = self.classifier.classify(scaled_elem, orig_w, orig_h)
            scaled_elements.append(classified_elem)

        scaled_texts = []
        for txt in raw_texts:
            s_tbox = self.preprocessor.scale_bbox(
                txt["bbox"], scale_x, scale_y, orig_w, orig_h
            )
            s_tcenter = self.preprocessor.scale_center(
                txt["center"], scale_x, scale_y, orig_w, orig_h
            )
            scaled_txt = dict(txt)
            scaled_txt["bbox"] = s_tbox
            scaled_txt["center"] = s_tcenter
            scaled_texts.append(scaled_txt)

        # Step 7: Element Fusion
        fused_elements = self.fusion.fuse(
            detected_elements=scaled_elements,
            detected_texts=scaled_texts,
            screen_width=orig_w,
            screen_height=orig_h,
        )

        # Refine classification and clickability on fused elements
        refined_elements = [
            self.classifier.classify(elem, orig_w, orig_h) for elem in fused_elements
        ]
        t_fusion = (time.perf_counter() - t0) * 1000.0

        t_total = (time.perf_counter() - t_total_start) * 1000.0

        # Record and report execution metrics
        self.last_metrics = {
            "load_time_ms": round(t_load, 2),
            "preprocess_time_ms": round(t_preprocess, 2),
            "detection_time_ms": round(t_detection, 2),
            "ocr_time_ms": round(t_ocr, 2),
            "fusion_time_ms": round(t_fusion, 2),
            "total_time_ms": round(t_total, 2),
            "elements_detected": len(refined_elements),
        }

        logger.info(
            "Screen Analysis Completed: %dx%d px | %d elements extracted | Total Time: %.1f ms "
            "(Load: %.1fms, Prep: %.1fms, Det: %.1fms, OCR: %.1fms, Fusion: %.1fms)",
            orig_w, orig_h, len(refined_elements), t_total,
            t_load, t_preprocess, t_detection, t_ocr, t_fusion,
        )

        # Step 8: Standard JSON Output
        return {
            "screen": {
                "width": orig_w,
                "height": orig_h,
            },
            "elements": refined_elements,
        }

    def visualize(
        self,
        image_source: Union[str, Path, bytes, np.ndarray, Image.Image],
        elements: List[Dict[str, Any]],
        output_path: str | Path,
    ) -> Path:
        """
        Draw visual annotations (bounding boxes, labels, text) on screenshot for inspection.
        """
        container = ScreenshotLoader.load(image_source)
        annotated = container.bgr.copy()

        # Color palette for each UI element type (BGR format)
        color_palette = {
            "button": (245, 130, 48),       # Blue/Orange
            "input field": (60, 180, 75),   # Green
            "checkbox": (230, 25, 75),      # Red
            "icon": (145, 30, 180),         # Purple
            "menu": (70, 240, 240),         # Cyan
            "image": (240, 50, 230),        # Magenta
            "link": (0, 130, 200),          # Deep Blue
            "textbox": (255, 225, 25),      # Yellow
        }

        for elem in elements:
            x1, y1, x2, y2 = elem["bbox"]
            elem_type = elem.get("type", "textbox")
            color = color_palette.get(elem_type, (200, 200, 200))

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Center point
            cx, cy = elem.get("center", [int((x1+x2)/2), int((y1+y2)/2)])
            cv2.circle(annotated, (cx, cy), 3, (0, 0, 255), -1)

            # Label tag
            label_text = f"#{elem.get('id', '')} {elem_type}"
            if "text" in elem:
                label_text += f": {elem['text'][:15]}"
            elif "description" in elem:
                label_text += f": {elem['description']}"

            # Label background box
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1
            (tw, th), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
            label_y1 = max(0, y1 - th - 6)
            cv2.rectangle(
                annotated,
                (x1, label_y1),
                (x1 + tw + 6, label_y1 + th + 6),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label_text,
                (x1 + 3, label_y1 + th + 2),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA,
            )

        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_p), annotated)
        logger.info("Saved visual annotation overlay to: %s", out_p.resolve())
        return out_p


def generate_sample_screenshot(output_path: str | Path) -> Path:
    """
    Generates a realistic desktop application screenshot containing:
    - Menu bar ('File', 'Edit', 'View', 'Help')
    - Search input field with search icon
    - Form with username and email input fields
    - Checkboxes ('Enable notifications', 'Auto-start on boot')
    - Buttons ('Save Changes', 'Cancel')
    - Action icon ('Settings' gear icon)
    """
    width, height = 1280, 720
    img = Image.new("RGB", (width, height), color=(243, 244, 246))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)

    # 1. Window Top Navigation / Menu Bar
    draw.rectangle([0, 0, width, 50], fill=(30, 41, 59))
    draw.text((25, 16), "Screen Analysis Studio", fill=(255, 255, 255))
    draw.text((240, 16), "File", fill=(203, 213, 225))
    draw.text((300, 16), "Edit", fill=(203, 213, 225))
    draw.text((360, 16), "View", fill=(203, 213, 225))
    draw.text((420, 16), "Help", fill=(203, 213, 225))

    # Top-right Settings Icon button
    draw.rectangle([width - 120, 10, width - 40, 40], fill=(51, 65, 85), outline=(71, 85, 105))
    draw.text((width - 105, 16), "Settings", fill=(255, 255, 255))

    # 2. Main Content Card
    draw.rectangle([60, 80, width - 60, height - 60], fill=(255, 255, 255), outline=(226, 232, 240), width=2)

    # Card Title
    draw.text((100, 110), "Account Settings & Preferences", fill=(15, 23, 42))

    # Search Bar (Icon + Input Field)
    draw.rectangle([100, 150, 450, 190], fill=(248, 250, 252), outline=(203, 213, 225), width=2)
    # Search icon simulation
    draw.ellipse([115, 162, 127, 174], outline=(100, 116, 139), width=2)
    draw.line([125, 172, 131, 178], fill=(100, 116, 139), width=2)
    draw.text((140, 162), "Search settings...", fill=(148, 163, 184))

    # Input Field 1: Username
    draw.text((100, 220), "Username", fill=(51, 65, 85))
    draw.rectangle([100, 245, 550, 290], fill=(255, 255, 255), outline=(148, 163, 184), width=2)
    draw.text((115, 257), "agent_user_42", fill=(30, 41, 59))

    # Input Field 2: Email
    draw.text((100, 315), "Email Address", fill=(51, 65, 85))
    draw.rectangle([100, 340, 550, 385], fill=(255, 255, 255), outline=(148, 163, 184), width=2)
    draw.text((115, 352), "agent@arena.ai", fill=(30, 41, 59))

    # Checkbox 1: Enable notifications (checked)
    draw.rectangle([100, 415, 122, 437], fill=(59, 130, 246), outline=(37, 99, 235), width=2)
    draw.line([104, 426, 109, 432], fill=(255, 255, 255), width=2)
    draw.line([109, 432, 118, 420], fill=(255, 255, 255), width=2)
    draw.text((135, 418), "Enable notifications", fill=(30, 41, 59))

    # Checkbox 2: Auto-start on boot (unchecked)
    draw.rectangle([100, 460, 122, 482], fill=(255, 255, 255), outline=(148, 163, 184), width=2)
    draw.text((135, 463), "Auto-start on boot", fill=(30, 41, 59))

    # Profile Image Box
    draw.rectangle([650, 150, 950, 385], fill=(241, 245, 249), outline=(203, 213, 225), width=2)
    draw.text((750, 260), "[ Avatar Image ]", fill=(100, 116, 139))

    # Buttons
    # Button 1: Save Changes (Primary Blue Button)
    draw.rectangle([100, 530, 260, 580], fill=(37, 99, 235), outline=(29, 78, 216), width=2)
    draw.text((130, 545), "Save Changes", fill=(255, 255, 255))

    # Button 2: Cancel (Secondary Gray Button)
    draw.rectangle([280, 530, 400, 580], fill=(241, 245, 249), outline=(203, 213, 225), width=2)
    draw.text((320, 545), "Cancel", fill=(51, 65, 85))

    # Link: Learn more
    draw.text((430, 548), "Learn more about permissions", fill=(37, 99, 235))
    draw.line([430, 565, 620, 565], fill=(37, 99, 235), width=1)

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_p)
    logger.info("Generated realistic sample UI screenshot at: %s", out_p.resolve())
    return out_p


def main():
    """Command-line interface for the Screen Analysis Engine."""
    parser = argparse.ArgumentParser(
        description="Screen Analysis Engine for Computer Use Agents"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input screenshot (PNG, JPG, BMP). If omitted, a sample screenshot is generated.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save output JSON. Defaults to screen_analyzer/output/example.json",
    )
    parser.add_argument(
        "--detector",
        type=str,
        choices=["yolo", "cv", "hybrid", "omniparser"],
        default="hybrid",
        help="UI Element Detector backend (default: hybrid)",
    )
    parser.add_argument(
        "--ocr",
        type=str,
        choices=["paddleocr", "heuristic", "auto"],
        default="auto",
        help="OCR Engine backend (default: auto)",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=1280,
        help="Max screenshot dimension for detection resize (default: 1280 for weak CPU)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save annotated image overlay with bounding boxes and labels",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run on generated demo screenshot and display results",
    )

    args = parser.parse_args()

    # Determine paths
    default_output_dir = current_dir / "output"
    default_output_dir.mkdir(parents=True, exist_ok=True)

    if args.image is None or args.demo:
        demo_image_path = default_output_dir / "sample_screen.png"
        generate_sample_screenshot(demo_image_path)
        input_image = demo_image_path
    else:
        input_image = Path(args.image)

    output_json_path = (
        Path(args.output) if args.output else default_output_dir / "example.json"
    )

    # Initialize Engine with CLI options
    config = ScreenAnalyzerConfig(
        detector_type=args.detector,
        ocr_engine=args.ocr,
        max_image_dim=args.max_dim,
        output_dir=str(default_output_dir),
    )
    engine = ScreenAnalysisEngine(config)

    print(f"\n=======================================================")
    print(f"🚀 Screen Analysis Engine (Computer Use Agent UI Extractor)")
    print(f"=======================================================")
    print(f"Screenshot : {input_image}")
    print(f"Detector   : {config.detector_type}")
    print(f"OCR Engine : {config.ocr_engine}")
    print(f"Max Res    : {config.max_image_dim}px (Weak CPU Optimized)")
    print(f"Device     : {config.device.upper()}\n")

    # Run Analysis
    result = engine.analyze(input_image)

    # Save JSON Output
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=config.indent_json)
    print(f"✅ JSON Output saved to: {output_json_path.resolve()}")

    # Visualize if requested
    if args.visualize or True:  # Default to saving visual preview for easy verification
        vis_path = default_output_dir / "annotated_screen.png"
        engine.visualize(input_image, result["elements"], vis_path)
        print(f"🎨 Visual overlay saved to: {vis_path.resolve()}")

    # Display Metrics and Sample Output
    metrics = engine.last_metrics
    print("\n⏱️  Execution Timing (ms):")
    print(f"   - Screenshot Load  : {metrics.get('load_time_ms', 0):.1f} ms")
    print(f"   - Preprocessing    : {metrics.get('preprocess_time_ms', 0):.1f} ms")
    print(f"   - UI Detection     : {metrics.get('detection_time_ms', 0):.1f} ms")
    print(f"   - OCR Detection    : {metrics.get('ocr_time_ms', 0):.1f} ms")
    print(f"   - Element Fusion   : {metrics.get('fusion_time_ms', 0):.1f} ms")
    print(f"   ----------------------------------------")
    print(f"   ⚡ Total Time      : {metrics.get('total_time_ms', 0):.1f} ms")
    print(f"   📦 Elements Found  : {metrics.get('elements_detected', 0)}\n")

    print("📄 Structured JSON Preview (first 3 elements):")
    sample_preview = {
        "screen": result["screen"],
        "elements": result["elements"][:3],
    }
    print(json.dumps(sample_preview, indent=2))
    print(f"\n(... {len(result['elements'])} total UI elements extracted ...)\n")


if __name__ == "__main__":
    main()
