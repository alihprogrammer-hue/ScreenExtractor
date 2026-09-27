"""
Configuration module for the Screen Analysis Engine.
Provides dataclass-based configuration with environment/dict overrides.
Optimized for low-resource environments (CPU only, 8GB RAM).
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import os


@dataclass
class ScreenAnalyzerConfig:
    """Configuration settings for ScreenAnalyzer engine."""

    # -------------------------------------------------------------
    # System & Execution Settings (Optimized for Weak CPU Systems)
    # -------------------------------------------------------------
    device: str = "cpu"
    max_image_dim: int = 1280  # Max dimension (width or height) for detection resize
    resize_for_detection: bool = True  # Resize large screenshots before detection
    log_level: str = "INFO"

    # -------------------------------------------------------------
    # UI Element Detector Settings
    # -------------------------------------------------------------
    # Options: "yolo", "cv", "hybrid", "omniparser"
    detector_type: str = "hybrid"
    yolo_model_path: str = field(
        default_factory=lambda: str(
            Path(__file__).resolve().parent / "models" / "yolov8_ui.pt"
        )
    )
    yolo_conf_threshold: float = 0.25
    yolo_iou_threshold: float = 0.45
    yolo_imgsz: int = 640
    ui_classes: List[str] = field(
        default_factory=lambda: [
            "button",
            "icon",
            "checkbox",
            "textbox",
            "menu",
            "image",
            "link",
            "input field",
        ]
    )

    # -------------------------------------------------------------
    # OCR Engine Settings
    # -------------------------------------------------------------
    # Options: "paddleocr", "heuristic", "auto" (tries PaddleOCR, falls back to heuristic)
    ocr_engine: str = "auto"
    ocr_lang: str = "en"
    ocr_conf_threshold: float = 0.35
    ocr_use_textline_orientation: bool = False
    ocr_det_model_dir: Optional[str] = None
    ocr_rec_model_dir: Optional[str] = None

    # -------------------------------------------------------------
    # Element Fusion Settings
    # -------------------------------------------------------------
    # Minimum ratio of text bounding box area inside container to be absorbed
    fusion_containment_threshold: float = 0.45
    # Whether standalone OCR text (not inside any UI container) becomes an element
    fusion_include_orphan_texts: bool = True
    fusion_orphan_text_type: str = "textbox"

    # -------------------------------------------------------------
    # Output Settings
    # -------------------------------------------------------------
    output_dir: str = field(
        default_factory=lambda: str(Path(__file__).resolve().parent / "output")
    )
    save_annotated_image: bool = True
    indent_json: int = 2

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScreenAnalyzerConfig":
        """Build configuration from dictionary, ignoring unexpected keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_json(cls, json_path: str | Path) -> "ScreenAnalyzerConfig":
        """Load configuration from JSON file."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def to_json(self, json_path: str | Path) -> None:
        """Save configuration to JSON file."""
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=self.indent_json)
