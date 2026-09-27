"""
OCR Text Detection subpackage for Screen Analysis Engine.
"""

from .ocr_engine import (
    BaseOCREngine,
    PaddleOCREngine,
    HeuristicOCREngine,
    OCREngineFactory,
)

__all__ = [
    "BaseOCREngine",
    "PaddleOCREngine",
    "HeuristicOCREngine",
    "OCREngineFactory",
]
