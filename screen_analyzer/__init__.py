"""
ScreenAnalyzer: Standard Screen Analysis Engine for Computer Use Agents.
Extracts visible UI elements, text, and bounding boxes into structured JSON.
"""

from .config import ScreenAnalyzerConfig
from .main import ScreenAnalysisEngine

__all__ = [
    "ScreenAnalyzerConfig",
    "ScreenAnalysisEngine",
]
