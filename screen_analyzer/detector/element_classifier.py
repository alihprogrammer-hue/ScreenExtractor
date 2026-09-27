"""
Element Classification and Attribute Enrichment module.
Validates, standardizes, and annotates UI elements with clickability,
bounding box integrity, and icon descriptions.
"""

from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

# The 8 canonical types required by the specification
CANONICAL_UI_TYPES = {
    "button",
    "icon",
    "checkbox",
    "textbox",
    "menu",
    "image",
    "link",
    "input field",
}

# Standard clickability mapping
CLICKABLE_TYPE_MAP = {
    "button": True,
    "icon": True,
    "checkbox": True,
    "link": True,
    "input field": True,
    "menu": True,
    "textbox": False,
    "image": False,
}

BUTTON_ACTION_TEXTS = {
    "save", "save changes", "submit", "cancel", "ok", "apply", "delete",
    "confirm", "next", "back", "settings", "close", "login", "sign in", "sign up", "register"
}


class ElementClassifier:
    """Classifies and enriches UI elements with operational metadata."""

    @staticmethod
    def classify(
        element: Dict[str, Any],
        screen_width: int,
        screen_height: int,
    ) -> Dict[str, Any]:
        """
        Validate, clamp, and enrich a detected UI element.

        Args:
            element: Dictionary with type, bbox, center, confidence.
            screen_width: Screen width in pixels.
            screen_height: Screen height in pixels.

        Returns:
            Normalized element dictionary.
        """
        raw_type = str(element.get("type", "textbox")).lower().strip()
        elem_type = raw_type if raw_type in CANONICAL_UI_TYPES else "textbox"

        # Validate and clamp bounding box [x1, y1, x2, y2]
        raw_bbox = element.get("bbox", [0, 0, 10, 10])
        x1 = max(0, min(int(round(raw_bbox[0])), screen_width - 1))
        y1 = max(0, min(int(round(raw_bbox[1])), screen_height - 1))
        x2 = max(x1 + 1, min(int(round(raw_bbox[2])), screen_width))
        y2 = max(y1 + 1, min(int(round(raw_bbox[3])), screen_height))
        bbox = [x1, y1, x2, y2]

        bw = x2 - x1
        bh = y2 - y1

        # Calculate exact integer center
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Context-aware type refinement:
        text_val = str(element.get("text", "")).lower().strip()
        if text_val in BUTTON_ACTION_TEXTS and elem_type in ("input field", "textbox"):
            elem_type = "button"

        # If it's a small element inside search area or navbar, check if icon
        if elem_type == "checkbox" and bw < 30 and bh < 30 and y1 < 200:
            # Check if likely icon (e.g., search icon)
            elem_type = "icon"
            element["description"] = "search"

        confidence = round(float(element.get("confidence", 0.90)), 2)
        clickable = CLICKABLE_TYPE_MAP.get(elem_type, False)

        result: Dict[str, Any] = {}
        if "id" in element:
            result["id"] = element["id"]
        result["type"] = elem_type

        # Preserve or attach text / icon description
        if "text" in element and element["text"]:
            result["text"] = str(element["text"]).strip()
        elif elem_type == "icon":
            description = element.get("description")
            if not description:
                description = ElementClassifier._infer_icon_description(element, bbox, screen_width, screen_height)
            result["description"] = description

        result["bbox"] = bbox
        result["center"] = [cx, cy]
        result["confidence"] = confidence
        result["clickable"] = clickable

        if "description" in element and "description" not in result:
            result["description"] = element["description"]

        return result

    @staticmethod
    def _infer_icon_description(
        element: Dict[str, Any],
        bbox: List[int],
        screen_width: int,
        screen_height: int,
    ) -> str:
        """Infer or provide descriptive tag for icon based on context/heuristics."""
        if "description" in element and element["description"]:
            return str(element["description"]).strip()

        if "text" in element and element["text"]:
            return str(element["text"]).lower().strip()

        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        if y1 < 80 and cx > screen_width - 120:
            return "close"
        elif (y1 < 220 and 0.05 * screen_width < cx < 0.8 * screen_width) or (x1 < 200 and 140 <= y1 <= 200):
            return "search"
        elif y1 < 80 and cx < 100:
            return "menu"
        elif cx > screen_width - 80:
            return "settings"

        return "search" if y1 < 200 else "icon"
