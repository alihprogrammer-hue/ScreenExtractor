"""
Element Fusion module for Screen Analysis Engine.
Fuses OCR text detections with UI element detections into structured,
unified UI elements with hierarchical containment logic and 1-based sequential IDs.
"""

from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class ElementFusion:
    """
    Fuses OCR text detections into containing UI elements (buttons, inputs, menus, etc.)
    and assigns structured, deterministic IDs and attributes.
    """

    def __init__(
        self,
        containment_threshold: float = 0.40,
        include_orphan_texts: bool = True,
        orphan_text_type: str = "textbox",
        row_group_tolerance: int = 25,
    ):
        self.containment_threshold = containment_threshold
        self.include_orphan_texts = include_orphan_texts
        self.orphan_text_type = orphan_text_type
        self.row_group_tolerance = row_group_tolerance

    def fuse(
        self,
        detected_elements: List[Dict[str, Any]],
        detected_texts: List[Dict[str, Any]],
        screen_width: int,
        screen_height: int,
    ) -> List[Dict[str, Any]]:
        """
        Merge UI element detections and OCR text detections.

        Args:
            detected_elements: UI elements from YOLO/CV detector.
            detected_texts: Text elements from OCR engine.
            screen_width: Original screen width.
            screen_height: Original screen height.

        Returns:
            List of fused UI element dictionaries with unique 'id' values.
        """
        # Make working copies
        containers = [dict(elem) for elem in detected_elements]
        texts = [dict(txt) for txt in detected_texts]

        # Track which texts have been absorbed
        consumed_text_indices = set()

        # For each text, find the best matching UI container
        # If multiple containers contain the text, choose the smallest one by area (most specific)
        for t_idx, txt in enumerate(texts):
            tb = txt["bbox"]
            t_area = max(1, (tb[2] - tb[0]) * (tb[3] - tb[1]))
            tcx, tcy = (tb[0] + tb[2]) / 2.0, (tb[1] + tb[3]) / 2.0

            best_c_idx = None
            smallest_area = float("inf")

            for c_idx, container in enumerate(containers):
                cb = container["bbox"]
                c_area = (cb[2] - cb[0]) * (cb[3] - cb[1])

                # Calculate intersection
                xi1 = max(tb[0], cb[0])
                yi1 = max(tb[1], cb[1])
                xi2 = min(tb[2], cb[2])
                yi2 = min(tb[3], cb[3])
                inter_w = max(0, xi2 - xi1)
                inter_h = max(0, yi2 - yi1)
                inter_area = inter_w * inter_h

                containment_ratio = inter_area / float(t_area)
                center_inside = (cb[0] <= tcx <= cb[2]) and (cb[1] <= tcy <= cb[3])

                # Absorbed if area overlap exceeds threshold or center is inside container
                if (containment_ratio >= self.containment_threshold or center_inside) and c_area < smallest_area:
                    smallest_area = c_area
                    best_c_idx = c_idx

            if best_c_idx is not None:
                consumed_text_indices.add(t_idx)
                if "_matched_texts" not in containers[best_c_idx]:
                    containers[best_c_idx]["_matched_texts"] = []
                containers[best_c_idx]["_matched_texts"].append(txt)

        # Merge matched texts into each container
        for container in containers:
            matched = container.pop("_matched_texts", [])
            if matched:
                # Prefer real semantic text over fallback generic 'Text_...' labels
                semantic_texts = [t for t in matched if not str(t.get("text", "")).startswith("Text_")]
                chosen_texts = semantic_texts if semantic_texts else matched

                # Sort matched texts: top-to-bottom, left-to-right
                chosen_texts.sort(key=lambda t: (t["bbox"][1] // 10, t["bbox"][0]))
                merged_text = " ".join(t["text"].strip() for t in chosen_texts if t.get("text"))
                if merged_text:
                    container["text"] = merged_text

                # If container is an icon, set or update description
                if container.get("type") == "icon" and "description" not in container:
                    container["description"] = merged_text.lower()

        # Handle remaining unconsumed OCR texts (orphan texts)
        orphan_elements: List[Dict[str, Any]] = []
        if self.include_orphan_texts:
            for t_idx, txt in enumerate(texts):
                if t_idx not in consumed_text_indices:
                    tb = txt["bbox"]
                    cx = int((tb[0] + tb[2]) / 2)
                    cy = int((tb[1] + tb[3]) / 2)
                    raw_t = txt.get("text", "").strip()
                    # Skip noise text
                    if not raw_t or raw_t.startswith("Text_"):
                        continue

                    orphan_elements.append({
                        "type": self.orphan_text_type,
                        "text": raw_t,
                        "bbox": tb,
                        "center": [cx, cy],
                        "confidence": txt.get("confidence", 0.90),
                        "clickable": False,
                    })

        all_elements = containers + orphan_elements

        # Spatial sorting: group by vertical rows, then sort horizontally left-to-right
        all_elements.sort(
            key=lambda e: (e["bbox"][1] // self.row_group_tolerance, e["bbox"][0])
        )

        # Assign clean, sequential integer IDs starting from 1
        final_elements: List[Dict[str, Any]] = []
        for current_id, elem in enumerate(all_elements, start=1):
            formatted = self._format_element_dict(current_id, elem)
            final_elements.append(formatted)

        return final_elements

    @staticmethod
    def _format_element_dict(elem_id: int, elem: Dict[str, Any]) -> Dict[str, Any]:
        """Format dictionary keys in the exact order requested by user specifications."""
        out: Dict[str, Any] = {
            "id": elem_id,
            "type": elem["type"],
        }

        # Place text or description right after type as in examples
        if "text" in elem and elem["text"]:
            out["text"] = elem["text"]
        elif elem["type"] == "icon" and "description" in elem:
            out["description"] = elem["description"]

        out["bbox"] = elem["bbox"]
        out["center"] = elem["center"]
        out["confidence"] = round(float(elem.get("confidence", 0.90)), 2)

        if "clickable" in elem:
            out["clickable"] = bool(elem["clickable"])

        # Include description if present and not already added
        if "description" in elem and "description" not in out:
            out["description"] = elem["description"]

        return out
