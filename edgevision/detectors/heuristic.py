from __future__ import annotations

from PIL import Image

from edgevision.config import HeuristicDetectorConfig
from edgevision.image_utils import connected_components, foreground_mask_from_border, image_to_array
from edgevision.schemas import BBox, Detection


class HeuristicPillDetector:
    """Dependency-light baseline detector based on foreground components.

    This detector exists to validate the full pipeline contract on simple images.
    Production detection should use the YOLO adapter trained on the `archive/`
    dataset.
    """

    def __init__(self, config: HeuristicDetectorConfig | None = None):
        self.config = config or HeuristicDetectorConfig()

    def detect(self, image: Image.Image) -> list[Detection]:
        array = image_to_array(image)
        height, width = array.shape[:2]
        image_area = height * width
        mask = foreground_mask_from_border(array, threshold=self.config.foreground_threshold)

        detections: list[Detection] = []
        for component in connected_components(mask):
            area_ratio = component.area / image_area
            if area_ratio < self.config.min_area_ratio:
                continue
            if area_ratio > self.config.max_area_ratio:
                continue

            bbox = BBox(component.x1, component.y1, component.x2, component.y2)
            fill_ratio = component.area / max(bbox.area, 1.0)
            confidence = max(0.05, min(0.99, 0.35 + 0.65 * fill_ratio))
            detections.append(
                Detection(
                    bbox=bbox,
                    confidence=confidence,
                    label="pill",
                    source="heuristic",
                )
            )

        return detections

