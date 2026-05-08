from __future__ import annotations

import unittest

from PIL import Image, ImageDraw

from edgevision.config import EdgeVisionConfig
from edgevision.pipeline import PillInspectionPipeline
from edgevision.schemas import BBox, Detection


class FakeBottleDetector:
    def detect(self, image: Image.Image) -> list[Detection]:
        return [
            Detection(
                bbox=BBox(20, 20, 90, 110),
                confidence=0.91,
                label="bottle",
                source="fake",
            )
        ]


class GenericObjectFlowTest(unittest.TestCase):
    def test_non_pill_detection_keeps_detector_label_without_pill_identity(self) -> None:
        config = EdgeVisionConfig()
        config.identifier.apply_to_detection_labels = ["pill"]
        config.quality.apply_to_detection_labels = ["pill"]

        pipeline = PillInspectionPipeline(config)
        pipeline.detector = FakeBottleDetector()

        image = Image.new("RGB", (140, 140), (128, 128, 128))
        draw = ImageDraw.Draw(image)
        draw.rectangle((20, 20, 90, 110), fill=(230, 230, 230))
        report = pipeline.inspect_image(image, image_path="generic.jpg")

        self.assertEqual(report.total_count, 1)
        self.assertEqual(report.count_by_detection_label, {"bottle": 1})
        self.assertEqual(report.count_by_type, {"bottle": 1})
        self.assertEqual(report.items[0].identity.label, "bottle")
        self.assertEqual(report.items[0].quality.measurements["quality_applied"], 0.0)


if __name__ == "__main__":
    unittest.main()
