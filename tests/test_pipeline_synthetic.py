from __future__ import annotations

import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from edgevision.config import EdgeVisionConfig
from edgevision.pipeline import PillInspectionPipeline
from edgevision.reporting import save_report_json


class SyntheticPipelineTest(unittest.TestCase):
    def test_detects_and_reports_two_synthetic_pills(self) -> None:
        image = Image.new("RGB", (320, 220), (132, 132, 132))
        draw = ImageDraw.Draw(image)
        draw.ellipse((40, 60, 120, 140), fill=(235, 210, 170))
        draw.rounded_rectangle((190, 70, 280, 130), radius=25, fill=(120, 190, 230))

        config = EdgeVisionConfig()
        config.detector.heuristic.foreground_threshold = 35
        config.detector.heuristic.min_area_ratio = 0.002
        config.identifier.unknown_threshold = 0.99

        tmp = Path("runs") / "tests" / "synthetic_pipeline"
        if tmp.exists():
            for file_path in tmp.rglob("*"):
                if file_path.is_file():
                    file_path.unlink()
            for dir_path in sorted((path for path in tmp.rglob("*") if path.is_dir()), reverse=True):
                dir_path.rmdir()
        tmp.mkdir(parents=True, exist_ok=True)

        with self.subTest(output_dir=str(tmp)):
            pipeline = PillInspectionPipeline(config)
            report = pipeline.inspect_image(image, image_path="synthetic.jpg", output_dir=tmp)
            save_report_json(report, tmp)

            self.assertEqual(report.total_count, 2)
            self.assertTrue((tmp / "annotated.jpg").exists())
            self.assertTrue((tmp / "report.json").exists())
            self.assertEqual(len(list((tmp / "crops").glob("*.jpg"))), 2)


if __name__ == "__main__":
    unittest.main()
