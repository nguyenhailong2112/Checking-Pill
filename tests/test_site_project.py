from __future__ import annotations

import unittest
from pathlib import Path

from PIL import Image

from edgevision.config import EdgeVisionConfig
from edgevision.site_project import (
    add_reference_image,
    apply_site_project_to_config,
    init_site_project,
    summarize_site_project,
)


class SiteProjectTest(unittest.TestCase):
    def test_init_and_add_reference(self) -> None:
        root = Path("runs") / "tests" / "site_project"
        project = init_site_project(
            root,
            name="demo_site",
            detector_class_names=["pill", "bottle", "thermometer"],
            identity_detection_labels=["pill"],
            quality_detection_labels=["pill"],
        )

        source = root / "captures" / "round_white.jpg"
        Image.new("RGB", (64, 64), (240, 240, 240)).save(source)
        target = add_reference_image(root, label="Round White", image_path=source)
        summary = summarize_site_project(root)

        config = apply_site_project_to_config(EdgeVisionConfig(), root)

        self.assertEqual(project.detector_class_names, ["pill", "bottle", "thermometer"])
        self.assertTrue(target.exists())
        self.assertEqual(summary["reference_counts"], {"Round_White": 1})
        self.assertEqual(config.identifier.apply_to_detection_labels, ["pill"])
        self.assertEqual(config.quality.apply_to_detection_labels, ["pill"])


if __name__ == "__main__":
    unittest.main()
