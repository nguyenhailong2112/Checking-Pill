from __future__ import annotations

import csv
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from edgevision.config import IdentifierConfig
from edgevision.identifiers.reference_gallery import ReferenceGalleryIdentifier


class ReferenceGalleryTest(unittest.TestCase):
    def test_loads_manifest_gallery(self) -> None:
        root = Path("runs") / "tests" / "reference_gallery"
        root.mkdir(parents=True, exist_ok=True)

        red = root / "red.jpg"
        blue = root / "blue.jpg"
        self._make_pill(red, (230, 80, 80))
        self._make_pill(blue, (70, 140, 230))

        manifest = root / "manifest.csv"
        with manifest.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["label", "image_path", "is_ref"])
            writer.writeheader()
            writer.writerow({"label": "red_pill", "image_path": "red.jpg", "is_ref": "True"})
            writer.writerow({"label": "blue_pill", "image_path": "blue.jpg", "is_ref": "True"})

        identifier = ReferenceGalleryIdentifier(
            IdentifierConfig(
                reference_manifest=str(manifest),
                reference_image_root=str(root),
                unknown_threshold=0.5,
            )
        )

        query = Image.open(red).convert("RGB")
        result = identifier.identify(query)
        self.assertEqual(result.label, "red_pill")
        self.assertFalse(result.is_unknown)

    @staticmethod
    def _make_pill(path: Path, color: tuple[int, int, int]) -> None:
        image = Image.new("RGB", (160, 160), (128, 128, 128))
        draw = ImageDraw.Draw(image)
        draw.ellipse((40, 40, 120, 120), fill=color)
        image.save(path)


if __name__ == "__main__":
    unittest.main()

