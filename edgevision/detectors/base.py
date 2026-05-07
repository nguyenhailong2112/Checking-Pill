from __future__ import annotations

from typing import Protocol

from PIL import Image

from edgevision.schemas import Detection


class PillDetector(Protocol):
    def detect(self, image: Image.Image) -> list[Detection]:
        """Return raw pill detections for an RGB image."""

