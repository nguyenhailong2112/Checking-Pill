from __future__ import annotations

from typing import Protocol

from PIL import Image

from edgevision.schemas import IdentificationResult


class PillIdentifier(Protocol):
    def identify(self, crop: Image.Image) -> IdentificationResult:
        """Return pill identity prediction for a normalized crop."""

