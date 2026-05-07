from __future__ import annotations

from typing import Protocol

from PIL import Image

from edgevision.schemas import IdentificationResult, QualityResult


class QualityInspector(Protocol):
    def inspect(
        self,
        crop: Image.Image,
        identity: IdentificationResult | None = None,
    ) -> QualityResult:
        """Return OK/NG/REVIEW quality result for a normalized crop."""

