from __future__ import annotations

import numpy as np
from PIL import Image

from edgevision.config import QualityConfig
from edgevision.image_utils import foreground_mask_from_border, image_to_array
from edgevision.schemas import IdentificationResult, QualityResult, QualityStatus


class RuleBasedQualityInspector:
    def __init__(self, config: QualityConfig | None = None):
        self.config = config or QualityConfig()

    def inspect(
        self,
        crop: Image.Image,
        identity: IdentificationResult | None = None,
    ) -> QualityResult:
        array = image_to_array(crop)
        gray = np.asarray(crop.convert("L"), dtype=np.float32)
        mask = foreground_mask_from_border(array, threshold=28)

        foreground_ratio = float(mask.mean())
        border_touch_ratio = self._border_touch_ratio(mask)
        solidity = self._bbox_solidity(mask)
        dark_spot_ratio = self._dark_spot_ratio(gray, mask)

        flags: list[str] = []
        penalties = 0.0

        if foreground_ratio < self.config.min_foreground_ratio:
            flags.append("foreground_too_small")
            penalties += 0.35
        if foreground_ratio > self.config.max_foreground_ratio:
            flags.append("foreground_too_large")
            penalties += 0.25
        if solidity < self.config.min_mask_solidity:
            flags.append("shape_irregular")
            penalties += 0.25
        if border_touch_ratio > self.config.max_border_touch_ratio:
            flags.append("touching_crop_border")
            penalties += 0.2
        if dark_spot_ratio > self.config.max_dark_spot_ratio:
            flags.append("dark_surface_anomaly")
            penalties += 0.25
        if identity is not None and identity.is_unknown:
            flags.append("identity_unknown")
            penalties += 0.12

        score = max(0.0, 1.0 - penalties)
        if score < self.config.review_score_threshold and flags:
            status = QualityStatus.NG if any(flag in flags for flag in ("shape_irregular", "dark_surface_anomaly")) else QualityStatus.REVIEW
        elif flags:
            status = QualityStatus.REVIEW
        else:
            status = QualityStatus.OK

        return QualityResult(
            status=status,
            score=score,
            flags=flags,
            measurements={
                "foreground_ratio": foreground_ratio,
                "mask_solidity": solidity,
                "border_touch_ratio": border_touch_ratio,
                "dark_spot_ratio": dark_spot_ratio,
            },
        )

    @staticmethod
    def _border_touch_ratio(mask: np.ndarray) -> float:
        if mask.size == 0:
            return 0.0
        border = np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])
        return float(border.mean())

    @staticmethod
    def _bbox_solidity(mask: np.ndarray) -> float:
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return 0.0
        bbox_area = float((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))
        return float(mask.sum() / max(bbox_area, 1.0))

    @staticmethod
    def _dark_spot_ratio(gray: np.ndarray, mask: np.ndarray) -> float:
        values = gray[mask]
        if len(values) < 16:
            return 0.0
        threshold = max(0.0, float(np.percentile(values, 20) - 25.0))
        return float((values < threshold).mean())

