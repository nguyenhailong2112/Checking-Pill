from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QualityStatus(str, Enum):
    OK = "OK"
    NG = "NG"
    REVIEW = "REVIEW"


@dataclass(frozen=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_int_list(self) -> list[int]:
        return [round(self.x1), round(self.y1), round(self.x2), round(self.y2)]

    def clipped(self, width: int, height: int) -> "BBox":
        return BBox(
            x1=min(max(self.x1, 0), width),
            y1=min(max(self.y1, 0), height),
            x2=min(max(self.x2, 0), width),
            y2=min(max(self.y2, 0), height),
        )

    def padded(self, ratio: float, width: int, height: int) -> "BBox":
        pad_x = self.width * ratio
        pad_y = self.height * ratio
        return BBox(
            self.x1 - pad_x,
            self.y1 - pad_y,
            self.x2 + pad_x,
            self.y2 + pad_y,
        ).clipped(width, height)


@dataclass
class Detection:
    bbox: BBox
    confidence: float
    label: str = "pill"
    source: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": self.bbox.to_int_list(),
            "confidence": round(float(self.confidence), 6),
            "label": self.label,
            "source": self.source,
        }


@dataclass
class Candidate:
    label: str
    confidence: float
    distance: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(float(self.confidence), 6),
            "distance": None if self.distance is None else round(float(self.distance), 6),
        }


@dataclass
class IdentificationResult:
    label: str
    confidence: float
    candidates: list[Candidate] = field(default_factory=list)
    is_unknown: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(float(self.confidence), 6),
            "is_unknown": self.is_unknown,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass
class QualityResult:
    status: QualityStatus
    score: float
    flags: list[str] = field(default_factory=list)
    measurements: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "score": round(float(self.score), 6),
            "flags": self.flags,
            "measurements": {
                key: round(float(value), 6) for key, value in self.measurements.items()
            },
        }


@dataclass
class PillItem:
    item_id: int
    detection: Detection
    crop_path: str | None
    identity: IdentificationResult
    quality: QualityResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.item_id,
            "bbox": self.detection.bbox.to_int_list(),
            "det_conf": round(float(self.detection.confidence), 6),
            "type": self.identity.label,
            "type_conf": round(float(self.identity.confidence), 6),
            "quality": self.quality.status.value,
            "quality_score": round(float(self.quality.score), 6),
            "flags": self.quality.flags,
            "crop_path": self.crop_path,
            "identity": self.identity.to_dict(),
            "quality_details": self.quality.to_dict(),
        }


@dataclass
class InspectionReport:
    image_path: str
    image_width: int
    image_height: int
    total_count: int
    count_by_type: dict[str, int]
    items: list[PillItem]
    image_status: QualityStatus
    precheck: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_path": self.image_path,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "total_count": self.total_count,
            "count_by_type": self.count_by_type,
            "image_status": self.image_status.value,
            "precheck": self.precheck,
            "items": [item.to_dict() for item in self.items],
        }

