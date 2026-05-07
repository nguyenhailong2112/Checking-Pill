from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class HeuristicDetectorConfig:
    min_area_ratio: float = 0.0008
    max_area_ratio: float = 0.35
    foreground_threshold: int = 32
    merge_distance_px: int = 0


@dataclass
class YoloDetectorConfig:
    weights_path: str | None = None
    image_size: int = 832
    device: str | None = None


@dataclass
class DetectorConfig:
    backend: str = "heuristic"
    confidence_threshold: float = 0.25
    nms_iou_threshold: float = 0.45
    heuristic: HeuristicDetectorConfig = field(default_factory=HeuristicDetectorConfig)
    yolo: YoloDetectorConfig = field(default_factory=YoloDetectorConfig)


@dataclass
class CropperConfig:
    padding_ratio: float = 0.12
    output_size: int = 224
    save_crops: bool = True


@dataclass
class IdentifierConfig:
    backend: str = "reference_gallery"
    reference_root: str | None = None
    reference_manifest: str | None = None
    reference_image_root: str | None = None
    reference_only: bool = True
    unknown_threshold: float = 0.82
    top_k: int = 5


@dataclass
class QualityConfig:
    min_foreground_ratio: float = 0.08
    max_foreground_ratio: float = 0.92
    min_mask_solidity: float = 0.72
    max_border_touch_ratio: float = 0.18
    max_dark_spot_ratio: float = 0.16
    review_score_threshold: float = 0.62


@dataclass
class EdgeVisionConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    cropper: CropperConfig = field(default_factory=CropperConfig)
    identifier: IdentifierConfig = field(default_factory=IdentifierConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)


def _get_nested(data: dict[str, Any], key: str, default: dict[str, Any]) -> dict[str, Any]:
    value = data.get(key, default)
    if value is None:
        return default
    if not isinstance(value, dict):
        raise TypeError(f"Expected object for config section '{key}', got {type(value).__name__}")
    return value


def load_config(path: str | Path | None = None) -> EdgeVisionConfig:
    if path is None:
        return EdgeVisionConfig()

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    detector_raw = _get_nested(raw, "detector", {})
    heuristic_raw = _get_nested(detector_raw, "heuristic", {})
    yolo_raw = _get_nested(detector_raw, "yolo", {})
    cropper_raw = _get_nested(raw, "cropper", {})
    identifier_raw = _get_nested(raw, "identifier", {})
    quality_raw = _get_nested(raw, "quality", {})

    detector = DetectorConfig(
        backend=detector_raw.get("backend", "heuristic"),
        confidence_threshold=detector_raw.get("confidence_threshold", 0.25),
        nms_iou_threshold=detector_raw.get("nms_iou_threshold", 0.45),
        heuristic=HeuristicDetectorConfig(**heuristic_raw),
        yolo=YoloDetectorConfig(**yolo_raw),
    )

    return EdgeVisionConfig(
        detector=detector,
        cropper=CropperConfig(**cropper_raw),
        identifier=IdentifierConfig(**identifier_raw),
        quality=QualityConfig(**quality_raw),
    )
