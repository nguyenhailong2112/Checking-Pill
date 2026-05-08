from __future__ import annotations

from collections import Counter
from pathlib import Path
from time import perf_counter

from PIL import Image

from edgevision.config import EdgeVisionConfig
from edgevision.cropper import CropNormalizer
from edgevision.detectors.heuristic import HeuristicPillDetector
from edgevision.detectors.ultralytics_yolo import UltralyticsYOLODetector
from edgevision.identifiers.reference_gallery import ReferenceGalleryIdentifier
from edgevision.image_utils import ensure_dir, load_rgb_image
from edgevision.postprocess import filter_detections
from edgevision.precheck import run_image_precheck
from edgevision.quality.rule_based import RuleBasedQualityInspector
from edgevision.schemas import (
    Candidate,
    IdentificationResult,
    InspectionReport,
    PillItem,
    QualityResult,
    QualityStatus,
)
from edgevision.visualization import annotate_report


class PillInspectionPipeline:
    def __init__(self, config: EdgeVisionConfig | None = None):
        self.config = config or EdgeVisionConfig()
        self.detector = self._build_detector()
        self.cropper = CropNormalizer(self.config.cropper)
        self.identifier = ReferenceGalleryIdentifier(self.config.identifier)
        self.quality_inspector = RuleBasedQualityInspector(self.config.quality)

    def _build_detector(self):
        backend = self.config.detector.backend.lower()
        if backend == "auto":
            weights_path = self.config.detector.yolo.weights_path
            if weights_path and Path(weights_path).exists():
                try:
                    return UltralyticsYOLODetector(self.config.detector.yolo)
                except RuntimeError:
                    return HeuristicPillDetector(self.config.detector.heuristic)
            return HeuristicPillDetector(self.config.detector.heuristic)
        if backend == "heuristic":
            return HeuristicPillDetector(self.config.detector.heuristic)
        if backend in {"yolo", "ultralytics", "ultralytics_yolo"}:
            return UltralyticsYOLODetector(self.config.detector.yolo)
        raise ValueError(f"Unsupported detector backend: {self.config.detector.backend}")

    def inspect_path(self, image_path: str | Path, output_dir: str | Path | None = None) -> InspectionReport:
        image = load_rgb_image(image_path)
        return self.inspect_image(image=image, image_path=str(image_path), output_dir=output_dir)

    def inspect_image(
        self,
        image: Image.Image,
        image_path: str = "<memory>",
        output_dir: str | Path | None = None,
    ) -> InspectionReport:
        if output_dir is not None:
            ensure_dir(output_dir)

        started_at = perf_counter()
        precheck = run_image_precheck(image)
        after_precheck = perf_counter()
        raw_detections = [] if not precheck["usable"] else self.detector.detect(image)
        after_detector = perf_counter()
        detections = filter_detections(
            raw_detections,
            confidence_threshold=self.config.detector.confidence_threshold,
            iou_threshold=self.config.detector.nms_iou_threshold,
            image_width=image.width,
            image_height=image.height,
        )
        after_postprocess = perf_counter()

        items: list[PillItem] = []
        for index, detection in enumerate(detections, start=1):
            crop = self.cropper.crop(image, detection)
            crop_path = self.cropper.maybe_save(crop, output_dir=output_dir, item_id=index)
            identity = (
                self.identifier.identify(crop)
                if self._label_enabled(detection.label, self.config.identifier.apply_to_detection_labels)
                else self._identity_from_detection(detection)
            )
            quality = (
                self.quality_inspector.inspect(crop, identity=identity)
                if self._label_enabled(detection.label, self.config.quality.apply_to_detection_labels)
                else self._quality_not_applied()
            )
            items.append(
                PillItem(
                    item_id=index,
                    detection=detection,
                    crop_path=crop_path,
                    identity=identity,
                    quality=quality,
                )
            )

        after_items = perf_counter()
        count_by_detection_label = dict(Counter(item.detection.label for item in items))
        count_by_type = dict(Counter(item.identity.label for item in items))
        image_status = self._fuse_image_status(items, precheck)
        report = InspectionReport(
            image_path=image_path,
            image_width=image.width,
            image_height=image.height,
            total_count=len(items),
            count_by_type=count_by_type,
            items=items,
            image_status=image_status,
            count_by_detection_label=count_by_detection_label,
            precheck=precheck,
            runtime={
                "precheck_ms": (after_precheck - started_at) * 1000.0,
                "detector_ms": (after_detector - after_precheck) * 1000.0,
                "postprocess_ms": (after_postprocess - after_detector) * 1000.0,
                "items_ms": (after_items - after_postprocess) * 1000.0,
                "total_ms": (after_items - started_at) * 1000.0,
            },
        )

        if output_dir is not None:
            annotated = annotate_report(image, report)
            annotated.save(Path(output_dir) / "annotated.jpg", quality=95)

        return report

    @staticmethod
    def _label_enabled(label: str, enabled_labels: list[str]) -> bool:
        if not enabled_labels:
            return False
        normalized = {value.lower() for value in enabled_labels}
        return "*" in normalized or label.lower() in normalized

    @staticmethod
    def _identity_from_detection(detection) -> IdentificationResult:
        confidence = max(0.0, min(1.0, float(detection.confidence)))
        return IdentificationResult(
            label=detection.label,
            confidence=confidence,
            candidates=[Candidate(label=detection.label, confidence=confidence, distance=None)],
            is_unknown=False,
        )

    @staticmethod
    def _quality_not_applied() -> QualityResult:
        return QualityResult(
            status=QualityStatus.OK,
            score=1.0,
            flags=[],
            measurements={"quality_applied": 0.0},
        )

    @staticmethod
    def _fuse_image_status(items: list[PillItem], precheck: dict) -> QualityStatus:
        if not precheck.get("usable", False):
            return QualityStatus.REVIEW
        if any(item.quality.status == QualityStatus.NG for item in items):
            return QualityStatus.NG
        if any(item.quality.status == QualityStatus.REVIEW for item in items):
            return QualityStatus.REVIEW
        return QualityStatus.OK
