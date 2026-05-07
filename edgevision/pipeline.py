from __future__ import annotations

from collections import Counter
from pathlib import Path

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
from edgevision.schemas import InspectionReport, PillItem, QualityStatus
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

        precheck = run_image_precheck(image)
        raw_detections = [] if not precheck["usable"] else self.detector.detect(image)
        detections = filter_detections(
            raw_detections,
            confidence_threshold=self.config.detector.confidence_threshold,
            iou_threshold=self.config.detector.nms_iou_threshold,
            image_width=image.width,
            image_height=image.height,
        )

        items: list[PillItem] = []
        for index, detection in enumerate(detections, start=1):
            crop = self.cropper.crop(image, detection)
            crop_path = self.cropper.maybe_save(crop, output_dir=output_dir, item_id=index)
            identity = self.identifier.identify(crop)
            quality = self.quality_inspector.inspect(crop, identity=identity)
            items.append(
                PillItem(
                    item_id=index,
                    detection=detection,
                    crop_path=crop_path,
                    identity=identity,
                    quality=quality,
                )
            )

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
            precheck=precheck,
        )

        if output_dir is not None:
            annotated = annotate_report(image, report)
            annotated.save(Path(output_dir) / "annotated.jpg", quality=95)

        return report

    @staticmethod
    def _fuse_image_status(items: list[PillItem], precheck: dict) -> QualityStatus:
        if not precheck.get("usable", False):
            return QualityStatus.REVIEW
        if any(item.quality.status == QualityStatus.NG for item in items):
            return QualityStatus.NG
        if any(item.quality.status == QualityStatus.REVIEW for item in items):
            return QualityStatus.REVIEW
        return QualityStatus.OK

