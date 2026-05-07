from __future__ import annotations

from edgevision.schemas import BBox, Detection


def bbox_iou(a: BBox, b: BBox) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    union = a.area + b.area - inter
    if union <= 1e-12:
        return 0.0
    return inter / union


def non_max_suppression(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    ordered = sorted(detections, key=lambda detection: detection.confidence, reverse=True)
    kept: list[Detection] = []

    for detection in ordered:
        if all(bbox_iou(detection.bbox, existing.bbox) <= iou_threshold for existing in kept):
            kept.append(detection)

    return kept


def filter_detections(
    detections: list[Detection],
    confidence_threshold: float,
    iou_threshold: float,
    image_width: int,
    image_height: int,
) -> list[Detection]:
    valid: list[Detection] = []
    image_area = image_width * image_height

    for detection in detections:
        bbox = detection.bbox.clipped(image_width, image_height)
        if detection.confidence < confidence_threshold:
            continue
        if bbox.area <= 1 or bbox.area > image_area:
            continue
        valid.append(Detection(bbox=bbox, confidence=detection.confidence, label=detection.label, source=detection.source))

    return non_max_suppression(valid, iou_threshold=iou_threshold)

