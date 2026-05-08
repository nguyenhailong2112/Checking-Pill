from __future__ import annotations

import json
from pathlib import Path

from edgevision.image_utils import ensure_dir
from edgevision.schemas import InspectionReport


def save_report_json(report: InspectionReport, output_dir: str | Path) -> Path:
    directory = ensure_dir(output_dir)
    report_path = directory / "report.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report.to_dict(), file, ensure_ascii=False, indent=2)
    return report_path


def summarize_report_rows(report: InspectionReport) -> list[dict[str, str | int | float]]:
    if not report.items:
        return [
            {
                "image_path": report.image_path,
                "image_status": report.image_status.value,
                "total_count": report.total_count,
                "item_id": "",
                "bbox": "",
                "det_label": "",
                "det_conf": "",
                "type": "",
                "type_conf": "",
                "quality": "",
                "quality_score": "",
                "flags": "",
                "crop_path": "",
            }
        ]

    rows = []
    for item in report.items:
        rows.append(
            {
                "image_path": report.image_path,
                "image_status": report.image_status.value,
                "total_count": report.total_count,
                "item_id": item.item_id,
                "bbox": " ".join(str(value) for value in item.detection.bbox.to_int_list()),
                "det_label": item.detection.label,
                "det_conf": round(float(item.detection.confidence), 6),
                "type": item.identity.label,
                "type_conf": round(float(item.identity.confidence), 6),
                "quality": item.quality.status.value,
                "quality_score": round(float(item.quality.score), 6),
                "flags": "|".join(item.quality.flags),
                "crop_path": item.crop_path or "",
            }
        )
    return rows
