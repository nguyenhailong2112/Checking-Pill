from __future__ import annotations

import csv
from pathlib import Path

from edgevision.image_utils import ensure_dir
from edgevision.pipeline import PillInspectionPipeline
from edgevision.reporting import save_report_json, summarize_report_rows
from edgevision.schemas import InspectionReport


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_image_paths(input_path: str | Path) -> list[Path]:
    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {path}")
        return [path]

    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    images = [
        file_path
        for file_path in path.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images)


def run_batch(
    pipeline: PillInspectionPipeline,
    input_path: str | Path,
    output_dir: str | Path,
) -> list[InspectionReport]:
    output_root = ensure_dir(output_dir)
    reports: list[InspectionReport] = []

    for index, image_path in enumerate(iter_image_paths(input_path), start=1):
        stem = image_path.stem
        image_output_dir = output_root / f"{index:04d}_{stem}"
        report = pipeline.inspect_path(image_path, output_dir=image_output_dir)
        save_report_json(report, image_output_dir)
        reports.append(report)

    save_summary_csv(reports, output_root / "summary.csv")
    return reports


def save_summary_csv(reports: list[InspectionReport], output_path: str | Path) -> Path:
    rows = []
    for report in reports:
        rows.extend(summarize_report_rows(report))

    fieldnames = [
        "image_path",
        "image_status",
        "total_count",
        "item_id",
        "bbox",
        "det_label",
        "det_conf",
        "type",
        "type_conf",
        "quality",
        "quality_score",
        "flags",
        "crop_path",
    ]

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path
