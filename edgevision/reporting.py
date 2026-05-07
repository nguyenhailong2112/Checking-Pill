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

