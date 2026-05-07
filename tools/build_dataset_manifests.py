from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_epillid_identity_manifest(source_root: Path, output_path: Path) -> dict:
    labels_csv = source_root / "ePillID_data" / "ePillID_data" / "all_labels.csv"
    image_root = source_root / "ePillID_data" / "ePillID_data" / "classification_data"
    if not labels_csv.exists():
        return {"exists": False, "reason": f"missing {labels_csv}"}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    missing_images = 0

    with labels_csv.open("r", encoding="utf-8", newline="") as src, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        reader = csv.DictReader(src)
        fieldnames = list(reader.fieldnames or []) + ["abs_image_path", "image_exists"]
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            abs_image_path = image_root / row["image_path"]
            exists = abs_image_path.exists()
            if not exists:
                missing_images += 1
            row["abs_image_path"] = str(abs_image_path)
            row["image_exists"] = str(exists)
            writer.writerow(row)
            rows_written += 1

    return {
        "exists": True,
        "rows": rows_written,
        "missing_images": missing_images,
        "output": str(output_path),
    }


def build_mediseg_manifest(source_root: Path, output_path: Path) -> dict:
    mediseg_root = source_root / "MEDISEG" / "MEDISEG"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for subset in ("3pills", "32pills"):
        images_dir = mediseg_root / subset / "images"
        if not images_dir.exists():
            continue
        for image_path in images_dir.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append({"subset": subset, "abs_image_path": str(image_path), "file_name": image_path.name})

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["subset", "file_name", "abs_image_path"])
        writer.writeheader()
        writer.writerows(rows)

    return {"rows": len(rows), "output": str(output_path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-dir", default="data/manifests")
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "source_root": str(source_root),
        "epillid_identity": build_epillid_identity_manifest(
            source_root,
            output_dir / "epillid_identity_manifest.csv",
        ),
        "mediseg_unlabeled": build_mediseg_manifest(
            source_root,
            output_dir / "mediseg_unlabeled_manifest.csv",
        ),
    }

    report_path = output_dir / "manifest_build_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

