from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def count_files(path: Path, suffixes: set[str] | None = None) -> tuple[int, int]:
    count = 0
    size = 0
    if not path.exists():
        return count, size
    for file_path in path.rglob("*"):
        if not file_path.is_file():
            continue
        if suffixes is not None and file_path.suffix.lower() not in suffixes:
            continue
        count += 1
        size += file_path.stat().st_size
    return count, size


def audit_yolo_archive(root: Path) -> dict:
    archive = root / "archive"
    label_files = list((archive / "labels").rglob("*.txt")) if (archive / "labels").exists() else []
    non_empty = 0
    boxes = 0
    for label_file in label_files:
        text = label_file.read_text(encoding="utf-8").strip()
        if text:
            non_empty += 1
            boxes += len(text.splitlines())

    train_images, train_size = count_files(archive / "images" / "train", {".jpg", ".jpeg", ".png"})
    val_images, val_size = count_files(archive / "images" / "val", {".jpg", ".jpeg", ".png"})

    return {
        "train_images": train_images,
        "val_images": val_images,
        "image_size_mb": round((train_size + val_size) / 1024 / 1024, 3),
        "label_files": len(label_files),
        "non_empty_label_files": non_empty,
        "empty_label_files": len(label_files) - non_empty,
        "boxes": boxes,
    }


def audit_epillid(root: Path) -> dict:
    labels_path = root / "ePillID_data" / "ePillID_data" / "all_labels.csv"
    if not labels_path.exists():
        return {"exists": False}

    rows = []
    with labels_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    label_counts = Counter(row["label"] for row in rows)
    is_ref_counts = Counter(row["is_ref"] for row in rows)
    is_front_counts = Counter(row["is_front"] for row in rows)

    return {
        "exists": True,
        "rows": len(rows),
        "labels": len(label_counts),
        "is_ref": dict(is_ref_counts),
        "is_front": dict(is_front_counts),
        "min_rows_per_label": min(label_counts.values()) if label_counts else 0,
        "max_rows_per_label": max(label_counts.values()) if label_counts else 0,
    }


def audit_mediseg(root: Path) -> dict:
    mediseg = root / "MEDISEG" / "MEDISEG"
    metadata = mediseg / "metadata.csv"
    rows = 0
    if metadata.exists():
        with metadata.open("r", encoding="utf-8", newline="") as file:
            rows = sum(1 for _ in csv.DictReader(file))

    images_3, size_3 = count_files(mediseg / "3pills" / "images", {".jpg", ".jpeg", ".png"})
    images_32, size_32 = count_files(mediseg / "32pills" / "images", {".jpg", ".jpeg", ".png"})
    return {
        "metadata_rows": rows,
        "3pills_images": images_3,
        "32pills_images": images_32,
        "image_size_mb": round((size_3 + size_32) / 1024 / 1024, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, help="Path to ePillID benchmark source folder.")
    parser.add_argument("--output", help="Optional JSON output path.")
    args = parser.parse_args()

    root = Path(args.source_root)
    report = {
        "source_root": str(root),
        "archive_yolo": audit_yolo_archive(root),
        "epillid": audit_epillid(root),
        "mediseg": audit_mediseg(root),
    }

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

