from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_label_lines(label_path: Path) -> list[str]:
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    return [line.strip() for line in text.splitlines() if line.strip()]


def validate_label_line(line: str) -> tuple[bool, str | None, int | None]:
    parts = line.split()
    if len(parts) != 5:
        return False, "expected_5_columns", None

    try:
        class_id_float = float(parts[0])
        values = [float(value) for value in parts[1:]]
    except ValueError:
        return False, "non_numeric_value", None

    class_id = int(class_id_float)
    if class_id_float != class_id:
        return False, "class_id_not_integer", None

    x_center, y_center, width, height = values
    if not all(0.0 <= value <= 1.0 for value in values):
        return False, "normalized_value_out_of_range", class_id
    if width <= 0.0 or height <= 0.0:
        return False, "non_positive_box_size", class_id
    if x_center - width / 2 < -1e-6 or x_center + width / 2 > 1.0 + 1e-6:
        return False, "box_exceeds_image_x", class_id
    if y_center - height / 2 < -1e-6 or y_center + height / 2 > 1.0 + 1e-6:
        return False, "box_exceeds_image_y", class_id

    return True, None, class_id


def collect_split(dataset_root: Path, split: str) -> dict[str, Any]:
    images_dir = dataset_root / "images" / split
    labels_dir = dataset_root / "labels" / split
    image_files = {
        path.stem: path
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    } if images_dir.exists() else {}
    label_files = {
        path.stem: path
        for path in labels_dir.rglob("*.txt")
        if path.is_file()
    } if labels_dir.exists() else {}

    missing_labels = sorted(set(image_files) - set(label_files))
    labels_without_images = sorted(set(label_files) - set(image_files))
    invalid_examples: list[dict[str, Any]] = []
    invalid_reason_counts: Counter[str] = Counter()
    class_counts: Counter[int] = Counter()
    boxes = 0
    empty_labels = 0
    boxes_per_file: list[int] = []

    for stem, label_path in sorted(label_files.items()):
        lines = read_label_lines(label_path)
        if not lines:
            empty_labels += 1
        boxes_per_file.append(len(lines))
        for line_number, line in enumerate(lines, start=1):
            ok, reason, class_id = validate_label_line(line)
            if class_id is not None:
                class_counts[class_id] += 1
            if ok:
                boxes += 1
            else:
                invalid_reason_counts[reason or "unknown"] += 1
                if len(invalid_examples) < 30:
                    invalid_examples.append(
                        {
                            "file": str(label_path),
                            "line_number": line_number,
                            "line": line,
                            "reason": reason,
                        }
                    )

    histogram: defaultdict[str, int] = defaultdict(int)
    for value in boxes_per_file:
        key = "0" if value == 0 else "1" if value == 1 else "2-5" if value <= 5 else "6-20" if value <= 20 else "21+"
        histogram[key] += 1

    return {
        "split": split,
        "images": len(image_files),
        "labels": len(label_files),
        "missing_labels": missing_labels[:100],
        "missing_labels_count": len(missing_labels),
        "labels_without_images": labels_without_images[:100],
        "labels_without_images_count": len(labels_without_images),
        "empty_label_files": empty_labels,
        "non_empty_label_files": len(label_files) - empty_labels,
        "boxes": boxes,
        "class_counts": {str(key): value for key, value in sorted(class_counts.items())},
        "invalid_reason_counts": dict(invalid_reason_counts),
        "invalid_examples": invalid_examples,
        "boxes_per_file_histogram": dict(sorted(histogram.items())),
    }


def write_data_yaml(dataset_root: Path, output_path: Path) -> None:
    text = "\n".join(
        [
            f"path: {dataset_root.as_posix()}",
            "train: images/train",
            "val: images/val",
            "test:",
            "names:",
            "  0: pill",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, help="YOLO dataset root containing images/ and labels/.")
    parser.add_argument("--output", help="Optional JSON audit output path.")
    parser.add_argument("--write-data-yaml", help="Optional path to write a YOLO data.yaml.")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    report = {
        "dataset_root": str(dataset_root),
        "splits": {
            split: collect_split(dataset_root, split)
            for split in ("train", "val")
        },
    }
    report["summary"] = {
        "images": sum(split["images"] for split in report["splits"].values()),
        "labels": sum(split["labels"] for split in report["splits"].values()),
        "boxes": sum(split["boxes"] for split in report["splits"].values()),
        "invalid_lines": sum(
            sum(split["invalid_reason_counts"].values()) for split in report["splits"].values()
        ),
        "missing_labels": sum(split["missing_labels_count"] for split in report["splits"].values()),
        "labels_without_images": sum(
            split["labels_without_images_count"] for split in report["splits"].values()
        ),
    }

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")

    if args.write_data_yaml:
        write_data_yaml(dataset_root, Path(args.write_data_yaml))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

