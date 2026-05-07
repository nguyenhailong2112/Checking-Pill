from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sanitize_yolo_line(line: str) -> tuple[str | None, str]:
    parts = line.split()
    if len(parts) != 5:
        return None, "dropped_expected_5_columns"

    try:
        class_id = int(float(parts[0]))
        x_center, y_center, width, height = [float(value) for value in parts[1:]]
    except ValueError:
        return None, "dropped_non_numeric"

    if width <= 0 or height <= 0:
        return None, "dropped_non_positive_size"

    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2

    clipped_x1 = min(max(x1, 0.0), 1.0)
    clipped_y1 = min(max(y1, 0.0), 1.0)
    clipped_x2 = min(max(x2, 0.0), 1.0)
    clipped_y2 = min(max(y2, 0.0), 1.0)

    if clipped_x2 <= clipped_x1 or clipped_y2 <= clipped_y1:
        return None, "dropped_empty_after_clip"

    new_width = clipped_x2 - clipped_x1
    new_height = clipped_y2 - clipped_y1
    new_x_center = clipped_x1 + new_width / 2
    new_y_center = clipped_y1 + new_height / 2

    status = "unchanged" if (x1, y1, x2, y2) == (clipped_x1, clipped_y1, clipped_x2, clipped_y2) else "clipped"
    return (
        f"{class_id} {new_x_center:.6f} {new_y_center:.6f} {new_width:.6f} {new_height:.6f}",
        status,
    )


def export_split(source_root: Path, output_root: Path, split: str) -> dict[str, Any]:
    source_images = source_root / "images" / split
    source_labels = source_root / "labels" / split
    output_images = output_root / "images" / split
    output_labels = output_root / "labels" / split
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    images = [
        path for path in source_images.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    stats = {
        "split": split,
        "images_copied": 0,
        "labels_written": 0,
        "unchanged_boxes": 0,
        "clipped_boxes": 0,
        "dropped_boxes": 0,
        "missing_source_labels": 0,
    }

    for image_path in images:
        relative = image_path.relative_to(source_images)
        target_image = output_images / relative
        target_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, target_image)
        stats["images_copied"] += 1

        label_path = source_labels / relative.with_suffix(".txt")
        target_label = output_labels / relative.with_suffix(".txt")
        target_label.parent.mkdir(parents=True, exist_ok=True)

        if not label_path.exists():
            target_label.write_text("", encoding="utf-8")
            stats["missing_source_labels"] += 1
            stats["labels_written"] += 1
            continue

        sanitized_lines: list[str] = []
        text = label_path.read_text(encoding="utf-8").strip()
        for line in text.splitlines() if text else []:
            sanitized, status = sanitize_yolo_line(line)
            if sanitized is None:
                stats["dropped_boxes"] += 1
            else:
                sanitized_lines.append(sanitized)
                if status == "clipped":
                    stats["clipped_boxes"] += 1
                else:
                    stats["unchanged_boxes"] += 1

        target_label.write_text("\n".join(sanitized_lines) + ("\n" if sanitized_lines else ""), encoding="utf-8")
        stats["labels_written"] += 1

    return stats


def write_data_yaml(output_root: Path) -> None:
    data_yaml = output_root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {output_root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "test:",
                "names:",
                "  0: pill",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, help="Original YOLO archive dataset root.")
    parser.add_argument("--output-root", required=True, help="Output root for sanitized YOLO dataset.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()

    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output root is not empty: {output_root}. Pass --overwrite to reuse it.")

    output_root.mkdir(parents=True, exist_ok=True)
    report = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "splits": {
            split: export_split(source_root, output_root, split)
            for split in ("train", "val")
        },
    }
    report["summary"] = {
        "images_copied": sum(split["images_copied"] for split in report["splits"].values()),
        "labels_written": sum(split["labels_written"] for split in report["splits"].values()),
        "clipped_boxes": sum(split["clipped_boxes"] for split in report["splits"].values()),
        "dropped_boxes": sum(split["dropped_boxes"] for split in report["splits"].values()),
        "missing_source_labels": sum(split["missing_source_labels"] for split in report["splits"].values()),
    }

    write_data_yaml(output_root)
    report_path = output_root / "export_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

