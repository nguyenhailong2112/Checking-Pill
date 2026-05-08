from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full YOLO dataset audit")
    p.add_argument("--dataset-root", required=True, help="YOLO dataset root")
    p.add_argument("--report-json", default="", help="Optional output json report path")
    p.add_argument("--max-examples", type=int, default=30, help="Max invalid examples to keep")
    return p.parse_args()


def read_data_yaml_class_names(dataset_root: Path) -> dict[int, str]:
    """
    Parse simple YOLO data.yaml names section, supporting:
      names:
        0: pill
        1: bottle
    or:
      names: [pill, bottle]
    If parsing fails, return {}.
    """
    yaml_path = dataset_root / "data.yaml"
    if not yaml_path.exists():
        return {}

    text = yaml_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    names_map: dict[int, str] = {}
    in_names_block = False

    # Try dict style block first
    for line in lines:
        s = line.rstrip()
        if s.strip().startswith("names:"):
            in_names_block = True
            # inline list style
            if "[" in s and "]" in s:
                inside = s[s.find("[") + 1 : s.rfind("]")]
                items = [x.strip().strip("'\"") for x in inside.split(",") if x.strip()]
                return {i: n for i, n in enumerate(items)}
            continue

        if in_names_block:
            if not s.strip():
                continue
            # next top-level key => break
            if not s.startswith(" ") and ":" in s:
                break
            # expected "  0: pill"
            t = s.strip()
            if ":" in t:
                k, v = t.split(":", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k.isdigit():
                    names_map[int(k)] = v

    return names_map


def validate_yolo_line(line: str) -> tuple[bool, str, int | None]:
    parts = line.split()
    if len(parts) != 5:
        return False, "expected_5_columns", None
    try:
        class_id = int(float(parts[0]))
        x_center, y_center, width, height = [float(v) for v in parts[1:]]
    except ValueError:
        return False, "non_numeric_value", None

    if class_id < 0:
        return False, "negative_class_id", class_id

    if not all(0.0 <= value <= 1.0 for value in (x_center, y_center, width, height)):
        return False, "normalized_value_out_of_range", class_id
    if width <= 0 or height <= 0:
        return False, "non_positive_box_size", class_id
    if x_center - width / 2 < -1e-6 or x_center + width / 2 > 1.0 + 1e-6:
        return False, "box_exceeds_image_x", class_id
    if y_center - height / 2 < -1e-6 or y_center + height / 2 > 1.0 + 1e-6:
        return False, "box_exceeds_image_y", class_id

    return True, "", class_id


def gather_split(dataset_root: Path, split: str, max_examples: int) -> dict[str, Any]:
    images_dir = dataset_root / "images" / split
    labels_dir = dataset_root / "labels" / split

    image_files = sorted(
        p for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ) if images_dir.exists() else []
    label_files = sorted(labels_dir.rglob("*.txt")) if labels_dir.exists() else []

    image_stems = {p.stem for p in image_files}
    label_stems = {p.stem for p in label_files}

    images_without_labels = sorted(image_stems - label_stems)
    labels_without_images = sorted(label_stems - image_stems)

    class_counter = Counter()
    invalid_reasons = Counter()
    invalid_examples: list[dict[str, Any]] = []
    empty_label_files = 0
    total_boxes = 0

    for lp in label_files:
        txt = lp.read_text(encoding="utf-8", errors="ignore").strip()
        if not txt:
            empty_label_files += 1
            continue
        for line_no, line in enumerate(txt.splitlines(), start=1):
            ok, reason, class_id = validate_yolo_line(line)
            if ok:
                total_boxes += 1
                class_counter[class_id] += 1
            else:
                invalid_reasons[reason] += 1
                if len(invalid_examples) < max_examples:
                    invalid_examples.append({
                        "file": str(lp),
                        "line_no": line_no,
                        "reason": reason,
                        "line": line,
                    })

    return {
        "split": split,
        "images": len(image_files),
        "labels": len(label_files),
        "boxes": total_boxes,
        "empty_label_files": empty_label_files,
        "images_without_labels_count": len(images_without_labels),
        "labels_without_images_count": len(labels_without_images),
        "images_without_labels_examples": images_without_labels[:20],
        "labels_without_images_examples": labels_without_images[:20],
        "invalid_lines": sum(invalid_reasons.values()),
        "invalid_reasons": dict(invalid_reasons),
        "invalid_examples": invalid_examples,
        "class_distribution": dict(sorted(class_counter.items(), key=lambda x: x[0])),
    }


def main() -> int:
    args = parse_args()
    root = Path(args.dataset_root).resolve()

    data_yaml_names = read_data_yaml_class_names(root)

    splits = {}
    for split in ("train", "val"):
        splits[split] = gather_split(root, split, args.max_examples)

    # aggregate
    agg_class_counter = Counter()
    agg_invalid = Counter()
    total = defaultdict(int)
    for s in splits.values():
        for k in ("images", "labels", "boxes", "empty_label_files", "invalid_lines"):
            total[k] += s[k]
        agg_class_counter.update(s["class_distribution"])
        agg_invalid.update(s["invalid_reasons"])

    class_ids = sorted(int(k) for k in agg_class_counter.keys())
    max_class_id = max(class_ids) if class_ids else -1
    inferred_num_classes = max_class_id + 1 if max_class_id >= 0 else 0

    # Build recommended class-names placeholder
    recommended_class_names = []
    for cid in range(inferred_num_classes):
        if cid in data_yaml_names:
            recommended_class_names.append(data_yaml_names[cid])
        else:
            recommended_class_names.append(f"class_{cid}")

    report = {
        "dataset_root": str(root),
        "splits": splits,
        "summary": {
            "images": total["images"],
            "labels": total["labels"],
            "boxes": total["boxes"],
            "empty_label_files": total["empty_label_files"],
            "invalid_lines": total["invalid_lines"],
            "invalid_reasons": dict(agg_invalid),
        },
        "classes": {
            "class_ids_found": class_ids,
            "num_classes_inferred": inferred_num_classes,
            "class_distribution_all": dict(sorted((int(k), v) for k, v in agg_class_counter.items())),
            "data_yaml_names": {int(k): v for k, v in data_yaml_names.items()},
            "recommended_class_names_ordered": recommended_class_names,
            "recommended_cli": "--class-names " + " ".join(recommended_class_names) if recommended_class_names else "",
        },
    }

    print("=" * 80)
    print("YOLO DATASET AUDIT")
    print("=" * 80)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print("-" * 80)
    print("Classes:")
    print(json.dumps(report["classes"], ensure_ascii=False, indent=2))
    print("-" * 80)
    for split in ("train", "val"):
        print(f"[{split}]")
        print(json.dumps({
            "images": report["splits"][split]["images"],
            "labels": report["splits"][split]["labels"],
            "boxes": report["splits"][split]["boxes"],
            "invalid_lines": report["splits"][split]["invalid_lines"],
            "invalid_reasons": report["splits"][split]["invalid_reasons"],
            "images_without_labels_count": report["splits"][split]["images_without_labels_count"],
            "labels_without_images_count": report["splits"][split]["labels_without_images_count"],
        }, ensure_ascii=False, indent=2))
    print("=" * 80)

    if args.report_json:
        out = Path(args.report_json).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved report: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())