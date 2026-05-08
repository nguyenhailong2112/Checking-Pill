from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an EdgeVision YOLO detector on Colab.")
    parser.add_argument("--dataset-root", required=True, help="YOLO root containing images/ and labels/.")
    parser.add_argument("--data-yaml", help="Existing Ultralytics data.yaml. If omitted, one is written.")
    parser.add_argument("--class-names", nargs="+", default=["pill"], help="Detector class names in class-id order.")
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics model seed, e.g. yolo11n.pt or yolo11s.pt.")
    parser.add_argument("--output-dir", required=True, help="Folder where best.pt, last.pt, and reports are copied.")
    parser.add_argument("--project", default="runs/detect", help="Ultralytics training project folder.")
    parser.add_argument("--name", default="edgevision_detector", help="Ultralytics run name.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=832)
    parser.add_argument("--batch", type=int, default=-1, help="-1 lets Ultralytics auto-select batch size.")
    parser.add_argument("--device", default=None, help="Use 0 for Colab GPU, cpu for CPU, or omit for auto.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--single-cls", action="store_true", help="Train as one class even if labels contain classes.")
    parser.add_argument("--export-format", default="", help="Optional export format such as onnx.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data_yaml = Path(args.data_yaml).resolve() if args.data_yaml else write_data_yaml(dataset_root, args.class_names)
    audit = audit_yolo_dataset(dataset_root, class_count=len(args.class_names))
    (output_dir / "dataset_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if audit["summary"]["images"] == 0:
        raise ValueError(f"No training images were found under {dataset_root}")
    if audit["summary"]["invalid_lines"] > 0:
        raise ValueError(f"Dataset has invalid YOLO labels. See {output_dir / 'dataset_audit.json'}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install Ultralytics first: pip install ultralytics") from exc

    model = YOLO(args.model)
    result = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=args.project,
        name=args.name,
        cache=args.cache,
        single_cls=args.single_cls,
        cos_lr=True,
        close_mosaic=10,
        plots=True,
        verbose=True,
    )

    save_dir = Path(getattr(result, "save_dir", Path(args.project) / args.name))
    weights_dir = save_dir / "weights"
    best_pt = weights_dir / "best.pt"
    last_pt = weights_dir / "last.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"Training finished but best.pt was not found: {best_pt}")

    shutil.copy2(best_pt, output_dir / "best.pt")
    if last_pt.exists():
        shutil.copy2(last_pt, output_dir / "last.pt")
    shutil.copy2(data_yaml, output_dir / "data.yaml")

    trained = YOLO(str(best_pt))
    metrics = trained.val(data=str(data_yaml), imgsz=args.imgsz, device=args.device, plots=True)
    metrics_dict = getattr(metrics, "results_dict", {})
    (output_dir / "val_metrics.json").write_text(
        json.dumps(to_jsonable(metrics_dict), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    export_path = None
    if args.export_format:
        export_path = trained.export(format=args.export_format, imgsz=args.imgsz, device=args.device)

    summary = {
        "dataset_root": str(dataset_root),
        "data_yaml": str(data_yaml),
        "class_names": args.class_names,
        "ultralytics_save_dir": str(save_dir),
        "best_pt": str(output_dir / "best.pt"),
        "last_pt": str(output_dir / "last.pt") if last_pt.exists() else None,
        "export_path": None if export_path is None else str(export_path),
        "metrics": to_jsonable(metrics_dict),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


def write_data_yaml(dataset_root: Path, class_names: list[str]) -> Path:
    lines = [
        f"path: {dataset_root.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test:",
        "names:",
    ]
    for index, name in enumerate(class_names):
        lines.append(f"  {index}: {name}")
    lines.append("")
    data_yaml = dataset_root / "data.yaml"
    data_yaml.write_text("\n".join(lines), encoding="utf-8")
    return data_yaml


def audit_yolo_dataset(dataset_root: Path, class_count: int) -> dict[str, Any]:
    splits = {split: audit_split(dataset_root, split, class_count) for split in ("train", "val")}
    return {
        "dataset_root": str(dataset_root),
        "splits": splits,
        "summary": {
            "images": sum(split["images"] for split in splits.values()),
            "labels": sum(split["labels"] for split in splits.values()),
            "boxes": sum(split["boxes"] for split in splits.values()),
            "invalid_lines": sum(split["invalid_lines"] for split in splits.values()),
        },
    }


def audit_split(dataset_root: Path, split: str, class_count: int) -> dict[str, Any]:
    images_dir = dataset_root / "images" / split
    labels_dir = dataset_root / "labels" / split
    image_files = [
        path
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ] if images_dir.exists() else []
    label_files = list(labels_dir.rglob("*.txt")) if labels_dir.exists() else []

    invalid_reasons: Counter[str] = Counter()
    boxes = 0
    for label_path in label_files:
        text = label_path.read_text(encoding="utf-8").strip()
        for line in text.splitlines() if text else []:
            ok, reason = validate_yolo_line(line, class_count)
            if ok:
                boxes += 1
            else:
                invalid_reasons[reason] += 1

    return {
        "images": len(image_files),
        "labels": len(label_files),
        "boxes": boxes,
        "invalid_lines": sum(invalid_reasons.values()),
        "invalid_reasons": dict(invalid_reasons),
    }


def validate_yolo_line(line: str, class_count: int) -> tuple[bool, str]:
    parts = line.split()
    if len(parts) != 5:
        return False, "expected_5_columns"
    try:
        class_id = int(float(parts[0]))
        x_center, y_center, width, height = [float(value) for value in parts[1:]]
    except ValueError:
        return False, "non_numeric_value"
    if class_id < 0 or class_id >= class_count:
        return False, "class_id_out_of_range"
    if not all(0.0 <= value <= 1.0 for value in (x_center, y_center, width, height)):
        return False, "normalized_value_out_of_range"
    if width <= 0 or height <= 0:
        return False, "non_positive_box_size"
    if x_center - width / 2 < -1e-6 or x_center + width / 2 > 1.0 + 1e-6:
        return False, "box_exceeds_image_x"
    if y_center - height / 2 < -1e-6 or y_center + height / 2 > 1.0 + 1e-6:
        return False, "box_exceeds_image_y"
    return True, ""


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


if __name__ == "__main__":
    raise SystemExit(main())
