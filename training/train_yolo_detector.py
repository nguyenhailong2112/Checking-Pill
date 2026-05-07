from __future__ import annotations

import argparse
from pathlib import Path


def ensure_data_yaml(dataset_root: Path, output_path: Path) -> Path:
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
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a one-class pill YOLO detector.")
    parser.add_argument("--dataset-root", required=True, help="YOLO dataset root containing images/ and labels/.")
    parser.add_argument("--data-yaml", help="Optional existing YOLO data.yaml path.")
    parser.add_argument("--model", default="yolo11s.pt", help="YOLO model checkpoint/name, e.g. yolo11s.pt.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=832)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None, help="Ultralytics device string, e.g. 0 or cpu.")
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="pill_detector_yolo")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exist-ok", action="store_true")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is required. In Colab run: pip install -U ultralytics"
        ) from exc

    dataset_root = Path(args.dataset_root).resolve()
    data_yaml = Path(args.data_yaml).resolve() if args.data_yaml else ensure_data_yaml(
        dataset_root,
        Path(args.project) / args.name / "data.yaml",
    )

    model = YOLO(args.model)
    train_kwargs = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "patience": args.patience,
        "seed": args.seed,
        "exist_ok": args.exist_ok,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device

    model.train(**train_kwargs)

    output_dir = Path(args.project) / args.name
    print(f"Training completed. Check weights under: {output_dir / 'weights'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

