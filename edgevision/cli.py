from __future__ import annotations

import argparse
import json
from pathlib import Path

from edgevision.config import load_config
from edgevision.pipeline import PillInspectionPipeline
from edgevision.reporting import save_report_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="edgevision")
    subparsers = parser.add_subparsers(dest="command", required=True)

    infer = subparsers.add_parser("infer", help="Run pill inspection on one image.")
    infer.add_argument("--image", required=True, help="Input image path.")
    infer.add_argument("--output", required=True, help="Output directory.")
    infer.add_argument("--config", help="Optional JSON config path.")
    infer.add_argument("--reference-root", help="Override identifier.reference_root.")
    infer.add_argument("--reference-manifest", help="CSV manifest with label and image_path columns.")
    infer.add_argument("--reference-image-root", help="Root used to resolve relative manifest image_path values.")
    infer.add_argument("--detector-backend", choices=["heuristic", "yolo"], help="Override detector backend.")
    infer.add_argument("--yolo-weights", help="YOLO weights path for yolo backend.")

    return parser


def run_infer(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.reference_root:
        config.identifier.reference_root = args.reference_root
        config.identifier.reference_manifest = None
    if args.reference_manifest:
        config.identifier.reference_manifest = args.reference_manifest
        config.identifier.reference_root = None
    if args.reference_image_root:
        config.identifier.reference_image_root = args.reference_image_root
    if args.detector_backend:
        config.detector.backend = args.detector_backend
    if args.yolo_weights:
        config.detector.backend = "yolo"
        config.detector.yolo.weights_path = args.yolo_weights

    output_dir = Path(args.output)
    pipeline = PillInspectionPipeline(config)
    report = pipeline.inspect_path(args.image, output_dir=output_dir)
    save_report_json(report, output_dir)

    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "infer":
        return run_infer(args)
    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
