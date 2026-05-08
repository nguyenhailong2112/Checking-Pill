from __future__ import annotations

import argparse
import json
from pathlib import Path

from edgevision.batch import run_batch
from edgevision.config import load_config
from edgevision.demo import run_demo
from edgevision.pipeline import PillInspectionPipeline
from edgevision.reporting import save_report_json
from edgevision.site_project import (
    add_reference_image,
    apply_site_project_to_config,
    init_site_project,
    summarize_site_project,
)


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="Optional JSON config path.")
    parser.add_argument("--site-project", help="Optional site project folder.")
    parser.add_argument("--reference-root", help="Override identifier.reference_root.")
    parser.add_argument("--reference-manifest", help="CSV manifest with label and image_path columns.")
    parser.add_argument("--reference-image-root", help="Root used to resolve relative manifest image_path values.")
    parser.add_argument("--detector-backend", choices=["auto", "heuristic", "yolo"], help="Override detector backend.")
    parser.add_argument("--yolo-weights", help="YOLO weights path for yolo backend.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="edgevision")
    subparsers = parser.add_subparsers(dest="command", required=True)

    infer = subparsers.add_parser("infer", help="Run pill inspection on one image.")
    infer.add_argument("--image", required=True, help="Input image path.")
    infer.add_argument("--output", required=True, help="Output directory.")
    add_runtime_arguments(infer)

    batch = subparsers.add_parser("batch", help="Run pill inspection on a folder of images.")
    batch.add_argument("--input", required=True, help="Input image file or image folder.")
    batch.add_argument("--output", required=True, help="Output directory.")
    add_runtime_arguments(batch)

    demo = subparsers.add_parser("demo", help="Create and run a deterministic pill demo.")
    demo.add_argument("--output", required=True, help="Output directory.")

    project_init = subparsers.add_parser("project-init", help="Create a site project folder.")
    project_init.add_argument("--path", required=True, help="Project folder to create.")
    project_init.add_argument("--name", default="edgevision_site", help="Human-readable project name.")
    project_init.add_argument(
        "--classes",
        nargs="+",
        default=["pill"],
        help="Detector class names, for example: pill bottle blister thermometer syringe.",
    )
    project_init.add_argument(
        "--identity-labels",
        nargs="+",
        default=["pill"],
        help="Detector labels that should run pill/type identification.",
    )
    project_init.add_argument(
        "--quality-labels",
        nargs="+",
        default=["pill"],
        help="Detector labels that should run OK/NG quality inspection.",
    )

    project_add_reference = subparsers.add_parser(
        "project-add-reference",
        help="Copy a labeled reference image into a site project gallery.",
    )
    project_add_reference.add_argument("--project", required=True, help="Site project folder.")
    project_add_reference.add_argument("--label", required=True, help="Reference label/type name.")
    project_add_reference.add_argument("--image", required=True, help="Reference image path.")

    project_summary = subparsers.add_parser("project-summary", help="Summarize a site project.")
    project_summary.add_argument("--project", required=True, help="Site project folder.")

    return parser


def build_pipeline_from_args(args: argparse.Namespace) -> PillInspectionPipeline:
    config = load_config(args.config)
    if getattr(args, "site_project", None):
        config = apply_site_project_to_config(config, args.site_project)
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
    return PillInspectionPipeline(config)


def run_infer(args: argparse.Namespace) -> int:
    output_dir = Path(args.output)
    pipeline = build_pipeline_from_args(args)
    report = pipeline.inspect_path(args.image, output_dir=output_dir)
    save_report_json(report, output_dir)

    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0


def run_batch_command(args: argparse.Namespace) -> int:
    pipeline = build_pipeline_from_args(args)
    reports = run_batch(pipeline, input_path=args.input, output_dir=args.output)
    print(
        json.dumps(
            {
                "input": args.input,
                "output": args.output,
                "images": len(reports),
                "total_pills": sum(report.total_count for report in reports),
                "total_objects": sum(report.total_count for report in reports),
                "summary_csv": str(Path(args.output) / "summary.csv"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def run_demo_command(args: argparse.Namespace) -> int:
    report = run_demo(args.output)
    print(
        json.dumps(
            {
                "output": args.output,
                "image_path": report.image_path,
                "total_count": report.total_count,
                "count_by_type": report.count_by_type,
                "image_status": report.image_status.value,
                "report_json": str(Path(args.output) / "inference" / "report.json"),
                "annotated_image": str(Path(args.output) / "inference" / "annotated.jpg"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def run_project_init(args: argparse.Namespace) -> int:
    project = init_site_project(
        args.path,
        name=args.name,
        detector_class_names=args.classes,
        identity_detection_labels=args.identity_labels,
        quality_detection_labels=args.quality_labels,
    )
    print(
        json.dumps(
            {
                "project_path": args.path,
                "project": project.to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def run_project_add_reference(args: argparse.Namespace) -> int:
    target = add_reference_image(args.project, label=args.label, image_path=args.image)
    print(
        json.dumps(
            {
                "project_path": args.project,
                "label": args.label,
                "reference_image": str(target),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def run_project_summary(args: argparse.Namespace) -> int:
    print(json.dumps(summarize_site_project(args.project), ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "infer":
        return run_infer(args)
    if args.command == "batch":
        return run_batch_command(args)
    if args.command == "demo":
        return run_demo_command(args)
    if args.command == "project-init":
        return run_project_init(args)
    if args.command == "project-add-reference":
        return run_project_add_reference(args)
    if args.command == "project-summary":
        return run_project_summary(args)
    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
