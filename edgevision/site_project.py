from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from edgevision.config import EdgeVisionConfig
from edgevision.image_utils import ensure_dir


PROJECT_FILE = "site_project.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class SiteProject:
    name: str
    detector_class_names: list[str] = field(default_factory=lambda: ["pill"])
    identity_detection_labels: list[str] = field(default_factory=lambda: ["pill"])
    quality_detection_labels: list[str] = field(default_factory=lambda: ["pill"])
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    )
    version: int = 1

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SiteProject":
        return cls(
            name=str(data.get("name", "site_project")),
            detector_class_names=list(data.get("detector_class_names", ["pill"])),
            identity_detection_labels=list(data.get("identity_detection_labels", ["pill"])),
            quality_detection_labels=list(data.get("quality_detection_labels", ["pill"])),
            created_at=str(data.get("created_at", "")) or datetime.now(timezone.utc).isoformat(),
            version=int(data.get("version", 1)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "created_at": self.created_at,
            "detector_class_names": self.detector_class_names,
            "identity_detection_labels": self.identity_detection_labels,
            "quality_detection_labels": self.quality_detection_labels,
        }


def init_site_project(
    project_dir: str | Path,
    name: str,
    detector_class_names: list[str] | None = None,
    identity_detection_labels: list[str] | None = None,
    quality_detection_labels: list[str] | None = None,
) -> SiteProject:
    root = ensure_dir(project_dir)
    class_names = detector_class_names or ["pill"]
    project = SiteProject(
        name=name,
        detector_class_names=class_names,
        identity_detection_labels=identity_detection_labels or ["pill"],
        quality_detection_labels=quality_detection_labels or ["pill"],
    )

    for relative in (
        "captures",
        "reference_gallery",
        "quality/OK",
        "quality/NG",
        "quality/REVIEW",
        "annotations/yolo/images/train",
        "annotations/yolo/images/val",
        "annotations/yolo/labels/train",
        "annotations/yolo/labels/val",
        "models",
        "runs",
        "exports",
    ):
        ensure_dir(root / relative)

    write_site_project(root, project)
    write_yolo_data_yaml(root, project.detector_class_names)
    write_project_readme(root, project)
    return project


def load_site_project(project_dir: str | Path) -> SiteProject:
    root = Path(project_dir)
    project_path = root / PROJECT_FILE
    if not project_path.exists():
        raise FileNotFoundError(f"Site project metadata does not exist: {project_path}")
    return SiteProject.from_dict(json.loads(project_path.read_text(encoding="utf-8")))


def write_site_project(project_dir: str | Path, project: SiteProject) -> Path:
    path = Path(project_dir) / PROJECT_FILE
    path.write_text(json.dumps(project.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def add_reference_image(
    project_dir: str | Path,
    label: str,
    image_path: str | Path,
    copy_file: bool = True,
) -> Path:
    root = Path(project_dir)
    load_site_project(root)
    source = Path(image_path)
    if not source.exists():
        raise FileNotFoundError(f"Reference image does not exist: {source}")
    if source.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported reference image extension: {source}")

    safe_label = sanitize_label(label)
    target_dir = ensure_dir(root / "reference_gallery" / safe_label)
    target = target_dir / source.name
    if copy_file:
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
    else:
        target = source
    return target


def apply_site_project_to_config(config: EdgeVisionConfig, project_dir: str | Path) -> EdgeVisionConfig:
    root = Path(project_dir)
    project = load_site_project(root)
    reference_root = root / "reference_gallery"
    if reference_root.exists():
        config.identifier.reference_root = str(reference_root)
        config.identifier.reference_manifest = None
    config.identifier.apply_to_detection_labels = project.identity_detection_labels
    config.quality.apply_to_detection_labels = project.quality_detection_labels
    return config


def summarize_site_project(project_dir: str | Path) -> dict[str, Any]:
    root = Path(project_dir)
    project = load_site_project(root)
    reference_counts: dict[str, int] = {}
    reference_root = root / "reference_gallery"
    if reference_root.exists():
        for label_dir in sorted(path for path in reference_root.iterdir() if path.is_dir()):
            reference_counts[label_dir.name] = sum(
                1
                for path in label_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )

    yolo_root = root / "annotations" / "yolo"
    yolo_counts = {
        split: {
            "images": _count_files(yolo_root / "images" / split, IMAGE_EXTENSIONS),
            "labels": _count_files(yolo_root / "labels" / split, {".txt"}),
        }
        for split in ("train", "val")
    }

    return {
        "project": project.to_dict(),
        "reference_counts": reference_counts,
        "yolo_counts": yolo_counts,
        "data_yaml": str(yolo_root / "data.yaml"),
    }


def write_yolo_data_yaml(project_dir: str | Path, class_names: list[str]) -> Path:
    root = Path(project_dir)
    yolo_root = root / "annotations" / "yolo"
    lines = [
        f"path: {yolo_root.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test:",
        "names:",
    ]
    for index, name in enumerate(class_names):
        lines.append(f"  {index}: {name}")
    lines.append("")
    path = yolo_root / "data.yaml"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_project_readme(project_dir: str | Path, project: SiteProject) -> Path:
    path = Path(project_dir) / "README.md"
    text = f"""# {project.name}

This folder is an EdgeVision site project. It stores task-specific captures,
YOLO annotations, reference images, quality examples, trained weights, and
runtime outputs.

Detector classes:

{_format_list(project.detector_class_names)}

Identity is applied to detector labels:

{_format_list(project.identity_detection_labels)}

Quality inspection is applied to detector labels:

{_format_list(project.quality_detection_labels)}
"""
    path.write_text(text, encoding="utf-8")
    return path


def sanitize_label(label: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in label.strip())
    return cleaned or "label"


def _format_list(values: list[str]) -> str:
    return "\n".join(f"- `{value}`" for value in values)


def _count_files(root: Path, suffixes: set[str]) -> int:
    if not root.exists():
        return 0
    return sum(1 for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)
