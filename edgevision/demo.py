from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from edgevision.config import EdgeVisionConfig
from edgevision.image_utils import ensure_dir
from edgevision.pipeline import PillInspectionPipeline
from edgevision.reporting import save_report_json
from edgevision.schemas import InspectionReport


def create_demo_assets(root: str | Path) -> tuple[Path, Path]:
    """Create a deterministic pill scene and a tiny reference gallery."""

    root_path = ensure_dir(root)
    input_dir = ensure_dir(root_path / "input")
    reference_root = ensure_dir(root_path / "reference_gallery")

    _create_reference_image(
        reference_root / "RoundCream" / "ref_001.jpg",
        shape="ellipse",
        fill=(236, 211, 170),
    )
    _create_reference_image(
        reference_root / "BlueCaplet" / "ref_001.jpg",
        shape="caplet",
        fill=(115, 188, 230),
    )

    scene_path = input_dir / "demo_pills.jpg"
    scene = Image.new("RGB", (760, 460), (130, 130, 130))
    draw = ImageDraw.Draw(scene)

    draw.ellipse((90, 125, 225, 260), fill=(236, 211, 170))
    draw.ellipse((270, 170, 405, 305), fill=(236, 211, 170))
    draw.rounded_rectangle((500, 145, 670, 250), radius=48, fill=(115, 188, 230))

    defect = [(585, 165), (654, 180), (646, 226), (595, 236), (558, 210)]
    draw.polygon(defect, fill=(28, 34, 38))
    scene.save(scene_path, quality=95)

    return scene_path, reference_root


def run_demo(output_dir: str | Path) -> InspectionReport:
    output_root = ensure_dir(output_dir)
    scene_path, reference_root = create_demo_assets(output_root / "assets")

    config = EdgeVisionConfig()
    config.detector.backend = "heuristic"
    config.detector.heuristic.foreground_threshold = 34
    config.detector.heuristic.min_area_ratio = 0.004
    config.identifier.reference_root = str(reference_root)
    config.identifier.unknown_threshold = 0.84
    config.quality.max_dark_spot_ratio = 0.05
    config.quality.review_score_threshold = 0.80

    inference_dir = ensure_dir(output_root / "inference")
    pipeline = PillInspectionPipeline(config)
    report = pipeline.inspect_path(scene_path, output_dir=inference_dir)
    save_report_json(report, inference_dir)
    return report


def _create_reference_image(path: Path, shape: str, fill: tuple[int, int, int]) -> None:
    ensure_dir(path.parent)
    image = Image.new("RGB", (224, 224), (130, 130, 130))
    draw = ImageDraw.Draw(image)
    if shape == "ellipse":
        draw.ellipse((45, 45, 179, 179), fill=fill)
    elif shape == "caplet":
        draw.rounded_rectangle((32, 72, 192, 152), radius=40, fill=fill)
    else:
        raise ValueError(f"Unsupported demo shape: {shape}")
    image.save(path, quality=95)
