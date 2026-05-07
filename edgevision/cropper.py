from __future__ import annotations

from pathlib import Path

from PIL import Image

from edgevision.config import CropperConfig
from edgevision.image_utils import ensure_dir
from edgevision.schemas import Detection


class CropNormalizer:
    def __init__(self, config: CropperConfig | None = None):
        self.config = config or CropperConfig()

    def crop(self, image: Image.Image, detection: Detection) -> Image.Image:
        bbox = detection.bbox.padded(
            self.config.padding_ratio,
            width=image.width,
            height=image.height,
        )
        crop = image.crop(tuple(bbox.to_int_list()))
        return crop.resize((self.config.output_size, self.config.output_size), Image.Resampling.LANCZOS)

    def maybe_save(self, crop: Image.Image, output_dir: str | Path | None, item_id: int) -> str | None:
        if not self.config.save_crops or output_dir is None:
            return None

        crop_dir = ensure_dir(Path(output_dir) / "crops")
        crop_path = crop_dir / f"pill_{item_id:04d}.jpg"
        crop.save(crop_path, quality=95)
        return str(crop_path)

