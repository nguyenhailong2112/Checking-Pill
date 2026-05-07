from __future__ import annotations

import numpy as np
from PIL import Image


def run_image_precheck(image: Image.Image) -> dict[str, float | bool | str]:
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    height, width = gray.shape
    contrast = float(gray.std())

    dx = np.abs(np.diff(gray, axis=1)).mean() if width > 1 else 0.0
    dy = np.abs(np.diff(gray, axis=0)).mean() if height > 1 else 0.0
    sharpness_proxy = float((dx + dy) / 2.0)

    usable = width >= 64 and height >= 64 and contrast >= 4.0
    reason = "ok" if usable else "image_too_small_or_low_contrast"

    return {
        "usable": bool(usable),
        "reason": reason,
        "width": float(width),
        "height": float(height),
        "contrast": contrast,
        "sharpness_proxy": sharpness_proxy,
    }

