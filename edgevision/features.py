from __future__ import annotations

import numpy as np
from PIL import Image

from edgevision.image_utils import foreground_mask_from_border, image_to_array


def extract_pill_feature(image: Image.Image, bins: int = 8) -> np.ndarray:
    array = image_to_array(image)
    mask = foreground_mask_from_border(array, threshold=28)
    pixels = array[mask]

    if len(pixels) < 16:
        pixels = array.reshape(-1, 3)

    hist_parts = []
    for channel in range(3):
        hist, _ = np.histogram(pixels[:, channel], bins=bins, range=(0, 256), density=True)
        hist_parts.append(hist.astype(np.float32))

    foreground_ratio = np.array([mask.mean()], dtype=np.float32)
    ys, xs = np.where(mask)
    if len(xs) > 0:
        width = max(float(xs.max() - xs.min() + 1), 1.0)
        height = max(float(ys.max() - ys.min() + 1), 1.0)
        aspect = np.array([width / height], dtype=np.float32)
    else:
        aspect = np.array([1.0], dtype=np.float32)

    mean_rgb = pixels.mean(axis=0).astype(np.float32) / 255.0
    std_rgb = pixels.std(axis=0).astype(np.float32) / 255.0

    feature = np.concatenate(hist_parts + [foreground_ratio, aspect, mean_rgb, std_rgb])
    norm = np.linalg.norm(feature)
    if norm > 1e-12:
        feature = feature / norm
    return feature.astype(np.float32)

