from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class Component:
    x1: int
    y1: int
    x2: int
    y2: int
    area: int


def load_rgb_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def image_to_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def estimate_background_rgb(array: np.ndarray) -> np.ndarray:
    top = array[0, :, :]
    bottom = array[-1, :, :]
    left = array[:, 0, :]
    right = array[:, -1, :]
    border = np.concatenate([top, bottom, left, right], axis=0)
    return np.median(border.astype(np.float32), axis=0)


def foreground_mask_from_border(array: np.ndarray, threshold: float) -> np.ndarray:
    background = estimate_background_rgb(array)
    distance = np.linalg.norm(array.astype(np.float32) - background[None, None, :], axis=2)
    return distance > threshold


def connected_components(mask: np.ndarray) -> list[Component]:
    if mask.ndim != 2:
        raise ValueError("connected_components expects a 2D mask")

    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[Component] = []

    for y in range(height):
        for x in range(width):
            if visited[y, x] or not mask[y, x]:
                continue

            queue: deque[tuple[int, int]] = deque([(x, y)])
            visited[y, x] = True
            min_x = max_x = x
            min_y = max_y = y
            area = 0

            while queue:
                cx, cy = queue.popleft()
                area += 1
                min_x = min(min_x, cx)
                max_x = max(max_x, cx)
                min_y = min(min_y, cy)
                max_y = max(max_y, cy)

                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if nx < 0 or ny < 0 or nx >= width or ny >= height:
                        continue
                    if visited[ny, nx] or not mask[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    queue.append((nx, ny))

            components.append(Component(min_x, min_y, max_x + 1, max_y + 1, area))

    return components


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory

