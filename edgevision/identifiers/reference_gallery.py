from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from edgevision.config import IdentifierConfig
from edgevision.features import extract_pill_feature
from edgevision.image_utils import cosine_similarity
from edgevision.schemas import Candidate, IdentificationResult


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ReferenceGalleryIdentifier:
    """Nearest-neighbor pill identifier from folder-based references.

    Expected layout:

    ```text
    reference_root/
      Panadol Extra/
        ref_1.jpg
        ref_2.jpg
      Aspirin/
        ref_1.jpg
    ```
    """

    def __init__(self, config: IdentifierConfig | None = None):
        self.config = config or IdentifierConfig()
        self.prototypes: dict[str, np.ndarray] = {}
        if self.config.reference_manifest:
            self.load_reference_manifest(
                self.config.reference_manifest,
                image_root=self.config.reference_image_root,
                reference_only=self.config.reference_only,
            )
        elif self.config.reference_root:
            self.load_reference_root(self.config.reference_root)

    def load_reference_root(self, reference_root: str | Path) -> None:
        root = Path(reference_root)
        if not root.exists():
            raise FileNotFoundError(f"Reference root does not exist: {root}")

        features_by_label: dict[str, list[np.ndarray]] = defaultdict(list)
        for image_path in root.rglob("*"):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if image_path.parent == root:
                label = image_path.stem
            else:
                label = image_path.parent.name
            with Image.open(image_path) as image:
                features_by_label[label].append(extract_pill_feature(image.convert("RGB")))

        self.prototypes = {
            label: np.mean(np.stack(features), axis=0)
            for label, features in features_by_label.items()
            if features
        }
        for label, prototype in self.prototypes.items():
            norm = np.linalg.norm(prototype)
            if norm > 1e-12:
                self.prototypes[label] = prototype / norm

    def load_reference_manifest(
        self,
        manifest_path: str | Path,
        image_root: str | Path | None = None,
        reference_only: bool = True,
    ) -> None:
        manifest = Path(manifest_path)
        if not manifest.exists():
            raise FileNotFoundError(f"Reference manifest does not exist: {manifest}")

        root = Path(image_root) if image_root is not None else manifest.parent
        features_by_label: dict[str, list[np.ndarray]] = defaultdict(list)

        with manifest.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            required = {"label", "image_path"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Reference manifest is missing columns: {sorted(missing)}")

            for row in reader:
                if reference_only and row.get("is_ref") not in {None, "", "True", "true", "1", "1.0"}:
                    continue

                label = row["label"]
                image_path = Path(row["image_path"])
                if not image_path.is_absolute():
                    image_path = root / image_path
                if not image_path.exists():
                    continue

                with Image.open(image_path) as image:
                    features_by_label[label].append(extract_pill_feature(image.convert("RGB")))

        self.prototypes = {
            label: np.mean(np.stack(features), axis=0)
            for label, features in features_by_label.items()
            if features
        }
        for label, prototype in self.prototypes.items():
            norm = np.linalg.norm(prototype)
            if norm > 1e-12:
                self.prototypes[label] = prototype / norm

    def identify(self, crop: Image.Image) -> IdentificationResult:
        if not self.prototypes:
            return IdentificationResult(
                label="Unknown",
                confidence=0.0,
                candidates=[],
                is_unknown=True,
            )

        feature = extract_pill_feature(crop)
        scored = []
        for label, prototype in self.prototypes.items():
            similarity = cosine_similarity(feature, prototype)
            confidence = max(0.0, min(1.0, (similarity + 1.0) / 2.0))
            scored.append(Candidate(label=label, confidence=confidence, distance=1.0 - similarity))

        scored.sort(key=lambda candidate: candidate.confidence, reverse=True)
        candidates = scored[: self.config.top_k]
        best = candidates[0]
        is_unknown = best.confidence < self.config.unknown_threshold
        return IdentificationResult(
            label="Unknown" if is_unknown else best.label,
            confidence=best.confidence,
            candidates=candidates,
            is_unknown=is_unknown,
        )
