from __future__ import annotations

from pathlib import Path

from PIL import Image

from edgevision.config import YoloDetectorConfig
from edgevision.schemas import BBox, Detection


class UltralyticsYOLODetector:
    """YOLO detector adapter.

    This adapter is intentionally optional. Install `ultralytics` and provide a
    weights path before using it.
    """

    def __init__(self, config: YoloDetectorConfig):
        if config.weights_path is None:
            raise ValueError("YOLO backend requires detector.yolo.weights_path")

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "The ultralytics package is required for YOLO inference. "
                "Install EdgeVision with the 'yolo' extra or install ultralytics."
            ) from exc

        self.config = config
        self.model = YOLO(str(Path(config.weights_path)))

    def detect(self, image: Image.Image) -> list[Detection]:
        results = self.model.predict(
            image,
            imgsz=self.config.image_size,
            device=self.config.device,
            verbose=False,
        )
        detections: list[Detection] = []

        for result in results:
            if result.boxes is None:
                continue
            xyxy = result.boxes.xyxy.cpu().numpy()
            conf = result.boxes.conf.cpu().numpy()
            cls = result.boxes.cls.cpu().numpy()
            for box, score, class_id in zip(xyxy, conf, cls):
                label = "pill" if int(class_id) == 0 else str(int(class_id))
                detections.append(
                    Detection(
                        bbox=BBox(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                        confidence=float(score),
                        label=label,
                        source="ultralytics_yolo",
                    )
                )

        return detections

