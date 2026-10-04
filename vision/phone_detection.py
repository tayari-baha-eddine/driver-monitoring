"""Phone detection using YOLOv8 (Ultralytics) on GPU."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class PhoneDetection:
    bbox: tuple[int, int, int, int]   # (x1, y1, x2, y2)
    confidence: float


class PhoneDetector:
    """
    Detects 'cell phone' class (COCO id=67) with YOLOv8.

    Automatically uses CUDA if available, otherwise CPU.
    """

    PHONE_CLASS_NAMES = {"cell phone", "phone", "mobile phone"}

    def __init__(
        self,
        model_path: str = "models/pytorch/yolov8n.pt",
        conf: float = 0.4,
        iou: float = 0.5,
        device: str | None = None,
    ) -> None:
        # Auto-select device
        if device is None:
            device = "0" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Ensure parent directory exists
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)

        # Load model (downloads yolov8n.pt if missing)
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou

    def detect(self, frame_bgr: np.ndarray) -> list[PhoneDetection]:
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
            device=self.device,
        )[0]

        detections: list[PhoneDetection] = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            name = self.model.names.get(cls_id, "")
            if name in self.PHONE_CLASS_NAMES:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append(PhoneDetection(
                    bbox=(x1, y1, x2, y2),
                    confidence=float(box.conf[0]),
                ))
        return detections

    def annotate(
        self,
        frame_bgr: np.ndarray,
        detections: list[PhoneDetection],
        color: tuple[int, int, int] = (0, 0, 255),
    ) -> None:
        """Draw bounding boxes + labels on the frame."""
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
            label = f"PHONE {det.confidence:.2f}"
            cv2.putText(
                frame_bgr, label,
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
            )