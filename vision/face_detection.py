"""Face detection + 468 landmarks via MediaPipe FaceMesh."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class FaceResult:
    landmarks: np.ndarray               # (468, 2) pixel coords
    landmarks_norm: np.ndarray          # (468, 2) normalized [0,1]
    bbox: tuple[int, int, int, int]     # (x1, y1, x2, y2)
    confidence: float


class FaceDetector:
    """Wrapper MediaPipe FaceMesh with structured output."""

    def __init__(
        self,
        max_faces: int = 1,
        refine_landmarks: bool = True,
        detection_conf: float = 0.5,
        tracking_conf: float = 0.5,
    ) -> None:
        self._mp_face_mesh = mp.solutions.face_mesh
        self._mesh = self._mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )

    def process(self, frame_bgr: np.ndarray) -> Optional[FaceResult]:
        h, w = frame_bgr.shape[:2]

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        face = results.multi_face_landmarks[0]
        pts_norm = np.array([[lm.x, lm.y] for lm in face.landmark], dtype=np.float32)
        pts_px = (pts_norm * np.array([w, h], dtype=np.float32)).astype(np.int32)

        x1, y1 = pts_px.min(axis=0)
        x2, y2 = pts_px.max(axis=0)

        return FaceResult(
            landmarks=pts_px,
            landmarks_norm=pts_norm,
            bbox=(int(x1), int(y1), int(x2), int(y2)),
            confidence=1.0,
        )

    def close(self) -> None:
        self._mesh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()