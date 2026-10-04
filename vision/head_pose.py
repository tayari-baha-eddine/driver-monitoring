"""Head pose estimation (yaw / pitch / roll) via solvePnP.

Includes a neutral-pose calibration step to compensate for the
constant pitch/yaw offset that solvePnP produces on MediaPipe landmarks.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np


# 3D facial model points (generic face, millimeters)
MODEL_POINTS_3D = np.array([
    (0.0,    0.0,    0.0),      # 0 - nose tip
    (0.0,  -63.6,  -12.5),      # 1 - chin
    (-43.3, 32.7,  -26.0),      # 2 - left eye outer corner
    (43.3,  32.7,  -26.0),      # 3 - right eye outer corner
    (-28.9, -28.9, -24.1),      # 4 - mouth left corner
    (28.9,  -28.9, -24.1),      # 5 - mouth right corner
], dtype=np.float64)

# Corresponding MediaPipe FaceMesh landmark indices
LANDMARK_IDS = [1, 152, 33, 263, 61, 291]


class HeadDirection(str, Enum):
    FRONT = "FRONT"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    UP = "UP"
    DOWN = "DOWN"


@dataclass
class HeadPose:
    pitch: float
    yaw: float
    roll: float
    direction: HeadDirection


class HeadPoseEstimator:
    """
    Estimates head orientation using solvePnP with automatic
    neutral-pose calibration.

    Calibration workflow:
      1. The first `calibration_frames` frames are used to learn the
         neutral pose (assumes the driver looks straight ahead).
      2. All subsequent angles are reported RELATIVE to that neutral pose.
      3. Direction classification uses threshold on the corrected angles.
    """

    def __init__(
        self,
        yaw_threshold: float = 25.0,
        pitch_threshold: float = 20.0,
        calibration_frames: int = 60,
    ) -> None:
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self._calibration_frames = calibration_frames

        # Calibration buffers (pitch and yaw)
        self._pitch_buffer: deque[float] = deque(maxlen=calibration_frames)
        self._yaw_buffer: deque[float] = deque(maxlen=calibration_frames)
        self._calibrated = False
        self._pitch_offset: float = 0.0
        self._yaw_offset: float = 0.0

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def _feed_calibration(self, raw_pitch: float, raw_yaw: float) -> bool:
        """Accumulate samples for the neutral-pose reference."""
        if self._calibrated:
            return True

        self._pitch_buffer.append(raw_pitch)
        self._yaw_buffer.append(raw_yaw)

        if len(self._pitch_buffer) >= self._calibration_frames:
            self._pitch_offset = float(np.median(self._pitch_buffer))
            self._yaw_offset = float(np.median(self._yaw_buffer))
            self._calibrated = True

        return self._calibrated

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(
        self,
        landmarks: np.ndarray,
        frame_shape: tuple[int, int, int],
    ) -> HeadPose:
        h, w = frame_shape[:2]
        image_points = landmarks[LANDMARK_IDS].astype(np.float64)

        # Camera intrinsics (focal length approximated as image width)
        focal = float(w)
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array([
            [focal, 0.0,   center[0]],
            [0.0,   focal, center[1]],
            [0.0,   0.0,   1.0],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        ok, rvec, _ = cv2.solvePnP(
            MODEL_POINTS_3D,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return HeadPose(0.0, 0.0, 0.0, HeadDirection.FRONT)

        rmat, _ = cv2.Rodrigues(rvec)
        angles, *_ = cv2.RQDecomp3x3(rmat)
        raw_pitch, raw_yaw, roll = (float(a) for a in angles)

        # Feed calibration until ready
        self._feed_calibration(raw_pitch, raw_yaw)

        # Apply offsets (relative pose)
        pitch = raw_pitch - self._pitch_offset
        yaw = raw_yaw - self._yaw_offset

        direction = self._classify(yaw, pitch)

        return HeadPose(
            pitch=round(pitch, 2),
            yaw=round(yaw, 2),
            roll=round(roll, 2),
            direction=direction,
        )

    # ------------------------------------------------------------------
    # Direction classification
    # ------------------------------------------------------------------

    def _classify(self, yaw: float, pitch: float) -> HeadDirection:
        """
        Convention after calibration:
          - Turning head to the RIGHT  -> yaw decreases  (negative)
          - Turning head to the LEFT   -> yaw increases  (positive)
          - Looking UP                 -> pitch increases (positive)
          - Looking DOWN               -> pitch decreases (negative)
        """
        # Yaw sign fixed for MediaPipe + this model
        if yaw > self.yaw_threshold:
            return HeadDirection.RIGHT
        if yaw < -self.yaw_threshold:
            return HeadDirection.LEFT

        # Pitch
        if pitch > self.pitch_threshold:
            return HeadDirection.UP
        if pitch < -self.pitch_threshold:
            return HeadDirection.DOWN

        return HeadDirection.FRONT

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    @property
    def pitch_offset(self) -> float:
        return self._pitch_offset

    @property
    def yaw_offset(self) -> float:
        return self._yaw_offset