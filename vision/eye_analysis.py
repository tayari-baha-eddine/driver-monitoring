"""Eye and mouth analysis: EAR, MAR, open/closed state."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


# ============================================================
# MediaPipe FaceMesh landmark indices (468 points)
# ============================================================

# Eye landmarks (6 points per eye, standard EAR formula)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Mouth landmarks for MAR (inner lips, standard formula)
MOUTH_LEFT_CORNER = 61
MOUTH_RIGHT_CORNER = 291
MOUTH_TOP_INNER = 13
MOUTH_BOTTOM_INNER = 14
MOUTH_TOP_INNER_2 = 81
MOUTH_BOTTOM_INNER_2 = 311
MOUTH_TOP_INNER_3 = 82
MOUTH_BOTTOM_INNER_3 = 312


class EyeState(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    UNKNOWN = "UNKNOWN"


@dataclass
class EyeMetrics:
    ear_left: float
    ear_right: float
    ear_avg: float
    state: EyeState
    mar: float = 0.0
    yawning: bool = False


# ============================================================
# Helper functions
# ============================================================

def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def compute_ear(landmarks: np.ndarray, indices: list[int]) -> float:
    """
    Eye Aspect Ratio.
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Open eyes  : EAR ~= 0.25 - 0.40
    Closed eyes: EAR ~= 0.05 - 0.18
    """
    p = landmarks[indices].astype(np.float32)
    v1 = _dist(p[1], p[5])
    v2 = _dist(p[2], p[4])
    h = _dist(p[0], p[3]) + 1e-6
    return (v1 + v2) / (2.0 * h)


def compute_mar(landmarks: np.ndarray) -> float:
    """
    Mouth Aspect Ratio (standard formula using inner lips).

    MAR = (d(13,14) + d(81,311) + d(82,312)) / (2 * d(61,291))

    Mouth closed : MAR ~= 0.05 - 0.20
    Mouth open   : MAR ~= 0.40 - 1.00
    """
    p = landmarks.astype(np.float32)

    # Vertical distances between inner lip top/bottom pairs
    v1 = _dist(p[MOUTH_TOP_INNER], p[MOUTH_BOTTOM_INNER])
    v2 = _dist(p[MOUTH_TOP_INNER_2], p[MOUTH_BOTTOM_INNER_2])
    v3 = _dist(p[MOUTH_TOP_INNER_3], p[MOUTH_BOTTOM_INNER_3])

    # Horizontal distance between mouth corners
    h = _dist(p[MOUTH_LEFT_CORNER], p[MOUTH_RIGHT_CORNER])

    return (v1 + v2 + v3) / (2.0 * h + 1e-6)


# ============================================================
# Main analyzer
# ============================================================

class EyeAnalyzer:
    """
    Computes EAR/MAR and classifies eye state.
    Auto-calibrates baseline EAR from the first N frames
    (assumes eyes are open during calibration).
    """

    def __init__(
        self,
        ear_threshold: float = 0.20,
        mar_threshold: float = 0.40,
        calibration_frames: int = 60,
    ) -> None:
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        self._calibration_frames = calibration_frames
        self._calibration_buffer: deque[float] = deque(maxlen=calibration_frames)
        self._calibrated = False
        self._baseline_ear: Optional[float] = None

    def calibrate(self, ear: float) -> bool:
        """Collect samples to compute a personalized EAR threshold."""
        if self._calibrated:
            return True

        self._calibration_buffer.append(ear)

        if len(self._calibration_buffer) >= self._calibration_frames:
            self._baseline_ear = float(np.median(self._calibration_buffer))
            # Closed-eye threshold at 75% of baseline EAR
            self.ear_threshold = self._baseline_ear * 0.75
            self._calibrated = True

        return self._calibrated

    def analyze(self, landmarks: np.ndarray) -> EyeMetrics:
        ear_l = compute_ear(landmarks, LEFT_EYE)
        ear_r = compute_ear(landmarks, RIGHT_EYE)
        ear = (ear_l + ear_r) / 2.0
        mar = compute_mar(landmarks)

        if not self._calibrated:
            self.calibrate(ear)

        state = EyeState.CLOSED if ear < self.ear_threshold else EyeState.OPEN

        return EyeMetrics(
            ear_left=round(ear_l, 3),
            ear_right=round(ear_r, 3),
            ear_avg=round(ear, 3),
            state=state,
            mar=round(mar, 3),
            yawning=mar > self.mar_threshold,
        )

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    @property
    def baseline_ear(self) -> Optional[float]:
        return self._baseline_ear