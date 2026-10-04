"""Distraction detection: gaze away + phone usage."""

from __future__ import annotations

import time
from dataclasses import dataclass

from vision.head_pose import HeadDirection, HeadPose


@dataclass
class DistractionResult:
    distracted: bool
    by_gaze: bool
    by_phone: bool
    gaze_duration: float
    phone_duration: float
    score: float  # 0-100


class DistractionDetector:
    """
    Triggers distraction when:
      - Head direction != FRONT sustained for gaze_threshold seconds
      - Phone detected sustained for phone_threshold seconds
    """

    def __init__(
        self,
        gaze_threshold: float = 1.0,
        phone_threshold: float = 1.0,
    ) -> None:
        self.gaze_threshold = gaze_threshold
        self.phone_threshold = phone_threshold

        self._gaze_start: float | None = None
        self._phone_start: float | None = None

    def update(
        self,
        pose: HeadPose,
        phone_detected: bool,
    ) -> DistractionResult:
        now = time.time()

        # Gaze away
        if pose.direction != HeadDirection.FRONT:
            if self._gaze_start is None:
                self._gaze_start = now
        else:
            self._gaze_start = None

        # Phone
        if phone_detected:
            if self._phone_start is None:
                self._phone_start = now
        else:
            self._phone_start = None

        gaze_dur = now - self._gaze_start if self._gaze_start else 0.0
        phone_dur = now - self._phone_start if self._phone_start else 0.0

        by_gaze = gaze_dur >= self.gaze_threshold
        by_phone = phone_dur >= self.phone_threshold

        score = min(
            100.0,
            50.0 * min(gaze_dur / self.gaze_threshold, 1.5)
            + 50.0 * min(phone_dur / self.phone_threshold, 1.5),
        )

        return DistractionResult(
            distracted=by_gaze or by_phone,
            by_gaze=by_gaze,
            by_phone=by_phone,
            gaze_duration=round(gaze_dur, 2),
            phone_duration=round(phone_dur, 2),
            score=round(score, 1),
        )