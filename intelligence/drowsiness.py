"""Drowsiness detection: EAR duration + PERCLOS + yawning."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

from vision.eye_analysis import EyeMetrics, EyeState


@dataclass
class DrowsinessResult:
    drowsy: bool
    microsleep: bool
    yawning: bool
    closed_duration: float
    perclos: float          # ratio of closed frames in recent window
    score: float            # 0-100


class DrowsinessDetector:
    """
    Rules:
      - Eyes CLOSED for >= closed_threshold (1.5s)   -> DROWSY
      - Eyes CLOSED for >= microsleep_threshold (3s) -> MICROSLEEP
      - PERCLOS >= perclos_threshold over 30s window -> DROWSY
      - MAR > threshold sustained                     -> YAWNING
    """

    def __init__(
        self,
        closed_threshold: float = 1.5,
        microsleep_threshold: float = 3.0,
        perclos_window: float = 30.0,
        perclos_threshold: float = 0.30,
        yawn_window: float = 3.0,
        yawn_ratio_threshold: float = 0.5,
    ) -> None:
        self.closed_threshold = closed_threshold
        self.microsleep_threshold = microsleep_threshold
        self.perclos_window = perclos_window
        self.perclos_threshold = perclos_threshold
        self.yawn_window = yawn_window
        self.yawn_ratio_threshold = yawn_ratio_threshold

        self._closed_start: float | None = None
        self._closed_duration: float = 0.0
        self._samples: deque[tuple[float, bool]] = deque()
        self._yawn_samples: deque[tuple[float, bool]] = deque()

    def update(self, eye: EyeMetrics) -> DrowsinessResult:
        now = time.time()

        # Track continuous closed duration
        if eye.state == EyeState.CLOSED:
            if self._closed_start is None:
                self._closed_start = now
            self._closed_duration = now - self._closed_start
        else:
            self._closed_start = None
            self._closed_duration = 0.0

        # PERCLOS buffer
        self._samples.append((now, eye.state == EyeState.CLOSED))
        self._trim(self._samples, now, self.perclos_window)
        closed_count = sum(1 for _, c in self._samples if c)
        perclos = closed_count / max(len(self._samples), 1)

        # Yawn buffer
        self._yawn_samples.append((now, eye.yawning))
        self._trim(self._yawn_samples, now, self.yawn_window)
        yawn_ratio = (
            sum(1 for _, y in self._yawn_samples if y)
            / max(len(self._yawn_samples), 1)
        )

        drowsy = (
            self._closed_duration >= self.closed_threshold
            or perclos >= self.perclos_threshold
        )
        microsleep = self._closed_duration >= self.microsleep_threshold

        score = min(
            100.0,
            60.0 * min(self._closed_duration / self.microsleep_threshold, 1.0)
            + 40.0 * min(perclos / self.perclos_threshold, 1.0),
        )

        return DrowsinessResult(
            drowsy=drowsy,
            microsleep=microsleep,
            yawning=yawn_ratio > self.yawn_ratio_threshold,
            closed_duration=round(self._closed_duration, 2),
            perclos=round(perclos, 3),
            score=round(score, 1),
        )

    @staticmethod
    def _trim(buf: deque, now: float, window: float) -> None:
        while buf and now - buf[0][0] > window:
            buf.popleft()