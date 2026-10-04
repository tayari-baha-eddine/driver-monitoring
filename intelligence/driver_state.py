"""Driver state machine: combines drowsiness + distraction signals."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

from intelligence.distraction import DistractionResult
from intelligence.drowsiness import DrowsinessResult


class DriverState(str, Enum):
    ALERT = "ALERT"
    DROWSY = "DROWSY"
    DISTRACTED = "DISTRACTED"
    CRITICAL = "CRITICAL"
    NO_DRIVER = "NO_DRIVER"


@dataclass
class DriverStateResult:
    state: DriverState
    reason: str


class DriverStateEngine:
    """
    Priority (most critical first):
      1. NO_DRIVER  : no face for >= no_driver_timeout seconds
      2. CRITICAL   : microsleep OR confirmed phone usage
      3. DROWSY     : drowsiness detected
      4. DISTRACTED : gaze away sustained
      5. ALERT      : everything OK
    """

    def __init__(self, no_driver_timeout: float = 2.0) -> None:
        self.no_driver_timeout = no_driver_timeout
        self._no_face_start: float | None = None

    def update(
        self,
        face_present: bool,
        drowsy: DrowsinessResult,
        distracted: DistractionResult,
    ) -> DriverStateResult:
        now = time.time()

        # NO_DRIVER
        if not face_present:
            if self._no_face_start is None:
                self._no_face_start = now
            elif now - self._no_face_start >= self.no_driver_timeout:
                return DriverStateResult(
                    DriverState.NO_DRIVER,
                    "Aucun conducteur detecte",
                )
        else:
            self._no_face_start = None

        # CRITICAL
        if drowsy.microsleep:
            return DriverStateResult(
                DriverState.CRITICAL,
                f"Micro-sommeil ({drowsy.closed_duration:.1f}s)",
            )
        if distracted.by_phone:
            return DriverStateResult(
                DriverState.CRITICAL,
                "Telephone au volant",
            )

        # DROWSY
        if drowsy.drowsy:
            return DriverStateResult(
                DriverState.DROWSY,
                f"Fermeture yeux {drowsy.closed_duration:.1f}s",
            )

        # DISTRACTED
        if distracted.distracted:
            if distracted.by_gaze:
                return DriverStateResult(
                    DriverState.DISTRACTED,
                    f"Regard detourne {distracted.gaze_duration:.1f}s",
                )

        return DriverStateResult(DriverState.ALERT, "")