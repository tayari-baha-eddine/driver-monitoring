"""Risk Engine: weighted multi-signal fusion + temporal stabilization.

Transforms instantaneous detector outputs into a stable, temporally-confirmed
driver state using:

  1. Weighted fusion of all signals into a continuous risk score (0-100)
  2. Exponential Moving Average (EMA) smoothing
  3. Threshold-based classification with hysteresis
  4. Temporal confirmation (state must persist N frames to be confirmed)
  5. Minimum dwell time (state cannot change faster than X seconds)
  6. Per-state confirmation speed (CRITICAL confirms fast, ALERT confirms slow)

This turns a set of independent detectors into a real driver monitoring algorithm.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from intelligence.distraction import DistractionResult
from intelligence.drowsiness import DrowsinessResult
from intelligence.driver_state import DriverState
from vision.head_pose import HeadPose


# ============================================================
# Risk components & result
# ============================================================

@dataclass
class RiskComponents:
    drowsiness: float = 0.0     # 0-1
    gaze: float = 0.0           # 0-1
    head_pose: float = 0.0      # 0-1
    phone: float = 0.0          # 0-1
    microsleep: float = 0.0     # 0-1


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class RiskResult:
    instantaneous: float        # 0-100 (before smoothing)
    smoothed: float             # 0-100 (after EMA)
    components: RiskComponents
    risk_level: str


# ============================================================
# Risk Engine (signal fusion + smoothing)
# ============================================================

class RiskEngine:
    """
    Fuses all signals into a single smoothed risk score (0-100).

    Formula:
        raw = Σ w_i · c_i         (weighted sum of components)
        risk_smoothed(t) = α·raw + (1-α)·risk_smoothed(t-1)

    Critical overrides:
        - microsleep   -> risk forced >= microsleep_override
        - phone + drowsy -> risk forced >= phone_override

    Default weights (auto-normalized):
        drowsiness : 0.35
        gaze       : 0.20
        head_pose  : 0.15
        phone      : 0.30
    """

    DEFAULT_WEIGHTS = {
        "drowsiness": 0.35,
        "gaze":       0.20,
        "head_pose":  0.15,
        "phone":      0.30,
    }

    def __init__(
        self,
        weights: Optional[dict[str, float]] = None,
        ema_alpha: float = 0.20,
        microsleep_override: float = 0.95,
        phone_override: float = 0.80,
        history_size: int = 300,
    ) -> None:
        weights = dict(weights or self.DEFAULT_WEIGHTS)
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        self.weights = weights

        self.ema_alpha = ema_alpha
        self.microsleep_override = microsleep_override
        self.phone_override = phone_override

        self._smoothed: float = 0.0
        self._raw: float = 0.0
        self._initialized = False
        self._history: deque[float] = deque(maxlen=history_size)

    # ------------------------------------------------------------------

    def update(
        self,
        drowsy: DrowsinessResult,
        distracted: DistractionResult,
        pose: HeadPose,
        phone_detected: bool,
    ) -> RiskResult:
        components = self._compute_components(drowsy, distracted, pose, phone_detected)

        # Weighted fusion
        raw = (
            self.weights["drowsiness"] * components.drowsiness
            + self.weights["gaze"] * components.gaze
            + self.weights["head_pose"] * components.head_pose
            + self.weights["phone"] * components.phone
        )

        # Critical overrides (safety-critical signals dominate)
        if drowsy.microsleep:
            raw = max(raw, self.microsleep_override)
        if phone_detected and drowsy.drowsy:
            raw = max(raw, self.phone_override)

        raw_score = min(100.0, raw * 100.0)

        # EMA smoothing
        if not self._initialized:
            self._smoothed = raw_score
            self._initialized = True
        else:
            self._smoothed = (
                self.ema_alpha * raw_score
                + (1.0 - self.ema_alpha) * self._smoothed
            )

        self._raw = raw_score
        self._history.append(self._smoothed)

        return RiskResult(
            instantaneous=round(raw_score, 1),
            smoothed=round(self._smoothed, 1),
            components=components,
            risk_level=self._level(self._smoothed).value,
        )

    # ------------------------------------------------------------------

    def _compute_components(
        self,
        drowsy: DrowsinessResult,
        distracted: DistractionResult,
        pose: HeadPose,
        phone_detected: bool,
    ) -> RiskComponents:
        # Drowsiness: direct from score (0-100 -> 0-1)
        c_drow = min(1.0, drowsy.score / 100.0)

        # Gaze: duration-based, saturates at 3s
        gaze_dur = distracted.gaze_duration if distracted.by_gaze else 0.0
        c_gaze = min(1.0, gaze_dur / 3.0)

        # Head pose: yaw + pitch magnitude (relative to neutral 0)
        head_mag = (abs(pose.yaw) + abs(pose.pitch)) / 90.0
        c_head = min(1.0, head_mag)

        # Phone: boolean with a 2s duration ramp
        if phone_detected:
            phone_dur = distracted.phone_duration if distracted.by_phone else 0.5
            c_phone = min(1.0, phone_dur / 2.0)
        else:
            c_phone = 0.0

        # Microsleep: binary
        c_micro = 1.0 if drowsy.microsleep else 0.0

        return RiskComponents(
            drowsiness=round(c_drow, 3),
            gaze=round(c_gaze, 3),
            head_pose=round(c_head, 3),
            phone=round(c_phone, 3),
            microsleep=round(c_micro, 3),
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _level(score: float) -> RiskLevel:
        if score >= 70.0:
            return RiskLevel.CRITICAL
        if score >= 45.0:
            return RiskLevel.HIGH
        if score >= 20.0:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    @property
    def smoothed(self) -> float:
        return self._smoothed

    @property
    def raw(self) -> float:
        return self._raw

    @property
    def history(self) -> list[float]:
        return list(self._history)


# ============================================================
# Temporal State Engine (hysteresis + confirmation)
# ============================================================

@dataclass
class TemporalStateResult:
    state: str
    reason: str
    pending_state: str          # state currently being confirmed (or "")
    pending_progress: float     # 0-1
    risk_smoothed: float
    risk_instant: float
    confirmed: bool


class TemporalStateEngine:
    """
    Converts raw signals + risk score into a stable driver state.

    Per-state confirmation speed (frames required to switch):
        CRITICAL   :  4  (fastest - safety critical)
        NO_DRIVER  :  5  (fast - don't miss driver absence)
        DROWSY     : 10
        DISTRACTED : 10
        ALERT      : 20  (slowest - avoid flickering back to ALERT)

    Plus a min_dwell_seconds floor to avoid rapid state changes.
    """

    CONFIRM_FRAMES = {
        "CRITICAL":   4,
        "NO_DRIVER":  5,
        "DROWSY":    10,
        "DISTRACTED": 10,
        "ALERT":     20,
    }

    def __init__(
        self,
        confirm_frames: Optional[dict[str, int]] = None,
        min_dwell_seconds: float = 1.0,
        no_driver_timeout: float = 2.0,
    ) -> None:
        self.confirm_frames = confirm_frames or dict(self.CONFIRM_FRAMES)
        self.min_dwell_seconds = min_dwell_seconds
        self.no_driver_timeout = no_driver_timeout

        # State tracking
        self._state: DriverState = DriverState.ALERT
        self._reason: str = ""
        self._state_started_at: float = time.time()

        # Pending transition
        self._pending: Optional[DriverState] = None
        self._pending_count: int = 0
        self._pending_reason: str = ""

        # No-driver tracking
        self._no_face_start: Optional[float] = None

        # Stats
        self._total_transitions: int = 0

    # ------------------------------------------------------------------

    def update(
        self,
        face_present: bool,
        drowsy: DrowsinessResult,
        distracted: DistractionResult,
        risk: RiskResult,
        now: Optional[float] = None,
    ) -> TemporalStateResult:
        now = now or time.time()

        # ---------- NO_DRIVER (fast path, bypasses hysteresis) ----------
        if not face_present:
            if self._no_face_start is None:
                self._no_face_start = now
            if now - self._no_face_start >= self.no_driver_timeout:
                self._commit(DriverState.NO_DRIVER, "Aucun conducteur detecte", now)
                return self._make_result(risk, confirmed=True)
        else:
            self._no_face_start = None

        # ---------- Instantaneous classification ----------
        raw_state, raw_reason = self._classify(drowsy, distracted, risk)

        # ---------- Temporal confirmation ----------
        if raw_state != self._state:
            if self._pending != raw_state:
                self._pending = raw_state
                self._pending_count = 0
                self._pending_reason = raw_reason
            self._pending_count += 1

            dwell = now - self._state_started_at
            required = self.confirm_frames.get(raw_state.value, 10)

            if (self._pending_count >= required
                    and dwell >= self.min_dwell_seconds):
                self._commit(raw_state, raw_reason, now)
        else:
            # Raw state agrees with current -> cancel pending, refresh reason
            self._pending = None
            self._pending_count = 0
            self._reason = raw_reason

        return self._make_result(risk, confirmed=self._pending is None)

    # ------------------------------------------------------------------

    @staticmethod
    def _classify(
        drowsy: DrowsinessResult,
        distracted: DistractionResult,
        risk: RiskResult,
    ) -> tuple[DriverState, str]:
        """Instantaneous classification (no temporal logic)."""
        # Priority order: CRITICAL > DROWSY > DISTRACTED > ALERT
        if drowsy.microsleep:
            return (DriverState.CRITICAL,
                    f"Micro-sommeil ({drowsy.closed_duration:.1f}s)")

        if distracted.by_phone:
            return (DriverState.CRITICAL, "Telephone au volant")

        if risk.risk_level == "CRITICAL":
            return (DriverState.CRITICAL, f"Risque critique ({risk.smoothed:.0f})")

        if drowsy.drowsy:
            return (DriverState.DROWSY,
                    f"Fermeture yeux {drowsy.closed_duration:.1f}s")

        if distracted.by_gaze:
            return (DriverState.DISTRACTED,
                    f"Regard detourne {distracted.gaze_duration:.1f}s")

        if risk.risk_level == "HIGH":
            return (DriverState.DISTRACTED,
                    f"Risque eleve ({risk.smoothed:.0f})")

        return (DriverState.ALERT, "")

    # ------------------------------------------------------------------

    def _commit(self, new_state: DriverState, reason: str, now: float) -> None:
        """Commit a state transition."""
        if new_state == self._state:
            self._reason = reason
            return
        self._state = new_state
        self._reason = reason
        self._state_started_at = now
        self._pending = None
        self._pending_count = 0
        self._pending_reason = ""
        self._total_transitions += 1

    def _make_result(self, risk: RiskResult, confirmed: bool) -> TemporalStateResult:
        pending = self._pending.value if self._pending else ""
        required = (self.confirm_frames.get(self._pending.value, 10)
                    if self._pending else 1)
        progress = (self._pending_count / required
                    if self._pending and required > 0 else 1.0)

        return TemporalStateResult(
            state=self._state.value,
            reason=self._reason,
            pending_state=pending,
            pending_progress=round(min(1.0, progress), 2),
            risk_smoothed=risk.smoothed,
            risk_instant=risk.instantaneous,
            confirmed=confirmed,
        )

    # ------------------------------------------------------------------

    @property
    def state(self) -> DriverState:
        return self._state

    @property
    def total_transitions(self) -> int:
        return self._total_transitions

    @property
    def state_age(self) -> float:
        return time.time() - self._state_started_at