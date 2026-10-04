"""Attention / vigilance scores (0-100) with exponential smoothing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from intelligence.distraction import DistractionResult
from intelligence.driver_state import DriverState
from intelligence.drowsiness import DrowsinessResult


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class ScoreResult:
    attention: float       # 100 = fully attentive, 0 = inattentive
    vigilance: float       # 100 = fully vigilant, 0 = asleep
    risk_level: RiskLevel


class AttentionScorer:
    """
    Combines driver state + drowsiness score + distraction score
    with exponential smoothing to avoid jumps.
    """

    def __init__(self, alpha: float = 0.15) -> None:
        self.alpha = alpha
        self._attention: float = 100.0
        self._vigilance: float = 100.0

    def update(
        self,
        state: DriverState,
        drowsy: DrowsinessResult,
        distracted: DistractionResult,
    ) -> ScoreResult:
        att_target, vig_target = self._targets(state, drowsy, distracted)

        self._attention = (1 - self.alpha) * self._attention + self.alpha * att_target
        self._vigilance = (1 - self.alpha) * self._vigilance + self.alpha * vig_target

        risk = self._risk_level(self._attention, self._vigilance)

        return ScoreResult(
            attention=round(self._attention, 1),
            vigilance=round(self._vigilance, 1),
            risk_level=risk,
        )

    @staticmethod
    def _targets(
        state: DriverState,
        drowsy: DrowsinessResult,
        distracted: DistractionResult,
    ) -> tuple[float, float]:
        if state == DriverState.CRITICAL:
            return 20.0, 15.0

        if state == DriverState.DROWSY:
            vig = max(20.0, 100.0 - drowsy.score)
            att = max(30.0, 100.0 - drowsy.score)
            return att, vig

        if state == DriverState.DISTRACTED:
            att = max(30.0, 100.0 - distracted.score)
            return att, 90.0

        if state == DriverState.NO_DRIVER:
            return 0.0, 0.0

        # ALERT : mild reduction from residual drowsiness
        return 100.0 - 0.3 * drowsy.score, 100.0 - 0.3 * drowsy.score

    @staticmethod
    def _risk_level(att: float, vig: float) -> RiskLevel:
        worst = min(att, vig)
        if worst >= 75:
            return RiskLevel.LOW
        if worst >= 50:
            return RiskLevel.MEDIUM
        if worst >= 25:
            return RiskLevel.HIGH
        return RiskLevel.CRITICAL