"""Unit tests for the attention scorer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from intelligence.distraction import DistractionResult
from intelligence.driver_state import DriverState
from intelligence.drowsiness import DrowsinessResult
from intelligence.scoring import AttentionScorer


def _drowsy(score: float = 0.0, micro: bool = False) -> DrowsinessResult:
    return DrowsinessResult(
        drowsy=micro or score > 50,
        microsleep=micro,
        yawning=False,
        closed_duration=0.0,
        perclos=0.0,
        score=score,
    )


def _distr(score: float = 0.0, by_phone: bool = False) -> DistractionResult:
    return DistractionResult(
        distracted=score > 50 or by_phone,
        by_gaze=not by_phone and score > 50,
        by_phone=by_phone,
        gaze_duration=0.0,
        phone_duration=0.0,
        score=score,
    )


def test_alert_full_attention():
    s = AttentionScorer(alpha=1.0)
    r = s.update(DriverState.ALERT, _drowsy(0), _distr(0))
    assert r.attention == 100.0
    assert r.vigilance == 100.0
    assert r.risk_level.value == "LOW"


def test_critical_drops_score():
    s = AttentionScorer(alpha=1.0)
    r = s.update(DriverState.CRITICAL, _drowsy(100, True), _distr(100, True))
    assert r.attention < 50
    assert r.vigilance < 50
    assert r.risk_level.value in ("HIGH", "CRITICAL")


def test_drowsy_reduces_vigilance():
    s = AttentionScorer(alpha=1.0)
    r = s.update(DriverState.DROWSY, _drowsy(80), _distr(0))
    assert r.vigilance < 50
    assert r.attention < 100


def test_no_driver_zero_scores():
    s = AttentionScorer(alpha=1.0)
    r = s.update(DriverState.NO_DRIVER, _drowsy(0), _distr(0))
    assert r.attention == 0.0
    assert r.vigilance == 0.0
    assert r.risk_level.value == "CRITICAL"


def test_smoothing_is_gradual():
    """With small alpha, score shouldn't jump to target immediately."""
    s = AttentionScorer(alpha=0.1)
    # Feed 1 frame of ALERT (target 100) starting from 100
    s.update(DriverState.ALERT, _drowsy(0), _distr(0))
    # Then 1 frame of CRITICAL (target 20) - should NOT drop instantly
    r = s.update(DriverState.CRITICAL, _drowsy(100, True), _distr(100, True))
    assert r.attention > 80  # hasn't dropped to 20 yet