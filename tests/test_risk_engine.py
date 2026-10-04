"""Unit tests for the Risk Engine and Temporal State Engine."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from intelligence.distraction import DistractionResult
from intelligence.drowsiness import DrowsinessResult
from intelligence.risk_engine import RiskEngine, TemporalStateEngine
from vision.head_pose import HeadDirection, HeadPose


def _drowsy(score=0.0, micro=False, drowsy=False) -> DrowsinessResult:
    return DrowsinessResult(drowsy, micro, False, 0.0, 0.0, score)


def _distr(score=0.0, by_gaze=False, by_phone=False,
           gaze_dur=0.0, phone_dur=0.0) -> DistractionResult:
    return DistractionResult(
        distracted=by_gaze or by_phone,
        by_gaze=by_gaze, by_phone=by_phone,
        gaze_duration=gaze_dur, phone_duration=phone_dur, score=score,
    )


def _front() -> HeadPose:
    return HeadPose(0.0, 0.0, 0.0, HeadDirection.FRONT)


def _looking_left() -> HeadPose:
    return HeadPose(0.0, 45.0, 0.0, HeadDirection.LEFT)


# ---- RiskEngine ----

def test_risk_low_when_all_clear():
    e = RiskEngine()
    r = e.update(_drowsy(), _distr(), _front(), phone_detected=False)
    assert r.smoothed < 10
    assert r.risk_level == "LOW"


def test_microsleep_forces_high_risk():
    e = RiskEngine()
    r = e.update(_drowsy(score=100, micro=True), _distr(), _front(), False)
    assert r.smoothed > 80


def test_phone_plus_drowsy_forces_high():
    e = RiskEngine()
    for _ in range(20):
        r = e.update(
            _drowsy(score=80, drowsy=True),
            _distr(by_phone=True, phone_dur=2.0),
            _front(),
            phone_detected=True,
        )
    assert r.smoothed > 60


def test_ema_smoothing():
    """EMA should smooth out spikes."""
    e = RiskEngine(ema_alpha=0.1)
    for _ in range(30):
        e.update(_drowsy(), _distr(), _front(), False)

    r = e.update(
        _drowsy(score=100),
        _distr(score=100, by_gaze=True, gaze_dur=3.0),
        _looking_left(),
        False,
    )
    assert r.smoothed < r.instantaneous
    assert r.smoothed < 80


# ---- TemporalStateEngine ----

def test_state_confirm_requires_frames():
    """A single DROWSY frame should not change state immediately."""
    te = TemporalStateEngine(min_dwell_seconds=0.0)
    drowsy = _drowsy(score=100, drowsy=True)
    risk = RiskEngine().update(drowsy, _distr(), _front(), False)
    res = te.update(True, drowsy, _distr(), risk)
    assert res.state == "ALERT"
    assert res.pending_state == "DROWSY"
    assert res.confirmed is False


def test_state_commits_after_confirmation():
    te = TemporalStateEngine(min_dwell_seconds=0.0)
    drowsy = _drowsy(score=100, drowsy=True)
    re = RiskEngine()
    for _ in range(15):
        risk = re.update(drowsy, _distr(), _front(), False)
        res = te.update(True, drowsy, _distr(), risk)
    assert res.state == "DROWSY"


def test_no_driver_detected():
    """
    NO_DRIVER requires the no-face timer to exceed the timeout.
    First call arms the timer, second call (after timeout) triggers.
    """
    te = TemporalStateEngine(no_driver_timeout=0.1)
    drowsy = _drowsy()
    risk = RiskEngine().update(drowsy, _distr(), _front(), False)

    # First call: arms the no-face timer (no commit yet)
    te.update(False, drowsy, _distr(), risk)

    time.sleep(0.15)

    # Second call: timeout exceeded -> NO_DRIVER committed
    res = te.update(False, drowsy, _distr(), risk)
    assert res.state == "NO_DRIVER"


def test_transitions_counter():
    te = TemporalStateEngine(min_dwell_seconds=0.0)
    drowsy = _drowsy(score=100, drowsy=True)
    re = RiskEngine()
    for _ in range(15):
        risk = re.update(drowsy, _distr(), _front(), False)
        te.update(True, drowsy, _distr(), risk)
    assert te.total_transitions >= 1