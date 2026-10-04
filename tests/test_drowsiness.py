"""Unit tests for the drowsiness detector."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from intelligence.drowsiness import DrowsinessDetector
from vision.eye_analysis import EyeMetrics, EyeState


def _eye(state: EyeState, mar: float = 0.10) -> EyeMetrics:
    ear = 0.30 if state == EyeState.OPEN else 0.10
    return EyeMetrics(ear, ear, ear, state, mar, mar > 0.40)


def test_open_eyes_not_drowsy():
    d = DrowsinessDetector()
    r = d.update(_eye(EyeState.OPEN))
    assert r.drowsy is False
    assert r.microsleep is False
    assert r.closed_duration == 0.0


def test_closed_eyes_trigger_drowsy():
    d = DrowsinessDetector(closed_threshold=0.1, microsleep_threshold=0.3)
    d.update(_eye(EyeState.CLOSED))
    time.sleep(0.15)
    r = d.update(_eye(EyeState.CLOSED))
    assert r.drowsy is True
    assert r.microsleep is False


def test_microsleep_triggered():
    d = DrowsinessDetector(closed_threshold=0.05, microsleep_threshold=0.15)
    d.update(_eye(EyeState.CLOSED))
    time.sleep(0.20)
    r = d.update(_eye(EyeState.CLOSED))
    assert r.microsleep is True


def test_closed_duration_resets():
    """
    Test that closed_duration resets to 0 when eyes reopen.
    PERCLOS is disabled in this test (set very high) so we isolate
    the 'continuous closed duration' rule.
    """
    d = DrowsinessDetector(
        closed_threshold=10.0,     # disable duration-based drowsy
        microsleep_threshold=20.0,
        perclos_window=30.0,
        perclos_threshold=0.99,    # disable PERCLOS rule
    )
    d.update(_eye(EyeState.CLOSED))
    time.sleep(0.05)
    d.update(_eye(EyeState.OPEN))
    r = d.update(_eye(EyeState.OPEN))
    assert r.closed_duration == 0.0
    assert r.drowsy is False


def test_yawn_detection():
    d = DrowsinessDetector(yawn_window=0.5)
    for _ in range(5):
        d.update(_eye(EyeState.OPEN, mar=0.7))
    time.sleep(0.1)
    r = d.update(_eye(EyeState.OPEN, mar=0.7))
    assert r.yawning is True


def test_perclos_accumulates():
    """Alternating/continuous closed samples should trigger PERCLOS drowsy."""
    d = DrowsinessDetector(
        closed_threshold=10.0,     # disable duration rule
        microsleep_threshold=20.0,
        perclos_window=1.0,
        perclos_threshold=0.30,
    )
    for _ in range(20):
        d.update(_eye(EyeState.CLOSED))
        time.sleep(0.02)
    r = d.update(_eye(EyeState.CLOSED))
    assert r.perclos > 0.3
    assert r.drowsy is True