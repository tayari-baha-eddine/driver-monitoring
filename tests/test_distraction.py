"""Unit tests for the distraction detector."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from intelligence.distraction import DistractionDetector
from vision.head_pose import HeadDirection, HeadPose


def _pose(direction: HeadDirection) -> HeadPose:
    yaw = pitch = 0.0
    if direction == HeadDirection.LEFT:
        yaw = 40.0
    elif direction == HeadDirection.RIGHT:
        yaw = -40.0
    elif direction == HeadDirection.UP:
        pitch = 30.0
    elif direction == HeadDirection.DOWN:
        pitch = -30.0
    return HeadPose(pitch, yaw, 0.0, direction)


def test_front_not_distracted():
    d = DistractionDetector()
    r = d.update(_pose(HeadDirection.FRONT), False)
    assert r.distracted is False
    assert r.by_gaze is False
    assert r.by_phone is False


def test_gaze_away_triggers():
    d = DistractionDetector(gaze_threshold=0.1)
    d.update(_pose(HeadDirection.RIGHT), False)
    time.sleep(0.15)
    r = d.update(_pose(HeadDirection.RIGHT), False)
    assert r.by_gaze is True
    assert r.distracted is True


def test_phone_triggers():
    d = DistractionDetector(phone_threshold=0.1)
    d.update(_pose(HeadDirection.FRONT), True)
    time.sleep(0.15)
    r = d.update(_pose(HeadDirection.FRONT), True)
    assert r.by_phone is True
    assert r.distracted is True


def test_gaze_resets_when_looking_front():
    d = DistractionDetector(gaze_threshold=5.0)
    d.update(_pose(HeadDirection.LEFT), False)
    time.sleep(0.05)
    d.update(_pose(HeadDirection.FRONT), False)
    r = d.update(_pose(HeadDirection.FRONT), False)
    assert r.gaze_duration == 0.0
    assert r.by_gaze is False


def test_both_signals_combine():
    d = DistractionDetector(gaze_threshold=0.1, phone_threshold=0.1)
    d.update(_pose(HeadDirection.RIGHT), True)
    time.sleep(0.15)
    r = d.update(_pose(HeadDirection.RIGHT), True)
    assert r.by_gaze is True
    assert r.by_phone is True
    assert r.score > 50