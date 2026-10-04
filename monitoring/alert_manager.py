"""Alert manager: audio + visual alerts based on driver state."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pygame

from intelligence.driver_state import DriverState

logger = logging.getLogger(__name__)


ALERT_COLORS: dict[DriverState, tuple[int, int, int]] = {
    DriverState.ALERT:      (0, 200, 0),      # green
    DriverState.DISTRACTED: (0, 200, 255),    # yellow
    DriverState.DROWSY:     (0, 140, 255),    # orange
    DriverState.CRITICAL:   (0, 0, 255),      # red
    DriverState.NO_DRIVER:  (128, 128, 128),  # gray
}


class AlertManager:
    """
    - Plays 'warning' sound for DROWSY / DISTRACTED
    - Plays 'critical' sound for CRITICAL
    - Cooldown prevents repeated beeps
    """

    def __init__(
        self,
        warning_sound: str = "assets/sounds/warning.wav",
        critical_sound: str = "assets/sounds/critical.wav",
        cooldown: float = 3.0,
    ) -> None:
        self.cooldown = cooldown
        self._last_alert: dict[DriverState, float] = {}
        self._sounds: dict[str, pygame.mixer.Sound] = {}
        self._enabled = False

        try:
            pygame.mixer.init()
            self._enabled = True
            self._load("warning", warning_sound)
            self._load("critical", critical_sound)
        except Exception as e:
            logger.warning(f"Audio disabled: {e}")

    def _load(self, key: str, path: str) -> None:
        p = Path(path)
        if p.exists():
            self._sounds[key] = pygame.mixer.Sound(str(p))
        else:
            logger.warning(f"Sound not found: {path}")

    def trigger(self, state: DriverState) -> None:
        if not self._enabled:
            return

        key = self._sound_key(state)
        if key is None:
            return

        now = time.time()
        if now - self._last_alert.get(state, 0.0) < self.cooldown:
            return

        if key in self._sounds:
            self._sounds[key].play()
            self._last_alert[state] = now
            logger.info(f"[ALERT] {state.value}")

    @staticmethod
    def _sound_key(state: DriverState) -> str | None:
        if state == DriverState.CRITICAL:
            return "critical"
        if state in (DriverState.DROWSY, DriverState.DISTRACTED):
            return "warning"
        return None

    @staticmethod
    def color_for(state: DriverState) -> tuple[int, int, int]:
        return ALERT_COLORS.get(state, (255, 255, 255))

    def close(self) -> None:
        if self._enabled:
            pygame.mixer.quit()