"""Performance monitor: FPS, latency, CPU, RAM."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False


@dataclass
class PerformanceStats:
    fps: float = 0.0
    latency_ms: float = 0.0
    cpu_percent: float = 0.0
    ram_mb: float = 0.0
    frames: int = 0


class PerformanceMonitor:
    """Rolling FPS / latency / CPU / RAM monitor."""

    def __init__(self, window: int = 30) -> None:
        self._intervals: deque[float] = deque(maxlen=window)
        self._stage_times: dict[str, deque[float]] = {}
        self._last_t: float = time.perf_counter()
        self._frames: int = 0
        self._process = psutil.Process() if _PSUTIL else None

    def tick(self) -> None:
        now = time.perf_counter()
        self._intervals.append(now - self._last_t)
        self._last_t = now
        self._frames += 1

    def stage(self, name: str):
        return _StageTimer(self, name)

    def _record_stage(self, name: str, dt: float) -> None:
        self._stage_times.setdefault(name, deque(maxlen=30)).append(dt)

    def stats(self) -> PerformanceStats:
        avg_dt = (sum(self._intervals) / len(self._intervals)) if self._intervals else 0.0
        fps = (1.0 / avg_dt) if avg_dt > 0 else 0.0
        cpu = self._process.cpu_percent(interval=None) if self._process else 0.0
        ram = (self._process.memory_info().rss / 1024 / 1024) if self._process else 0.0
        return PerformanceStats(
            fps=round(fps, 1),
            latency_ms=round(avg_dt * 1000, 1),
            cpu_percent=round(cpu, 1),
            ram_mb=round(ram, 1),
            frames=self._frames,
        )

    def stage_stats(self) -> dict[str, float]:
        return {
            name: round((sum(v) / len(v)) * 1000, 2)
            for name, v in self._stage_times.items() if v
        }


class _StageTimer:
    def __init__(self, monitor: PerformanceMonitor, name: str) -> None:
        self.monitor = monitor
        self.name = name

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.monitor._record_stage(self.name, time.perf_counter() - self.t0)