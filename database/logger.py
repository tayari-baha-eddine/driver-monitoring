"""SQLite event logger with driver identity and risk metrics."""

from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

DEFAULT_DB = "data/driver_events.db"

# Tables only (no indexes here — indexes are created AFTER migrations)
SCHEMA_TABLES = """
CREATE TABLE IF NOT EXISTS events (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts             REAL    NOT NULL,
    state          TEXT    NOT NULL,
    reason         TEXT,
    attention      REAL,
    vigilance      REAL,
    drowsiness     REAL,
    distraction    REAL,
    risk_instant   REAL,
    risk_smoothed  REAL,
    ear            REAL,
    mar            REAL,
    yaw            REAL,
    pitch          REAL,
    roll           REAL,
    phone          INTEGER DEFAULT 0,
    fps            REAL,
    latency_ms     REAL,
    driver         TEXT    DEFAULT 'unknown'
);
"""

SCHEMA_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_events_ts     ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_state  ON events(state);
CREATE INDEX IF NOT EXISTS idx_events_driver ON events(driver);
"""

# Columns that might be missing on old databases (idempotent migrations)
MIGRATIONS = [
    ("driver",        "ALTER TABLE events ADD COLUMN driver TEXT DEFAULT 'unknown'"),
    ("risk_instant",  "ALTER TABLE events ADD COLUMN risk_instant REAL DEFAULT 0"),
    ("risk_smoothed", "ALTER TABLE events ADD COLUMN risk_smoothed REAL DEFAULT 0"),
]


@dataclass
class Event:
    ts: float
    state: str
    reason: str = ""
    attention: float = 0.0
    vigilance: float = 0.0
    drowsiness: float = 0.0
    distraction: float = 0.0
    risk_instant: float = 0.0
    risk_smoothed: float = 0.0
    ear: float = 0.0
    mar: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    phone: int = 0
    fps: float = 0.0
    latency_ms: float = 0.0
    driver: str = "unknown"


class EventLogger:
    """Writes to SQLite on state change OR every N seconds."""

    def __init__(
        self,
        db_path: str = DEFAULT_DB,
        periodic_interval: float = 5.0,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.periodic_interval = periodic_interval
        self._last_state: str | None = None
        self._last_log_ts: float = 0.0
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as con:
            # 1) Create table if missing (no index yet)
            con.executescript(SCHEMA_TABLES)

            # 2) Run migrations FIRST (add missing columns)
            cols = {row[1] for row in
                    con.execute("PRAGMA table_info(events)").fetchall()}
            for col, sql in MIGRATIONS:
                if col not in cols:
                    con.execute(sql)

            # 3) Only NOW create indexes (safe: columns exist)
            con.executescript(SCHEMA_INDEXES)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(str(self.db_path))
        try:
            yield con
            con.commit()
        finally:
            con.close()

    def log(self, event: Event, force: bool = False) -> bool:
        now = event.ts or time.time()
        transition = event.state != self._last_state
        periodic = now - self._last_log_ts >= self.periodic_interval

        if not (force or transition or periodic):
            return False

        self._insert(event)
        self._last_state = event.state
        self._last_log_ts = now
        return True

    def _insert(self, event: Event) -> None:
        data = asdict(event)
        cols = ", ".join(data.keys())
        placeholders = ", ".join(["?"] * len(data))
        with self._connect() as con:
            con.execute(
                f"INSERT INTO events ({cols}) VALUES ({placeholders})",
                tuple(data.values()),
            )

    def fetch_recent(self, limit: int = 500) -> list[dict]:
        with self._connect() as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                "SELECT * FROM events ORDER BY ts DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        with self._connect() as con:
            total = con.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            by_state = dict(con.execute(
                "SELECT state, COUNT(*) FROM events GROUP BY state"
            ).fetchall())
            by_driver = dict(con.execute(
                "SELECT driver, COUNT(*) FROM events GROUP BY driver"
            ).fetchall())
        return {"total": total, "by_state": by_state, "by_driver": by_driver}