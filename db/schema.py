"""
db/schema.py
============
SQLite schema initialization for the Campus Monitor system.

Tables:
  persons    — registered students (name, reg_no, embedding)
  attendance — timestamped recognition events

Python 3.6+ compatible. Uses only stdlib sqlite3.
"""

from __future__ import annotations

import sqlite3
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default database location relative to project root
_PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_DB_PATH = _PROJECT_ROOT / "data" / "attendance.db"

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------
_CREATE_PERSONS = """
CREATE TABLE IF NOT EXISTS persons (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL,
    reg_no        TEXT NOT NULL UNIQUE,
    embedding     BLOB NOT NULL,
    face_path     TEXT,
    registered_at TEXT DEFAULT (datetime('now', 'localtime'))
);
"""

# NOTE: SQLite < 3.31 does not support GENERATED ALWAYS AS.
# session_date is stored as a plain TEXT column and set explicitly in crud.py.
_CREATE_ATTENDANCE = """
CREATE TABLE IF NOT EXISTS attendance (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id     INTEGER REFERENCES persons(id) ON DELETE SET NULL,
    timestamp     TEXT DEFAULT (datetime('now', 'localtime')),
    confidence    REAL,
    session_date  TEXT,
    snapshot_path TEXT,
    period_name   TEXT
);
"""

# Migrations for databases created before this version
_MIGRATE_ATTENDANCE_SNAPSHOT = """
ALTER TABLE attendance ADD COLUMN snapshot_path TEXT;
"""
_MIGRATE_ATTENDANCE_PERIOD = """
ALTER TABLE attendance ADD COLUMN period_name TEXT;
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_persons_reg_no ON persons(reg_no);",
    "CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance(session_date);",
    "CREATE INDEX IF NOT EXISTS idx_attendance_person ON attendance(person_id);",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """
    Open (and initialize) the SQLite database.

    Args:
        db_path: Path to the .db file. Uses data/attendance.db by default.

    Returns:
        sqlite3.Connection with row_factory set to sqlite3.Row (dict-like rows).
    """
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")   # safer concurrent writes
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db(db_path: Optional[str] = None) -> sqlite3.Connection:
    """
    Create tables and indexes if they don't exist.

    Safe to call multiple times (uses CREATE IF NOT EXISTS).

    Args:
        db_path: Path to the .db file.

    Returns:
        Open sqlite3.Connection ready for use.
    """
    conn = get_connection(db_path)
    with conn:
        conn.execute(_CREATE_PERSONS)
        conn.execute(_CREATE_ATTENDANCE)
        for idx_sql in _CREATE_INDEXES:
            conn.execute(idx_sql)
        # Migrations for older database versions
        try:
            conn.execute(_MIGRATE_ATTENDANCE_SNAPSHOT)
        except Exception:
            pass  # Column already exists
        try:
            conn.execute(_MIGRATE_ATTENDANCE_PERIOD)
        except Exception:
            pass  # Column already exists
    logger.info("Database initialized at %s", db_path or DEFAULT_DB_PATH)
    return conn
