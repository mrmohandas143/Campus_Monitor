"""
db/crud.py
==========
Create / Read / Update / Delete operations for the Campus Monitor database.

Depends on: db/schema.py (for connection + table definitions)
            core/embedder.py (for embedding serialization)

All functions accept an open sqlite3.Connection. Callers are responsible
for opening the connection via db.schema.init_db() or get_connection().

Python 3.6+ compatible.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.embedder import FaceEmbedder

logger = logging.getLogger(__name__)

# Cooldown: minimum seconds between two attendance records for the same person
DEFAULT_COOLDOWN_SECONDS = 300  # 5 minutes


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
def register_person(
    conn: sqlite3.Connection,
    name: str,
    reg_no: str,
    embedding: np.ndarray,
    face_path: Optional[str] = None,
) -> int:
    """
    Register a new student with their face embedding.

    Args:
        conn:      Open database connection.
        name:      Student's full name.
        reg_no:    Registration/roll number (must be unique).
        embedding: 128-d float32 numpy array from FaceEmbedder.embed_face().
        face_path: Optional path to saved face crop image.

    Returns:
        Row ID of the newly inserted person.

    Raises:
        sqlite3.IntegrityError: If reg_no already exists.
    """
    blob = FaceEmbedder.serialize(embedding)
    with conn:
        cursor = conn.execute(
            "INSERT INTO persons (name, reg_no, embedding, face_path) VALUES (?, ?, ?, ?)",
            (name, reg_no, blob, face_path),
        )
    person_id = cursor.lastrowid
    logger.info("Registered: %s (%s) → id=%d", name, reg_no, person_id)
    return person_id


def update_embedding(
    conn: sqlite3.Connection,
    reg_no: str,
    embedding: np.ndarray,
) -> bool:
    """
    Update the stored embedding for an existing person (e.g. after multi-shot averaging).

    Returns True if the person was found and updated, False otherwise.
    """
    blob = FaceEmbedder.serialize(embedding)
    with conn:
        cursor = conn.execute(
            "UPDATE persons SET embedding=? WHERE reg_no=?",
            (blob, reg_no),
        )
    updated = cursor.rowcount > 0
    if updated:
        logger.info("Updated embedding for reg_no=%s", reg_no)
    return updated


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------
def get_all_persons(conn: sqlite3.Connection) -> List[Dict]:
    """
    Return all registered persons with their embeddings deserialized.

    Returns:
        List of dicts: {id, name, reg_no, embedding (np.ndarray), face_path, registered_at}
    """
    rows = conn.execute(
        "SELECT id, name, reg_no, embedding, face_path, registered_at FROM persons"
    ).fetchall()

    result = []
    for row in rows:
        result.append({
            "id": row["id"],
            "name": row["name"],
            "reg_no": row["reg_no"],
            "embedding": FaceEmbedder.deserialize(row["embedding"]),
            "face_path": row["face_path"],
            "registered_at": row["registered_at"],
        })
    return result


def get_person_by_reg_no(
    conn: sqlite3.Connection, reg_no: str
) -> Optional[Dict]:
    """Fetch a single person by registration number. Returns None if not found."""
    row = conn.execute(
        "SELECT id, name, reg_no, embedding, face_path, registered_at "
        "FROM persons WHERE reg_no=?",
        (reg_no,),
    ).fetchone()

    if row is None:
        return None
    return {
        "id": row["id"],
        "name": row["name"],
        "reg_no": row["reg_no"],
        "embedding": FaceEmbedder.deserialize(row["embedding"]),
        "face_path": row["face_path"],
        "registered_at": row["registered_at"],
    }


def person_exists(conn: sqlite3.Connection, reg_no: str) -> bool:
    """Quick existence check without fetching embedding blob."""
    row = conn.execute(
        "SELECT 1 FROM persons WHERE reg_no=?", (reg_no,)
    ).fetchone()
    return row is not None


# ---------------------------------------------------------------------------
# Attendance logging
# ---------------------------------------------------------------------------
def log_attendance(
    conn: sqlite3.Connection,
    person_id: int,
    confidence: float,
    period_name: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Log an attendance event using period-based deduplication.

    A person is logged **at most once per period per day**.  If
    ``period_name`` is None (outside all class periods), the event is
    silently skipped.

    Args:
        conn:        Open database connection.
        person_id:   ID from the persons table.
        confidence:  Cosine similarity score from the recognizer.
        period_name: Name of the current class period (e.g. "Period 1"),
                     or None when outside all periods.

    Returns:
        ``(True, reason)``  when a new row is inserted.
        ``(False, reason)`` when skipped (outside period or already logged).
    """
    if period_name is None:
        return False, "Outside class periods — not logged"

    today   = date.today().isoformat()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Was this person already logged in this period today?
    existing = conn.execute(
        """
        SELECT id, timestamp, out_time FROM attendance
        WHERE person_id   = ?
          AND session_date = ?
          AND period_name  = ?
        LIMIT 1
        """,
        (person_id, today, period_name),
    ).fetchone()

    if existing:
        existing_id = existing["id"]
        in_time_str = existing["timestamp"]

        # If the same person crosses the camera within 1 min (60s), do NOT set out_time
        try:
            in_dt  = datetime.strptime(in_time_str, "%Y-%m-%d %H:%M:%S")
            now_dt = datetime.strptime(now_str, "%Y-%m-%d %H:%M:%S")
            elapsed = (now_dt - in_dt).total_seconds()
        except Exception:
            elapsed = 0

        if elapsed < 60:
            logger.debug(
                "Skipping out_time for person_id=%d: within 1 min (%.1fs) of in-time",
                person_id, elapsed,
            )
            return False, f"Within 1 min of entry — out-time not set"

        # Update out_time if time has progressed beyond 1 min and different from current out_time
        if existing["out_time"] != now_str:
            with conn:
                conn.execute(
                    "UPDATE attendance SET out_time = ? WHERE id = ?",
                    (now_str, existing_id),
                )
            time_part = now_str.split()[1] if " " in now_str else now_str
            logger.debug(
                "Out-time updated: person_id=%d  period=%s  out_time=%s",
                person_id, period_name, now_str,
            )
            return False, f"Out-time updated ({time_part})"
        return False, f"Already logged in {period_name}"

    with conn:
        conn.execute(
            "INSERT INTO attendance "
            "(person_id, timestamp, confidence, session_date, period_name) "
            "VALUES (?, ?, ?, ?, ?)",
            (person_id, now_str, confidence, today, period_name),
        )

    logger.debug(
        "Attendance logged: person_id=%d  period=%s  confidence=%.3f",
        person_id, period_name, confidence,
    )
    return True, f"Logged in {period_name}"


def update_out_time(
    conn: sqlite3.Connection,
    person_id: int,
    out_time: Optional[str] = None,
    period_name: Optional[str] = None,
    session_date: Optional[str] = None,
) -> bool:
    """
    Update the out_time for a person's attendance record.

    If period_name is given, updates the record for that period today (or session_date).
    Otherwise updates the latest attendance record for the person on session_date (default: today).

    Args:
        conn:         Open database connection.
        person_id:    ID from the persons table.
        out_time:     Out-time string formatted as 'YYYY-MM-DD HH:MM:SS' (default: now).
        period_name:  Specific period name, or None for latest record today.
        session_date: YYYY-MM-DD date string (default: today).

    Returns:
        True if an attendance record was found and updated, False otherwise.
    """
    target_date = session_date or date.today().isoformat()
    now_str = out_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if period_name:
        row = conn.execute(
            """
            SELECT id FROM attendance
            WHERE person_id = ? AND session_date = ? AND period_name = ?
            ORDER BY id DESC LIMIT 1
            """,
            (person_id, target_date, period_name),
        ).fetchone()
    else:
        row = conn.execute(
            """
            SELECT id FROM attendance
            WHERE person_id = ? AND session_date = ?
            ORDER BY id DESC LIMIT 1
            """,
            (person_id, target_date),
        ).fetchone()

    if row is None:
        return False

    with conn:
        conn.execute(
            "UPDATE attendance SET out_time = ? WHERE id = ?",
            (now_str, row["id"]),
        )
    logger.debug("Updated out_time for person_id=%d to %s", person_id, now_str)
    return True


def log_unknown(conn: sqlite3.Connection) -> None:
    """Deprecated stub — unknown persons are now auto-registered via the engine."""
    logger.warning("log_unknown() called but unknown persons are now auto-registered.")


def log_unknown_with_cooldown(
    conn: sqlite3.Connection,
    cooldown_seconds: int = 60,
    snapshot_path: Optional[str] = None,
) -> Tuple[bool, str]:
    """Deprecated stub — unknown persons are now auto-registered via the engine."""
    logger.warning("log_unknown_with_cooldown() called but unknown persons are now auto-registered.")
    return False, "Deprecated — use engine auto-registration"


# ---------------------------------------------------------------------------
# Queries for export / reporting
# ---------------------------------------------------------------------------
def get_attendance_for_date(
    conn: sqlite3.Connection,
    session_date: Optional[str] = None,
) -> List[Dict]:
    """
    Fetch attendance records for a given date (default: today).

    Returns:
        List of dicts: {attendance_id, name, reg_no, timestamp, out_time,
                        confidence, session_date, period_name}
        Auto-registered unknowns appear with their generated name/reg_no.
    """
    target_date = session_date or date.today().isoformat()
    rows = conn.execute(
        """
        SELECT
            a.id           AS attendance_id,
            COALESCE(p.name,   'Unknown') AS name,
            COALESCE(p.reg_no, '')        AS reg_no,
            a.timestamp,
            a.out_time,
            a.confidence,
            a.session_date,
            a.period_name
        FROM attendance a
        LEFT JOIN persons p ON a.person_id = p.id
        WHERE a.session_date = ?
        ORDER BY a.timestamp ASC
        """,
        (target_date,),
    ).fetchall()

    return [dict(row) for row in rows]


def get_attendance_range(
    conn: sqlite3.Connection,
    start_date: str,
    end_date: str,
) -> List[Dict]:
    """Fetch attendance records between start_date and end_date (inclusive, YYYY-MM-DD)."""
    rows = conn.execute(
        """
        SELECT
            a.id           AS attendance_id,
            COALESCE(p.name,   'Unknown') AS name,
            COALESCE(p.reg_no, '')        AS reg_no,
            a.timestamp,
            a.out_time,
            a.confidence,
            a.session_date,
            a.period_name
        FROM attendance a
        LEFT JOIN persons p ON a.person_id = p.id
        WHERE a.session_date BETWEEN ? AND ?
        ORDER BY a.session_date ASC, a.timestamp ASC
        """,
        (start_date, end_date),
    ).fetchall()

    return [dict(row) for row in rows]


def get_daily_summary(
    conn: sqlite3.Connection,
    session_date: Optional[str] = None,
) -> Dict:
    """
    Summarize today's attendance.

    Returns:
        {date, total_present, total_unknown, records: [...]}
        ``total_unknown`` counts auto-registered unknown persons (reg_no LIKE 'UNK_%').
    """
    target_date = session_date or date.today().isoformat()
    records = get_attendance_for_date(conn, target_date)

    unknown = [r for r in records if r["reg_no"].startswith("UNK_")]
    known   = [r for r in records if not r["reg_no"].startswith("UNK_")]

    # Deduplicate by reg_no (first appearance = arrival time, latest out_time)
    seen = {}
    for r in known:
        reg = r["reg_no"]
        if reg not in seen:
            seen[reg] = dict(r)
        else:
            latest_time = r.get("out_time") or r.get("timestamp")
            if latest_time:
                seen[reg]["out_time"] = latest_time
    unique_known = list(seen.values())

    return {
        "date": target_date,
        "total_present": len(unique_known),
        "total_unknown": len(unknown),
        "records": unique_known,
    }


def get_unknown_alerts_for_date(
    conn: sqlite3.Connection,
    session_date: Optional[str] = None,
) -> List[Dict]:
    """Fetch auto-registered unknown person records for a given date.

    Unknown persons are identified by ``reg_no LIKE 'UNK_%'``.

    Returns:
        List of dicts: {attendance_id, name, reg_no, timestamp, session_date,
                        period_name, snapshot_path}
    """
    target_date = session_date or date.today().isoformat()
    rows = conn.execute(
        """
        SELECT
            a.id           AS attendance_id,
            p.name,
            p.reg_no,
            a.timestamp,
            a.session_date,
            a.period_name,
            p.face_path    AS snapshot_path
        FROM attendance a
        JOIN persons p ON a.person_id = p.id
        WHERE p.reg_no LIKE 'UNK_%'
          AND a.session_date = ?
        ORDER BY a.timestamp ASC
        """,
        (target_date,),
    ).fetchall()
    return [dict(row) for row in rows]


def get_unknown_alerts_range(
    conn: sqlite3.Connection,
    start_date: str,
    end_date: str,
) -> List[Dict]:
    """Fetch auto-registered unknown person records between start_date and end_date (inclusive)."""
    rows = conn.execute(
        """
        SELECT
            a.id           AS attendance_id,
            p.name,
            p.reg_no,
            a.timestamp,
            a.session_date,
            a.period_name,
            p.face_path    AS snapshot_path
        FROM attendance a
        JOIN persons p ON a.person_id = p.id
        WHERE p.reg_no LIKE 'UNK_%'
          AND a.session_date BETWEEN ? AND ?
        ORDER BY a.session_date ASC, a.timestamp ASC
        """,
        (start_date, end_date),
    ).fetchall()
    return [dict(row) for row in rows]
