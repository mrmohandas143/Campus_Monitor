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
    cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS,
) -> Tuple[bool, str]:
    """
    Log an attendance event, respecting the cooldown window.

    If the same person was already logged within `cooldown_seconds`,
    this call is a no-op (returns False).

    Args:
        conn:             Open database connection.
        person_id:        ID from persons table.
        confidence:       Cosine similarity score from recognizer.
        cooldown_seconds: Minimum gap between records for the same person.

    Returns:
        Tuple (logged: bool, reason: str).
        logged=True  → new record inserted.
        logged=False → within cooldown, record skipped.
    """
    today = date.today().isoformat()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Check cooldown: was this person logged in the last N seconds?
    recent = conn.execute(
        """
        SELECT timestamp FROM attendance
        WHERE person_id=?
          AND session_date=?
        ORDER BY timestamp DESC
        LIMIT 1
        """,
        (person_id, today),
    ).fetchone()

    if recent:
        last_ts = datetime.strptime(recent["timestamp"], "%Y-%m-%d %H:%M:%S")
        elapsed = (datetime.now() - last_ts).total_seconds()
        if elapsed < cooldown_seconds:
            return False, f"Cooldown active ({elapsed:.0f}s / {cooldown_seconds}s)"

    with conn:
        conn.execute(
            "INSERT INTO attendance (person_id, timestamp, confidence, session_date) "
            "VALUES (?, ?, ?, ?)",
            (person_id, now_str, confidence, today),
        )

    logger.debug("Attendance logged: person_id=%d  confidence=%.3f", person_id, confidence)
    return True, "Logged"


def log_unknown(conn: sqlite3.Connection) -> None:
    """Log an unrecognized person (person_id=NULL)."""
    today = date.today().isoformat()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with conn:
        conn.execute(
            "INSERT INTO attendance (person_id, timestamp, confidence, session_date) "
            "VALUES (NULL, ?, NULL, ?)",
            (now_str, today),
        )


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
        List of dicts: {attendance_id, name, reg_no, timestamp, confidence, session_date}
        Unknown persons appear with name='Unknown', reg_no=''.
    """
    target_date = session_date or date.today().isoformat()
    rows = conn.execute(
        """
        SELECT
            a.id         AS attendance_id,
            COALESCE(p.name, 'Unknown')   AS name,
            COALESCE(p.reg_no, '')        AS reg_no,
            a.timestamp,
            a.confidence,
            a.session_date
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
            a.id         AS attendance_id,
            COALESCE(p.name, 'Unknown')   AS name,
            COALESCE(p.reg_no, '')        AS reg_no,
            a.timestamp,
            a.confidence,
            a.session_date
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
    """
    target_date = session_date or date.today().isoformat()
    records = get_attendance_for_date(conn, target_date)

    known = [r for r in records if r["name"] != "Unknown"]
    unknown = [r for r in records if r["name"] == "Unknown"]

    # Deduplicate by reg_no (first appearance = arrival time)
    seen = set()
    unique_known = []
    for r in known:
        if r["reg_no"] not in seen:
            seen.add(r["reg_no"])
            unique_known.append(r)

    return {
        "date": target_date,
        "total_present": len(unique_known),
        "total_unknown": len(unknown),
        "records": unique_known,
    }
