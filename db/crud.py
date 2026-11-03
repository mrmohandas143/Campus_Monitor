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
    Log an attendance event for today's session with 1-minute entry guard:

    - First sighting today: logs in-time (marked Inside).
    - If someone enters the camera and again enters within 1 min:
      it is NOT taken as having left and is NOT marked outside (remains Inside).
    - If sighting occurs >= 60 seconds after entry:
      it updates out-time (marked Outside).
    - If a person who was marked outside re-crosses the camera later:
      they are re-marked Inside.

    Args:
        conn:        Open database connection.
        person_id:   ID from the persons table.
        confidence:  Cosine similarity score from the recognizer.
        period_name: Kept for backwards compatibility (ignored).

    Returns:
        ``(True, reason)``  when initial in-time or re-entry is logged.
        ``(False, reason)`` when skipped within 1 min or out-time is updated.
    """
    today   = date.today().isoformat()
    now     = datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")

    # Was this person already logged today?
    existing = conn.execute(
        """
        SELECT id, timestamp, out_time FROM attendance
        WHERE person_id   = ?
          AND session_date = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (person_id, today),
    ).fetchone()

    if existing:
        existing_id  = existing["id"]
        in_time_str  = existing["timestamp"]
        out_time_str = existing["out_time"]

        # Parse in-time from DB
        try:
            in_dt = datetime.strptime(in_time_str, "%Y-%m-%d %H:%M:%S")
        except Exception:
            in_dt = now

        # If person is currently INSIDE (out_time is None)
        if not out_time_str:
            elapsed = (now - in_dt).total_seconds()

            # Rule: If someone enters the camera and again enters within 1 min (60s),
            # it is NOT taken as he left and does NOT mark outside.
            if elapsed < 60:
                logger.debug(
                    "Person person_id=%d entered again within 1 min (%.1fs). Still inside.",
                    person_id, elapsed,
                )
                return False, f"Within 1 min of entry ({int(elapsed)}s) — still inside (not left)"

            # Beyond 1 min: person is now leaving (marked outside with out_time)
            with conn:
                conn.execute(
                    "UPDATE attendance SET out_time = ? WHERE id = ?",
                    (now_str, existing_id),
                )
            time_part = now_str.split()[1] if " " in now_str else now_str
            logger.debug(
                "Out-time updated: person_id=%d  out_time=%s",
                person_id, now_str,
            )
            return False, f"Out-time updated ({time_part})"
        else:
            # Person was marked outside (had out_time), but is now crossing camera again
            try:
                out_dt = datetime.strptime(out_time_str, "%Y-%m-%d %H:%M:%S")
                elapsed_since_out = (now - out_dt).total_seconds()
            except Exception:
                elapsed_since_out = 999

            if elapsed_since_out < 60:
                return False, f"Within 1 min of exit ({int(elapsed_since_out)}s)"

            # Re-entering the room: update in-time to now and clear out_time so they are marked INSIDE
            with conn:
                conn.execute(
                    "UPDATE attendance SET timestamp = ?, out_time = NULL WHERE id = ?",
                    (now_str, existing_id),
                )
            time_part = now_str.split()[1] if " " in now_str else now_str
            logger.debug("Person %d re-entered at %s — marked inside", person_id, time_part)
            return True, f"Re-entered at {time_part} (marked inside)"

    # First entry today: record in-time and mark INSIDE
    with conn:
        conn.execute(
            "INSERT INTO attendance "
            "(person_id, timestamp, confidence, session_date) "
            "VALUES (?, ?, ?, ?)",
            (person_id, now_str, confidence, today),
        )

    logger.debug(
        "Attendance logged: person_id=%d  confidence=%.3f",
        person_id, confidence,
    )
    return True, "Logged (Inside)"


def update_out_time(
    conn: sqlite3.Connection,
    person_id: int,
    out_time: Optional[str] = None,
    period_name: Optional[str] = None,
    session_date: Optional[str] = None,
) -> bool:
    """
    Update the out_time for a person's attendance record.

    Updates the latest attendance record for the person on session_date (default: today).

    Args:
        conn:         Open database connection.
        person_id:    ID from the persons table.
        out_time:     Out-time string formatted as 'YYYY-MM-DD HH:MM:SS' (default: now).
        period_name:  Ignored (for backwards compatibility).
        session_date: YYYY-MM-DD date string (default: today).

    Returns:
        True if an attendance record was found and updated, False otherwise.
    """
    target_date = session_date or date.today().isoformat()
    now_str = out_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
                        confidence, session_date}
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
            a.id           AS attendance_id,
            COALESCE(p.name,   'Unknown') AS name,
            COALESCE(p.reg_no, '')        AS reg_no,
            a.timestamp,
            a.out_time,
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
            latest_time = r.get("out_time")
            # If any record for the student has out_time is None, they are currently inside!
            if latest_time is None:
                seen[reg]["out_time"] = None
            elif seen[reg].get("out_time"):
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
                        snapshot_path}
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
