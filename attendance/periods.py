"""
attendance/periods.py
=====================
Defines the 8 class periods and provides helpers to determine
which period is currently active.

Periods:
    Period 1 :  09:00 – 10:00
    Period 2 :  10:00 – 10:50
    Period 3 :  11:05 – 11:55
    Period 4 :  11:55 – 12:45
    Period 5 :  13:30 – 14:20
    Period 6 :  14:20 – 15:10
    Period 7 :  15:20 – 16:10
    Period 8 :  16:10 – 17:00

Python 3.6+ compatible.
"""

from __future__ import annotations

from datetime import datetime, time
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Period definitions  (name, start, end — end is exclusive)
# ---------------------------------------------------------------------------
PERIODS: List[Dict] = [
    {"name": "Period 1", "start": time(9,  0),  "end": time(10,  0)},
    {"name": "Period 2", "start": time(10, 0),  "end": time(10, 50)},
    {"name": "Period 3", "start": time(11, 5),  "end": time(11, 55)},
    {"name": "Period 4", "start": time(11, 55), "end": time(12, 45)},
    {"name": "Period 5", "start": time(13, 30), "end": time(14, 20)},
    {"name": "Period 6", "start": time(14, 20), "end": time(15, 10)},
    {"name": "Period 7", "start": time(15, 20), "end": time(16, 10)},
    {"name": "Period 8", "start": time(16, 10), "end": time(17,  0)},
]


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------
def get_current_period(now: Optional[datetime] = None) -> Optional[Dict]:
    """Return the active period dict, or None if outside all periods.

    Args:
        now: datetime to check (defaults to ``datetime.now()``).

    Returns:
        ``{"name": str, "start": time, "end": time}`` or ``None``.
    """
    t = (now or datetime.now()).time()
    for p in PERIODS:
        if p["start"] <= t < p["end"]:
            return p
    return None


def is_within_any_period(now: Optional[datetime] = None) -> bool:
    """Return True if *now* falls inside any defined period."""
    return get_current_period(now) is not None


def next_period(now: Optional[datetime] = None) -> Optional[Dict]:
    """Return the next upcoming period, or None if after the last period for today.

    Args:
        now: datetime to check (defaults to ``datetime.now()``).

    Returns:
        Next period dict or ``None``.
    """
    t = (now or datetime.now()).time()
    for p in PERIODS:
        if p["start"] > t:
            return p
    return None


def period_label(now: Optional[datetime] = None) -> str:
    """Return a short human-readable label for display in the HUD.

    Examples: ``"Period 3 (11:05-11:55)"`` or ``"-- Break --"``
    """
    p = get_current_period(now)
    if p is None:
        nxt = next_period(now)
        if nxt:
            return f"Break  (next: {p['name']} {nxt['start'].strftime('%H:%M')})" if False \
                else f"Break  (next: {nxt['name']} @ {nxt['start'].strftime('%H:%M')})"
        return "After hours"
    return f"{p['name']}  {p['start'].strftime('%H:%M')}-{p['end'].strftime('%H:%M')}"
