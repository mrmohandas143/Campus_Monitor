"""
attendance/alerts.py
====================
Alert utilities for unknown-person detection events.

Provides:
    sound_alert()   — plays an audible beep (Windows winsound, fallback: bell)
    console_alert() — prints a prominent ANSI-colored alert banner to stdout
    export_alerts() — writes unknown-person events to an Excel (.xlsx) file

Python 3.6+ compatible. No third-party dependencies beyond openpyxl (already required).
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------
def _r(s: str) -> str: return f"\033[91m{s}\033[0m"
def _y(s: str) -> str: return f"\033[93m{s}\033[0m"
def _b(s: str) -> str: return f"\033[1m{s}\033[0m"


# ---------------------------------------------------------------------------
# Sound alert
# ---------------------------------------------------------------------------
def sound_alert(repeat: int = 3) -> None:
    """Play an audible beep to alert the operator.

    Uses ``winsound.Beep`` on Windows (built-in, no install needed).
    Falls back to printing the ASCII bell character on other platforms.

    Args:
        repeat: Number of beeps to play.
    """
    try:
        import winsound
        for _ in range(repeat):
            winsound.Beep(1000, 300)   # 1000 Hz, 300 ms
    except (ImportError, RuntimeError):
        # Non-Windows or audio unavailable — use terminal bell
        print("\a" * repeat, end="", flush=True)


# ---------------------------------------------------------------------------
# Console alert
# ---------------------------------------------------------------------------
def console_alert(snapshot_path: Optional[str] = None) -> None:
    """Print a prominent alert banner to the terminal.

    Args:
        snapshot_path: Path to the saved face-crop snapshot (may be None).
    """
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    line = "=" * 62
    print(_r(_b(f"\n  {'!' * 3}  UNKNOWN PERSON DETECTED  {'!' * 3}")))
    print(_r(f"  {line}"))
    print(_r(f"  Time     : {ts}"))
    if snapshot_path:
        print(_r(f"  Snapshot : {snapshot_path}"))
    else:
        print(_y(f"  Snapshot : (not saved)"))
    print(_r(f"  {line}\n"))


# ---------------------------------------------------------------------------
# Excel export for alerts
# ---------------------------------------------------------------------------
_ALERT_HEADER_FILL = PatternFill("solid", fgColor="7B0000")   # dark red
_ALERT_HEADER_FONT = Font(name="Calibri", bold=True, color="FFFFFF", size=11)
_EVEN_FILL         = PatternFill("solid", fgColor="FFE0E0")   # light red
_ODD_FILL          = PatternFill("solid", fgColor="FFFFFF")
_ALERT_DATA_FONT   = Font(name="Calibri", color="C00000", size=10)
_THIN_BORDER       = Border(
    left=Side(style="thin", color="BFBFBF"),
    right=Side(style="thin", color="BFBFBF"),
    top=Side(style="thin", color="BFBFBF"),
    bottom=Side(style="thin", color="BFBFBF"),
)
_CENTER = Alignment(horizontal="center", vertical="center")
_LEFT   = Alignment(horizontal="left",   vertical="center")

_ALERT_COLUMNS = [
    ("S.No",      8),
    ("Date",     13),
    ("Time",     10),
    ("Snapshot", 50),
]


def export_alerts(
    records: List[dict],
    output_path: Optional[str],
    label: str,
    title: str,
) -> str:
    """Write unknown-person alert records to an Excel file.

    Args:
        records:     List of dicts from ``crud.get_unknown_alerts_for_date`` /
                     ``crud.get_unknown_alerts_range``.
        output_path: Destination ``.xlsx`` path (None → default under data/exports/).
        label:       Short label used in the default filename.
        title:       Title string written at the top of the sheet.

    Returns:
        Absolute path of the written ``.xlsx`` file.
    """
    if output_path is None:
        project_root = Path(__file__).parent.parent
        exports_dir = project_root / "data" / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(exports_dir / f"alerts_{label}.xlsx")

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Unknown Alerts"

    num_cols = len(_ALERT_COLUMNS)
    last_col = get_column_letter(num_cols)

    # -- Title row --
    ws.merge_cells(f"A1:{last_col}1")
    title_cell = ws["A1"]
    title_cell.value = title
    title_cell.font = Font(name="Calibri", bold=True, size=14, color="7B0000")
    title_cell.alignment = _CENTER
    ws.row_dimensions[1].height = 28

    # -- Generated timestamp --
    ws.merge_cells(f"A2:{last_col}2")
    ts_cell = ws["A2"]
    ts_cell.value = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    ts_cell.font = Font(name="Calibri", italic=True, size=9, color="808080")
    ts_cell.alignment = _CENTER
    ws.row_dimensions[2].height = 16

    # -- Summary --
    ws.merge_cells(f"A3:{last_col}3")
    summary_cell = ws["A3"]
    summary_cell.value = f"Total Unknown Detections: {len(records)}"
    summary_cell.font = Font(name="Calibri", bold=True, size=10, color="7B0000")
    summary_cell.alignment = _CENTER
    ws.row_dimensions[3].height = 18

    ws.row_dimensions[4].height = 6  # blank separator

    # -- Headers (row 5) --
    header_row = 5
    for col_idx, (col_name, col_width) in enumerate(_ALERT_COLUMNS, start=1):
        cell = ws.cell(row=header_row, column=col_idx, value=col_name)
        cell.font = _ALERT_HEADER_FONT
        cell.fill = _ALERT_HEADER_FILL
        cell.alignment = _CENTER
        cell.border = _THIN_BORDER
        ws.column_dimensions[get_column_letter(col_idx)].width = col_width
    ws.row_dimensions[header_row].height = 20

    # -- Data rows --
    for row_offset, record in enumerate(records):
        data_row = header_row + 1 + row_offset
        try:
            ts = datetime.strptime(record["timestamp"], "%Y-%m-%d %H:%M:%S")
            date_str = ts.strftime("%d-%m-%Y")
            time_str = ts.strftime("%H:%M:%S")
        except Exception:
            date_str = record.get("session_date", "")
            time_str = record.get("timestamp", "")

        snapshot = record.get("snapshot_path") or "—"
        row_data = [row_offset + 1, date_str, time_str, snapshot]

        fill = _EVEN_FILL if row_offset % 2 == 0 else _ODD_FILL
        for col_idx, value in enumerate(row_data, start=1):
            cell = ws.cell(row=data_row, column=col_idx, value=value)
            cell.fill = fill
            cell.border = _THIN_BORDER
            cell.alignment = _LEFT if col_idx == 4 else _CENTER
            cell.font = _ALERT_DATA_FONT
        ws.row_dimensions[data_row].height = 16

    ws.freeze_panes = ws.cell(row=header_row + 1, column=1)
    ws.auto_filter.ref = (
        f"A{header_row}:{last_col}{header_row + len(records)}"
    )

    wb.save(output_path)
    logger.info("Exported %d alert records to %s", len(records), output_path)
    print(f"  Exported {len(records)} alert record(s) \u2192 {output_path}")
    return output_path
