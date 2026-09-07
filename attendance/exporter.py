"""
attendance/exporter.py
======================
Exports attendance records from SQLite to Excel (.xlsx) using openpyxl.

No pandas dependency — openpyxl only, for minimal Jetson Nano footprint.

Python 3.6+ compatible.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional

import openpyxl
from openpyxl.styles import (
    Alignment, Font, PatternFill, Border, Side
)
from openpyxl.utils import get_column_letter

from db import crud

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
_HEADER_FILL   = PatternFill("solid", fgColor="1F3864")   # dark navy
_HEADER_FONT   = Font(name="Calibri", bold=True, color="FFFFFF", size=11)
_EVEN_FILL     = PatternFill("solid", fgColor="DCE6F1")   # light blue
_ODD_FILL      = PatternFill("solid", fgColor="FFFFFF")
_PRESENT_FONT  = Font(name="Calibri", color="1E7B34", bold=True, size=10)   # green
_UNKNOWN_FONT  = Font(name="Calibri", color="C00000", size=10)               # red
_INSIDE_FONT   = Font(name="Calibri", color="1E7B34", bold=True, size=10)   # green
_OUT_FONT      = Font(name="Calibri", color="595959", size=10)               # gray
_TOTAL_FONT    = Font(name="Calibri", bold=True, size=11, color="1F3864")
_TOTAL_COUNT   = Font(name="Calibri", bold=True, size=11, color="1E7B34")
_TOTAL_FILL    = PatternFill("solid", fgColor="E9EEF4")
_DEFAULT_FONT  = Font(name="Calibri", size=10)
_THIN_BORDER   = Border(
    left=Side(style="thin", color="BFBFBF"),
    right=Side(style="thin", color="BFBFBF"),
    top=Side(style="thin", color="BFBFBF"),
    bottom=Side(style="thin", color="BFBFBF"),
)
_DOUBLE_BOTTOM = Border(
    left=Side(style="thin", color="BFBFBF"),
    right=Side(style="thin", color="BFBFBF"),
    top=Side(style="thin", color="1F3864"),
    bottom=Side(style="double", color="1F3864"),
)
_CENTER = Alignment(horizontal="center", vertical="center")
_LEFT   = Alignment(horizontal="left",   vertical="center")


# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------
_COLUMNS = [
    ("S.No",       8),
    ("Name",      24),
    ("Reg No",    14),
    ("Date",      13),
    ("Period",    12),
    ("In Time",   12),
    ("Out Time",  12),
    ("Inside",    10),
    ("Confidence",13),
    ("Status",    12),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def export_daily(
    conn: sqlite3.Connection,
    output_path: Optional[str] = None,
    session_date: Optional[str] = None,
) -> str:
    """
    Export one day's attendance to an Excel file.

    Args:
        conn:         Open SQLite connection.
        output_path:  Destination .xlsx path. Defaults to data/attendance_YYYY-MM-DD.xlsx.
        session_date: YYYY-MM-DD string. Defaults to today.

    Returns:
        Absolute path of the written .xlsx file.
    """
    target_date = session_date or date.today().isoformat()
    records = crud.get_attendance_for_date(conn, target_date)
    return _write_excel(records, output_path, target_date, title=f"Attendance — {target_date}")


def export_range(
    conn: sqlite3.Connection,
    start_date: str,
    end_date: str,
    output_path: Optional[str] = None,
) -> str:
    """
    Export attendance for a date range to an Excel file.

    Args:
        conn:        Open SQLite connection.
        start_date:  YYYY-MM-DD (inclusive).
        end_date:    YYYY-MM-DD (inclusive).
        output_path: Destination .xlsx path.

    Returns:
        Absolute path of the written .xlsx file.
    """
    records = crud.get_attendance_range(conn, start_date, end_date)
    label = f"{start_date}_to_{end_date}"
    title = f"Attendance — {start_date} to {end_date}"
    return _write_excel(records, output_path, label, title=title)


# ---------------------------------------------------------------------------
# Internal writer
# ---------------------------------------------------------------------------
def _default_path(label: str) -> str:
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    exports_dir = project_root / "data" / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    return str(exports_dir / f"attendance_{label}.xlsx")


def _write_excel(
    records: List[dict],
    output_path: Optional[str],
    label: str,
    title: str,
) -> str:
    path = output_path or _default_path(label)

    # Filter out unknown persons — do not add unknown person data to the Excel sheet
    known_records = [
        r for r in records
        if not (
            r.get("reg_no", "").startswith("UNK_")
            or r.get("name", "").startswith("Unknown")
            or not r.get("reg_no")
        )
    ]

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Attendance"

    last_col = get_column_letter(len(_COLUMNS))

    # -- Title row --
    ws.merge_cells(f"A1:{last_col}1")
    title_cell = ws["A1"]
    title_cell.value = title
    title_cell.font = Font(name="Calibri", bold=True, size=14, color="1F3864")
    title_cell.alignment = _CENTER
    ws.row_dimensions[1].height = 28

    # -- Generated timestamp --
    ws.merge_cells(f"A2:{last_col}2")
    ts_cell = ws["A2"]
    ts_cell.value = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    ts_cell.font = Font(name="Calibri", italic=True, size=9, color="808080")
    ts_cell.alignment = _CENTER
    ws.row_dimensions[2].height = 16

    # -- Summary row --
    unique_present = len({r["reg_no"] for r in known_records if r.get("reg_no")})
    inside_count   = sum(1 for r in known_records if not r.get("out_time"))

    ws.merge_cells(f"A3:{last_col}3")
    summary_cell = ws["A3"]
    summary_cell.value = (
        f"Total Present (Unique): {unique_present}   |   Total Records: {len(known_records)}   |   Persons Inside: {inside_count}"
    )
    summary_cell.font = Font(name="Calibri", bold=True, size=10, color="1F3864")
    summary_cell.alignment = _CENTER
    ws.row_dimensions[3].height = 18

    # Blank separator
    ws.row_dimensions[4].height = 6

    # -- Column headers (row 5) --
    header_row = 5
    for col_idx, (col_name, col_width) in enumerate(_COLUMNS, start=1):
        cell = ws.cell(row=header_row, column=col_idx, value=col_name)
        cell.font = _HEADER_FONT
        cell.fill = _HEADER_FILL
        cell.alignment = _CENTER
        cell.border = _THIN_BORDER
        ws.column_dimensions[get_column_letter(col_idx)].width = col_width
    ws.row_dimensions[header_row].height = 20

    # -- Data rows --
    for row_offset, record in enumerate(known_records):
        data_row = header_row + 1 + row_offset

        # Parse in_time from timestamp
        try:
            ts = datetime.strptime(record["timestamp"], "%Y-%m-%d %H:%M:%S")
            date_str = ts.strftime("%d-%m-%Y")
            in_time_str = ts.strftime("%H:%M:%S")
        except Exception:
            date_str = record.get("session_date", "")
            in_time_str = record.get("timestamp", "") or "—"

        # Parse out_time
        out_time_raw = record.get("out_time")
        if out_time_raw:
            try:
                ots = datetime.strptime(out_time_raw, "%Y-%m-%d %H:%M:%S")
                out_time_str = ots.strftime("%H:%M:%S")
            except Exception:
                out_time_str = str(out_time_raw)
        else:
            out_time_str = "—"

        is_inside = (not out_time_raw) or (out_time_raw == "") or (out_time_str == "—")
        inside_str = "Yes" if is_inside else "No"

        confidence = record.get("confidence")
        conf_str   = f"{confidence:.3f}" if confidence is not None else "—"
        period_str = record.get("period_name") or "—"
        status     = "Present"

        row_data = [
            row_offset + 1,           # S.No (col 1)
            record["name"],           # Name (col 2)
            record["reg_no"] or "—",  # Reg No (col 3)
            date_str,                 # Date (col 4)
            period_str,               # Period (col 5)
            in_time_str,              # In Time (col 6)
            out_time_str,             # Out Time (col 7)
            inside_str,               # Inside (col 8)
            conf_str,                 # Confidence (col 9)
            status,                   # Status (col 10)
        ]

        fill = _EVEN_FILL if row_offset % 2 == 0 else _ODD_FILL

        for col_idx, value in enumerate(row_data, start=1):
            cell = ws.cell(row=data_row, column=col_idx, value=value)
            cell.fill = fill
            cell.border = _THIN_BORDER
            cell.alignment = _CENTER if col_idx != 2 else _LEFT

            if col_idx == 8:  # Inside column
                cell.font = _INSIDE_FONT if is_inside else _OUT_FONT
            elif col_idx == 10:  # Status column (col 10)
                cell.font = _PRESENT_FONT
            else:
                cell.font = _DEFAULT_FONT

        ws.row_dimensions[data_row].height = 16

    # -- Total row at the end of the table showing count of persons inside --
    total_row = header_row + len(known_records) + 1
    ws.merge_cells(f"A{total_row}:G{total_row}")
    total_label_cell = ws[f"A{total_row}"]
    total_label_cell.value = "Total Persons Inside:"
    total_label_cell.font = _TOTAL_FONT
    total_label_cell.alignment = Alignment(horizontal="right", vertical="center")

    count_cell = ws.cell(row=total_row, column=8, value=inside_count)
    count_cell.font = _TOTAL_COUNT
    count_cell.alignment = _CENTER

    for col_idx in range(1, len(_COLUMNS) + 1):
        c = ws.cell(row=total_row, column=col_idx)
        c.fill = _TOTAL_FILL
        c.border = _DOUBLE_BOTTOM

    ws.row_dimensions[total_row].height = 22

    # -- Freeze header rows --
    ws.freeze_panes = ws.cell(row=header_row + 1, column=1)

    # -- Auto-filter on header row --
    ws.auto_filter.ref = (
        f"A{header_row}:{get_column_letter(len(_COLUMNS))}{header_row + len(known_records)}"
    )

    wb.save(path)
    logger.info("Exported %d records to %s", len(known_records), path)
    print(f"  Exported {len(known_records)} records -> {path}")
    return path
