import os
import sys
import tempfile
import sqlite3
import numpy as np
from datetime import datetime, timedelta
import openpyxl

sys.path.insert(0, ".")

from db.schema import init_db
from db import crud
from attendance.exporter import export_daily, export_range


def test_schema_and_out_time():
    print("[TEST 1] Testing DB schema migration, out_time column, and 1-min threshold...")
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test.db")
    conn = init_db(db_path)
    try:
        # Check columns of attendance
        cols = [c[1] for c in conn.execute("PRAGMA table_info(attendance)").fetchall()]
        assert "out_time" in cols, f"out_time column missing from attendance table: {cols}"
        assert "timestamp" in cols, f"timestamp column missing: {cols}"
        assert "period_name" not in cols, f"period_name should NOT be in attendance table: {cols}"
        print("  [OK] out_time column exists and period_name is removed from attendance table")
        
        # Register a test student
        fake_emb = np.random.rand(128).astype(np.float32)
        fake_emb /= np.linalg.norm(fake_emb)
        pid = crud.register_person(conn, "Test Student", "TEST001", fake_emb)
        assert pid > 0
        
        # 1. Log initial attendance (In-Time)
        logged, reason = crud.log_attendance(conn, pid, confidence=0.88)
        assert logged is True, f"Expected initial log to be True, got {logged} ({reason})"
        print("  [OK] In-time logged successfully:", reason)
        
        # Verify initial record in DB
        records = crud.get_attendance_for_date(conn)
        assert len(records) == 1
        r = records[0]
        assert r["reg_no"] == "TEST001"
        assert r["timestamp"] is not None
        assert r["out_time"] is None
        initial_in_time = r["timestamp"]
        print("  [OK] Initial record has timestamp and out_time=None")
        
        # 2. Test 1-minute rule: if same person crosses camera within 1 min, do NOT set out_time
        logged_within_1m, reason_within_1m = crud.log_attendance(conn, pid, confidence=0.91)
        assert logged_within_1m is False
        assert "within 1 min" in reason_within_1m.lower(), f"Expected within 1 min warning, got: {reason_within_1m}"
        
        # Verify out_time is STILL None!
        records_1m = crud.get_attendance_for_date(conn)
        assert records_1m[0]["out_time"] is None, "out_time should NOT be set within 1 minute of in-time!"
        print("  [OK] Within 1 min: out-time was NOT set (remained None)")
        
        # 3. Simulate detection after >= 60 seconds (e.g. 5 minutes later)
        past_time = (datetime.now() - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
        conn.execute("UPDATE attendance SET timestamp=? WHERE id=?", (past_time, r["attendance_id"]))
        
        logged_after_5m, reason_after_5m = crud.log_attendance(conn, pid, confidence=0.90)
        assert logged_after_5m is False
        assert "Out-time updated" in reason_after_5m, f"Expected Out-time updated, got: {reason_after_5m}"
        
        records_after = crud.get_attendance_for_date(conn)
        assert records_after[0]["out_time"] is not None
        print("  [OK] After >= 1 min: out-time was updated successfully:", reason_after_5m)
        
    finally:
        conn.close()


def test_excel_export_and_inside_column():
    print("\n[TEST 2] Testing Excel export Inside column, count at end, and unknown exclusion...")
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test_export.db")
    conn = init_db(db_path)
    try:
        fake_emb = np.random.rand(128).astype(np.float32)
        fake_emb /= np.linalg.norm(fake_emb)
        
        # 1. Student 1: Inside (has in-time, out-time is None)
        pid1 = crud.register_person(conn, "Alice InRoom", "REG2024001", fake_emb)
        # 2. Student 2: Left (has in-time, out-time is set)
        pid2 = crud.register_person(conn, "Bob LeftRoom", "REG2024002", fake_emb)
        # 3. Auto-registered unknown (should be excluded)
        pid3 = crud.register_person(conn, "Unknown_20260906_120000", "UNK_20260906_120000", fake_emb)
        
        # Log attendance for Student 1 (Inside: out_time=None)
        crud.log_attendance(conn, pid1, confidence=0.92)
        
        # Log attendance for Student 2 (Left: out_time is set)
        crud.log_attendance(conn, pid2, confidence=0.95)
        crud.update_out_time(conn, pid2, out_time="2026-09-06 10:00:00")
        
        # Log attendance for Unknown
        crud.log_attendance(conn, pid3, confidence=0.0)
        
        export_file = os.path.join(tmpdir, "test_attendance.xlsx")
        out_path = export_daily(conn, output_path=export_file)
        assert os.path.exists(out_path)
        
        wb = openpyxl.load_workbook(out_path)
        ws = wb.active
        
        # Check header row (row 5)
        headers = [ws.cell(row=5, column=c).value for c in range(1, 10)]
        expected_headers = [
            "S.No", "Name", "Reg No", "Date", "In Time", "Out Time", "Inside", "Confidence", "Status"
        ]
        assert headers == expected_headers, f"Headers mismatch: {headers} vs {expected_headers}"
        print("  [OK] Excel headers match expected [In Time, Out Time, Inside, etc.]")
        
        # Data rows should be 2 (Alice and Bob), Unknown excluded!
        # Row 6: Alice
        alice_row = [ws.cell(row=6, column=c).value for c in range(1, 10)]
        assert alice_row[1] == "Alice InRoom"
        assert alice_row[5] == "—"      # Out Time
        assert alice_row[6] == "Yes"    # Inside
        print("  [OK] Student without out-time has Inside='Yes'")
        
        # Row 7: Bob
        bob_row = [ws.cell(row=7, column=c).value for c in range(1, 10)]
        assert bob_row[1] == "Bob LeftRoom"
        assert bob_row[5] == "10:00:00" # Out Time
        assert bob_row[6] == "No"       # Inside
        print("  [OK] Student with out-time has Inside='No'")
        
        # Row 8: Total Persons Inside row at the end of the table
        total_label = ws["A8"].value
        inside_count_val = ws["G8"].value
        assert "Total Persons Inside" in total_label, f"Expected total label at row 8, got '{total_label}'"
        assert inside_count_val == 1, f"Expected inside count 1 at G8, got {inside_count_val}"
        print(f"  [OK] Total row at end of table correctly shows '{total_label}' -> Count: {inside_count_val}")
        
        # Check summary cell (A3)
        summary_val = ws["A3"].value
        assert "Currently Inside: 1" in summary_val, f"Expected 'Currently Inside: 1' in summary, got '{summary_val}'"
        assert "Unknown" not in summary_val
        print("  [OK] Top summary row (A3) displays:", summary_val)
        
    finally:
        conn.close()


def test_engine_unknown_capture_logic():
    print("\n[TEST 3] Testing AttendanceEngine unknown capture logic...")
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test_engine.db")
    conn = init_db(db_path)
    try:
        from attendance.engine import AttendanceEngine
        engine = AttendanceEngine(conn)
        engine.load_gallery()
        
        # Check initial persons and attendance counts
        p_count_before = conn.execute("SELECT count(*) FROM persons").fetchone()[0]
        a_count_before = conn.execute("SELECT count(*) FROM attendance").fetchone()[0]
        
        # Mock frame and face
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        fake_face = np.array([50, 50, 100, 100, 70, 70, 90, 70, 80, 90, 70, 110, 90, 110, 0.95], dtype=np.float32)
        fake_emb = np.random.rand(128).astype(np.float32)
        fake_emb /= np.linalg.norm(fake_emb)
        
        # 1. Test _capture_unknown
        is_new, snap = engine._capture_unknown(fake_frame, fake_face, fake_emb)
        assert is_new is True
        assert snap is not None
        assert os.path.exists(snap)
        print("  [OK] Unknown face snapshot saved locally at:", snap)
        
        # 2. Test cooldown: immediate same embedding should NOT re-capture
        is_new2, snap2 = engine._capture_unknown(fake_frame, fake_face, fake_emb)
        assert is_new2 is False
        assert snap2 is None
        print("  [OK] Cooldown prevented duplicate unknown snapshot")
        
        # 3. Ensure no DB inserts were made for unknown person
        p_count_after = conn.execute("SELECT count(*) FROM persons").fetchone()[0]
        a_count_after = conn.execute("SELECT count(*) FROM attendance").fetchone()[0]
        assert p_count_after == p_count_before, "persons table was modified for unknown face!"
        assert a_count_after == a_count_before, "attendance table was modified for unknown face!"
        print("  [OK] Database persons and attendance tables were NOT populated for unknown person")
    finally:
        conn.close()


def test_repeated_entry_within_1_min_never_marks_outside():
    print("\n[TEST 4] Testing requirement: entering again within 1 min does NOT take as left and does NOT mark outside...")
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test_guard.db")
    conn = init_db(db_path)
    try:
        fake_emb = np.random.rand(128).astype(np.float32)
        fake_emb /= np.linalg.norm(fake_emb)
        pid = crud.register_person(conn, "Mohan Guard Test", "GUARD001", fake_emb)

        # 1. First entry into camera
        logged, reason = crud.log_attendance(conn, pid, confidence=0.89)
        assert logged is True
        
        # Check status: Inside, out_time is None
        rec = crud.get_attendance_for_date(conn)[0]
        assert rec["out_time"] is None, "out_time should be None on initial entry"
        summary = crud.get_daily_summary(conn)
        inside_count = sum(1 for r in summary["records"] if not r.get("out_time"))
        assert inside_count == 1, "Person should be marked INSIDE"
        print("  [OK] Initial entry: logged in, out_time=None, marked INSIDE (inside_count=1)")

        # 2. Enters camera again after 15 seconds (within 1 min)
        # Simulate in-time was 15 seconds ago
        t_15s_ago = (datetime.now() - timedelta(seconds=15)).strftime("%Y-%m-%d %H:%M:%S")
        conn.execute("UPDATE attendance SET timestamp=? WHERE id=?", (t_15s_ago, rec["attendance_id"]))

        logged_15s, reason_15s = crud.log_attendance(conn, pid, confidence=0.92)
        assert logged_15s is False
        assert "still inside" in reason_15s.lower() or "within 1 min" in reason_15s.lower()
        rec_15s = crud.get_attendance_for_date(conn)[0]
        assert rec_15s["out_time"] is None, "out_time must NOT be set when entering again after 15s"
        summary_15s = crud.get_daily_summary(conn)
        inside_count_15s = sum(1 for r in summary_15s["records"] if not r.get("out_time"))
        assert inside_count_15s == 1, "Person must STILL be marked INSIDE after 15s"
        print(f"  [OK] Sighting after 15s: {reason_15s} -> out_time=None, still marked INSIDE")

        # 3. Enters camera again after 45 seconds (still within 1 min)
        t_45s_ago = (datetime.now() - timedelta(seconds=45)).strftime("%Y-%m-%d %H:%M:%S")
        conn.execute("UPDATE attendance SET timestamp=? WHERE id=?", (t_45s_ago, rec["attendance_id"]))

        logged_45s, reason_45s = crud.log_attendance(conn, pid, confidence=0.91)
        assert logged_45s is False
        rec_45s = crud.get_attendance_for_date(conn)[0]
        assert rec_45s["out_time"] is None, "out_time must NOT be set when entering again after 45s"
        summary_45s = crud.get_daily_summary(conn)
        inside_count_45s = sum(1 for r in summary_45s["records"] if not r.get("out_time"))
        assert inside_count_45s == 1, "Person must STILL be marked INSIDE after 45s"
        print("  [OK] Sighting after 45s: out_time=None, still marked INSIDE")

        # 4. Enters camera again after 59 seconds (still within 1 min)
        t_59s_ago = (datetime.now() - timedelta(seconds=59)).strftime("%Y-%m-%d %H:%M:%S")
        conn.execute("UPDATE attendance SET timestamp=? WHERE id=?", (t_59s_ago, rec["attendance_id"]))

        logged_59s, reason_59s = crud.log_attendance(conn, pid, confidence=0.90)
        assert logged_59s is False
        rec_59s = crud.get_attendance_for_date(conn)[0]
        assert rec_59s["out_time"] is None, "out_time must NOT be set when entering again after 59s"
        summary_59s = crud.get_daily_summary(conn)
        inside_count_59s = sum(1 for r in summary_59s["records"] if not r.get("out_time"))
        assert inside_count_59s == 1, "Person must STILL be marked INSIDE after 59s"
        print("  [OK] Sighting after 59s: out_time=None, still marked INSIDE")

        # 5. Crossing camera after >= 60 seconds (person actually leaving)
        t_90s_ago = (datetime.now() - timedelta(seconds=90)).strftime("%Y-%m-%d %H:%M:%S")
        conn.execute("UPDATE attendance SET timestamp=? WHERE id=?", (t_90s_ago, rec["attendance_id"]))

        logged_90s, reason_90s = crud.log_attendance(conn, pid, confidence=0.91)
        assert logged_90s is False
        assert "out-time updated" in reason_90s.lower()
        rec_90s = crud.get_attendance_for_date(conn)[0]
        assert rec_90s["out_time"] is not None, "out_time SHOULD be set after 90s (person leaving)"
        summary_90s = crud.get_daily_summary(conn)
        inside_count_90s = sum(1 for r in summary_90s["records"] if not r.get("out_time"))
        assert inside_count_90s == 0, "Person should now be marked OUTSIDE (left)"
        print(f"  [OK] Sighting after 90s (leaving): {reason_90s} -> out_time recorded, marked OUTSIDE")

    finally:
        conn.close()


if __name__ == "__main__":
    test_schema_and_out_time()
    test_excel_export_and_inside_column()
    test_engine_unknown_capture_logic()
    test_repeated_entry_within_1_min_never_marks_outside()
    print("\nALL CUSTOM TESTS PASSED!")
