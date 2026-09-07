"""
cli/main.py
===========
Command-line interface for the Campus Monitor system.

Subcommands:
    register  — Capture face from camera and register a student
    monitor   — Live camera loop: detect, recognize, log attendance
    export    — Export attendance to Excel
    list      — Print all registered students
    report    — Print today's attendance summary

Usage examples:
    python cli/main.py register --name "Alice Kumar" --reg-no "CS2024001"
    python cli/main.py monitor --camera 0 --cooldown 300
    python cli/main.py export --date 2024-08-13
    python cli/main.py export --start 2024-08-01 --end 2024-08-13
    python cli/main.py list
    python cli/main.py report
    python cli/main.py report --date 2024-08-13

    # Jetson Nano with CUDA OpenCV:
    python cli/main.py monitor --cuda

Python 3.6+ compatible.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Allow running from project root
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import cv2

from attendance.engine import AttendanceEngine
from attendance.exporter import export_daily, export_range
from attendance.alerts import sound_alert, console_alert, export_alerts
from attendance.periods import period_label
from db.schema import init_db
from db import crud

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("campus_monitor")


# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------
def _g(s): return f"\033[92m{s}\033[0m"
def _r(s): return f"\033[91m{s}\033[0m"
def _y(s): return f"\033[93m{s}\033[0m"
def _b(s): return f"\033[1m{s}\033[0m"
def _c(s): return f"\033[96m{s}\033[0m"


# ---------------------------------------------------------------------------
# Subcommand: register
# ---------------------------------------------------------------------------
def cmd_register(args: argparse.Namespace) -> int:
    """Capture N frames from camera, average embeddings, register student."""
    conn = init_db(args.db)
    engine = AttendanceEngine(conn, use_cuda=args.cuda)
    engine.load_gallery()

    name   = args.name.strip()
    reg_no = args.reg_no.strip()

    if crud.person_exists(conn, reg_no):
        print(_y(f"[WARN] {reg_no} is already registered. Use --force to overwrite."))
        if not args.force:
            return 1

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(_r(f"[ERROR] Cannot open camera index {args.camera}"))
        return 1

    print(_b(f"\n  Registering: {name}  ({reg_no})"))
    print(f"  Camera: index {args.camera}  |  Samples to capture: {args.samples}")
    print(f"  Press SPACE to capture, Q to quit.\n")

    import numpy as np
    collected_embeddings = []
    face_save_dir = str(_PROJECT_ROOT / "data" / "faces")

    while len(collected_embeddings) < args.samples:
        ret, frame = cap.read()
        if not ret:
            print(_r("  [ERROR] Failed to read frame."))
            break

        # Detect and draw faces for visual feedback
        faces = engine.face_detector.detect(frame)
        for bbox in engine.face_detector.bboxes(faces):
            x1, y1, x2, y2 = bbox
            color = (0, 255, 0) if len(faces) == 1 else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        n_captured = len(collected_embeddings)
        status_text = f"Captured: {n_captured}/{args.samples}  |  SPACE=capture  Q=quit"
        face_hint   = "1 face detected — ready" if len(faces) == 1 else f"{len(faces)} faces"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
        cv2.putText(frame, face_hint,   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0,255,0) if len(faces) == 1 else (0,165,255), 1)
        cv2.imshow(f"Register — {name}", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            print("  Aborted.")
            cap.release()
            cv2.destroyAllWindows()
            return 1

        if key == ord(" "):
            if len(faces) != 1:
                print(_y(f"  [SKIP] Need exactly 1 face, detected {len(faces)}"))
                continue

            emb = engine.embedder.embed_face(frame, faces[0])
            collected_embeddings.append(emb)
            print(f"  [{len(collected_embeddings)}/{args.samples}] Sample captured.")

    cap.release()
    cv2.destroyAllWindows()

    if not collected_embeddings:
        print(_r("  No samples captured. Registration cancelled."))
        return 1

    # Average all collected embeddings
    avg_embedding = np.mean(np.stack(collected_embeddings, axis=0), axis=0)
    avg_embedding /= (np.linalg.norm(avg_embedding) + 1e-10)

    # Save face crop from last good frame (reuse `frame` which had 1 face)
    face_path = None
    if len(faces) == 1:
        try:
            aligned = engine.embedder._recognizer.alignCrop(frame, faces[0])
            save_path = Path(face_save_dir) / f"{reg_no}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), aligned)
            face_path = str(save_path)
        except Exception:
            pass

    if args.force and crud.person_exists(conn, reg_no):
        crud.update_embedding(conn, reg_no, avg_embedding)
        print(_g(f"\n  [OK] Updated embedding for {name} ({reg_no})"))
    else:
        try:
            person_id = crud.register_person(conn, name, reg_no, avg_embedding, face_path)
            print(_g(f"\n  [OK] Registered {name} ({reg_no})  ->  DB id={person_id}"))
        except Exception as exc:
            print(_r(f"\n  [ERROR] {exc}"))
            return 1

    engine.load_gallery()
    print(f"  Gallery now has {engine.recognizer.gallery_size()} person(s).\n")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: monitor
# ---------------------------------------------------------------------------
def cmd_monitor(args: argparse.Namespace) -> int:
    """Live camera loop — detect faces, recognize, mark attendance."""
    conn   = init_db(args.db)
    engine = AttendanceEngine(conn, use_cuda=args.cuda)
    n      = engine.load_gallery()

    if n == 0:
        print(_y("[WARN] Gallery is empty. Unknown faces will be auto-registered."))
        print("       Use  python cli/main.py register --name NAME --reg-no REG_NO\n")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(_r(f"[ERROR] Cannot open camera index {args.camera}"))
        return 1

    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height or 480)

    print(_b(f"\n  [MONITOR] Starting camera {args.camera}"))
    print(f"  Gallery: {n} person(s)  |  Attendance: period-based (once per class period)")
    print(f"  Press Q or ESC to stop.\n")

    frame_count   = 0
    process_every = args.every
    alert_frames  = 0   # remaining frames to show alert overlay

    while True:
        ret, frame = cap.read()
        if not ret:
            print(_r("  [ERROR] Camera read failed."))
            break

        frame_count += 1
        display = frame.copy()

        if frame_count % process_every == 0:
            results = engine.process_frame(frame)

            for res in results:
                x1, y1, x2, y2 = res["bbox"]
                matched = res["matched"]
                is_new_unknown = res.get("alert", False)

                # Box color: green=known, orange=new-unknown auto-registered, red=unknown
                if is_new_unknown:
                    color = (0, 128, 255)   # orange for newly auto-registered
                elif matched:
                    color = (0, 200, 0)     # green for known
                else:
                    color = (0, 0, 220)     # red fallback

                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

                label = res["name"]
                if matched and not is_new_unknown:
                    label += f" {res['confidence']:.2f}"
                if res.get("logged"):
                    label += " [✓]"

                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(display, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                cv2.putText(display, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

                period = res.get("period") or "(outside period)"

                if is_new_unknown:
                    snap = res.get("snapshot_path")
                    sound_alert()
                    console_alert(snap)
                    print(_r(f"  [UNKNOWN PERSON]   Detected (picture stored locally)"))
                    if snap:
                        print(_r(f"  [PICTURE SAVED]    {snap}"))
                    alert_frames = 60
                elif matched and res.get("logged"):
                    print(_g(f"  [IN-TIME LOGGED]   {res['name']:20s} {res['reg_no']:14s}  conf={res['confidence']:.3f}  {period}"))
                elif matched:
                    reason = res.get("log_reason", "")
                    if "Out-time" in reason:
                        print(_c(f"  [OUT-TIME]         {res['name']:20s} {res['reg_no']:14s}  {reason}"))
                    else:
                        print(f"  [STATUS]           {res['name']:20s} {res['reg_no']:14s}  {reason}")


        # --- Alert overlay ---
        if alert_frames > 0:
            h, w = display.shape[:2]
            cv2.rectangle(display, (0, 0), (w, 55), (0, 70, 200), -1)
            cv2.putText(display, "!! UNKNOWN PERSON DETECTED — PICTURE SAVED !!",
                        (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            alert_frames -= 1

        # --- HUD (period + frame counter) ---
        cur_period = period_label()
        hud = f"{cur_period}  |  Gallery: {n}  |  Frame: {frame_count}  |  Q=quit"
        cv2.putText(display, hud, (8, display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        cv2.imshow("Campus Monitor", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n  Monitor stopped.")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: export
# ---------------------------------------------------------------------------
def cmd_export(args: argparse.Namespace) -> int:
    """Export attendance records to Excel."""
    conn = init_db(args.db)

    if args.start and args.end:
        path = export_range(conn, args.start, args.end, args.output)
    else:
        path = export_daily(conn, args.output, args.date)

    print(_g(f"  [OK] Exported -> {path}"))
    return 0


# ---------------------------------------------------------------------------
# Subcommand: export-alerts
# ---------------------------------------------------------------------------
def cmd_export_alerts(args: argparse.Namespace) -> int:
    """Export unknown-person alert records to a red-themed Excel file."""
    conn = init_db(args.db)

    if args.start and args.end:
        records = crud.get_unknown_alerts_range(conn, args.start, args.end)
        label   = f"{args.start}_to_{args.end}"
        title   = f"Unknown Person Alerts — {args.start} to {args.end}"
    else:
        from datetime import date as _date
        target = args.date or _date.today().isoformat()
        records = crud.get_unknown_alerts_for_date(conn, target)
        label   = target
        title   = f"Unknown Person Alerts — {target}"

    if not records:
        print(_y(f"  [INFO] No unknown-person alerts found for the requested period."))
        return 0

    path = export_alerts(records, args.output, label, title)
    print(_g(f"  [OK] Alerts exported -> {path}"))
    return 0


# ---------------------------------------------------------------------------
# Subcommand: list
# ---------------------------------------------------------------------------
def cmd_list(args: argparse.Namespace) -> int:
    """Print all registered students."""
    conn = init_db(args.db)
    persons = crud.get_all_persons(conn)

    if not persons:
        print(_y("  No registered persons found."))
        print("  Run:  python cli/main.py register --name NAME --reg-no REG_NO")
        return 0

    print(_b(f"\n  {'ID':>4}  {'Name':<25}  {'Reg No':<14}  {'Registered At'}"))
    print("  " + "-" * 65)
    for p in persons:
        print(f"  {p['id']:>4}  {p['name']:<25}  {p['reg_no']:<14}  {p['registered_at']}")
    print(f"\n  Total: {len(persons)} person(s)\n")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: report
# ---------------------------------------------------------------------------
def cmd_report(args: argparse.Namespace) -> int:
    """Print attendance summary for a given date."""
    conn = init_db(args.db)
    summary = crud.get_daily_summary(conn, args.date)

    inside_count = sum(1 for r in summary["records"] if not r.get("out_time"))

    print(_b(f"\n  Attendance Report — {summary['date']}"))
    print("  " + "=" * 55)
    print(f"  Present (unique): {_g(str(summary['total_present']))}")
    print(f"  Persons Inside:   {_c(str(inside_count))}")
    print(f"  Unknown faces:    {_r(str(summary['total_unknown']))}")
    print()

    if summary["records"]:
        print(f"  {'Name':<22}  {'Reg No':<14}  {'In Time':<10}  {'Out Time':<10}  {'Inside'}")
        print("  " + "-" * 70)
        for r in summary["records"]:
            try:
                from datetime import datetime as dt
                in_t = dt.strptime(r["timestamp"], "%Y-%m-%d %H:%M:%S").strftime("%H:%M:%S")
            except Exception:
                in_t = r.get("timestamp") or "—"

            out_raw = r.get("out_time")
            if out_raw:
                try:
                    from datetime import datetime as dt
                    out_t = dt.strptime(out_raw, "%Y-%m-%d %H:%M:%S").strftime("%H:%M:%S")
                except Exception:
                    out_t = str(out_raw)
                inside_str = "No"
            else:
                out_t = "—"
                inside_str = "Yes"

            print(f"  {r['name']:<22}  {r['reg_no']:<14}  {in_t:<10}  {out_t:<10}  {inside_str}")
        print("  " + "-" * 70)
        print(f"  Total Persons Inside: {_g(str(inside_count))}")
    else:
        print(_y("  No attendance records for this date."))

    print()
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="campus_monitor",
        description="Campus face recognition attendance system",
    )
    parser.add_argument(
        "--db", default=None,
        help="Path to SQLite database (default: data/attendance.db)"
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # -- register --
    p_reg = sub.add_parser("register", help="Register a new student")
    p_reg.add_argument("--name",    required=True, help="Student full name")
    p_reg.add_argument("--reg-no",  required=True, dest="reg_no", help="Registration number")
    p_reg.add_argument("--camera",  type=int, default=0, help="Camera index (default 0)")
    p_reg.add_argument("--samples", type=int, default=5,
                       help="Number of face samples to average (default 5)")
    p_reg.add_argument("--cuda",    action="store_true", help="Use CUDA backend")
    p_reg.add_argument("--force",   action="store_true", help="Overwrite existing registration")
    p_reg.set_defaults(func=cmd_register)

    # -- monitor --
    p_mon = sub.add_parser("monitor", help="Run live attendance monitoring")
    p_mon.add_argument("--camera", type=int,   default=0,    help="Camera index")
    p_mon.add_argument("--every",  type=int,   default=3,    help="Process every Nth frame")
    p_mon.add_argument("--width",  type=int,   default=None, help="Camera capture width")
    p_mon.add_argument("--height", type=int,   default=None, help="Camera capture height")
    p_mon.add_argument("--cuda",   action="store_true",      help="Use CUDA backend")
    p_mon.set_defaults(func=cmd_monitor)

    # -- export --
    p_exp = sub.add_parser("export", help="Export attendance to Excel")
    p_exp.add_argument("--date",   default=None, help="YYYY-MM-DD (default: today)")
    p_exp.add_argument("--start",  default=None, help="Start date for range export")
    p_exp.add_argument("--end",    default=None, help="End date for range export")
    p_exp.add_argument("--output", default=None, help="Output .xlsx path")
    p_exp.set_defaults(func=cmd_export)

    # -- export-alerts --
    p_eal = sub.add_parser("export-alerts", help="Export unknown-person alerts to Excel")
    p_eal.add_argument("--date",   default=None, help="YYYY-MM-DD (default: today)")
    p_eal.add_argument("--start",  default=None, help="Start date for range export")
    p_eal.add_argument("--end",    default=None, help="End date for range export")
    p_eal.add_argument("--output", default=None, help="Output .xlsx path")
    p_eal.set_defaults(func=cmd_export_alerts)

    # -- list --
    p_lst = sub.add_parser("list", help="List all registered students")
    p_lst.set_defaults(func=cmd_list)

    # -- report --
    p_rep = sub.add_parser("report", help="Print attendance summary")
    p_rep.add_argument("--date", default=None, help="YYYY-MM-DD (default: today)")
    p_rep.set_defaults(func=cmd_report)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
