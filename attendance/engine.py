"""
attendance/engine.py
====================
Orchestrates the full pipeline for one frame:
    detect faces → embed → recognize → auto-register unknowns → log attendance

Unknown persons are automatically registered in the database so that the
same face will be recognised (and not re-alerted) on subsequent frames.
Attendance is logged at most once per class period per person.

This is the single entry point called by the CLI monitor loop.
It does NOT touch the camera — that is the CLI's responsibility.

Python 3.6+ compatible.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from attendance.periods import get_current_period
from core.detector import FaceDetector, PersonDetector
from core.embedder import FaceEmbedder
from core.recognizer import FaceRecognizer
from db import crud

_PROJECT_ROOT = Path(__file__).parent.parent
_ALERTS_DIR   = _PROJECT_ROOT / "data" / "alerts"

logger = logging.getLogger(__name__)


class AttendanceEngine:
    """
    Full detect → embed → recognize → auto-register unknowns → log pipeline.

    Unknown faces are automatically registered in the database (name:
    ``Unknown_<timestamp>``, reg_no: ``UNK_<timestamp>``) so the camera
    will recognise the same person on the next frame and not alert again.

    Attendance is logged at most **once per class period** per person.

    Usage::

        engine = AttendanceEngine(conn)
        engine.load_gallery()   # call once at startup

        # In the camera loop:
        results = engine.process_frame(frame)

    Args:
        conn:     Open sqlite3.Connection (from db.schema.init_db()).
        use_cuda: Pass True on Jetson Nano with JetPack CUDA OpenCV.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        use_cuda: bool = False,
    ) -> None:
        self._conn = conn

        self.face_detector   = FaceDetector(use_cuda=use_cuda)
        self.person_detector = PersonDetector(use_cuda=use_cuda)
        self.embedder        = FaceEmbedder(use_cuda=use_cuda)
        self.recognizer      = FaceRecognizer()

        # In-memory cooldown tracking for unknown face snapshots
        self._unknown_cooldown = 30.0  # seconds between alerts for the same unknown face
        self._recent_unknowns: List[Tuple[np.ndarray, float]] = []

        logger.info("AttendanceEngine initialized")

    def load_gallery(self) -> int:
        """
        Load all registered persons from DB into the recognizer gallery.
        Call this once at startup, and again after `register` adds a new person.

        Returns:
            Number of persons loaded.
        """
        persons = crud.get_all_persons(self._conn)
        self.recognizer.load_gallery(persons)
        return self.recognizer.gallery_size()

    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Run the full pipeline on a single BGR frame.

        Steps:
          1. Determine current class period
          2. Detect persons (HOG) — for headcount
          3. Detect faces (YuNet)
          4. Embed each detected face (SFace)
          5. Recognize each embedding against gallery
          6a. Known person   → log attendance for current period (once per period)
          6b. Unknown person → auto-register + reload gallery + alert + log attendance

        Args:
            frame: BGR image from cv2.VideoCapture.

        Returns:
            List of result dicts per detected face::

              {
                "bbox"          : (x1, y1, x2, y2),
                "name"          : str,              # registered or auto-generated name
                "reg_no"        : str,
                "person_id"     : int | None,
                "confidence"    : float,
                "matched"       : bool,
                "logged"        : bool,
                "log_reason"    : str,
                "alert"         : bool,              # True when new unknown auto-registered
                "snapshot_path" : str | None,        # face-crop path for new unknowns
                "period"        : str | None,        # active period name, or None
              }
        """
        results = []

        # Step 1: Determine active period
        period      = get_current_period()
        period_name = period["name"] if period else None

        # Step 2: Person headcount (non-blocking — HOG may miss headshots)
        persons = self.person_detector.detect(frame)
        if persons:
            logger.debug("Person headcount: %d", len(persons))

        # Step 3: Face detection
        faces = self.face_detector.detect(frame)
        if len(faces) == 0:
            return results

        # Step 4: Embed all faces in one pass
        embeddings = self.embedder.embed_batch(frame, faces)

        # Step 5 + 6: Recognize and log
        bboxes               = self.face_detector.bboxes(faces)
        recognition_results  = self.recognizer.recognize_batch(embeddings)

        for i, rec in enumerate(recognition_results):
            logged        = False
            log_reason    = "Not matched"
            alert         = False
            snapshot_path: Optional[str] = None

            if rec["matched"]:
                # --- Known person: log attendance (records in-time, updates out-time on repeat) ---
                logged, log_reason = crud.log_attendance(
                    self._conn,
                    person_id=rec["person_id"],
                    confidence=rec["confidence"],
                    period_name=period_name,
                )
            else:
                # --- Unknown person: take picture and store locally; do NOT log to attendance/excel ---
                is_new, snap = self._capture_unknown(frame, faces[i], embeddings[i])
                if is_new and snap:
                    alert = True
                    snapshot_path = snap
                    log_reason = "Unknown person (picture saved locally)"
                else:
                    log_reason = "Unknown person (already captured)"

                rec = {
                    **rec,
                    "name": "Unknown",
                    "reg_no": "",
                    "person_id": None,
                    "matched": False,
                }

            results.append({
                "bbox":          bboxes[i],
                "name":          rec["name"],
                "reg_no":        rec["reg_no"],
                "person_id":     rec["person_id"],
                "confidence":    rec["confidence"],
                "matched":       rec["matched"],
                "logged":        logged,
                "log_reason":    log_reason,
                "alert":         alert,
                "snapshot_path": snapshot_path,
                "period":        period_name,
            })

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _capture_unknown(
        self,
        frame:     np.ndarray,
        face,
        embedding: np.ndarray,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check cooldown, crop and save unknown face snapshot locally to data/alerts/.

        Returns:
            ``(is_new_unknown, snapshot_path)``
        """
        import time as _time
        now = _time.time()
        # Clean up entries older than cooldown
        self._recent_unknowns = [
            (emb, ts) for (emb, ts) in self._recent_unknowns
            if (now - ts) < self._unknown_cooldown
        ]

        # Check if matched against any recent unknown embedding
        for emb, _ in self._recent_unknowns:
            sim = float(np.dot(embedding, emb) / (
                np.linalg.norm(embedding) * np.linalg.norm(emb) + 1e-10
            ))
            if sim >= 0.363:
                return False, None

        # New unknown person: capture and save picture locally
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        snap = self._save_face_crop(frame, face, filename=f"unknown_{ts_str}.jpg")
        self._recent_unknowns.append((embedding, now))
        logger.info("Saved picture of unknown person to %s", snap)
        return True, snap
    # ------------------------------------------------------------------
    def _auto_register_unknown(
        self,
        frame:     np.ndarray,
        face,
        embedding: np.ndarray,
    ) -> Tuple[Optional[int], str, str, Optional[str]]:
        """Crop, save, and register an unknown face automatically.

        Args:
            frame:     Full BGR frame from the camera.
            face:      YuNet face detection row.
            embedding: Pre-computed SFace embedding for this face.

        Returns:
            ``(person_id, name, reg_no, snapshot_path)``
            or ``(None, '', '', None)`` on failure.
        """
        ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
        name   = f"Unknown_{ts}"
        reg_no = f"UNK_{ts}"

        # Save face crop
        snap = self._save_face_crop(frame, face, filename=f"unknown_{ts}.jpg")

        try:
            person_id = crud.register_person(
                self._conn, name, reg_no, embedding, snap
            )
            logger.info(
                "Auto-registered unknown: %s (%s) → id=%d",
                name, reg_no, person_id,
            )
            return person_id, name, reg_no, snap
        except Exception as exc:
            logger.warning("Auto-registration failed: %s", exc)
            return None, "", "", None

    def _save_face_crop(
        self,
        frame:    np.ndarray,
        face,
        filename: str,
    ) -> Optional[str]:
        """Align-crop a face and save it to ``data/alerts/``.

        Returns absolute path to the saved file, or None on failure.
        """
        try:
            _ALERTS_DIR.mkdir(parents=True, exist_ok=True)
            aligned = self.embedder._recognizer.alignCrop(frame, face)
            path    = str(_ALERTS_DIR / filename)
            cv2.imwrite(path, aligned)
            return path
        except Exception as exc:
            logger.warning("Could not save face crop: %s", exc)
            return None

    def register_person_from_frame(
        self,
        frame: np.ndarray,
        name: str,
        reg_no: str,
        face_save_dir: Optional[str] = None,
        n_samples: int = 1,
    ) -> Dict:
        """
        Detect, embed, and register a single person from a frame.

        If n_samples > 1, call this method multiple times and pass the
        averaged embedding via db.crud.update_embedding() for better accuracy.
        This method handles the single-shot case.

        Args:
            frame:         BGR image containing exactly one prominent face.
            name:          Student's full name.
            reg_no:        Student's registration number.
            face_save_dir: If given, saves the aligned face crop as PNG.
            n_samples:     Ignored here; documented for multi-shot workflow.

        Returns:
            {success: bool, person_id: int | None, message: str}
        """
        faces = self.face_detector.detect(frame)
        if len(faces) == 0:
            return {"success": False, "person_id": None, "message": "No face detected in frame."}

        if len(faces) > 1:
            logger.warning("Multiple faces detected during registration. Using largest.")
            # Pick face with largest bounding box area
            areas = [(f[2] * f[3]) for f in faces]
            best = int(np.argmax(areas))
            faces = faces[best:best+1]

        embedding = self.embedder.embed_face(frame, faces[0])

        # Optionally save face crop
        face_path = None
        if face_save_dir:
            from pathlib import Path
            save_dir = Path(face_save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            aligned = self.embedder._recognizer.alignCrop(frame, faces[0])
            face_path = str(save_dir / f"{reg_no}.png")
            cv2.imwrite(face_path, aligned)

        try:
            person_id = crud.register_person(
                self._conn, name, reg_no, embedding, face_path
            )
            # Reload gallery so the new person is immediately recognizable
            self.load_gallery()
            return {
                "success": True,
                "person_id": person_id,
                "message": f"Registered {name} ({reg_no}) successfully.",
            }
        except Exception as exc:
            return {"success": False, "person_id": None, "message": str(exc)}
