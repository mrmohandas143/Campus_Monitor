"""
attendance/engine.py
====================
Orchestrates the full pipeline for one frame:
    detect faces → embed → recognize → log attendance

This is the single entry point called by the CLI monitor loop.
It does NOT touch the camera — that is the CLI's responsibility.

Python 3.6+ compatible.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Dict, List, Optional

import cv2
import numpy as np

from core.detector import FaceDetector, PersonDetector
from core.embedder import FaceEmbedder
from core.recognizer import FaceRecognizer
from db import crud

logger = logging.getLogger(__name__)


class AttendanceEngine:
    """
    Full detect → embed → recognize → log pipeline.

    Usage:
        engine = AttendanceEngine(conn)
        engine.load_gallery()   # call once after init, or after registering new students

        # In the camera loop:
        results = engine.process_frame(frame)
        # results is a list of dicts per detected face with recognition + log outcome

    Args:
        conn:              Open sqlite3.Connection (from db.schema.init_db()).
        use_cuda:          Pass True on Jetson Nano with JetPack CUDA OpenCV.
        cooldown_seconds:  Seconds between duplicate attendance entries per person.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        use_cuda: bool = False,
        cooldown_seconds: int = crud.DEFAULT_COOLDOWN_SECONDS,
    ) -> None:
        self._conn = conn
        self._cooldown = cooldown_seconds

        self.face_detector = FaceDetector(use_cuda=use_cuda)
        self.person_detector = PersonDetector(use_cuda=use_cuda)
        self.embedder = FaceEmbedder(use_cuda=use_cuda)
        self.recognizer = FaceRecognizer()

        logger.info("AttendanceEngine initialized (cooldown=%ds)", cooldown_seconds)

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
          1. Detect persons (HOG) — for headcount
          2. Detect faces (YuNet)
          3. Embed each detected face (SFace)
          4. Recognize each embedding against gallery
          5. Log attendance for matched persons (with cooldown)

        Args:
            frame: BGR image from cv2.VideoCapture.

        Returns:
            List of result dicts per detected face:
              {
                "bbox"       : (x1, y1, x2, y2),     # face bounding box
                "name"       : str,                    # recognized name or "Unknown"
                "reg_no"     : str,
                "person_id"  : int | None,
                "confidence" : float,
                "matched"    : bool,
                "logged"     : bool,                   # True if attendance was logged
                "log_reason" : str,
              }
        """
        results = []

        # Step 1: Person headcount (non-blocking — HOG may miss headshots)
        persons = self.person_detector.detect(frame)
        if persons:
            logger.debug("Person headcount: %d", len(persons))

        # Step 2: Face detection
        faces = self.face_detector.detect(frame)
        if len(faces) == 0:
            return results

        # Step 3: Embed all faces in one pass
        embeddings = self.embedder.embed_batch(frame, faces)

        # Step 4 + 5: Recognize and log
        bboxes = self.face_detector.bboxes(faces)
        recognition_results = self.recognizer.recognize_batch(embeddings)

        for i, rec in enumerate(recognition_results):
            logged = False
            log_reason = "Not matched"

            if rec["matched"]:
                logged, log_reason = crud.log_attendance(
                    self._conn,
                    person_id=rec["person_id"],
                    confidence=rec["confidence"],
                    cooldown_seconds=self._cooldown,
                )
            else:
                crud.log_unknown(self._conn)
                log_reason = "Unknown face logged"

            results.append({
                "bbox": bboxes[i],
                "name": rec["name"],
                "reg_no": rec["reg_no"],
                "person_id": rec["person_id"],
                "confidence": rec["confidence"],
                "matched": rec["matched"],
                "logged": logged,
                "log_reason": log_reason,
            })

        return results

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
