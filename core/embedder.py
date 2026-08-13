"""
core/embedder.py
================
Face embedding generation using cv2.FaceRecognizerSF (SFace ONNX).

Hard constraints honored:
  - No TensorFlow, PyTorch, DeepFace, or Ultralytics
  - Embedding model: SFace (face_recognition_sface_2021dec.onnx) loaded via OpenCV DNN
  - Alignment: cv2.FaceRecognizerSF.alignCrop() requires YuNet-format detections
    (the (N, 15) array from FaceDetector.detect())

SFace thresholds (from OpenCV Zoo paper):
  - Cosine similarity >= 0.363  → same person  (used for recognition)
  - L2 distance       <= 1.128  → same person  (alternative metric)

Python 3.6+ compatible (Jetson Nano / JetPack 4.6 / 5.x).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_MODEL_DIR = Path(__file__).parent.parent / "models"
_SFACE_MODEL = _MODEL_DIR / "embedder" / "face_recognition_sface_2021dec.onnx"

# ---------------------------------------------------------------------------
# SFace thresholds
# ---------------------------------------------------------------------------
COSINE_THRESHOLD = 0.363   # >= this → same person
L2_THRESHOLD = 1.128        # <= this → same person


# ---------------------------------------------------------------------------
# FaceEmbedder
# ---------------------------------------------------------------------------
class FaceEmbedder:
    """
    Generates 128-d face embeddings using OpenCV's SFace recognizer.

    Requirements:
      - OpenCV >= 4.5.4 (cv2.FaceRecognizerSF must be available)
      - face_recognition_sface_2021dec.onnx in models/embedder/

    The embed_face() method expects face detections in YuNet format (shape (15,)
    per face), which is produced directly by FaceDetector.detect(). This is
    intentional — YuNet provides the 5 landmarks (eye centers, nose tip, mouth
    corners) that SFace uses for geometric alignment before embedding.

    Args:
        model_path: Path to SFace ONNX file. Defaults to models/embedder/ path.
        use_cuda:   Use CUDA backend (Jetson Nano with JetPack OpenCV).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        use_cuda: bool = False,
    ) -> None:
        path = Path(model_path) if model_path else _SFACE_MODEL
        if not path.exists():
            raise FileNotFoundError(
                f"SFace model not found: {path}\n"
                "Run:  python scripts/download_models.py"
            )

        if use_cuda:
            backend = cv2.dnn.DNN_BACKEND_CUDA
            target = cv2.dnn.DNN_TARGET_CUDA
            self._recognizer = cv2.FaceRecognizerSF.create(
                model=str(path), config="", backend_id=backend, target_id=target
            )
        else:
            # Don't pass backend/target on CPU — OpenCV 5 graph engine warns on explicit CPU target
            self._recognizer = cv2.FaceRecognizerSF.create(
                model=str(path), config=""
            )
        logger.info("FaceEmbedder ready (SFace, use_cuda=%s)", use_cuda)

    # ------------------------------------------------------------------
    # Core embedding
    # ------------------------------------------------------------------
    def embed_face(self, frame: np.ndarray, face_row: np.ndarray) -> np.ndarray:
        """
        Align and embed a single face.

        Args:
            frame:    Full BGR frame (H, W, 3).
            face_row: Single detection row from FaceDetector.detect(), shape (15,).
                      Format: [x, y, w, h, x_re, y_re, x_le, y_le,
                               x_nose, y_nose, x_rmth, y_rmth, x_lmth, y_lmth, score]

        Returns:
            128-d float32 numpy array (L2-normalized by SFace internally).

        Raises:
            ValueError: If face_row has wrong shape.
        """
        if face_row.shape != (15,):
            raise ValueError(
                f"face_row must have shape (15,); got {face_row.shape}. "
                "Pass a single row from FaceDetector.detect()."
            )

        aligned = self._recognizer.alignCrop(frame, face_row)
        embedding = self._recognizer.feature(aligned)   # shape (1, 128)
        return embedding.flatten()                       # shape (128,)

    def embed_batch(
        self, frame: np.ndarray, faces: np.ndarray
    ) -> np.ndarray:
        """
        Embed all detected faces in a frame.

        Args:
            frame: Full BGR frame.
            faces: Detection array from FaceDetector.detect(), shape (N, 15).

        Returns:
            numpy array shape (N, 128). Empty array if no faces.
        """
        if len(faces) == 0:
            return np.empty((0, 128), dtype=np.float32)

        embeddings = [self.embed_face(frame, faces[i]) for i in range(len(faces))]
        return np.stack(embeddings, axis=0)

    # ------------------------------------------------------------------
    # Similarity metrics
    # ------------------------------------------------------------------
    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.

        Returns:
            float in [-1, 1]. Values >= COSINE_THRESHOLD (0.363) indicate same person.
        """
        e1 = emb1.reshape(1, -1).astype(np.float32)
        e2 = emb2.reshape(1, -1).astype(np.float32)
        return float(
            self._recognizer.match(e1, e2, cv2.FaceRecognizerSF_FR_COSINE)
        )

    def l2_distance(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute L2 distance between two embeddings.

        Returns:
            float >= 0. Values <= L2_THRESHOLD (1.128) indicate same person.
        """
        e1 = emb1.reshape(1, -1).astype(np.float32)
        e2 = emb2.reshape(1, -1).astype(np.float32)
        return float(
            self._recognizer.match(e1, e2, cv2.FaceRecognizerSF_FR_NORM_L2)
        )

    def is_same_person(
        self,
        emb1: np.ndarray,
        emb2: np.ndarray,
        metric: str = "cosine",
    ) -> Tuple[bool, float]:
        """
        Determine if two embeddings belong to the same person.

        Args:
            emb1, emb2: 128-d embeddings.
            metric:     "cosine" or "l2". Default "cosine".

        Returns:
            Tuple (is_same: bool, score: float).
            For cosine: score is similarity (higher = more similar).
            For l2:     score is distance (lower = more similar).
        """
        if metric == "cosine":
            score = self.cosine_similarity(emb1, emb2)
            return score >= COSINE_THRESHOLD, score
        elif metric == "l2":
            score = self.l2_distance(emb1, emb2)
            return score <= L2_THRESHOLD, score
        else:
            raise ValueError(f"Unknown metric '{metric}'. Use 'cosine' or 'l2'.")

    # ------------------------------------------------------------------
    # Serialization helpers (used by db/crud.py in Stage 2)
    # ------------------------------------------------------------------
    @staticmethod
    def serialize(embedding: np.ndarray) -> bytes:
        """Convert 128-d float32 embedding to bytes for SQLite BLOB storage."""
        return embedding.astype(np.float32).tobytes()

    @staticmethod
    def deserialize(blob: bytes) -> np.ndarray:
        """Restore 128-d float32 embedding from SQLite BLOB bytes."""
        return np.frombuffer(blob, dtype=np.float32)
