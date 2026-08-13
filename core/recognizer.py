"""
core/recognizer.py
==================
Matches a face embedding against all registered persons in the database.

No machine learning framework — pure numpy cosine similarity against
stored embeddings loaded from SQLite.

Python 3.6+ compatible.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.embedder import COSINE_THRESHOLD

logger = logging.getLogger(__name__)

# Return type for a recognition result
RecognitionResult = Dict  # {name, reg_no, person_id, confidence, matched: bool}


class FaceRecognizer:
    """
    Compares a query embedding against a gallery of registered embeddings.

    The gallery is loaded once from the database and cached in memory.
    Call reload() to refresh when new students are registered mid-session.

    Args:
        threshold: Cosine similarity threshold for positive match.
                   Default: 0.363 (SFace paper recommendation).
    """

    def __init__(self, threshold: float = COSINE_THRESHOLD) -> None:
        self._threshold = threshold
        self._gallery: List[Dict] = []   # [{name, reg_no, person_id, embedding}]
        logger.info("FaceRecognizer initialized (threshold=%.3f)", threshold)

    # ------------------------------------------------------------------
    # Gallery management
    # ------------------------------------------------------------------
    def load_gallery(self, persons: List[Dict]) -> None:
        """
        Load registered persons into memory.

        Args:
            persons: List of person dicts from db.crud.get_all_persons().
                     Each dict must have keys: id, name, reg_no, embedding.
        """
        self._gallery = [
            {
                "person_id": p["id"],
                "name": p["name"],
                "reg_no": p["reg_no"],
                "embedding": p["embedding"].astype(np.float32),
            }
            for p in persons
        ]
        logger.info("Gallery loaded: %d registered persons", len(self._gallery))

    def gallery_size(self) -> int:
        return len(self._gallery)

    # ------------------------------------------------------------------
    # Recognition
    # ------------------------------------------------------------------
    def recognize(self, query_embedding: np.ndarray) -> RecognitionResult:
        """
        Find the best matching person for a query face embedding.

        Args:
            query_embedding: 128-d float32 array from FaceEmbedder.embed_face().

        Returns:
            Dict with keys:
              name       : str  — recognized name, or "Unknown"
              reg_no     : str  — registration number, or ""
              person_id  : int | None
              confidence : float — best cosine similarity score
              matched    : bool — True if confidence >= threshold
        """
        if not self._gallery:
            return _unknown_result(0.0)

        query = query_embedding.astype(np.float32).flatten()
        q_norm = query / (np.linalg.norm(query) + 1e-10)

        # Vectorized cosine similarity against all gallery embeddings
        gallery_matrix = np.stack(
            [g["embedding"] / (np.linalg.norm(g["embedding"]) + 1e-10)
             for g in self._gallery]
        )  # shape (N, 128)

        similarities = gallery_matrix @ q_norm   # shape (N,)
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        if best_score >= self._threshold:
            match = self._gallery[best_idx]
            logger.debug(
                "Recognized: %s (%s) score=%.3f",
                match["name"], match["reg_no"], best_score
            )
            return {
                "name": match["name"],
                "reg_no": match["reg_no"],
                "person_id": match["person_id"],
                "confidence": best_score,
                "matched": True,
            }

        logger.debug("Unknown face. Best score=%.3f < threshold=%.3f", best_score, self._threshold)
        return _unknown_result(best_score)

    def recognize_batch(
        self, embeddings: np.ndarray
    ) -> List[RecognitionResult]:
        """
        Recognize multiple faces at once (from a single frame).

        Args:
            embeddings: shape (N, 128) array from FaceEmbedder.embed_batch().

        Returns:
            List of N RecognitionResult dicts, one per face.
        """
        return [self.recognize(embeddings[i]) for i in range(len(embeddings))]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _unknown_result(score: float) -> RecognitionResult:
    return {
        "name": "Unknown",
        "reg_no": "",
        "person_id": None,
        "confidence": score,
        "matched": False,
    }
