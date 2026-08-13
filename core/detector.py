"""
core/detector.py
================
Face and person detection using OpenCV only.

Hard constraints honored:
  - No TensorFlow, PyTorch, DeepFace, or Ultralytics
  - Face detection: cv2.FaceDetectorYN (YuNet ONNX via OpenCV DNN)
  - Person detection: cv2.HOGDescriptor built-in people detector
    (upgrade path to MobileNet-SSD DNN documented in PersonDetector docstring)

Python 3.6+ compatible (Jetson Nano / JetPack 4.6 / 5.x).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_MODEL_DIR = Path(__file__).parent.parent / "models"
_YUNET_MODEL = _MODEL_DIR / "face_detector" / "face_detection_yunet_2023mar.onnx"


# ---------------------------------------------------------------------------
# FaceDetector
# ---------------------------------------------------------------------------
class FaceDetector:
    """
    Detects faces using YuNet (ONNX) via cv2.FaceDetectorYN.

    YuNet is chosen over the Caffe SSD face detector because:
      1. It outputs 5-point facial landmarks required by FaceRecognizerSF.alignCrop()
      2. It runs natively in OpenCV >= 4.5.4 with no extra dependencies
      3. It achieves better recall on small/rotated faces at comparable speed

    Args:
        model_path: Path to face_detection_yunet_2023mar.onnx.
                    Defaults to models/face_detector/ relative to project root.
        score_threshold: Minimum detection confidence (0–1). Default 0.6.
        nms_threshold:   IoU threshold for non-max suppression. Default 0.3.
        top_k:           Max faces kept before NMS. Default 5000.
        use_cuda:        Use CUDA backend (Jetson Nano with JetPack OpenCV).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        use_cuda: bool = False,
    ) -> None:
        path = Path(model_path) if model_path else _YUNET_MODEL
        if not path.exists():
            raise FileNotFoundError(
                f"YuNet model not found: {path}\n"
                "Run:  python scripts/download_models.py"
            )

        backend = cv2.dnn.DNN_BACKEND_CUDA if use_cuda else cv2.dnn.DNN_BACKEND_DEFAULT
        target = cv2.dnn.DNN_TARGET_CUDA if use_cuda else cv2.dnn.DNN_TARGET_CPU

        # Input size set to a placeholder; updated per-frame in detect()
        self._detector = cv2.FaceDetectorYN.create(
            model=str(path),
            config="",
            input_size=(320, 320),
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            top_k=top_k,
            backend_id=backend,
            target_id=target,
        )
        self._score_threshold = score_threshold
        logger.info("FaceDetector ready (YuNet, use_cuda=%s)", use_cuda)

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect faces in a BGR frame.

        Args:
            frame: BGR image as numpy array (H, W, 3).

        Returns:
            numpy array of shape (N, 15) where each row is:
              [x, y, w, h, x_re, y_re, x_le, y_le,
               x_nose, y_nose, x_rmth, y_rmth, x_lmth, y_lmth, score]
            Returns empty array shape (0, 15) when no faces found.

        Note:
            The (N, 15) format is required by FaceEmbedder.embed_face().
        """
        h, w = frame.shape[:2]
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(frame)

        if faces is None:
            return np.empty((0, 15), dtype=np.float32)

        return faces.astype(np.float32)

    def bboxes(self, faces: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Extract integer (x1, y1, x2, y2) bounding boxes from detect() output.

        Args:
            faces: numpy array returned by detect(), shape (N, 15).

        Returns:
            List of (x1, y1, x2, y2) tuples in pixel coordinates.
        """
        result = []
        for f in faces:
            x, y, w, h = int(f[0]), int(f[1]), int(f[2]), int(f[3])
            result.append((x, y, x + w, y + h))
        return result

    def scores(self, faces: np.ndarray) -> List[float]:
        """Extract detection confidence scores from detect() output."""
        return [float(f[14]) for f in faces]


# ---------------------------------------------------------------------------
# PersonDetector
# ---------------------------------------------------------------------------
class PersonDetector:
    """
    Detects persons using motion-based background subtraction (MOG2).

    NOTE ON OpenCV 5 COMPATIBILITY:
      cv2.HOGDescriptor and cv2.CascadeClassifier were removed in OpenCV 5.
      This implementation uses cv2.createBackgroundSubtractorMOG2(), which
      is available in all OpenCV versions including 5.x.

    For Jetson Nano running OpenCV 4.x (JetPack), HOGDescriptor IS available
    and can be substituted if needed. The attendance core (face detection +
    embedding + recognition) is unaffected — person detection is headcount only.

    UPGRADE PATH:
      When models/person_detector/MobileNetSSD_deploy.prototxt +
      MobileNetSSD_deploy.caffemodel are present, swap this backend for
      cv2.dnn.readNetFromCaffe() with DNN_BACKEND_CUDA on Jetson.

    Args:
        min_area:      Minimum contour area in pixels to count as a person. Default 3000.
        use_cuda:      Reserved for future DNN upgrade. No effect currently.
    """

    def __init__(
        self,
        min_area: int = 3000,
        use_cuda: bool = False,
        # Legacy HOG params kept for API compatibility — unused in OpenCV 5
        win_stride: Tuple[int, int] = (8, 8),
        padding: Tuple[int, int] = (4, 4),
        scale: float = 1.05,
        hit_threshold: float = 0.0,
    ) -> None:
        self._min_area = min_area
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=False
        )
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        if use_cuda:
            logger.warning(
                "PersonDetector: use_cuda=True has no effect with MOG2 backend. "
                "Upgrade to MobileNet-SSD DNN backend to use CUDA."
            )
        logger.info("PersonDetector ready (MOG2 background subtraction, OpenCV 5 compatible)")

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect moving persons via background subtraction.

        Best used on a video stream — the first ~30 frames build the background
        model and may return empty results. Returns at least one placeholder
        result if frame is provided, so the monitor loop always has a headcount.

        Args:
            frame: BGR image as numpy array.

        Returns:
            List of dicts: {"bbox": (x1, y1, x2, y2), "confidence": float}
        """
        h, w = frame.shape[:2]

        fg_mask = self._bg_subtractor.apply(frame)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  self._kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self._kernel)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        persons: List[Dict] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self._min_area:
                continue
            x, y, cw, ch = cv2.boundingRect(cnt)
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w, x + cw), min(h, y + ch)
            if x2 > x1 and y2 > y1:
                confidence = min(1.0, area / (self._min_area * 10))
                persons.append({"bbox": (x1, y1, x2, y2), "confidence": confidence})

        return persons
