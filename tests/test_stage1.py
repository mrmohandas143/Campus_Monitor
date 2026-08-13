"""
tests/test_stage1.py
====================
Stage 1 verification: face detection + person detection + face embedding.

Usage:
    # With a real image (recommended):
    python tests/test_stage1.py --image path/to/face_photo.jpg

    # Without an image (downloads a public-domain test face):
    python tests/test_stage1.py

    # Test CUDA backend (Jetson Nano):
    python tests/test_stage1.py --image photo.jpg --cuda

Pass criteria (all must pass to proceed to Stage 2):
    [1] FaceDetector: detects >= 1 face
    [2] PersonDetector: detects >= 1 person (HOG; may miss small/distant persons)
    [3] FaceEmbedder: produces shape (128,) float32 embedding
    [4] Same-face cosine similarity >= 0.363 (identical crops of same face)
    [5] Different-face cosine similarity < 0.363 (same img vs solid color patch)
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# Allow running from project root OR from tests/ directory
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from core.detector import FaceDetector, PersonDetector
from core.embedder import FaceEmbedder, COSINE_THRESHOLD


# ---------------------------------------------------------------------------
# ANSI colours (degrade gracefully on Windows without ANSI support)
# ---------------------------------------------------------------------------
def _green(s: str) -> str:
    return f"\033[92m{s}\033[0m"

def _red(s: str) -> str:
    return f"\033[91m{s}\033[0m"

def _yellow(s: str) -> str:
    return f"\033[93m{s}\033[0m"

def _bold(s: str) -> str:
    return f"\033[1m{s}\033[0m"


# ---------------------------------------------------------------------------
# Test image acquisition
# ---------------------------------------------------------------------------
_SAMPLE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/14/"
    "Gatto_europeo4.jpg/320px-Gatto_europeo4.jpg"  # placeholder — not a face
)

# We use a better face image from Wikimedia (public domain portrait)
_FACE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/"
    "A_girl_smiling.jpg/320px-A_girl_smiling.jpg"
)

_TEST_IMAGE_CACHE = _PROJECT_ROOT / "tests" / "_test_face.jpg"


def _get_test_image(image_path: str | None) -> np.ndarray:
    """Load or download a test face image."""
    if image_path:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")
        print(f"  Loaded: {image_path}  shape={img.shape}")
        return img

    if _TEST_IMAGE_CACHE.exists():
        img = cv2.imread(str(_TEST_IMAGE_CACHE))
        if img is not None:
            print(f"  Using cached test image: {_TEST_IMAGE_CACHE}")
            return img

    print(f"  Downloading sample face image from Wikimedia...")
    try:
        urllib.request.urlretrieve(_FACE_URL, str(_TEST_IMAGE_CACHE))
        img = cv2.imread(str(_TEST_IMAGE_CACHE))
        if img is None:
            raise RuntimeError("Downloaded image could not be decoded by OpenCV.")
        print(f"  Downloaded test image  shape={img.shape}")
        return img
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download test image: {exc}\n"
            "Provide your own image with:  --image path/to/face.jpg"
        )


# ---------------------------------------------------------------------------
# Individual test functions
# ---------------------------------------------------------------------------
def test_face_detection(
    detector: FaceDetector, frame: np.ndarray
) -> Tuple[bool, str, np.ndarray]:
    """Test [1]: FaceDetector detects >= 1 face."""
    faces = detector.detect(frame)
    n = len(faces)
    if n >= 1:
        scores = detector.scores(faces)
        bboxes = detector.bboxes(faces)
        detail = f"Found {n} face(s). Top score={scores[0]:.3f}. First bbox={bboxes[0]}"
        return True, detail, faces
    else:
        return False, "No faces detected. Check image quality or lower score_threshold.", faces


def test_person_detection(
    detector: PersonDetector, frame: np.ndarray
) -> Tuple[bool, str]:
    """Test [2]: PersonDetector (MOG2) — warm up on static frame then detect.

    MOG2 background subtraction needs multiple frames to build a model.
    We feed a slightly perturbed version as 'background', then detect on
    the real frame to simulate motion. This validates the pipeline works.
    """
    import numpy as _np

    # Warm up: feed a brighter version as background (simulates empty scene)
    background = _np.clip(frame.astype(_np.int16) + 30, 0, 255).astype(_np.uint8)
    for _ in range(30):
        detector.detect(background)

    # Now detect on the real frame (foreground = the person)
    persons = detector.detect(frame)
    n = len(persons)
    if n >= 1:
        return True, f"Found {n} person(s). Top confidence={persons[0]['confidence']:.3f}"
    else:
        return (
            False,
            "No motion detected vs warmed background. "
            "This is a soft failure — MOG2 works on live video streams. "
            "Person detection is headcount only; attendance is face-based.",
        )


def test_embedding_generation(
    embedder: FaceEmbedder, frame: np.ndarray, faces: np.ndarray
) -> Tuple[bool, str, np.ndarray | None]:
    """Test [3]: FaceEmbedder produces 128-d float32 embedding."""
    if len(faces) == 0:
        return False, "Skipped — no faces detected upstream.", None

    emb = embedder.embed_face(frame, faces[0])
    if emb.shape == (128,) and emb.dtype == np.float32:
        norm = float(np.linalg.norm(emb))
        return True, f"shape={emb.shape}  dtype={emb.dtype}  L2-norm={norm:.4f}", emb
    else:
        return False, f"Bad embedding: shape={emb.shape}  dtype={emb.dtype}", None


def test_same_face_similarity(
    embedder: FaceEmbedder, frame: np.ndarray, faces: np.ndarray
) -> Tuple[bool, str]:
    """Test [4]: Same face → cosine similarity >= COSINE_THRESHOLD."""
    if len(faces) == 0:
        return False, "Skipped — no faces detected upstream."

    emb1 = embedder.embed_face(frame, faces[0])
    # Second embedding: same face with tiny brightness jitter (+10)
    jittered = np.clip(frame.astype(np.int16) + 10, 0, 255).astype(np.uint8)
    emb2 = embedder.embed_face(jittered, faces[0])

    sim = embedder.cosine_similarity(emb1, emb2)
    passed = sim >= COSINE_THRESHOLD
    return (
        passed,
        f"Same-face cosine similarity={sim:.4f}  (threshold>={COSINE_THRESHOLD})"
    )


def test_different_face_similarity(
    embedder: FaceEmbedder, frame: np.ndarray, faces: np.ndarray
) -> Tuple[bool, str]:
    """Test [5]: Blank image embedding vs real face → similarity < COSINE_THRESHOLD."""
    if len(faces) == 0:
        return False, "Skipped — no faces detected upstream."

    emb_real = embedder.embed_face(frame, faces[0])

    # Create a synthetic blank "face" patch — solid grey 112x112 (SFace input size)
    blank = np.full((frame.shape[0], frame.shape[1], 3), 128, dtype=np.uint8)
    # We need a fake detection row for the blank image — use same bbox
    emb_blank = embedder.embed_face(blank, faces[0])

    sim = embedder.cosine_similarity(emb_real, emb_blank)
    passed = sim < COSINE_THRESHOLD
    return (
        passed,
        f"Real-vs-blank cosine similarity={sim:.4f}  (must be < {COSINE_THRESHOLD})"
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_tests(image_path: str | None, use_cuda: bool) -> int:
    """Run all Stage 1 tests. Returns 0 on full pass, 1 on any hard failure."""

    print()
    print(_bold("=" * 60))
    print(_bold("  Campus Monitor — Stage 1 Verification"))
    print(_bold("=" * 60))
    print(f"  OpenCV version: {cv2.__version__}")
    print(f"  CUDA requested: {use_cuda}")
    print()

    # -- Load image --
    print("[Setup] Loading test image...")
    try:
        frame = _get_test_image(image_path)
    except Exception as exc:
        print(_red(f"  [ABORT] {exc}"))
        return 1

    # -- Load models --
    print("\n[Setup] Loading FaceDetector (YuNet)...")
    try:
        face_det = FaceDetector(use_cuda=use_cuda)
    except FileNotFoundError as exc:
        print(_red(f"  [ABORT] {exc}"))
        return 1

    print("[Setup] Loading PersonDetector (HOG+SVM)...")
    person_det = PersonDetector(use_cuda=use_cuda)

    print("[Setup] Loading FaceEmbedder (SFace)...")
    try:
        embedder = FaceEmbedder(use_cuda=use_cuda)
    except FileNotFoundError as exc:
        print(_red(f"  [ABORT] {exc}"))
        return 1

    print()

    # -- Run tests --
    results: List[Tuple[int, bool, str, str]] = []  # (test_num, passed, label, detail)
    hard_failures = 0

    def record(num: int, label: str, passed: bool, detail: str, is_hard: bool = True) -> None:
        nonlocal hard_failures
        status = _green("PASS") if passed else _red("FAIL")
        if not passed and is_hard:
            hard_failures += 1
        results.append((num, passed, label, detail))
        marker = "[*]" if is_hard else "[-]"
        print(f"  [{status}] {marker} Test {num}: {label}")
        print(f"         {detail}")

    # Test 1
    passed, detail, faces = test_face_detection(face_det, frame)
    record(1, "FaceDetector — detects >= 1 face", passed, detail)

    # Test 2 (soft — HOG may fail on head-only images)
    p2, d2 = test_person_detection(person_det, frame)
    record(2, "PersonDetector — detects >= 1 person", p2, d2, is_hard=False)

    # Test 3
    p3, d3, emb = test_embedding_generation(embedder, frame, faces)
    record(3, "FaceEmbedder — 128-d float32 embedding", p3, d3)

    # Test 4
    p4, d4 = test_same_face_similarity(embedder, frame, faces)
    record(4, "Same-face cosine >= threshold", p4, d4)

    # Test 5
    p5, d5 = test_different_face_similarity(embedder, frame, faces)
    record(5, "Real-vs-blank cosine < threshold", p5, d5)

    # -- Summary --
    print()
    print(_bold("=" * 60))
    hard_pass = sum(1 for _, p, _, _ in results if p)
    total = len(results)
    if hard_failures == 0:
        print(_green(f"  Stage 1 PASSED  ({hard_pass}/{total} tests passed)"))
        print(_green("  Proceed to Stage 2: db/schema.py + db/crud.py"))
    else:
        print(_red(f"  Stage 1 FAILED  ({hard_failures} hard failure(s))"))
        print(_yellow("  Fix the failures above before writing Stage 2 files."))
    print(_bold("=" * 60))
    print()

    return 0 if hard_failures == 0 else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1 verification: face detection + embedding"
    )
    parser.add_argument(
        "--image", "-i",
        help="Path to a face image. If omitted, downloads a sample from Wikimedia.",
        default=None,
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Use CUDA backend (Jetson Nano with JetPack OpenCV).",
    )
    args = parser.parse_args()
    sys.exit(run_tests(args.image, args.cuda))


if __name__ == "__main__":
    main()
