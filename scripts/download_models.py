"""
scripts/download_models.py
==========================
Downloads YuNet (face detection) and SFace (face embedding) ONNX models
from the official OpenCV Zoo GitHub repository.

Usage:
    python scripts/download_models.py

No arguments needed. Models are saved to:
    models/face_detector/face_detection_yunet_2023mar.onnx
    models/embedder/face_recognition_sface_2021dec.onnx

JETSON NANO WARNING:
    This script downloads model files only. It does NOT install or reinstall
    OpenCV. Do NOT run 'pip install opencv-contrib-python' on Jetson Nano —
    that will override the CUDA-enabled OpenCV installed by JetPack.
"""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from pathlib import Path
from typing import Dict, NamedTuple

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
class ModelSpec(NamedTuple):
    url: str
    dest: str          # relative to project root
    sha256: str        # expected SHA-256 (first 16 chars sufficient for sanity)


_PROJECT_ROOT = Path(__file__).parent.parent

MODELS: Dict[str, ModelSpec] = {
    "yunet_face_detector": ModelSpec(
        url=(
            "https://github.com/opencv/opencv_zoo/raw/main/models/"
            "face_detection_yunet/face_detection_yunet_2023mar.onnx"
        ),
        dest="models/face_detector/face_detection_yunet_2023mar.onnx",
        sha256="9b0b9fc4e1f21ab7b53b516d8ba77d79",  # MD5-like prefix for quick check
    ),
    "sface_embedder": ModelSpec(
        url=(
            "https://github.com/opencv/opencv_zoo/raw/main/models/"
            "face_recognition_sface/face_recognition_sface_2021dec.onnx"
        ),
        dest="models/embedder/face_recognition_sface_2021dec.onnx",
        sha256="",  # skip hash check for this one; size check instead
    ),
}

# Minimum expected file sizes in bytes (sanity check, not security)
MIN_SIZES: Dict[str, int] = {
    "yunet_face_detector": 300_000,   # ~360 KB
    "sface_embedder":       37_000_000,  # ~37 MB
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded / total_size * 100)
        bar_len = 40
        filled = int(bar_len * pct / 100)
        bar = "=" * filled + "-" * (bar_len - filled)
        mb_done = downloaded / 1_048_576
        mb_total = total_size / 1_048_576
        sys.stdout.write(f"\r  [{bar}] {pct:5.1f}%  {mb_done:.1f}/{mb_total:.1f} MB")
        sys.stdout.flush()
    if downloaded >= total_size > 0:
        sys.stdout.write("\n")


def _download(name: str, spec: ModelSpec) -> None:
    dest = _PROJECT_ROOT / spec.dest
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        size = dest.stat().st_size
        min_size = MIN_SIZES.get(name, 0)
        if size >= min_size:
            print(f"  [SKIP] {dest.name} already present ({size / 1_048_576:.1f} MB)")
            return
        else:
            print(f"  [WARN] {dest.name} exists but is too small ({size} bytes). Re-downloading.")
            dest.unlink()

    print(f"  Downloading {name} ...")
    print(f"  URL: {spec.url}")
    try:
        urllib.request.urlretrieve(spec.url, str(dest), reporthook=_progress_hook)
    except Exception as exc:
        print(f"\n  [ERROR] Download failed: {exc}")
        if dest.exists():
            dest.unlink()
        raise

    size = dest.stat().st_size
    print(f"  [OK] Saved to {dest}  ({size / 1_048_576:.1f} MB)")

    # Sanity size check
    min_size = MIN_SIZES.get(name, 0)
    if size < min_size:
        dest.unlink()
        raise RuntimeError(
            f"Downloaded file is too small ({size} bytes < {min_size} expected). "
            "Possibly a partial download or 404 error page saved as file."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("Campus Monitor — Model Download Script")
    print("=" * 60)
    print()

    # Jetson warning
    jetson_hint = (
        "\n  JETSON NANO: This script downloads ONNX model files only.\n"
        "  Do NOT run 'pip install opencv-contrib-python' on Jetson Nano —\n"
        "  it will replace your CUDA-enabled JetPack OpenCV build.\n"
    )
    if sys.platform.startswith("linux") and os.path.exists("/etc/nv_tegra_release"):
        print("[!] Jetson device detected.")
        print(jetson_hint)

    errors = []
    for name, spec in MODELS.items():
        print(f"\n[{name}]")
        try:
            _download(name, spec)
        except Exception as exc:
            errors.append((name, str(exc)))

    print()
    if errors:
        print("[FAIL] Some models failed to download:")
        for name, msg in errors:
            print(f"  - {name}: {msg}")
        sys.exit(1)
    else:
        print("[OK] All models downloaded successfully.")
        print()
        print("Next step:")
        print("  python tests/test_stage1.py --image <path_to_face_image>")


if __name__ == "__main__":
    main()
