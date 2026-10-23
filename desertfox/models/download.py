"""
ONNX Model Download Utility for DesertFox.

Downloads pre-trained SCRFD (face detection) and ArcFace (face embedding)
ONNX models from the InsightFace model zoo.

Usage:
    python3 -m desertfox.models.download                  # Download all models
    python3 -m desertfox.models.download --model scrfd     # Detection model only
    python3 -m desertfox.models.download --model arcface   # Embedding model only
    python3 -m desertfox.models.download --list            # List available models
"""

import os
import sys
import hashlib
import argparse
import logging
from typing import Dict, Optional
from urllib.request import urlretrieve
from urllib.error import URLError

logger = logging.getLogger(__name__)

# Default model storage directory
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "weights")

# Model registry: name → (url, filename, sha256, description)
MODEL_REGISTRY: Dict[str, dict] = {
    "scrfd_10g": {
        "url": "https://huggingface.co/immich-app/buffalo_l/resolve/main/detection/model.onnx",
        "filename": "scrfd_10g_bnkps.onnx",
        "sha256": None,
        "description": "SCRFD 10GF — High-accuracy face detector with keypoints (Buffalo_l)",
        "type": "detector",
    },
    "arcface_r50": {
        "url": "https://huggingface.co/immich-app/buffalo_l/resolve/main/recognition/model.onnx",
        "filename": "w600k_r50.onnx",
        "sha256": None,
        "description": "ArcFace ResNet-50 — 512-d face embedding (WebFace600K / Buffalo_l)",
        "type": "embedder",
    },
}

# Default model selections
DEFAULT_DETECTOR = "scrfd_10g"
DEFAULT_EMBEDDER = "arcface_r50"


def _progress_hook(block_num: int, block_size: int, total_size: int):
    """Download progress indicator."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 // total_size)
        mb_done = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\r    [{percent:3d}%] {mb_done:.1f} / {mb_total:.1f} MB")
    else:
        mb_done = downloaded / (1024 * 1024)
        sys.stdout.write(f"\r    {mb_done:.1f} MB downloaded...")
    sys.stdout.flush()


def _verify_sha256(filepath: str, expected: str) -> bool:
    """Verifies file SHA-256 checksum."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest() == expected


def download_model(
    model_name: str,
    model_dir: str = DEFAULT_MODEL_DIR,
    force: bool = False
) -> str:
    """
    Downloads an ONNX model from the registry.

    Args:
        model_name: Key in MODEL_REGISTRY (e.g., "scrfd_2.5g", "arcface_r100")
        model_dir: Directory to save the model file
        force: Re-download even if file exists

    Returns:
        str: Absolute path to the downloaded model file
    """
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")

    info = MODEL_REGISTRY[model_name]
    os.makedirs(model_dir, exist_ok=True)
    filepath = os.path.join(model_dir, info["filename"])

    if os.path.exists(filepath) and not force:
        print(f"  [✓] {model_name}: Already downloaded → {filepath}")
        return filepath

    print(f"  [↓] Downloading {model_name}: {info['description']}")
    print(f"      URL: {info['url']}")

    try:
        urlretrieve(info["url"], filepath, reporthook=_progress_hook)
        print()  # Newline after progress bar
    except (URLError, OSError) as e:
        print(f"\n  [✗] Download failed: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
        raise

    # Verify checksum if available
    if info["sha256"]:
        if _verify_sha256(filepath, info["sha256"]):
            print(f"  [✓] SHA-256 verified: {filepath}")
        else:
            os.remove(filepath)
            raise ValueError(f"SHA-256 mismatch for {model_name}! File may be corrupted.")
    else:
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  [✓] Downloaded: {filepath} ({size_mb:.1f} MB)")

    return filepath


def get_model_path(model_name: str, model_dir: str = DEFAULT_MODEL_DIR) -> Optional[str]:
    """Returns the path to a model if it exists locally, otherwise None."""
    if model_name not in MODEL_REGISTRY:
        return None
    filepath = os.path.join(model_dir, MODEL_REGISTRY[model_name]["filename"])
    return filepath if os.path.exists(filepath) else None


def get_default_detector_path(model_dir: str = DEFAULT_MODEL_DIR) -> Optional[str]:
    """Returns path to the default face detector model if available."""
    return get_model_path(DEFAULT_DETECTOR, model_dir)


def get_default_embedder_path(model_dir: str = DEFAULT_MODEL_DIR) -> Optional[str]:
    """Returns path to the default face embedder model if available."""
    return get_model_path(DEFAULT_EMBEDDER, model_dir)


def list_models():
    """Prints all available models in the registry."""
    print("\n  DesertFox ONNX Model Registry")
    print("  " + "=" * 70)
    for name, info in MODEL_REGISTRY.items():
        status = "✓ Downloaded" if get_model_path(name) else "✗ Not downloaded"
        default = ""
        if name == DEFAULT_DETECTOR:
            default = " [DEFAULT DETECTOR]"
        elif name == DEFAULT_EMBEDDER:
            default = " [DEFAULT EMBEDDER]"
        print(f"  {name}{default}")
        print(f"    {info['description']}")
        print(f"    Status: {status}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Download ONNX models for DesertFox face detection and embedding"
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY.keys()) + ["scrfd", "arcface", "all"],
        default="all",
        help="Which model(s) to download"
    )
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="Model storage directory")
    parser.add_argument("--force", action="store_true", help="Re-download even if already present")
    parser.add_argument("--list", action="store_true", help="List available models and exit")

    args = parser.parse_args()

    if args.list:
        list_models()
        return

    print("\n[*] DesertFox Model Downloader")
    print("=" * 50)

    if args.model == "all":
        targets = [DEFAULT_DETECTOR, DEFAULT_EMBEDDER]
    elif args.model == "scrfd":
        targets = [DEFAULT_DETECTOR]
    elif args.model == "arcface":
        targets = [DEFAULT_EMBEDDER]
    else:
        targets = [args.model]

    for model_name in targets:
        try:
            download_model(model_name, model_dir=args.model_dir, force=args.force)
        except Exception as e:
            print(f"  [!] Error downloading {model_name}: {e}")
            sys.exit(1)

    print("\n[✓] All models ready.\n")


if __name__ == "__main__":
    main()
