"""Pipeline configuration — root paths and supported video extensions."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
VIDEO_DIR = ROOT / "data" / "videos"
OUTPUT_DIR = ROOT / "data" / "output"
WEIGHTS_DIR = ROOT / "weights"

SUPPORTED_VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")
