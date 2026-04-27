from pathlib import Path

ROOT = Path(__file__).resolve().parent
VIDEO_DIR = ROOT / "data" / "videos"
WEIGHTS_DIR = ROOT / "weights"

SUPPORTED_VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")
