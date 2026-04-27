from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from config import SUPPORTED_VIDEO_EXTENSIONS


def list_videos(directory: Path) -> list[Path]:
    """Return a sorted list of video files in *directory*.

    Args:
        directory: Folder to scan.

    Returns:
        Sorted list of :class:`pathlib.Path` objects whose suffix (lowercased)
        is in :data:`SUPPORTED_VIDEO_EXTENSIONS`.
    """
    return sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
    )


def frame_count(video_path: Path) -> int:
    """Return the total number of frames in the video at *video_path*.

    Args:
        video_path: Path to the video file.

    Returns:
        Total frame count as an integer.
    """
    cap = cv2.VideoCapture(str(video_path))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def read_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    """Read a single frame from *video_path* and return it as an RGB uint8 array.

    Args:
        video_path: Path to the video file.
        frame_idx:  Zero-based frame index.

    Returns:
        RGB image, shape (H, W, 3), dtype uint8.

    Raises:
        IndexError: if *frame_idx* is out of bounds.
    """
    cap = cv2.VideoCapture(str(video_path))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_idx < 0 or frame_idx >= total:
            raise IndexError(
                f"frame_idx {frame_idx} is out of bounds for video with {total} frames: "
                f"{video_path}"
            )
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, bgr = cap.read()
        if not ok:
            raise IndexError(f"Failed to read frame {frame_idx} from {video_path}.")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()
