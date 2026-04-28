from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np


def process_video(
    pipeline,
    video_path: Path,
    output_dir: Path,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, Path]:
    """Run *pipeline* on every frame of *video_path* and write the results to *output_dir*.

    Produces two videos:
        mask.mp4      — final alpha matte (greyscale, displayed as colour).
        composite.mp4 — subject composited on a black background.

    Args:
        pipeline:    A :class:`MattingPipeline` instance (already loaded).
        video_path:  Path to the source video.
        output_dir:  Folder that will receive the two output videos (created if absent).
        on_progress: Optional callback called after each frame with (frame_idx, total_frames).

    Returns:
        Dict with keys ``"mask"`` and ``"composite"`` pointing to the written files.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_dir.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    mask_path = output_dir / "mask.mp4"
    composite_path = output_dir / "composite.mp4"

    mask_writer = cv2.VideoWriter(str(mask_path), fourcc, fps, (w, h))
    composite_writer = cv2.VideoWriter(str(composite_path), fourcc, fps, (w, h))

    idx = 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            result = pipeline.process_frame(frame_rgb)

            # Mask: float32 [0,1] → uint8 grey → 3-channel for the writer
            mask_uint8 = (result["final_mask"] * 255).astype(np.uint8)
            mask_writer.write(cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2BGR))

            composite_writer.write(cv2.cvtColor(result["final"], cv2.COLOR_RGB2BGR))

            idx += 1
            if on_progress:
                on_progress(idx, total)
    finally:
        cap.release()
        mask_writer.release()
        composite_writer.release()

    return {"mask": mask_path, "composite": composite_path}
