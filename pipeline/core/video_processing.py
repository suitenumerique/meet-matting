"""
Video processing utilities for running the pipeline on video files.
Includes support for frame skipping to speed up processing.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np


def process_video(
    pipeline,
    video_path: Path,
    output_dir: Path,
    on_progress: Callable[[int, int, float], None] | None = None,
    skip_frames: int = 1,
) -> dict[str, Path]:
    """Run *pipeline* on frames of *video_path* and write results to *output_dir*.

    Produces four videos:
        mask.mp4      -- final alpha matte (greyscale, shown as BGR).
        raw.mp4       -- composite using the raw mask (before postprocessing).
        composite.mp4 -- subject composited over the chosen background colour.
        original.mp4  -- unprocessed source frames.

    For skipped frames (idx % skip_frames != 0), the last computed mask is
    reused and re-composited onto the current source frame so the output video
    always runs at the original frame rate without frozen frames.

    Args:
        pipeline:    A MattingPipeline instance (already loaded).
        video_path:  Path to the source video.
        output_dir:  Folder that will receive the output videos (created if absent).
        on_progress: Optional callback called after each frame with (done, total, fps).
        skip_frames: Run the model 1 frame every N frames; reuse last mask in between.

    Returns:
        Dict with keys "mask", "raw", "composite", and "original".
    """
    cap = cv2.VideoCapture(str(video_path))
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_dir.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"avc1")

    mask_path      = output_dir / "mask.mp4"
    raw_path       = output_dir / "raw.mp4"
    composite_path = output_dir / "composite.mp4"
    original_path  = output_dir / "original.mp4"

    mask_writer      = cv2.VideoWriter(str(mask_path),      fourcc, orig_fps, (w, h))
    raw_writer       = cv2.VideoWriter(str(raw_path),       fourcc, orig_fps, (w, h))
    composite_writer = cv2.VideoWriter(str(composite_path), fourcc, orig_fps, (w, h))
    original_writer  = cv2.VideoWriter(str(original_path),  fourcc, orig_fps, (w, h))

    idx = 0
    last_result: dict | None = None
    last_raw_bgr: np.ndarray | None = None
    fps_val = orig_fps

    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            if idx % skip_frames == 0 or last_result is None:
                iter_start = time.time()
                last_result = pipeline.process_frame(frame_rgb)
                fps_val = 1.0 / max(time.time() - iter_start, 0.001)

                # Raw composite: model output before postprocessing.
                raw_alpha = last_result["raw_mask"][:, :, np.newaxis]
                last_raw_bgr = cv2.cvtColor(
                    (frame_rgb * raw_alpha).astype(np.uint8), cv2.COLOR_RGB2BGR
                )
            else:
                # Reuse last mask, re-composite onto the current source frame.
                mask = last_result["final_mask"]
                mask3 = mask[..., None]
                final = (
                    (frame_rgb * mask3 + pipeline._bg * (1.0 - mask3))
                    .clip(0, 255)
                    .astype(np.uint8)
                )
                last_result = {
                    "final_mask": mask,
                    "final": final,
                    "raw_mask": last_result.get("raw_mask"),
                }
                # Re-compute raw composite on the current frame with the cached raw mask.
                if last_result["raw_mask"] is not None:
                    raw_alpha = last_result["raw_mask"][:, :, np.newaxis]
                    last_raw_bgr = cv2.cvtColor(
                        (frame_rgb * raw_alpha).astype(np.uint8), cv2.COLOR_RGB2BGR
                    )

            mask_uint8 = (last_result["final_mask"] * 255).astype(np.uint8)
            mask_writer.write(cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2BGR))
            if last_raw_bgr is not None:
                raw_writer.write(last_raw_bgr)
            composite_writer.write(cv2.cvtColor(last_result["final"], cv2.COLOR_RGB2BGR))
            original_writer.write(bgr)

            idx += 1
            if on_progress:
                on_progress(idx, total, fps_val)
    finally:
        cap.release()
        mask_writer.release()
        raw_writer.release()
        composite_writer.release()
        original_writer.release()

    return {
        "mask":      mask_path,
        "raw":       raw_path,
        "composite": composite_path,
        "original":  original_path,
    }
