"""
Video processing utilities for running the pipeline on video files.
Includes support for frame skipping to speed up processing.
"""
from __future__ import annotations
from collections.abc import Callable
from pathlib import Path
import cv2
import numpy as np
import time

def process_video(
    pipeline,
    video_path: Path,
    output_dir: Path,
    on_progress: Callable[[int, int, float], None] | None = None,
    skip_frames: int = 1,
) -> dict[str, Path]:
    """Run *pipeline* on frames of *video_path* and write results to *output_dir*.

    Args:
        pipeline:    A :class:`MattingPipeline` instance.
        video_path:  Path to the source video.
        output_dir:  Folder that will receive the output videos.
        on_progress: Optional callback(done, total, fps).
        skip_frames: Process 1 frame every N frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    # Adjust output FPS based on skipping
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    # Keep original FPS for smooth output
    output_fps = orig_fps
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_dir.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"avc1")

    mask_path = output_dir / "mask.mp4"
    composite_path = output_dir / "composite.mp4"

    mask_writer = cv2.VideoWriter(str(mask_path), fourcc, output_fps, (w, h))
    composite_writer = cv2.VideoWriter(str(composite_path), fourcc, output_fps, (w, h))

    idx = 0
    last_result = None
    start_time = time.time()
    
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            
            frame_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            
            # Determine if we should run inference or reuse last result
            if idx % skip_frames == 0 or last_result is None:
                iter_start = time.time()
                last_result = pipeline.process_frame(frame_rgb)
                iter_end = time.time()
                fps_val = 1.0 / max(iter_end - iter_start, 0.001)
            else:
                # REUSE MASK but apply to CURRENT frame for fluidity
                # We need to re-composite because the background/original changed
                mask = last_result["final_mask"]
                comp_mask = mask[:, :, np.newaxis]
                final = (frame_rgb * comp_mask).astype(np.uint8)
                
                last_result = {
                    "final_mask": mask,
                    "final": final
                }
                fps_val = orig_fps # Dummy for skipped frames

            # Always write to output to maintain original FPS
            mask_uint8 = (last_result["final_mask"] * 255).astype(np.uint8)
            mask_writer.write(cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2BGR))
            composite_writer.write(cv2.cvtColor(last_result["final"], cv2.COLOR_RGB2BGR))
            
            if on_progress:
                on_progress(idx + 1, total, fps_val)
            
            idx += 1
    finally:
        cap.release()
        mask_writer.release()
        composite_writer.release()

    return {"mask": mask_path, "composite": composite_path}
