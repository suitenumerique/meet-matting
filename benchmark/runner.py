"""
Main engine of the Video Matting benchmark.

Orchestrates the full workflow for each (video, model) pair:
  1. Inference   → saved masks + measured latencies
  2. Evaluation  → metrics computed vs. Ground Truth
  3. Report      → aggregated results in CSV / JSON

Metric computation is strictly separated from latency measurement.
"""

import csv
import json
import logging
import shutil
import time
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from typing import Dict, Iterable, List, Optional, Tuple, Generator
import itertools
from concurrent.futures import ThreadPoolExecutor
import threading
import queue

from .config import (
    GROUND_TRUTH_DIR,
    LATENCY_N_FRAMES,
    LATENCY_PERCENTILE,
    LATENCY_WARMUP_FRAMES,
    OUTPUT_DIR,
    RESULTS_CSV_FILENAME,
    RESULTS_JSON_FILENAME,
    TEMP_RESULTS_DIR,
    VIDEOS_DIR,
)
from .metrics import compute_all_metrics
from .models.base import BaseModelWrapper

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Video utilities
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _get_video_info(video_path: Path) -> Tuple[int, float, Tuple[int, int]]:
    """Return (num_frames, fps, (h, w)) for a video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Unable to open the video: {video_path}")

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return num_frames, fps, (w, h)


class VideoPrefetcher:
    """Reads video frames in a separate thread to saturate inference."""
    def __init__(self, video_path: Path, queue_size: int = 32):
        self.video_path = video_path
        self.queue = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        # Attach the Streamlit context if running inside a dashboard to avoid warnings
        try:
            from streamlit.runtime.scriptrunner import get_script_run_context, add_script_run_context
            ctx = get_script_run_context()
            if ctx:
                add_script_run_context(self.thread)
        except ImportError:
            pass

        self.thread.start()
        return self

    def _run(self):
        cap = cv2.VideoCapture(str(self.video_path))
        while not self.stopped:
            if not self.queue.full():
                ret, frame = cap.read()
                if not ret:
                    self.stopped = True
                    break
                self.queue.put(frame)
            else:
                time.sleep(0.001)
        cap.release()

    def __iter__(self):
        while not self.stopped or not self.queue.empty():
            try:
                frame = self.queue.get(timeout=1.0)
                yield frame
            except queue.Empty:
                continue

    def stop(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)


def _iter_video_frames(video_path: Path) -> Generator[np.ndarray, None, None]:
    """Frame generator for a video (saves RAM)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Unable to open the video: {video_path}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        cap.release()


def _read_video_frames(video_path: Path) -> Tuple[List[np.ndarray], float]:
    """
    Read every frame of a video (deprecated, prefer _iter_video_frames).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Unable to open the video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames, fps


def _load_ground_truth_masks(gt_dir: Path, num_frames: int) -> List[np.ndarray]:
    """
    Load GT masks from a directory.

    Supports two formats:
      - Folder of images (PNG/JPG) sorted by name.
      - Video of masks (each frame = a grayscale mask).

    Returns:
        List of float32 masks in [0, 1], of length min(available, num_frames).
    """
    masks = []

    if gt_dir.is_dir():
        # Image-folder mode
        image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
        files = sorted(
            f for f in gt_dir.iterdir() if f.suffix.lower() in image_exts
        )
        for f in files[:num_frames]:
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                masks.append(img.astype(np.float32) / 255.0)
    elif gt_dir.is_file():
        # Mask-video mode
        cap = cv2.VideoCapture(str(gt_dir))
        while len(masks) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            masks.append(gray.astype(np.float32) / 255.0)
        cap.release()

    logger.info("GT loaded: %d masks from %s", len(masks), gt_dir)
    return masks


def _save_masks(masks: List[np.ndarray], output_dir: Path, start_idx: int = 0) -> None:
    """Save a list of masks as PNGs into a directory using a thread pool."""
    output_dir.mkdir(parents=True, exist_ok=True)

    def _save_single(idx, mask):
        mask_u8 = (mask * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(str(output_dir / f"mask_{idx:06d}.png"), mask_u8)

    with ThreadPoolExecutor(max_workers=4) as executor:
        for i, mask in enumerate(masks):
            executor.submit(_save_single, start_idx + i, mask)


def _batched(iterable, n):
    """Group the elements of an iterable into batches of size n."""
    it = iter(iterable)
    while True:
        batch = list(itertools.islice(it, n))
        if not batch:
            break
        yield batch


import imageio

def _save_segmented_masks(masks: List[np.ndarray], frames: List[np.ndarray], output_dir: Path) -> None:
    """Save frames with the subject cut out on a black background."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, (mask, frame) in enumerate(zip(masks, frames)):
        # Robust BGR(A) conversion
        if len(frame.shape) == 3 and frame.shape[2] == 4:
            f_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        else:
            f_bgr = frame

        segmented = (f_bgr * mask.squeeze()[:, :, np.newaxis]).astype(np.uint8)
        cv2.imwrite(str(output_dir / f"segmented_{i:06d}.png"), segmented)


def _save_segmented_video(
    masks: List[np.ndarray],
    frames: List[np.ndarray],
    output_path: Path,
    fps: float
) -> None:
    """Compile a video of the cut-out subject on a black background as fast as possible."""
    if not masks or not frames:
        return

    # High-performance encoder (H.264) via imageio (uses bundled ffmpeg-static)
    # We specify 'libx264' with a reasonable speed/quality trade-off
    import sys
    # Use the hardware encoder on Mac when possible
    codec = 'h264_videotoolbox' if sys.platform == 'darwin' else 'libx264'

    writer = imageio.get_writer(
        str(output_path),
        fps=fps,
        codec=codec,
        # quality=7 is replaced by bitrate for videotoolbox if needed,
        # but imageio tries to map quality to the ffmpeg params.
        pixelformat='yuv420p',
        ffmpeg_log_level='error'
    )

    for mask, frame in zip(masks, frames):
        # BGR -> RGB conversion
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Binary mask to avoid darkening the person
        mask_binary = (mask.squeeze() > 0.5).astype(np.float32)

        # Cut-out
        segmented = (frame_rgb * mask_binary[:, :, np.newaxis]).astype(np.uint8)
        writer.append_data(segmented)

    writer.close()


def _save_masks_as_video_fast(masks: List[np.ndarray], output_path: Path, fps: float) -> None:
    """Compile a video of masks as fast as possible."""
    if not masks:
        return

    import sys
    codec = 'h264_videotoolbox' if sys.platform == 'darwin' else 'libx264'

    writer = imageio.get_writer(
        str(output_path),
        fps=fps,
        codec=codec,
        pixelformat='yuv420p',
        ffmpeg_log_level='error'
    )

    for mask in masks:
        # Squeeze to make sure it is (H, W) and not (H, W, 1)
        mask_2d = mask.squeeze()
        mask_u8 = (mask_2d * 255).astype(np.uint8)
        # Convert grayscale -> RGB for H.264 compatibility
        mask_rgb = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2RGB)
        writer.append_data(mask_rgb)

    writer.close()


def _load_masks(masks_dir: Path) -> List[np.ndarray]:
    """Reload saved masks from a directory."""
    files = sorted(masks_dir.glob("mask_*.png"))
    masks = []
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            masks.append(img.astype(np.float32) / 255.0)
    return masks


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Step 1: Inference
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_inference(
    model: BaseModelWrapper,
    video_path: Path,
    output_dir: Path,
    batch_size: int = 8,
    collect_masks: bool = True,
) -> Dict:
    """
    Run inference on a video with streaming prefetching and batching.
    Collects masks for evaluation; latency is measured separately by
    measure_latency() under batch=1 conditions.
    """
    total_frames = 0
    masks_in_ram = [] if collect_masks else None

    num_frames, fps, (w, h) = _get_video_info(video_path)
    input_shape = (3, h, w)

    model.reset_state()
    logger.info("Inference %s on %s (prefetch active)…", model.name, video_path.name)

    prefetcher = VideoPrefetcher(video_path, queue_size=batch_size * 2).start()

    with tqdm(total=num_frames, desc=f"  ⚡ {model.name}", unit="frame") as pbar:
        frame_idx = 0
        for batch in _batched(prefetcher, batch_size):
            masks = model.predict_batch(batch)

            for m in masks:
                if collect_masks:
                    masks_in_ram.append(m)
                frame_idx += 1

            if output_dir:
                _save_masks(masks, output_dir, start_idx=frame_idx - len(batch))

            pbar.update(len(batch))
            total_frames += len(batch)

    prefetcher.stop()

    flops = model.get_flops(input_shape)
    logger.info("%s — %d frames processed | FLOPs: %.2e", model.name, total_frames, flops)

    return {
        "flops_per_frame": flops,
        "num_frames": total_frames,
        "masks": masks_in_ram,
    }


def measure_latency(
    model: BaseModelWrapper,
    video_path: Path,
    warmup_frames: int = LATENCY_WARMUP_FRAMES,
    n_frames: int = LATENCY_N_FRAMES,
) -> Dict:
    """
    Measure frame-by-frame latency under real conditions (batch=1).

    Loads the first (warmup_frames + n_frames) frames of the video into
    RAM, resets the model state, runs the warmup without recording, then
    times each predict_batch([frame]) call individually.
    The model stays in steady state throughout (recurrent state preserved),
    which reflects the steady-state latency of a streaming pipeline.

    Args:
        warmup_frames: Number of warm-up frames (not measured).
        n_frames:      Number of frames actually timed.

    Returns:
        Dict with latency_p95_ms, latency_mean_ms, latency_std_ms.
    """
    total_needed = warmup_frames + n_frames
    frames: List[np.ndarray] = []
    cap = cv2.VideoCapture(str(video_path))
    while len(frames) < total_needed:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if len(frames) <= warmup_frames:
        logger.warning(
            "measure_latency: video too short (%d frames) for %d warmup + %d measurements.",
            len(frames), warmup_frames, n_frames,
        )
        return {"latency_p95_ms": 0.0, "latency_mean_ms": 0.0, "latency_std_ms": 0.0}

    model.reset_state()
    logger.info(
        "Latency %s: %d warmup + %d measurements (batch=1)…",
        model.name, warmup_frames, len(frames) - warmup_frames,
    )

    for frame in frames[:warmup_frames]:
        model.predict_batch([frame])

    latencies: List[float] = []
    for frame in frames[warmup_frames:]:
        t0 = time.perf_counter()
        model.predict_batch([frame])
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    arr = np.array(latencies)
    p95 = float(np.percentile(arr, LATENCY_PERCENTILE))
    logger.info(
        "%s — p95 latency: %.2f ms | Mean: %.2f ms",
        model.name, p95, arr.mean(),
    )
    return {
        "latency_p95_ms": p95,
        "latency_mean_ms": float(arr.mean()),
        "latency_std_ms": float(arr.std()),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Step 2: Evaluation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_evaluation(
    masks_dir: Optional[Path],
    gt_masks: List[np.ndarray],
    video_path: Optional[Path] = None,
    masks: Optional[List[np.ndarray]] = None,
) -> Dict:
    """
    Compute the metrics vs. GT.
    Accepts either a directory of masks (disk) or a list (RAM).
    """
    pred_masks = masks if masks else _load_masks(masks_dir)

    if not pred_masks:
        logger.error("No predicted mask found in %s", masks_dir)
        return {
            "iou_mean": 0.0,
            "iou_std": 0.0,
            "boundary_f_mean": 0.0,
            "boundary_f_std": 0.0,
            "flow_warping_error": 0.0,
        }

    logger.info("Evaluation: %d predicted masks vs. %d GT", len(pred_masks), len(gt_masks))

    frames_iter = _iter_video_frames(video_path) if video_path else None

    metrics = compute_all_metrics(pred_masks, gt_masks, frames_iter)

    logger.info(
        "Results — IoU: %.4f ± %.4f | BoundaryF: %.4f ± %.4f | FWE: %.4f",
        metrics["iou_mean"],
        metrics["iou_std"],
        metrics["boundary_f_mean"],
        metrics["boundary_f_std"],
        metrics["flow_warping_error"],
    )

    return metrics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Video / GT discovery
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def discover_datasets(
    videos_dir: Path = VIDEOS_DIR,
    gt_dir: Path = GROUND_TRUTH_DIR,
) -> List[Tuple[Path, Path]]:
    """
    Discover (video, ground_truth) pairs.

    Naming convention:
      - dataset/videos/video_001.mp4
      - dataset/ground_truth/video_001/  (folder of PNG masks)
      OR
      - dataset/ground_truth/video_001.mp4  (video of masks)

    Returns:
        List of (video_path, gt_path) tuples.
    """
    if not videos_dir.exists():
        logger.warning("Videos folder not found: %s", videos_dir)
        return []

    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    pairs = []

    for video_file in sorted(videos_dir.iterdir()):
        if video_file.suffix.lower() not in video_exts:
            continue

        stem = video_file.stem

        # Look for the matching GT
        gt_folder = gt_dir / stem
        gt_video = gt_dir / f"{stem}.mp4"
        gt_video_avi = gt_dir / f"{stem}.avi"

        if gt_folder.is_dir():
            pairs.append((video_file, gt_folder))
            logger.info("Dataset discovered: %s ↔ %s/", video_file.name, gt_folder.name)
        elif gt_video.is_file():
            pairs.append((video_file, gt_video))
            logger.info("Dataset discovered: %s ↔ %s", video_file.name, gt_video.name)
        elif gt_video_avi.is_file():
            pairs.append((video_file, gt_video_avi))
            logger.info("Dataset discovered: %s ↔ %s", video_file.name, gt_video_avi.name)
        else:
            logger.warning(
                "No GT found for %s (looked for: %s/, %s, %s)",
                video_file.name, gt_folder, gt_video, gt_video_avi,
            )

    logger.info("Total: %d (video, GT) pairs discovered.", len(pairs))
    return pairs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_benchmark(
    models: List[BaseModelWrapper],
    videos_dir: Path = VIDEOS_DIR,
    gt_dir: Path = GROUND_TRUTH_DIR,
    output_dir: Path = OUTPUT_DIR,
    temp_dir: Path = TEMP_RESULTS_DIR,
    num_videos: Optional[int] = None,
    random_selection: bool = False,
    save_masks: bool = False,
    save_video: bool = False,
    save_segmented: bool = False,
) -> List[Dict]:
    """
    Run the full benchmark for every model over every video.

    Args:
        random_selection: If True, pick num_videos videos at random.
        save_masks: If True, store the binary masks (PNG).
        save_video: If True, compile the binary masks into a video (MP4).
        save_segmented: If True, store the subject on a black background
                        (images and video).
    """
    import random

    # Prepare directories
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    masks_final_dir = output_dir / "masks"
    if save_masks:
        masks_final_dir.mkdir(parents=True, exist_ok=True)

    # Discover datasets
    datasets = discover_datasets(videos_dir, gt_dir)
    if not datasets:
        logger.error(
            "No dataset found. Make sure there are videos in %s "
            "and matching GT in %s.",
            videos_dir, gt_dir,
        )
        return []

    # Select the number of videos
    if random_selection:
        random.shuffle(datasets)

    if num_videos is not None:
        datasets = datasets[:num_videos]
        logger.info("Processing %d videos (%s selection).",
                    len(datasets), "random" if random_selection else "ordered")

    all_results = []
    total_combos = len(models) * len(datasets)

    logger.info("=" * 72)
    logger.info(
        "BENCHMARK: %d model(s) × %d video(s) = %d combinations",
        len(models), len(datasets), total_combos,
    )
    logger.info("=" * 72)

    for model in models:
        logger.info("─" * 72)
        logger.info("Model: %s", model.name)
        logger.info("─" * 72)

        # Load the model
        try:
            model.load()
        except Exception as e:
            logger.error("Failed to load %s: %s", model.name, e)
            # Record an error result for each video
            for video_path, _ in datasets:
                all_results.append({
                    "model": model.name,
                    "video": video_path.name,
                    "status": "LOAD_ERROR",
                    "error": str(e),
                })
            continue

        for video_path, gt_path in datasets:
            logger.info("\n📹 Video: %s", video_path.name)
            result_entry = {
                "model": model.name,
                "video": video_path.name,
                "status": "OK",
            }

            try:
                # ── Video info ──
                num_frames, fps, (w_h) = _get_video_info(video_path)
                result_entry["fps_source"] = fps
                result_entry["resolution"] = f"{w_h[0]}x{w_h[1]}"

                # ── Load the GT ──
                gt_masks = _load_ground_truth_masks(gt_path, num_frames)
                if not gt_masks:
                    logger.warning("No GT available for %s.", video_path.name)

                # ── Step 1: Inference (streaming + prefetch) ──
                # Avoid writing to disk if we only care about the metrics
                masks_output = temp_dir / f"{model.name.replace(' ', '_')}_{video_path.stem}"

                # If save_masks is False we don't pass masks_output to run_inference,
                # but we still ask it to collect masks in RAM for evaluation.
                inference_result = run_inference(
                    model,
                    video_path,
                    masks_output if save_masks else None,
                    collect_masks=True,
                )

                latency_result = measure_latency(model, video_path)

                result_entry.update({
                    "latency_p95_ms": round(latency_result["latency_p95_ms"], 2),
                    "latency_mean_ms": round(latency_result["latency_mean_ms"], 2),
                    "latency_std_ms": round(latency_result["latency_std_ms"], 2),
                    "flops_per_frame": inference_result["flops_per_frame"],
                    "num_frames": inference_result["num_frames"],
                })

                # ── Step 2: Evaluation (RAM or disk) ──
                if gt_masks:
                    eval_result = run_evaluation(
                        masks_output if save_masks else None,
                        gt_masks,
                        video_path,
                        masks=inference_result.get("masks")
                    )
                    result_entry.update({
                        "iou_mean": round(eval_result["iou_mean"], 4),
                        "iou_std": round(eval_result["iou_std"], 4),
                        "boundary_f_mean": round(eval_result["boundary_f_mean"], 4),
                        "boundary_f_std": round(eval_result["boundary_f_std"], 4),
                        "flow_warping_error": round(eval_result["flow_warping_error"], 4),
                    })
                else:
                    result_entry.update({
                        "iou_mean": None,
                        "iou_std": None,
                        "boundary_f_mean": None,
                        "boundary_f_std": None,
                        "flow_warping_error": None,
                    })

                # ── Permanent saving if requested ──
                if save_masks or save_video or save_segmented:
                    dest_base = masks_final_dir / model.name.replace(" ", "_") / video_path.stem
                    dest_base.mkdir(parents=True, exist_ok=True)

                    # Use the masks in RAM if disk writes were bypassed
                    m_temp = inference_result.get("masks")
                    if not m_temp or len(m_temp) == 0:
                        m_temp = _load_masks(masks_output)

                    if save_masks:
                        _save_masks(m_temp, dest_base)
                        logger.info("💾 PNG masks saved to: %s", dest_base)

                    if save_video:
                        video_out_path = dest_base.parent / f"{video_path.stem}_{model.name.replace(' ', '_')}_mask.mp4"
                        _save_masks_as_video_fast(m_temp, video_out_path, fps)
                        logger.info("🎬 Mask video saved: %s", video_out_path)

                    if save_segmented:
                        # Re-read the frames for the cut-out
                        frames_list, _ = _read_video_frames(video_path)
                        seg_video_path = dest_base.parent / f"{video_path.stem}_{model.name.replace(' ', '_')}_segmented.mp4"
                        _save_segmented_video(m_temp, frames_list, seg_video_path, fps)
                        logger.info("🎨 Cut-out video saved: %s", seg_video_path)

                # ── Clean up temporary masks ──
                if masks_output.exists():
                    shutil.rmtree(masks_output)
                    logger.info("🗑️  Temporary masks removed: %s", masks_output)

            except Exception as e:
                logger.error("Error for %s / %s: %s", model.name, video_path.name, e)
                result_entry["status"] = "ERROR"
                result_entry["error"] = str(e)

            all_results.append(result_entry)

        # Release the model
        model.cleanup()

    # ── Generate the reports ──
    _save_csv_report(all_results, output_dir)
    _save_json_report(all_results, output_dir)

    logger.info("=" * 72)
    logger.info("✅ BENCHMARK COMPLETE — %d results recorded.", len(all_results))
    logger.info("   CSV  : %s", output_dir / RESULTS_CSV_FILENAME)
    logger.info("   JSON : %s", output_dir / RESULTS_JSON_FILENAME)
    logger.info("=" * 72)

    return all_results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Report generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _save_csv_report(results: List[Dict], output_dir: Path) -> None:
    """Save the results as CSV."""
    if not results:
        return

    csv_path = output_dir / RESULTS_CSV_FILENAME
    fieldnames = [
        "model",
        "video",
        "status",
        "resolution",
        "fps_source",
        "num_frames",
        "latency_p95_ms",
        "latency_mean_ms",
        "latency_std_ms",
        "flops_per_frame",
        "iou_mean",
        "iou_std",
        "boundary_f_mean",
        "boundary_f_std",
        "flow_warping_error",
        "error",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    logger.info("CSV report saved: %s", csv_path)


def _save_json_report(results: List[Dict], output_dir: Path) -> None:
    """Save the results as JSON (richer, includes metadata)."""
    if not results:
        return

    json_path = output_dir / RESULTS_JSON_FILENAME

    # Clean up non-serialisable values
    clean_results = []
    for r in results:
        clean = {}
        for k, v in r.items():
            if isinstance(v, (np.floating, np.integer)):
                clean[k] = float(v)
            elif isinstance(v, np.ndarray):
                clean[k] = v.tolist()
            else:
                clean[k] = v
        clean_results.append(clean)

    report = {
        "benchmark_version": "1.0.0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "num_models": len(set(r.get("model", "") for r in results)),
        "num_videos": len(set(r.get("video", "") for r in results)),
        "results": clean_results,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("JSON report saved: %s", json_path)
