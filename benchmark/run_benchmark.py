#!/usr/bin/env python3
"""
CLI entry point for the Video Matting benchmark.

Usage:
  python -m benchmark.run_benchmark                   # All models
  python -m benchmark.run_benchmark --models rvm modnet  # Specific models
  python -m benchmark.run_benchmark --list-models      # List models
  python -m benchmark.run_benchmark --videos-dir /path  # Custom directory

Examples:
  # Full benchmark
  cd background-segmentation
  python -m benchmark.run_benchmark

  # Only MediaPipe and RVM
  python -m benchmark.run_benchmark --models mediapipe_portrait rvm
"""

import argparse
import logging
import sys
from pathlib import Path

from .config import (
    GROUND_TRUTH_DIR,
    LOG_FORMAT,
    LOG_LEVEL,
    OUTPUT_DIR,
    TEMP_RESULTS_DIR,
    VIDEOS_DIR,
)
from .models import MODEL_REGISTRY
from .runner import run_benchmark


def setup_logging(level: str = LOG_LEVEL) -> None:
    """Configure logging with timestamped format + colours via StreamHandler."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=LOG_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="🎬 Video Matting model benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available models:
  mediapipe_portrait          MediaPipe Portrait Segmenter
  mediapipe_selfie_multiclass MediaPipe Selfie Multiclass
  mediapipe_landscape         MediaPipe Landscape Segmenter
  rvm                         Robust Video Matting (MobileNetV3)
  mobilenetv3_lraspp          MobileNetV3 + LRASPP Head
  trimap_matting              Trimap-based Matting (GrabCut)
  modnet                      MODNet
  pphumanseg_v2               PP-HumanSeg V2
  efficient_vit               EfficientViT
        """,
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_REGISTRY.keys()),
        default=None,
        help="Models to benchmark (default: all).",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Display the list of available models and exit.",
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=VIDEOS_DIR,
        help=f"Directory with source videos (default: {VIDEOS_DIR}).",
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=GROUND_TRUTH_DIR,
        help=f"Directory with Ground Truth (default: {GROUND_TRUTH_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory for reports (default: {OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--num-videos",
        type=int,
        default=None,
        help="Maximum number of videos to process (default: all).",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Select videos randomly if --num-videos is used.",
    )
    parser.add_argument(
        "--save-masks",
        action="store_true",
        help="Save output masks to output/masks/ (PNG images).",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Compile and save output masks as MP4 video.",
    )
    parser.add_argument(
        "--save-segmented",
        action="store_true",
        help="Apply the mask to the source video and save the subject on a black background.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=LOG_LEVEL,
        help=f"Log level (default: {LOG_LEVEL}).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # ── List models ──
    if args.list_models:
        print("\n📋 Available models:\n")
        for key, cls in MODEL_REGISTRY.items():
            instance = cls()
            print(f"  {key:<30s}  →  {instance.name}")
        print()
        sys.exit(0)

    # ── Model selection ──
    model_keys = args.models or list(MODEL_REGISTRY.keys())
    models = []
    for key in model_keys:
        if key not in MODEL_REGISTRY:
            logger.error("Unknown model: '%s'", key)
            sys.exit(1)
        models.append(MODEL_REGISTRY[key]())

    logger.info("Selected models: %s", [m.name for m in models])

    # ── Checks ──
    if not args.videos_dir.exists():
        logger.error("Videos directory not found: %s", args.videos_dir)
        logger.info("Create the folder and add MP4 videos inside.")
        args.videos_dir.mkdir(parents=True, exist_ok=True)
        sys.exit(1)

    if not args.gt_dir.exists():
        logger.warning(
            "GT directory not found: %s — quality metrics "
            "will not be computed.",
            args.gt_dir,
        )
        args.gt_dir.mkdir(parents=True, exist_ok=True)

    # ── Run the benchmark ──
    results = run_benchmark(
        models=models,
        videos_dir=args.videos_dir,
        gt_dir=args.gt_dir,
        output_dir=args.output_dir,
        temp_dir=TEMP_RESULTS_DIR,
        num_videos=args.num_videos,
        random_selection=args.shuffle,
        save_masks=args.save_masks,
        save_video=args.save_video,
        save_segmented=args.save_segmented,
    )

    if not results:
        logger.warning("No results produced. Check your data.")
        sys.exit(1)

    # Summary
    print("\n" + "=" * 72)
    print("📊 BENCHMARK SUMMARY")
    print("=" * 72)

    header = f"{'Model':<30s} {'Video':<20s} {'IoU':>8s} {'BndF':>8s} {'FWE':>8s} {'p95(ms)':>10s} {'FLOPs':>12s}"
    print(header)
    print("─" * len(header))

    for r in results:
        if r.get("status") != "OK":
            print(f"{'⚠ ' + r['model']:<30s} {r.get('video', 'N/A'):<20s}  {'ERROR':>8s}")
            continue

        iou = f"{r.get('iou_mean', 0):.4f}" if r.get("iou_mean") is not None else "N/A"
        bf = f"{r.get('boundary_f_mean', 0):.4f}" if r.get("boundary_f_mean") is not None else "N/A"
        fwe = f"{r.get('flow_warping_error', 0):.4f}" if r.get("flow_warping_error") is not None else "N/A"
        p95 = f"{r.get('latency_p95_ms', 0):.2f}"
        flops = r.get("flops_per_frame", -1)
        flops_str = f"{flops:.2e}" if flops and flops > 0 else "N/A"

        print(
            f"{r['model']:<30s} {r['video']:<20s} {iou:>8s} {bf:>8s} "
            f"{fwe:>8s} {p95:>10s} {flops_str:>12s}"
        )

    print("=" * 72)
    print(f"📁 Reports: {args.output_dir}/")
    print()


if __name__ == "__main__":
    main()
