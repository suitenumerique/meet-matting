"""
Centralized configuration for the benchmark.

Defines paths, default thresholds and global parameters.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BENCHMARK_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = BENCHMARK_ROOT / "dataset"
VIDEOS_DIR = DATASET_DIR / "videos"
GROUND_TRUTH_DIR = DATASET_DIR / "ground_truth"
TEMP_RESULTS_DIR = BENCHMARK_ROOT / "temp_results"
OUTPUT_DIR = BENCHMARK_ROOT / "output"

# ──────────────────────────────────────────────
# Inference parameters
# ──────────────────────────────────────────────
# Target resolution for the models (H, W). None = native resolution.
DEFAULT_INPUT_SIZE = (256, 256)

# Number of warm-up frames ignored for latency measurement (main pass)
WARMUP_FRAMES = 5

# Percentile used for latency
LATENCY_PERCENTILE = 95

# Dedicated latency pass (batch=1, real frames)
LATENCY_WARMUP_FRAMES = 20   # warm-up frames, not measured
LATENCY_N_FRAMES = 50        # frames actually timed

# Binarization threshold for masks (for models producing alpha mattes)
MASK_THRESHOLD = 0.5

# ──────────────────────────────────────────────
# Metric parameters
# ──────────────────────────────────────────────
# Radius in pixels for the Boundary F-measure
BOUNDARY_TOLERANCE_PX = 3

# ──────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────
RESULTS_CSV_FILENAME = "benchmark_results.csv"
RESULTS_JSON_FILENAME = "benchmark_results.json"

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
