"""
Configuration centralisée pour le benchmark.

Définit les chemins, les seuils par défaut et les paramètres globaux.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# Chemins
# ──────────────────────────────────────────────
BENCHMARK_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = BENCHMARK_ROOT / "dataset"
VIDEOS_DIR = DATASET_DIR / "videos"
GROUND_TRUTH_DIR = DATASET_DIR / "ground_truth"
TEMP_RESULTS_DIR = BENCHMARK_ROOT / "temp_results"
OUTPUT_DIR = BENCHMARK_ROOT / "output"

# ──────────────────────────────────────────────
# Paramètres d'inférence
# ──────────────────────────────────────────────
# Résolution cible pour les modèles (H, W). None = résolution native.
DEFAULT_INPUT_SIZE = (256, 256)

# Nombre de frames de warm-up ignorées pour la mesure de latence (passe principale)
WARMUP_FRAMES = 5

# Percentile pour la latence
LATENCY_PERCENTILE = 95

# Passe latence dédiée (batch=1, frames réelles)
LATENCY_WARMUP_FRAMES = 20   # frames de chauffe, non mesurées
LATENCY_N_FRAMES = 50        # frames effectivement chronométrées

# Seuil de binarisation des masques (pour les modèles produisant des alpha mattes)
MASK_THRESHOLD = 0.5

# ──────────────────────────────────────────────
# Paramètres des métriques
# ──────────────────────────────────────────────
# Rayon en pixels pour le Boundary F-measure
BOUNDARY_TOLERANCE_PX = 3

# ──────────────────────────────────────────────
# Rapport
# ──────────────────────────────────────────────
RESULTS_CSV_FILENAME = "benchmark_results.csv"
RESULTS_JSON_FILENAME = "benchmark_results.json"

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
