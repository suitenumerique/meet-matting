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

# Datasets disponibles
DATASETS = {
    "Synthetic (Fond Faux)": BENCHMARK_ROOT / "dataset",
    "Real (Vidéos Masques)": BENCHMARK_ROOT / "videos_masques",
}

# Par défaut, on utilise le premier dataset
DEFAULT_DATASET_DIR = DATASETS["Synthetic (Fond Faux)"]

# Chemins dynamiques (seront mis à jour par le runner si besoin,
# mais on garde ces variables pour la compatibilité descendante)
VIDEOS_DIR = DEFAULT_DATASET_DIR / "videos"
GROUND_TRUTH_DIR = DEFAULT_DATASET_DIR / "ground_truth"

TEMP_RESULTS_DIR = BENCHMARK_ROOT / "temp_results"
OUTPUT_DIR = BENCHMARK_ROOT / "output"


# ──────────────────────────────────────────────
# Paramètres d'inférence
# ──────────────────────────────────────────────
# Résolution cible pour les modèles (H, W). None = résolution native.
DEFAULT_INPUT_SIZE = (256, 256)

# Nombre de frames de warm-up ignorées pour la mesure de latence
WARMUP_FRAMES = 5

# Percentile pour la latence
LATENCY_PERCENTILE = 95

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
