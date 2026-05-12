#!/usr/bin/env python3
"""
Point d'entrée CLI pour le benchmark de Video Matting.

Usage :
  python -m benchmark.run_benchmark                   # Tous les modèles
  python -m benchmark.run_benchmark --models rvm modnet  # Modèles spécifiques
  python -m benchmark.run_benchmark --list-models      # Lister les modèles
  python -m benchmark.run_benchmark --videos-dir /path  # Répertoire custom

Exemples :
  # Benchmark complet
  cd background-segmentation
  python -m benchmark.run_benchmark

  # Seulement MediaPipe et RVM
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
    """Configure le logging avec format horodaté + couleurs via StreamHandler."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=LOG_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="🎬 Benchmark de modèles de Video Matting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modèles disponibles :
  mediapipe_portrait          MediaPipe Portrait Segmenter
  mediapipe_selfie_multiclass MediaPipe Selfie Multiclass
  mediapipe_landscape         MediaPipe Landscape Segmenter
  rvm                         Robust Video Matting (MobileNetV3)
  mobilenetv3_lraspp          MobileNetV3 + LRASPP Head
  trimap_matting              Trimap-based Matting (GrabCut)
  modnet                      MODNet
  pphumanseg_v2               PP-HumanSeg V2
  segformer                   SegFormer-B0 (ADE20K, transformer-light)
        """,
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_REGISTRY.keys()),
        default=None,
        help="Modèles à benchmarker (défaut : tous).",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Affiche la liste des modèles disponibles et quitte.",
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=VIDEOS_DIR,
        help=f"Répertoire des vidéos sources (défaut : {VIDEOS_DIR}).",
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=GROUND_TRUTH_DIR,
        help=f"Répertoire du Ground Truth (défaut : {GROUND_TRUTH_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Répertoire de sortie des rapports (défaut : {OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--num-videos",
        type=int,
        default=None,
        help="Nombre maximum de vidéos à traiter (défaut : toutes).",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Sélectionne les vidéos aléatoirement si --num-videos est utilisé.",
    )
    parser.add_argument(
        "--save-masks",
        action="store_true",
        help="Sauvegarde les masques de sortie dans le dossier output/masks/ (images PNG).",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Compile et sauvegarde les masques de sortie en vidéo MP4.",
    )
    parser.add_argument(
        "--save-segmented",
        action="store_true",
        help="Applique le masque sur la vidéo source et sauvegarde le sujet sur fond noir.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Seuil de binarisation des masques (défaut : 0.5). Ignoré si --analyze-thresholds.",
    )
    parser.add_argument(
        "--analyze-thresholds",
        action="store_true",
        help="Active le balayage de seuils [0.1, 0.9] et calcule le meilleur seuil par modèle.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=LOG_LEVEL,
        help=f"Niveau de log (défaut : {LOG_LEVEL}).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # ── Lister les modèles ──
    if args.list_models:
        print("\n📋 Modèles disponibles :\n")
        for key, cls in MODEL_REGISTRY.items():
            instance = cls()
            print(f"  {key:<30s}  →  {instance.name}")
        print()
        sys.exit(0)

    # ── Sélection des modèles ──
    model_keys = args.models or list(MODEL_REGISTRY.keys())
    models = []
    for key in model_keys:
        if key not in MODEL_REGISTRY:
            logger.error("Modèle inconnu : '%s'", key)
            sys.exit(1)
        models.append(MODEL_REGISTRY[key]())

    logger.info("Modèles sélectionnés : %s", [m.name for m in models])

    # ── Vérifications ──
    if not args.videos_dir.exists():
        logger.error("Répertoire vidéos introuvable : %s", args.videos_dir)
        logger.info("Crée le dossier et ajoute des vidéos MP4 dedans.")
        args.videos_dir.mkdir(parents=True, exist_ok=True)
        sys.exit(1)

    if not args.gt_dir.exists():
        logger.warning(
            "Répertoire GT introuvable : %s — les métriques de qualité ne seront pas calculées.",
            args.gt_dir,
        )
        args.gt_dir.mkdir(parents=True, exist_ok=True)

    # ── Lancer le benchmark ──
    bench = run_benchmark(
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
        analyze_thresholds=args.analyze_thresholds,
        threshold=args.threshold,
    )
    results = bench["results"]
    global_fps = bench["global_pipeline_fps"]
    best_thresholds = bench["best_thresholds"]

    if not results:
        logger.warning("Aucun résultat produit. Vérifie tes données.")
        sys.exit(1)

    # Résumé final
    print("\n" + "=" * 72)
    print("📊 RÉSUMÉ DU BENCHMARK")
    print("=" * 72)

    header = (
        f"{'Modèle':<30s} {'Vidéo':<20s} "
        f"{'IoU':>8s} {'BndF':>8s} {'FWE':>8s} "
        f"{'p95(ms)':>10s} {'FPS':>8s} {'pipFPS':>8s}"
    )
    print(header)
    print("─" * len(header))

    for r in results:
        if r.get("status") != "OK":
            print(f"{'⚠ ' + r['model']:<30s} {r.get('video', 'N/A'):<20s}  {'ERREUR':>8s}")
            continue

        iou = f"{r.get('iou_mean', 0):.4f}" if r.get("iou_mean") is not None else "N/A"
        bf = f"{r.get('boundary_f_mean', 0):.4f}" if r.get("boundary_f_mean") is not None else "N/A"
        fwe = (
            f"{r.get('flow_warping_error', 0):.4f}"
            if r.get("flow_warping_error") is not None
            else "N/A"
        )
        p95 = f"{r.get('latency_p95_ms', 0):.2f}"
        fps_val = r.get("fps", 0)
        fps_str = f"{fps_val:.1f}" if fps_val else "N/A"
        pipe_val = r.get("pipeline_fps", 0)
        pipe_str = f"{pipe_val:.1f}" if pipe_val else "N/A"

        print(
            f"{r['model']:<30s} {r['video']:<20s} "
            f"{iou:>8s} {bf:>8s} {fwe:>8s} "
            f"{p95:>10s} {fps_str:>8s} {pipe_str:>8s}"
        )

    print("=" * 72)
    print(f"FPS pipeline global : {global_fps:.2f}")
    if best_thresholds:
        print("Meilleurs seuils par modèle (argmax IoU moyen) :")
        for m, t in sorted(best_thresholds.items()):
            print(f"  {m:<30s}  →  {t:.2f}")
    print(f"📁 Rapports : {args.output_dir}/")
    print()


if __name__ == "__main__":
    main()
