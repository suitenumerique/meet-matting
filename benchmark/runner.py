"""
Moteur principal du benchmark de Video Matting.

Orchestre le workflow complet pour chaque couple (vidéo, modèle) :
  1. Inférence  → masques sauvegardés + latences mesurées
  2. Évaluation → métriques calculées vs Ground Truth
  3. Rapport    → résultats agrégés en CSV/JSON

Le calcul des métriques est strictement séparé de la mesure de latence.
"""

import csv
import itertools
import json
import logging
import queue
import shutil
import threading
import time
import warnings
from collections import defaultdict
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import imageio
import numpy as np
from tqdm import tqdm

from .config import (
    GROUND_TRUTH_DIR,
    LATENCY_PERCENTILE,
    OUTPUT_DIR,
    RESULTS_CSV_FILENAME,
    RESULTS_JSON_FILENAME,
    TEMP_RESULTS_DIR,
    VIDEOS_DIR,
    WARMUP_FRAMES,
)
from .metrics import compute_all_metrics
from .models.base import BaseModelWrapper

# Silence Streamlit spam
logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").disabled = True
warnings.filterwarnings("ignore", message="missing ScriptRunContext")

logger = logging.getLogger(__name__)


class BenchmarkStoppedError(Exception):
    """Levée quand l'utilisateur demande l'arrêt du benchmark en cours."""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Utilitaires vidéo
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _get_video_info(video_path: Path) -> tuple[int, float, tuple[int, int]]:
    """Retourne (nombre_frames, fps, (h, w)) d'une vidéo."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Impossible d'ouvrir la vidéo : {video_path}")

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return num_frames, fps, (w, h)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Ground-Truth format helpers (chroma key)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_GREEN_LO = np.array([35, 50, 50], dtype=np.uint8)
_GREEN_HI = np.array([85, 255, 255], dtype=np.uint8)


def _is_chroma_key(frame_bgr: np.ndarray, threshold: float = 0.15) -> bool:
    """Return True if ≥threshold of pixels fall in the HSV green range."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, _GREEN_LO, _GREEN_HI)
    return float(green.sum()) / (255.0 * green.size) >= threshold


def _chroma_key_to_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """Convert a chroma-key frame (green bg) to a float32 [0,1] foreground mask."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, _GREEN_LO, _GREEN_HI)
    return cv2.bitwise_not(green).astype(np.float32) / 255.0


def get_frame_at(video_path: Path, frame_idx: int) -> np.ndarray | None:
    """Return a single BGR frame at index frame_idx, or None on failure."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


class VideoPrefetcher:
    """Lit et pré-traite les frames dans un thread séparé."""

    def __init__(
        self, video_path: Path, queue_size: int = 128, target_size: tuple[int, int] | None = None
    ):
        self.video_path = video_path
        self.queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=queue_size)
        self.target_size = target_size  # (W, H)
        self.stopped = False
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
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

                # OPTIMISATION VITALE : Resize immédiat pour alléger la file RAM
                if self.target_size:
                    frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_NEAREST)

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
    """Générateur de frames pour une vidéo (économise la RAM)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Impossible d'ouvrir la vidéo : {video_path}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        cap.release()


def _read_video_frames(video_path: Path) -> tuple[list[np.ndarray], float]:
    """
    Lit toutes les frames d'une vidéo (obsolète, préféré _iter_video_frames).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Impossible d'ouvrir la vidéo : {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames, fps


def _load_ground_truth_masks(gt_dir: Path, num_frames: int) -> list[np.ndarray]:
    """
    Load GT masks from a directory or video file.

    Supports two formats:
      - Folder of images (PNG/JPG) sorted by name.
      - Video file (each frame = a mask).

    GT format is auto-detected on the first frame:
      - If ≥15 % of pixels are green (chroma-key), the foreground is extracted
        by inverting the green mask (person = 1.0, green bg = 0.0).
      - Otherwise the frame is converted to greyscale and normalised to [0, 1].

    Returns:
        List of float32 masks in [0, 1], of length min(available, num_frames).
    """
    masks = []
    use_chroma: bool | None = None  # determined on first readable frame

    def _frame_to_mask(frame: np.ndarray) -> np.ndarray:
        nonlocal use_chroma
        if frame.ndim == 3 and frame.shape[2] >= 3:
            if use_chroma is None:
                use_chroma = _is_chroma_key(frame)
                logger.info(
                    "GT format auto-detected: %s",
                    "chroma-key green" if use_chroma else "greyscale mask",
                )
            if use_chroma:
                return _chroma_key_to_mask(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            if use_chroma is None:
                use_chroma = False
            gray = frame
        return gray.astype(np.uint8)

    if gt_dir.is_dir():
        image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
        files = sorted(f for f in gt_dir.iterdir() if f.suffix.lower() in image_exts)
        for f in files[:num_frames]:
            img = cv2.imread(str(f))
            if img is not None:
                masks.append(_frame_to_mask(img))
    elif gt_dir.is_file():
        # Mode vidéo : On gère le cas des vidéos à fond vert (Green Screen)
        cap = cv2.VideoCapture(str(gt_dir))
        while len(masks) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Détection automatique : si la saturation moyenne est faible, c'est probablement un masque binaire
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            avg_saturation = np.mean(hsv[:, :, 1])

            if avg_saturation > 10:
                # Détection du VERT (Background) par dominance de canal
                # Un pixel est vert si G > R + tolerance et G > B + tolerance
                b, g, r = cv2.split(frame)
                is_green = (g > (r.astype(np.int16) + 15)) & (g > (b.astype(np.int16) + 15))

                # La personne est l'inverse du vert
                human_mask = (~is_green).astype(np.uint8) * 255
                masks.append(human_mask.astype(np.uint8))
            else:
                # C'est probablement déjà un masque binaire ou niveaux de gris
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                masks.append(gray.astype(np.uint8))
        cap.release()

    logger.info("GT chargé et traité : %d masques extraits depuis %s", len(masks), gt_dir)
    return masks


def _save_masks(masks: list[np.ndarray], output_dir: Path, start_idx: int = 0) -> None:
    """Sauvegarde une liste de masques en PNG dans un répertoire via un pool de threads."""
    output_dir.mkdir(parents=True, exist_ok=True)

    def _save_single(idx, mask):
        if mask.dtype == np.uint8:
            mask_u8 = mask
        else:
            mask_u8 = (mask * 255).astype(np.uint8)
        # IMWRITE_PNG_COMPRESSION=1 : Compression MINIMALE pour une vitesse maximale
        cv2.imwrite(
            str(output_dir / f"mask_{idx:06d}.png"), mask_u8, [cv2.IMWRITE_PNG_COMPRESSION, 1]
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        for i, mask in enumerate(masks):
            executor.submit(_save_single, start_idx + i, mask)


def _batched(iterable, n):
    """Regroupe les éléments d'un itérable par lots de taille n."""
    it = iter(iterable)
    while True:
        batch = list(itertools.islice(it, n))
        if not batch:
            break
        yield batch


def _save_segmented_masks(
    masks: list[np.ndarray], frames: list[np.ndarray], output_dir: Path
) -> None:
    """Sauvegarde les frames avec le sujet détouré sur fond noir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, (mask, frame) in enumerate(zip(masks, frames, strict=True)):
        # Conversion robuste BGR(A)
        if len(frame.shape) == 3 and frame.shape[2] == 4:
            f_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        else:
            f_bgr = frame

        h, w = f_bgr.shape[:2]
        # Redimensionnement du masque si nécessaire
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Normalisation uint8/float
        if mask.dtype == np.uint8:
            mask_norm = mask.astype(np.float32) / 255.0
        else:
            mask_norm = mask.astype(np.float32)

        segmented = (f_bgr * mask_norm.squeeze()[:, :, np.newaxis]).astype(np.uint8)
        cv2.imwrite(str(output_dir / f"segmented_{i:06d}.png"), segmented)


def _save_segmented_video(
    masks: list[np.ndarray], frames: list[np.ndarray], output_path: Path, fps: float
) -> None:
    """Compile une vidéo du sujet détouré sur fond noir de façon ultra-rapide."""
    if not masks or not frames:
        return

    # Encoder haute performance (H.264) via imageio (utilise ffmpeg-static interne)
    # On spécifie 'libx264' et une qualité raisonnable pour la vitesse
    import sys

    # Utiliser l'encodeur matériel sur Mac si possible
    codec = "h264_videotoolbox" if sys.platform == "darwin" else "libx264"

    writer = imageio.get_writer(
        str(output_path),
        fps=fps,
        codec=codec,
        pixelformat="yuv420p",
        ffmpeg_log_level="error",
        macro_block_size=1,  # Évite les warnings sur les dimensions non divisibles par 16
    )

    for mask, frame in zip(masks, frames, strict=True):
        # Conversion BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]

        # Redimensionnement auto
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Masque binaire robuste
        threshold = 127 if mask.dtype == np.uint8 else 0.5
        mask_binary = (mask.squeeze() >= threshold).astype(np.float32)

        # Détourage
        segmented = (frame_rgb * mask_binary[:, :, np.newaxis]).astype(np.uint8)
        writer.append_data(segmented)

    writer.close()


def _save_masks_as_video_fast(masks: list[np.ndarray], output_path: Path, fps: float) -> None:
    """Compile une vidéo des masques de façon ultra-rapide."""
    if not masks:
        return

    import sys

    codec = "h264_videotoolbox" if sys.platform == "darwin" else "libx264"

    writer = imageio.get_writer(
        str(output_path),
        fps=fps,
        codec=codec,
        pixelformat="yuv420p",
        ffmpeg_log_level="error",
        macro_block_size=1,
    )

    for mask in masks:
        # Squeeze pour s'assurer que c'est du (H, W) et non (H, W, 1)
        mask_2d = mask.squeeze()

        if mask_2d.dtype == np.uint8:
            mask_u8 = mask_2d
        else:
            mask_u8 = (mask_2d * 255).astype(np.uint8)

        # Convertir Gris -> RGB pour compatibilité H.264
        mask_rgb = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2RGB)
        writer.append_data(mask_rgb)

    writer.close()
    logger.info("🎬 Vidéo masque sauvegardée : %s", output_path)


def _save_eval_debug_videos(
    preds: list[np.ndarray],
    gts: list[np.ndarray],
    frames: list[np.ndarray],
    output_base: Path,
    fps: float,
) -> None:
    """Génère des vidéos d'intersection et d'union pour débugger les métriques."""
    if not frames or not preds or not gts:
        return

    h, w = frames[0].shape[:2]

    # Chemins de sortie
    parent = output_base.parent
    inter_path = parent / f"{output_base.name}_DEBUG_intersection.mp4"
    union_path = parent / f"{output_base.name}_DEBUG_union.mp4"

    writer_inter = imageio.get_writer(str(inter_path), fps=fps, macro_block_size=1)
    writer_union = imageio.get_writer(str(union_path), fps=fps, macro_block_size=1)

    # Garder la longueur minimale
    n = min(len(preds), len(gts), len(frames))

    for i in range(n):
        p, g, f = preds[i], gts[i], frames[i]

        # Binarisation
        p_bin = (p > 127 if p.dtype == np.uint8 else p > 0.5).astype(np.uint8)

        # Le GT est souvent une frame BGR d'une vidéo
        if g.ndim == 3:
            g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
        g_res = cv2.resize(g, (w, h), interpolation=cv2.INTER_NEAREST)
        g_bin = (g_res > 127 if g_res.dtype == np.uint8 else g_res > 0.5).astype(np.uint8)

        p_res = cv2.resize(p_bin, (w, h), interpolation=cv2.INTER_NEAREST)

        # Intersection : Les pixels communs (Vrais Positifs)
        inter = np.logical_and(p_res, g_bin).astype(np.uint8) * 255
        inter_bgr = cv2.cvtColor(inter, cv2.COLOR_GRAY2BGR)
        inter_vis = cv2.addWeighted(cv2.cvtColor(f, cv2.COLOR_RGB2BGR), 0.3, inter_bgr, 0.7, 0)
        writer_inter.append_data(cv2.cvtColor(inter_vis, cv2.COLOR_BGR2RGB))

        # Union : Visualisation des erreurs
        # Rouge = IA seule, Bleu = GT seul, Blanc = Les deux
        union_vis = cv2.cvtColor(f, cv2.COLOR_RGB2BGR).copy()

        # Overlay IA (Rouge)
        mask_ia = np.zeros_like(union_vis)
        mask_ia[p_res > 0] = [0, 0, 255]

        # Overlay GT (Bleu)
        mask_gt = np.zeros_like(union_vis)
        mask_gt[g_bin > 0] = [255, 0, 0]

        # Somme des overlays
        fusion = cv2.addWeighted(mask_ia, 0.5, mask_gt, 0.5, 0)
        union_vis = cv2.addWeighted(union_vis, 0.3, fusion, 0.7, 0)

        writer_union.append_data(cv2.cvtColor(union_vis, cv2.COLOR_BGR2RGB))

    writer_inter.close()
    writer_union.close()
    logger.info(
        "🧪 Vidéos de debug générées : %s (Intersection) et %s (Union)",
        inter_path.name,
        union_path.name,
    )


def _load_masks(masks_dir: Path) -> list[np.ndarray]:
    """Recharge les masques sauvegardés depuis un répertoire."""
    files = sorted(masks_dir.glob("mask_*.png"))
    masks = []
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            masks.append(img.astype(np.float32) / 255.0)
    return masks


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Étape 1 : Inférence
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_inference(
    model: BaseModelWrapper,
    video_path: Path,
    output_dir: Path | None,
    batch_size: int = 8,
    collect_masks: bool = True,
    stop_event: threading.Event | None = None,
) -> dict:
    """
    Exécute l'inférence sur une vidéo en streaming avec prefetching et batching.
    """
    latencies: list[float] = []
    total_frames = 0
    masks_in_ram: list[np.ndarray] | None = [] if collect_masks else None

    num_frames, fps, (w, h) = _get_video_info(video_path)
    input_shape = (3, h, w)

    model.reset_state()
    logger.info("Inférence %s sur %s (prefetch actif)…", model.name, video_path.name)

    # Préparer le prefetcher avec la taille d'entrée du modèle (si fixe)
    target_size = None
    if model.input_size:
        # Convertir (H, W) -> (W, H) pour OpenCV
        target_size = (model.input_size[1], model.input_size[0])

    # Optimisation spécifique pour Mac : Limitation des threads OpenCV pour éviter les conflits
    import cv2

    cv2.setNumThreads(1)

    # On bypass le prefetcher pour MediaPipe pour éviter la surcharge de threads
    cap = cv2.VideoCapture(str(video_path))
    target_size = None
    if model.input_size:
        target_size = (model.input_size[1], model.input_size[0])

    with tqdm(total=num_frames, desc=f"  ⚡ {model.name}", unit="frame") as pbar:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if stop_event is not None and stop_event.is_set():
                cap.release()
                raise BenchmarkStoppedError(
                    f"Arrêt demandé à la frame {frame_idx} ({model.name} / {video_path.name})"
                )

            t_start = time.perf_counter()
            # On resize directement ici pour être le plus rapide possible
            if target_size:
                frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_NEAREST)

            mask = model.predict(frame)
            t_end = time.perf_counter()

            latency = (t_end - t_start) * 1000.0
            if frame_idx >= WARMUP_FRAMES:
                latencies.append(latency)

            if collect_masks:
                assert masks_in_ram is not None  # garanti par collect_masks=True à l'init
                if mask is None:
                    raise RuntimeError(
                        f"Le modèle {model.name} a renvoyé None à la frame {frame_idx} "
                        f"({video_path.name}). Aucun masque ne peut être produit."
                    )
                mask_u8 = (mask * 255).astype(np.uint8)
                masks_in_ram.append(mask_u8)

            frame_idx += 1
            pbar.update(1)
            total_frames += 1

        cap.release()

    # Calcul des statistiques
    if not latencies:
        raise RuntimeError(
            f"Aucune latence collectée pour {model.name} sur {video_path.name}: "
            f"vidéo trop courte (≤ {WARMUP_FRAMES} frames) ou aucune frame lue."
        )
    latencies_arr = np.array(latencies)
    p95 = float(np.percentile(latencies_arr, LATENCY_PERCENTILE))
    mean_ms = float(latencies_arr.mean())
    # FPS d'inférence pure (inverse de la latence moyenne par frame)
    fps_inference = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    # Garde input_shape référencé pour compat (les modèles peuvent en avoir besoin)
    _ = input_shape

    result = {
        "latencies_ms": latencies,
        "latency_p95_ms": p95,
        "latency_mean_ms": mean_ms,
        "latency_std_ms": float(latencies_arr.std()),
        "fps": fps_inference,
        "num_frames": total_frames,
        "masks": masks_in_ram,
    }

    logger.info(
        "%s — Latence p95: %.2f ms | Moyenne: %.2f ms | FPS inférence: %.1f",
        model.name,
        p95,
        mean_ms,
        fps_inference,
    )

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Étape 2 : Évaluation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_evaluation(
    masks_dir: Path | None,
    gt_masks: list[np.ndarray],
    video_path: Path | None = None,
    masks: list[np.ndarray] | None = None,
    threshold: float = 0.5,
    frames: list[np.ndarray] | None = None,
) -> dict:
    """
    Calcule les métriques vs GT au seuil donné.
    Peut prendre soit un répertoire de masques (disque), soit une liste (RAM).
    Si `frames` est fourni, il est réutilisé directement pour le FWE (évite une relecture disque).
    """
    if masks:
        pred_masks = masks
    else:
        if masks_dir is None:
            raise ValueError("run_evaluation: il faut fournir 'masks' ou 'masks_dir'.")
        pred_masks = _load_masks(masks_dir)

    if not pred_masks:
        raise RuntimeError(
            f"Aucun masque prédit trouvé (masks_dir={masks_dir}). "
            "L'évaluation ne peut pas se faire."
        )

    logger.info("Évaluation : %d masques prédits vs %d GT", len(pred_masks), len(gt_masks))

    frames_iter: list[np.ndarray] | None
    if frames is not None:
        frames_iter = frames
    elif video_path is not None:
        frames_iter = list(_iter_video_frames(video_path))
    else:
        frames_iter = None

    metrics = compute_all_metrics(pred_masks, gt_masks, frames_iter, threshold=threshold)

    logger.info(
        "Résultats — IoU: %.4f ± %.4f | BoundaryF: %.4f ± %.4f | FWE: %.4f",
        metrics["iou_mean"],
        metrics["iou_std"],
        metrics["boundary_f_mean"],
        metrics["boundary_f_std"],
        metrics["flow_warping_error"],
    )

    return metrics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Découverte des vidéos et GT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def discover_datasets(
    videos_dir: Path = VIDEOS_DIR,
    gt_dir: Path = GROUND_TRUTH_DIR,
) -> list[tuple[Path, Path]]:
    """
    Découvre les couples (vidéo, ground_truth).

    Convention de nommage :
      - dataset/videos/video_001.mp4
      - dataset/ground_truth/video_001/  (dossier de masques PNG)
      OU
      - dataset/ground_truth/video_001.mp4  (vidéo de masques)

    Returns:
        Liste de tuples (video_path, gt_path).
    """
    if not videos_dir.exists():
        logger.warning("Dossier vidéos introuvable : %s", videos_dir)
        return []

    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    pairs = []

    for video_file in sorted(videos_dir.iterdir()):
        if video_file.suffix.lower() not in video_exts:
            continue

        stem = video_file.stem

        # Chercher le GT correspondant
        gt_folder = gt_dir / stem
        gt_video = gt_dir / f"{stem}.mp4"
        gt_video_avi = gt_dir / f"{stem}.avi"

        if gt_folder.is_dir():
            pairs.append((video_file, gt_folder))
            logger.info("Dataset découvert : %s ↔ %s/", video_file.name, gt_folder.name)
        elif gt_video.is_file():
            pairs.append((video_file, gt_video))
            logger.info("Dataset découvert : %s ↔ %s", video_file.name, gt_video.name)
        elif gt_video_avi.is_file():
            pairs.append((video_file, gt_video_avi))
            logger.info("Dataset découvert : %s ↔ %s", video_file.name, gt_video_avi.name)
        else:
            # C'est souvent normal pour certaines vidéos d'un dataset de ne pas avoir de GT.
            # On log en DEBUG pour éviter de spammer le terminal lors du discovery UI.
            logger.debug(
                "Pas de GT trouvé pour %s (cherché : %s/, %s, %s)",
                video_file.name,
                gt_folder,
                gt_video,
                gt_video_avi,
            )

    logger.debug("Total : %d couples (vidéo, GT) découverts.", len(pairs))
    return pairs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Boucle principale
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_benchmark(
    models: list[BaseModelWrapper],
    videos_dir: Path = VIDEOS_DIR,
    gt_dir: Path = GROUND_TRUTH_DIR,
    output_dir: Path = OUTPUT_DIR,
    temp_dir: Path = TEMP_RESULTS_DIR,
    num_videos: int | None = None,
    random_selection: bool = False,
    video_indices: list[int] | None = None,
    save_masks: bool = False,
    save_video: bool = False,
    save_segmented: bool = False,
    progress_callback: Callable[[int, int, str], None] | None = None,
    on_result: Callable[[dict], None] | None = None,
    analyze_thresholds: bool = False,
    threshold: float = 0.5,
    stop_event: threading.Event | None = None,
) -> dict[str, Any]:
    """
    Exécute le benchmark complet pour tous les modèles sur toutes les vidéos.

    Args:
        progress_callback: Fonction appelée à chaque étape (current, total, message).
        threshold: Seuil de binarisation utilisé quand analyze_thresholds=False.
        analyze_thresholds: Si True, effectue un balayage de seuils [0.1, 0.9] et
            calcule le meilleur seuil par modèle (argmax IoU moyenné sur les vidéos).

    Returns:
        Dict avec les clés:
            - "results": list[dict] des résultats par (modèle, vidéo)
            - "best_thresholds": dict[model_name, float] (vide si analyze_thresholds=False)
            - "global_pipeline_fps": float — FPS pipeline global du benchmark
            - "wall_clock_total_s": float — durée totale en secondes
            - "history_dir": Path | None — dossier d'historique créé
    """
    import random

    # Préparer les répertoires
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    masks_final_dir = output_dir / "masks"
    if save_masks or save_video or save_segmented:
        masks_final_dir.mkdir(parents=True, exist_ok=True)

    # ── Dossier d'historique numéroté (un par benchmark lancé) ──
    history_root = output_dir / "history"
    history_root.mkdir(parents=True, exist_ok=True)
    existing = sorted(p for p in history_root.glob("benchmark_*") if p.is_dir())
    if existing:
        last_id_str = existing[-1].name.split("_", 1)[1]
        next_id = int(last_id_str) + 1
    else:
        next_id = 1
    run_dir = history_root / f"benchmark_{next_id:04d}"
    run_dir.mkdir()
    logger.info("📁 Historique du benchmark : %s", run_dir)

    # Chemins générés pendant ce run — utilisés par le dashboard pour le nettoyage en cas d'arrêt.
    model_dirs_created: list[Path] = [run_dir]

    # Découvrir les datasets
    datasets = discover_datasets(videos_dir, gt_dir)
    if not datasets:
        raise FileNotFoundError(
            f"Aucun dataset trouvé. Vérifie que des vidéos sont dans {videos_dir} "
            f"et des GT correspondants dans {gt_dir}."
        )

    # Sélection par indices spécifiques (si fournis)
    if video_indices is not None and len(video_indices) > 0:
        datasets = [datasets[i] for i in video_indices if i < len(datasets)]
        logger.info("Traitement de %d vidéos (sélection par indices).", len(datasets))
    else:
        # Sélection du nombre de vidéos (comportement d'origine)
        if random_selection:
            random.shuffle(datasets)

        if num_videos is not None and num_videos > 0:
            datasets = datasets[:num_videos]
            logger.info(
                "Traitement de %d vidéos (sélection %s).",
                len(datasets),
                "aléatoire" if random_selection else "ordonnée",
            )

    all_results: list[dict] = []
    total_combos = len(models) * len(datasets)
    current_combo = 0
    total_frames_global = 0
    wall_clock_start = time.perf_counter()

    logger.info("=" * 72)
    logger.info(
        "BENCHMARK : %d modèle(s) × %d vidéo(s) = %d combinaisons",
        len(models),
        len(datasets),
        total_combos,
    )
    logger.info("=" * 72)

    if progress_callback:
        progress_callback(0, total_combos, "Démarrage du benchmark...")

    _stopped = False
    try:
        for model in models:
            logger.info("─" * 72)
            logger.info("Modèle : %s", model.name)
            logger.info("─" * 72)

            # Charger le modèle
            try:
                model.load()
            except Exception as e:
                logger.error("Échec du chargement de %s : %s", model.name, e)
                for video_path, _ in datasets:
                    all_results.append(
                        {
                            "model": model.name,
                            "video": video_path.name,
                            "status": "LOAD_ERROR",
                            "error": str(e),
                        }
                    )
                continue

            try:
                for video_path, gt_path in datasets:
                    logger.info("\n📹 Vidéo : %s", video_path.name)
                    result_entry: dict[str, Any] = {
                        "model": model.name,
                        "video": video_path.name,
                        "status": "OK",
                    }

                    # Wall-clock englobant inférence + GT + évaluation + export
                    t_combo_start = time.perf_counter()

                    try:
                        # ── Infos vidéo ──
                        num_frames, fps_src, (w_h) = _get_video_info(video_path)
                        result_entry["fps_source"] = fps_src
                        result_entry["resolution"] = f"{w_h[0]}x{w_h[1]}"

                        # ── Charger le GT ──
                        gt_masks = _load_ground_truth_masks(gt_path, num_frames)
                        if not gt_masks:
                            raise FileNotFoundError(
                                f"Pas de GT disponible pour {video_path.name} "
                                f"(cherché dans {gt_path}). Le calcul des métriques est impossible."
                            )

                        # ── Étape 1 : Inférence ──
                        masks_output = (
                            temp_dir / f"{model.name.replace(' ', '_')}_{video_path.stem}"
                        )
                        inference_result = run_inference(
                            model,
                            video_path,
                            masks_output if save_masks else None,
                            collect_masks=True,
                            stop_event=stop_event,
                        )

                        result_entry.update(
                            {
                                "latency_p95_ms": round(inference_result["latency_p95_ms"], 2),
                                "latency_mean_ms": round(inference_result["latency_mean_ms"], 2),
                                "latency_std_ms": round(inference_result["latency_std_ms"], 2),
                                "fps": round(inference_result["fps"], 2),
                                "num_frames": inference_result["num_frames"],
                            }
                        )

                        # ── Charger les frames une seule fois (FWE + save_segmented) ──
                        source_frames, _ = _read_video_frames(video_path)

                        # ── Étape 2 : Évaluation ──
                        eval_result = run_evaluation(
                            masks_output if save_masks else None,
                            gt_masks,
                            video_path=None,
                            masks=inference_result.get("masks"),
                            threshold=threshold,
                            frames=source_frames,
                        )
                        result_entry.update(
                            {
                                "iou_mean": round(eval_result["iou_mean"], 4),
                                "iou_std": round(eval_result["iou_std"], 4),
                                "boundary_f_mean": round(eval_result["boundary_f_mean"], 4),
                                "boundary_f_std": round(eval_result["boundary_f_std"], 4),
                                "flow_warping_error": round(eval_result["flow_warping_error"], 4),
                                "threshold": threshold,
                            }
                        )

                        # ── Threshold sensitivity analysis ──
                        if analyze_thresholds:
                            pred_masks_all = inference_result.get("masks")
                            if not pred_masks_all:
                                raise RuntimeError(
                                    f"analyze_thresholds=True mais aucun masque collecté pour "
                                    f"{model.name} / {video_path.name}."
                                )
                            n_th = min(len(pred_masks_all), len(gt_masks))
                            th_range = [round(t * 0.1, 1) for t in range(1, 10)]
                            th_analysis: dict[str, dict] = {}
                            for t in th_range:
                                th_m = compute_all_metrics(
                                    pred_masks_all[:n_th],
                                    gt_masks[:n_th],
                                    frames=source_frames[:n_th],
                                    threshold=t,
                                )
                                th_analysis[str(t)] = {
                                    "iou_mean": round(th_m["iou_mean"], 4),
                                    "boundary_f_mean": round(th_m["boundary_f_mean"], 4),
                                    "flow_warping_error": round(th_m["flow_warping_error"], 4),
                                }
                            result_entry["threshold_analysis"] = th_analysis

                        # ── Sauvegarde permanente si demandée ──
                        if save_masks or save_video or save_segmented:
                            model_dir = masks_final_dir / model.name.replace(" ", "_")
                            if model_dir not in model_dirs_created:
                                model_dirs_created.append(model_dir)
                            dest_base = model_dir / video_path.stem
                            dest_base.mkdir(parents=True, exist_ok=True)

                            m_temp = inference_result.get("masks")
                            if not m_temp or len(m_temp) == 0:
                                m_temp = _load_masks(masks_output)

                            if save_masks:
                                _save_masks(m_temp, dest_base)
                                logger.info("💾 Masques PNG sauvegardés dans : %s", dest_base)

                            if save_video:
                                video_out_path = (
                                    dest_base.parent
                                    / f"{video_path.stem}_{model.name.replace(' ', '_')}_mask.mp4"
                                )
                                _save_masks_as_video_fast(m_temp, video_out_path, fps_src)
                                logger.info("🎬 Vidéo masque sauvegardée : %s", video_out_path)

                            if save_segmented:
                                seg_video_path = (
                                    dest_base.parent
                                    / f"{video_path.stem}_{model.name.replace(' ', '_')}_segmented.mp4"
                                )
                                _save_segmented_video(
                                    m_temp, source_frames, seg_video_path, fps_src
                                )
                                logger.info("🎨 Vidéo détourée sauvegardée : %s", seg_video_path)

                        # ── Nettoyage des masques temporaires ──
                        if masks_output.exists():
                            shutil.rmtree(masks_output)
                            logger.info("🗑️  Masques temporaires supprimés : %s", masks_output)

                    except BenchmarkStoppedError:
                        raise
                    except Exception as e:
                        logger.exception("Erreur pour %s / %s : %s", model.name, video_path.name, e)
                        result_entry["status"] = "ERROR"
                        result_entry["error"] = str(e)

                    # Wall-clock de la combinaison → FPS pipeline
                    combo_elapsed = time.perf_counter() - t_combo_start
                    n_frames_combo = int(result_entry.get("num_frames") or 0)
                    result_entry["wall_clock_s"] = round(combo_elapsed, 3)
                    result_entry["pipeline_fps"] = (
                        round(n_frames_combo / combo_elapsed, 2) if combo_elapsed > 0 else 0.0
                    )
                    total_frames_global += n_frames_combo

                    all_results.append(result_entry)

                    current_combo += 1
                    if progress_callback:
                        progress_callback(
                            current_combo,
                            total_combos,
                            f"Traité : {model.name} / {video_path.name}",
                        )

                    if on_result:
                        on_result(result_entry)

            except BenchmarkStoppedError:
                raise
            finally:
                model.cleanup()

    except BenchmarkStoppedError:
        _stopped = True
        logger.info("🛑 Benchmark arrêté par l'utilisateur.")

    wall_clock_total = time.perf_counter() - wall_clock_start
    global_pipeline_fps = total_frames_global / wall_clock_total if wall_clock_total > 0 else 0.0

    # En cas d'arrêt : ne pas écrire de rapports partiels, laisser le dashboard nettoyer.
    if _stopped:
        logger.info(
            "🛑 Benchmark interrompu — %d résultat(s) partiel(s), aucun rapport écrit.",
            len(all_results),
        )
        return {
            "results": all_results,
            "best_thresholds": {},
            "global_pipeline_fps": 0.0,
            "wall_clock_total_s": round(wall_clock_total, 2),
            "history_dir": run_dir,
            "stopped": True,
            "model_dirs_created": model_dirs_created,
        }

    # ── Meilleur seuil par modèle (argmax IoU moyenné sur les vidéos) ──
    best_thresholds: dict[str, float] = {}
    if analyze_thresholds:
        best_thresholds = _compute_best_thresholds(all_results)
        # Reporter le meilleur seuil et ses métriques dans chaque ligne du résultat
        for r in all_results:
            if r.get("status") != "OK":
                continue
            ta = r.get("threshold_analysis")
            if not ta:
                continue
            model_name = r["model"]
            if model_name not in best_thresholds:
                raise RuntimeError(
                    f"Pas de seuil optimal calculé pour {model_name}: "
                    f"threshold_analysis manquant ou inconsistant."
                )
            t_best = best_thresholds[model_name]
            metrics_best = ta[str(t_best)]
            r["best_threshold"] = t_best
            r["iou_mean"] = metrics_best["iou_mean"]
            r["boundary_f_mean"] = metrics_best["boundary_f_mean"]
            r["flow_warping_error"] = metrics_best["flow_warping_error"]
            r["threshold"] = t_best

    # ── Générer les rapports (output/ + run_dir/) ──
    _save_csv_report(all_results, output_dir, analyze_thresholds=analyze_thresholds)
    _save_json_report(all_results, output_dir)
    _save_csv_report(all_results, run_dir, analyze_thresholds=analyze_thresholds)
    _save_json_report(all_results, run_dir)

    summary = {
        "id": next_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "models": sorted(m.name for m in models),
        "num_videos": len(datasets),
        "num_results": len(all_results),
        "global_pipeline_fps": round(global_pipeline_fps, 2),
        "wall_clock_total_s": round(wall_clock_total, 2),
        "analyze_thresholds": analyze_thresholds,
        "threshold": None if analyze_thresholds else threshold,
        "best_thresholds": best_thresholds,
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("=" * 72)
    logger.info("✅ BENCHMARK TERMINÉ — %d résultats enregistrés.", len(all_results))
    logger.info("   CSV  : %s", output_dir / RESULTS_CSV_FILENAME)
    logger.info("   JSON : %s", output_dir / RESULTS_JSON_FILENAME)
    logger.info("   History : %s", run_dir)
    logger.info(
        "   FPS pipeline global : %.2f (frames=%d, wall=%.2fs)",
        global_pipeline_fps,
        total_frames_global,
        wall_clock_total,
    )
    logger.info("=" * 72)

    return {
        "results": all_results,
        "best_thresholds": best_thresholds,
        "global_pipeline_fps": global_pipeline_fps,
        "wall_clock_total_s": wall_clock_total,
        "history_dir": run_dir,
        "stopped": False,
        "model_dirs_created": model_dirs_created,
    }


def _compute_best_thresholds(results: list[dict]) -> dict[str, float]:
    """Pour chaque modèle, retourne le seuil qui maximise la moyenne d'IoU
    sur l'ensemble des vidéos analysées.

    Lève une RuntimeError si un modèle a des analyses partielles (certains seuils
    manquants pour certaines vidéos).
    """
    by_model_threshold: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r.get("status") != "OK":
            continue
        ta = r.get("threshold_analysis")
        if not ta:
            continue
        for t_str, metrics in ta.items():
            if "iou_mean" not in metrics:
                raise RuntimeError(
                    f"threshold_analysis pour {r.get('model')} / {r.get('video')} "
                    f"au seuil {t_str} n'a pas d'iou_mean."
                )
            by_model_threshold[r["model"]][t_str].append(float(metrics["iou_mean"]))

    best: dict[str, float] = {}
    for model_name, per_threshold in by_model_threshold.items():
        # Vérifier que tous les seuils ont le même nombre d'échantillons
        counts = {len(v) for v in per_threshold.values()}
        if len(counts) > 1:
            raise RuntimeError(
                f"Analyse de seuils incohérente pour {model_name}: "
                f"nombre d'échantillons varie selon le seuil ({counts})."
            )
        means = {float(t): sum(v) / len(v) for t, v in per_threshold.items()}
        best_t = max(means, key=lambda k: means[k])
        best[model_name] = best_t
        logger.info(
            "🎯 Meilleur seuil pour %s : %.2f (IoU moyen = %.4f)",
            model_name,
            best_t,
            means[best_t],
        )
    return best


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Report generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _save_csv_report(
    results: list[dict],
    output_dir: Path,
    analyze_thresholds: bool = False,
) -> None:
    """Sauvegarde les résultats en CSV — une ligne par (modèle, vidéo).

    Quand analyze_thresholds=True, une colonne `best_threshold` est ajoutée
    et les métriques iou_mean, boundary_f_mean, flow_warping_error
    correspondent à ce meilleur seuil (déjà résolu par le caller).
    """
    if not results:
        return

    csv_path = output_dir / RESULTS_CSV_FILENAME

    base_fields = [
        "model",
        "video",
        "threshold",
        "status",
        "resolution",
        "fps_source",
        "num_frames",
        "latency_p95_ms",
        "latency_mean_ms",
        "latency_std_ms",
        "fps",
        "pipeline_fps",
        "wall_clock_s",
        "iou_mean",
        "boundary_f_mean",
        "boundary_f_std",
        "flow_warping_error",
        "error",
    ]
    if analyze_thresholds:
        fieldnames = ["model", "video", "best_threshold"] + base_fields[2:]
    else:
        fieldnames = base_fields

    rows: list[dict] = []
    for r in results:
        row = {k: v for k, v in r.items() if k != "threshold_analysis"}
        if analyze_thresholds and r.get("status") == "OK" and "best_threshold" not in row:
            raise RuntimeError(
                f"analyze_thresholds=True mais best_threshold manquant pour "
                f"{r.get('model')} / {r.get('video')}. Anomalie dans la pipeline."
            )
        rows.append(row)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    logger.info("Rapport CSV sauvegardé : %s", csv_path)


def _save_json_report(results: list[dict], output_dir: Path) -> None:
    """Sauvegarde les résultats en JSON (plus riche, inclut les métadonnées)."""
    if not results:
        return

    json_path = output_dir / RESULTS_JSON_FILENAME

    # Nettoyer les valeurs non-sérialisables
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

    logger.info("Rapport JSON sauvegardé : %s", json_path)
