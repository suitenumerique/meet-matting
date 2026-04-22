"""
Moteur principal du benchmark de Video Matting.

Orchestre le workflow complet pour chaque couple (vidéo, modèle) :
  1. Inférence  → masques sauvegardés + latences mesurées
  2. Évaluation → métriques calculées vs Ground Truth
  3. Rapport    → résultats agrégés en CSV/JSON

Le calcul des métriques est strictement séparé de la mesure de latence.
"""

import csv
import json
import logging
import shutil
import time
import warnings
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from typing import Dict, Iterable, List, Optional, Tuple, Generator
import itertools
from concurrent.futures import ThreadPoolExecutor
import threading
import queue

# Silence Streamlit spam
logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").disabled = True
warnings.filterwarnings("ignore", message="missing ScriptRunContext")

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

logger = logging.getLogger(__name__)



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Utilitaires vidéo
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _get_video_info(video_path: Path) -> Tuple[int, float, Tuple[int, int]]:
    """Retourne (nombre_frames, fps, (h, w)) d'une vidéo."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Impossible d'ouvrir la vidéo : {video_path}")
    
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return num_frames, fps, (w, h)


class VideoPrefetcher:
    """Lit les frames d'une vidéo dans un thread séparé pour saturer l'inférence."""
    def __init__(self, video_path: Path, queue_size: int = 32):
        self.video_path = video_path
        self.queue = queue.Queue(maxsize=queue_size)
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
        raise IOError(f"Impossible d'ouvrir la vidéo : {video_path}")
    
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
    Lit toutes les frames d'une vidéo (obsolète, préféré _iter_video_frames).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Impossible d'ouvrir la vidéo : {video_path}")

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
    Charge les masques GT depuis un répertoire ou une vidéo.
    
    Si le GT est une vidéo couleur, on applique un traitement de type Chromakey 
    pour extraire l'humain (tout ce qui n'est ni vert ni noir).
    """
    masks = []

    if gt_dir.is_dir():
        # Mode dossier d'images
        image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
        files = sorted(
            f for f in gt_dir.iterdir() if f.suffix.lower() in image_exts
        )
        for f in files[:num_frames]:
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                masks.append(img.astype(np.float32) / 255.0)
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
            
            if avg_saturation > 30: # Seuil arbitraire pour détecter un fond vert
                # 1. Détection du VERT (Background)
                lower_green = np.array([35, 40, 40])
                upper_green = np.array([85, 255, 255])
                mask_green = cv2.inRange(hsv, lower_green, upper_green)
                
                # 2. Détection du NOIR (Logos/Bords sombres en background)
                lower_black = np.array([0, 0, 0])
                upper_black = np.array([180, 255, 50]) # V < 50
                mask_black = cv2.inRange(hsv, lower_black, upper_black)
                
                # 3. Le fond est l'union du vert et du noir
                background_mask = cv2.bitwise_or(mask_green, mask_black)
                
                # 4. Le sujet humain est l'inverse du fond
                human_mask = cv2.bitwise_not(background_mask)
                
                # Nettoyage morphologique
                kernel = np.ones((5, 5), np.uint8)
                human_mask = cv2.morphologyEx(human_mask, cv2.MORPH_OPEN, kernel)
                human_mask = cv2.morphologyEx(human_mask, cv2.MORPH_CLOSE, kernel)
                masks.append(human_mask.astype(np.float32) / 255.0)
            else:
                # C'est probablement déjà un masque binaire ou niveaux de gris
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                masks.append(gray.astype(np.float32) / 255.0)
        cap.release()


    logger.info("GT chargé et traité : %d masques extraits depuis %s", len(masks), gt_dir)
    return masks



def _save_masks(masks: List[np.ndarray], output_dir: Path, start_idx: int = 0) -> None:
    """Sauvegarde une liste de masques en PNG dans un répertoire via un pool de threads."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    def _save_single(idx, mask):
        mask_u8 = (mask * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(str(output_dir / f"mask_{idx:06d}.png"), mask_u8)

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


import imageio

def _save_segmented_masks(masks: List[np.ndarray], frames: List[np.ndarray], output_dir: Path) -> None:
    """Sauvegarde les frames avec le sujet détouré sur fond noir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, (mask, frame) in enumerate(zip(masks, frames)):
        # Conversion robuste BGR(A)
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
    """Compile une vidéo du sujet détouré sur fond noir de façon ultra-rapide."""
    if not masks or not frames:
        return
    
    # Encoder haute performance (H.264) via imageio (utilise ffmpeg-static interne)
    # On spécifie 'libx264' et une qualité raisonnable pour la vitesse
    import sys
    # Utiliser l'encodeur matériel sur Mac si possible
    codec = 'h264_videotoolbox' if sys.platform == 'darwin' else 'libx264'
    
    writer = imageio.get_writer(
        str(output_path), 
        fps=fps, 
        codec=codec, 
        # quality=7 est remplacé par bitrate pour videotoolbox si nécessaire, 
        # mais imageio tente de mapper quality vers les params ffmpeg.
        pixelformat='yuv420p',
        ffmpeg_log_level='error'
    )
    
    for mask, frame in zip(masks, frames):
        # Conversion BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Masque binaire pour éviter d'assombrir l'humain
        mask_binary = (mask.squeeze() > 0.5).astype(np.float32)
        
        # Détourage
        segmented = (frame_rgb * mask_binary[:, :, np.newaxis]).astype(np.uint8)
        writer.append_data(segmented)
    
    writer.close()


def _save_masks_as_video_fast(masks: List[np.ndarray], output_path: Path, fps: float) -> None:
    """Compile une vidéo des masques de façon ultra-rapide."""
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
        # Squeeze pour s'assurer que c'est du (H, W) et non (H, W, 1)
        mask_2d = mask.squeeze()
        mask_u8 = (mask_2d * 255).astype(np.uint8)
        # Convertir Gris -> RGB pour compatibilité H.264
        mask_rgb = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2RGB)
        writer.append_data(mask_rgb)
    
    writer.close()


def _load_masks(masks_dir: Path) -> List[np.ndarray]:
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
    output_dir: Path,
    batch_size: int = 8,
    collect_masks: bool = True,
) -> Dict:
    """
    Exécute l'inférence sur une vidéo en streaming avec prefetching et batching.
    """
    latencies = []
    total_frames = 0
    masks_in_ram = [] if collect_masks else None
    
    num_frames, fps, (w, h) = _get_video_info(video_path)
    input_shape = (3, h, w)

    model.reset_state()
    logger.info("Inférence %s sur %s (prefetch actif)…", model.name, video_path.name)

    # Préparer le prefetcher
    prefetcher = VideoPrefetcher(video_path, queue_size=batch_size * 2).start()

    with tqdm(total=num_frames, desc=f"  ⚡ {model.name}", unit="frame") as pbar:
        frame_idx = 0
        for batch in _batched(prefetcher, batch_size):
            t_start = time.perf_counter()
            masks = model.predict_batch(batch)
            t_end = time.perf_counter()
            
            latency_per_frame = ((t_end - t_start) * 1000.0) / len(batch)
            
            for m in masks:
                if frame_idx >= WARMUP_FRAMES:
                    latencies.append(latency_per_frame)
                if collect_masks:
                    masks_in_ram.append(m)
                frame_idx += 1

            # Sauvegarde disque uniquement si nécessaire (IO bypass)
            if output_dir:
                _save_masks(masks, output_dir, start_idx=frame_idx - len(batch))
            
            pbar.update(len(batch))
            total_frames += len(batch)

    prefetcher.stop()

    # Calcul des stats
    latencies_arr = np.array(latencies) if latencies else np.array([0.0])
    p95 = float(np.percentile(latencies_arr, LATENCY_PERCENTILE))
    flops = model.get_flops(input_shape)

    result = {
        "latencies_ms": latencies,
        "latency_p95_ms": p95,
        "latency_mean_ms": float(latencies_arr.mean()),
        "latency_std_ms": float(latencies_arr.std()),
        "flops_per_frame": flops,
        "num_frames": total_frames,
        "masks": masks_in_ram,
    }

    logger.info(
        "%s — Latence p95: %.2f ms | Moyenne: %.2f ms | FLOPs: %.2e",
        model.name, p95, result["latency_mean_ms"], flops,
    )

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Étape 2 : Évaluation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_evaluation(
    masks_dir: Optional[Path],
    gt_masks: List[np.ndarray],
    video_path: Optional[Path] = None,
    masks: Optional[List[np.ndarray]] = None,
) -> Dict:
    """
    Calcule les métriques vs GT.
    Peut prendre soit un répertoire de masques (disque), soit une liste (RAM).
    """
    pred_masks = masks if masks else _load_masks(masks_dir)

    if not pred_masks:
        logger.error("Aucun masque prédit trouvé dans %s", masks_dir)
        return {
            "iou_mean": 0.0,
            "iou_std": 0.0,
            "boundary_f_mean": 0.0,
            "boundary_f_std": 0.0,
            "flow_warping_error": 0.0,
        }

    logger.info("Évaluation : %d masques prédits vs %d GT", len(pred_masks), len(gt_masks))

    frames_iter = _iter_video_frames(video_path) if video_path else None
    
    metrics = compute_all_metrics(pred_masks, gt_masks, frames_iter)

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
) -> List[Tuple[Path, Path]]:
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
                video_file.name, gt_folder, gt_video, gt_video_avi,
            )

    logger.debug("Total : %d couples (vidéo, GT) découverts.", len(pairs))
    return pairs



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Boucle principale
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
from typing import Dict, Iterable, List, Optional, Tuple, Generator, Callable

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
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> List[Dict]:
    """
    Exécute le benchmark complet pour tous les modèles sur toutes les vidéos.

    Args:
        progress_callback: Fonction appelée à chaque étape (current, total, message).
    """
    import random

    # Préparer les répertoires
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    masks_final_dir = output_dir / "masks"
    if save_masks:
        masks_final_dir.mkdir(parents=True, exist_ok=True)

    # Découvrir les datasets
    datasets = discover_datasets(videos_dir, gt_dir)
    if not datasets:
        logger.error(
            "Aucun dataset trouvé. Vérifie que des vidéos sont dans %s "
            "et des GT correspondants dans %s.",
            videos_dir, gt_dir,
        )
        return []

    # Sélection du nombre de vidéos
    if random_selection:
        random.shuffle(datasets)
    
    if num_videos is not None:
        datasets = datasets[:num_videos]
        logger.info("Traitement de %d vidéos (sélection %s).", 
                    len(datasets), "aléatoire" if random_selection else "ordonnée")

    all_results = []
    total_combos = len(models) * len(datasets)
    current_combo = 0

    logger.info("=" * 72)
    logger.info(
        "BENCHMARK : %d modèle(s) × %d vidéo(s) = %d combinaisons",
        len(models), len(datasets), total_combos,
    )
    logger.info("=" * 72)

    if progress_callback:
        progress_callback(0, total_combos, "Démarrage du benchmark...")

    for model in models:

        logger.info("─" * 72)
        logger.info("Modèle : %s", model.name)
        logger.info("─" * 72)

        # Charger le modèle
        try:
            model.load()
        except Exception as e:
            logger.error("Échec du chargement de %s : %s", model.name, e)
            # Enregistrer un résultat d'erreur pour chaque vidéo
            for video_path, _ in datasets:
                all_results.append({
                    "model": model.name,
                    "video": video_path.name,
                    "status": "LOAD_ERROR",
                    "error": str(e),
                })
            continue

        for video_path, gt_path in datasets:
            logger.info("\n📹 Vidéo : %s", video_path.name)
            result_entry = {
                "model": model.name,
                "video": video_path.name,
                "status": "OK",
            }

            try:
                # ── Infos vidéo ──
                num_frames, fps, (w_h) = _get_video_info(video_path)
                result_entry["fps_source"] = fps
                result_entry["resolution"] = f"{w_h[0]}x{w_h[1]}"

                # ── Charger le GT ──
                gt_masks = _load_ground_truth_masks(gt_path, num_frames)
                if not gt_masks:
                    logger.warning("Pas de GT disponible pour %s.", video_path.name)

                # ── Étape 1 : Inférence (Streaming + Prefetch) ──
                # On évite d'écrire sur disque si on ne veut que les métriques
                masks_output = temp_dir / f"{model.name.replace(' ', '_')}_{video_path.stem}"
                
                # Si save_masks est False, on ne passe pas de masks_output à run_inference
                # mais on demande de collecter les masques en RAM pour l'évaluation.
                inference_result = run_inference(
                    model, 
                    video_path, 
                    masks_output if save_masks else None,
                    collect_masks=True
                )

                result_entry.update({
                    "latency_p95_ms": round(inference_result["latency_p95_ms"], 2),
                    "latency_mean_ms": round(inference_result["latency_mean_ms"], 2),
                    "latency_std_ms": round(inference_result["latency_std_ms"], 2),
                    "flops_per_frame": inference_result["flops_per_frame"],
                    "num_frames": inference_result["num_frames"],
                })

                # ── Étape 2 : Évaluation (RAM ou Disque) ──
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

                # ── Sauvegarde permanente si demandée ──
                if save_masks or save_video or save_segmented:
                    dest_base = masks_final_dir / model.name.replace(" ", "_") / video_path.stem
                    dest_base.mkdir(parents=True, exist_ok=True)
                    
                    # Utiliser les masques en RAM si l'écriture disque a été bypassée
                    m_temp = inference_result.get("masks")
                    if not m_temp or len(m_temp) == 0:
                        m_temp = _load_masks(masks_output)

                    if save_masks:
                        _save_masks(m_temp, dest_base)
                        logger.info("💾 Masques PNG sauvegardés dans : %s", dest_base)
                        
                    if save_video:
                        video_out_path = dest_base.parent / f"{video_path.stem}_{model.name.replace(' ', '_')}_mask.mp4"
                        _save_masks_as_video_fast(m_temp, video_out_path, fps)
                        logger.info("🎬 Vidéo masque sauvegardée : %s", video_out_path)

                    if save_segmented:
                        # Lire les frames à nouveau pour le détourage
                        frames_list, _ = _read_video_frames(video_path)
                        seg_video_path = dest_base.parent / f"{video_path.stem}_{model.name.replace(' ', '_')}_segmented.mp4"
                        _save_segmented_video(m_temp, frames_list, seg_video_path, fps)
                        logger.info("🎨 Vidéo détourée sauvegardée : %s", seg_video_path)

                # ── Nettoyage des masques temporaires ──
                if masks_output.exists():
                    shutil.rmtree(masks_output)
                    logger.info("🗑️  Masques temporaires supprimés : %s", masks_output)

            except Exception as e:
                logger.error("Erreur pour %s / %s : %s", model.name, video_path.name, e)
                result_entry["status"] = "ERROR"
                result_entry["error"] = str(e)

            all_results.append(result_entry)
            
            # Mise à jour progression
            current_combo += 1
            if progress_callback:
                progress_callback(current_combo, total_combos, f"Traité : {model.name} / {video_path.name}")

        # Libérer le modèle
        model.cleanup()


    # ── Générer les rapports ──
    _save_csv_report(all_results, output_dir)
    _save_json_report(all_results, output_dir)

    logger.info("=" * 72)
    logger.info("✅ BENCHMARK TERMINÉ — %d résultats enregistrés.", len(all_results))
    logger.info("   CSV  : %s", output_dir / RESULTS_CSV_FILENAME)
    logger.info("   JSON : %s", output_dir / RESULTS_JSON_FILENAME)
    logger.info("=" * 72)

    return all_results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Génération de rapports
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _save_csv_report(results: List[Dict], output_dir: Path) -> None:
    """Sauvegarde les résultats en CSV."""
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

    logger.info("Rapport CSV sauvegardé : %s", csv_path)


def _save_json_report(results: List[Dict], output_dir: Path) -> None:
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
