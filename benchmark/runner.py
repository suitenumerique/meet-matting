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
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Generator
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Ground-Truth format helpers (chroma key)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _is_chroma_key(frame_bgr: np.ndarray, threshold: float = 0.15) -> bool:
    """Return True if ≥threshold of pixels fall in the HSV green range."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255))
    return float(green.sum()) / (255.0 * green.size) >= threshold


def _chroma_key_to_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """Convert a chroma-key frame (green bg) to a float32 [0,1] foreground mask."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255))
    return cv2.bitwise_not(green).astype(np.float32) / 255.0


def get_frame_at(video_path: Path, frame_idx: int) -> Optional[np.ndarray]:
    """Return a single BGR frame at index frame_idx, or None on failure."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


class VideoPrefetcher:
    """Lit et pré-traite les frames dans un thread séparé."""
    def __init__(self, video_path: Path, queue_size: int = 128, target_size: Optional[Tuple[int, int]] = None):
        self.video_path = video_path
        self.queue = queue.Queue(maxsize=queue_size)
        self.target_size = target_size # (W, H)
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
    use_chroma: Optional[bool] = None  # determined on first readable frame

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
                
                # Nettoyage morphologique léger pour les bords
                kernel = np.ones((3, 3), np.uint8)
                masks.append(human_mask.astype(np.uint8))
            else:
                # C'est probablement déjà un masque binaire ou niveaux de gris
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                masks.append(gray.astype(np.uint8))
        cap.release()


    logger.info("GT chargé et traité : %d masques extraits depuis %s", len(masks), gt_dir)
    return masks



def _save_masks(masks: List[np.ndarray], output_dir: Path, start_idx: int = 0) -> None:
    """Sauvegarde une liste de masques en PNG dans un répertoire via un pool de threads."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    def _save_single(idx, mask):
        if mask.dtype == np.uint8:
            mask_u8 = mask
        else:
            mask_u8 = (mask * 255).astype(np.uint8)
        # IMWRITE_PNG_COMPRESSION=1 : Compression MINIMALE pour une vitesse maximale
        cv2.imwrite(str(output_dir / f"mask_{idx:06d}.png"), mask_u8, [cv2.IMWRITE_PNG_COMPRESSION, 1])

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
        pixelformat='yuv420p',
        ffmpeg_log_level='error',
        macro_block_size=1 # Évite les warnings sur les dimensions non divisibles par 16
    )
    
    for mask, frame in zip(masks, frames):
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
        ffmpeg_log_level='error',
        macro_block_size=1
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
    preds: List[np.ndarray], 
    gts: List[np.ndarray], 
    frames: List[np.ndarray], 
    output_base: Path, 
    fps: float
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
        if g.ndim == 3: g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
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
    logger.info("🧪 Vidéos de debug générées : %s (Intersection) et %s (Union)", inter_path.name, union_path.name)


def _load_masks(masks_dir: Path) -> List[np.ndarray]:
    """Recharge les masques sauvegardés depuis un répertoire."""
    files = sorted(masks_dir.glob("mask_*.png"))
    masks = []
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            masks.append(img.astype(np.float32) / 255.0)
    return masks


def load_masks_from_mask_video(mask_video_path: Path) -> List[np.ndarray]:
    """
    Reload float32 [0,1] masks from a *_mask.mp4 output file.

    The mask MP4 was saved as a greyscale video encoded in RGB
    (cv2.COLOR_GRAY2RGB). This function reverses that conversion.
    """
    masks = []
    cap = cv2.VideoCapture(str(mask_video_path))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        masks.append(gray.astype(np.uint8))
    cap.release()
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

    # Préparer le prefetcher avec la taille d'entrée du modèle (si fixe)
    target_size = None
    if model.input_size:
        # Convertir (H, W) -> (W, H) pour OpenCV
        target_size = (model.input_size[1], model.input_size[0])

    # Optimisation spécifique pour Mac : Limitation des threads OpenCV pour éviter les conflits
    import cv2
    cv2.setNumThreads(1)

    effective_batch_size = 1

    # On bypass le prefetcher pour MediaPipe pour éviter la surcharge de threads
    cap = cv2.VideoCapture(str(video_path))
    target_size = None
    if model.input_size:
        target_size = (model.input_size[1], model.input_size[0])

    with tqdm(total=num_frames, desc=f"  ⚡ {model.name}", unit="frame") as pbar:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            
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
                # SÉCURITÉ ABSOLUE : On vérifie que mask n'est pas None avant TOUTE opération
                if mask is not None:
                    try:
                        mask_u8 = (mask * 255).astype(np.uint8)
                        masks_in_ram.append(mask_u8)
                    except Exception as e:
                        logger.error(f"Erreur multiplication masque : {e}")
                        masks_in_ram.append(np.zeros((h, w), dtype=np.uint8))
                else:
                    # Si le modèle a échoué (None), on met un masque noir au lieu de crasher
                    masks_in_ram.append(np.zeros((h, w), dtype=np.uint8))
            
            frame_idx += 1
            pbar.update(1)
            total_frames += 1
            
        cap.release()

    # Calcul des statistiques
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
    video_indices: Optional[List[int]] = None,
    save_masks: bool = False,
    save_video: bool = False,
    save_segmented: bool = False,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    on_result: Optional[Callable[[Dict], None]] = None,
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

                    # ── Vidéos de DEBUG EVAL (Intersection / Union) ──
                    if gt_masks and len(gt_masks) > 0:
                        frames_list_debug, _ = _read_video_frames(video_path)
                        _save_eval_debug_videos(m_temp, gt_masks, frames_list_debug, dest_base, fps)

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
            
            if on_result:
                on_result(result_entry)

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
#  Post-hoc metrics on saved outputs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _load_cached_latency(output_dir: Path) -> Dict[str, Dict]:
    """Read per-model latency/FLOPs cached in benchmark_results.csv.

    Returns a dict: model_name → {latency_p95_ms, latency_mean_ms,
    latency_std_ms, flops_per_frame} (values may be None if not present).
    """
    csv_path = output_dir / RESULTS_CSV_FILENAME
    if not csv_path.exists():
        return {}

    cache: Dict[str, Dict] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row.get("model", "").strip()
            if not model or model in cache:
                continue
            entry: Dict = {}
            for key in ("latency_p95_ms", "latency_mean_ms", "latency_std_ms", "flops_per_frame"):
                raw = row.get(key, "").strip()
                try:
                    entry[key] = float(raw) if raw else None
                except ValueError:
                    entry[key] = None
            # Only cache if at least one latency value is non-None and non-zero
            if any(entry.get(k) for k in ("latency_p95_ms", "latency_mean_ms")):
                cache[model] = entry
    return cache


def _find_model_class_by_dir_name(dir_name: str):
    """Return the model *class* whose .name matches dir_name (with _ → space)."""
    from .models import MODEL_REGISTRY

    target = dir_name.replace("_", " ")
    for cls in MODEL_REGISTRY.values():
        try:
            if cls().name == target:
                return cls
        except Exception:
            continue
    return None


def compute_metrics_on_output(
    output_dir: Path = OUTPUT_DIR,
    gt_dir: Path = GROUND_TRUTH_DIR,
    videos_dir: Path = VIDEOS_DIR,
    model_filter: Optional[List[str]] = None,
    threshold: float = 0.5,
    measure_missing_latency: bool = True,
    on_result: Optional[Callable[[Dict], None]] = None,
    on_latency_status: Optional[Callable[[str], None]] = None,
) -> List[Dict]:
    """
    Compute quality metrics on previously generated *_mask.mp4 files.

    Latency/FLOPs are resolved in this order:
      1. Read from benchmark_results.csv (cache).
      2. If not cached and measure_missing_latency=True: load the model,
         run measure_latency() on a reference video, then unload.
      3. Otherwise left as None.

    Args:
        output_dir:               Root output directory (contains masks/).
        gt_dir:                   Folder of ground-truth videos/images.
        videos_dir:               Folder of source videos (needed for FWE).
        model_filter:             If set, only process these model_dir names.
        threshold:                Binarisation threshold forwarded to metrics.
        measure_missing_latency:  Re-measure latency when not in CSV.
        on_result:                Callback fired with each result dict.
        on_latency_status:        Callback fired with status strings during
                                  latency measurement.

    Returns:
        List of result dicts (same schema as run_benchmark).
    """
    from .metrics import compute_iou, compute_boundary_f_measure, compute_flow_warping_error

    masks_root = output_dir / "masks"
    if not masks_root.is_dir():
        logger.warning("No masks/ folder found in %s", output_dir)
        return []

    # ── Step 0: load latency cache ────────────────────────────────────────────
    latency_cache = _load_cached_latency(output_dir)
    logger.info("Latency cache: %d models found in CSV.", len(latency_cache))

    # Find one reference video for latency measurement (any will do)
    ref_videos = sorted(videos_dir.glob("*.mp4")) if videos_dir.is_dir() else []
    ref_video: Optional[Path] = ref_videos[0] if ref_videos else None

    all_results: List[Dict] = []

    model_dirs = sorted(
        d for d in masks_root.iterdir()
        if d.is_dir() and (model_filter is None or d.name in model_filter)
    )

    for model_dir in model_dirs:
        model_name = model_dir.name.replace("_", " ")
        mask_files = sorted(model_dir.glob("*_mask.mp4"))
        if not mask_files:
            continue

        # ── Step 1: resolve latency for this model ────────────────────────────
        lat_entry: Dict = latency_cache.get(model_name, {})

        if not lat_entry and measure_missing_latency and ref_video is not None:
            model_cls = _find_model_class_by_dir_name(model_dir.name)
            if model_cls is not None:
                try:
                    msg = f"Mesure de la latence pour {model_name}…"
                    logger.info(msg)
                    if on_latency_status:
                        on_latency_status(msg)

                    model_inst = model_cls()
                    model_inst.load()

                    # FLOPs: use first frame resolution from the reference video
                    _, _, (w_ref, h_ref) = _get_video_info(ref_video)
                    flops = model_inst.get_flops((3, h_ref, w_ref))

                    lat_result = measure_latency(model_inst, ref_video)
                    lat_entry = {**lat_result, "flops_per_frame": flops}
                    model_inst.cleanup()

                    msg = f"Latence {model_name}: p95={lat_entry['latency_p95_ms']:.1f} ms"
                    logger.info(msg)
                    if on_latency_status:
                        on_latency_status(msg)
                except Exception as lat_err:
                    logger.warning("Latency measurement failed for %s: %s", model_name, lat_err)
                    if on_latency_status:
                        on_latency_status(f"⚠️ Latence non disponible pour {model_name}: {lat_err}")
            else:
                logger.warning("Model class not found for dir '%s'.", model_dir.name)
                if on_latency_status:
                    on_latency_status(f"⚠️ Classe modèle introuvable pour {model_dir.name}")

        # ── Step 2: compute quality metrics per video ─────────────────────────
        for mask_file in mask_files:
            video_stem = mask_file.stem.replace(f"_{model_dir.name}_mask", "")
            gt_path = gt_dir / f"{video_stem}.mp4"
            video_path = videos_dir / f"{video_stem}.mp4"

            result_entry: Dict = {
                "model": model_name,
                "video": f"{video_stem}.mp4",
                "status": "OK",
                **lat_entry,
            }

            try:
                pred_masks = load_masks_from_mask_video(mask_file)
                if not pred_masks:
                    raise ValueError(f"No frames in {mask_file}")

                gt_masks = _load_ground_truth_masks(gt_path, len(pred_masks)) if gt_path.exists() else []

                if not gt_masks:
                    logger.warning("No GT found for %s.", video_stem)
                    result_entry["status"] = "NO_GT"
                    all_results.append(result_entry)
                    if on_result:
                        try:
                            on_result(result_entry)
                        except Exception:
                            pass
                    continue

                n = min(len(pred_masks), len(gt_masks))

                iou = compute_iou(pred_masks[:n], gt_masks[:n], threshold=threshold)

                bf_scores = [
                    compute_boundary_f_measure(p, g, threshold=threshold)
                    for p, g in zip(pred_masks[:n], gt_masks[:n])
                ]

                fwe = 0.0
                if video_path.exists():
                    fwe = compute_flow_warping_error(
                        pred_masks[:n],
                        _iter_video_frames(video_path),
                        threshold=threshold,
                    )
                else:
                    logger.warning("Source video not found for FWE: %s", video_path)

                result_entry.update({
                    "iou_mean": round(iou, 4),
                    "iou_std": 0.0,
                    "boundary_f_mean": round(float(np.mean(bf_scores)), 4),
                    "boundary_f_std": round(float(np.std(bf_scores)), 4),
                    "flow_warping_error": round(fwe, 4),
                    "num_frames": n,
                    "threshold": threshold,
                })

                logger.info(
                    "%s / %s — IoU=%.4f  BF=%.4f  FWE=%.4f",
                    model_name, video_stem, iou, np.mean(bf_scores), fwe,
                )

            except Exception as e:
                logger.error("metrics error %s / %s: %s", model_name, video_stem, e)
                result_entry["status"] = "ERROR"
                result_entry["error"] = str(e)

            all_results.append(result_entry)
            if on_result is not None:
                try:
                    on_result(result_entry)
                except Exception as cb_exc:
                    logger.debug("on_result callback error: %s", cb_exc)

    _save_csv_report(all_results, output_dir)
    _save_json_report(all_results, output_dir)
    logger.info("Post-hoc metrics done: %d results saved.", len(all_results))
    return all_results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Report generation
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
