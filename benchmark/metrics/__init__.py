"""
Fonctions de calcul des métriques de qualité pour le Video Matting.
Version Optimisée (Vision Pro) :
- Alignement Auto (Shift & Flip invariant)
- FWE Turbo via DIS Flow + ROI Optimization
- Parallélisation ThreadPool (8 workers)
- Streaming support (Mémoire efficiente)
"""

import logging
import gc
from typing import Iterable, List, Optional, Tuple, Generator, Dict, Union
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from .performance_metrics import (
    _get_alignment_params,
    _apply_alignment,
    compute_iou as _compute_iou_opt,
    compute_boundary_f_measure as _compute_bf_parallel,
    compute_flow_warping_error_dis as _compute_fwe_opt
)

logger = logging.getLogger(__name__)

def compute_all_metrics(
    pred_masks: Union[List[np.ndarray], Iterable[np.ndarray]],
    gt_masks: Union[List[np.ndarray], Iterable[np.ndarray]],
    frames: Optional[Iterable[np.ndarray]] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Calcule toutes les métriques de segmentation vidéo avec les optimisations "Expert".
    
    Args:
        pred_masks: Masques prédits (uint8 ou float32).
        gt_masks: Masques Ground Truth.
        frames: Flux de frames vidéo originales (requis pour FWE).
        threshold: Seuil de binarisation.
    """
    # 1. IoU Global (avec alignement auto intégré et streaming)
    # On utilise les versions optimisées de performance_metrics.py
    iou_val = _compute_iou_opt(pred_masks, gt_masks, threshold=threshold)
    
    # 2. Boundary F-measure (Parallélisé, 8 workers)
    # BF est très lent, donc on parallélisera par frame.
    bf_val = _compute_bf_parallel(pred_masks, gt_masks, threshold=threshold, num_workers=8)
    
    # 3. Flow Warping Error (FWE)
    # Requiert la vidéo originale pour le flux optique + validité Lai.
    fwe_val = 0.0
    if frames is not None:
        fwe_val = _compute_fwe_opt(frames, pred_masks, threshold=threshold)
    
    # Nettoyage explicite des buffers OpenCV et cache numpy
    gc.collect()

    results = {
        "iou_mean": float(iou_val),
        "iou_std": 0.0, # L'IoU de zone est global par définition
        "boundary_f_mean": float(bf_val),
        "boundary_f_std": 0.0,
        "flow_warping_error": float(fwe_val),
    }
    
    return results

# Reste du fichier maintenu pour compatibilité si des fonctions individuelles sont appelées
def compute_iou(pred_masks, gt_masks, threshold=0.5):
    return _compute_iou_opt(pred_masks, gt_masks, threshold)

def compute_boundary_f_measure(pred_masks, gt_masks, threshold=0.5):
    return _compute_bf_parallel(pred_masks, gt_masks, threshold)

def compute_flow_warping_error(masks, frames, threshold=0.5):
    return _compute_fwe_opt(frames, masks, threshold)
