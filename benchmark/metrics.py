"""
Fonctions de calcul des métriques de qualité pour le Video Matting.

Métriques implémentées :
  - IoU (Intersection over Union)
  - Boundary F-measure (F1 sur les contours)
  - Flow Warping Error (stabilité temporelle)
"""

import logging
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial

# bound_ratio par défaut (règle DAVIS) : la tolérance de matching de contour
# est 0.8% de la diagonale de l'image (~2 px à 480p, ~4 px à 1080p).
DEFAULT_BOUND_RATIO = 0.008

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  IoU — Intersection over Union
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_iou(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    threshold: float = 0.5,
) -> float:
    """
    Calcule l'IoU global sur toute la séquence vidéo.

    Intersection et union sont cumulées sur l'ensemble des pixels de toutes
    les frames, puis le ratio est calculé une seule fois. C'est un IoU
    pixel-pondéré : les frames où l'objet est plus grand contribuent
    davantage au score final.

    Conventions (alignées sur metrics/performance_metrics.py) :
      - Binarisation avec `threshold` (défaut 0.5, seuil inclusif).
      - Si l'union est vide sur toute la séquence -> retourne 1.0 (accord
        parfait par convention : les deux masques sont vides partout).
      - Si les shapes diffèrent entre pred et GT sur une frame, le masque
        prédit est redimensionné sur la shape du GT (interpolation NEAREST).

    Args:
        pred_masks: Liste de masques prédits, chacun (H, W) ou (H, W, 1),
                    valeurs dans [0, 1].
        gt_masks:   Liste de masques ground truth, chacun (H, W), dans [0, 1].
        threshold:  Seuil de binarisation.

    Returns:
        IoU global dans [0, 1].
    """
    n = min(len(pred_masks), len(gt_masks))
    if n == 0:
        return 1.0

    total_inter = 0
    total_union = 0
    for i in range(n):
        pred = np.asarray(pred_masks[i]).squeeze()
        gt = np.asarray(gt_masks[i]).squeeze()

        pred_bin = (pred >= threshold).astype(np.uint8)
        gt_bin = (gt >= threshold).astype(np.uint8)

        if pred_bin.shape != gt_bin.shape:
            logger.debug(
                "IoU: resize pred %s -> gt %s", pred_bin.shape, gt_bin.shape
            )
            pred_bin = cv2.resize(
                pred_bin, (gt_bin.shape[1], gt_bin.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        total_inter += int(np.logical_and(pred_bin, gt_bin).sum())
        total_union += int(np.logical_or(pred_bin, gt_bin).sum())

    if total_union == 0:
        return 1.0

    return float(total_inter / total_union)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Boundary F-measure
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _extract_boundary(mask_bin: np.ndarray) -> np.ndarray:
    """
    Extrait un contour 1-pixel par soustraction morphologique :
        boundary = mask - erode(mask)

    Noyau croix 3x3 (4-voisinage). Contrairement à Canny, cette méthode
    capture aussi les bords des trous internes du masque, ce qui les
    pénalise dans le F-measure — comportement standard DAVIS, pertinent
    pour le matting (les trous produisent des artefacts visibles).
    """
    mask_u8 = mask_bin.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    eroded = cv2.erode(mask_u8, kernel, iterations=1)
    return mask_u8 - eroded


def compute_boundary_f_measure(
    pred: np.ndarray,
    gt: np.ndarray,
    bound_ratio: float = DEFAULT_BOUND_RATIO,
    threshold: float = 0.5,
) -> float:
    """
    Boundary F-measure binaire d'une frame (Perazzi et al. 2016, DAVIS).

    Méthode alignée sur metrics/performance_metrics.py :
      - Extraction du contour par érosion morphologique (trous internes
        pénalisés — règle DAVIS, souhaitable pour le matting).
      - Tolérance spatiale adaptée à la résolution :
            bound_radius = max(1, round(bound_ratio * sqrt(H^2 + W^2)))
        Avec bound_ratio=0.008 (DAVIS), cela donne ~2 px à 480p, ~4 px à
        1080p. Un pixel prédit est "matché" s'il tombe dans le contour GT
        dilaté par un disque de rayon bound_radius (approx morphologique
        du matching bipartite de Martin et al. 2004).

    Args:
        pred:        Masque prédit (H, W) ou (H, W, 1), valeurs dans [0, 1].
        gt:          Masque ground truth (H, W), valeurs dans [0, 1].
        bound_ratio: Fraction de la diagonale utilisée comme rayon de
                     tolérance. Défaut 0.008 (règle DAVIS).
        threshold:   Seuil de binarisation (inclusif).

    Returns:
        Score F1 des contours dans [0, 1].
    """
    pred = np.asarray(pred).squeeze()
    gt = np.asarray(gt).squeeze()

    pred_bin = (pred >= threshold).astype(np.uint8)
    gt_bin = (gt >= threshold).astype(np.uint8)

    if pred_bin.shape != gt_bin.shape:
        logger.debug(
            "BoundaryF: resize pred %s -> gt %s", pred_bin.shape, gt_bin.shape
        )
        pred_bin = cv2.resize(
            pred_bin, (gt_bin.shape[1], gt_bin.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    pred_boundary = _extract_boundary(pred_bin)
    gt_boundary = _extract_boundary(gt_bin)

    n_pred = int(pred_boundary.sum())
    n_gt = int(gt_boundary.sum())

    # Cas limites : aucun contour d'un côté
    if n_pred == 0 and n_gt == 0:
        return 1.0  # accord : pas d'objet nulle part
    if n_pred == 0 or n_gt == 0:
        return 0.0  # un seul contour existe -> pas de matching possible

    # Rayon de tolérance en pixels (règle DAVIS : % de la diagonale)
    h, w = gt_bin.shape[:2]
    diag = np.sqrt(h ** 2 + w ** 2)
    bound_radius = max(1, int(np.round(bound_ratio * diag)))
    ksize = 2 * bound_radius + 1
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

    # Matching morphologique : un pixel prédit est matché s'il tombe dans
    # le contour GT dilaté (numérateur de la précision). Symétriquement
    # pour le recall. Les deux comptes diffèrent quand les contours ont
    # des densités différentes.
    pred_dil = cv2.dilate(pred_boundary, disk)
    gt_dil = cv2.dilate(gt_boundary, disk)
    matched_pred = int(np.logical_and(pred_boundary, gt_dil).sum())
    matched_gt = int(np.logical_and(gt_boundary, pred_dil).sum())

    precision = matched_pred / n_pred
    recall = matched_gt / n_gt

    if precision + recall == 0:
        return 0.0

    return float(2.0 * precision * recall / (precision + recall))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Flow Warping Error — Stabilité temporelle
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_flow_warping_error(
    masks: List[np.ndarray],
    frames: Optional[Iterable[np.ndarray]] = None,
) -> float:
    """
    Calcule le Flow Warping Error moyen sur une séquence de masques.

    Mesure la cohérence temporelle : warp le masque t vers t+1 via le flux
    optique, puis mesure l'erreur entre le masque warpé et le masque réel t+1.

    Args:
        masks: Liste de masques (H, W) en float [0, 1], ordonnés temporellement.
        frames: Itérable de frames RGB (H, W, 3) correspondantes pour le calcul
                du flux optique. Si None, utilise les masques eux-mêmes.

    Returns:
        Erreur moyenne de warping dans [0, 1]. Plus bas = meilleure stabilité.
    """
    if len(masks) < 2:
        return 0.0

    errors = []
    
    # Préparer l'itérateur de frames si fourni
    frame_it = iter(frames) if frames is not None else None
    frame_curr = next(frame_it, None) if frame_it else None

    for i in range(len(masks) - 1):
        mask_curr = masks[i].astype(np.float32)
        mask_next = masks[i + 1].astype(np.float32)

        # Resize si nécessaire
        if mask_curr.shape != mask_next.shape:
            mask_curr = cv2.resize(
                mask_curr, (mask_next.shape[1], mask_next.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        
        frame_next = next(frame_it, None) if frame_it else None
        
        if frame_curr is not None and frame_next is not None:
            if len(frame_curr.shape) == 3:
                if frame_curr.shape[2] == 3:
                    gray_curr = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2GRAY)
                elif frame_curr.shape[2] == 4:
                    gray_curr = cv2.cvtColor(frame_curr, cv2.COLOR_BGRA2GRAY)
                else:
                    gray_curr = frame_curr[:, :, 0]
            else:
                gray_curr = frame_curr

            if len(frame_next.shape) == 3:
                if frame_next.shape[2] == 3:
                    gray_next = cv2.cvtColor(frame_next, cv2.COLOR_BGR2GRAY)
                elif frame_next.shape[2] == 4:
                    gray_next = cv2.cvtColor(frame_next, cv2.COLOR_BGRA2GRAY)
                else:
                    gray_next = frame_next[:, :, 0]
            else:
                gray_next = frame_next
        else:
            # Utiliser les masques comme référence de flux
            gray_curr = (mask_curr.squeeze() * 255).astype(np.uint8)
            gray_next = (mask_next.squeeze() * 255).astype(np.uint8)

        # ── Optimisation Performance : Downsample pour le flux optique ──
        # La stabilité temporelle peut être évaluée avec précision en basse résolution.
        target_w = 480
        h_orig, w_orig = gray_curr.shape[:2]
        if w_orig > target_w:
            scale = target_w / w_orig
            target_h = int(h_orig * scale)
            g_curr_small = cv2.resize(gray_curr, (target_w, target_h), interpolation=cv2.INTER_AREA)
            g_next_small = cv2.resize(gray_next, (target_w, target_h), interpolation=cv2.INTER_AREA)
            m_curr_small = cv2.resize(mask_curr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            m_next_small = cv2.resize(mask_next, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        else:
            g_curr_small, g_next_small = gray_curr, gray_next
            m_curr_small, m_next_small = mask_curr, mask_next

        # Calcul du flux optique dense (DIS) sur résolution réduite — Beaucoup plus rapide que Farneback
        if hasattr(cv2, 'DISOpticalFlow_create'):
            dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
            flow = dis.calc(g_curr_small, g_next_small, None)
        else:
            # Fallback Farneback si DIS absent
            flow = cv2.calcOpticalFlowFarneback(
                g_curr_small, g_next_small,
                None,
                pyr_scale=0.5,
                levels=2,
                winsize=11,
                iterations=2,
                poly_n=5,
                poly_sigma=1.1,
                flags=0,
            )

        # Warping du masque courant vers le frame suivant
        h_s, w_s = m_curr_small.shape[:2]
        flow_map_x = np.arange(w_s, dtype=np.float32)[np.newaxis, :] + flow[..., 0]
        flow_map_y = np.arange(h_s, dtype=np.float32)[:, np.newaxis] + flow[..., 1]
    
        warped_mask = cv2.remap(
            m_curr_small,
            flow_map_x,
            flow_map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # Erreur L1 entre masque warpé et masque réel (sur résolution réduite)
        error = np.abs(warped_mask - m_next_small).mean()
        errors.append(error)
        
        # Le frame "next" devient le "curr" pour la prochaine itération
        frame_curr = frame_next

    return float(np.mean(errors))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Agrégation par vidéo
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _compute_single_frame_bf(pair):
    """Calcul Boundary F-measure pour une frame (helper multiprocessing)."""
    pred, gt = pair
    return compute_boundary_f_measure(pred, gt)


def compute_all_metrics(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    frames: Optional[Iterable[np.ndarray]] = None,
) -> dict:
    """
    Calcule toutes les métriques de qualité sur une séquence vidéo.

    IoU : calculé globalement sur la séquence (pixel-weighted, un seul
    ratio pour toute la vidéo). `iou_std` est exposé à 0.0 — la métrique
    n'a plus de dispersion par frame. La clé est conservée pour
    compatibilité avec les rapports en aval.
    """
    n = min(len(pred_masks), len(gt_masks))
    if n == 0:
        logger.error("Aucun masque disponible pour le calcul des métriques.")
        return {
            "iou_mean": 0.0,
            "iou_std": 0.0,
            "boundary_f_mean": 0.0,
            "boundary_f_std": 0.0,
            "flow_warping_error": 0.0,
        }

    # IoU global sur toute la séquence (un seul ratio agrégé)
    iou_global = compute_iou(pred_masks[:n], gt_masks[:n])

    # Boundary F par frame (parallélisé — ThreadPool, OpenCV libère le GIL)
    pairs = list(zip(pred_masks[:n], gt_masks[:n]))
    with ThreadPoolExecutor(max_workers=8) as executor:
        boundary_fs = list(executor.map(_compute_single_frame_bf, pairs))

    # Flow warping error sur les prédictions
    fwe = compute_flow_warping_error(
        pred_masks[:n],
        frames,
    )

    return {
        "iou_mean": iou_global,
        "iou_std": 0.0,
        "boundary_f_mean": float(np.mean(boundary_fs)),
        "boundary_f_std": float(np.std(boundary_fs)),
        "flow_warping_error": fwe,
    }
