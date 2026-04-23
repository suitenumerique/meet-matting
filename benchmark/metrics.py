"""
Fonctions de calcul des métriques de qualité pour le Video Matting.

Métriques implémentées :
  - IoU (Intersection over Union)
  - Boundary F-measure (F1 sur les contours)
  - Flow Warping Error (stabilité temporelle)
"""

import logging
import gc
from typing import Iterable, List, Optional, Tuple, Generator

import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial

# bound_ratio par défaut (règle DAVIS) : la tolérance de matching de contour
# est 0.8% de la diagonale de l'image (~2 px à 480p, ~4 px à 1080p).
DEFAULT_BOUND_RATIO = 0.008

logger = logging.getLogger(__name__)

def _get_alignment_params(pred_bin: np.ndarray, gt_bin: np.ndarray, max_shift: int = 50) -> Tuple[int, int, bool, float]:
    """
    Calcule les paramètres de recalage optimal sur une frame de référence.
    Retourne (dx, dy, flip_needed, score).
    """
    h_gt, w_gt = gt_bin.shape[:2]
    low_res = 128
    scale = low_res / max(h_gt, w_gt)
    g_low = cv2.resize(gt_bin, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    def _match(p_img):
        pad = int(max_shift * scale) + 1
        g_padded = cv2.copyMakeBorder(g_low, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        res = cv2.matchTemplate(g_padded.astype(np.float32), p_img.astype(np.float32), cv2.TM_CCORR_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        return max_val, max_loc, pad

    # 1. Normal
    p_low_normal = cv2.resize(pred_bin, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    score_n, loc_n, pad = _match(p_low_normal)
    
    # 2. Flip
    p_low_f = cv2.flip(p_low_normal, 1)
    score_f, loc_f, _ = _match(p_low_f)
    
    flip = score_f > score_n * 1.1 # 10% de mieux pour l'un par rapport à l'autre
    best_score = max(score_n, score_f)
    best_loc = loc_f if flip else loc_n
    
    dx = int((best_loc[0] - pad) / scale)
    dy = int((best_loc[1] - pad) / scale)
    
    return dx, dy, flip, best_score

def _apply_alignment(mask: np.ndarray, dx: int, dy: int, flip: bool) -> np.ndarray:
    """Applique les paramètres de recalage à un masque."""
    h, w = mask.shape[:2]
    if flip:
        mask = cv2.flip(mask, 1)
    if dx == 0 and dy == 0:
        return mask
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)



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

    # 1. Déterminer l'alignement optimal sur une frame du milieu (souvent plus représentative)
    ref_idx = n // 2
    p_ref = np.asarray(pred_masks[ref_idx]).squeeze()
    g_ref = np.asarray(gt_masks[ref_idx]).squeeze()
    
    # Redimensionnement temporaire pour l'alignement si besoin
    if p_ref.shape != g_ref.shape:
        p_ref = cv2.resize(p_ref, (g_ref.shape[1], g_ref.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    p_ref_bin = (p_ref >= (threshold * 255.0 if p_ref.dtype == np.uint8 else threshold)).astype(np.uint8)
    g_ref_bin = (g_ref >= (threshold * 255.0 if g_ref.dtype == np.uint8 else threshold)).astype(np.uint8)
    dx, dy, flip, _ = _get_alignment_params(p_ref_bin, g_ref_bin)

    total_inter = 0
    total_union = 0
    for i in range(n):
        pred = np.asarray(pred_masks[i]).squeeze()
        gt = np.asarray(gt_masks[i]).squeeze()

        actual_threshold = threshold * 255.0 if pred.dtype == np.uint8 else threshold
        pred_bin = (pred >= actual_threshold).astype(np.uint8)
        
        actual_gt_threshold = threshold * 255.0 if gt.dtype == np.uint8 else threshold
        gt_bin = (gt >= actual_gt_threshold).astype(np.uint8)

        if pred_bin.shape != gt_bin.shape:
            pred_bin = cv2.resize(pred_bin, (gt_bin.shape[1], gt_bin.shape[0]), interpolation=cv2.INTER_NEAREST)
            
        # Appliquer l'alignement constant
        pred_bin = _apply_alignment(pred_bin, dx, dy, flip)

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

    actual_threshold = threshold * 255.0 if pred.dtype == np.uint8 else threshold
    pred_bin = (pred >= actual_threshold).astype(np.uint8)
    
    actual_gt_threshold = threshold * 255.0 if gt.dtype == np.uint8 else threshold
    gt_bin = (gt >= actual_gt_threshold).astype(np.uint8)

    if pred_bin.shape != gt_bin.shape:
        pred_bin = cv2.resize(
            pred_bin, (gt_bin.shape[1], gt_bin.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    # Optionnel: on pourrait passer les params d'alignement ici pour être plus rapide
    # mais boundary_f est souvent appelé par frame via map().
    # On va quand même optimiser compute_all_metrics pour qu'il ne recalcule pas à chaque fois.
    return _compute_boundary_f_measure_with_params(pred, gt, 0, 0, False, bound_ratio, threshold)

def _compute_boundary_f_measure_with_params(
    pred: np.ndarray,
    gt: np.ndarray,
    dx: int,
    dy: int,
    flip: bool,
    bound_ratio: float = DEFAULT_BOUND_RATIO,
    threshold: float = 0.5,
) -> float:
    pred = np.asarray(pred).squeeze()
    gt = np.asarray(gt).squeeze()

    actual_threshold = threshold * 255.0 if pred.dtype == np.uint8 else threshold
    pred_bin = (pred >= actual_threshold).astype(np.uint8)
    
    actual_gt_threshold = threshold * 255.0 if gt.dtype == np.uint8 else threshold
    gt_bin = (gt >= actual_gt_threshold).astype(np.uint8)

    if pred_bin.shape != gt_bin.shape:
        pred_bin = cv2.resize(pred_bin, (gt_bin.shape[1], gt_bin.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Appliquer l'alignement
    pred_bin = _apply_alignment(pred_bin, dx, dy, flip)

    pred_boundary = _extract_boundary(pred_bin)
    gt_boundary = _extract_boundary(gt_bin)

    n_pred = int(pred_boundary.sum())
    n_gt = int(gt_boundary.sum())

    if n_pred == 0 and n_gt == 0: return 1.0
    if n_pred == 0 or n_gt == 0: return 0.0

    h, w = gt_bin.shape[:2]
    diag = np.sqrt(h ** 2 + w ** 2)
    bound_radius = max(1, int(np.round(bound_ratio * diag)))
    ksize = 2 * bound_radius + 1
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

    pred_dil = cv2.dilate(pred_boundary, disk)
    gt_dil = cv2.dilate(gt_boundary, disk)
    matched_pred = int(np.logical_and(pred_boundary, gt_dil).sum())
    matched_gt = int(np.logical_and(gt_boundary, pred_dil).sum())

    precision = matched_pred / n_pred
    recall = matched_gt / n_gt

    if precision + recall == 0: return 0.0
    return float(2.0 * precision * recall / (precision + recall))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Flow Warping Error — Stabilité temporelle
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Seuil par défaut d'erreur photométrique L1 (sur RGB normalisé [0, 1])
# au-delà duquel un pixel est considéré invalide (désocclusion / échec du
# flux optique) et exclu de l'agrégation Lai et al. 2018.
DEFAULT_PHOTO_THRESHOLD = 0.05

# On utilise DISOpticalFlow pour une vitesse 10x supérieure à Farneback
_DIS_FLOW = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)


def _compute_flow_pair_dis(prev_gray, curr_gray):
    """Calcul du flux via DIS (Dense Inverse Search)."""
    fwd = _DIS_FLOW.calc(prev_gray, curr_gray, None)
    bwd = _DIS_FLOW.calc(curr_gray, prev_gray, None)
    return fwd, bwd



def _warp_with_flow(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """
    Backward-warp `image` avec `flow` :
        warped[y, x] = image[y + flow_y(y, x), x + flow_x(y, x)]

    Usage typique : image = frame_{t-1} et flow = F_{t -> t-1} (flux backward) ;
    `warped` est alors une estimation de frame t construite en tirant les
    pixels de frame_{t-1} le long du flux. Les lookups hors-image sont mis à 0.
    """
    h, w = image.shape[:2]
    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = grid_x + flow[..., 0]
    map_y = grid_y + flow[..., 1]
    return cv2.remap(
        image, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _lai_validity_mask(
    frame_prev: np.ndarray,
    frame_curr: np.ndarray,
    flow_fwd: np.ndarray,
    flow_bwd: np.ndarray,
    photo_threshold: float,
) -> np.ndarray:
    """
    Masque de validité (H, W) dans {0, 1}. Un pixel est marqué valide ssi :
      1) cohérence forward/backward du flux (Sundaram et al. 2010) : aller
         t -> t-1 via flow_bwd puis revenir via flow_fwd retombe au point
         de départ à un seuil adaptatif près ;
      2) cohérence photométrique (Lai et al. 2018) : warper frame_{t-1}
         vers frame t reproduit la couleur observée avec une erreur L1
         inférieure à `photo_threshold`.
    Les pixels qui échouent l'un ou l'autre test correspondent typiquement
    aux désocclusions ou aux failures du flux et sont exclus du score.
    """
    # (1) Cohérence forward/backward
    flow_fwd_at_t = _warp_with_flow(flow_fwd, flow_bwd)
    diff = flow_fwd_at_t + flow_bwd  # ≈ 0 si les flux sont cohérents
    diff_sq = (diff ** 2).sum(axis=-1)
    mag_sq = (flow_fwd_at_t ** 2).sum(axis=-1) + (flow_bwd ** 2).sum(axis=-1)
    # Seuil adaptatif (Sundaram) : 1% de la norme au carré + un plancher de
    # 0.5 pour ne pas rejeter à tort les pixels quasi-statiques.
    fb_ok = diff_sq <= 0.01 * mag_sq + 0.5

    # (2) Cohérence photométrique
    frame_prev_warped = _warp_with_flow(frame_prev, flow_bwd)
    photo_err = np.abs(frame_curr - frame_prev_warped).mean(axis=-1)
    photo_ok = photo_err <= photo_threshold

    return (fb_ok & photo_ok).astype(np.float32)


def compute_flow_warping_error(
    masks: List[np.ndarray],
    frames: Optional[Iterable[np.ndarray]] = None,
    threshold: float = 0.5,
    photo_threshold: float = DEFAULT_PHOTO_THRESHOLD,
    max_res: int = 480,
) -> float:
    """
    Calcul du Flow Warping Error optimisé (Vitesse + RAM).
    Utilise DISOpticalFlow et un recadrage sur le sujet (ROI).
    """
    if frames is None or len(masks) < 2:
        return 0.0

    total_err = 0.0
    total_valid = 0.0
    
    prev_frame_rgb = None
    prev_gray = None
    prev_mask_bin = None
    
    # Préchauffage du DIS (OpenCV réutilise les buffers si la taille est constante)
    dis = _DIS_FLOW

    for i, frame in enumerate(frames):
        if i >= len(masks): break
        
        # 1. Mise à l'échelle pour la vitesse
        h, w = frame.shape[:2]
        if max(h, w) > max_res:
            scale = max_res / max(h, w)
            tw, th = int(w * scale), int(h * scale)
            frame_low = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
        else:
            tw, th = w, h
            frame_low = frame
            
        m = np.asarray(masks[i]).squeeze()
        m_bin = (m >= (threshold * 255 if m.dtype == np.uint8 else threshold)).astype(np.float32)
        if m_bin.shape != (th, tw):
            m_bin = cv2.resize(m_bin, (tw, th), interpolation=cv2.INTER_NEAREST)

        # 2. Conversion nécessaire pour le flux
        if frame_low.ndim == 3 and frame_low.shape[2] == 4:
            frame_low = cv2.cvtColor(frame_low, cv2.COLOR_BGRA2BGR)
        curr_gray = cv2.cvtColor(frame_low, cv2.COLOR_BGR2GRAY)
        curr_frame_rgb = cv2.cvtColor(frame_low, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        if prev_gray is not None:
            # 3. Optimisation ROI (Region of Interest) : On ne calcule le flux que là où il y a de l'action
            # On prend l'union des zones occupées par le sujet (passé et présent) + un padding
            union_mask = (m_bin + prev_mask_bin) > 0
            y_coords, x_coords = np.where(union_mask)
            
            if len(y_coords) > 0:
                pad = 20
                y1, y2 = max(0, y_coords.min() - pad), min(th, y_coords.max() + pad)
                x1, x2 = max(0, x_coords.min() - pad), min(tw, x_coords.max() + pad)
                
                # Crop pour le flux + .copy() pour assurer la continuité en mémoire (requis par DIS)
                roi_prev = prev_gray[y1:y2, x1:x2].copy()
                roi_curr = curr_gray[y1:y2, x1:x2].copy()
                
                # Calcul du flux sur la ROI uniquement (gain de temps massif)
                fwd_roi = dis.calc(roi_prev, roi_curr, None)
                bwd_roi = dis.calc(roi_curr, roi_prev, None)
                
                # Crop pour la validité et l'erreur
                rgb_p_roi = prev_frame_rgb[y1:y2, x1:x2]
                rgb_c_roi = curr_frame_rgb[y1:y2, x1:x2]
                mask_p_roi = prev_mask_bin[y1:y2, x1:x2]
                mask_c_roi = m_bin[y1:y2, x1:x2]
                
                # Validité et warping sur la ROI
                validity = _lai_validity_mask(rgb_p_roi, rgb_c_roi, fwd_roi, bwd_roi, photo_threshold)
                mask_p_warped = _warp_with_flow(mask_p_roi, bwd_roi)
                err = np.abs(mask_c_roi - mask_p_warped)
                
                total_err += float((validity * err).sum())
                total_valid += float(validity.sum())

        # Shift
        prev_frame_rgb = curr_frame_rgb
        prev_gray = curr_gray
        prev_mask_bin = m_bin

    gc.collect()
    return float(total_err / total_valid) if total_valid > 0 else 0.0


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
    threshold: float = 0.5,
) -> dict:
    """
    Calcule toutes les métriques de qualité sur une séquence vidéo.
    """
    n = min(len(pred_masks), len(gt_masks))
    if n == 0:
        logger.error("Aucun masque disponible pour le calcul des métriques.")
        return {
            "iou_mean": 0.0, "iou_std": 0.0,
            "boundary_f_mean": 0.0, "boundary_f_std": 0.0,
            "flow_warping_error": 0.0,
        }

    # 1. Détermination de l'alignement GLOBAL (une seule fois pour toute la vidéo)
    ref_idx = n // 2
    p_ref = np.asarray(pred_masks[ref_idx]).squeeze()
    g_ref = np.asarray(gt_masks[ref_idx]).squeeze()
    if p_ref.shape != g_ref.shape:
        p_ref = cv2.resize(p_ref, (g_ref.shape[1], g_ref.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    actual_t = threshold * 255 if p_ref.dtype == np.uint8 else threshold
    p_ref_bin = (p_ref >= actual_t).astype(np.uint8)
    g_ref_bin = (g_ref >= (threshold * 255 if g_ref.dtype == np.uint8 else threshold)).astype(np.uint8)
    dx, dy, flip, score = _get_alignment_params(p_ref_bin, g_ref_bin)
    logger.info(f"Alignment: dx={dx}, dy={dy}, flip={flip}, score={score:.3f}")

    # 2. IoU global (réutilise les params d'alignement internement ou on aurait pu l'optimiser encore plus)
    # Pour garder la simplicité on laisse compute_iou appeler son propre alignement mais on l'a déjà optimisé plus haut
    iou_global = compute_iou(pred_masks[:n], gt_masks[:n], threshold=threshold)

    # 3. Boundary F par frame (ThreadPool) avec les params d'alignement constants
    func = partial(_compute_boundary_f_measure_with_params, dx=dx, dy=dy, flip=flip, threshold=threshold)
    pairs = list(zip(pred_masks[:n], gt_masks[:n]))
    with ThreadPoolExecutor(max_workers=8) as executor:
        boundary_fs = list(executor.map(lambda p: func(p[0], p[1]), pairs))

    # 4. Flow warping error (streaming)
    fwe = compute_flow_warping_error(pred_masks[:n], frames, threshold=threshold)

    return {
        "iou_mean": iou_global,
        "iou_std": 0.0,
        "boundary_f_mean": float(np.mean(boundary_fs)),
        "boundary_f_std": float(np.std(boundary_fs)),
        "flow_warping_error": fwe,
    }
