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
# Seuil par défaut d'erreur photométrique L1 (sur RGB normalisé [0, 1])
# au-delà duquel un pixel est considéré invalide (désocclusion / échec du
# flux optique) et exclu de l'agrégation Lai et al. 2018.
DEFAULT_PHOTO_THRESHOLD = 0.05

# Paramètres Farneback (mêmes valeurs que metrics/performance_metrics.py)
_FARNEBACK_PARAMS = dict(
    pyr_scale=0.5, levels=3, winsize=15, iterations=3,
    poly_n=5, poly_sigma=1.2, flags=0,
)


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
) -> float:
    """
    Flow Warping Error (Lai et al. 2018) via Farneback, pleine résolution.

    Pour chaque paire (t-1, t) :
      - Estime les flux forward F_{t-1 -> t} et backward F_{t -> t-1}
        au niveau Farneback.
      - Warpe le masque t-1 dans le référentiel de t via le flux backward.
      - Calcule l'erreur L1 avec le masque t, pondérée par un masque de
        validité (cohérence fwd/bwd + photométrique).
    L'erreur finale est une moyenne globale pondérée par le nombre de
    pixels valides sur toute la séquence — et non une moyenne par frame.

    Note : la boucle fusionne le calcul du flux et l'agrégation pour éviter
    de stocker tous les flux simultanément en mémoire.

    Args:
        masks:           Liste de masques (H, W) ou (H, W, 1) dans [0, 1],
                         ordonnés temporellement.
        frames:          Itérable de frames BGR (convention OpenCV), (H, W, 3)
                         ou (H, W, 4). Requis : sans frames l'erreur n'est
                         pas calculable de façon sensée -> 0.0 + warning.
        threshold:       Seuil de binarisation des masques (inclusif).
        photo_threshold: Erreur L1 max (RGB normalisé) pour qu'un pixel soit
                         considéré valide. Défaut 0.05.

    Returns:
        Erreur moyenne pondérée dans [0, 1]. Plus bas = meilleure stabilité.
    """
    if frames is None:
        logger.warning(
            "FWE : aucune frame RGB fournie — la métrique requiert les frames "
            "source pour la cohérence photométrique. Retour 0.0."
        )
        return 0.0

    if len(masks) < 2:
        return 0.0

    # --- Matérialiser les frames et les convertir BGR -> RGB float32 [0, 1] ---
    frame_list = list(frames)
    t_max = min(len(masks), len(frame_list))
    if t_max < 2:
        return 0.0

    video_list: List[np.ndarray] = []
    for bgr in frame_list[:t_max]:
        if bgr.ndim == 3 and bgr.shape[2] == 4:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        video_list.append(rgb)
    video = np.stack(video_list, axis=0).astype(np.float32) / 255.0  # (T, H, W, 3)
    t_max, h, w = video.shape[:3]

    # --- Binariser + aligner les masques sur (h, w) ---
    mask_stack: List[np.ndarray] = []
    for m in masks[:t_max]:
        m_sq = np.asarray(m).squeeze().astype(np.float32)
        if m_sq.max() > 1.0:
            m_sq /= 255.0
        m_bin = (m_sq >= threshold).astype(np.float32)
        if m_bin.shape != (h, w):
            m_bin = cv2.resize(m_bin, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        mask_stack.append(m_bin)
    masks_arr = np.stack(mask_stack, axis=0)  # (T, H, W)

    # --- Niveaux de gris pour Farneback ---
    gray = np.stack([
        cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        for f in video
    ])  # (T, H, W) uint8

    # --- Boucle principale : flux bidirectionnel + agrégation streaming ---
    total_err = 0.0
    total_valid = 0.0
    for t in range(1, t_max):
        flow_fwd = cv2.calcOpticalFlowFarneback(
            gray[t - 1], gray[t], None, **_FARNEBACK_PARAMS,
        )
        flow_bwd = cv2.calcOpticalFlowFarneback(
            gray[t], gray[t - 1], None, **_FARNEBACK_PARAMS,
        )

        validity = _lai_validity_mask(
            video[t - 1], video[t], flow_fwd, flow_bwd, photo_threshold,
        )
        mask_prev_warped = _warp_with_flow(masks_arr[t - 1], flow_bwd)
        err = np.abs(masks_arr[t] - mask_prev_warped)

        total_err += float((validity * err).sum())
        total_valid += float(validity.sum())

    if total_valid <= 0:
        return 0.0
    return float(total_err / total_valid)


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
