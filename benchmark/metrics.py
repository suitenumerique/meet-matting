"""
Fonctions de calcul des métriques de qualité pour le Video Matting.

Métriques :
  - IoU (Intersection over Union)
  - Boundary F-measure (F1 sur les contours, méthode DAVIS)
  - Flow Warping Error (stabilité temporelle, Lai et al. 2018)
"""

import gc
import logging
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Fraction de la diagonale utilisée comme rayon de tolérance (règle DAVIS)
DEFAULT_BOUND_RATIO = 0.008
# Seuil L1 photométrique pour le masque de validité du flux optique
DEFAULT_PHOTO_THRESHOLD = 0.05
# Résolution max pour BF (pixels côté long) — 4× moins de surface à 540p vs 1080p
_BF_MAX_RES = 540
# Résolution max pour FWE (DIS optical flow)
_FWE_MAX_RES = 320

# DIS optical flow — singleton réutilisant les buffers internes d'OpenCV
_DIS_FLOW = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Utilitaires communs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _to_binary(mask: np.ndarray, threshold: float) -> np.ndarray:
    """Binarise un masque float32 [0,1] ou uint8 [0,255] en tableau bool.

    Utilise une comparaison dans le dtype natif pour éviter l'upcast en float64
    que NumPy ferait sinon (×20 plus lent sur de grands tableaux uint8).
    """
    m = np.asarray(mask).squeeze()
    if m.dtype == np.uint8:
        return m >= np.uint8(int(threshold * 255))
    return m >= threshold


def _resize_to_match(arr: np.ndarray, ref_shape: Tuple[int, int]) -> np.ndarray:
    """Redimensionne arr à ref_shape (H,W) si nécessaire (INTER_NEAREST)."""
    if arr.shape[:2] == ref_shape:
        return arr
    return cv2.resize(
        arr.astype(np.uint8), (ref_shape[1], ref_shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  IoU — Intersection over Union
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_iou(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    threshold: float = 0.5,
) -> float:
    """
    IoU global pixel-pondéré sur toute la séquence vidéo.

    Intersection et union sont cumulées sur l'ensemble des pixels de toutes
    les frames, puis le ratio est calculé une seule fois.

    Returns:
        IoU ∈ [0, 1]. Retourne 1.0 si l'union est vide (deux masques vides).
    """
    n = min(len(pred_masks), len(gt_masks))
    if n == 0:
        return 1.0

    total_inter = 0
    total_union = 0
    for pred, gt in zip(pred_masks[:n], gt_masks[:n]):
        p_bin = _to_binary(pred, threshold)
        g_bin = _to_binary(gt, threshold)
        if p_bin.shape != g_bin.shape:
            p_bin = _resize_to_match(p_bin.astype(np.uint8), g_bin.shape[:2]).astype(bool)
        total_inter += int(np.bitwise_and(p_bin, g_bin).sum())
        total_union += int(np.bitwise_or(p_bin, g_bin).sum())

    return float(total_inter / total_union) if total_union > 0 else 1.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Boundary F-measure (méthode DAVIS)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _extract_boundary(mask_u8: np.ndarray) -> np.ndarray:
    """Contour 1-pixel : mask - erode(mask). Noyau croix 3×3 (4-voisinage)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    return mask_u8 - cv2.erode(mask_u8, kernel, iterations=1)


def _make_disk(bound_radius: int) -> np.ndarray:
    """Élément structurant elliptique de rayon bound_radius."""
    k = 2 * bound_radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _bf_disk_for_shape(h: int, w: int, bound_ratio: float = DEFAULT_BOUND_RATIO) -> np.ndarray:
    """Calcule le disk adapté à une résolution (H, W)."""
    diag = np.sqrt(h ** 2 + w ** 2)
    return _make_disk(max(1, int(np.round(bound_ratio * diag))))


def compute_boundary_f_measure(
    pred: np.ndarray,
    gt: np.ndarray,
    bound_ratio: float = DEFAULT_BOUND_RATIO,
    threshold: float = 0.5,
    _disk: Optional[np.ndarray] = None,
) -> float:
    """
    Boundary F-measure binaire d'une frame (Perazzi et al. 2016, DAVIS).

    Travaille à _BF_MAX_RES (540p) pour la vitesse — le bound_radius reste
    proportionnellement correct car il est basé sur la diagonale.

    Args:
        pred:    Masque prédit (H, W) ou (H, W, 1), valeurs ∈ [0, 1].
        gt:      Masque ground truth (H, W), valeurs ∈ [0, 1].
        _disk:   Élément structurant pré-calculé (optionnel, pour la vitesse).

    Returns:
        Score F1 des contours ∈ [0, 1].
    """
    p_bin = _to_binary(pred, threshold).astype(np.uint8)
    g_bin = _to_binary(gt, threshold).astype(np.uint8)
    if p_bin.shape != g_bin.shape:
        p_bin = _resize_to_match(p_bin, g_bin.shape[:2])

    # Downscale pour vitesse (4× moins de surface à 540p vs 1080p)
    h, w = g_bin.shape[:2]
    if max(h, w) > _BF_MAX_RES:
        scale = _BF_MAX_RES / max(h, w)
        nw, nh = int(w * scale), int(h * scale)
        p_bin = cv2.resize(p_bin, (nw, nh), interpolation=cv2.INTER_NEAREST)
        g_bin = cv2.resize(g_bin, (nw, nh), interpolation=cv2.INTER_NEAREST)
        h, w = nh, nw
        _disk = None  # recalculer pour la nouvelle résolution

    p_bound = _extract_boundary(p_bin)
    g_bound = _extract_boundary(g_bin)

    n_pred = int(p_bound.sum())
    n_gt   = int(g_bound.sum())
    if n_pred == 0 and n_gt == 0:
        return 1.0
    if n_pred == 0 or n_gt == 0:
        return 0.0

    if _disk is None:
        _disk = _bf_disk_for_shape(h, w, bound_ratio)

    p_dil = cv2.dilate(p_bound, _disk)
    g_dil = cv2.dilate(g_bound, _disk)

    precision = int(np.logical_and(p_bound, g_dil).sum()) / n_pred
    recall    = int(np.logical_and(g_bound, p_dil).sum()) / n_gt

    if precision + recall == 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Flow Warping Error (Lai et al. 2018)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _warp(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Backward-warp rapide en pré-calculant une grille fixe."""
    h, w = flow.shape[:2]
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = gx + flow[..., 0]
    map_y = gy + flow[..., 1]
    return cv2.remap(image, map_x, map_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _lai_validity_mask(
    frame_prev: np.ndarray,
    frame_curr: np.ndarray,
    flow_fwd: np.ndarray,
    flow_bwd: np.ndarray,
    photo_threshold: float,
) -> np.ndarray:
    """Masque de validité {0,1} : cohérence forward/backward + photométrique."""
    fwd_at_t = _warp(flow_fwd, flow_bwd)
    diff_sq = ((fwd_at_t + flow_bwd) ** 2).sum(axis=-1)
    mag_sq  = ((fwd_at_t ** 2).sum(axis=-1) + (flow_bwd ** 2).sum(axis=-1))
    fb_ok = diff_sq <= 0.01 * mag_sq + 0.5

    photo_err = np.abs(frame_curr - _warp(frame_prev, flow_bwd)).mean(axis=-1)
    return (fb_ok & (photo_err <= photo_threshold)).astype(np.float32)


def compute_flow_warping_error(
    masks: List[np.ndarray],
    frames: Optional[Iterable[np.ndarray]] = None,
    threshold: float = 0.5,
    photo_threshold: float = DEFAULT_PHOTO_THRESHOLD,
    max_res: int = _FWE_MAX_RES,
    frame_step: int = 2,
) -> float:
    """
    Flow Warping Error (Lai et al. 2018) via DIS optical flow.

    Args:
        max_res:    Résolution maximale (côté long) pour le calcul du flux.
                    Défaut 320 (4× moins de pixels qu'à 1280p).
        frame_step: Sous-échantillonnage temporel : 1 = toutes les frames,
                    2 = 1 sur 2, etc. Réduit le temps de 1/frame_step.

    Returns:
        FWE ∈ [0, 1], ou 0.0 si < 2 paires de frames valides.
    """
    if frames is None or len(masks) < 2:
        return 0.0

    total_err   = 0.0
    total_valid = 0.0
    prev_gray = prev_rgb = prev_mask = None
    dis = _DIS_FLOW

    for i, frame in enumerate(frames):
        if i >= len(masks):
            break

        # Sous-échantillonnage temporel
        if i % frame_step != 0:
            continue

        # Downscale pour vitesse
        h, w = frame.shape[:2]
        if max(h, w) > max_res:
            scale = max_res / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        th, tw = frame.shape[:2]

        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        m = np.asarray(masks[i]).squeeze()
        t = threshold * 255.0 if m.dtype == np.uint8 else threshold
        m_bin = (m >= t).astype(np.float32)
        if m_bin.shape != (th, tw):
            m_bin = cv2.resize(m_bin, (tw, th), interpolation=cv2.INTER_NEAREST)

        if prev_gray is not None:
            union = (m_bin + prev_mask) > 0
            ys, xs = np.where(union)
            if len(ys) > 0:
                pad = 16
                y1, y2 = max(0, ys.min() - pad), min(th, ys.max() + pad)
                x1, x2 = max(0, xs.min() - pad), min(tw, xs.max() + pad)

                rp = prev_gray[y1:y2, x1:x2].copy()
                rc = curr_gray[y1:y2, x1:x2].copy()
                fwd = dis.calc(rp, rc, None)
                bwd = dis.calc(rc, rp, None)

                validity = _lai_validity_mask(
                    prev_rgb[y1:y2, x1:x2], curr_rgb[y1:y2, x1:x2],
                    fwd, bwd, photo_threshold,
                )
                warped = _warp(prev_mask[y1:y2, x1:x2], bwd)
                err = np.abs(m_bin[y1:y2, x1:x2] - warped)
                total_err   += float((validity * err).sum())
                total_valid += float(validity.sum())

        prev_gray = curr_gray
        prev_rgb  = curr_rgb
        prev_mask = m_bin

    gc.collect()
    return float(total_err / total_valid) if total_valid > 0 else 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Agrégation par vidéo
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _compute_single_frame_bf(args: Tuple) -> float:
    """Helper picklable pour ThreadPoolExecutor."""
    pred, gt, disk = args
    return compute_boundary_f_measure(pred, gt, _disk=disk)


def compute_all_metrics(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    frames: Optional[Iterable[np.ndarray]] = None,
    threshold: float = 0.5,
) -> dict:
    """
    Calcule IoU + Boundary F + Flow Warping Error sur une séquence vidéo.

    Optimisations :
      - Disk kernel BF pré-calculé une fois à la résolution downsamplée.
      - BF calculé en parallèle sur 8 threads (OpenCV libère le GIL).
      - FWE avec sous-échantillonnage temporel (frame_step=2) et max_res=320.
    """
    n = min(len(pred_masks), len(gt_masks))
    if n == 0:
        logger.error("Aucun masque disponible.")
        return {"iou_mean": 0.0, "iou_std": 0.0,
                "boundary_f_mean": 0.0, "boundary_f_std": 0.0,
                "flow_warping_error": 0.0}

    # Disk BF pré-calculé à la résolution effective (après downsample à 540p)
    ref = np.asarray(gt_masks[0]).squeeze()
    h, w = ref.shape[:2]
    if max(h, w) > _BF_MAX_RES:
        scale = _BF_MAX_RES / max(h, w)
        h, w = int(h * scale), int(w * scale)
    disk = _bf_disk_for_shape(h, w)

    iou_global = compute_iou(pred_masks[:n], gt_masks[:n], threshold=threshold)

    # BF en parallèle (ThreadPool — OpenCV libère le GIL pour les ops morpho)
    n_workers = min(8, n)
    args = [(p, g, disk) for p, g in zip(pred_masks[:n], gt_masks[:n])]
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        boundary_fs = list(executor.map(_compute_single_frame_bf, args))

    fwe = compute_flow_warping_error(pred_masks[:n], frames, threshold=threshold)

    return {
        "iou_mean": iou_global,
        "iou_std": 0.0,
        "boundary_f_mean": float(np.mean(boundary_fs)),
        "boundary_f_std": float(np.std(boundary_fs)),
        "flow_warping_error": fwe,
    }
