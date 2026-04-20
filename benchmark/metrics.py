"""
Fonctions de calcul des métriques de qualité pour le Video Matting.

Métriques implémentées :
  - IoU (Intersection over Union)
  - Boundary F-measure (F1 sur les contours)
  - Flow Warping Error (stabilité temporelle)
"""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import BOUNDARY_TOLERANCE_PX

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  IoU — Intersection over Union
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Calcule l'IoU entre un masque prédit et un masque ground truth.

    Les deux masques doivent être binaires (0/1) ou seront binarisés
    avec un seuil de 0.5.

    Args:
        pred: Masque prédit (H, W), valeurs dans [0, 1].
        gt:   Masque ground truth (H, W), valeurs dans [0, 1].

    Returns:
        Score IoU dans [0, 1]. Retourne 0.0 si l'union est vide.
    """
    pred_bin = (pred > 0.5).astype(np.uint8)
    gt_bin = (gt > 0.5).astype(np.uint8)

    # Resize automatique si dimensions différentes
    if pred_bin.shape != gt_bin.shape:
        logger.debug(
            "IoU: resize pred %s -> gt %s", pred_bin.shape, gt_bin.shape
        )
        pred_bin = cv2.resize(
            pred_bin, (gt_bin.shape[1], gt_bin.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Boundary F-measure
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _extract_boundary(mask: np.ndarray) -> np.ndarray:
    """Extrait les pixels de contour d'un masque binaire via Canny."""
    mask_u8 = (mask * 255).astype(np.uint8)
    edges = cv2.Canny(mask_u8, 50, 150)
    return (edges > 0).astype(np.uint8)


def _boundary_precision_recall(
    pred_boundary: np.ndarray,
    gt_boundary: np.ndarray,
    tolerance_px: int = BOUNDARY_TOLERANCE_PX,
) -> Tuple[float, float]:
    """
    Calcule Precision et Recall entre contours pred et GT
    avec une tolérance spatiale (dilatation morphologique).
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * tolerance_px + 1, 2 * tolerance_px + 1)
    )

    # Dilater GT pour calculer la précision (pred hits dans zone GT)
    gt_dilated = cv2.dilate(gt_boundary, kernel)
    # Dilater pred pour calculer le recall (GT hits dans zone pred)
    pred_dilated = cv2.dilate(pred_boundary, kernel)

    pred_count = pred_boundary.sum()
    gt_count = gt_boundary.sum()

    if pred_count == 0 and gt_count == 0:
        return 1.0, 1.0
    if pred_count == 0:
        return 0.0, 0.0
    if gt_count == 0:
        return 0.0, 0.0

    precision = float(np.logical_and(pred_boundary, gt_dilated).sum() / pred_count)
    recall = float(np.logical_and(gt_boundary, pred_dilated).sum() / gt_count)

    return precision, recall


def compute_boundary_f_measure(
    pred: np.ndarray,
    gt: np.ndarray,
    tolerance_px: int = BOUNDARY_TOLERANCE_PX,
) -> float:
    """
    Calcule le Boundary F-measure (F1 sur les contours).

    Mesure la qualité de la segmentation sur les bords — métrique critique
    pour le matting vidéo où la qualité des cheveux/doigts est essentielle.

    Args:
        pred: Masque prédit (H, W), valeurs dans [0, 1].
        gt:   Masque ground truth (H, W), valeurs dans [0, 1].
        tolerance_px: Tolérance en pixels pour le matching de contours.

    Returns:
        Score F1 des contours dans [0, 1].
    """
    pred_bin = (pred > 0.5).astype(np.uint8)
    gt_bin = (gt > 0.5).astype(np.uint8)

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

    precision, recall = _boundary_precision_recall(
        pred_boundary, gt_boundary, tolerance_px
    )

    if precision + recall == 0:
        return 0.0

    f_measure = 2.0 * precision * recall / (precision + recall)
    return float(f_measure)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Flow Warping Error — Stabilité temporelle
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_flow_warping_error(
    masks: List[np.ndarray],
    frames: Optional[List[np.ndarray]] = None,
) -> float:
    """
    Calcule le Flow Warping Error moyen sur une séquence de masques.

    Mesure la cohérence temporelle : warp le masque t vers t+1 via le flux
    optique, puis mesure l'erreur entre le masque warpé et le masque réel t+1.

    Args:
        masks: Liste de masques (H, W) en float [0, 1], ordonnés temporellement.
        frames: Liste de frames RGB (H, W, 3) correspondantes pour le calcul
                du flux optique. Si None, utilise les masques eux-mêmes.

    Returns:
        Erreur moyenne de warping dans [0, 1]. Plus bas = meilleure stabilité.
    """
    if len(masks) < 2:
        return 0.0

    errors = []

    for i in range(len(masks) - 1):
        mask_curr = masks[i].astype(np.float32)
        mask_next = masks[i + 1].astype(np.float32)

        # Resize si nécessaire
        if mask_curr.shape != mask_next.shape:
            mask_curr = cv2.resize(
                mask_curr, (mask_next.shape[1], mask_next.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        # Frames pour le flux optique
        if frames is not None and len(frames) > i + 1:
            frame_curr = frames[i]
            frame_next = frames[i + 1]
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

        # Calcul du flux optique dense (Farneback) sur résolution réduite
        flow = cv2.calcOpticalFlowFarneback(
            g_curr_small, g_next_small,
            None,
            pyr_scale=0.5,
            levels=2,        # Réduit de 3 à 2 pour plus de vitesse
            winsize=11,      # Réduit de 15 à 11
            iterations=2,    # Réduit de 3 à 2
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

    return float(np.mean(errors))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Agrégation par vidéo
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_all_metrics(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    frames: Optional[List[np.ndarray]] = None,
) -> dict:
    """
    Calcule toutes les métriques de qualité sur une séquence vidéo.

    Args:
        pred_masks: Liste des masques prédits (H, W), float [0, 1].
        gt_masks:   Liste des masques GT (H, W), float [0, 1].
        frames:     Liste de frames RGB optionnelles pour le flux optique.

    Returns:
        Dict avec les clés :
          - iou_mean, iou_std
          - boundary_f_mean, boundary_f_std
          - flow_warping_error
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

    ious = []
    boundary_fs = []

    for i in range(n):
        ious.append(compute_iou(pred_masks[i], gt_masks[i]))
        boundary_fs.append(compute_boundary_f_measure(pred_masks[i], gt_masks[i]))

    # Flow warping error sur les prédictions
    fwe = compute_flow_warping_error(
        pred_masks[:n],
        frames[:n] if frames else None,
    )

    return {
        "iou_mean": float(np.mean(ious)),
        "iou_std": float(np.std(ious)),
        "boundary_f_mean": float(np.mean(boundary_fs)),
        "boundary_f_std": float(np.std(boundary_fs)),
        "flow_warping_error": fwe,
    }
