"""
Quality metric computation for Video Matting.

Implemented metrics:
  - IoU (Intersection over Union)
  - Boundary F-measure (F1 on contours)
  - Flow Warping Error (temporal stability)
"""

import logging
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial

# Default bound_ratio (DAVIS rule): the contour-matching tolerance is
# 0.8% of the image diagonal (~2 px at 480p, ~4 px at 1080p).
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
    Compute the global IoU over the whole video sequence.

    Intersection and union are cumulated over all pixels of every frame,
    then the ratio is computed once. This is a pixel-weighted IoU:
    frames where the object is larger contribute more to the final score.

    Conventions (aligned with metrics/performance_metrics.py):
      - Binarisation with `threshold` (default 0.5, inclusive threshold).
      - If the union is empty across the whole sequence -> return 1.0
        (perfect agreement by convention: both masks are empty everywhere).
      - If shapes differ between pred and GT on a frame, the predicted
        mask is resized to the GT shape (NEAREST interpolation).

    Args:
        pred_masks: List of predicted masks, each (H, W) or (H, W, 1),
                    values in [0, 1].
        gt_masks:   List of ground-truth masks, each (H, W), values in [0, 1].
        threshold:  Binarisation threshold.

    Returns:
        Global IoU in [0, 1].
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
    Extract a 1-pixel contour via morphological subtraction:
        boundary = mask - erode(mask)

    3x3 cross kernel (4-neighbourhood). Unlike Canny, this method also
    captures the edges of internal holes in the mask, penalising them
    in the F-measure — DAVIS-standard behaviour, relevant for matting
    (holes produce visible artifacts).
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
    Binary boundary F-measure of a frame (Perazzi et al. 2016, DAVIS).

    Method aligned with metrics/performance_metrics.py:
      - Contour extraction by morphological erosion (internal holes are
        penalised — DAVIS rule, desirable for matting).
      - Spatial tolerance scaled with the resolution:
            bound_radius = max(1, round(bound_ratio * sqrt(H^2 + W^2)))
        With bound_ratio=0.008 (DAVIS), this gives ~2 px at 480p, ~4 px
        at 1080p. A predicted pixel is "matched" if it falls within the
        GT contour dilated by a disk of radius bound_radius (morphological
        approximation of the bipartite matching of Martin et al. 2004).

    Args:
        pred:        Predicted mask (H, W) or (H, W, 1), values in [0, 1].
        gt:          Ground-truth mask (H, W), values in [0, 1].
        bound_ratio: Fraction of the diagonal used as tolerance radius.
                     Default 0.008 (DAVIS rule).
        threshold:   Binarisation threshold (inclusive).

    Returns:
        Contour F1 score in [0, 1].
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

    # Edge cases: no contour on either side
    if n_pred == 0 and n_gt == 0:
        return 1.0  # agreement: no object anywhere
    if n_pred == 0 or n_gt == 0:
        return 0.0  # only one contour exists -> no matching possible

    # Tolerance radius in pixels (DAVIS rule: % of the diagonal)
    h, w = gt_bin.shape[:2]
    diag = np.sqrt(h ** 2 + w ** 2)
    bound_radius = max(1, int(np.round(bound_ratio * diag)))
    ksize = 2 * bound_radius + 1
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

    # Morphological matching: a predicted pixel is matched if it falls
    # within the dilated GT contour (numerator of precision). Symmetric
    # for recall. The two counts differ when the contours have different
    # densities.
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
#  Flow Warping Error — Temporal stability
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Default L1 photometric-error threshold (on normalised RGB [0, 1])
# above which a pixel is considered invalid (disocclusion / optical-flow
# failure) and excluded from the aggregation (Lai et al. 2018).
DEFAULT_PHOTO_THRESHOLD = 0.05

# Farneback parameters (same values as metrics/performance_metrics.py)
_FARNEBACK_PARAMS = dict(
    pyr_scale=0.5, levels=3, winsize=15, iterations=3,
    poly_n=5, poly_sigma=1.2, flags=0,
)


def _warp_with_flow(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """
    Backward-warp `image` with `flow`:
        warped[y, x] = image[y + flow_y(y, x), x + flow_x(y, x)]

    Typical usage: image = frame_{t-1} and flow = F_{t -> t-1} (backward
    flow); `warped` is then an estimate of frame t built by pulling the
    pixels of frame_{t-1} along the flow. Out-of-image lookups are set to 0.
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
    Validity mask (H, W) in {0, 1}. A pixel is flagged valid iff:
      1) forward/backward flow consistency (Sundaram et al. 2010): going
         t -> t-1 via flow_bwd and then back via flow_fwd lands near the
         starting point, up to an adaptive threshold;
      2) photometric consistency (Lai et al. 2018): warping frame_{t-1}
         onto frame t reproduces the observed colour with an L1 error
         below `photo_threshold`.
    Pixels that fail either check typically correspond to disocclusions
    or flow failures and are excluded from the score.
    """
    # (1) Forward/backward consistency
    flow_fwd_at_t = _warp_with_flow(flow_fwd, flow_bwd)
    diff = flow_fwd_at_t + flow_bwd  # ≈ 0 if the two flows are consistent
    diff_sq = (diff ** 2).sum(axis=-1)
    mag_sq = (flow_fwd_at_t ** 2).sum(axis=-1) + (flow_bwd ** 2).sum(axis=-1)
    # Adaptive threshold (Sundaram): 1% of the squared magnitude, plus a
    # 0.5 floor so that near-static pixels are not wrongly rejected.
    fb_ok = diff_sq <= 0.01 * mag_sq + 0.5

    # (2) Photometric consistency
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
    Flow Warping Error (Lai et al. 2018) via Farneback, at full resolution.

    For each pair (t-1, t):
      - Estimate the forward flow F_{t-1 -> t} and backward flow
        F_{t -> t-1} at the Farneback level.
      - Warp the mask at t-1 into the frame t reference via the backward
        flow.
      - Compute the L1 error with the mask at t, weighted by a validity
        mask (fwd/bwd consistency + photometric).
    The final error is a global mean weighted by the number of valid
    pixels over the whole sequence — not a per-frame average.

    Note: the loop fuses flow computation and aggregation to avoid
    storing every flow in memory at once.

    Args:
        masks:           List of masks (H, W) or (H, W, 1) in [0, 1],
                         ordered temporally.
        frames:          Iterable of BGR frames (OpenCV convention),
                         (H, W, 3) or (H, W, 4). Required: without frames
                         the error cannot be sensibly computed -> 0.0 +
                         warning.
        threshold:       Mask binarisation threshold (inclusive).
        photo_threshold: Max L1 error (on normalised RGB) for a pixel to
                         be considered valid. Default 0.05.

    Returns:
        Weighted mean error in [0, 1]. Lower = better stability.
    """
    if frames is None:
        logger.warning(
            "FWE: no RGB frames provided — the metric requires the source "
            "frames for photometric consistency. Returning 0.0."
        )
        return 0.0

    if len(masks) < 2:
        return 0.0

    # --- Materialise frames and convert BGR -> RGB float32 [0, 1] ---
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

    # --- Binarise + align masks on (h, w) ---
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

    # --- Grayscale for Farneback ---
    gray = np.stack([
        cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        for f in video
    ])  # (T, H, W) uint8

    # --- Main loop: bidirectional flow + streaming aggregation ---
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
#  Per-video aggregation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _compute_single_frame_bf(pair):
    """Compute the Boundary F-measure for a single frame (multiprocessing helper)."""
    pred, gt = pair
    return compute_boundary_f_measure(pred, gt)


def compute_all_metrics(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    frames: Optional[Iterable[np.ndarray]] = None,
) -> dict:
    """
    Compute every quality metric for a video sequence.

    IoU: computed globally on the sequence (pixel-weighted, a single
    ratio for the whole video). `iou_std` is exposed as 0.0 — the metric
    no longer has per-frame dispersion. The key is kept for compatibility
    with downstream reports.
    """
    n = min(len(pred_masks), len(gt_masks))
    if n == 0:
        logger.error("No mask available for metric computation.")
        return {
            "iou_mean": 0.0,
            "iou_std": 0.0,
            "boundary_f_mean": 0.0,
            "boundary_f_std": 0.0,
            "flow_warping_error": 0.0,
        }

    # Global IoU over the whole sequence (single aggregated ratio)
    iou_global = compute_iou(pred_masks[:n], gt_masks[:n])

    # Boundary F per frame (parallelised — ThreadPool, OpenCV releases the GIL)
    pairs = list(zip(pred_masks[:n], gt_masks[:n]))
    with ThreadPoolExecutor(max_workers=8) as executor:
        boundary_fs = list(executor.map(_compute_single_frame_bf, pairs))

    # Flow warping error on the predictions
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
