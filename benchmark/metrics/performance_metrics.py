"""
Performance metrics for the video segmentation evaluation pipeline.

All metric functions take mask videos of shape (T, H, W) or (T, H, W, C)
and return a scalar (optionally also a per-frame vector for finer analysis).
"""

import numpy as np
import cv2
import gc
from typing import List, Tuple, Union, Optional, Iterable
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# =============================================================================
# Global IoU (Intersection over Union)
# =============================================================================

def compute_iou(
    pred_masks: Union[List[np.ndarray], np.ndarray],
    gt_masks: Union[List[np.ndarray], np.ndarray],
    threshold: float = 0.5,
    apply_auto_align: bool = True
) -> float:
    """
    Compute the global IoU between predicted and ground-truth masks.
    Supports Lists, Generates, or Arrays. Handles automatic resizing and optional alignment.
    """
    # Convert to numpy arrays if they are lists or iterators
    if not isinstance(pred_masks, np.ndarray): pred_masks = np.array(list(pred_masks))
    if not isinstance(gt_masks, np.ndarray): gt_masks = np.array(list(gt_masks))
    
    if pred_masks.ndim == 2: # Single frame (H, W) -> add time dim
        pred_masks = pred_masks[np.newaxis, ...]
    if gt_masks.ndim == 2:
        gt_masks = gt_masks[np.newaxis, ...]

    if pred_masks.shape[0] != gt_masks.shape[0]:
        n = min(pred_masks.shape[0], gt_masks.shape[0])
        pred_masks = pred_masks[:n]
        gt_masks = gt_masks[:n]

    # Pre-determined alignment if requested
    dx, dy, flip = 0, 0, False
    if apply_auto_align and pred_masks.ndim >= 3:
        # Align on the middle frame
        mid = pred_masks.shape[0] // 2
        p_ref = pred_masks[mid]
        g_ref = gt_masks[mid]
        
        # Prepare for alignment helper
        p_ref_bin = (p_ref >= (threshold * 255 if p_ref.dtype == np.uint8 else threshold)).astype(np.uint8)
        g_ref_bin = (g_ref >= (threshold * 255 if g_ref.dtype == np.uint8 else threshold)).astype(np.uint8)
        
        if p_ref_bin.shape != g_ref_bin.shape:
             p_ref_bin = cv2.resize(p_ref_bin, (g_ref_bin.shape[1], g_ref_bin.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        dx, dy, flip, _ = _get_alignment_params(p_ref_bin, g_ref_bin)

    total_inter = 0
    total_union = 0
    
    for t in range(pred_masks.shape[0]):
        p = pred_masks[t]
        g = gt_masks[t]
        
        if p.ndim == 3: p = p.mean(axis=-1)
        if g.ndim == 3: g = g.mean(axis=-1)
        
        p_t = threshold * 255 if p.dtype == np.uint8 else threshold
        g_t = threshold * 255 if g.dtype == np.uint8 else threshold
        
        p_bin = (p >= p_t).astype(np.uint8)
        g_bin = (g >= g_t).astype(np.uint8)
        
        if p_bin.shape != g_bin.shape:
            p_bin = cv2.resize(p_bin, (g_bin.shape[1], g_bin.shape[0]), interpolation=cv2.INTER_NEAREST)
            
        if apply_auto_align:
            p_bin = _apply_alignment(p_bin, dx, dy, flip)
            
        total_inter += np.logical_and(p_bin, g_bin).sum()
        total_union += np.logical_or(p_bin, g_bin).sum()

    if total_union == 0:
        return 1.0

    return float(total_inter / total_union)


# =============================================================================
# Boundary F-measure (Perazzi et al. 2016, DAVIS)
# =============================================================================

def _extract_boundary(mask: np.ndarray) -> np.ndarray:
    """
    Extract a 1-pixel-wide boundary from a binary mask via morphological
    subtraction: boundary = mask - erode(mask).

    Note: this yields *all* boundaries, including the borders of internal
    holes. Holes in the predicted mask are therefore penalized by the
    F-measure, which is the DAVIS-standard behavior and desirable for our
    use case (holes produce visible artifacts in background blur).
    """
    mask_u8 = mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    eroded = cv2.erode(mask_u8, kernel, iterations=1)
    return mask_u8 - eroded


def _f_measure_frame(pred: np.ndarray, gt: np.ndarray, bound_radius: int) -> float:
    """Boundary F-measure for a single binary frame."""
    pred_boundary = _extract_boundary(pred)
    gt_boundary = _extract_boundary(gt)

    n_pred = int(pred_boundary.sum())
    n_gt = int(gt_boundary.sum())

    # Edge cases
    if n_pred == 0 and n_gt == 0:
        return 1.0  # Both agree: no object present.
    if n_pred == 0 or n_gt == 0:
        return 0.0  # Only one contour exists -> no possible matching.

    # Tolerance disk used as the dilation structuring element.
    ksize = 2 * bound_radius + 1
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    pred_dil = cv2.dilate(pred_boundary, disk)
    gt_dil = cv2.dilate(gt_boundary, disk)

    # Matching (morphological approximation of the bipartite matching from
    # Martin et al. 2004): a predicted pixel is "matched" if it lies within
    # the dilated GT contour (numerator of precision); a GT pixel is matched
    # if it lies within the dilated predicted contour (numerator of recall).
    # The two counts are distinct because the two boundaries may have
    # different thicknesses / densities.
    matched_pred = np.logical_and(pred_boundary, gt_dil).sum()
    matched_gt = np.logical_and(gt_boundary, pred_dil).sum()

    precision = matched_pred / n_pred
    recall = matched_gt / n_gt

    if precision + recall == 0:
        return 0.0
    
    return float(2 * precision * recall / (precision + recall))


def compute_boundary_f_measure(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    threshold: float = 0.5,
    bound_ratio: float = 0.008,
    return_per_frame: bool = False,
    num_workers: int = 1,
):
    """
    Binary boundary F-measure (Perazzi et al. 2016, DAVIS), averaged per frame.
    Supports parallelization via num_workers.
    """
    # Convert to numpy arrays if they are lists or iterators
    if not isinstance(pred_mask, np.ndarray): pred_mask = np.array(list(pred_mask))
    if not isinstance(gt_mask, np.ndarray): gt_mask = np.array(list(gt_mask))

    # Collapse channel dim if present
    if pred_mask.ndim == 4 and pred_mask.shape[-1] == 1: pred_mask = pred_mask.squeeze(-1)
    if gt_mask.ndim == 4 and gt_mask.shape[-1] == 1: gt_mask = gt_mask.squeeze(-1)
    if pred_mask.ndim == 4: pred_mask = pred_mask.mean(axis=-1)
    if gt_mask.ndim == 4: gt_mask = gt_mask.mean(axis=-1)

    if pred_mask.shape != gt_mask.shape:
        # Si mismatch de résolution, on redimensionne la prédiction sur le GT
        T, H_gt, W_gt = gt_mask.shape[:3]
        if pred_mask.shape[0] != T:
            # Tronquage si nombre de frames différent
            min_t = min(pred_mask.shape[0], T)
            pred_mask = pred_mask[:min_t]
            gt_mask = gt_mask[:min_t]
            T = min_t
        
        # Redimensionnement (plus efficace si fait globalement si possible, sinon frame par frame)
        # Pour éviter l'explosion RAM sur de très grosses vidéos, on pourrait le faire dans la boucle,
        # mais ici on reste simple pour la boucle parallèle.
        if pred_mask.shape[1:3] != (H_gt, W_gt):
            new_preds = np.zeros((T, H_gt, W_gt), dtype=pred_mask.dtype)
            for t in range(T):
                new_preds[t] = cv2.resize(pred_mask[t], (W_gt, H_gt), interpolation=cv2.INTER_NEAREST)
            pred_mask = new_preds

    # Normalize/Threshold/Binarize
    # Ensure we have a 3D batch (T, H, W)
    if pred_mask.ndim == 2: # (H, W) -> (1, H, W)
        pred_mask = pred_mask[np.newaxis, ...]
        gt_mask = gt_mask[np.newaxis, ...]
    elif pred_mask.ndim == 3 and pred_mask.shape[-1] <= 4 and pred_mask.shape[0] > 10:
        # Cas ambigu: (H, W, C) ou (T, H, W) ? 
        # Si T > 10, c'est probablement (T, H, W). Sinon on traite comme (H, W, C).
        pass 
    elif pred_mask.ndim == 3 and pred_mask.shape[-1] <= 4:
        # Probablement une seule frame RGB(A) : (H, W, C) -> (1, H, W)
        pred_mask = pred_mask.mean(axis=-1)[np.newaxis, ...]
        gt_mask = gt_mask.mean(axis=-1)[np.newaxis, ...]

    T = pred_mask.shape[0]
    H, W = pred_mask.shape[1:3]
    diag = np.sqrt(H ** 2 + W ** 2)
    bound_radius = max(1, int(np.round(bound_ratio * diag)))
    
    # Pre-processing for thresholding to avoid repeated branches
    p_thresh = threshold * 255 if pred_mask.dtype == np.uint8 else threshold
    g_thresh = threshold * 255 if gt_mask.dtype == np.uint8 else threshold
    
    p_bin = (pred_mask >= p_thresh).astype(np.uint8)
    g_bin = (gt_mask >= g_thresh).astype(np.uint8)

    if num_workers > 1:
        func = partial(_f_measure_frame, bound_radius=bound_radius)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            per_frame = np.array(list(executor.map(lambda t: func(p_bin[t], g_bin[t]), range(T))), dtype=np.float32)
    else:
        per_frame = np.array([_f_measure_frame(p_bin[t], g_bin[t], bound_radius) for t in range(T)], dtype=np.float32)
    
    mean_f = float(per_frame.mean())
    return (mean_f, per_frame) if return_per_frame else mean_f


# =============================================================================
# Flow Warping Error (Lai et al. 2018)
# =============================================================================

# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------

def _prepare_video(video: np.ndarray) -> np.ndarray:
    """(T, H, W, 3) RGB -> float32 in [0, 1]."""
    video = video.astype(np.float32)
    if video.max() > 1.0:
        video /= 255.0
    return video


def _prepare_mask(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """(T, H, W) or (T, H, W, C) -> float32 binary {0, 1} of shape (T, H, W)."""
    if mask.ndim == 4:
        mask = mask.mean(axis=-1)
    mask = mask.astype(np.float32)
    if mask.max() > 1.0:
        mask /= 255.0
    return (mask >= threshold).astype(np.float32)


def _warp(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """
    Backward-warp `image` using `flow`:
        warped[y, x] = image[y + flow_y(y, x), x + flow_x(y, x)]

    Typical usage: with image = frame_{t-1} and flow = F_{t -> t-1}
    (the backward flow from frame t to frame t-1), `warped` is an estimate
    of frame t built by pulling pixels from frame t-1 along the flow.
    Out-of-image lookups are set to 0.
    """
    H, W = image.shape[:2]
    grid_y, grid_x = np.mgrid[0:H, 0:W].astype(np.float32)
    map_x = grid_x + flow[..., 0]
    map_y = grid_y + flow[..., 1]
    return cv2.remap(
        image, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _validity_mask(
    frame_prev: np.ndarray, frame_curr: np.ndarray,
    flow_fwd: np.ndarray, flow_bwd: np.ndarray,
    photo_threshold: float,
) -> np.ndarray:
    """
    Per-pixel validity mask (H, W) in {0, 1}. A pixel is flagged valid iff:
      1) forward/backward flow consistency (Sundaram et al. 2010): going
         from t to t-1 via flow_bwd and back via flow_fwd returns (near)
         the starting location;
      2) photometric consistency (Lai et al. 2018): warping frame t-1 onto
         frame t reproduces the observed color with small L1 error.
    Pixels failing either check typically correspond to disocclusions or
    to flow-estimation failures and are excluded from the stability score.
    """
    # (1) Forward/backward flow consistency.
    # At each pixel (y, x) in frame t we want to compare flow_bwd(y, x) with
    # -flow_fwd(source), where source = (y, x) + flow_bwd(y, x).
    # `flow_fwd_at_t` samples flow_fwd at that source position.
    flow_fwd_at_t = _warp(flow_fwd, flow_bwd)
    diff = flow_fwd_at_t + flow_bwd  # ~ 0 if the two flows are consistent
    diff_sq = (diff ** 2).sum(axis=-1)
    mag_sq = (flow_fwd_at_t ** 2).sum(axis=-1) + (flow_bwd ** 2).sum(axis=-1)
    # Adaptive threshold from Sundaram et al.: tolerate up to 1% of the
    # squared magnitude, plus a 0.5 floor so that near-static pixels (where
    # the ratio would otherwise be overly strict) are not wrongly rejected.
    fb_ok = diff_sq <= 0.01 * mag_sq + 0.5

    # (2) Photometric consistency.
    frame_prev_warped = _warp(frame_prev, flow_bwd)
    photo_err = np.abs(frame_curr - frame_prev_warped).mean(axis=-1)
    photo_ok = photo_err <= photo_threshold

    return (fb_ok & photo_ok).astype(np.float32)


def _aggregate_warping_error(
    video: np.ndarray, masks: np.ndarray,
    flows_fwd: np.ndarray, flows_bwd: np.ndarray,
    photo_threshold: float, return_per_frame: bool,
):
    """
    L1 mask-warping error averaged over all valid pixels of the video.

    `mean_err` is the pixel-weighted global average (the standard Lai et al.
    definition and the metric of record). `per_frame` holds the frame-level
    average for each pair (t-1, t) and is useful for diagnostics only:
    frames with very few valid pixels can dominate a naive frame-wise mean.
    """
    T = video.shape[0]
    per_frame = np.zeros(T - 1, dtype=np.float32)
    total_err, total_valid = 0.0, 0.0
    for t in range(1, T):
        V = _validity_mask(
            video[t - 1], video[t],
            flows_fwd[t - 1], flows_bwd[t - 1],
            photo_threshold,
        )
        # Pull mask_{t-1} into frame t's coordinate system.
        mask_prev_warped = _warp(masks[t - 1], flows_bwd[t - 1])
        err = np.abs(masks[t] - mask_prev_warped)
        n_valid = V.sum()
        per_frame[t - 1] = (V * err).sum() / n_valid if n_valid > 0 else 0.0
        total_err += (V * err).sum()
        total_valid += n_valid
    mean_err = float(total_err / total_valid) if total_valid > 0 else 0.0
    return (mean_err, per_frame) if return_per_frame else mean_err


# -----------------------------------------------------------------------------
# Variant 1: Farneback (fast, CPU)
# -----------------------------------------------------------------------------

def compute_flow_warping_error_farneback(
    video: np.ndarray,
    pred_mask: np.ndarray,
    threshold: float = 0.5,
    photo_threshold: float = 0.05,
    return_per_frame: bool = False,
):
    """
    Flow warping error using OpenCV's Farneback optical flow. Fast, CPU-only,
    approximate. Intended for rapid model ranking during iteration.

    Note on bias: Farneback produces over-smoothed flow around motion
    boundaries, which may slightly penalize models with sharper, more
    accurate mask contours. The forward/backward consistency check
    mitigates this, but for final numbers prefer the RAFT
    variant below.

    Args:
        video:            (T, H, W, 3), uint8 or float, RGB.
        pred_mask:        (T, H, W) or (T, H, W, C).
        threshold:        Mask binarization threshold.
        photo_threshold:  Max photometric error (on [0, 1] RGB) for a
                          pixel to be considered valid. Default: 0.05.
        return_per_frame: If True, also return a (T-1,) vector of per-pair
                          errors.

    Returns:
        float (mean error); lower = more temporally stable.
    """
    video_f = _prepare_video(video)
    masks = _prepare_mask(pred_mask, threshold)
    T, H, W = masks.shape
    if T < 2:
        return (0.0, np.zeros(0, dtype=np.float32)) if return_per_frame else 0.0

    # Farneback requires grayscale uint8 inputs.
    gray = np.stack([
        cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        for f in video_f
    ])
    flows_fwd = np.zeros((T - 1, H, W, 2), dtype=np.float32)
    flows_bwd = np.zeros((T - 1, H, W, 2), dtype=np.float32)
    params = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                  poly_n=5, poly_sigma=1.2, flags=0)
    for t in range(1, T):
        # Forward flow F_{t-1 -> t}.
        flows_fwd[t - 1] = cv2.calcOpticalFlowFarneback(gray[t - 1], gray[t], None, **params)
        # Backward flow F_{t -> t-1}.
        flows_bwd[t - 1] = cv2.calcOpticalFlowFarneback(gray[t], gray[t - 1], None, **params)

    return _aggregate_warping_error(video_f, masks, flows_fwd, flows_bwd,
                                    photo_threshold, return_per_frame)


# =============================================================================
# Alignment Helpers (Auto-Recalage)
# =============================================================================

def _get_alignment_params(pred_bin: np.ndarray, gt_bin: np.ndarray, max_shift: int = 50) -> Tuple[int, int, bool, float]:
    """
    Calcule les paramètres de recalage optimal (Translation + Flip) sur une frame.
    """
    h_gt, w_gt = gt_bin.shape[:2]
    low_res = 128
    scale = low_res / max(h_gt, w_gt)
    g_low = cv2.resize(gt_bin.astype(np.uint8), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    def _match(p_img):
        pad = int(max_shift * scale) + 1
        g_padded = cv2.copyMakeBorder(g_low, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        res = cv2.matchTemplate(g_padded.astype(np.float32), p_img.astype(np.float32), cv2.TM_CCORR_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        return max_val, max_loc, pad

    p_low_normal = cv2.resize(pred_bin.astype(np.uint8), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    score_n, loc_n, pad = _match(p_low_normal)
    
    p_low_f = cv2.flip(p_low_normal, 1)
    score_f, loc_f, _ = _match(p_low_f)
    
    flip = score_f > score_n * 1.1
    best_score = max(score_n, score_f)
    best_loc = loc_f if flip else loc_n
    
    dx = int((best_loc[0] - pad) / scale)
    dy = int((best_loc[1] - pad) / scale)
    
    return dx, dy, flip, best_score

def _apply_alignment(mask: np.ndarray, dx: int, dy: int, flip: bool) -> np.ndarray:
    """Applique les paramètres de recalage."""
    h, w = mask.shape[:2]
    if flip:
        mask = cv2.flip(mask, 1)
    if dx == 0 and dy == 0:
        return mask
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# =============================================================================
# Variant 3: DIS Flow (Ultra-fast, CPU)
# =============================================================================

_DIS_FLOW = None

def compute_flow_warping_error_dis(
    video: np.ndarray,
    pred_mask: np.ndarray,
    threshold: float = 0.5,
    photo_threshold: float = 0.05,
    max_res: int = 480,
    return_per_frame: bool = False,
):
    """
    Flow warping error via DISOpticalFlow (OpenCV).
    Très rapide et optimisé pour le streaming.
    """
    global _DIS_FLOW
    if _DIS_FLOW is None:
        _DIS_FLOW = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    
    # Ensure numpy arrays
    if not isinstance(video, np.ndarray): video = np.array(list(video))
    if not isinstance(pred_mask, np.ndarray): pred_mask = np.array(list(pred_mask))

    video_f = _prepare_video(video)
    masks = _prepare_mask(pred_mask, threshold)
    T, H, W = masks.shape
    if T < 2:
        return (0.0, np.zeros(0, dtype=np.float32)) if return_per_frame else 0.0

    # Downsampling for flow speed if needed
    if max(H, W) > max_res:
        scale = max_res / max(H, W)
        th, tw = int(H * scale), int(W * scale)
    else:
        th, tw = H, W

    per_frame = np.zeros(T - 1, dtype=np.float32)
    total_err, total_valid = 0.0, 0.0
    
    prev_gray = None
    prev_frame_f = None
    prev_mask = None

    for t in range(T):
        curr_frame_f = video_f[t]
        if max(H, W) > max_res:
            curr_frame_f = cv2.resize(curr_frame_f, (tw, th), interpolation=cv2.INTER_AREA)
            curr_mask = cv2.resize(masks[t], (tw, th), interpolation=cv2.INTER_NEAREST)
        else:
            curr_mask = masks[t]
            
        curr_gray = cv2.cvtColor((curr_frame_f * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        if prev_gray is not None:
            fwd = _DIS_FLOW.calc(prev_gray, curr_gray, None)
            bwd = _DIS_FLOW.calc(curr_gray, prev_gray, None)
            
            V = _validity_mask(prev_frame_f, curr_frame_f, fwd, bwd, photo_threshold)
            mask_prev_warped = _warp(prev_mask, bwd)
            err = np.abs(curr_mask - mask_prev_warped)
            
            n_valid = V.sum()
            err_val = (V * err).sum()
            
            if t-1 < len(per_frame):
                per_frame[t - 1] = err_val / n_valid if n_valid > 0 else 0.0
            
            total_err += err_val
            total_valid += n_valid

        prev_gray = curr_gray
        prev_frame_f = curr_frame_f
        prev_mask = curr_mask

    gc.collect()
    mean_err = float(total_err / total_valid) if total_valid > 0 else 0.0
    return (mean_err, per_frame) if return_per_frame else mean_err


# -----------------------------------------------------------------------------
# Variant 2: RAFT (accurate, GPU recommended)
# -----------------------------------------------------------------------------

def compute_flow_warping_error_raft(
    video: np.ndarray,
    pred_mask: np.ndarray,
    threshold: float = 0.5,
    photo_threshold: float = 0.05,
    device: str = None,
    batch_size: int = 4,
    return_per_frame: bool = False,
):
    """
    Flow warping error using RAFT (torchvision, pretrained). Slower than
    Farneback but substantially more accurate at motion boundaries.
    Intended for final numbers.

    First call will download the pretrained weights (~200 MB, internet
    required).

    Additional args:
        device:      'cuda' or 'cpu'. Auto-detected if None.
        batch_size:  Number of frame pairs processed in parallel on GPU.
                     Reduce if you hit OOM on high-resolution video.
    """
    import torch
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    video_f = _prepare_video(video)
    masks = _prepare_mask(pred_mask, threshold)
    T, H, W = masks.shape
    if T < 2:
        return (0.0, np.zeros(0, dtype=np.float32)) if return_per_frame else 0.0

    weights = Raft_Large_Weights.DEFAULT
    transforms = weights.transforms()
    model = raft_large(weights=weights, progress=False).to(device).eval()

    # RAFT requires spatial dims divisible by 8. We pad (rather than resize)
    # so we don't have to rescale flow vectors afterwards; padding is then
    # cropped off the output flow.
    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8
    video_t = torch.from_numpy(video_f).permute(0, 3, 1, 2)  # (T, 3, H, W)
    if pad_h or pad_w:
        video_t = torch.nn.functional.pad(
            video_t, (0, pad_w, 0, pad_h), mode="replicate"
        )

    flows_fwd = np.zeros((T - 1, H, W, 2), dtype=np.float32)
    flows_bwd = np.zeros((T - 1, H, W, 2), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, T - 1, batch_size):
            end = min(start + batch_size, T - 1)
            img1 = video_t[start:end].to(device)
            img2 = video_t[start + 1:end + 1].to(device)
            img1_n, img2_n = transforms(img1, img2)
            # RAFT returns a list of refined flow predictions across
            # iterations; the last element is the highest-quality one.
            fwd = model(img1_n, img2_n)[-1]  # (B, 2, H_pad, W_pad)
            bwd = model(img2_n, img1_n)[-1]
            # Crop padding off and move back to (H, W, 2) numpy layout.
            fwd = fwd[:, :, :H, :W].permute(0, 2, 3, 1).cpu().numpy()
            bwd = bwd[:, :, :H, :W].permute(0, 2, 3, 1).cpu().numpy()
            flows_fwd[start:end] = fwd
            flows_bwd[start:end] = bwd

    return _aggregate_warping_error(video_f, masks, flows_fwd, flows_bwd,
                                    photo_threshold, return_per_frame)