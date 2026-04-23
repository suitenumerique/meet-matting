"""
Performance metrics for the video segmentation evaluation pipeline.

All metric functions take mask videos of shape (T, H, W) or (T, H, W, C)
and return a scalar (optionally also a per-frame vector for finer analysis).
"""

import numpy as np
import cv2

# =============================================================================
# Global IoU (Intersection over Union)
# =============================================================================

def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5) -> float:
    """
    Compute the global IoU between the predicted and ground-truth masks,
    aggregated over all pixels of the video.

    Args:
        pred_mask: Predicted mask, shape (T, H, W) or (T, H, W, C).
        gt_mask:   Ground-truth mask, same shape as pred_mask.
        threshold: Binarization threshold in [0, 1]. Default: 0.5.

    Returns:
        IoU in [0, 1].
    """
    if pred_mask.shape != gt_mask.shape:
        raise ValueError(
            f"Masks must have the same shape. "
            f"Got pred={pred_mask.shape}, gt={gt_mask.shape}"
        )

    # If the .mp4 was decoded as RGB, collapse the channel dim; the three
    # channels of a mask are identical in practice.
    if pred_mask.ndim == 4:
        pred_mask = pred_mask.mean(axis=-1)
    if gt_mask.ndim == 4:
        gt_mask = gt_mask.mean(axis=-1)

    # Cast to float and normalize from [0, 255] to [0, 1] if needed.
    pred_mask = pred_mask.astype(np.float32)
    gt_mask = gt_mask.astype(np.float32)
    if pred_mask.max() > 1.0:
        pred_mask /= 255.0
    if gt_mask.max() > 1.0:
        gt_mask /= 255.0

    # Binarize
    pred_bin = pred_mask >= threshold
    gt_bin = gt_mask >= threshold

    # IoU = |A ∩ B| / |A ∪ B|
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()

    if union == 0:
        # Both masks are fully empty -> perfect agreement by convention.
        return 1.0

    return float(intersection / union)


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
):
    """
    Binary boundary F-measure (Perazzi et al. 2016, DAVIS), averaged per frame.

    Note: this implementation treats internal holes as boundaries and
    therefore penalizes them. This is the DAVIS-standard behavior and is
    appropriate for the video-conferencing background-blur use case.

    Args:
        pred_mask:        Predicted mask, shape (T, H, W) or (T, H, W, C).
        gt_mask:          Ground-truth mask, same shape as pred_mask.
        threshold:        Binarization threshold in [0, 1]. Default: 0.5.
        bound_ratio:      Tolerance expressed as a fraction of the image
                          diagonal. DAVIS default: 0.008 (~2 px at 480p).
        return_per_frame: If True, also returns the per-frame F-measure
                          vector (useful for plots / temporal analyses).

    Returns:
        float, or (float, np.ndarray) if return_per_frame=True.
    """

    if pred_mask.shape != gt_mask.shape:
        raise ValueError(
            f"Masks must have the same shape. "
            f"Got pred={pred_mask.shape}, gt={gt_mask.shape}"
        )

    # Collapse channel dim if masks were decoded as RGB.
    if pred_mask.ndim == 4:
        pred_mask = pred_mask.mean(axis=-1)
    if gt_mask.ndim == 4:
        gt_mask = gt_mask.mean(axis=-1)

    # Normalize from [0, 255] to [0, 1] if needed.
    pred_mask = pred_mask.astype(np.float32)
    gt_mask = gt_mask.astype(np.float32)
    if pred_mask.max() > 1.0:
        pred_mask /= 255.0
    if gt_mask.max() > 1.0:
        gt_mask /= 255.0

    # Binarize
    pred_bin = pred_mask >= threshold
    gt_bin = gt_mask >= threshold

    # Tolerance radius in pixels (DAVIS rule).
    T, H, W = pred_bin.shape
    diag = np.sqrt(H ** 2 + W ** 2)
    bound_radius = max(1, int(np.round(bound_ratio * diag)))

    per_frame = np.array(
        [_f_measure_frame(pred_bin[t], gt_bin[t], bound_radius) for t in range(T)],
        dtype=np.float32,
    )
    mean_f = float(per_frame.mean())

    if return_per_frame:
        return mean_f, per_frame
    return mean_f


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