"""
Temporal mask warping via dense optical flow (DIS).

Reference: Kroeger, Timofte, Dai, Van Gool (ECCV 2016).
"Fast Optical Flow using Dense Inverse Search."

Motivation: EMA blending without motion compensation causes "ghosting" -- the
previous mask bleeds into regions where the person has moved away, leaving
semi-transparent artefacts. This postprocessor uses the optical flow field to
warp the previous mask to the current camera position before blending.

Algorithm
---------
1. Compute the dense forward optical flow F from frame t-1 to frame t using
   cv2.DISOpticalFlow (ULTRAFAST preset, CPU-only, ~5 ms on 1080p).
   F[y, x] = (dx, dy): pixel (x, y) in frame t-1 displaced to (x+dx, y+dy).

2. Backward warp approximation via cv2.remap:
   For each pixel (x', y') in the output (current-frame coordinates), sample
   the previous mask at its approximate source position:
       map_x[y'][x'] = x' - dx[y'][x']
       map_y[y'][x'] = y' - dy[y'][x']
   (exact backward flow would require inverting F; this first-order approximation
   is standard and accurate for small-to-moderate inter-frame motion.)

3. Blend warped mask with current mask:
       out = alpha * mask_current + (1 - alpha) * mask_warped

   Adaptive blend (optional, inspired by 1 Euro Filter speed-based logic):
       mean_flow = mean ||F[y,x]|| over all pixels
       alpha = clip(alpha_min + gamma * mean_flow, alpha_min, 1.0)
   When the scene is fast-moving, alpha rises toward 1.0 (trust current mask
   fully). When the scene is nearly static, alpha stays near alpha_min
   (warp contributes more, smoothing temporal noise).
"""

from __future__ import annotations

import cv2
import numpy as np
from core.base import Postprocessor
from core.parameters import ParameterSpec
from core.registry import postprocessors


@postprocessors.register
class OpticalFlowWarp(Postprocessor):
    name = "optical_flow_warp"
    hidden = True  # disponible en skip strategy — masqué dans post-process
    description = "Warp the previous mask with DIS optical flow before blending to remove ghosting."
    details = (
        "Reference: Kroeger et al. (ECCV 2016) -- DIS optical flow.\n"
        "Algorithm:\n"
        "  1. Compute forward flow F (frame t-1 -> frame t) with DIS ULTRAFAST.\n"
        "  2. Warp previous mask to current frame via backward-map approximation:\n"
        "       map_x[x',y'] = x' - F_x[x',y']\n"
        "       map_y[x',y'] = y' - F_y[x',y']\n"
        "  3. Blend: out = alpha * mask_current + (1-alpha) * mask_warped\n"
        "Adaptive blend (inspired by 1 Euro Filter):\n"
        "  alpha rises with mean flow magnitude so fast motion trusts the\n"
        "  current mask fully and slow/static scenes get more temporal smoothing.\n"
        "Combine upstream with EMA or One Euro for best results."
    )

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self._dis = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        self._prev_gray: np.ndarray | None = None
        self._prev_mask: np.ndarray | None = None
        # Pre-allocated coordinate grids; rebuilt when frame size changes.
        self._x_grid: np.ndarray | None = None
        self._y_grid: np.ndarray | None = None
        self._grid_shape: tuple[int, int] | None = None

    @classmethod
    def parameter_specs(cls) -> list[ParameterSpec]:
        return [
            ParameterSpec(
                name="alpha_min",
                type="float",
                default=0.15,
                label="Min blend (alpha_min)",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                help=(
                    "Blend weight of the current mask when scene is static. "
                    "1 - alpha_min is the contribution of the warped previous mask."
                ),
            ),
            ParameterSpec(
                name="adaptive",
                type="bool",
                default=True,
                label="Adaptive blend",
                help=(
                    "When enabled, alpha rises with mean optical flow magnitude "
                    "(fast motion -> trust current mask fully). "
                    "Disable to use a fixed alpha_min blend."
                ),
            ),
            ParameterSpec(
                name="gamma",
                type="float",
                default=0.05,
                label="Flow sensitivity (gamma)",
                min_value=0.0,
                max_value=1.0,
                step=0.005,
                help=(
                    "Rate at which alpha increases per unit of mean flow magnitude "
                    "(pixels/frame). Effective only when Adaptive blend is enabled. "
                    "alpha = clip(alpha_min + gamma * mean_flow, alpha_min, 1.0)"
                ),
            ),
        ]

    def reset(self) -> None:
        self._prev_gray = None
        self._prev_mask = None

    def _ensure_grids(self, h: int, w: int) -> None:
        """Allocate or reuse coordinate grids for cv2.remap."""
        if self._grid_shape == (h, w):
            return
        self._y_grid, self._x_grid = np.mgrid[0:h, 0:w].astype(np.float32)
        self._grid_shape = (h, w)

    def __call__(self, mask: np.ndarray, original_frame: np.ndarray) -> np.ndarray:
        alpha_min = float(self.params["alpha_min"])
        adaptive = bool(self.params["adaptive"])
        gamma = float(self.params["gamma"])

        h, w = mask.shape

        cur_gray = cv2.cvtColor(original_frame, cv2.COLOR_RGB2GRAY)

        # Cold start: no previous frame yet.
        if self._prev_gray is None or self._prev_mask is None:
            self._prev_gray = cur_gray
            self._prev_mask = mask.copy()
            return mask

        if self._prev_gray.shape != cur_gray.shape:
            # Frame size changed (e.g., zoom crop); reset and skip this frame.
            self._prev_gray = cur_gray
            self._prev_mask = mask.copy()
            return mask

        # --- Step 1: dense forward optical flow (prev -> cur) ---------------
        flow = self._dis.calc(
            self._prev_gray,
            cur_gray,
            np.zeros((*cur_gray.shape, 2), dtype=np.float32),
        )
        # flow shape: (H, W, 2), dtype float32
        # flow[y, x] = (dx, dy): pixel (x,y) in prev moved to (x+dx, y+dy) in cur.

        # --- Step 2: backward-warp approximation ----------------------------
        self._ensure_grids(h, w)
        assert self._x_grid is not None
        assert self._y_grid is not None
        # For cur pixel (x', y'), its source in prev is approximately (x'-dx, y'-dy).
        map_x = self._x_grid - flow[:, :, 0]
        map_y = self._y_grid - flow[:, :, 1]

        warped = cv2.remap(
            self._prev_mask,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # --- Step 3: blend --------------------------------------------------
        if adaptive:
            # Mean Euclidean flow magnitude across the frame.
            flow_mag = np.mean(np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2))
            alpha = float(np.clip(alpha_min + gamma * flow_mag, alpha_min, 1.0))
        else:
            alpha = alpha_min

        out = alpha * mask + (1.0 - alpha) * warped
        out = np.clip(out, 0.0, 1.0).astype(np.float32)

        # Store for next frame.
        self._prev_gray = cur_gray
        self._prev_mask = out

        return out
