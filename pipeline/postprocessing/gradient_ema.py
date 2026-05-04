"""
Gradient-based Temporal EMA (IIR smoothing driven by spatial gradient).

Algorithm
---------
At each frame t, the output mask M_t is computed as a pixel-wise weighted blend:

    M_t = W * M_pred + (1 - W) * M_{t-1}

where W is a (H, W) float32 matrix derived from the spatial gradient of the
current RGB frame -- NOT from motion or optical flow.

Weight map construction
-----------------------
1. Convert the current frame to grayscale.
2. Compute the Sobel gradient magnitude:
       G_x = Sobel(gray, dx=1, dy=0)
       G_y = Sobel(gray, dx=0, dy=1)
       mag  = sqrt(G_x^2 + G_y^2)
3. Normalise mag to [w_min, w_max] with a per-frame min-max rescaling:
       W = w_min + (w_max - w_min) * (mag - mag.min()) / (mag.max() - mag.min() + eps)

Rationale
---------
- Strong gradient pixel (physical edge: clothing on a wall, hair on background):
  W -> w_max  =>  M_t closely tracks the model prediction.
  The model prediction is trusted so that edges snap into place instantly.

- Flat-zone pixel (homogeneous region: bare torso, plain background):
  W -> w_min  =>  M_t largely keeps the previous mask M_{t-1}.
  The history damps flickering caused by per-frame probability noise in flat areas.

All operations are vectorised over (H, W) with NumPy and OpenCV.
No Python loops over pixels.
"""

from __future__ import annotations

import cv2
import numpy as np
from core.base import Postprocessor
from core.parameters import ParameterSpec
from core.registry import postprocessors

_EPS = 1e-6


@postprocessors.register
class GradientEMA(Postprocessor):
    name = "gradient_ema"
    description = (
        "Gradient-based IIR EMA: high blend weight on physical edges, "
        "strong temporal smoothing on flat zones."
    )
    details = (
        "Algorithm: M_t = W * M_pred + (1-W) * M_{t-1}\n"
        "W is a per-pixel matrix derived from the Sobel gradient magnitude of\n"
        "the current frame, normalised to [w_min, w_max].\n"
        "Edge pixels (high gradient) get W -> w_max: mask snaps to the\n"
        "model prediction so moving edges are tracked sharply.\n"
        "Flat pixels (low gradient) get W -> w_min: mask keeps the previous\n"
        "frame value to suppress flickering in uniform regions.\n"
        "All computations are vectorised -- no per-pixel Python loops."
    )

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self._prev_mask: np.ndarray | None = None

    @classmethod
    def parameter_specs(cls) -> list[ParameterSpec]:
        return [
            ParameterSpec(
                name="w_min",
                type="float",
                default=0.05,
                label="w_min (flat zones)",
                min_value=0.0,
                max_value=0.50,
                step=0.01,
                help=(
                    "Blend weight applied to flat, uniform regions (low gradient). "
                    "Lower = stronger temporal smoothing, less flicker on homogeneous areas."
                ),
            ),
            ParameterSpec(
                name="w_max",
                type="float",
                default=0.85,
                label="w_max (edges)",
                min_value=0.50,
                max_value=1.00,
                step=0.01,
                help=(
                    "Blend weight applied to sharp physical edges (high gradient). "
                    "Higher = mask snaps faster to the current model prediction on edges."
                ),
            ),
        ]

    def reset(self) -> None:
        self._prev_mask = None

    # ------------------------------------------------------------------
    @staticmethod
    def _weight_map(frame_rgb: np.ndarray, w_min: float, w_max: float) -> np.ndarray:
        """Compute the per-pixel blend weight matrix W from the spatial gradient.

        Parameters
        ----------
        frame_rgb : np.ndarray, shape (H, W, 3), dtype uint8
            Current RGB frame (un-preprocessed original).
        w_min, w_max : float
            Target range for the normalised weight map.

        Returns
        -------
        W : np.ndarray, shape (H, W), dtype float32, values in [w_min, w_max]
        """
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

        # Sobel derivatives (ksize=3, float32 output -- handles negative values correctly)
        g_x = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
        g_y = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)

        # Gradient magnitude (always non-negative)
        mag = np.sqrt(g_x * g_x + g_y * g_y)

        # Per-frame min-max normalisation into [w_min, w_max]
        mag_min = mag.min()
        mag_max = mag.max()
        span = mag_max - mag_min

        if span < _EPS:
            # Flat frame: no gradient information -- use w_min everywhere
            return np.full(mag.shape, w_min, dtype=np.float32)

        # Vectorised rescaling: strong gradient -> high W, flat zone -> low W
        W = w_min + (w_max - w_min) * (mag - mag_min) / span
        return W.astype(np.float32)

    # ------------------------------------------------------------------
    def __call__(self, mask: np.ndarray, original_frame: np.ndarray) -> np.ndarray:
        """Apply gradient-based EMA to the current mask.

        Parameters
        ----------
        mask : np.ndarray, shape (H, W), dtype float32, range [0, 1]
            Raw (or previously post-processed) alpha matte for the current frame.
        original_frame : np.ndarray, shape (H, W, 3), dtype uint8, RGB
            Un-preprocessed source frame used to compute the gradient weight map.

        Returns
        -------
        M_t : np.ndarray, shape (H, W), dtype float32, range [0, 1]
        """
        w_min = float(self.params["w_min"])
        w_max = float(self.params["w_max"])

        # Cold start: no history yet -- return the current mask as-is.
        if self._prev_mask is None or self._prev_mask.shape != mask.shape:
            self._prev_mask = mask.copy()
            return mask

        # Build the spatial-gradient weight matrix (fully vectorised, no Python loops)
        W = self._weight_map(original_frame, w_min, w_max)

        # Pixel-wise IIR blend: M_t = W * M_pred + (1 - W) * M_{t-1}
        M_t = W * mask + (1.0 - W) * self._prev_mask

        self._prev_mask = M_t  # keep reference without copy (W * mask already new array)

        return np.clip(M_t, 0.0, 1.0).astype(np.float32)
