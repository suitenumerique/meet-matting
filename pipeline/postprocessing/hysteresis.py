"""
Hysteresis thresholding for stable binary mask output.

A pixel activates only when it exceeds T_high and deactivates only when it
drops below T_low.  This eliminates the class of flickering where a pixel's
probability oscillates around a single fixed threshold.

Reference: Canny (1986), "A Computational Approach to Edge Detection",
IEEE TPAMI 8(6):679-698.
"""

import numpy as np
from core.base import Postprocessor
from core.parameters import ParameterSpec
from core.registry import postprocessors


@postprocessors.register
class HysteresisThreshold(Postprocessor):
    name = "hysteresis"
    description = "Stable binarization: activates at T_high, deactivates below T_low. Eliminates boundary flicker."

    def __init__(self, **params):
        """Initialise with params and allocate internal buffers."""
        super().__init__(**params)
        self._prev_mask: np.ndarray | None = None

    @classmethod
    def parameter_specs(cls):
        """Return the list of tunable parameters for this component."""
        return [
            ParameterSpec(
                name="t_high",
                type="float",
                default=0.65,
                label="High threshold",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                help="Pixels above this value switch to foreground.",
            ),
            ParameterSpec(
                name="t_low",
                type="float",
                default=0.35,
                label="Low threshold",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                help="Active pixels remain foreground until they drop below this value.",
            ),
        ]

    def reset(self):
        """Clear internal state; called by the pipeline between videos."""
        self._prev_mask = None

    def __call__(self, mask: np.ndarray, original_frame: np.ndarray) -> np.ndarray:
        """Apply hysteresis thresholding to *mask* using stored pixel states.

        Args:
            mask:           Alpha matte, shape (H, W), dtype float32, range [0, 1].
            original_frame: Original RGB frame, shape (H, W, 3), dtype uint8 (unused).

        Returns:
            Binary alpha matte, shape (H, W), dtype float32, values in {0.0, 1.0}.
        """
        t_high = self.params["t_high"]
        t_low = self.params["t_low"]

        above_high = mask >= t_high
        above_low = mask >= t_low

        if self._prev_mask is None or self._prev_mask.shape != mask.shape:
            # Cold start: use the strict high threshold only.
            result = above_high
        else:
            prev_active = self._prev_mask > 0.5
            # A pixel is active if:
            #   - it just crossed T_high (newly activated), or
            #   - it was already active AND is still above T_low (hysteresis hold).
            result = above_high | (above_low & prev_active)

        out = result.astype(np.float32)
        self._prev_mask = out
        return out
