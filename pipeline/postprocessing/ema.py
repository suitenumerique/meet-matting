"""
Exponential Moving Average (EMA) post-processor.
Smooths mask transitions over time to reduce flickering.
"""

import numpy as np
from core.base import Postprocessor
from core.parameters import ParameterSpec
from core.registry import postprocessors


@postprocessors.register
class TemporalSmoothing(Postprocessor):
    name = "ema"
    description = "Smooths the mask over time using Exponential Moving Average."

    def __init__(self, **params):
        """Initialise with params and allocate internal buffers."""
        super().__init__(**params)
        self._prev_mask = None

    @classmethod
    def parameter_specs(cls):
        """Return the list of tunable parameters for this component."""
        return [
            ParameterSpec(
                name="alpha",
                type="float",
                default=0.15,
                label="Smoothing (Alpha)",
                min_value=0.01,
                max_value=1.0,
                step=0.01,
                help="Smoothing factor. Lower = smoother but more lag.",
            ),
        ]

    def reset(self):
        """Clear internal state; called by the pipeline between videos."""
        self._prev_mask = None

    def __call__(self, mask: np.ndarray, original_frame: np.ndarray) -> np.ndarray:
        """Apply EMA smoothing to *mask* using the stored previous frame mask.

        Args:
            mask:           Alpha matte, shape (H, W), dtype float32, range [0, 1].
            original_frame: Original RGB frame, shape (H, W, 3), dtype uint8 (unused).

        Returns:
            Smoothed alpha matte, shape (H, W), dtype float32, range [0, 1].
        """
        if self._prev_mask is None or self._prev_mask.shape != mask.shape:
            self._prev_mask = mask.copy()
            return mask

        alpha = self.params["alpha"]
        # raw_mask = alpha * current + (1-alpha) * prev
        smoothed = alpha * mask + (1.0 - alpha) * self._prev_mask
        self._prev_mask = smoothed.copy()

        return smoothed
