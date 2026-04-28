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
        super().__init__(**params)
        self._prev_mask = None

    @classmethod
    def parameter_specs(cls):
        return [
            ParameterSpec(
                name="alpha",
                type="float",
                default=0.4,
                label="Smoothing (Alpha)",
                min_value=0.05,
                max_value=1.0,
                step=0.05,
                help="Smoothing factor. Lower = smoother but more lag.",
            ),
        ]

    def __call__(self, mask: np.ndarray, original_frame: np.ndarray) -> np.ndarray:
        if self._prev_mask is None or self._prev_mask.shape != mask.shape:
            self._prev_mask = mask.copy()
            return mask

        alpha = self.params["alpha"]
        # raw_mask = alpha * current + (1-alpha) * prev
        smoothed = alpha * mask + (1.0 - alpha) * self._prev_mask
        self._prev_mask = smoothed.copy()
        
        return smoothed
