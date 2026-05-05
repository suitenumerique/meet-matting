"""Threshold postprocessor — binarises the mask at a configurable cutoff value."""

import numpy as np
from core.base import Postprocessor
from core.parameters import ParameterSpec
from core.registry import postprocessors


@postprocessors.register
class Threshold(Postprocessor):
    name = "threshold"
    description = "Binarizes the mask at a chosen cutoff."

    @classmethod
    def parameter_specs(cls):
        """Return the list of tunable parameters for this component."""
        return [
            ParameterSpec(
                name="cutoff",
                type="float",
                default=0.5,
                label="Cutoff",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
            ),
        ]

    def __call__(self, mask, original_frame):
        """Binarize *mask* at ``cutoff``.

        Args:
            mask:           Alpha matte, shape (H, W), dtype float32, range [0, 1].
            original_frame: Original RGB frame, shape (H, W, 3), dtype uint8 (unused).

        Returns:
            Binary mask, shape (H, W), dtype float32, values in {0.0, 1.0}.
        """
        return (mask >= self.params["cutoff"]).astype(np.float32)
