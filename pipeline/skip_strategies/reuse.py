"""Reuse skip strategy — returns the last inferred mask unchanged for every skipped frame."""

import numpy as np
from core.base import SkipStrategy
from core.registry import skip_strategies


@skip_strategies.register
class Reuse(SkipStrategy):
    name = "reuse"
    description = "Reuses the last inferred mask as-is for skipped frames."

    @classmethod
    def parameter_specs(cls):
        """Return the list of tunable parameters for this component."""
        return []

    def __call__(
        self,
        current_frame: np.ndarray,
        prev_frame: np.ndarray,
        prev_mask: np.ndarray,
    ) -> np.ndarray:
        """Return *prev_mask* unchanged.

        Args:
            current_frame: RGB frame to fill (unused), shape (H, W, 3), dtype uint8.
            prev_frame:    Last inferred frame (unused), shape (H, W, 3), dtype uint8.
            prev_mask:     Mask from the last inferred frame, shape (H, W), float32, [0, 1].

        Returns:
            *prev_mask* unmodified.
        """
        return prev_mask
