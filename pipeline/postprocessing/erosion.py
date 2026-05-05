"""Erosion postprocessor — shrinks the mask by a configurable radius to tighten edges."""

import cv2
import numpy as np
from core.base import Postprocessor
from core.parameters import ParameterSpec
from core.registry import postprocessors


@postprocessors.register
class Erosion(Postprocessor):
    name = "erosion"
    description = "Erodes the mask by a given number of pixels to tighten edges."

    @classmethod
    def parameter_specs(cls):
        """Return the list of tunable parameters for this component."""
        return [
            ParameterSpec(
                name="radius",
                type="int",
                default=2,
                label="Erosion radius (px)",
                min_value=1,
                max_value=20,
                step=1,
                help="Radius of the erosion kernel in pixels. Larger values shrink the mask more aggressively.",
            ),
        ]

    def reset(self):
        """No temporal state to clear."""
        pass

    def __call__(self, mask: np.ndarray, original_frame: np.ndarray) -> np.ndarray:
        """Erode *mask* by *radius* pixels to tighten foreground edges.

        Args:
            mask:           Alpha matte, shape (H, W), dtype float32, range [0, 1].
            original_frame: Original RGB frame, shape (H, W, 3), dtype uint8 (unused).

        Returns:
            Eroded alpha matte, shape (H, W), dtype float32, range [0, 1].
        """
        radius = int(self.params["radius"])
        size = 2 * radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        m_u8 = (mask * 255.0).astype(np.uint8)
        eroded = cv2.erode(m_u8, kernel)
        return eroded.astype(np.float32) * (1.0 / 255.0)
