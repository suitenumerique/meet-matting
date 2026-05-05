"""Gaussian blur preprocessor — softens the frame to reduce high-frequency noise before inference."""

import cv2
from core.base import Preprocessor
from core.parameters import ParameterSpec
from core.registry import preprocessors


@preprocessors.register
class GaussianBlur(Preprocessor):
    name = "gaussian_blur"
    description = "Smooths the frame with a Gaussian kernel before inference."

    @classmethod
    def parameter_specs(cls):
        """Return the list of tunable parameters for this component."""
        return [
            ParameterSpec(
                name="kernel_size",
                type="int",
                default=5,
                label="Kernel size",
                min_value=1,
                max_value=31,
                step=2,
                help="Must be odd. Larger = more blur.",
            ),
            ParameterSpec(
                name="sigma",
                type="float",
                default=1.0,
                label="Sigma",
                min_value=0.1,
                max_value=10.0,
                step=0.1,
            ),
        ]

    def __call__(self, frame):
        """Apply Gaussian blur to *frame*.

        Args:
            frame: RGB image, shape (H, W, 3), dtype uint8.

        Returns:
            Blurred RGB image, shape (H, W, 3), dtype uint8.
        """
        k = self.params["kernel_size"]
        if k % 2 == 0:
            k += 1  # cv2 requires odd kernel size; be defensive.
        return cv2.GaussianBlur(frame, (k, k), self.params["sigma"])
