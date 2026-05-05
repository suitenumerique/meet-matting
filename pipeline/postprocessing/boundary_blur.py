"""Boundary blur postprocessor — softens jagged mask edges with a Gaussian applied only to transition zones."""

import cv2
import numpy as np
from core.base import Postprocessor
from core.parameters import ParameterSpec
from core.registry import postprocessors


@postprocessors.register
class BoundaryBlur(Postprocessor):
    name = "boundary_blur"
    description = "Applies a weighted separable blur specifically to the mask boundaries."
    details = (
        "This filter identifies the transition areas (frontiers) of the mask and applies "
        "a Gaussian blur only to those regions. This helps soften jagged edges without "
        "affecting the overall mask structure or confidence in solid regions."
    )

    @classmethod
    def parameter_specs(cls) -> list[ParameterSpec]:
        """Return the list of tunable parameters for this component."""
        return [
            ParameterSpec(
                name="sigma",
                type="float",
                default=1.5,
                label="Blur Strength (Sigma)",
                min_value=0.1,
                max_value=5.0,
                step=0.1,
                help="Standard deviation of the Gaussian kernel. Higher = softer edges.",
            ),
            ParameterSpec(
                name="edge_width",
                type="int",
                default=3,
                label="Edge Detection Width",
                min_value=1,
                max_value=11,
                step=2,
                help="Size of the dilation used to define the 'boundary' zone.",
            ),
            ParameterSpec(
                name="mix_factor",
                type="float",
                default=1.0,
                label="Mix Factor",
                min_value=0.0,
                max_value=1.0,
                step=0.1,
                help="How much of the blurred boundary to mix back (1.0 = full blur).",
            ),
        ]

    def reset(self) -> None:
        """No temporal state to clear."""
        pass

    def __call__(self, mask: np.ndarray, original_frame: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur only to the mask boundary zone.

        Args:
            mask:           Alpha matte, shape (H, W), dtype float32, range [0, 1].
            original_frame: Original RGB frame, shape (H, W, 3), dtype uint8 (unused).

        Returns:
            Refined alpha matte with softened edges, shape (H, W), dtype float32.
        """
        if self.params["sigma"] <= 0.1:
            return mask

        # 1. Calculate the 'boundary' mask (where alpha is between 0 and 1)
        # Or more robustly: where there is a gradient
        # We use dilation - erosion to find the contour
        ksize = int(self.params["edge_width"])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

        # Working with uint8 for efficiency in morphological ops
        m_u8 = (mask * 255).astype(np.uint8)
        dilated = cv2.dilate(m_u8, kernel)
        eroded = cv2.erode(m_u8, kernel)
        boundary_zone = (dilated - eroded).astype(np.float32) / 255.0

        # 2. Apply Gaussian Blur (Separable) to the whole mask
        sigma = self.params["sigma"]
        # Kernel size is usually 4*sigma + 1
        k_blur = int(4 * sigma + 1) | 1
        blurred_mask = cv2.GaussianBlur(mask, (k_blur, k_blur), sigma)

        # 3. Weighted Mix: Result = Original + (Blurred - Original) * BoundaryZone * MixFactor
        # This ensures that only the boundary zone is affected.
        mix = self.params["mix_factor"]
        result = mask + (blurred_mask - mask) * boundary_zone * mix

        return np.clip(result, 0.0, 1.0)
