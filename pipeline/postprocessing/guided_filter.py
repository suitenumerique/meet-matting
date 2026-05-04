"""
Edge-preserving mask refinement via a fast guided filter (He et al., 2013).
Uses cv2.boxFilter (O(n), no opencv-contrib required).
The RGB frame acts as a structural guide so mask edges align with real image edges.
"""

import cv2
import numpy as np
from core.base import Postprocessor
from core.parameters import ParameterSpec
from core.registry import postprocessors


def _guided_filter(guide_rgb: np.ndarray, mask: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Fast guided filter using box-filter mean approximation.

    Args:
        guide_rgb: RGB image, shape (H, W, 3), dtype uint8.
        mask:      Alpha matte, shape (H, W), dtype float32, range [0, 1].
        radius:    Box filter half-size (kernel = 2*radius+1).
        eps:       Regularization term (prevents division by near-zero variance).

    Returns:
        Filtered mask, shape (H, W), dtype float32, range [0, 1].
    """
    ksize = (2 * radius + 1, 2 * radius + 1)

    # Grayscale guide normalized to [0, 1] -- float32 throughout to avoid repeated casts
    guide = cv2.cvtColor(guide_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) * (1.0 / 255.0)

    mean_I = cv2.boxFilter(guide, ddepth=-1, ksize=ksize)
    mean_p = cv2.boxFilter(mask, ddepth=-1, ksize=ksize)
    mean_Ip = cv2.boxFilter(guide * mask, ddepth=-1, ksize=ksize)
    mean_II = cv2.boxFilter(guide * guide, ddepth=-1, ksize=ksize)

    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, ddepth=-1, ksize=ksize)
    mean_b = cv2.boxFilter(b, ddepth=-1, ksize=ksize)

    result = mean_a * guide + mean_b
    return np.clip(result, 0.0, 1.0)


@postprocessors.register
class GuidedFilter(Postprocessor):
    name = "guided_filter"
    hidden = True  # disponible en upsampling — masqué dans post-process
    description = "Realigns mask edges to image edges using the RGB frame as a guide."

    @classmethod
    def parameter_specs(cls):
        return [
            ParameterSpec(
                name="radius",
                type="int",
                default=5,
                label="Radius",
                min_value=1,
                max_value=15,
                step=1,
                help="Spatial extent of the filter. Larger = smoother edges.",
            ),
            ParameterSpec(
                name="eps",
                type="float",
                default=1e-6,
                label="Regularization (eps)",
                min_value=1e-6,
                max_value=0.1,
                step=0.001,
                help="Edge sensitivity. Higher values reduce edge sharpness.",
            ),
        ]

    def reset(self):
        pass

    def __call__(self, mask: np.ndarray, original_frame: np.ndarray) -> np.ndarray:
        return _guided_filter(original_frame, mask, self.params["radius"], self.params["eps"])
