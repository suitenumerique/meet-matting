"""
Morphological mask cleanup.
Closing (5x5) fills small internal holes; Opening (3x3) removes isolated noise.
"""

import cv2
import numpy as np
from core.base import Postprocessor
from core.parameters import ParameterSpec
from core.registry import postprocessors

# Module-level kernels -- allocated once, reused every frame
_KERNEL_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
_KERNEL_OPEN  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


@postprocessors.register
class MorphologyCleanup(Postprocessor):
    name = "morphology"
    description = "Fills small holes (Closing 5x5) then removes isolated noise (Opening 3x3)."

    @classmethod
    def parameter_specs(cls):
        return [
            ParameterSpec(
                name="enabled",
                type="bool",
                default=True,
                label="Enable morphological cleanup",
                help="Closing 5x5 fills gaps; Opening 3x3 removes isolated pixels.",
            ),
        ]

    def reset(self):
        pass

    def __call__(self, mask: np.ndarray, original_frame: np.ndarray) -> np.ndarray:
        if not self.params["enabled"]:
            return mask

        # float32 [0,1] -> uint8 [0,255] for OpenCV morphology ops
        m_u8 = (mask * 255.0).astype(np.uint8)
        m_u8 = cv2.morphologyEx(m_u8, cv2.MORPH_CLOSE, _KERNEL_CLOSE)
        m_u8 = cv2.morphologyEx(m_u8, cv2.MORPH_OPEN,  _KERNEL_OPEN)
        return m_u8.astype(np.float32) * (1.0 / 255.0)
