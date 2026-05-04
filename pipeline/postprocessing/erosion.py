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
        pass

    def __call__(self, mask: np.ndarray, original_frame: np.ndarray) -> np.ndarray:
        radius = int(self.params["radius"])
        size = 2 * radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        m_u8 = (mask * 255.0).astype(np.uint8)
        eroded = cv2.erode(m_u8, kernel)
        return eroded.astype(np.float32) * (1.0 / 255.0)
