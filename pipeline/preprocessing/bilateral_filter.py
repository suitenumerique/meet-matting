import cv2
import numpy as np
from core.base import Preprocessor
from core.parameters import ParameterSpec
from core.registry import preprocessors


@preprocessors.register
class BilateralFilter(Preprocessor):
    name = "bilateral_filter"
    description = "Smooths the frame while preserving sharp edges. Great for noise reduction."

    @classmethod
    def parameter_specs(cls):
        return [
            ParameterSpec(
                name="diameter",
                type="int",
                default=9,
                label="Diameter",
                min_value=1,
                max_value=25,
                step=1,
                help="Diameter of each pixel neighborhood. Larger = slower but cleaner.",
            ),
            ParameterSpec(
                name="sigma_color",
                type="float",
                default=75.0,
                label="Sigma Color",
                min_value=1.0,
                max_value=200.0,
                step=5.0,
                help="Higher value means pixels with larger color difference will be blurred together.",
            ),
            ParameterSpec(
                name="sigma_space",
                type="float",
                default=75.0,
                label="Sigma Space",
                min_value=1.0,
                max_value=200.0,
                step=5.0,
                help="Higher value means pixels farther away will influence each other.",
            ),
        ]

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """Apply Bilateral Filter to *frame*.

        Args:
            frame: RGB image, shape (H, W, 3), dtype uint8.

        Returns:
            Edge-preserved smoothed RGB image.
        """
        d = self.params["diameter"]
        sc = self.params["sigma_color"]
        ss = self.params["sigma_space"]
        
        # cv2.bilateralFilter works on RGB images
        return cv2.bilateralFilter(frame, d, sc, ss)
