import cv2
import numpy as np
from core.base import Preprocessor
from core.parameters import ParameterSpec
from core.registry import preprocessors


@preprocessors.register
class CLAHE(Preprocessor):
    name = "clahe"
    description = "Contrast Limited Adaptive Histogram Equalization to improve detail in shadows."

    @classmethod
    def parameter_specs(cls):
        return [
            ParameterSpec(
                name="clip_limit",
                type="float",
                default=2.0,
                label="Clip Limit",
                min_value=0.1,
                max_value=10.0,
                step=0.1,
                help="Threshold for contrast limiting. Higher = more contrast.",
            ),
            ParameterSpec(
                name="tile_grid_size",
                type="int",
                default=8,
                label="Tile Grid Size",
                min_value=2,
                max_value=32,
                step=1,
                help="Size of grid for histogram equalization. Larger = more global.",
            ),
        ]

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE to the luminance channel of the RGB *frame*.

        Args:
            frame: RGB image, shape (H, W, 3), dtype uint8.

        Returns:
            Contrast-enhanced RGB image.
        """
        # Convert RGB to LAB to apply CLAHE only to the L channel (luminance)
        # This avoids changing the color balance.
        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        grid_size = self.params["tile_grid_size"]
        clahe = cv2.createCLAHE(
            clipLimit=self.params["clip_limit"], 
            tileGridSize=(grid_size, grid_size)
        )
        
        cl = clahe.apply(l)

        # Merge the CLAHE enhanced L-channel back with A and B channels
        limg = cv2.merge((cl, a, b))

        # Convert back to RGB
        return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
