import cv2
import numpy as np
from core.base import Postprocessor
from core.parameters import ParameterSpec
from core.registry import postprocessors


@postprocessors.register
class Morphology(Postprocessor):
    name = "morphology"
    description = "Applies morphological operations (Opening/Closing) to clean the mask."

    @classmethod
    def parameter_specs(cls):
        return [
            ParameterSpec(
                name="kernel_size",
                type="int",
                default=3,
                label="Kernel Size",
                min_value=0,
                max_value=15,
                step=1,
                help="Radius of the morphological kernel. 0 means disabled.",
            ),
            ParameterSpec(
                name="mode",
                type="choice",
                default="close_open",
                label="Mode",
                choices=["close", "open", "close_open"],
                help="Close: fills holes. Open: removes noise. Close-Open: does both.",
            ),
        ]

    def __call__(self, mask, original_frame):
        """Clean mask using morphological operations.
        
        Args:
            mask: Alpha matte [0, 1] float32.
            original_frame: Original RGB frame.
            
        Returns:
            Cleaned mask.
        """
        size = self.params.get("kernel_size", 0)
        if size <= 0:
            return mask
            
        mode = self.params.get("mode", "close_open")
        
        # Morphology works best on uint8 [0, 255]
        m_u8 = (mask * 255).astype(np.uint8)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size * 2 + 1, size * 2 + 1))
        
        if mode == "close" or mode == "close_open":
            m_u8 = cv2.morphologyEx(m_u8, cv2.MORPH_CLOSE, kernel)
        
        if mode == "open" or mode == "close_open":
            m_u8 = cv2.morphologyEx(m_u8, cv2.MORPH_OPEN, kernel)
            
        return (m_u8 / 255.0).astype(np.float32)
