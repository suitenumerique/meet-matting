"""
Connected Component Analysis (CCA) post-processor.
Removes small isolated mask islands (artifacts).
"""

import cv2
import numpy as np
from core.base import Postprocessor
from core.parameters import ParameterSpec
from core.registry import postprocessors


@postprocessors.register
class ConnectedComponents(Postprocessor):
    name = "cca"
    description = "Removes small isolated artifacts using Connected Component Analysis."

    @classmethod
    def parameter_specs(cls):
        return [
            ParameterSpec(
                name="min_area_ratio",
                type="float",
                default=0.05,
                label="Min Area Ratio",
                min_value=0.01,
                max_value=0.5,
                step=0.01,
                help="Components smaller than this ratio of the total mask area will be removed.",
            ),
        ]

    def reset(self):
        pass

    def __call__(self, mask: np.ndarray, original_frame: np.ndarray) -> np.ndarray:
        if np.sum(mask) == 0:
            return mask

        # 1. Convert to binary u8 for robust component detection
        # We use a mid-threshold (127) to isolate strong components
        m_u8 = (mask * 255).astype(np.uint8)
        _, binary_mask = cv2.threshold(m_u8, 127, 255, cv2.THRESH_BINARY)
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask)

        if num_labels <= 1:
            return mask

        # 2. Calculate total area of "confident" mask
        total_area = np.sum(binary_mask > 0)
        if total_area == 0:
            return mask

        refined_mask = mask.copy()
        min_area = total_area * self.params["min_area_ratio"]

        # 3. Remove small components
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                # Zero out this component in the final soft mask
                refined_mask[labels == i] = 0

        return refined_mask
