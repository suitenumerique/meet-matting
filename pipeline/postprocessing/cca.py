"""
Connected Component Analysis (CCA) post-processor.
Removes small isolated mask islands (artifacts).

Optimized: uses a vectorized LUT approach instead of per-component Python loops.
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
        # Fast exit: completely empty mask
        if not np.any(mask > 0.0):
            return mask

        # 1. Binarize for robust component detection (uint8 for OpenCV)
        m_u8 = (mask * 255).astype(np.uint8)
        _, binary_mask = cv2.threshold(m_u8, 127, 255, cv2.THRESH_BINARY)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask)

        if num_labels <= 1:
            return mask

        # 2. Vectorized LUT: build a keep/discard table for ALL labels at once
        total_area = np.sum(stats[1:, cv2.CC_STAT_AREA])  # exclude background (label 0)
        if total_area == 0:
            return mask

        min_area = total_area * self.params["min_area_ratio"]

        # LUT: 1.0 = keep, 0.0 = discard.  Label 0 (background) always gets 0.
        lut = np.zeros(num_labels, dtype=np.float32)
        areas = stats[:, cv2.CC_STAT_AREA]
        lut[1:] = (areas[1:] >= min_area).astype(np.float32)

        # 3. Apply LUT in one vectorized indexing op (no Python loop)
        keep_mask = lut[labels]  # (H, W) float32, 0.0 or 1.0

        # Multiply: preserves soft alpha values for kept components, zeros discarded
        return mask * keep_mask
