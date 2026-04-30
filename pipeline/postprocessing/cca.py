"""
Connected Component Analysis (CCA) post-processor.

Two complementary filters:
  - top_n        : keep only the N largest blobs (0 = keep all).
                   Most effective against furniture/object ghost detections.
  - min_area_pct : drop blobs below X % of the frame area.
                   Removes sub-pixel noise and tiny stray detections.

Typical anti-ghost setup: top_n=1 (or 2 if two people), min_area_pct=0.5.
"""

import cv2
import numpy as np
from core.base import Postprocessor
from core.parameters import ParameterSpec
from core.registry import postprocessors

_BINARISE_THRESH = 0.15  # mask value below which a pixel is treated as background


@postprocessors.register
class ConnectedComponents(Postprocessor):
    name = "cca"
    description = "Keeps the N largest blobs and removes tiny artefacts — eliminates ghost detections on furniture / objects."
    details = (
        "top_n=1  → single-person scenes (most aggressive ghost removal)\n"
        "top_n=2  → two-person scenes\n"
        "top_n=0  → disable the top-N filter, only min_area_pct applies\n\n"
        "min_area_pct filters blobs smaller than X% of the frame, independently of top_n."
    )

    @classmethod
    def parameter_specs(cls):
        return [
            ParameterSpec(
                name="top_n",
                type="int",
                default=1,
                label="Keep top N blobs (0 = all)",
                min_value=0,
                max_value=10,
                step=1,
                help=(
                    "Keep only the N largest connected components. "
                    "Set to 1 for a single person; 2 for two people. "
                    "0 disables this filter."
                ),
            ),
            ParameterSpec(
                name="min_area_pct",
                type="float",
                default=0.5,
                label="Min blob area (% of frame)",
                min_value=0.0,
                max_value=20.0,
                step=0.1,
                help=(
                    "Remove blobs whose area is smaller than this percentage of the "
                    "total frame area. Applied after top_n selection."
                ),
            ),
        ]

    def reset(self):
        pass

    def __call__(self, mask: np.ndarray, original_frame: np.ndarray) -> np.ndarray:
        if np.all(mask < _BINARISE_THRESH):
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

        return refined
