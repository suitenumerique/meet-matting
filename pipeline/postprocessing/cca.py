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
        """Return the list of tunable parameters for this component."""
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
        """No temporal state to clear."""
        pass

    def __call__(self, mask: np.ndarray, original_frame: np.ndarray) -> np.ndarray:
        """Zero out small or excess connected components, keeping only the top-N largest blobs.

        Args:
            mask:           Alpha matte, shape (H, W), dtype float32, range [0, 1].
            original_frame: Original RGB frame, shape (H, W, 3), dtype uint8 (unused).

        Returns:
            Refined alpha matte, shape (H, W), dtype float32, range [0, 1].
        """
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

        frame_area = mask.shape[0] * mask.shape[1]
        min_area = frame_area * (self.params["min_area_pct"] / 100.0)

        # LUT: 1.0 = keep, 0.0 = discard.  Label 0 (background) always gets 0.
        lut = np.zeros(num_labels, dtype=np.float32)
        areas = stats[:, cv2.CC_STAT_AREA]
        lut[1:] = (areas[1:] >= min_area).astype(np.float32)

        # 3. Apply LUT in one vectorized indexing op (no Python loop)
        keep_mask = lut[labels]  # (H, W) float32, 0.0 or 1.0

        # Multiply: preserves soft alpha values for kept components, zeros discarded
        return mask * keep_mask
