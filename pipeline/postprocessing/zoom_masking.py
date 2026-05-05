"""
Post-processor that strictly masks anything outside the detected Person Zoom bboxes.
Useful for cleaning up ghosts and ensuring privacy.
"""

import numpy as np
from core import context
from core.base import Postprocessor
from core.registry import postprocessors


@postprocessors.register
class PersonZoomMasking(Postprocessor):
    name = "zoom_masking"
    description = "Forces the mask to zero outside the detected Person Zoom boxes."

    @classmethod
    def parameter_specs(cls):
        """Return the list of tunable parameters for this component."""
        return []  # No params needed

    def reset(self):
        """No temporal state to clear."""
        pass

    def __call__(self, mask: np.ndarray, original_frame: np.ndarray) -> np.ndarray:
        """Zero out *mask* outside the bounding boxes stored in the shared context.

        Reads ``person_bboxes`` and ``person_zoom_active`` from :mod:`core.context`.
        If zoom is not active or no boxes are present, *mask* is returned unchanged.

        Args:
            mask:           Alpha matte, shape (H, W), dtype float32, range [0, 1].
            original_frame: Original RGB frame, shape (H, W, 3), dtype uint8 (unused).

        Returns:
            Masked alpha matte — zeroed outside detected bounding boxes.
        """
        bboxes = context.get_val("person_bboxes", [])
        zoom_active = context.get_val("person_zoom_active", False)

        if not zoom_active or not bboxes:
            return mask

        h, w = mask.shape
        strict_mask = np.zeros((h, w), dtype=np.float32)
        for x1, y1, x2, y2 in bboxes:
            strict_mask[y1:y2, x1:x2] = 1.0

        return mask * strict_mask
