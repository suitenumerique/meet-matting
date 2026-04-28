"""
Post-processor that strictly masks anything outside the detected Person Zoom bboxes.
Useful for cleaning up ghosts and ensuring privacy.
"""

import numpy as np
from core.base import Postprocessor
from core.parameters import ParameterSpec
from core.registry import postprocessors
from core import context

@postprocessors.register
class PersonZoomMasking(Postprocessor):
    name = "zoom_masking"
    description = "Forces the mask to zero outside the detected Person Zoom boxes."

    @classmethod
    def parameter_specs(cls):
        return [] # No params needed

    def reset(self):
        pass

    def __call__(self, mask: np.ndarray, original_frame: np.ndarray) -> np.ndarray:
        bboxes = context.get_val("person_bboxes", [])
        zoom_active = context.get_val("person_zoom_active", False)
        
        if not zoom_active or not bboxes:
            return mask
            
        h, w = mask.shape
        strict_mask = np.zeros((h, w), dtype=np.float32)
        for (x1, y1, x2, y2) in bboxes:
            strict_mask[y1:y2, x1:x2] = 1.0
            
        return mask * strict_mask
