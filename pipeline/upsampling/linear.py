import cv2
import numpy as np
from core.base import UpsamplingMethod
from core.registry import upsamplers


@upsamplers.register
class Linear(UpsamplingMethod):
    name = "linear"
    description = "Bilinear interpolation — fast, no guidance."

    @classmethod
    def parameter_specs(cls):
        return []

    def _upsample_impl(self, low_res_mask: np.ndarray, guide: np.ndarray) -> np.ndarray:
        H_h, W_h = guide.shape[:2]
        if low_res_mask.shape[:2] == (H_h, W_h):
            return low_res_mask
        return cv2.resize(low_res_mask, (W_h, H_h), interpolation=cv2.INTER_LINEAR)
