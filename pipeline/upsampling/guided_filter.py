import cv2
import numpy as np
from core.base import UpsamplingMethod
from core.parameters import ParameterSpec
from core.registry import upsamplers


@upsamplers.register
class GuidedFilter(UpsamplingMethod):
    name = "guided_filter"
    description = (
        "Guided filter upsampling — bilinear upsample then edge-aware refinement "
        "guided by the high-res frame. O(N) complexity regardless of radius."
    )

    @classmethod
    def parameter_specs(cls):
        return [
            ParameterSpec(
                name="radius",
                type="int",
                default=8,
                label="Radius (px)",
                min_value=1,
                max_value=64,
                step=1,
                help="Filter radius in high-res pixels. Literature: 4–60 depending on resolution.",
            ),
            ParameterSpec(
                name="eps",
                type="float",
                default=1e-2,
                label="Epsilon",
                min_value=1e-4,
                max_value=0.1,
                step=1e-4,
                help="Regularisation (guide normalised to [0,1]). Literature: 1e-5–0.1. "
                "Smaller = edges preserved more aggressively. Fine matting: ~1e-4; smooth blending: ~0.05.",
            ),
        ]

    def upsample(self, low_res_mask: np.ndarray, guide: np.ndarray) -> np.ndarray:
        """Bilinear-upsample then refine with a guided filter.

        Args:
            low_res_mask: float32 (H_l, W_l), values in [0, 1].
            guide:        uint8 RGB (H_h, W_h, 3), full-resolution frame.

        Returns:
            float32 (H_h, W_h), values in [0, 1].
        """
        H_h, W_h = guide.shape[:2]

        mask_up = cv2.resize(low_res_mask, (W_h, H_h), interpolation=cv2.INTER_LINEAR)

        # Normalise guide to [0, 1] so that ε is on the same scale as Var(I).
        # Without this, Var(I) ~ thousands (uint8) and ε ∈ [0, 1] has no effect.
        guide_f = guide.astype(np.float32) / 255.0

        result = cv2.ximgproc.guidedFilter(
            guide=guide_f,
            src=mask_up,
            radius=int(self.params["radius"]),
            eps=float(self.params["eps"]),
        )

        return np.clip(result, 0.0, 1.0).astype(np.float32)
