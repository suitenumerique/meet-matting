import cv2
import numpy as np
from core.base import UpsamplingMethod
from core.parameters import ParameterSpec
from core.registry import upsamplers


@upsamplers.register
class JointBilateral(UpsamplingMethod):
    name = "joint_bilateral"
    description = "Joint bilateral upsampling — edge-aware, guided by the high-res frame."

    @classmethod
    def parameter_specs(cls):
        return [
            ParameterSpec(
                name="radius",
                type="int",
                default=2,
                label="Radius (low-res px)",
                min_value=1,
                max_value=4,
                step=1,
                help="Neighbourhood radius in low-resolution space. Literature: 1–3. Larger = smoother but slower.",
            ),
            ParameterSpec(
                name="sigma_s",
                type="float",
                default=1.0,
                label="Spatial sigma",
                min_value=0.1,
                max_value=3.0,
                step=0.1,
                help="Spatial falloff in low-res pixels. Literature: 0.5–3.0.",
            ),
            ParameterSpec(
                name="sigma_r",
                type="float",
                default=0.1,
                label="Range sigma",
                min_value=0.01,
                max_value=0.5,
                step=0.01,
                help="Sensitivity to luma differences in the guide [0,1]. "
                "Literature: 0.05–0.25. Smaller = sharper edges.",
            ),
        ]

    def upsample(self, low_res_mask: np.ndarray, guide: np.ndarray) -> np.ndarray:
        """Joint bilateral upsample *low_res_mask* guided by the full-resolution *guide*.

        Algorithm (Kopf et al. 2007): for every high-res output pixel x, aggregate
        low-res neighbours weighted by spatial proximity (sigma_s) and colour
        similarity between the high-res guide at x and the low-res guide at each
        neighbour (sigma_r).

        Args:
            low_res_mask: float32 (H_l, W_l), values in [0, 1].
            guide:        uint8 RGB (H_h, W_h, 3), full-resolution frame.

        Returns:
            float32 (H_h, W_h), values in [0, 1].
        """
        H_h, W_h = guide.shape[:2]
        H_l, W_l = low_res_mask.shape

        if (H_l, W_l) == (H_h, W_h):
            return low_res_mask

        radius = int(self.params["radius"])
        sigma_s = float(self.params["sigma_s"])
        sigma_r = float(self.params["sigma_r"])

        # Downscale guide to low-res for range comparisons
        guide_low = cv2.resize(guide, (W_l, H_l), interpolation=cv2.INTER_AREA)

        # Convert both guides to float32 luma [0, 1] for the range kernel
        def _luma(img: np.ndarray) -> np.ndarray:
            r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32) / 255.0

        luma_h = _luma(guide)  # (H_h, W_h)
        luma_l = _luma(guide_low)  # (H_l, W_l)

        # Float low-res coordinates for every high-res pixel
        ys_l = np.arange(H_h, dtype=np.float32) * (H_l / H_h)  # (H_h,)
        xs_l = np.arange(W_h, dtype=np.float32) * (W_l / W_h)  # (W_h,)

        # Integer base indices in low-res grid
        j_base = np.floor(ys_l).astype(np.int32)  # (H_h,)
        i_base = np.floor(xs_l).astype(np.int32)  # (W_h,)

        output = np.zeros((H_h, W_h), dtype=np.float32)
        weight_sum = np.zeros((H_h, W_h), dtype=np.float32)

        inv2ss = 1.0 / (2.0 * sigma_s**2)
        inv2sr = 1.0 / (2.0 * sigma_r**2)

        for dj in range(-radius, radius + 1):
            for di in range(-radius, radius + 1):
                j = np.clip(j_base + dj, 0, H_l - 1)  # (H_h,)
                i = np.clip(i_base + di, 0, W_l - 1)  # (W_h,)

                # Spatial distance (in low-res units) between high-res pixel and neighbour
                dy = ys_l - (j_base + dj)  # (H_h,)
                dx = xs_l - (i_base + di)  # (W_h,)
                w_s = np.exp(-(dy[:, None] ** 2 + dx[None, :] ** 2) * inv2ss)  # (H_h, W_h)

                # Range distance: high-res luma at (y,x) vs low-res luma at neighbour (j,i)
                luma_nb = luma_l[j[:, None], i[None, :]]  # (H_h, W_h)
                w_r = np.exp(-((luma_h - luma_nb) ** 2) * inv2sr)  # (H_h, W_h)

                w = w_s * w_r
                output += w * low_res_mask[j[:, None], i[None, :]]
                weight_sum += w

        return (output / np.maximum(weight_sum, 1e-8)).astype(np.float32)
