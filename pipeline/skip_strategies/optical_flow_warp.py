import cv2
import numpy as np
from core.base import SkipStrategy
from core.parameters import ParameterSpec
from core.registry import skip_strategies

_FLOW_METHODS = ["dis_ultrafast", "dis_fast", "dis_medium", "farneback"]

_DIS_PRESETS = {
    "dis_ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
    "dis_fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
    "dis_medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
}


def _build_remap(flow: np.ndarray):
    """Convert a (H, W, 2) backward flow field into cv2.remap coordinate maps."""
    h, w = flow.shape[:2]
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    map_x, map_y = np.meshgrid(xs, ys)
    map_x += flow[..., 0]
    map_y += flow[..., 1]
    return map_x, map_y


@skip_strategies.register
class OpticalFlowWarp(SkipStrategy):
    name = "optical_flow_warp"
    description = (
        "Warps the previous mask to the current frame using dense optical flow (backward warp). "
        "Works best when the subject moves; may drift at edges if only the background moves."
    )

    @classmethod
    def parameter_specs(cls):
        return [
            ParameterSpec(
                name="flow_method",
                type="choice",
                default="dis_fast",
                label="Flow method",
                choices=_FLOW_METHODS,
                help=(
                    "Dense optical flow algorithm. "
                    "DIS variants are fast; Farneback is slower but sometimes smoother."
                ),
            ),
            ParameterSpec(
                name="blend",
                type="float",
                default=0.0,
                label="Blend with reuse (0 = pure warp, 1 = pure reuse)",
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                help=(
                    "Mix the warped mask with the plain reused mask. "
                    "Useful to dampen artefacts on static subjects or moving backgrounds."
                ),
            ),
        ]

    def reset(self):
        pass

    def __call__(
        self,
        current_frame: np.ndarray,
        prev_frame: np.ndarray,
        prev_mask: np.ndarray,
    ) -> np.ndarray:
        method = self.params["flow_method"]
        blend = float(self.params["blend"])

        h, w = current_frame.shape[:2]
        fw, fh = max(1, w // 4), max(1, h // 4)
        cur_small = cv2.resize(current_frame, (fw, fh), interpolation=cv2.INTER_LINEAR)
        prev_small = cv2.resize(prev_frame, (fw, fh), interpolation=cv2.INTER_LINEAR)

        prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_RGB2GRAY)
        current_gray = cv2.cvtColor(cur_small, cv2.COLOR_RGB2GRAY)

        # Backward flow: for each pixel in current_frame, where does it come from in prev_frame.
        # We compute flow from current → prev so that remap(prev_mask, flow) lands in the right spot.
        if method in _DIS_PRESETS:
            dis = cv2.DISOpticalFlow_create(_DIS_PRESETS[method])
            flow = dis.calc(current_gray, prev_gray, None)
        else:  # farneback
            flow = cv2.calcOpticalFlowFarneback(
                current_gray,
                prev_gray,
                flow=None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )

        # Scale flow vectors from flow resolution up to full-res mask space.
        mask_h, mask_w = prev_mask.shape[:2]
        if fw != mask_w or fh != mask_h:
            flow = cv2.resize(flow, (mask_w, mask_h), interpolation=cv2.INTER_LINEAR)
            flow[..., 0] *= mask_w / fw
            flow[..., 1] *= mask_h / fh

        map_x, map_y = _build_remap(flow)
        warped = cv2.remap(
            prev_mask,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        warped = warped.clip(0.0, 1.0)

        if blend > 0.0:
            warped = (1.0 - blend) * warped + blend * prev_mask

        return warped.astype(np.float32)
