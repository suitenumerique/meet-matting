"""
Anchored Mask postprocessor.

Problem targeted: a background object (frame, furniture, wall) is spatially
adjacent to the person and gets detected as foreground because the model is
uncertain at the boundary — the object then "bleeds" into the mask.

Principle:
  1. Extract an "anchor core" — pixels where the model is confident the
     region is person (alpha >= anchor_threshold).
  2. Dilate the core by max_expand_px pixels → this is the *allowed zone*.
  3. Zero out every mask pixel outside that zone.

Why this beats simple morphological opening:
  - Opening erodes uniformly and destroys soft edges (hair, fingers).
  - Here the core is always the high-confidence body/face; its dilation
    naturally follows the person's shape and preserves soft edges within
    reach, while blocking objects that are only adjacent to low-confidence
    boundary pixels.

Typical tuning:
  anchor_threshold = 0.55–0.65  (lower = more permissive anchors)
  max_expand_px    = 20–40      (larger = allows more soft edge recovery)
"""

import cv2
import numpy as np
from core.base import Postprocessor
from core.parameters import ParameterSpec
from core.registry import postprocessors


@postprocessors.register
class AnchoredMask(Postprocessor):
    name = "anchored_mask"
    description = (
        "Removes adjacent background objects by constraining the mask to regions "
        "reachable from high-confidence person pixels."
    )
    details = (
        "Best used AFTER the model and BEFORE temporal filters.\n"
        "If the ghost still appears, lower anchor_threshold or increase max_expand_px.\n"
        "If it cuts into the person's edges, increase max_expand_px."
    )

    @classmethod
    def parameter_specs(cls):
        return [
            ParameterSpec(
                name="anchor_threshold",
                type="float",
                default=0.60,
                label="Anchor threshold",
                min_value=0.30,
                max_value=0.90,
                step=0.05,
                help=(
                    "Pixels with alpha ≥ this value are treated as certain person "
                    "and used as anchors. Lower = more anchors (more permissive). "
                    "If anchor is too strict the person's limbs get clipped."
                ),
            ),
            ParameterSpec(
                name="max_expand_px",
                type="int",
                default=30,
                label="Max expansion (px)",
                min_value=5,
                max_value=120,
                step=5,
                help=(
                    "Maximum distance (pixels) the mask can extend from any anchor pixel. "
                    "Increase if hair/arm edges are being cut off. "
                    "Decrease to be more aggressive against adjacent ghosts."
                ),
            ),
        ]

    def reset(self):
        pass

    def __call__(self, mask: np.ndarray, original_frame: np.ndarray) -> np.ndarray:
        anchor_thr = float(self.params["anchor_threshold"])
        expand_px = int(self.params["max_expand_px"])

        # --- 1. Anchor core: high-confidence person pixels -------------------
        core = (mask >= anchor_thr).astype(np.uint8)

        if core.sum() == 0:
            # No high-confidence pixels found — skip to avoid blank frame
            return mask

        # --- 2. Dilate core → allowed zone -----------------------------------
        k_size = 2 * expand_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        allowed = cv2.dilate(core, kernel)  # uint8 binary

        # --- 3. Gate the original soft mask with the allowed zone ------------
        return mask * allowed.astype(np.float32)
