"""
Orchestration layer for the matting pipeline.
Handles global features like 'Person Zoom' (multi-crop inference)
so that all models can benefit from it automatically.
"""

import cv2
import numpy as np

from core import context
from core.base import MattingModel, Postprocessor, Preprocessor


class MattingPipeline:
    """Orchestrates preprocessing → model inference → postprocessing for a single frame."""

    def __init__(
        self,
        preprocessors: list[Preprocessor],
        model: MattingModel,
        postprocessors: list[Postprocessor],
        bg_color: tuple[int, int, int] = (0, 0, 0),
    ):
        self.preprocessors = preprocessors
        self.model = model
        self.postprocessors = postprocessors
        self._bg = np.array(bg_color, dtype=np.float32)[None, None, :]  # (1, 1, 3)

    def process_frame(self, frame: np.ndarray) -> dict:
        """Run the full pipeline on one frame."""
        context.clear()

        original = frame.copy()
        inference_frame = frame.copy()
        debug_frame = frame.copy()

        # 1. Preprocessing (populates context with bboxes, etc.)
        for pre in self.preprocessors:
            debug_frame = pre(debug_frame)

        # 2. Model Inference (with automatic Person Zoom support)
        bboxes = context.get_val("person_bboxes", [])
        zoom_active = context.get_val("person_zoom_active", False)
        h_orig, w_orig = frame.shape[:2]

        if zoom_active and bboxes:
            # Global Zoom Logic: runs the model on each crop and merges
            mask_full = np.zeros((h_orig, w_orig), dtype=np.float32)
            for x1, y1, x2, y2 in bboxes:
                crop = inference_frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # The model just sees a crop, it doesn't know it's a crop
                mask_small = self.model.infer(crop)

                # Resize back to crop size and paste
                mask_crop = cv2.resize(
                    mask_small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR
                )
                mask_full[y1:y2, x1:x2] = np.maximum(mask_full[y1:y2, x1:x2], mask_crop)

            raw_mask = mask_full
        else:
            # Standard full-frame inference
            raw_mask = self.model.infer(inference_frame)

        # 3. Postprocessing
        final_mask = raw_mask.copy()
        for post in self.postprocessors:
            final_mask = post(final_mask, original)

        # 4. Final Compositing — alpha blend over chosen background colour
        mask3 = final_mask[..., None]  # (H, W, 1)
        final = (original * mask3 + self._bg * (1.0 - mask3)).clip(0, 255).astype(np.uint8)

        return {
            "original": original,
            "preprocessed": debug_frame,
            "raw_mask": raw_mask,
            "final_mask": final_mask,
            "final": final,
        }
