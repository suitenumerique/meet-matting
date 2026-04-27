from __future__ import annotations

import numpy as np

from core.base import MattingModel, Postprocessor, Preprocessor


class MattingPipeline:
    """Orchestrates preprocessing → model inference → postprocessing for a single frame."""

    def __init__(
        self,
        preprocessors: list[Preprocessor],
        model: MattingModel,
        postprocessors: list[Postprocessor],
    ):
        self.preprocessors = preprocessors
        self.model = model
        self.postprocessors = postprocessors

    def process_frame(self, frame: np.ndarray) -> dict:
        """Run the full pipeline on one frame.

        Args:
            frame: RGB image, shape (H, W, 3), dtype uint8.

        Returns:
            A dict with keys:
                ``original``     — uint8 (H, W, 3), unmodified input frame.
                ``preprocessed`` — uint8 (H, W, 3), frame after all preprocessors.
                ``raw_mask``     — float32 (H, W) in [0, 1], raw model output.
                ``final_mask``   — float32 (H, W) in [0, 1], after postprocessors.
                ``final``        — uint8 (H, W, 3), original composited with final_mask.
        """
        original = frame

        preprocessed = frame.copy()
        for pre in self.preprocessors:
            preprocessed = pre(preprocessed)

        raw_mask = self.model.infer(preprocessed)

        final_mask = raw_mask.copy()
        for post in self.postprocessors:
            final_mask = post(final_mask, original)

        # Composite uses the original (un-preprocessed) frame.
        final = (original * final_mask[..., None]).astype(np.uint8)

        return {
            "original": original,
            "preprocessed": preprocessed,
            "raw_mask": raw_mask,
            "final_mask": final_mask,
            "final": final,
        }
