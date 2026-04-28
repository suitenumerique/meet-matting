"""
Orchestration layer for the matting pipeline.
Handles global features like 'Person Zoom' (multi-crop inference) 
with context padding to avoid edge artifacts.
"""
import cv2
import numpy as np
from core.base import MattingModel, Postprocessor, Preprocessor
from core import context

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

    def reset(self):
        """Reset state of all components (counters, buffers, etc.)."""
        for pre in self.preprocessors:
            pre.reset()
        self.model.reset()
        for post in self.postprocessors:
            post.reset()

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
            mask_full = np.zeros((h_orig, w_orig), dtype=np.float32)
            
            for (x1, y1, x2, y2) in bboxes:
                # Add CONTEXT PADDING (20%) to help the model distinguish bg from person
                bw, bh = x2 - x1, y2 - y1
                pad_w, pad_h = int(bw * 0.2), int(bh * 0.2)
                
                # Expanded coordinates (clipped to frame)
                ex1 = max(0, x1 - pad_w)
                ey1 = max(0, y1 - pad_h)
                ex2 = min(w_orig, x2 + pad_w)
                ey2 = min(h_orig, y2 + pad_h)
                
                crop = inference_frame[ey1:ey2, ex1:ex2]
                if crop.size == 0: continue
                
                # Inference on expanded crop
                mask_expanded = self.model.infer(crop)
                if mask_expanded.ndim == 3: mask_expanded = mask_expanded.squeeze(-1)
                
                # Resize mask back to expanded crop size
                mask_expanded = cv2.resize(mask_expanded, (ex2-ex1, ey2-ey1), interpolation=cv2.INTER_LINEAR)
                
                # Extract only the original bbox area from the expanded mask
                mask_crop = mask_expanded[y1-ey1 : y2-ey1, x1-ex1 : x2-ex1]
                
                # Paste into full mask
                mask_full[y1:y2, x1:x2] = np.maximum(mask_full[y1:y2, x1:x2], mask_crop)
            
            raw_mask = mask_full
        else:
            raw_mask = self.model.infer(inference_frame)

        # 3. Postprocessing
        final_mask = raw_mask.copy()
        for post in self.postprocessors:
            final_mask = post(final_mask, original)

        # 4. Final Compositing
        if final_mask.ndim == 3:
            final_mask = final_mask.squeeze(-1)
        
        comp_mask = final_mask[:, :, np.newaxis]
        final = (original * comp_mask).astype(np.uint8)

        return {
            "original": original,
            "preprocessed": debug_frame,
            "raw_mask": raw_mask,
            "final_mask": final_mask,
            "final": final,
        }
