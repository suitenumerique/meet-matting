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

        # 1. Preprocessing
        for pre in self.preprocessors:
            inference_frame = pre(inference_frame)

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

        # 5. Prepare debug view for UI (preprocessed frame + overlays)
        debug_frame = inference_frame.copy()
        for (x1, y1, x2, y2) in bboxes:
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(
                debug_frame,
                "ZOOM ZONE",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        # Draw Pose Landmarks if available
        if context.get_val("show_landmarks") and context.get_val("pose_landmarks"):
            all_pose_landmarks = context.get_val("pose_landmarks")
            h, w = debug_frame.shape[:2]
            
            # Key connections: (start_idx, end_idx)
            # Shoulders (11, 12), Hips (23, 24), Shoulders to Hips (11-23, 12-24), Arms, Legs
            connections = [
                (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), # Upper body
                (23, 24), (23, 25), (25, 27), (24, 26), (26, 28), # Lower body
                (11, 23), (12, 24), # Torso sides
                (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6) # Face (simplified)
            ]
            
            for pose_landmarks in all_pose_landmarks:
                # Draw connections
                for start_idx, end_idx in connections:
                    if start_idx < len(pose_landmarks) and end_idx < len(pose_landmarks):
                        p1 = pose_landmarks[start_idx]
                        p2 = pose_landmarks[end_idx]
                        # Only draw if visibility is decent
                        if getattr(p1, 'visibility', 1.0) > 0.5 and getattr(p2, 'visibility', 1.0) > 0.5:
                            c1 = (int(p1.x * w), int(p1.y * h))
                            c2 = (int(p2.x * w), int(p2.y * h))
                            cv2.line(debug_frame, c1, c2, (0, 255, 255), 2)

                # Draw points
                for i, lm in enumerate(pose_landmarks):
                    if getattr(lm, 'visibility', 1.0) > 0.5:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(debug_frame, (cx, cy), 3, (0, 255, 0), -1)

        return {
            "original": original,
            "preprocessed": debug_frame,
            "raw_mask": raw_mask,
            "final_mask": final_mask,
            "final": final,
        }
