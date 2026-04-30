"""
Orchestration layer for the matting pipeline.
Handles global features like 'Person Zoom' (multi-crop inference)
with context padding to avoid edge artifacts.
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
        bg_image: np.ndarray | None = None,
    ):
        self.preprocessors = preprocessors
        self.model = model
        self.postprocessors = postprocessors
        if bg_image is not None:
            self._bg = bg_image.astype(np.float32)  # (H, W, 3)
        else:
            self._bg = np.array(bg_color, dtype=np.float32)[None, None, :]  # (1, 1, 3)

        # ── Pre-allocated composition buffers (lazily sized on first frame) ──
        self._buf_mask3: np.ndarray | None = None   # uint8 (H, W, 3)
        self._buf_inv3: np.ndarray | None = None    # uint8 (H, W, 3)
        self._bg_u8: np.ndarray | None = None       # uint8 (H, W, 3)
        self._buf_shape: tuple[int, int] = (0, 0)   # (H, W) of last frame

    def reset(self):
        """Reset state of all components (counters, buffers, etc.)."""
        for pre in self.preprocessors:
            pre.reset()
        self.model.reset()
        for post in self.postprocessors:
            post.reset()

    def process_frame(self, frame: np.ndarray) -> dict:
        """Run the full pipeline on one frame with detailed profiling."""
        import time
        timings = {}
        t_start = time.perf_counter()

        context.clear()

        original = frame.copy()
        inference_frame = frame.copy()

        # 1. Preprocessing
        t_pre_start = time.perf_counter()
        for pre in self.preprocessors:
            t_c_start = time.perf_counter()
            inference_frame = pre(inference_frame)
            timings[f"pre_{pre.name}"] = time.perf_counter() - t_c_start
        timings["preprocessing_total"] = time.perf_counter() - t_pre_start

        # 2. Model Inference (with automatic Person Zoom support)
        t_model_start = time.perf_counter()
        context.set_val("upsampling_time", 0.0)  # Reset for this frame
        
        bboxes = context.get_val("person_bboxes", [])
        zoom_active = context.get_val("person_zoom_active", False)
        h_orig, w_orig = frame.shape[:2]

        if zoom_active and bboxes:
            mask_full = np.zeros((h_orig, w_orig), dtype=np.float32)

            for x1, y1, x2, y2 in bboxes:
                # Context padding (20%)
                bw, bh = x2 - x1, y2 - y1
                pad_w, pad_h = int(bw * 0.2), int(bh * 0.2)

                ex1 = max(0, x1 - pad_w)
                ey1 = max(0, y1 - pad_h)
                ex2 = min(w_orig, x2 + pad_w)
                ey2 = min(h_orig, y2 + pad_h)

                crop = inference_frame[ey1:ey2, ex1:ex2]
                if crop.size == 0:
                    continue

                mask_expanded = self.model.infer(crop)
                if mask_expanded.ndim == 3:
                    mask_expanded = mask_expanded.squeeze(-1)

                # Time the crop upsampling (back to original resolution of the crop)
                t_up_start = time.perf_counter()
                mask_expanded = cv2.resize(
                    mask_expanded, (ex2 - ex1, ey2 - ey1), interpolation=cv2.INTER_LINEAR
                )
                dt_up = time.perf_counter() - t_up_start
                context.set_val("upsampling_time", context.get_val("upsampling_time", 0.0) + dt_up)

                mask_crop = mask_expanded[y1 - ey1 : y2 - ey1, x1 - ex1 : x2 - ex1]
                mask_full[y1:y2, x1:x2] = np.maximum(mask_full[y1:y2, x1:x2], mask_crop)

            raw_mask = mask_full
        else:
            raw_mask = self.model.infer(inference_frame)

        while raw_mask.ndim > 2:
            raw_mask = raw_mask.squeeze(-1)
        
        timings["model_inference"] = time.perf_counter() - t_model_start
        timings["upsampling"] = context.get_val("upsampling_time", 0.0)

        # 3. Postprocessing
        t_post_start = time.perf_counter()
        final_mask = raw_mask.copy()
        for post in self.postprocessors:
            t_c_start = time.perf_counter()
            final_mask = post(final_mask, original)
            timings[f"post_{post.name}"] = time.perf_counter() - t_c_start
        timings["postprocessing_total"] = time.perf_counter() - t_post_start

        # 4. Final Compositing — uint8 fast path (OpenCV SIMD)
        t_comp_start = time.perf_counter()

        if final_mask.ndim == 3:
            final_mask = final_mask.squeeze(-1)

        # Lazy-allocate on resolution change
        if self._buf_shape != (h_orig, w_orig):
            self._buf_mask3 = np.empty((h_orig, w_orig, 3), dtype=np.uint8)
            self._buf_inv3  = np.empty((h_orig, w_orig, 3), dtype=np.uint8)
            self._buf_shape = (h_orig, w_orig)
            # Pre-compute background as uint8 at target resolution (once)
            if self._bg_image is not None:
                bg = cv2.resize(self._bg_image, (w_orig, h_orig))
                self._bg_u8 = bg if bg.dtype == np.uint8 else bg.astype(np.uint8)
            else:
                self._bg_u8 = np.full((h_orig, w_orig, 3), self._bg_color, dtype=np.uint8)

        # mask float32 [0,1] → uint8 [0,255]  (single channel, cheap)
        mask_u8 = np.clip(final_mask * 255.0, 0, 255).astype(np.uint8)

        # Expand to 3 channels (contiguous memory for SIMD)
        cv2.merge((mask_u8, mask_u8, mask_u8), dst=self._buf_mask3)
        np.subtract(255, self._buf_mask3, out=self._buf_inv3)

        # Alpha blend entirely in uint8:
        #   fg = saturate_uint8(original * mask3 / 255)
        #   bg = saturate_uint8(bg_u8 * inv_mask3 / 255)
        #   final = fg + bg
        _S = 1.0 / 255.0
        fg = cv2.multiply(original, self._buf_mask3, scale=_S, dtype=cv2.CV_8U)
        bg = cv2.multiply(self._bg_u8, self._buf_inv3, scale=_S, dtype=cv2.CV_8U)
        final = cv2.add(fg, bg)

        timings["compositing"] = time.perf_counter() - t_comp_start


        # 5. Prepare debug view
        # We draw boxes on a separate debug frame
        debug_frame = original.copy()
        if zoom_active:
            for x1, y1, x2, y2 in bboxes:
                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(debug_frame, "ZOOM", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw Pose Landmarks
        if context.get_val("show_landmarks") and context.get_val("pose_landmarks"):
            all_pose_landmarks = context.get_val("pose_landmarks")
            h, w = debug_frame.shape[:2]

            connections = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28), (11, 23), (12, 24), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6)]
            for pose_landmarks in all_pose_landmarks:
                for start_idx, end_idx in connections:
                    if start_idx < len(pose_landmarks) and end_idx < len(pose_landmarks):
                        p1, p2 = pose_landmarks[start_idx], pose_landmarks[end_idx]
                        if getattr(p1, "visibility", 1.0) > 0.5 and getattr(p2, "visibility", 1.0) > 0.5:
                            c1, c2 = (int(p1.x * w), int(p1.y * h)), (int(p2.x * w), int(p2.y * h))
                            cv2.line(debug_frame, c1, c2, (0, 255, 255), 2)
                for lm in pose_landmarks:
                    if getattr(lm, "visibility", 1.0) > 0.5:
                        cv2.circle(debug_frame, (int(lm.x * w), int(lm.y * h)), 3, (0, 255, 0), -1)

        timings["total_pipeline"] = time.perf_counter() - t_start

        return {
            "original": original,
            "preprocessed": debug_frame,
            "raw_mask": raw_mask,
            "final_mask": final_mask,
            "final": final,
            "timings": timings,
            "bboxes": bboxes if zoom_active else [],
            "zoom_active": zoom_active
        }
