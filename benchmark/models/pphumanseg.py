"""
Wrapper for PP-HumanSeg V2 (PaddleSeg).

PP-HumanSeg V2 is a lightweight human segmentation model developed by
Baidu/PaddlePaddle, optimised for mobile deployment.

This wrapper uses ONNX Runtime to avoid the PaddlePaddle dependency.
The ONNX model must be converted beforehand or downloaded.

Model: https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.9/contrib/PP-HumanSeg
"""

import logging
from pathlib import Path
import cv2
import numpy as np
from typing import List, Optional, Tuple

from .base import BaseModelWrapper

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_PATH = (
    Path(__file__).parent.parent / "weights" / "pphumanseg_v2.onnx"
)


class PPHumanSegV2Wrapper(BaseModelWrapper):
    """
    PP-HumanSeg V2 via ONNX Runtime.

    Expected input: (1, 3, 192, 192) normalised with PaddleSeg mean/std.
    Output: segmentation map (1, 2, H, W) — argmax to get the binary mask.
    """

    _INPUT_H = 192
    _INPUT_W = 192
    _MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/human_segmentation_pphumanseg/human_segmentation_pphumanseg_2023mar.onnx"

    def __init__(self, model_path: Optional[str] = None):
        self._model_path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
        self._session = None
        self._input_name = None

    @property
    def name(self) -> str:
        return "PP-HumanSeg V2"

    @property
    def input_size(self) -> Optional[Tuple[int, int]]:
        return (self._INPUT_H, self._INPUT_W)

    def load(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "onnxruntime is required. Install it via: pip install onnxruntime"
            ) from e

        if not self._model_path.exists():
            logger.info("PP-HumanSeg V2: downloading model from %s", self._MODEL_URL)
            self._download_model()

        providers = ort.get_available_providers()
        selected_providers = []
        if "CoreMLExecutionProvider" in providers:
            selected_providers.append("CoreMLExecutionProvider")
        if "CUDAExecutionProvider" in providers:
            selected_providers.append("CUDAExecutionProvider")
        selected_providers.append("CPUExecutionProvider")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # CoreML specific optimizations if on Mac
        actual_providers = []
        for p in selected_providers:
            if p == "CoreMLExecutionProvider":
                actual_providers.append(
                    ("CoreMLExecutionProvider", {
                        "MLComputeUnits": "ALL",
                        "convert_model_to_fp16": True  # Enable FP16 inference on Mac
                    })
                )
            else:
                actual_providers.append(p)

        self._session = ort.InferenceSession(
            str(self._model_path),
            providers=actual_providers,
            sess_options=sess_options
        )
        self._input_name = self._session.get_inputs()[0].name
        logger.info("PP-HumanSeg V2: ONNX model loaded (%s).", self._model_path.name)

    def _download_model(self) -> None:
        import urllib.request
        self._model_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("PP-HumanSeg V2: downloading to %s …", self._model_path)
        urllib.request.urlretrieve(self._MODEL_URL, str(self._model_path))
        logger.info("PP-HumanSeg V2: download complete.")

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        return self.predict_batch([frame_bgr])[0]

    def predict_batch(self, frames_bgr: List[np.ndarray]) -> List[np.ndarray]:
        if not frames_bgr:
            return []

        h_orig, w_orig = frames_bgr[0].shape[:2]

        if self._session is None:
            logger.warning("PP-HumanSeg V2: no model loaded, returning empty masks.")
            return [np.zeros((h_orig, w_orig), dtype=np.float32) for _ in frames_bgr]

        batch_size = len(frames_bgr)

        # Pre-processing
        tensors = []
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        std = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        for frame in frames_bgr:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (self._INPUT_W, self._INPUT_H))
            tensor = (frame_resized.astype(np.float32) / 255.0 - mean) / std
            tensor = np.transpose(tensor, (2, 0, 1))
            tensors.append(tensor)

        batch_tensor = np.stack(tensors, axis=0)

        # Inference
        # Note: some ONNX models (such as the one from OpenCV Zoo) have a fixed batch size of 1.
        # We check whether we can pass the entire batch or whether we need to loop.
        try:
            output = self._session.run(None, {self._input_name: batch_tensor})
            logits_batch = output[0]
        except Exception as e:
            if "Got: " in str(e) and "Expected: 1" in str(e):
                logger.debug("PP-HumanSeg: batching not supported by the model, falling back to iteration.")
                logits_batch = []
                for i in range(batch_size):
                    t = np.expand_dims(batch_tensor[i], axis=0)
                    out = self._session.run(None, {self._input_name: t})
                    logits_batch.append(out[0][0])
                logits_batch = np.array(logits_batch)
            else:
                raise e

        masks = []
        for i in range(batch_size):
            l = logits_batch[i]
            if l.ndim == 3 and l.shape[0] >= 2:
                # Manual softmax on a single item
                exp_l = np.exp(l - l.max(axis=0, keepdims=True))
                probs = exp_l / exp_l.sum(axis=0, keepdims=True)
                mask = probs[1]  # "Person" class
            elif l.ndim == 3 and l.shape[0] == 1:
                mask = 1.0 / (1.0 + np.exp(-l[0]))
            else:
                mask = l.squeeze()

            # Binarisation for PP-HumanSeg
            mask = (mask > 0.5).astype(np.float32)

            if mask.shape != (h_orig, w_orig):
                mask = cv2.resize(mask, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

            masks.append(mask)

        return masks

    def get_flops(self, input_shape: Tuple[int, int, int] = (3, 192, 192)) -> float:
        # PP-HumanSeg V2: ~90 MFLOPs at 192x192
        c, h, w = input_shape
        base_flops = 90e6
        scale = (h * w) / (192 * 192)
        return base_flops * scale

    def cleanup(self) -> None:
        self._session = None
        logger.info("PP-HumanSeg V2: ONNX session closed.")
