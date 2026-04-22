"""
Wrapper for MODNet — Trimap-Free Portrait Matting in Real Time.

MODNet is a single-stage matting network that does not require a trimap.
It directly produces a high-quality alpha matte.

This wrapper uses ONNX Runtime for inference.

Model: https://github.com/ZHKKKe/MODNet
"""

import logging
from pathlib import Path
import cv2
import numpy as np
from typing import List, Optional, Tuple

from .base import BaseModelWrapper

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "weights" / "modnet_photographic_portrait_matting.onnx"
_MODEL_URL = (
    "https://huggingface.co/Xenova/modnet/resolve/main/onnx/model.onnx?download=true"
)


class MODNetWrapper(BaseModelWrapper):
    """
    MODNet via ONNX Runtime.

    Expected input: (1, 3, H, W) normalised to [0, 1].
    Output: alpha matte (1, 1, H, W) in [0, 1].
    """

    def __init__(self, model_path: Optional[str] = None):
        self._model_path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
        self._session = None
        self._input_name = None
        self._ref_size = 512  # MODNet expects multiples of 32, typically 512

    @property
    def name(self) -> str:
        return "MODNet"

    @property
    def input_size(self) -> Optional[Tuple[int, int]]:
        return (self._ref_size, self._ref_size)

    def load(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "onnxruntime is required. Install it via: pip install onnxruntime"
            ) from e

        if not self._model_path.exists():
            logger.info("MODNet: downloading model from %s", _MODEL_URL)
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
        logger.info("MODNet: ONNX model loaded (%s).", self._model_path.name)

    def _download_model(self) -> None:
        import urllib.request

        self._model_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("MODNet: downloading to %s …", self._model_path)
        urllib.request.urlretrieve(_MODEL_URL, str(self._model_path))
        logger.info("MODNet: download complete.")

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        return self.predict_batch([frame_bgr])[0]

    def predict_batch(self, frames_bgr: List[np.ndarray]) -> List[np.ndarray]:
        if self._session is None:
            raise RuntimeError("MODNet: model not loaded. Call load() first.")

        batch_size = len(frames_bgr)
        if batch_size == 0:
            return []

        h_orig, w_orig = frames_bgr[0].shape[:2]

        # Pre-processing: BGR -> RGB, resize, normalise
        tensors = []
        for frame in frames_bgr:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (self._ref_size, self._ref_size))

            # Normalise [0, 1] then MODNet standardisation
            tensor = frame_resized.astype(np.float32) / 255.0
            tensor = (tensor - 0.5) / 0.5  # [-1, 1]
            tensor = np.transpose(tensor, (2, 0, 1))   # CHW
            tensors.append(tensor)

        # Concatenate to form a batch (N, 3, H, W)
        batch_tensor = np.stack(tensors, axis=0)

        # Inference
        try:
            output = self._session.run(None, {self._input_name: batch_tensor})
            alphas_batch = output[0]
        except Exception as e:
            if "Got: " in str(e) and "Expected: 1" in str(e):
                logger.debug("MODNet: batching not supported by the model, falling back to iteration.")
                alphas_batch = []
                for i in range(batch_size):
                    t = np.expand_dims(batch_tensor[i], axis=0)
                    out = self._session.run(None, {self._input_name: t})
                    alphas_batch.append(out[0][0])
                alphas_batch = np.array(alphas_batch)
            else:
                raise e

        masks = []
        for i in range(batch_size):
            mask = alphas_batch[i, 0]  # (H, W)
            mask = np.clip(mask, 0.0, 1.0)

            # Resize back to the original size
            if mask.shape != (h_orig, w_orig):
                mask = cv2.resize(mask, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

            masks.append(mask.astype(np.float32))

        return masks

    def get_flops(self, input_shape: Tuple[int, int, int] = (3, 256, 256)) -> float:
        # MODNet: ~4 GFLOPs at 512x512
        c, h, w = input_shape
        base_flops = 4e9
        scale = (h * w) / (512 * 512)
        return base_flops * scale

    def cleanup(self) -> None:
        self._session = None
        logger.info("MODNet: ONNX session closed.")
