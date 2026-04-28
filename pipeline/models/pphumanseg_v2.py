import logging
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from core.base import MattingModel
from core.parameters import ParameterSpec
from core.registry import models

logger = logging.getLogger(__name__)

_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/human_segmentation_pphumanseg/human_segmentation_pphumanseg_2023mar.onnx"
_WEIGHTS_DIR = Path(__file__).parent.parent / "weights"
_DEFAULT_MODEL_PATH = _WEIGHTS_DIR / "pphumanseg_v2.onnx"

_INPUT_H = 192
_INPUT_W = 192


@models.register
class PPHumanSegV2(MattingModel):
    name = "pphumanseg_v2"
    description = (
        "PP-HumanSeg V2 — lightweight PaddleSeg human segmenter via ONNX Runtime, 192×192."
    )

    _session = None
    _input_name: str | None = None
    upsampler = None

    @classmethod
    def parameter_specs(cls):
        return []

    def load(self, weights_path=None):
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "onnxruntime is required. Install it with: pip install onnxruntime"
            ) from e

        model_path = Path(weights_path) if weights_path else _DEFAULT_MODEL_PATH
        if not model_path.exists():
            logger.info("PP-HumanSeg V2: downloading model from %s", _MODEL_URL)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(_MODEL_URL, str(model_path))

        providers = ort.get_available_providers()
        actual_providers: list[str | tuple[str, dict[str, Any]]] = []
        if "CoreMLExecutionProvider" in providers:
            actual_providers.append(
                (
                    "CoreMLExecutionProvider",
                    {"MLComputeUnits": "ALL", "convert_model_to_fp16": True},
                )
            )
        if "CUDAExecutionProvider" in providers:
            actual_providers.append("CUDAExecutionProvider")
        actual_providers.append("CPUExecutionProvider")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            str(model_path), providers=actual_providers, sess_options=sess_options
        )
        self._input_name = self._session.get_inputs()[0].name

    def infer(self, frame: np.ndarray) -> np.ndarray:
        """Run high-performance inference on a single RGB frame."""
        if self._session is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        h, w = frame.shape[:2]

        # 1. Fast Resize (direct resize is faster than letterboxing)
        frame_resized = cv2.resize(frame, (_INPUT_W, _INPUT_H), interpolation=cv2.INTER_LINEAR)

        # 2. Fast Normalization (vectorized)
        tensor = (frame_resized.astype(np.float32) - 127.5) * (1.0 / 127.5)
        tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis]  # (1, 3, 192, 192)

        # 3. GPU Inference (CoreML/MPS)
        output = self._session.run(None, {self._input_name: tensor})
        logits = output[0][0]  # (C, H, W)

        # 4. Fast Sigmoid/Softmax
        if logits.shape[0] == 2:
            mask_low = 1.0 / (1.0 + np.exp(-(logits[1] - logits[0])))
        else:
            mask_low = 1.0 / (1.0 + np.exp(-logits[0]))

        # 5. Fast Post-processing
        mask_up = cv2.resize(mask_low, (w, h), interpolation=cv2.INTER_LINEAR)

        # Soft contrast stretch + clip
        return np.clip((mask_up - 0.45) * 10.0, 0.0, 1.0).astype(np.float32)
