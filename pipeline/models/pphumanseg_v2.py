"""PP-HumanSeg V2 pipeline model wrapper — lightweight ONNX-based human segmenter at 192×192."""

import logging
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from core.base import MattingModel
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
        """Return the list of tunable parameters for this component."""
        return []

    def load(self, weights_path=None):
        """Download weights if needed and initialise the inference session."""
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
        """Run inference on a single frame.

        This model expects 192x192 float32 input with normalization (mean=0.5, std=0.5).
        It handles its own resizing and upsampling to be self-contained.
        """
        if self._session is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        h_orig, w_orig = frame.shape[:2]
        target_h, target_w = _INPUT_H, _INPUT_W

        # 1. Pre-processing: Letterbox resize to target size
        scale = min(target_w / w_orig, target_h / h_orig)
        nw, nh = int(w_orig * scale), int(h_orig * scale)
        dx, dy = (target_w - nw) // 2, (target_h - nh) // 2

        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        frame_resized = cv2.resize(frame, (nw, nh), interpolation=interp)

        canvas = np.full((target_h, target_w, 3), 127, dtype=np.uint8)
        canvas[dy : dy + nh, dx : dx + nw] = frame_resized

        # 2. Normalization & Tensor conversion
        tensor = (canvas.astype(np.float32) / 255.0 - 0.5) / 0.5
        tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis]

        # 3. Inference
        output = self._session.run(None, {self._input_name: tensor})
        logits = output[0][0]  # (C, H, W)

        # Handling different model output formats (Softmax vs Sigmoid)
        if logits.ndim == 3 and logits.shape[0] >= 2:
            exp_logits = np.exp(logits - logits.max(axis=0, keepdims=True))
            probs = exp_logits / exp_logits.sum(axis=0, keepdims=True)
            mask = probs[1]
        elif logits.ndim == 3 and logits.shape[0] == 1:
            mask = 1.0 / (1.0 + np.exp(-logits[0]))
        else:
            mask = logits.squeeze()

        # 4. Post-processing: Remove letterbox padding, then upsample to original resolution
        mask_valid = mask[dy : dy + nh, dx : dx + nw].astype(np.float32)
        upsampled = self._apply_upsampler(mask_valid, frame)
        if upsampled.shape[:2] != (h_orig, w_orig):
            upsampled = cv2.resize(upsampled, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

        return upsampled.astype(np.float32)
