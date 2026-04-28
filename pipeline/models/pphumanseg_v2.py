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
        return [
            ParameterSpec(
                name="use_refinement",
                type="bool",
                default=True,
                label="Guided filter refinement",
                help="Refine mask edges with a guided filter (slower but sharper boundaries).",
            ),
            ParameterSpec(
                name="refine_radius",
                type="int",
                default=8,
                label="Refine radius",
                min_value=1,
                max_value=32,
                step=1,
                help="Guided filter radius.",
            ),
            ParameterSpec(
                name="refine_eps",
                type="float",
                default=1e-2,
                label="Refine eps",
                min_value=1e-4,
                max_value=1.0,
                step=1e-4,
                help="Guided filter regularisation term.",
            ),
        ]

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
        """Run inference on a single RGB frame.

        Args:
            frame: RGB image, shape (H, W, 3), dtype uint8.

        Returns:
            Alpha matte, shape (H, W), dtype float32, values in [0, 1].
        """
        if self._session is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        h, w = frame.shape[:2]

        # Letterbox resize to preserve aspect ratio
        scale = min(_INPUT_W / w, _INPUT_H / h)
        nw, nh = int(w * scale), int(h * scale)
        dx, dy = (_INPUT_W - nw) // 2, (_INPUT_H - nh) // 2

        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        frame_resized = cv2.resize(frame, (nw, nh), interpolation=interp)

        canvas = np.full((_INPUT_H, _INPUT_W, 3), 127, dtype=np.uint8)
        canvas[dy : dy + nh, dx : dx + nw] = frame_resized

        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        tensor = (canvas.astype(np.float32) / 255.0 - mean) / std
        tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis]  # (1, 3, H, W)

        output = self._session.run(None, {self._input_name: tensor})
        logits = output[0][0]  # (C, H, W)

        if logits.ndim == 3 and logits.shape[0] >= 2:
            exp_logits = np.exp(logits - logits.max(axis=0, keepdims=True))
            probs = exp_logits / exp_logits.sum(axis=0, keepdims=True)
            mask_padded = probs[1]
        elif logits.ndim == 3 and logits.shape[0] == 1:
            mask_padded = 1.0 / (1.0 + np.exp(-logits[0]))
        else:
            mask_padded = logits.squeeze()

        # Remove letterbox padding, then upsample to original resolution
        mask_low = mask_padded[dy : dy + nh, dx : dx + nw].astype(np.float32)
        if self.upsampler is not None:
            mask_up = self.upsampler.upsample(mask_low, frame)
        else:
            mask_up = cv2.resize(mask_low, (w, h), interpolation=cv2.INTER_LINEAR)

        if self.params["use_refinement"]:
            # frame is RGB but guidedFilter only needs luminance structure — BGR order doesn't matter
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            mask_up = cv2.ximgproc.guidedFilter(
                guide=frame_bgr,
                src=mask_up,
                radius=self.params["refine_radius"],
                eps=self.params["refine_eps"],
            )

        # Soft contrast stretch matching benchmark post-processing
        mask = np.clip((mask_up - 0.45) / (0.55 - 0.45), 0.0, 1.0)
        return mask.astype(np.float32)
