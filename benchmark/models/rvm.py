"""
Wrapper for Robust Video Matting (RVM).

RVM is a recurrent (GRU-based) model that maintains an internal state
between frames to produce temporally coherent alpha mattes.

This wrapper supports inference via:
  - ONNX Runtime (recommended for the benchmark, more portable)
  - Native PyTorch (fallback)

Model: https://github.com/PeterL1n/RobustVideoMatting
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .base import BaseModelWrapper

logger = logging.getLogger(__name__)

# Default local path for the ONNX model
_DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "weights" / "rvm_mobilenetv3_fp32.onnx"
_MODEL_URL = (
    "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/"
    "rvm_mobilenetv3_fp32.onnx"
)


class RVMWrapper(BaseModelWrapper):
    """
    Robust Video Matting via ONNX Runtime.

    Maintains recurrent states (r1..r4) between frames.
    reset_state() must be called at the start of each new video.
    """

    def __init__(self, model_path: Optional[str] = None, downsample_ratio: float = 0.25):
        self._model_path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
        self._downsample_ratio = downsample_ratio
        self._session = None
        self._recurrent_state = None

    @property
    def name(self) -> str:
        return "RVM (MobileNetV3)"

    @property
    def input_size(self) -> Optional[Tuple[int, int]]:
        return None  # Dynamic, depends on the video

    def load(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "onnxruntime is required. Install it via: pip install onnxruntime"
            ) from e

        # Download if missing
        if not self._model_path.exists():
            logger.info("RVM: downloading model from %s", _MODEL_URL)
            self._download_model()

        providers = ort.get_available_providers()
        # Prioritise hardware accelerators (CoreML on Mac, CUDA on NVIDIA)
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
        self.reset_state()
        logger.info("RVM: ONNX model loaded (%s).", self._model_path.name)

    def _download_model(self) -> None:
        """Download the ONNX model from GitHub."""
        import urllib.request

        self._model_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("RVM: downloading to %s …", self._model_path)
        urllib.request.urlretrieve(_MODEL_URL, str(self._model_path))
        logger.info("RVM: download complete.")

    def reset_state(self) -> None:
        """Reset the recurrent GRU states between videos."""
        self._recurrent_state = {
            "r1i": np.zeros((1, 1, 1, 1), dtype=np.float32),
            "r2i": np.zeros((1, 1, 1, 1), dtype=np.float32),
            "r3i": np.zeros((1, 1, 1, 1), dtype=np.float32),
            "r4i": np.zeros((1, 1, 1, 1), dtype=np.float32),
        }

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self._session is None:
            raise RuntimeError("RVM: model not loaded. Call load() first.")

        h_orig, w_orig = frame_bgr.shape[:2]

        # Pre-processing: BGR -> RGB, HWC -> NCHW, normalise [0,1]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = frame_rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))  # CHW
        tensor = np.expand_dims(tensor, axis=0)     # NCHW

        # Inference
        inputs = {
            "src": tensor,
            "r1i": self._recurrent_state["r1i"],
            "r2i": self._recurrent_state["r2i"],
            "r3i": self._recurrent_state["r3i"],
            "r4i": self._recurrent_state["r4i"],
            "downsample_ratio": np.array([self._downsample_ratio], dtype=np.float32),
        }

        outputs = self._session.run(None, inputs)

        # Outputs: [fgr, pha, r1o, r2o, r3o, r4o]
        alpha = outputs[1]  # (1, 1, H, W)

        # Update the recurrent state
        self._recurrent_state["r1i"] = outputs[2]
        self._recurrent_state["r2i"] = outputs[3]
        self._recurrent_state["r3i"] = outputs[4]
        self._recurrent_state["r4i"] = outputs[5]

        # Post-processing
        mask = alpha[0, 0]  # (H, W)
        mask = np.clip(mask, 0.0, 1.0)

        if mask.shape != (h_orig, w_orig):
            mask = cv2.resize(mask, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

        return mask.astype(np.float32)

    def get_flops(self, input_shape: Tuple[int, int, int] = (3, 256, 256)) -> float:
        # RVM MobileNetV3: ~600 MFLOPs at 256x256
        c, h, w = input_shape
        base_flops = 600e6  # for 256x256
        scale = (h * w) / (256 * 256)
        return base_flops * scale

    def cleanup(self) -> None:
        self._session = None
        self._recurrent_state = None
        logger.info("RVM: ONNX session closed.")
