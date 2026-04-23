"""
Wrapper for EfficientViT (segmentation).

EfficientViT is a lightweight Vision Transformer architecture, optimised
for fast inference on GPUs and edge devices.

This wrapper uses ONNX Runtime. The ONNX model must be exported locally
from the official repo (no pre-exported ONNX is published).

Code:    https://github.com/mit-han-lab/efficientvit
Weights: https://huggingface.co/han-cai/efficientvit-seg (PyTorch .pt)

Export (B1 ADE20K, 512x512):
    python assets/onnx_export.py \\
        --export_path <repo>/benchmark/weights/efficientvit_seg.onnx \\
        --task seg --model efficientvit-seg-b1-ade20k \\
        --resolution 512 512 --bs 1
"""

import logging
from pathlib import Path
import cv2
import numpy as np
from typing import List, Optional, Tuple

from .base import BaseModelWrapper

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_PATH = (
    Path(__file__).parent.parent / "weights" / "efficientvit_seg.onnx"
)


class EfficientViTWrapper(BaseModelWrapper):
    """
    EfficientViT Segmentation via ONNX Runtime.

    Expected input: (1, 3, 512, 512) normalised to ImageNet stats.
    Output: segmentation logits (1, num_classes, H, W).
    """

    _INPUT_SIZE = 512
    # In ADE20K, 'person' is index 12
    PERSON_CLASS_INDEX = 12

    def __init__(self, model_path: Optional[str] = None):
        self._model_path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
        self._session = None
        self._input_name = None

    @property
    def name(self) -> str:
        return "EfficientViT"

    @property
    def input_size(self) -> Optional[Tuple[int, int]]:
        return (self._INPUT_SIZE, self._INPUT_SIZE)

    def load(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "onnxruntime is required. Install it via: pip install onnxruntime"
            ) from e

        if not self._model_path.exists():
            logger.error(
                "EfficientViT: ONNX model not found at %s. "
                "Export it from the official repo: "
                "https://github.com/mit-han-lab/efficientvit (see module docstring).",
                self._model_path,
            )
            self._session = None
            return

        providers = ort.get_available_providers()
        selected_providers = []
        if "CoreMLExecutionProvider" in providers:
            selected_providers.append("CoreMLExecutionProvider")
        if "CUDAExecutionProvider" in providers:
            selected_providers.append("CUDAExecutionProvider")
        selected_providers.append("CPUExecutionProvider")

        try:
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
            logger.info("EfficientViT: ONNX model loaded (%s).", self._model_path.name)
        except Exception as e:
            logger.error("EfficientViT: error while creating the session: %s", e)
            if "INVALID_PROTOBUF" in str(e) or "Protobuf parsing failed" in str(e):
                logger.warning("EfficientViT: file appears corrupted. Removing %s for a future re-download.", self._model_path)
                if self._model_path.exists():
                    self._model_path.unlink()
            self._session = None

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        return self.predict_batch([frame_bgr])[0]

    def predict_batch(self, frames_bgr: List[np.ndarray]) -> List[np.ndarray]:
        if not frames_bgr:
            return []

        h_orig, w_orig = frames_bgr[0].shape[:2]

        if self._session is None:
            logger.warning("EfficientViT: no model loaded, returning empty masks.")
            return [np.zeros((h_orig, w_orig), dtype=np.float32) for _ in frames_bgr]

        batch_size = len(frames_bgr)

        # Determine the expected input shape
        input_meta = self._session.get_inputs()[0]
        input_shape = input_meta.shape
        logger.info("EfficientViT: input metadata - Shape: %s, Name: %s", input_shape, input_meta.name)

        # Try to guess the channel placement (usually 1 for NCHW, 3 for NHWC)
        if len(input_shape) == 4:
            if isinstance(input_shape[1], int) and input_shape[1] in [3, 4]:
                expected_channels = input_shape[1]
                layout = "NCHW"
            elif isinstance(input_shape[3], int) and input_shape[3] in [3, 4]:
                expected_channels = input_shape[3]
                layout = "NHWC"
            else:
                expected_channels = 3 # fallback
                layout = "NCHW"
        else:
            expected_channels = 3
            layout = "NCHW"

        logger.info("EfficientViT: detected layout: %s, expected channels: %d", layout, expected_channels)

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        tensors = []

        for frame in frames_bgr:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (self._INPUT_SIZE, self._INPUT_SIZE))
            img = (frame_resized.astype(np.float32) / 255.0 - mean) / std

            if expected_channels == 4:
                # Add an alpha channel (zeros)
                alpha = np.zeros((self._INPUT_SIZE, self._INPUT_SIZE, 1), dtype=np.float32)
                img = np.concatenate([img, alpha], axis=-1)

            if layout == "NCHW":
                tensor = np.transpose(img, (2, 0, 1))
            else:
                tensor = img

            tensors.append(tensor)

        batch_tensor = np.stack(tensors, axis=0)

        # Inference with automatic fallback logic (self-healing)
        try:
            output = self._session.run(None, {self._input_name: batch_tensor})
            logits_batch = output[0]
        except Exception as e:
            error_msg = str(e)
            # If the error indicates a channel mismatch (C: 3 instead of 4)
            if "channels C" in error_msg and "not equal to kernel channels" in error_msg:
                logger.info("EfficientViT: detected a 4-channel model. Transforming inputs...")
                # Rebuild the batch with 4 channels
                new_tensors = []
                for t in tensors: # t is (3, H, W)
                    # Convert to (H, W, 3)
                    img_3 = np.transpose(t, (1, 2, 0))
                    # Add alpha channel
                    alpha = np.zeros((self._INPUT_SIZE, self._INPUT_SIZE, 1), dtype=np.float32)
                    img_4 = np.concatenate([img_3, alpha], axis=-1)
                    # Back to (4, H, W)
                    new_tensors.append(np.transpose(img_4, (2, 0, 1)))

                batch_tensor = np.stack(new_tensors, axis=0)
                # Second attempt
                output = self._session.run(None, {self._input_name: batch_tensor})
                logits_batch = output[0]

            elif "Got: " in error_msg and "Expected: 1" in error_msg:
                logger.debug("EfficientViT: batching not supported by the model, falling back to iteration.")
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

            # Dynamic detection of the output format
            if l.ndim == 3: # Format (C, H, W) - segmentation
                num_classes = l.shape[0]
                # If it's a single class (sigmoid)
                if num_classes == 1:
                    mask = 1.0 / (1.0 + np.exp(-l[0]))
                else:
                    # Find the person class. If index 15 is out of bounds,
                    # take the max, or index 1 for Cityscapes
                    idx = self.PERSON_CLASS_INDEX if num_classes > self.PERSON_CLASS_INDEX else (1 if num_classes > 1 else 0)

                    # Softmax over the classes
                    exp_l = np.exp(l - l.max(axis=0, keepdims=True))
                    probs = exp_l / exp_l.sum(axis=0, keepdims=True)
                    mask = probs[idx]
            elif l.ndim == 1: # Format (C,) - error, this is classification
                logger.error("EfficientViT: the loaded model is a classification model, not a segmentation one.")
                mask = np.zeros((self._INPUT_SIZE, self._INPUT_SIZE), dtype=np.float32)
            else:
                mask = l.squeeze()
                if mask.ndim != 2:
                    mask = np.zeros((self._INPUT_SIZE, self._INPUT_SIZE), dtype=np.float32)

            # Post-processing: resize back to the original size
            if mask.shape != (h_orig, w_orig):
                mask = cv2.resize(mask, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

            masks.append(mask.astype(np.float32))

        return masks

    def get_flops(self, input_shape: Tuple[int, int, int] = (3, 512, 512)) -> float:
        # EfficientViT-L1: ~5 GFLOPs at 512x512
        c, h, w = input_shape
        base_flops = 5e9
        scale = (h * w) / (512 * 512)
        return base_flops * scale

    def cleanup(self) -> None:
        self._session = None
        logger.info("EfficientViT: ONNX session closed.")

