import urllib.request
from pathlib import Path

import cv2
import numpy as np
from core.base import MattingModel
from core.parameters import ParameterSpec
from core.registry import models

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "image_segmenter/selfie_segmenter_landscape/float16/latest/"
    "selfie_segmenter_landscape.tflite"
)
_WEIGHTS_DIR = Path(__file__).parent.parent / "weights"


@models.register
class MediapipeSelfielandscape(MattingModel):
    name = "mediapipe_selfie_landscape"
    description = "MediaPipe Selfie Landscape — fast binary segmenter optimised for landscape frames, 256×256 TFLite."

    _segmenter = None
    _mp = None
    _frame_count: int = 0

    @classmethod
    def parameter_specs(cls):
        return [
            ParameterSpec(
                name="use_gpu",
                type="bool",
                default=True,
                label="Use GPU delegate",
                help="Enable GPU acceleration for the TFLite model.",
            ),
        ]

    def load(self, weights_path=None):
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import (
                ImageSegmenter,
                ImageSegmenterOptions,
                RunningMode,
            )
        except ImportError as e:
            raise ImportError(
                "mediapipe is required. Install it with: pip install mediapipe"
            ) from e

        _WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        local_path = _WEIGHTS_DIR / _MODEL_URL.split("/")[-1]

        if not local_path.exists() or local_path.stat().st_size == 0:
            urllib.request.urlretrieve(_MODEL_URL, str(local_path))

        delegate = BaseOptions.Delegate.GPU if self.params["use_gpu"] else BaseOptions.Delegate.CPU
        options = ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=str(local_path), delegate=delegate),
            running_mode=RunningMode.VIDEO,
            output_category_mask=False,
            output_confidence_masks=True,
        )
        self._segmenter = ImageSegmenter.create_from_options(options)
        self._mp = mp
        self._frame_count = 0

    def infer(self, frame: np.ndarray) -> np.ndarray:
        """Run inference on a single RGB frame.

        Args:
            frame: RGB image, shape (H, W, 3), dtype uint8.

        Returns:
            Alpha matte, shape (H, W), dtype float32, values in [0, 1].
        """
        if self._segmenter is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        h, w = frame.shape[:2]
        frame_small = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_NEAREST)
        frame_rgba = cv2.cvtColor(frame_small, cv2.COLOR_RGB2RGBA)

        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGBA, data=frame_rgba)
        ts = int(self._frame_count * (1000 / 30))
        result = self._segmenter.segment_for_video(mp_image, ts)
        self._frame_count += 1

        if result and result.confidence_masks:
            mask_small = result.confidence_masks[0].numpy_view()
        else:
            return np.zeros((h, w), dtype=np.float32)

        if mask_small.shape[:2] != (h, w):
            mask_small = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_LINEAR)

        return mask_small.astype(np.float32)
