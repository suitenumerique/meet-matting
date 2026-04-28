from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from core.base import MattingModel
from core.parameters import ParameterSpec
from core.registry import models

logger = logging.getLogger(__name__)

_MODEL_URLS = {
    "portrait": (
        "https://storage.googleapis.com/mediapipe-models/"
        "image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"
    ),
    "landscape": (
        "https://storage.googleapis.com/mediapipe-models/"
        "image_segmenter/selfie_segmenter_landscape/float16/latest/"
        "selfie_segmenter_landscape.tflite"
    ),
}


class BaseMediapipeSelfie(MattingModel):
    _variant: str = ""
    _segmenter = None
    _frame_count = 0

    @classmethod
    def parameter_specs(cls):
        return [
            ParameterSpec(
                name="gpu",
                type="bool",
                default=True,
                label="Use GPU",
                help="Use GPU acceleration if available.",
            ),
        ]

    def load(self, weights_path: str | None = None):
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
                "mediapipe est requis. Installe-le via : pip install mediapipe"
            ) from e

        # Resolve weights path
        weights_dir = Path(__file__).parent.parent / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)

        model_filename = _MODEL_URLS[self._variant].split("/")[-1]
        local_path = weights_dir / model_filename

        if not local_path.exists() or local_path.stat().st_size == 0:
            logger.info(f"Downloading MediaPipe {self._variant} model...")
            urllib.request.urlretrieve(_MODEL_URLS[self._variant], str(local_path))

        delegate = (
            BaseOptions.Delegate.GPU if self.params.get("gpu", True) else BaseOptions.Delegate.CPU
        )

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
        if self._segmenter is None:
            self.load()

        try:
            h_orig, w_orig = frame.shape[:2]
            # MediaPipe Selfie expects 256x256
            frame_small = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LINEAR)
            
            # Metal delegate on macOS requires SRGBA (format 2)
            frame_rgba = cv2.cvtColor(frame_small, cv2.COLOR_RGB2RGBA)
            mp_image = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGBA, data=frame_rgba
            )
            
            ts = int(self._frame_count * (1000 / 30))  # Pseudo-timestamp at 30fps
            result = self._segmenter.segment_for_video(mp_image, ts)
            self._frame_count += 1

            if result and result.confidence_masks:
                mask_small = result.confidence_masks[0].numpy_view()

                if mask_small.shape[:2] != (h_orig, w_orig):
                    return cv2.resize(
                        mask_small, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR
                    ).astype(np.float32)
                return mask_small.astype(np.float32)

            return np.zeros((h_orig, w_orig), dtype=np.float32)
        except Exception as e:
            logger.error(f"MediaPipe {self._variant} inference error: {e}")
            return np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)

    def cleanup(self):
        if self._segmenter:
            self._segmenter.close()
            self._segmenter = None


@models.register
class MediapipePortrait(BaseMediapipeSelfie):
    name = "mediapipe_portrait"
    description = "MediaPipe Selfie Segmenter - Portrait (optimisé pour les visages proches)."
    _variant = "portrait"


@models.register
class MediapipeLandscape(BaseMediapipeSelfie):
    name = "mediapipe_landscape"
    description = "MediaPipe Selfie Segmenter - Landscape (champ plus large)."
    _variant = "landscape"
