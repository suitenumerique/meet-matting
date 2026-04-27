"""
Wrapper pour le modèle MediaPipe Selfie Segmenter (Portrait et Landscape).
"""

import logging
import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from .base import BaseModelWrapper

logger = logging.getLogger(__name__)

os.environ["GLOG_minloglevel"] = "2"

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


class BaseMediapipeWrapper(BaseModelWrapper):
    _variant: str = ""
    _segmenter = None
    _frame_count = 0

    @property
    def input_size(self) -> tuple[int, int] | None:
        return (256, 256)

    def load(self) -> None:
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

        weights_dir = Path(__file__).parent.parent / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)

        model_filename = _MODEL_URLS[self._variant].split("/")[-1]
        local_path = weights_dir / model_filename

        if not local_path.exists() or local_path.stat().st_size == 0:
            urllib.request.urlretrieve(_MODEL_URLS[self._variant], str(local_path))

        options = ImageSegmenterOptions(
            base_options=BaseOptions(
                model_asset_path=str(local_path), delegate=BaseOptions.Delegate.GPU
            ),
            running_mode=RunningMode.VIDEO,
            output_category_mask=False,
            output_confidence_masks=True,
        )
        self._segmenter = ImageSegmenter.create_from_options(options)
        self._mp = mp
        self._frame_count = 0

    def reset_state(self) -> None:
        self._frame_count = 0
        if self._segmenter is not None:
            try:
                self._segmenter.close()
            except Exception:
                pass
            self._segmenter = None
            self.load()

    def get_flops(self, input_shape: tuple[int, int, int] = (3, 256, 256)) -> float:
        estimates = {"portrait": 7.5e6, "selfie_multiclass": 9.2e6, "landscape": 8.1e6}
        return estimates.get(self._variant, 8.0e6)

    def predict(self, frame_bgr: np.ndarray, frame_rgb: np.ndarray | None = None) -> np.ndarray:
        if self._segmenter is None:
            raise RuntimeError(
                f"MediaPipe ({self._variant}): modèle non chargé. Appelle load() d'abord."
            )
        try:
            h_orig, w_orig = frame_bgr.shape[:2]
            frame_small = cv2.resize(frame_bgr, (256, 256), interpolation=cv2.INTER_NEAREST)
            frame_small_rgba = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGBA)

            mp_image = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGBA, data=frame_small_rgba
            )
            ts = int(self._frame_count * (1000 / 30))

            result = self._segmenter.segment_for_video(mp_image, ts)
            self._frame_count += 1

            if result and result.confidence_masks:
                if self._variant == "selfie_multiclass" and len(result.confidence_masks) > 1:
                    mask_small = 1.0 - result.confidence_masks[0].numpy_view()
                else:
                    mask_small = result.confidence_masks[0].numpy_view()

                if mask_small.shape[:2] != (h_orig, w_orig):
                    return cv2.resize(
                        mask_small, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST
                    ).astype(np.float32)
                return mask_small.astype(np.float32)

            return np.zeros((h_orig, w_orig), dtype=np.float32)
        except Exception as e:
            logger.error(f"Erreur interne MediaPipe ({self._variant}): {e}")
            return np.zeros((h_orig, w_orig), dtype=np.float32)

    def cleanup(self) -> None:
        if self._segmenter:
            self._segmenter.close()
        self._segmenter = None


class MediapipePortraitWrapper(BaseMediapipeWrapper):
    _variant = "portrait"

    @property
    def name(self) -> str:
        return "MediaPipe Portrait"


class MediapipeLandscapeWrapper(BaseMediapipeWrapper):
    _variant = "landscape"

    @property
    def name(self) -> str:
        return "MediaPipe Landscape"
