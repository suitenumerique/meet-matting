"""
Wrappers for the MediaPipe segmentation models.

Three configurations:
  - Portrait Segmenter     : optimised for front-facing portraits.
  - Selfie Multiclass      : segments into several classes (skin, hair, clothing…).
  - Landscape Segmenter    : segments the subject in wide shots.

Each wrapper uses the official MediaPipe Python API.
Models are downloaded automatically if absent.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .base import BaseModelWrapper

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
#  MediaPipe model URLs
# ──────────────────────────────────────────────
_MODEL_URLS = {
    "portrait": (
        "https://storage.googleapis.com/mediapipe-models/"
        "image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"
    ),
    "selfie_multiclass": (
        "https://storage.googleapis.com/mediapipe-models/"
        "image_segmenter/selfie_multiclass_256x256/float32/latest/"
        "selfie_multiclass_256x256.tflite"
    ),
    "landscape": (
        "https://storage.googleapis.com/mediapipe-models/"
        "image_segmenter/selfie_segmenter_landscape/float16/latest/"
        "selfie_segmenter_landscape.tflite"
    ),
}


class _BaseMediapipeWrapper(BaseModelWrapper):
    """
    Internal common class for the 3 MediaPipe variants.

    Uses mediapipe.tasks.vision.ImageSegmenter.
    """

    _variant: str = ""  # "portrait", "selfie_multiclass", "landscape"
    _segmenter = None

    @property
    def input_size(self) -> Optional[Tuple[int, int]]:
        return (256, 256)

    def load(self) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import (
                ImageSegmenter,
                ImageSegmenterOptions,
            )
        except ImportError as e:
            raise ImportError(
                "mediapipe is required. Install it via: pip install mediapipe"
            ) from e

        # Resolve the local path
        weights_dir = Path(__file__).parent.parent / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        model_filename = _MODEL_URLS[self._variant].split("/")[-1]
        local_path = weights_dir / model_filename

        # Download if missing or empty
        if not local_path.exists() or local_path.stat().st_size == 0:
            import urllib.request
            logger.info("%s: downloading from %s", self.name, _MODEL_URLS[self._variant])
            try:
                urllib.request.urlretrieve(_MODEL_URLS[self._variant], str(local_path))
            except Exception as e:
                logger.error("%s: download error: %s", self.name, e)
                raise RuntimeError(f"Failed to download model {self.name} from {_MODEL_URLS[self._variant]}: {e}")

        if not local_path.exists() or local_path.stat().st_size == 0:
            raise RuntimeError(f"Model file is missing or empty after download: {local_path}")

        # Initialise MediaPipe with the local path
        try:
            options = ImageSegmenterOptions(
                base_options=BaseOptions(model_asset_path=str(local_path)),
                output_category_mask=True,
            )
            self._segmenter = ImageSegmenter.create_from_options(options)
            self._mp = mp
            logger.info("%s: model loaded successfully from %s.", self.name, local_path)
        except Exception as e:
            logger.error("%s: MediaPipe initialisation error: %s", self.name, e)
            raise RuntimeError(f"MediaPipe initialisation error for {self.name}: {e}")

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self._segmenter is None:
            raise RuntimeError(f"{self.name}: model not loaded. Call load() first.")

        if frame_bgr is None or frame_bgr.size == 0:
            logger.warning("%s: invalid or empty frame received.", self.name)
            # Return a fallback empty mask (using h_orig/w_orig if available)
            try:
                h, w = frame_bgr.shape[:2]
                return np.zeros((h, w), dtype=np.float32)
            except:
                return np.zeros((256, 256), dtype=np.float32)

        h_orig, w_orig = frame_bgr.shape[:2]

        # Robust BGR(A) -> RGB conversion
        if frame_bgr.shape[2] == 4:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGRA2RGB)
        else:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=frame_rgb,
        )

        result = self._segmenter.segment(mp_image)

        if result.category_mask is not None:
            mask = result.category_mask.numpy_view().astype(np.float32)

            if self._variant == "selfie_multiclass":
                # For multiclass, 0 is background and 1-5 are the person
                mask = (mask > 0).astype(np.float32)
            else:
                # For Portrait/Landscape, here 0 appears to be the person
                mask = (mask == 0).astype(np.float32)
        elif result.confidence_masks:
            # Take the "person" class mask (index 1 in multiclass)
            idx = 1 if len(result.confidence_masks) > 1 else 0
            mask = result.confidence_masks[idx].numpy_view().astype(np.float32)
        else:
            logger.warning("%s: no mask returned, returning empty mask.", self.name)
            mask = np.zeros((h_orig, w_orig), dtype=np.float32)

        # Resize back to the original size
        if mask.shape[:2] != (h_orig, w_orig):
            mask = cv2.resize(mask, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

        return mask

    def get_flops(self, input_shape: Tuple[int, int, int] = (3, 256, 256)) -> float:
        # MediaPipe TFLite: no direct FLOPs counting.
        # Estimated values taken from the documentation:
        estimates = {
            "portrait": 7.5e6,        # ~7.5 MFLOPs
            "selfie_multiclass": 9.2e6,  # ~9.2 MFLOPs
            "landscape": 8.1e6,        # ~8.1 MFLOPs
        }
        return estimates.get(self._variant, -1.0)

    def cleanup(self) -> None:
        if self._segmenter is not None:
            self._segmenter.close()
            self._segmenter = None
        logger.info("%s: resources released.", self.name)


# ──────────────────────────────────────────────
#  Concrete variants
# ──────────────────────────────────────────────
class MediapipePortraitWrapper(_BaseMediapipeWrapper):
    _variant = "portrait"

    @property
    def name(self) -> str:
        return "MediaPipe Portrait"


class MediapipeSelfieMulticlassWrapper(_BaseMediapipeWrapper):
    _variant = "selfie_multiclass"

    @property
    def name(self) -> str:
        return "MediaPipe Selfie Multiclass"


class MediapipeLandscapeWrapper(_BaseMediapipeWrapper):
    _variant = "landscape"

    @property
    def name(self) -> str:
        return "MediaPipe Landscape"
