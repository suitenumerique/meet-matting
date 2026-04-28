"""
MediaPipe Selfie Segmenter models (Portrait & Landscape).
Refactored to be 'pure' models compatible with the Person Zoom pipeline.
Uses IMAGE mode to allow independent crop processing.
"""

import logging
import urllib.request
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import ImageSegmenter, ImageSegmenterOptions, RunningMode

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

    @classmethod
    def parameter_specs(cls):
        return [
            ParameterSpec(
                name="gpu",
                type="bool",
                default=True,
                label="Use GPU",
                help="Use GPU acceleration (Metal on macOS).",
            ),
        ]

    def reset(self):
        """Reset state (none for this model)."""
        pass

    def load(self, weights_path: str | None = None):
        # Resolve weights path
        weights_dir = Path(__file__).parent.parent / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)

        model_filename = _MODEL_URLS[self._variant].split("/")[-1]
        local_path = weights_dir / model_filename

        if not local_path.exists() or local_path.stat().st_size == 0:
            logger.info(f"Downloading MediaPipe {self._variant} model...")
            urllib.request.urlretrieve(_MODEL_URLS[self._variant], str(local_path))

        delegate = BaseOptions.Delegate.GPU if self.params.get("gpu", True) else BaseOptions.Delegate.CPU

        # CRITICAL: Use IMAGE mode for Person Zoom compatibility.
        # VIDEO mode's temporal memory fails when processing multiple crops per frame.
        options = ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=str(local_path), delegate=delegate),
            running_mode=RunningMode.IMAGE,
            output_category_mask=False,
            output_confidence_masks=True,
        )
        self._segmenter = ImageSegmenter.create_from_options(options)
        logger.info(f"Loaded MediaPipe Selfie {self._variant} in IMAGE mode.")

    def infer(self, frame: np.ndarray) -> np.ndarray:
        if self._segmenter is None:
            self.load()

        try:
            h_orig, w_orig = frame.shape[:2]
            
            # MediaPipe expects SRGBA for Metal/GPU
            frame_rgba = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=frame_rgba)
            
            # Inference on single image (crop or full frame)
            result = self._segmenter.segment(mp_image)

            if result and result.confidence_masks:
                # Get the alpha mask
                mask = result.confidence_masks[0].numpy_view()
                
                # Resize to original input size (important for crops!)
                if mask.shape[:2] != (h_orig, w_orig):
                    mask = cv2.resize(mask, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
                
                return mask.astype(np.float32)

            return np.zeros((h_orig, w_orig), dtype=np.float32)
        except Exception as e:
            logger.error(f"MediaPipe {self._variant} inference error: {e}")
            return np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)

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
