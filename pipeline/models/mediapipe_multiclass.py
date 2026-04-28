"""
MediaPipe Selfie Multiclass model.
Outputs a multiclass segmentation mask (Background, Hair, Body, etc.)
Merged into a single person mask.
"""

import logging
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path

from core.base import MattingModel
from core.parameters import ParameterSpec
from core.registry import models

logger = logging.getLogger(__name__)

@models.register
class MediapipeSelfieMulticlass(MattingModel):
    name = "mediapipe_multiclass"
    description = "MediaPipe Selfie Multiclass Segmenter (Background, Hair, Body, Clothes, Skin)."

    def __init__(self, **params):
        super().__init__(**params)
        self._segmenter = None

    @classmethod
    def parameter_specs(cls):
        return []

    def load(self, weights_path: str | None = None):
        # Resolve absolute path to the model file
        if weights_path:
            path = weights_path
        else:
            path = str(Path(__file__).parent.parent / "weights" / "selfie_multiclass_256x256.tflite")
            
        if not Path(path).exists():
            logger.error(f"Model file not found at: {path}")
            raise FileNotFoundError(f"MediaPipe model not found at {path}")

        base_options = python.BaseOptions(model_asset_path=path)
        options = vision.ImageSegmenterOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            output_category_mask=True,
        )
        self._segmenter = vision.ImageSegmenter.create_from_options(options)
        logger.info(f"Loaded MediaPipe Multiclass from {path}")

    def infer(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Pure inference: Image -> Mask. 
        Scaling/Cropping is now handled globally by the Pipeline.
        """
        if self._segmenter is None:
            self.load()

        # Convert to SRGBA for Metal compatibility on macOS
        frame_rgba = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2RGBA)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=frame_rgba)
        
        result = self._segmenter.segment(mp_image)
        
        # Category mask: 0=bg, 1=hair, 2=body, 3=face, 4=clothes, 5=skin
        mask = result.category_mask.numpy_view()
        # Merge all non-background categories into a person mask
        person_mask = (mask > 0).astype(np.float32)
        
        return person_mask
