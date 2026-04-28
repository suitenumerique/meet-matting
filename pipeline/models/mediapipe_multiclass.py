"""
MediaPipe Selfie Multiclass model.
Outputs a multiclass segmentation mask (Background, Hair, Body, etc.)
Merged into a single person mask.
Force GPU/Metal for performance.
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

    def reset(self):
        """Reset state (none for this model)."""
        pass

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
            ParameterSpec(
                name="include_hair",
                type="bool",
                default=True,
                label="Inclure Cheveux",
                help="Inclure les cheveux dans le masque final.",
            ),
            ParameterSpec(
                name="include_body_skin",
                type="bool",
                default=True,
                label="Inclure Peau (Corps)",
                help="Inclure la peau du corps dans le masque final.",
            ),
            ParameterSpec(
                name="include_face_skin",
                type="bool",
                default=True,
                label="Inclure Peau (Visage)",
                help="Inclure la peau du visage dans le masque final.",
            ),
            ParameterSpec(
                name="include_clothes",
                type="bool",
                default=True,
                label="Inclure Vêtements",
                help="Inclure les vêtements dans le masque final.",
            ),
            ParameterSpec(
                name="include_others",
                type="bool",
                default=True,
                label="Inclure Autres (Accessoires)",
                help="Inclure les accessoires et autres éléments dans le masque final.",
            ),
        ]

    def load(self, weights_path: str | None = None):
        # Resolve absolute path to the model file
        if weights_path:
            path = weights_path
        else:
            path = str(Path(__file__).parent.parent / "weights" / "selfie_multiclass_256x256.tflite")
            
        if not Path(path).exists():
            logger.error(f"Model file not found at: {path}")
            raise FileNotFoundError(f"MediaPipe model not found at {path}")

        # Configure GPU delegate
        delegate = python.BaseOptions.Delegate.GPU if self.params.get("gpu", True) else python.BaseOptions.Delegate.CPU
        
        base_options = python.BaseOptions(
            model_asset_path=path,
            delegate=delegate
        )
        
        options = vision.ImageSegmenterOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            output_category_mask=True,
        )
        self._segmenter = vision.ImageSegmenter.create_from_options(options)
        logger.info(f"Loaded MediaPipe Multiclass from {path} (Delegate: {delegate.name})")

    def infer(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Pure inference: Image -> Mask."""
        if self._segmenter is None:
            self.load()

        # Convert to SRGBA for Metal compatibility on macOS
        frame_rgba = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2RGBA)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=frame_rgba)
        
        result = self._segmenter.segment(mp_image)
        
        # Category mask values:
        # 0: bg, 1: hair, 2: body-skin, 3: face-skin, 4: clothes, 5: others
        mask = result.category_mask.numpy_view()
        
        # Build set of allowed classes
        allowed = []
        if self.params.get("include_hair", True): allowed.append(1)
        if self.params.get("include_body_skin", True): allowed.append(2)
        if self.params.get("include_face_skin", True): allowed.append(3)
        if self.params.get("include_clothes", True): allowed.append(4)
        if self.params.get("include_others", True): allowed.append(5)
        
        # Merge selected categories into a person mask
        if not allowed:
            person_mask = np.zeros_like(mask, dtype=np.float32)
        else:
            person_mask = np.isin(mask, allowed).astype(np.float32)
        
        return person_mask
