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
            output_category_mask=False,
            output_confidence_masks=True,
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
        
        # Build list of allowed indices
        # 0: bg, 1: hair, 2: body-skin, 3: face-skin, 4: clothes, 5: others
        allowed_indices = []
        if self.params.get("include_hair", True): allowed_indices.append(1)
        if self.params.get("include_body_skin", True): allowed_indices.append(2)
        if self.params.get("include_face_skin", True): allowed_indices.append(3)
        if self.params.get("include_clothes", True): allowed_indices.append(4)
        if self.params.get("include_others", True): allowed_indices.append(5)
        
        h, w = frame_rgb.shape[:2]
        if not allowed_indices or not result.confidence_masks:
            return np.zeros((h, w), dtype=np.float32)
            
        # Sum confidence masks of all selected categories
        person_mask = np.zeros((h, w), dtype=np.float32)
        for idx in allowed_indices:
            # Add the probability of this class (squeeze to remove trailing 1 if present)
            mask_data = result.confidence_masks[idx].numpy_view()
            if mask_data.ndim == 3:
                mask_data = mask_data.squeeze(-1)
            person_mask += mask_data
            
        # Clip to [0, 1] to avoid values > 1 at boundaries (though they should sum to 1 theoretically)
        person_mask = np.clip(person_mask, 0, 1)
        
        return person_mask
