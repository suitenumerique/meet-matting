"""
Wrapper pour le modèle MediaPipe Selfie Multiclass.
"""

import logging
import urllib.request
from pathlib import Path

from .mediapipe_selfie import BaseMediapipeWrapper

logger = logging.getLogger(__name__)

_MODEL_URLS = {
    "selfie_multiclass": (
        "https://storage.googleapis.com/mediapipe-models/"
        "image_segmenter/selfie_multiclass_256x256/float32/latest/"
        "selfie_multiclass_256x256.tflite"
    ),
}


class MediapipeSelfieMulticlassWrapper(BaseMediapipeWrapper):
    _variant = "selfie_multiclass"

    @property
    def name(self) -> str:
        """Return the model name."""
        return "MediaPipe Selfie Multiclass"

    def load(self) -> None:
        """Download weights if needed and initialise the inference session."""
        # Override load pour utiliser les bons URLs localement si besoin,
        # ou on injecte l'URL dans la classe parente.
        # Pour simplifier, on duplique la logique de chargement ou on l'adapte.
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
