"""
Wrappers pour les modèles MediaPipe de segmentation.

Trois configurations :
  - Portrait Segmenter     : optimisé pour les portraits de face.
  - Selfie Multiclass      : segmente en plusieurs classes (peau, cheveux, vêtements…).
  - Landscape Segmenter    : segmente le sujet dans des plans larges.

Chaque wrapper utilise l'API Python officielle de MediaPipe.
Les modèles sont téléchargés automatiquement si absents.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .base import BaseModelWrapper

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
#  URLs des modèles MediaPipe
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
    Classe interne commune aux 3 variantes MediaPipe.

    Utilise mediapipe.tasks.vision.ImageSegmenter.
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
                "mediapipe est requis. Installe-le via : pip install mediapipe"
            ) from e

        # Déterminer le chemin local
        weights_dir = Path(__file__).parent.parent / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        model_filename = _MODEL_URLS[self._variant].split("/")[-1]
        local_path = weights_dir / model_filename

        # Télécharger si absent ou vide
        if not local_path.exists() or local_path.stat().st_size == 0:
            import urllib.request
            logger.info("%s: téléchargement depuis %s", self.name, _MODEL_URLS[self._variant])
            try:
                urllib.request.urlretrieve(_MODEL_URLS[self._variant], str(local_path))
            except Exception as e:
                logger.error("%s: Erreur de téléchargement : %s", self.name, e)
                raise RuntimeError(f"Échec du téléchargement du modèle {self.name} de {_MODEL_URLS[self._variant]}: {e}")

        if not local_path.exists() or local_path.stat().st_size == 0:
            raise RuntimeError(f"Le fichier du modèle est manquant ou vide après téléchargement : {local_path}")

        # Initialisation MediaPipe avec le chemin local
        try:
            options = ImageSegmenterOptions(
                base_options=BaseOptions(model_asset_path=str(local_path)),
                output_category_mask=True,
            )
            self._segmenter = ImageSegmenter.create_from_options(options)
            self._mp = mp
            logger.info("%s: modèle chargé avec succès depuis %s.", self.name, local_path)
        except Exception as e:
            logger.error("%s: Erreur d'initialisation MediaPipe : %s", self.name, e)
            raise RuntimeError(f"Erreur d'initialisation MediaPipe pour {self.name}: {e}")

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self._segmenter is None:
            raise RuntimeError(f"{self.name}: modèle non chargé. Appelle load() d'abord.")

        if frame_bgr is None or frame_bgr.size == 0:
            logger.warning("%s: frame invalide ou vide reçue.", self.name)
            # Retourner un masque vide de secours (en utilisant h_orig/w_orig s'ils existent)
            try:
                h, w = frame_bgr.shape[:2]
                return np.zeros((h, w), dtype=np.float32)
            except:
                return np.zeros((256, 256), dtype=np.float32)

        h_orig, w_orig = frame_bgr.shape[:2]
        
        # Conversion robuste BGR(A) -> RGB
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
                # Pour le multiclass, 0 est le décor, 1-5 sont la personne
                mask = (mask > 0).astype(np.float32)
            else:
                # Pour Portrait/Landscape, chez vous 0 semble être la personne
                mask = (mask == 0).astype(np.float32)
        elif result.confidence_masks:
            # Prendre le masque de la classe "personne" (index 1 en multiclass)
            idx = 1 if len(result.confidence_masks) > 1 else 0
            mask = result.confidence_masks[idx].numpy_view().astype(np.float32)
        else:
            logger.warning("%s: aucun masque retourné, renvoi masque vide.", self.name)
            mask = np.zeros((h_orig, w_orig), dtype=np.float32)

        # Resize vers la taille originale
        if mask.shape[:2] != (h_orig, w_orig):
            mask = cv2.resize(mask, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

        return mask

    def get_flops(self, input_shape: Tuple[int, int, int] = (3, 256, 256)) -> float:
        # MediaPipe TFLite : pas de comptage FLOPs direct.
        # Valeurs estimées provenant de la documentation :
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
        logger.info("%s: ressources libérées.", self.name)


# ──────────────────────────────────────────────
#  Variantes concrètes
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
