"""
Wrapper pour MobileNetV3 avec tête de segmentation LRASPP.

Utilise le modèle pré-entraîné de torchvision.models.segmentation.
LRASPP (Lite R-ASPP) est un décodeur léger optimisé pour le mobile.

Ce modèle est entraîné sur COCO/VOC avec une classe "person" (label 15).
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from .base import BaseModelWrapper

logger = logging.getLogger(__name__)


class MobileNetV3LRASPPWrapper(BaseModelWrapper):
    """
    MobileNetV3-Large + LRASPP head via torchvision.

    Produit un masque de segmentation sémantique, dont on extrait
    la classe "person" (index 15 dans COCO/VOC).
    """

    PERSON_CLASS_INDEX = 15  # Index de la classe "person" dans COCO/VOC

    def __init__(self):
        self._model = None
        self._device = None

    @property
    def name(self) -> str:
        return "MobileNetV3 + LRASPP"

    @property
    def input_size(self) -> Optional[Tuple[int, int]]:
        return (256, 256)

    def load(self) -> None:
        try:
            import torch
            import torchvision.models.segmentation as seg_models
        except ImportError as e:
            raise ImportError(
                "torch et torchvision sont requis. "
                "Installe-les via : pip install torch torchvision"
            ) from e

        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")

        # Charger le modèle pré-entraîné
        self._model = seg_models.lraspp_mobilenet_v3_large(
            weights=seg_models.LRASPP_MobileNet_V3_Large_Weights.DEFAULT,
        )
        self._model.to(self._device)
        self._model.eval()

        self._torch = torch
        logger.info(
            "MobileNetV3+LRASPP: modèle chargé sur %s.", self._device
        )

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("MobileNetV3+LRASPP: modèle non chargé.")

        import torch

        h_orig, w_orig = frame_bgr.shape[:2]

        # Pre-processing
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (256, 256))

        # Normalisation ImageNet
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        tensor = (frame_resized.astype(np.float32) / 255.0 - mean) / std
        tensor = np.transpose(tensor, (2, 0, 1))  # CHW
        tensor = np.expand_dims(tensor, axis=0)     # NCHW

        input_tensor = torch.from_numpy(tensor).to(self._device)

        with torch.no_grad():
            output = self._model(input_tensor)["out"]  # (1, 21, H, W)

        # Extraire la probabilité de la classe "person"
        probs = torch.softmax(output, dim=1)
        person_mask = probs[0, self.PERSON_CLASS_INDEX].cpu().numpy()

        # Resize vers la taille originale
        if person_mask.shape != (h_orig, w_orig):
            person_mask = cv2.resize(
                person_mask, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR
            )

        return person_mask.astype(np.float32)

    def get_flops(self, input_shape: Tuple[int, int, int] = (3, 256, 256)) -> float:
        try:
            import torch
            from fvcore.nn import FlopCountAnalysis

            dummy = torch.randn(1, *input_shape).to(self._device)
            flops = FlopCountAnalysis(self._model, dummy)
            return float(flops.total())
        except ImportError:
            logger.warning(
                "fvcore non disponible, retour aux estimations."
            )
            # ~70 MFLOPs pour MobileNetV3-Large + LRASPP à 256x256
            c, h, w = input_shape
            return 70e6 * (h * w) / (256 * 256)
        except Exception as e:
            logger.warning("Erreur FLOPs MobileNetV3: %s", e)
            return -1.0

    def cleanup(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        logger.info("MobileNetV3+LRASPP: ressources libérées.")
