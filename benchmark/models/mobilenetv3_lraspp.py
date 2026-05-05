"""
Wrapper pour MobileNetV3 avec tête de segmentation LRASPP.

Utilise le modèle pré-entraîné de torchvision.models.segmentation.
LRASPP (Lite R-ASPP) est un décodeur léger optimisé pour le mobile.

Ce modèle est entraîné sur COCO/VOC avec une classe "person" (label 15).
"""

import logging

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
        """Initialise with no model loaded yet."""
        self._model = None
        self._device = None

    @property
    def name(self) -> str:
        """Return the model name."""
        return "MobileNetV3 + LRASPP"

    @property
    def input_size(self) -> tuple[int, int] | None:
        """Return the fixed 256×256 input size."""
        return (256, 256)

    def load(self) -> None:
        """Download weights if needed and initialise the inference session."""
        try:
            import torch
            import torchvision.models.segmentation as seg_models
        except ImportError as e:
            raise ImportError(
                "torch et torchvision sont requis. Installe-les via : pip install torch torchvision"
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
        logger.info("MobileNetV3+LRASPP: modèle chargé sur %s.", self._device)

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Run inference on a single BGR frame; delegates to predict_batch."""
        return self.predict_batch([frame_bgr])[0]

    def predict_batch(self, frames_bgr: list[np.ndarray]) -> list[np.ndarray]:
        """Run inference on a batch of BGR frames and return float32 masks.

        Args:
            frames_bgr: List of BGR images (H, W, 3), dtype uint8.

        Returns:
            List of alpha mattes (H, W), dtype float32, values in [0, 1].
        """
        if self._model is None:
            raise RuntimeError("MobileNetV3+LRASPP: modèle non chargé.")

        import torch

        batch_size = len(frames_bgr)
        if batch_size == 0:
            return []

        h_orig, w_orig = frames_bgr[0].shape[:2]

        # Pre-processing
        tensors = []
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        for frame in frames_bgr:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (256, 256))
            tensor = (frame_resized.astype(np.float32) / 255.0 - mean) / std
            tensor = np.transpose(tensor, (2, 0, 1))
            tensors.append(tensor)

        input_batch = torch.from_numpy(np.stack(tensors)).to(self._device)

        with torch.no_grad():
            output = self._model(input_batch)["out"]  # (N, 21, H, W)

        # Extraire la probabilité de la classe "person"
        probs = torch.softmax(output, dim=1)

        masks = []
        for i in range(batch_size):
            person_mask = probs[i, self.PERSON_CLASS_INDEX].cpu().numpy()

            # Resize vers la taille originale
            if person_mask.shape != (h_orig, w_orig):
                person_mask = cv2.resize(
                    person_mask, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR
                )
            masks.append(person_mask.astype(np.float32))

        return masks

    def get_flops(self, input_shape: tuple[int, int, int] = (3, 256, 256)) -> float:
        """Return estimated FLOPs (~70 MFLOPs at 256×256)."""
        # ~70 MFLOPs for MobileNetV3-Large + LRASPP at 256×256 (from torchvision model card)
        _, h, w = input_shape
        return 70e6 * (h * w) / (256 * 256)

    def cleanup(self) -> None:
        """Delete the model and free GPU memory."""
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
