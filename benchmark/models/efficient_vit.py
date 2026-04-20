"""
Wrapper pour EfficientViT (segmentation).

EfficientViT est une architecture Vision Transformer allégée, optimisée
pour l'inférence rapide sur GPU et edge devices.

Ce wrapper utilise ONNX Runtime. Le modèle ONNX doit être fourni
(converti depuis le repo officiel ou un export TorchScript).

Modèle : https://github.com/microsoft/Cream/tree/main/EfficientViT
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .base import BaseModelWrapper

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_PATH = (
    Path(__file__).parent.parent / "weights" / "efficientvit_seg.onnx"
)


class EfficientViTWrapper(BaseModelWrapper):
    """
    EfficientViT Segmentation via ONNX Runtime.

    Input attendu : (1, 3, 512, 512) normalisé ImageNet.
    Output : logits de segmentation (1, num_classes, H, W).
    """

    _INPUT_SIZE = 224
    PERSON_CLASS_INDEX = 15  # COCO person class
    _MODEL_URL = "https://github.com/PINTO0309/PINTO_model_zoo/raw/main/199_EfficientViT/models/efficientvit_b0_224x224.onnx"

    def __init__(self, model_path: Optional[str] = None):
        self._model_path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
        self._session = None
        self._input_name = None

    @property
    def name(self) -> str:
        return "EfficientViT"

    @property
    def input_size(self) -> Optional[Tuple[int, int]]:
        return (self._INPUT_SIZE, self._INPUT_SIZE)

    def load(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "onnxruntime est requis. Installe-le via : pip install onnxruntime"
            ) from e

        # Téléchargement automatique si absent
        if not self._model_path.exists():
            import urllib.request
            self._model_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("EfficientViT: téléchargement depuis %s", self._MODEL_URL)
            try:
                urllib.request.urlretrieve(self._MODEL_URL, str(self._model_path))
            except Exception as e:
                logger.error("EfficientViT: Erreur de téléchargement : %s", e)
                return

        if not self._model_path.exists():
            self._session = None
            return

        providers = ort.get_available_providers()
        selected_providers = []
        if "CoreMLExecutionProvider" in providers:
            selected_providers.append("CoreMLExecutionProvider")
        if "CUDAExecutionProvider" in providers:
            selected_providers.append("CUDAExecutionProvider")
        selected_providers.append("CPUExecutionProvider")

        try:
            self._session = ort.InferenceSession(
                str(self._model_path), providers=selected_providers
            )
            self._input_name = self._session.get_inputs()[0].name
            logger.info("EfficientViT: modèle ONNX chargé (%s).", self._model_path.name)
        except Exception as e:
            logger.error("EfficientViT: Erreur lors de la création de la session : %s", e)
            self._session = None

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        h_orig, w_orig = frame_bgr.shape[:2]

        if self._session is None:
            logger.warning(
                "EfficientViT: pas de modèle chargé, retour masque vide."
            )
            return np.zeros((h_orig, w_orig), dtype=np.float32)

        # Pre-processing : BGR -> RGB, resize, normalisation ImageNet
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self._INPUT_SIZE, self._INPUT_SIZE))

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        tensor = (frame_resized.astype(np.float32) / 255.0 - mean) / std
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)

        # Inférence
        output = self._session.run(None, {self._input_name: tensor})
        logits = output[0]

        # Extraire la classe "person"
        if logits.ndim == 4 and logits.shape[1] > self.PERSON_CLASS_INDEX:
            exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            mask = probs[0, self.PERSON_CLASS_INDEX]
        elif logits.ndim == 4 and logits.shape[1] == 1:
            mask = 1.0 / (1.0 + np.exp(-logits[0, 0]))
        else:
            mask = logits.squeeze()
            mask = np.clip(mask, 0.0, 1.0)

        mask = mask.astype(np.float32)

        if mask.shape != (h_orig, w_orig):
            mask = cv2.resize(mask, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

        return mask

    def get_flops(self, input_shape: Tuple[int, int, int] = (3, 512, 512)) -> float:
        # EfficientViT-L1 : ~5 GFLOPs à 512x512
        c, h, w = input_shape
        base_flops = 5e9
        scale = (h * w) / (512 * 512)
        return base_flops * scale

    def cleanup(self) -> None:
        self._session = None
        logger.info("EfficientViT: session ONNX fermée.")
