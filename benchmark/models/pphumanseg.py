"""
Wrapper pour PP-HumanSeg V2 (PaddleSeg).

PP-HumanSeg V2 est un modèle léger de segmentation humaine développé
par Baidu/PaddlePaddle, optimisé pour le déploiement mobile.

Ce wrapper utilise ONNX Runtime pour éviter la dépendance à PaddlePaddle.
Le modèle ONNX doit être converti au préalable ou téléchargé.

Modèle : https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.9/contrib/PP-HumanSeg
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .base import BaseModelWrapper

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_PATH = (
    Path(__file__).parent.parent / "weights" / "pphumanseg_v2.onnx"
)


class PPHumanSegV2Wrapper(BaseModelWrapper):
    """
    PP-HumanSeg V2 via ONNX Runtime.

    Input attendu : (1, 3, 192, 192) normalisé avec mean/std PaddleSeg.
    Output : segmentation map (1, 2, H, W) — argmax pour obtenir le masque binaire.
    """

    _INPUT_H = 192
    _INPUT_W = 192
    _MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/human_segmentation_pphumanseg/human_segmentation_pphumanseg_2023mar.onnx"

    def __init__(self, model_path: Optional[str] = None):
        self._model_path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
        self._session = None
        self._input_name = None

    @property
    def name(self) -> str:
        return "PP-HumanSeg V2"

    @property
    def input_size(self) -> Optional[Tuple[int, int]]:
        return (self._INPUT_H, self._INPUT_W)

    def load(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "onnxruntime est requis. Installe-le via : pip install onnxruntime"
            ) from e

        if not self._model_path.exists():
            logger.info("PP-HumanSeg V2: téléchargement du modèle depuis %s", self._MODEL_URL)
            self._download_model()

        providers = ort.get_available_providers()
        selected_providers = []
        if "CoreMLExecutionProvider" in providers:
            selected_providers.append("CoreMLExecutionProvider")
        if "CUDAExecutionProvider" in providers:
            selected_providers.append("CUDAExecutionProvider")
        selected_providers.append("CPUExecutionProvider")

        self._session = ort.InferenceSession(
            str(self._model_path), providers=selected_providers
        )
        self._input_name = self._session.get_inputs()[0].name
        logger.info("PP-HumanSeg V2: modèle ONNX chargé (%s).", self._model_path.name)

    def _download_model(self) -> None:
        import urllib.request
        self._model_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("PP-HumanSeg V2: téléchargement vers %s …", self._model_path)
        urllib.request.urlretrieve(self._MODEL_URL, str(self._model_path))
        logger.info("PP-HumanSeg V2: téléchargement terminé.")

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        h_orig, w_orig = frame_bgr.shape[:2]

        if self._session is None:
            # Mode placeholder : retourne un masque vide avec un warning
            logger.warning(
                "PP-HumanSeg V2: pas de modèle chargé, retour masque vide."
            )
            return np.zeros((h_orig, w_orig), dtype=np.float32)

        # Pre-processing
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self._INPUT_W, self._INPUT_H))

        # Normalisation PaddleSeg standard
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        tensor = (frame_resized.astype(np.float32) / 255.0 - mean) / std
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)

        # Inférence
        output = self._session.run(None, {self._input_name: tensor})

        # Output shape : (1, 2, H, W) — 2 classes (bg, fg)
        logits = output[0]

        if logits.ndim == 4 and logits.shape[1] >= 2:
            # Softmax sur l'axe des classes
            exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            mask = probs[0, 1]  # Classe "personne"
        elif logits.ndim == 4 and logits.shape[1] == 1:
            mask = 1.0 / (1.0 + np.exp(-logits[0, 0]))  # Sigmoid
        else:
            mask = logits.squeeze()

        # Binarisation pour PP-HumanSeg (modèle de segmentation, pas de matting)
        # Cela évite d'avoir un fond "assombri" (probabilités résiduelles)
        mask = (mask > 0.5).astype(np.float32)

        if mask.shape != (h_orig, w_orig):
            mask = cv2.resize(mask, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

        return mask

    def get_flops(self, input_shape: Tuple[int, int, int] = (3, 192, 192)) -> float:
        # PP-HumanSeg V2 : ~90 MFLOPs à 192x192
        c, h, w = input_shape
        base_flops = 90e6
        scale = (h * w) / (192 * 192)
        return base_flops * scale

    def cleanup(self) -> None:
        self._session = None
        logger.info("PP-HumanSeg V2: session ONNX fermée.")
