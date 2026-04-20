"""
Wrapper pour MODNet — Trimap-Free Portrait Matting in Real Time.

MODNet est un réseau de matting single-stage qui ne nécessite pas de trimap.
Il produit directement un alpha matte de haute qualité.

Ce wrapper utilise ONNX Runtime pour l'inférence.

Modèle : https://github.com/ZHKKKe/MODNet
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .base import BaseModelWrapper

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "weights" / "modnet_photographic_portrait_matting.onnx"
_MODEL_URL = (
    "https://huggingface.co/Xenova/modnet/resolve/main/onnx/model.onnx?download=true"
)


class MODNetWrapper(BaseModelWrapper):
    """
    MODNet via ONNX Runtime.

    Input attendu : (1, 3, H, W) normalisé [0, 1].
    Output : alpha matte (1, 1, H, W) dans [0, 1].
    """

    def __init__(self, model_path: Optional[str] = None):
        self._model_path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
        self._session = None
        self._input_name = None
        self._ref_size = 512  # MODNet attend des multiples de 32, typiquement 512

    @property
    def name(self) -> str:
        return "MODNet"

    @property
    def input_size(self) -> Optional[Tuple[int, int]]:
        return (self._ref_size, self._ref_size)

    def load(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "onnxruntime est requis. Installe-le via : pip install onnxruntime"
            ) from e

        if not self._model_path.exists():
            logger.info("MODNet: téléchargement du modèle depuis %s", _MODEL_URL)
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
        logger.info("MODNet: modèle ONNX chargé (%s).", self._model_path.name)

    def _download_model(self) -> None:
        import urllib.request

        self._model_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("MODNet: téléchargement vers %s …", self._model_path)
        urllib.request.urlretrieve(_MODEL_URL, str(self._model_path))
        logger.info("MODNet: téléchargement terminé.")

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self._session is None:
            raise RuntimeError("MODNet: modèle non chargé. Appelle load() d'abord.")

        h_orig, w_orig = frame_bgr.shape[:2]

        # Pre-processing : BGR -> RGB, resize, normalise
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self._ref_size, self._ref_size))

        # Normalisation [0, 1] puis standardisation MODNet
        tensor = frame_resized.astype(np.float32) / 255.0
        tensor = (tensor - 0.5) / 0.5  # [-1, 1]
        tensor = np.transpose(tensor, (2, 0, 1))   # CHW
        tensor = np.expand_dims(tensor, axis=0)      # NCHW

        # Inférence
        output = self._session.run(None, {self._input_name: tensor})
        alpha = output[0]  # (1, 1, H, W)

        mask = alpha[0, 0]  # (H, W)
        mask = np.clip(mask, 0.0, 1.0)

        # Resize vers la taille originale
        if mask.shape != (h_orig, w_orig):
            mask = cv2.resize(mask, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

        return mask.astype(np.float32)

    def get_flops(self, input_shape: Tuple[int, int, int] = (3, 256, 256)) -> float:
        # MODNet : ~4 GFLOPs à 512x512
        c, h, w = input_shape
        base_flops = 4e9
        scale = (h * w) / (512 * 512)
        return base_flops * scale

    def cleanup(self) -> None:
        self._session = None
        logger.info("MODNet: session ONNX fermée.")
