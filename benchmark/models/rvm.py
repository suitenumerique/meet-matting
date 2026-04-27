"""
Wrapper pour Robust Video Matting (RVM).

RVM est un modèle récurrent (GRU-based) qui maintient un état interne
entre les frames pour produire des alpha mattes temporellement cohérents.

Ce wrapper supporte l'inférence via :
  - ONNX Runtime (recommandé pour le benchmark, plus portable)
  - PyTorch natif (fallback)

Modèle : https://github.com/PeterL1n/RobustVideoMatting
"""

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .base import BaseModelWrapper

logger = logging.getLogger(__name__)

# Chemin local par défaut pour le modèle ONNX
_DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "weights" / "rvm_mobilenetv3_fp32.onnx"
_MODEL_URL = (
    "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/"
    "rvm_mobilenetv3_fp32.onnx"
)


class RVMWrapper(BaseModelWrapper):
    """
    Robust Video Matting via ONNX Runtime.

    Maintient des états récurrents (r1..r4) entre les frames.
    reset_state() doit être appelé au début de chaque nouvelle vidéo.
    """

    def __init__(self, model_path: str | None = None, downsample_ratio: float = 0.25):
        self._model_path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
        self._downsample_ratio = downsample_ratio
        self._session: Any = None
        self._recurrent_state: dict[str, np.ndarray] = {}

    @property
    def name(self) -> str:
        return "RVM (MobileNetV3)"

    @property
    def input_size(self) -> tuple[int, int] | None:
        return None  # Dynamique, dépend de la vidéo

    def load(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "onnxruntime est requis. Installe-le via : pip install onnxruntime"
            ) from e

        # Télécharger si absent
        if not self._model_path.exists():
            logger.info("RVM: téléchargement du modèle depuis %s", _MODEL_URL)
            self._download_model()

        providers = ort.get_available_providers()
        # Priorité aux accélérateurs matériels (CoreML sur Mac, CUDA sur NVIDIA)
        selected_providers = []
        if "CoreMLExecutionProvider" in providers:
            selected_providers.append("CoreMLExecutionProvider")
        if "CUDAExecutionProvider" in providers:
            selected_providers.append("CUDAExecutionProvider")
        selected_providers.append("CPUExecutionProvider")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # CoreML specific optimizations if on Mac
        actual_providers: list[str | tuple[str, dict[str, Any]]] = []
        for p in selected_providers:
            if p == "CoreMLExecutionProvider":
                actual_providers.append(("CoreMLExecutionProvider", {"MLComputeUnits": "ALL"}))
            else:
                actual_providers.append(p)

        self._session = ort.InferenceSession(
            str(self._model_path), providers=actual_providers, sess_options=sess_options
        )
        self.reset_state()
        logger.info("RVM: modèle ONNX chargé (%s).", self._model_path.name)

    def _download_model(self) -> None:
        """Télécharge le modèle ONNX depuis GitHub."""
        import urllib.request

        self._model_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("RVM: téléchargement vers %s …", self._model_path)
        urllib.request.urlretrieve(_MODEL_URL, str(self._model_path))
        logger.info("RVM: téléchargement terminé.")

    def reset_state(self) -> None:
        """Réinitialise les états récurrents GRU entre vidéos."""
        self._recurrent_state = {
            "r1i": np.zeros((1, 1, 1, 1), dtype=np.float32),
            "r2i": np.zeros((1, 1, 1, 1), dtype=np.float32),
            "r3i": np.zeros((1, 1, 1, 1), dtype=np.float32),
            "r4i": np.zeros((1, 1, 1, 1), dtype=np.float32),
        }

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self._session is None:
            raise RuntimeError("RVM: modèle non chargé. Appelle load() d'abord.")

        h_orig, w_orig = frame_bgr.shape[:2]

        # Pre-processing : BGR -> RGB, HWC -> NCHW, normalise [0,1]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = frame_rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))  # CHW
        tensor = np.expand_dims(tensor, axis=0)  # NCHW

        # Inférence
        inputs = {
            "src": tensor,
            "r1i": self._recurrent_state["r1i"],
            "r2i": self._recurrent_state["r2i"],
            "r3i": self._recurrent_state["r3i"],
            "r4i": self._recurrent_state["r4i"],
            "downsample_ratio": np.array([self._downsample_ratio], dtype=np.float32),
        }

        outputs = self._session.run(None, inputs)

        # Outputs : [fgr, pha, r1o, r2o, r3o, r4o]
        alpha = outputs[1]  # (1, 1, H, W)

        # Mettre à jour l'état récurrent
        self._recurrent_state["r1i"] = outputs[2]
        self._recurrent_state["r2i"] = outputs[3]
        self._recurrent_state["r3i"] = outputs[4]
        self._recurrent_state["r4i"] = outputs[5]

        # Post-processing
        mask = alpha[0, 0]  # (H, W)
        mask = np.clip(mask, 0.0, 1.0)

        if mask.shape != (h_orig, w_orig):
            mask = cv2.resize(mask, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

        return mask.astype(np.float32)

    def get_flops(self, input_shape: tuple[int, int, int] = (3, 256, 256)) -> float:
        # RVM MobileNetV3 : ~600 MFLOPs à 256x256
        c, h, w = input_shape
        base_flops = 600e6  # pour 256x256
        scale = (h * w) / (256 * 256)
        return base_flops * scale

    def cleanup(self) -> None:
        self._session = None
        self._recurrent_state = {}
        logger.info("RVM: session ONNX fermée.")
