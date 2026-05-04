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
from typing import Any

import cv2
import numpy as np

from .base import BaseModelWrapper

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "weights" / "pphumanseg_v2.onnx"


class PPHumanSegV2Wrapper(BaseModelWrapper):
    """
    PP-HumanSeg V2 via ONNX Runtime.

    Input attendu : (1, 3, 192, 192) normalisé avec mean/std PaddleSeg.
    Output : segmentation map (1, 2, H, W) — argmax pour obtenir le masque binaire.
    """

    _INPUT_H = 192
    _INPUT_W = 192
    _MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/human_segmentation_pphumanseg/human_segmentation_pphumanseg_2023mar.onnx"

    def __init__(
        self,
        model_path: str | None = None,
    ):
        self._model_path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
        self._session: Any = None
        self._input_name: str | None = None

    @property
    def name(self) -> str:
        return "PP-HumanSeg V2"

    @property
    def input_size(self) -> tuple[int, int] | None:
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

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # CoreML specific optimizations if on Mac
        actual_providers: list[str | tuple[str, dict[str, Any]]] = []
        for p in selected_providers:
            if p == "CoreMLExecutionProvider":
                actual_providers.append(
                    (
                        "CoreMLExecutionProvider",
                        {
                            "MLComputeUnits": "ALL",
                            "convert_model_to_fp16": True,  # Enable FP16 inference on Mac
                        },
                    )
                )
            else:
                actual_providers.append(p)

        self._session = ort.InferenceSession(
            str(self._model_path), providers=actual_providers, sess_options=sess_options
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
        """
        Exécute l'inférence sur une frame BGR.
        Gère le pre-processing et le post-processing en interne.
        """
        if self._session is None:
            raise RuntimeError("Modèle non chargé.")

        h_orig, w_orig = frame_bgr.shape[:2]

        # 1. Pre-processing: Letterbox
        scale = min(self._INPUT_W / w_orig, self._INPUT_H / h_orig)
        nw, nh = int(w_orig * scale), int(h_orig * scale)
        dx, dy = (self._INPUT_W - nw) // 2, (self._INPUT_H - nh) // 2

        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        frame_resized = cv2.resize(frame_bgr, (nw, nh), interpolation=interp)
        canvas = np.full((self._INPUT_H, self._INPUT_W, 3), 127, dtype=np.uint8)
        canvas[dy : dy + nh, dx : dx + nw] = frame_resized

        # 2. Normalisation
        tensor = (canvas.astype(np.float32) / 255.0 - 0.5) / 0.5
        tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis]

        # 3. Inférence
        output = self._session.run(None, {self._input_name: tensor})
        logits = output[0][0]

        # 4. Activation (Softmax/Sigmoid)
        if logits.ndim == 3 and logits.shape[0] >= 2:
            exp_logits = np.exp(logits - logits.max(axis=0, keepdims=True))
            probs = exp_logits / exp_logits.sum(axis=0, keepdims=True)
            mask = probs[1]
        elif logits.ndim == 3 and logits.shape[0] == 1:
            mask = 1.0 / (1.0 + np.exp(-logits[0]))
        else:
            mask = logits.squeeze()

        # 5. Post-processing: Crop & Resize back
        mask_valid = mask[dy : dy + nh, dx : dx + nw]
        mask_full = cv2.resize(mask_valid, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

        return mask_full.astype(np.float32)

    def predict_batch(self, frames_bgr: list[np.ndarray]) -> list[np.ndarray]:
        """
        Exécute l'inférence sur un lot de frames BGR.
        """
        # Pour PP-HumanSeg, on réutilise predict car le batching natif ONNX
        # compliquerait la gestion des différentes tailles d'entrée/padding.
        return [self.predict(f) for f in frames_bgr]

    def get_flops(self, input_shape: tuple[int, int, int] = (3, 192, 192)) -> float:
        # PP-HumanSeg V2 : ~90 MFLOPs à 192x192
        c, h, w = input_shape
        base_flops = 90e6
        scale = (h * w) / (192 * 192)
        return base_flops * scale

    def cleanup(self) -> None:
        self._session = None
        logger.info("PP-HumanSeg V2: session ONNX fermée.")
