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
        use_refinement: bool = True,
        refine_radius: int = 8,
        refine_eps: float = 1e-2,
    ):
        self._model_path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
        self._session: Any = None
        self._input_name: str | None = None

        # Paramètres de raffinement
        self._use_refinement = use_refinement
        self._refine_radius = refine_radius
        self._refine_eps = refine_eps

    @property
    def name(self) -> str:
        name = "PP-HumanSeg V2"
        if self._use_refinement:
            name += " (Refined)"
        return name

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
        return self.predict_batch([frame_bgr])[0]

    def predict_batch(self, frames_bgr: list[np.ndarray]) -> list[np.ndarray]:
        if not frames_bgr:
            return []

        h_orig, w_orig = frames_bgr[0].shape[:2]

        if self._session is None:
            logger.warning("PP-HumanSeg V2: pas de modèle chargé, retour masques vides.")
            return [np.zeros((h_orig, w_orig), dtype=np.float32) for _ in frames_bgr]

        batch_size = len(frames_bgr)

        # Pre-processing avec Letterbox pour préserver l'aspect ratio
        tensors = []
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        std = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        # Calcul du ratio de redimensionnement et du padding une seule fois (si toutes les frames ont la même taille)
        scale = min(self._INPUT_W / w_orig, self._INPUT_H / h_orig)
        nw, nh = int(w_orig * scale), int(h_orig * scale)
        dx, dy = (self._INPUT_W - nw) // 2, (self._INPUT_H - nh) // 2

        for frame in frames_bgr:
            # Note: On convertit en RGB car le modèle a été entraîné en RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Letterbox resize
            interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
            frame_resized = cv2.resize(frame_rgb, (nw, nh), interpolation=interp)

            # Create padded canvas
            canvas = np.full((self._INPUT_H, self._INPUT_W, 3), 127, dtype=np.uint8)  # Gray padding
            canvas[dy : dy + nh, dx : dx + nw, :] = frame_resized

            tensor = (canvas.astype(np.float32) / 255.0 - mean) / std
            tensor = np.transpose(tensor, (2, 0, 1))
            tensors.append(tensor)

        batch_tensor = np.stack(tensors, axis=0)

        # Inférence
        try:
            output = self._session.run(None, {self._input_name: batch_tensor})
            logits_batch = output[0]
        except Exception as e:
            if "Got: " in str(e) and "Expected: 1" in str(e):
                logger.debug("PP-HumanSeg: Batching non supporté, passage en boucle.")
                logits_batch = []
                for i in range(batch_size):
                    t = np.expand_dims(batch_tensor[i], axis=0)
                    out = self._session.run(None, {self._input_name: t})
                    logits_batch.append(out[0][0])
                logits_batch = np.array(logits_batch)
            else:
                raise e

        masks = []
        for i in range(batch_size):
            logits = logits_batch[i]
            # Softmax / Sigmoid selon le format d'output
            if logits.ndim == 3 and logits.shape[0] >= 2:
                exp_logits = np.exp(logits - logits.max(axis=0, keepdims=True))
                probs = exp_logits / exp_logits.sum(axis=0, keepdims=True)
                mask_low_padded = probs[1]
            elif logits.ndim == 3 and logits.shape[0] == 1:
                mask_low_padded = 1.0 / (1.0 + np.exp(-logits[0]))
            else:
                mask_low_padded = logits.squeeze()

            # Retirer le padding du masque
            mask_low = mask_low_padded[dy : dy + nh, dx : dx + nw]

            # 1. Upsample initial
            mask_up = cv2.resize(mask_low, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

            # 2. Raffinement via Guided Filter (optionnel mais recommandé)
            if self._use_refinement:
                guide = frames_bgr[i]
                mask_up = cv2.ximgproc.guidedFilter(
                    guide=guide, src=mask_up, radius=self._refine_radius, eps=self._refine_eps
                )

            # 3. Post-traitement du contraste (Sigmoïde douce)
            mask = np.clip((mask_up - 0.45) / (0.55 - 0.45), 0.0, 1.0)

            masks.append(mask)

        return masks

    def get_flops(self, input_shape: tuple[int, int, int] = (3, 192, 192)) -> float:
        # PP-HumanSeg V2 : ~90 MFLOPs à 192x192
        c, h, w = input_shape
        base_flops = 90e6
        scale = (h * w) / (192 * 192)
        return base_flops * scale

    def cleanup(self) -> None:
        self._session = None
        logger.info("PP-HumanSeg V2: session ONNX fermée.")
