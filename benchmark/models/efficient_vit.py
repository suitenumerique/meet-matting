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
import cv2
import numpy as np
from typing import List, Optional, Tuple

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
    # Utilisation d'un modèle HuggingFace plus stable
    _MODEL_URL = "https://huggingface.co/han-lab/efficientvit-seg-b0-ade20k/resolve/main/model.onnx"
    # Dans ADE20K, 'person' est souvent l'index 12
    PERSON_CLASS_INDEX = 12

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
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            # CoreML specific optimizations if on Mac
            actual_providers = []
            for p in selected_providers:
                if p == "CoreMLExecutionProvider":
                    actual_providers.append(
                        ("CoreMLExecutionProvider", {
                            "MLComputeUnits": "ALL",
                            "convert_model_to_fp16": True  # Enable FP16 inference on Mac
                        })
                    )
                else:
                    actual_providers.append(p)

            self._session = ort.InferenceSession(
                str(self._model_path), 
                providers=actual_providers,
                sess_options=sess_options
            )
            self._input_name = self._session.get_inputs()[0].name
            logger.info("EfficientViT: modèle ONNX chargé (%s).", self._model_path.name)
        except Exception as e:
            logger.error("EfficientViT: Erreur lors de la création de la session : %s", e)
            if "INVALID_PROTOBUF" in str(e) or "Protobuf parsing failed" in str(e):
                logger.warning("EfficientViT: Le fichier semble corrompu. Suppression de %s pour un futur téléchargement.", self._model_path)
                if self._model_path.exists():
                    self._model_path.unlink()
            self._session = None

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        return self.predict_batch([frame_bgr])[0]

    def predict_batch(self, frames_bgr: List[np.ndarray]) -> List[np.ndarray]:
        if not frames_bgr:
            return []
            
        h_orig, w_orig = frames_bgr[0].shape[:2]

        if self._session is None:
            logger.warning("EfficientViT: pas de modèle chargé, retour masques vides.")
            return [np.zeros((h_orig, w_orig), dtype=np.float32) for _ in frames_bgr]

        batch_size = len(frames_bgr)
        
        # Déterminer la forme d'entrée attendue
        input_meta = self._session.get_inputs()[0]
        input_shape = input_meta.shape
        logger.info("EfficientViT: Metadata entrée - Shape: %s, Name: %s", input_shape, input_meta.name)
        
        # On essaie de deviner l'emplacement des canaux (souvent 1 en NCHW, 3 en NHWC)
        if len(input_shape) == 4:
            if isinstance(input_shape[1], int) and input_shape[1] in [3, 4]:
                expected_channels = input_shape[1]
                layout = "NCHW"
            elif isinstance(input_shape[3], int) and input_shape[3] in [3, 4]:
                expected_channels = input_shape[3]
                layout = "NHWC"
            else:
                expected_channels = 3 # fallback
                layout = "NCHW"
        else:
            expected_channels = 3
            layout = "NCHW"

        logger.info("EfficientViT: Layout détecté: %s, Canaux attendus: %d", layout, expected_channels)
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        tensors = []
        
        for frame in frames_bgr:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (self._INPUT_SIZE, self._INPUT_SIZE))
            img = (frame_resized.astype(np.float32) / 255.0 - mean) / std
            
            if expected_channels == 4:
                # Ajout d'un canal alpha (zéros)
                alpha = np.zeros((self._INPUT_SIZE, self._INPUT_SIZE, 1), dtype=np.float32)
                img = np.concatenate([img, alpha], axis=-1)
            
            if layout == "NCHW":
                tensor = np.transpose(img, (2, 0, 1))
            else:
                tensor = img
                
            tensors.append(tensor)

        batch_tensor = np.stack(tensors, axis=0)

        # Inférence avec logique de repli automatique (Self-Healing)
        try:
            output = self._session.run(None, {self._input_name: batch_tensor})
            logits_batch = output[0]
        except Exception as e:
            error_msg = str(e)
            # Si l'erreur indique un problème de canaux (C: 3 au lieu de 4)
            if "channels C" in error_msg and "not equal to kernel channels" in error_msg:
                logger.info("EfficientViT: Détection d'un modèle à 4 canaux. Transformation en cours...")
                # Recréer le batch avec 4 canaux
                new_tensors = []
                for t in tensors: # t est (3, H, W)
                    # Convertir en (H, W, 3)
                    img_3 = np.transpose(t, (1, 2, 0))
                    # Ajouter canal Alpha
                    alpha = np.zeros((self._INPUT_SIZE, self._INPUT_SIZE, 1), dtype=np.float32)
                    img_4 = np.concatenate([img_3, alpha], axis=-1)
                    # Retour vers (4, H, W)
                    new_tensors.append(np.transpose(img_4, (2, 0, 1)))
                
                batch_tensor = np.stack(new_tensors, axis=0)
                # Deuxième tentative
                output = self._session.run(None, {self._input_name: batch_tensor})
                logits_batch = output[0]
            
            elif "Got: " in error_msg and "Expected: 1" in error_msg:
                logger.debug("EfficientViT: Batching non supporté par le modèle, repli sur itération.")
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
            l = logits_batch[i]
            
            # Détection dynamique du format de sortie
            if l.ndim == 3: # Format (C, H, W) - Segmentation
                num_classes = l.shape[0]
                # Si c'est une seule classe (sigmoïde)
                if num_classes == 1:
                    mask = 1.0 / (1.0 + np.exp(-l[0]))
                else:
                    # On cherche la classe personne. Si l'index 15 est out of bounds, 
                    # on prend le max ou l'index 1 si Cityscapes
                    idx = self.PERSON_CLASS_INDEX if num_classes > self.PERSON_CLASS_INDEX else (1 if num_classes > 1 else 0)
                    
                    # Softmax sur les classes
                    exp_l = np.exp(l - l.max(axis=0, keepdims=True))
                    probs = exp_l / exp_l.sum(axis=0, keepdims=True)
                    mask = probs[idx]
            elif l.ndim == 1: # Format (C,) - Erreur, c'est de la classification
                logger.error("EfficientViT: Le modèle chargé est un modèle de classification, pas de segmentation.")
                mask = np.zeros((self._INPUT_SIZE, self._INPUT_SIZE), dtype=np.float32)
            else:
                mask = l.squeeze()
                if mask.ndim != 2:
                    mask = np.zeros((self._INPUT_SIZE, self._INPUT_SIZE), dtype=np.float32)

            # Post-processing : resize vers la taille originale
            if mask.shape != (h_orig, w_orig):
                mask = cv2.resize(mask, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
            
            masks.append(mask.astype(np.float32))

        return masks

    def get_flops(self, input_shape: Tuple[int, int, int] = (3, 512, 512)) -> float:
        # EfficientViT-L1 : ~5 GFLOPs à 512x512
        c, h, w = input_shape
        base_flops = 5e9
        scale = (h * w) / (512 * 512)
        return base_flops * scale

    def cleanup(self) -> None:
        self._session = None
        logger.info("EfficientViT: session ONNX fermée.")
