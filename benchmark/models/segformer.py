"""
Wrapper PyTorch + MPS pour SegFormer-B0 (segmentation ADE20K).

Remplace EfficientViT — qui dépend de Triton/CUDA et ne tourne pas sur Mac —
par SegFormer-B0, transformer-seg léger de NVIDIA disponible via HuggingFace
transformers et compatible Apple Silicon.

Modèle : nvidia/segformer-b0-finetuned-ade-512-512
   https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512

~3.7M paramètres, ~8 GFLOPs à 512x512. Prédit 150 classes ADE20K ;
on extrait le canal "person" (index 12, convention CSAILVision/sceneparsing).
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from .base import BaseModelWrapper

logger = logging.getLogger(__name__)

_HF_MODEL_ID = "nvidia/segformer-b0-finetuned-ade-512-512"


class SegFormerWrapper(BaseModelWrapper):
    """
    SegFormer-B0 (ADE20K) — PyTorch natif sur MPS.

    Modèle généraliste (150 classes), pas spécialisé portrait : on extrait
    la classe "person". L'IoU sera plus modeste que MODNet/RVM mais > 0.
    """

    _INPUT_SIZE = 512
    _PERSON_CLASS_INDEX = 12  # 0-indexed ADE20K
    _IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, model_id: str = _HF_MODEL_ID):
        self._model_id = model_id
        self._model: Any = None
        self._device: Any = None

    @property
    def name(self) -> str:
        return "SegFormer-B0"

    @property
    def input_size(self) -> tuple[int, int] | None:
        return (self._INPUT_SIZE, self._INPUT_SIZE)

    def load(self) -> None:
        try:
            import torch
            from transformers import SegformerForSemanticSegmentation
        except ImportError as e:
            raise ImportError("SegFormer: `transformers` requis. pip install transformers") from e

        if torch.backends.mps.is_available():
            self._device = torch.device("mps")
        elif torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

        model = SegformerForSemanticSegmentation.from_pretrained(self._model_id)
        model.eval()
        model.to(self._device)

        self._model = model
        logger.info("SegFormer: device=%s, model=%s", self._device, self._model_id)

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        return self.predict_batch([frame_bgr])[0]

    def predict_batch(self, frames_bgr: list[np.ndarray]) -> list[np.ndarray]:
        import torch

        if not frames_bgr:
            return []
        if self._model is None:
            raise RuntimeError("SegFormer: modèle non chargé. Appelle load() d'abord.")

        h_orig, w_orig = frames_bgr[0].shape[:2]

        tensors = []
        for frame in frames_bgr:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (self._INPUT_SIZE, self._INPUT_SIZE))
            norm = (resized.astype(np.float32) / 255.0 - self._IMAGENET_MEAN) / self._IMAGENET_STD
            tensors.append(np.transpose(norm, (2, 0, 1)))  # CHW

        batch = torch.from_numpy(np.stack(tensors, axis=0)).to(self._device, non_blocking=True)

        with torch.inference_mode():
            outputs = self._model(pixel_values=batch)
            logits = outputs.logits  # [N, 150, H/4, W/4]

        # SegFormer renvoie une sortie 4× downsamplée → upsample
        logits = torch.nn.functional.interpolate(
            logits,
            size=(self._INPUT_SIZE, self._INPUT_SIZE),
            mode="bilinear",
            align_corners=False,
        )

        probs = torch.softmax(logits, dim=1)  # [N, 150, H, W]
        person_prob = probs[:, self._PERSON_CLASS_INDEX, :, :].detach().to("cpu").numpy()

        masks = []
        for i in range(len(frames_bgr)):
            mask = person_prob[i].astype(np.float32)
            if mask.shape != (h_orig, w_orig):
                mask = cv2.resize(mask, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
            masks.append(np.clip(mask, 0.0, 1.0))

        return masks

    def get_flops(self, input_shape: tuple[int, int, int] = (3, 512, 512)) -> float:
        # SegFormer-B0 : ~8 GFLOPs à 512x512 (paper Table 2).
        _, h, w = input_shape
        return 8e9 * (h * w) / (512 * 512)

    def cleanup(self) -> None:
        import gc

        import torch

        self._model = None
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()
        logger.info("SegFormer: ressources libérées.")
