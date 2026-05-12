"""
Wrapper PyTorch + MPS pour Robust Video Matting (RVM).

Suit le pattern officiel du README ByteDance :
    https://github.com/PeterL1n/RobustVideoMatting

L'architecture est chargée via `torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3")`
puis les poids officiels `rvm_mobilenetv3.pth` sont chargés depuis disque
(téléchargés depuis la release GitHub si absents).

Sur Apple Silicon (M1/M2/M3), l'inférence tourne sur MPS (GPU Apple) pour
maximiser le throughput — CoreML EP partitionne RVM en 48 segments et est
2× plus lent que CPU ; PyTorch+MPS est la voie la plus rapide testée.
"""

from __future__ import annotations

import gc
import logging
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .base import BaseModelWrapper

logger = logging.getLogger(__name__)

_WEIGHT_FILENAME = "rvm_mobilenetv3.pth"
_DEFAULT_WEIGHT_PATH = Path(__file__).parent.parent / "weights" / _WEIGHT_FILENAME
_WEIGHT_URL = (
    "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth"
)
_HUB_REPO = "PeterL1n/RobustVideoMatting"


class RVMWrapper(BaseModelWrapper):
    """
    Robust Video Matting — PyTorch natif sur Apple MPS.

    Maintient l'état récurrent `rec=[r1,r2,r3,r4]` entre les frames et le
    réinitialise via `reset_state()` (appelé par le runner au début de chaque
    vidéo). Le `downsample_ratio` est choisi automatiquement à la 1ère frame
    selon la table recommandée par le README officiel.
    """

    def __init__(
        self,
        weight_path: str | None = None,
        downsample_ratio: float | None = None,
    ):
        self._weight_path = Path(weight_path) if weight_path else _DEFAULT_WEIGHT_PATH
        self._downsample_override = downsample_ratio
        self._model: Any = None
        self._device: Any = None
        self._rec: list[Any] = [None, None, None, None]
        self._downsample_ratio: float | None = None

    @property
    def name(self) -> str:
        return "RVM (MobileNetV3)"

    @property
    def input_size(self) -> tuple[int, int] | None:
        return None  # Dynamique

    def load(self) -> None:
        import torch

        if not self._weight_path.exists():
            self._download_weights()

        if torch.backends.mps.is_available():
            self._device = torch.device("mps")
        elif torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

        try:
            model = torch.hub.load(_HUB_REPO, "mobilenetv3", pretrained=False, trust_repo=True)
        except TypeError:
            # Older hubconf signatures may not accept `pretrained`.
            model = torch.hub.load(_HUB_REPO, "mobilenetv3", trust_repo=True)

        state_dict = torch.load(self._weight_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        model.to(self._device)

        self._model = model
        self.reset_state()
        logger.info("RVM: device=%s, weights=%s", self._device, self._weight_path.name)

    def _download_weights(self) -> None:
        self._weight_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("RVM: téléchargement des poids depuis %s …", _WEIGHT_URL)
        urllib.request.urlretrieve(_WEIGHT_URL, str(self._weight_path))
        logger.info("RVM: téléchargement terminé → %s", self._weight_path)

    @staticmethod
    def _auto_downsample_ratio(h: int, w: int) -> float:
        """Table issue du README officiel RVM (`inference.py:auto_downsample_ratio`)."""
        m = min(h, w)
        if m <= 512:
            return 1.0
        if m <= 720:
            return 0.5
        if m <= 1080:
            return 0.375
        if m <= 2160:
            return 0.25
        return 0.125

    def reset_state(self) -> None:
        """Réinitialise les états récurrents ConvGRU et le ratio adaptatif."""
        self._rec = [None, None, None, None]
        self._downsample_ratio = self._downsample_override

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        import torch

        if self._model is None:
            raise RuntimeError("RVM: modèle non chargé. Appelle load() d'abord.")

        h_orig, w_orig = frame_bgr.shape[:2]

        if self._downsample_ratio is None:
            self._downsample_ratio = self._auto_downsample_ratio(h_orig, w_orig)
            logger.debug(
                "RVM: downsample_ratio=%.3f pour %dx%d",
                self._downsample_ratio,
                h_orig,
                w_orig,
            )

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        src = (
            torch.from_numpy(frame_rgb)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .div_(255.0)
            .to(self._device, non_blocking=True)
        )

        with torch.inference_mode():
            fgr, pha, *self._rec = self._model(src, *self._rec, self._downsample_ratio)

        alpha = pha[0, 0].detach().to("cpu").numpy().astype(np.float32)

        if alpha.shape != (h_orig, w_orig):
            alpha = cv2.resize(alpha, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

        return np.clip(alpha, 0.0, 1.0)

    def get_flops(self, input_shape: tuple[int, int, int] = (3, 1080, 1920)) -> float:
        # RVM MobileNetV3 : ~600 MFLOPs à 256x256 (référence officielle).
        _, h, w = input_shape
        ratio = self._downsample_ratio if self._downsample_ratio else 0.375
        h_eff = h * ratio
        w_eff = w * ratio
        return 600e6 * (h_eff * w_eff) / (256 * 256)

    def cleanup(self) -> None:
        import torch

        self._model = None
        self._rec = [None, None, None, None]
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()
        logger.info("RVM: ressources libérées.")
