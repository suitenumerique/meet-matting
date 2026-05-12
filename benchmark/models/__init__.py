"""
Model wrappers pour le benchmark de Video Matting.

Chaque wrapper hérite de BaseModelWrapper et implémente :
  - load()      : Chargement du modèle
  - predict()   : Inférence sur une frame
  - cleanup()   : Libération des ressources
"""

from .base import BaseModelWrapper
from .mediapipe_multiclass import MediapipeSelfieMulticlassWrapper
from .mediapipe_selfie import (
    MediapipeLandscapeWrapper,
    MediapipePortraitWrapper,
)
from .mobilenetv3_lraspp import MobileNetV3LRASPPWrapper
from .modnet import MODNetWrapper
from .pphumanseg import PPHumanSegV2Wrapper
from .rvm import RVMWrapper
from .segformer import SegFormerWrapper
from .trimap_matting import TrimapMattingWrapper

# Registre de tous les modèles disponibles pour le benchmark
MODEL_REGISTRY: dict[str, type[BaseModelWrapper]] = {
    "mediapipe_portrait": MediapipePortraitWrapper,
    "mediapipe_selfie_multiclass": MediapipeSelfieMulticlassWrapper,
    "mediapipe_landscape": MediapipeLandscapeWrapper,
    "rvm": RVMWrapper,
    "mobilenetv3_lraspp": MobileNetV3LRASPPWrapper,
    "trimap_matting": TrimapMattingWrapper,
    "modnet": MODNetWrapper,
    "pphumanseg_v2": PPHumanSegV2Wrapper,
    "segformer": SegFormerWrapper,
}

__all__ = [
    "BaseModelWrapper",
    "MODEL_REGISTRY",
    "MediapipePortraitWrapper",
    "MediapipeSelfieMulticlassWrapper",
    "MediapipeLandscapeWrapper",
    "RVMWrapper",
    "MobileNetV3LRASPPWrapper",
    "TrimapMattingWrapper",
    "MODNetWrapper",
    "PPHumanSegV2Wrapper",
    "SegFormerWrapper",
]
