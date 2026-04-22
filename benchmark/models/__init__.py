"""
Model wrappers for the Video Matting benchmark.

Each wrapper inherits from BaseModelWrapper and implements:
  - load()      : Load the model
  - predict()   : Run inference on a frame
  - cleanup()   : Release resources
"""

from .base import BaseModelWrapper
from .mediapipe_model import (
    MediapipeLandscapeWrapper,
    MediapipePortraitWrapper,
    MediapipeSelfieMulticlassWrapper,
)
from .rvm import RVMWrapper
from .mobilenetv3_lraspp import MobileNetV3LRASPPWrapper
from .trimap_matting import TrimapMattingWrapper
from .modnet import MODNetWrapper
from .pphumanseg import PPHumanSegV2Wrapper
from .efficient_vit import EfficientViTWrapper

# Registry of every model available for the benchmark
MODEL_REGISTRY: dict[str, type[BaseModelWrapper]] = {
    "mediapipe_portrait": MediapipePortraitWrapper,
    "mediapipe_selfie_multiclass": MediapipeSelfieMulticlassWrapper,
    "mediapipe_landscape": MediapipeLandscapeWrapper,
    "rvm": RVMWrapper,
    "mobilenetv3_lraspp": MobileNetV3LRASPPWrapper,
    "trimap_matting": TrimapMattingWrapper,
    "modnet": MODNetWrapper,
    "pphumanseg_v2": PPHumanSegV2Wrapper,
    "efficient_vit": EfficientViTWrapper,
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
    "EfficientViTWrapper",
]
