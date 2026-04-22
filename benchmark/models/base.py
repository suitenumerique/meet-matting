"""
Abstract base class for every segmentation-model wrapper.

Each model MUST implement:
  - name         : Human-readable model name.
  - load()       : Load into memory (weights, ONNX session, etc.).
  - predict()    : Inference on a BGR frame → float mask in [0, 1].
  - cleanup()    : Release GPU / memory resources.
  - get_flops()  : Estimate or measure of FLOPs per frame.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np


class BaseModelWrapper(ABC):
    """Common interface for every Video Matting model."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name (e.g. 'MediaPipe Portrait')."""
        ...

    @property
    def input_size(self) -> Optional[Tuple[int, int]]:
        """
        Expected input size (H, W). None if the size is dynamic.
        Used for automatic resizing before inference.
        """
        return None

    @abstractmethod
    def load(self) -> None:
        """
        Load the model into memory.

        This method is called once before the inference loop.
        It must download the weights if needed and initialise the session.
        """
        ...

    @abstractmethod
    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Run inference on a BGR frame.

        The implementation must handle pre-processing (resize, normalisation)
        and post-processing (thresholding, resize back to the original size).

        Args:
            frame_bgr: BGR image (H, W, 3) as uint8.

        Returns:
            Segmentation mask (H, W) as float32 in [0, 1].
            H, W must match the input frame dimensions.
        """
        ...

    def predict_batch(self, frames_bgr: List[np.ndarray]) -> List[np.ndarray]:
        """
        Run inference on a batch of BGR frames.

        By default this method loops over predict().
        It should be overridden for models that natively support batching
        (e.g. ONNX, PyTorch) in order to maximise GPU utilisation.

        Args:
            frames_bgr: List of BGR frames (H, W, 3) as uint8.

        Returns:
            List of masks (H, W) as float32 in [0, 1].
        """
        return [self.predict(f) for f in frames_bgr]

    @abstractmethod
    def get_flops(self, input_shape: Tuple[int, int, int] = (3, 256, 256)) -> float:
        """
        Return the number of FLOPs for one inference.

        Args:
            input_shape: Shape of the input (C, H, W).

        Returns:
            Number of FLOPs (Floating Point Operations). -1 if not measurable.
        """
        ...

    def reset_state(self) -> None:
        """
        Reset the model's internal state (for recurrent models such as
        RVM that maintain state between frames).

        Called at the start of each video.
        """
        pass

    def cleanup(self) -> None:
        """
        Release resources (GPU, ONNX sessions, etc.).

        Called after the benchmark finishes for this model.
        """
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} '{self.name}'>"
