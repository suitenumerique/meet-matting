from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from core.parameters import ParameterSpec


class Component(ABC):
    """Shared base for all pipeline components."""

    name: str
    description: str

    def __init__(self, **params):
        """Store params after validating that all required keys are present.

        A key is required if the corresponding ParameterSpec has no default.
        Currently all specs carry defaults, but validation is written defensively.
        """
        specs = {s.name: s for s in self.parameter_specs()}
        for spec_name, spec in specs.items():
            if spec_name not in params:
                if spec.default is None:
                    raise ValueError(
                        f"{self.__class__.__name__}: missing required parameter '{spec_name}'"
                    )
                params[spec_name] = spec.default
        self.params = params

    @classmethod
    @abstractmethod
    def parameter_specs(cls) -> list[ParameterSpec]:
        """Return the list of ParameterSpec objects that describe this component's parameters."""


class Preprocessor(Component, ABC):
    """Transforms a frame before it is passed to the matting model.

    Data contract:
        __call__ receives: np.ndarray, shape (H, W, 3), dtype uint8, RGB.
        __call__ returns:  np.ndarray, shape (H, W, 3), dtype uint8, RGB.
    """

    @abstractmethod
    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """Apply preprocessing to *frame* and return the modified frame.

        Args:
            frame: RGB image, shape (H, W, 3), dtype uint8.

        Returns:
            Preprocessed RGB image, shape (H, W, 3), dtype uint8.
        """


class MattingModel(Component, ABC):
    """Wraps an inference backend and produces an alpha matte.

    Data contract:
        infer receives: np.ndarray, shape (H, W, 3), dtype uint8, RGB.
        infer returns:  np.ndarray, shape (H, W),   dtype float32, range [0, 1].
    """

    @abstractmethod
    def load(self, weights_path: str | None) -> None:
        """Load model weights from *weights_path*.

        Must be called before :meth:`infer`. Implementations that carry their
        weights internally may treat *weights_path* as optional.

        Args:
            weights_path: Filesystem path to the weights file, or ``None``.
        """

    @abstractmethod
    def infer(self, frame: np.ndarray) -> np.ndarray:
        """Run inference on a single frame.

        Args:
            frame: RGB image, shape (H, W, 3), dtype uint8.

        Returns:
            Alpha matte, shape (H, W), dtype float32, values in [0, 1].
        """


class Postprocessor(Component, ABC):
    """Refines the raw mask produced by the model.

    Data contract:
        __call__ receives:
            mask:           np.ndarray, shape (H, W),   dtype float32, range [0, 1].
            original_frame: np.ndarray, shape (H, W, 3), dtype uint8, RGB (un-preprocessed).
        __call__ returns:
            np.ndarray, shape (H, W), dtype float32, same range [0, 1].
    """

    @abstractmethod
    def __call__(self, mask: np.ndarray, original_frame: np.ndarray) -> np.ndarray:
        """Refine *mask* using the original (un-preprocessed) *original_frame*.

        Args:
            mask:           Alpha matte, shape (H, W), dtype float32, range [0, 1].
            original_frame: Original RGB frame, shape (H, W, 3), dtype uint8.

        Returns:
            Refined alpha matte, shape (H, W), dtype float32, range [0, 1].
        """


class UpsamplingMethod(Component, ABC):
    """Upsamples a low-resolution mask to the resolution of a high-resolution guide image.

    Data contract:
        upsample receives:
            low_res_mask: np.ndarray, shape (H_l, W_l), dtype float32, range [0, 1].
            guide:        np.ndarray, shape (H_h, W_h, 3), dtype uint8, RGB.
        upsample returns:
            np.ndarray, shape (H_h, W_h), dtype float32, range [0, 1].
    """

    @abstractmethod
    def upsample(self, low_res_mask: np.ndarray, guide: np.ndarray) -> np.ndarray:
        """Upsample *low_res_mask* to the resolution of *guide*.

        Args:
            low_res_mask: Alpha matte, shape (H_l, W_l), dtype float32, range [0, 1].
            guide:        Full-resolution RGB frame used as upsampling guidance,
                          shape (H_h, W_h, 3), dtype uint8.

        Returns:
            Upsampled alpha matte, shape (H_h, W_h), dtype float32, range [0, 1].
        """
