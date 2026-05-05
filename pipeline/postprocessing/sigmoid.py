"""Sigmoid postprocessor — sharpens soft mask edges by remapping probabilities through a sigmoid."""

import numpy as np
from core.base import Postprocessor
from core.parameters import ParameterSpec
from core.registry import postprocessors


@postprocessors.register
class Sigmoid(Postprocessor):
    name = "sigmoid"
    description = "Pushes mask values toward 0 or 1 via a sigmoid centred on 0.5."

    @classmethod
    def parameter_specs(cls):
        """Return the list of tunable parameters for this component."""
        return [
            ParameterSpec(
                name="gain",
                type="float",
                default=10.0,
                label="Gain",
                min_value=1.0,
                max_value=50.0,
                step=0.5,
                help="Steepness of the sigmoid. Higher = harder edge (approaches threshold at ∞).",
            ),
        ]

    def __call__(self, mask: np.ndarray, original_frame: np.ndarray) -> np.ndarray:
        """Apply a sigmoid centred on 0.5 to *mask*.

        σ(x) = 1 / (1 + exp(-gain · (x - 0.5)))

        Optimized: pre-compute normalization scalars, single np.exp call,
        in-place operations to minimise temporary arrays.
        """
        gain = float(self.params["gain"])

        # Pre-compute normalization constants (scalars — free)
        s0 = 1.0 / (1.0 + np.exp(gain * 0.5))  # σ(0)
        s1 = 1.0 / (1.0 + np.exp(-gain * 0.5))  # σ(1)
        inv_range = np.float32(1.0 / (s1 - s0))
        s0_f32 = np.float32(s0)

        # Single in-place pass: avoid creating (x - 0.5) temporary
        # exponent = -gain * (mask - 0.5)
        exponent = np.subtract(mask, 0.5)  # in-place candidate
        exponent *= -gain
        np.exp(exponent, out=exponent)  # exp in-place
        exponent += 1.0  # 1 + exp(...)
        np.reciprocal(exponent, out=exponent)  # 1 / (1 + exp(...))

        # Normalize so f(0)=0, f(1)=1
        exponent -= s0_f32
        exponent *= inv_range

        return exponent
