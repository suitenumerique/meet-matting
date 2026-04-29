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

        Args:
            mask:           Alpha matte, shape (H, W), dtype float32, range [0, 1].
            original_frame: Original RGB frame (unused).

        Returns:
            Sharpened mask, shape (H, W), dtype float32, range [0, 1].
        """
        gain = float(self.params["gain"])

        def _s(x):
            return 1.0 / (1.0 + np.exp(-gain * (x - 0.5)))

        # Normalise so that f(0)=0 and f(1)=1 exactly.
        s0 = _s(0.0)
        s1 = _s(1.0)
        return ((_s(mask) - s0) / (s1 - s0)).astype(np.float32)
