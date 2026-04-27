import numpy as np
from core.base import MattingModel
from core.parameters import ParameterSpec
from core.registry import models


@models.register
class DummyPassthrough(MattingModel):
    name = "dummy_passthrough"
    description = "Returns an all-ones mask. Use to test the pipeline plumbing."

    @classmethod
    def parameter_specs(cls):
        return [
            ParameterSpec(
                name="mask_value",
                type="float",
                default=1.0,
                label="Mask value",
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                help="Constant value filled into the output mask.",
            ),
        ]

    def load(self, weights_path=None):
        pass  # No weights to load.

    def infer(self, frame):
        """Return a constant-valued mask.

        Args:
            frame: RGB image, shape (H, W, 3), dtype uint8.

        Returns:
            Alpha matte, shape (H, W), dtype float32, filled with ``mask_value``.
        """
        h, w = frame.shape[:2]
        return np.full((h, w), self.params["mask_value"], dtype=np.float32)
