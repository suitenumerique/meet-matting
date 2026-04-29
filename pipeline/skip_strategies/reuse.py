import numpy as np
from core.base import SkipStrategy
from core.registry import skip_strategies


@skip_strategies.register
class Reuse(SkipStrategy):
    name = "reuse"
    description = "Reuses the last inferred mask as-is for skipped frames."

    @classmethod
    def parameter_specs(cls):
        return []

    def __call__(
        self,
        current_frame: np.ndarray,
        prev_frame: np.ndarray,
        prev_mask: np.ndarray,
    ) -> np.ndarray:
        return prev_mask
