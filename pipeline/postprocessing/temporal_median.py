"""
Temporal median filter over a sliding window of N frames.

The median is more robust than the mean to single-frame spikes (sudden occlusion,
detection glitch, lighting transient) because it discards outlier values entirely
instead of smearing them across subsequent frames the way EMA does.

Use this upstream of EMA: median removes spikes, EMA smooths the residual jitter.

Causal window: output at time t uses frames [t, t-1, ..., t-N+1].
Effective lag: ~1 frame for smooth motion with N=3.
"""

from collections import deque

import numpy as np
from core.base import Postprocessor
from core.parameters import ParameterSpec
from core.registry import postprocessors


@postprocessors.register
class TemporalMedian(Postprocessor):
    name = "temporal_median"
    description = "Removes single-frame spikes by returning the median of the last N frames."

    def __init__(self, **params):
        """Initialise with params and allocate the frame buffer."""
        super().__init__(**params)
        self._buffer: deque | None = None

    @classmethod
    def parameter_specs(cls):
        """Return the list of tunable parameters for this component."""
        return [
            ParameterSpec(
                name="n_frames",
                type="int",
                default=3,
                label="Window size (frames)",
                min_value=3,
                max_value=7,
                step=2,
                help="Odd values only. Higher = more spike suppression but more lag.",
            ),
        ]

    def reset(self):
        """Clear the frame buffer so the filter re-initialises on the next frame."""
        self._buffer = None

    def __call__(self, mask: np.ndarray, original_frame: np.ndarray) -> np.ndarray:
        """Return the pixel-wise median over the last N frames including *mask*.

        Args:
            mask:           Alpha matte, shape (H, W), dtype float32, range [0, 1].
            original_frame: Original RGB frame, shape (H, W, 3), dtype uint8 (unused).

        Returns:
            Median-filtered mask, shape (H, W), dtype float32, range [0, 1].
        """
        n = self.params["n_frames"]

        if self._buffer is None or self._buffer[0].shape != mask.shape:
            # Warm-start: pre-fill with the first frame to avoid cold-start artifacts.
            self._buffer = deque([mask.copy()] * n, maxlen=n)
            # Buffer is all identical, median == mask -- return directly.
            return mask

        self._buffer.append(mask.copy())
        # np.median on a (N, H, W) float32 array; uses a partial-sort (O(N) per pixel).
        return np.median(np.stack(self._buffer, axis=0), axis=0).astype(np.float32)
