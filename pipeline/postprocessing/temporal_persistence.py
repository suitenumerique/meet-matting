"""
Temporal Persistence filter -- per-pixel track confirmation and debouncing.

Reference: Wojke, Bewley, Paulus (ICIP 2017).
"Simple Online and Realtime Tracking with a Deep Association Metric (DeepSORT)."

The DeepSORT paper defines a track lifecycle for bounding-box detections:
  - TENTATIVE: a new track needs n_confirm consecutive hits before being shown.
  - CONFIRMED: a confirmed track is kept alive across up to max_age misses.
  - DELETED:   a track is removed after max_age consecutive misses.

This postprocessor applies the same finite-state machine independently to every
pixel in the segmentation mask, treating each pixel as a 1-D binary track.

State per pixel:
  HIDDEN  (False) -> VISIBLE (True) : requires n_confirm consecutive positives.
  VISIBLE (True)  -> HIDDEN  (False): requires max_age  consecutive negatives.

Positive / negative is decided by comparing mask[p] against a threshold.
Output is a binary float32 mask (0.0 / 1.0).

This eliminates single-frame false positives (camera noise, model glitch) and
prevents rapid disappearance of a partially occluded person.
"""

from __future__ import annotations

import numpy as np
from core.base import Postprocessor
from core.parameters import ParameterSpec
from core.registry import postprocessors


@postprocessors.register
class TemporalPersistence(Postprocessor):
    name = "temporal_persistence"
    description = "DeepSORT-style debouncer: confirms a pixel after N hits, keeps it for K misses."
    details = (
        "Reference: Wojke et al. (DeepSORT, ICIP 2017).\n"
        "State machine per pixel:\n"
        "  HIDDEN -> VISIBLE: requires n_confirm consecutive frames above threshold.\n"
        "  VISIBLE -> HIDDEN: requires max_age  consecutive frames below threshold.\n"
        "Use after a temporal smoother (EMA, One Euro) so that threshold crossings\n"
        "are already stable before the state machine makes its decision."
    )

    def __init__(self, **params) -> None:
        """Initialise with params and allocate per-pixel state, confirm, and age arrays."""
        super().__init__(**params)
        # Per-pixel state: True = VISIBLE, False = HIDDEN.
        self._state: np.ndarray | None = None
        # Consecutive positive-detection counter (for HIDDEN pixels).
        self._confirm_count: np.ndarray | None = None
        # Consecutive negative-detection counter (for VISIBLE pixels).
        self._age_count: np.ndarray | None = None

    @classmethod
    def parameter_specs(cls) -> list[ParameterSpec]:
        """Return the list of tunable parameters for this component."""
        return [
            ParameterSpec(
                name="threshold",
                type="float",
                default=0.5,
                label="Detection threshold",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                help=(
                    "Pixel is 'detected' when mask value >= threshold. "
                    "Apply hysteresis or EMA upstream for stable crossings."
                ),
            ),
            ParameterSpec(
                name="n_confirm",
                type="int",
                default=3,
                label="Confirm frames (n_confirm)",
                min_value=1,
                max_value=30,
                step=1,
                help=(
                    "Number of consecutive positive frames required to switch "
                    "a pixel from HIDDEN to VISIBLE. Higher = fewer false positives."
                ),
            ),
            ParameterSpec(
                name="max_age",
                type="int",
                default=5,
                label="Max age (max_age)",
                min_value=1,
                max_value=60,
                step=1,
                help=(
                    "Number of consecutive negative frames tolerated before "
                    "a VISIBLE pixel is set back to HIDDEN. Higher = more persistence."
                ),
            ),
        ]

    def reset(self) -> None:
        """Clear all per-pixel state so the filter re-initialises on the next frame."""
        self._state = None
        self._confirm_count = None
        self._age_count = None

    def _init_state(self, mask: np.ndarray, threshold: float) -> None:
        """Cold start: initialise from first frame using the strict threshold."""
        self._state = mask >= threshold
        self._confirm_count = np.zeros(mask.shape, dtype=np.int32)
        self._age_count = np.zeros(mask.shape, dtype=np.int32)

    def __call__(self, mask: np.ndarray, original_frame: np.ndarray) -> np.ndarray:
        """Apply one step of the DeepSORT-style state machine to every pixel of *mask*.

        Args:
            mask:           Alpha matte, shape (H, W), dtype float32, range [0, 1].
            original_frame: Original RGB frame, shape (H, W, 3), dtype uint8 (unused).

        Returns:
            Binary mask, shape (H, W), dtype float32, values in {0.0, 1.0}.
        """
        threshold = float(self.params["threshold"])
        n_confirm = int(self.params["n_confirm"])
        max_age = int(self.params["max_age"])

        if self._state is None or self._state.shape != mask.shape:
            self._init_state(mask, threshold)
            assert self._state is not None
            return self._state.astype(np.float32)

        positive = mask >= threshold  # bool (H, W)
        negative = ~positive

        assert self._confirm_count is not None
        assert self._age_count is not None

        # --- Update HIDDEN pixels -------------------------------------------
        # Increment confirm counter on positive detection, reset on miss.
        hidden = ~self._state
        self._confirm_count = np.where(
            hidden & positive,
            self._confirm_count + 1,
            np.where(hidden & negative, 0, self._confirm_count),
        )

        # HIDDEN -> VISIBLE transition.
        newly_visible = hidden & (self._confirm_count >= n_confirm)

        # --- Update VISIBLE pixels ------------------------------------------
        # Increment age counter on negative detection, reset on positive.
        visible = self._state
        self._age_count = np.where(
            visible & negative,
            self._age_count + 1,
            np.where(visible & positive, 0, self._age_count),
        )

        # VISIBLE -> HIDDEN transition.
        newly_hidden = visible & (self._age_count >= max_age)

        # --- Apply state transitions atomically -----------------------------
        self._state = (self._state | newly_visible) & (~newly_hidden)

        # Reset counters for pixels that just changed state.
        self._confirm_count = np.where(newly_visible, 0, self._confirm_count)
        self._age_count = np.where(newly_hidden, 0, self._age_count)

        return self._state.astype(np.float32)
