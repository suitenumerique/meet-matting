"""
One Euro Filter applied pixelwise to the segmentation mask.

Reference: Casiez, Roussel, Vogel (CHI 2012).
"1 Euro Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems."

The core mathematics live in core/filters.py (alpha_from_cutoff, OneEuroFilter1D).
This postprocessor vectorises that same logic over (H, W) arrays with numpy
so that the entire mask is processed in a single pass without Python loops.

Adaptive cutoff per pixel:
    dx_hat[p] = a_d * (mask[p] - x_hat[p]) * f_s + (1 - a_d) * dx_hat[p]  (velocity EMA)
    f_c[p]    = min_cutoff + beta * |dx_hat[p]|                              (adaptive cutoff)
    a[p]      = 1 / (1 + tau[p] * f_s)   where tau[p] = 1/(2*pi*f_c[p])    (EMA coefficient)
    x_hat[p]  = a[p] * mask[p] + (1 - a[p]) * x_hat[p]                      (filtered estimate)
"""

from __future__ import annotations

import numpy as np
from core.base import Postprocessor
from core.filters import alpha_from_cutoff
from core.parameters import ParameterSpec
from core.registry import postprocessors


@postprocessors.register
class OneEuroFilterMask(Postprocessor):
    name = "one_euro"
    description = "Adaptive low-pass filter: strong smoothing at rest, low lag during motion."
    details = (
        "Reference: Casiez, Roussel, Vogel (CHI 2012).\n"
        "Principle: per-pixel EMA whose cutoff frequency f_c rises with signal speed.\n"
        "  f_c = min_cutoff + beta * |velocity|\n"
        "  alpha = 1 / (1 + f_s / (2*pi*f_c))\n"
        "  x_hat = alpha * mask + (1-alpha) * x_hat_prev\n"
        "min_cutoff: smoothing at rest (lower -> more anti-jitter, more lag).\n"
        "beta: reactivity to motion (higher -> less lag during transitions).\n"
        "d_cutoff: fixed cutoff for the velocity EMA (leave at 1.0 Hz)."
    )

    def __init__(self, **params) -> None:
        """Initialise with params and allocate per-pixel state buffers."""
        super().__init__(**params)
        self._x_hat: np.ndarray | None = None
        self._dx_hat: np.ndarray | None = None

    @classmethod
    def parameter_specs(cls) -> list[ParameterSpec]:
        """Return the list of tunable parameters for this component."""
        return [
            ParameterSpec(
                name="f_s",
                type="float",
                default=30.0,
                label="Frame rate (fps)",
                min_value=1.0,
                max_value=120.0,
                step=1.0,
                help="Frequency of the input signal. Set to your actual video fps.",
            ),
            ParameterSpec(
                name="min_cutoff",
                type="float",
                default=0.5,
                label="Min cutoff (Hz)",
                min_value=0.01,
                max_value=10.0,
                step=0.01,
                help="Cutoff at rest. Lower = smoother but more lag when stationary.",
            ),
            ParameterSpec(
                name="beta",
                type="float",
                default=2.0,
                label="Speed coefficient (beta)",
                min_value=0.0,
                max_value=20.0,
                step=0.1,
                help="How fast the cutoff rises with motion. Higher = less lag during movement.",
            ),
            ParameterSpec(
                name="d_cutoff",
                type="float",
                default=1.0,
                label="Derivative cutoff (Hz)",
                min_value=0.1,
                max_value=10.0,
                step=0.1,
                help="Smoothing of the velocity estimate. Recommended value: 1.0 Hz.",
            ),
        ]

    def reset(self) -> None:
        """Clear per-pixel state so the filter re-initialises on the next frame."""
        self._x_hat = None
        self._dx_hat = None

    def __call__(self, mask: np.ndarray, original_frame: np.ndarray) -> np.ndarray:
        """Apply one step of the One Euro Filter to *mask*.

        Args:
            mask:           Alpha matte, shape (H, W), dtype float32, range [0, 1].
            original_frame: Original RGB frame, shape (H, W, 3), dtype uint8 (unused).

        Returns:
            Filtered mask, shape (H, W), dtype float32, range [0, 1].
        """
        f_s = float(self.params["f_s"])
        min_cutoff = float(self.params["min_cutoff"])
        beta = float(self.params["beta"])
        d_cutoff = float(self.params["d_cutoff"])

        # Cold start: initialise state from the first frame.
        if self._x_hat is None or self._x_hat.shape != mask.shape:
            self._x_hat = mask.copy()
            self._dx_hat = np.zeros_like(mask)
            return mask

        # --- Velocity (derivative) estimate ---------------------------------
        # Raw finite-difference derivative at sample rate f_s.
        dx_raw = (mask - self._x_hat) * f_s

        # EMA on the derivative with a fixed cutoff d_cutoff.
        a_d = alpha_from_cutoff(d_cutoff, f_s)  # scalar; same for all pixels
        assert self._dx_hat is not None
        self._dx_hat = a_d * dx_raw + (1.0 - a_d) * self._dx_hat

        # --- Adaptive per-pixel cutoff --------------------------------------
        # f_c[p] = min_cutoff + beta * |dx_hat[p]|
        f_c = min_cutoff + beta * np.abs(self._dx_hat)  # shape (H, W)

        # alpha[p] = 1 / (1 + tau[p] * f_s)  where tau = 1/(2*pi*f_c)
        tau = 1.0 / (2.0 * np.pi * f_c)
        a = 1.0 / (1.0 + tau * f_s)  # shape (H, W)

        # --- Position EMA ---------------------------------------------------
        self._x_hat = a * mask + (1.0 - a) * self._x_hat

        return np.clip(self._x_hat, 0.0, 1.0).astype(np.float32)
