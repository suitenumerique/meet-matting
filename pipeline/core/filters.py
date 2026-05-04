"""
Reusable temporal filter primitives.

Reference: Casiez, Roussel, Vogel (CHI 2012).
"1 Euro Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems."

This module provides scalar (1D) and multi-dimensional wrappers that can
stabilise landmark coordinates, bounding-box positions, or any continuous
signal. For dense pixel-array smoothing see postprocessing/one_euro.py which
vectorises the same mathematics over numpy arrays.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Internal helper -- shared by this module and postprocessing/one_euro.py
# ---------------------------------------------------------------------------


def alpha_from_cutoff(cutoff: float, f_s: float) -> float:
    """Return the EMA coefficient for a given cutoff frequency and sample rate.

    Derivation from the 1 Euro Filter paper:
        tau   = 1 / (2 * pi * f_c)
        dt    = 1 / f_s
        alpha = (2 * pi * f_c * dt) / (2 * pi * f_c * dt + 1)
              = 1 / (1 + tau * f_s)
    """
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau * f_s)


# ---------------------------------------------------------------------------
# 1-D scalar filter
# ---------------------------------------------------------------------------


class OneEuroFilter1D:
    """One Euro Filter applied to a scalar (1-D) signal.

    Mathematically rigorous implementation of Casiez et al. (2012).
    The filter applies an EMA whose cutoff frequency adapts to signal speed:

        f_c   = min_cutoff + beta * |dx_hat|
        alpha = 1 / (1 + tau * f_s)   where tau = 1 / (2 * pi * f_c)
        x_hat = alpha * x + (1 - alpha) * x_hat_prev

    The velocity estimate dx_hat is itself smoothed by a fixed-cutoff EMA
    at d_cutoff Hz, as specified in Appendix A of the paper.

    Parameters
    ----------
    f_s : float
        Sample rate in Hz (frames per second of the signal).
    min_cutoff : float
        Minimum cutoff frequency in Hz. Controls jitter suppression at rest.
        Lower value -> more smoothing when the signal is stationary.
    beta : float
        Speed coefficient. Higher value -> faster response during fast motion,
        reducing lag. Set to 0 to get a plain fixed-cutoff EMA.
    d_cutoff : float
        Fixed cutoff frequency for the derivative (velocity) EMA. Default 1 Hz
        as recommended by the authors. Rarely needs tuning.
    """

    def __init__(
        self,
        f_s: float,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ) -> None:
        self.f_s = f_s
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_hat: float | None = None
        self._dx_hat: float = 0.0

    # ------------------------------------------------------------------
    def __call__(self, x: float) -> float:
        """Feed measurement *x* and return the filtered estimate."""
        if self._x_hat is None:
            self._x_hat = x
            return x

        # Raw derivative (finite difference at rate f_s)
        dx_raw = (x - self._x_hat) * self.f_s

        # Smooth the derivative with a fixed EMA
        a_d = alpha_from_cutoff(self.d_cutoff, self.f_s)
        self._dx_hat = a_d * dx_raw + (1.0 - a_d) * self._dx_hat

        # Adaptive cutoff driven by smoothed speed
        f_c = self.min_cutoff + self.beta * abs(self._dx_hat)
        a = alpha_from_cutoff(f_c, self.f_s)

        # Position EMA
        self._x_hat = a * x + (1.0 - a) * self._x_hat
        return self._x_hat

    def reset(self) -> None:
        self._x_hat = None
        self._dx_hat = 0.0


# ---------------------------------------------------------------------------
# Multi-dimensional wrappers
# ---------------------------------------------------------------------------


class OneEuroFilter2D:
    """One Euro Filter for a 2-D coordinate (x, y).

    Wraps two independent OneEuroFilter1D instances sharing the same
    hyperparameters, as each coordinate axis is treated independently.
    """

    def __init__(
        self,
        f_s: float,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ) -> None:
        kw = dict(f_s=f_s, min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
        self._fx = OneEuroFilter1D(**kw)
        self._fy = OneEuroFilter1D(**kw)

    def __call__(self, x: float, y: float) -> tuple[float, float]:
        return self._fx(x), self._fy(y)

    def reset(self) -> None:
        self._fx.reset()
        self._fy.reset()


class OneEuroFilter3D:
    """One Euro Filter for a 3-D coordinate (x, y, z)."""

    def __init__(
        self,
        f_s: float,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ) -> None:
        kw = dict(f_s=f_s, min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
        self._fx = OneEuroFilter1D(**kw)
        self._fy = OneEuroFilter1D(**kw)
        self._fz = OneEuroFilter1D(**kw)

    def __call__(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        return self._fx(x), self._fy(y), self._fz(z)

    def reset(self) -> None:
        self._fx.reset()
        self._fy.reset()
        self._fz.reset()
