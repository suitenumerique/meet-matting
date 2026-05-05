"""
Kalman filter (Constant Velocity model) applied pixelwise to the segmentation mask.

Each pixel is treated as an independent 1D signal with state [probability, velocity].
All operations are vectorized over (H, W) with NumPy -- no Python loops per frame.

Reference: Welch & Bishop (2006), "An Introduction to the Kalman Filter", UNC-CH TR 95-041.
"""

import numpy as np
from core.base import Postprocessor
from core.parameters import ParameterSpec
from core.registry import postprocessors


@postprocessors.register
class KalmanMask(Postprocessor):
    name = "kalman"
    description = "Kalman filter (CV model): estimates mask probability and its rate of change."
    details = (
        "Principe : filtre optimal qui maintient deux etats par pixel -- la probabilite "
        "courante et sa vitesse de variation -- et les corrige a chaque frame.\n"
        "- Predict : le filtre anticipe la prochaine valeur grace au modele physique.\n"
        "- Update : la mesure du modele corrige la prediction via le gain de Kalman.\n"
        "q_pos : bruit de processus sur la probabilite (augmenter si le masque change vite).\n"
        "q_vel : bruit de processus sur la vitesse (augmenter pour adapter plus vite).\n"
        "r_mes : bruit de mesure (augmenter pour lisser plus, diminuer pour suivre plus)."
    )

    def __init__(self, **params):
        """Initialise with params and allocate internal buffers."""
        super().__init__(**params)
        # State estimates -- all (H, W) float32.
        self._p: np.ndarray | None = None  # probability estimate
        self._v: np.ndarray | None = None  # velocity estimate
        # Upper-triangular of the 2x2 symmetric covariance matrix P.
        self._P00: np.ndarray | None = None
        self._P01: np.ndarray | None = None
        self._P11: np.ndarray | None = None

    @classmethod
    def parameter_specs(cls):
        """Return the list of tunable parameters for this component."""
        return [
            ParameterSpec(
                name="q_pos",
                type="float",
                default=0.01,
                label="Process noise -- position (q_pos)",
                min_value=0.0001,
                max_value=0.5,
                step=0.001,
                help="How much the probability can drift between frames beyond velocity. "
                "Higher = less prediction confidence = faster adaptation.",
            ),
            ParameterSpec(
                name="q_vel",
                type="float",
                default=0.001,
                label="Process noise -- velocity (q_vel)",
                min_value=0.0001,
                max_value=0.1,
                step=0.0001,
                help="How much velocity can change between frames. "
                "Keep small for slowly-varying masks.",
            ),
            ParameterSpec(
                name="r_mes",
                type="float",
                default=0.1,
                label="Measurement noise (r_mes)",
                min_value=0.001,
                max_value=1.0,
                step=0.001,
                help="Trust in the raw model output. Higher = more smoothing, lower = follows measurement.",
            ),
        ]

    def reset(self):
        """Clear all per-pixel state so the filter re-initialises on the next frame."""
        self._p = None
        self._v = None
        self._P00 = None
        self._P01 = None
        self._P11 = None

    def __call__(self, mask: np.ndarray, original_frame: np.ndarray) -> np.ndarray:
        """Run one Kalman predict-update step on *mask*.

        Args:
            mask:           Alpha matte, shape (H, W), dtype float32, range [0, 1].
            original_frame: Original RGB frame, shape (H, W, 3), dtype uint8 (unused).

        Returns:
            Filtered mask, shape (H, W), dtype float32, range [0, 1].
        """
        q_pos = self.params["q_pos"]
        q_vel = self.params["q_vel"]
        r_mes = self.params["r_mes"]

        if self._p is None or self._p.shape != mask.shape:
            self._p = mask.copy()
            self._v = np.zeros_like(mask)
            # Start with high uncertainty so the filter converges from the first measurement.
            self._P00 = np.ones_like(mask)
            self._P01 = np.zeros_like(mask)
            self._P11 = np.ones_like(mask)
            return mask

        assert self._v is not None
        assert self._P00 is not None
        assert self._P01 is not None
        assert self._P11 is not None

        # ── PREDICT ───────────────────────────────────────────────────────────
        # State transition: F = [[1, 1], [0, 1]] (dt = 1 frame).
        p_pred = self._p + self._v
        v_pred = self._v

        # Covariance predict: P_pred = F * P * F^T + Q.
        # With F = [[1,1],[0,1]] and Q = diag(q_pos, q_vel):
        P00_p = self._P00 + self._P01 + self._P01 + self._P11 + q_pos
        P01_p = self._P01 + self._P11
        P11_p = self._P11 + q_vel

        # ── UPDATE ────────────────────────────────────────────────────────────
        # Observation model: H = [1, 0], measurement z = mask.
        # Innovation covariance: S = P00_p + r.
        S = P00_p + r_mes

        # Kalman gain: K = P_pred * H^T / S = [P00_p, P01_p]^T / S.
        K0 = P00_p / S  # gain for position
        K1 = P01_p / S  # gain for velocity

        # Innovation.
        y = mask - p_pred

        # Updated state.
        p_new = p_pred + K0 * y
        v_new = v_pred + K1 * y

        # Updated covariance: P = (I - K*H) * P_pred.
        one_minus_K0 = 1.0 - K0
        P00_new = one_minus_K0 * P00_p
        P01_new = one_minus_K0 * P01_p
        P11_new = P11_p - K1 * P01_p

        self._p = p_new
        self._v = v_new
        self._P00 = P00_new
        self._P01 = P01_new
        self._P11 = P11_new

        return np.clip(p_new, 0.0, 1.0).astype(np.float32)
