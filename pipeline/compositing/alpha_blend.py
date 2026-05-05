"""Alpha blend compositor — standard over-operator: FG·α + BG·(1−α)."""

import numpy as np
from core.base import Compositor
from core.registry import compositors


@compositors.register
class AlphaBlend(Compositor):
    name = "alpha_blend"
    description = "Fusion classique FG·α + BG·(1−α). Rapide et sans artefact."

    @classmethod
    def parameter_specs(cls):
        """Return the list of tunable parameters for this component."""
        return []

    def composite(self, fg: np.ndarray, bg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Composite *fg* over *bg* using *alpha* via the standard lerp formula.

        Args:
            fg:    Foreground RGB frame, shape (H, W, 3), dtype uint8.
            bg:    Background, shape (H, W, 3), dtype float32, range [0, 255].
            alpha: Alpha matte, shape (H, W), dtype float32, range [0, 1].

        Returns:
            Composited image, shape (H, W, 3), dtype uint8.
        """
        # Lerp formula: bg + (fg - bg) * alpha
        # All in-place on fg_f to avoid extra allocations.
        # fg is uint8 → explicit float32 cast prevents numpy from upcasting to float64.
        fg_f = fg.astype(np.float32)  # [0, 255]
        mask3 = alpha[..., np.newaxis]
        fg_f -= bg  # fg - bg
        fg_f *= mask3  # (fg - bg) * alpha
        fg_f += bg  # bg + (fg - bg) * alpha  →  lerp result
        np.clip(fg_f, 0.0, 255.0, out=fg_f)
        return fg_f.astype(np.uint8)
