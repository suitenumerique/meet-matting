"""Identity postprocessor — passes the mask through unchanged (no-op baseline)."""

from core.base import Postprocessor
from core.registry import postprocessors


@postprocessors.register
class Identity(Postprocessor):
    name = "identity"
    description = "Passes the mask through unchanged."

    @classmethod
    def parameter_specs(cls):
        """Return the list of tunable parameters for this component."""
        return []

    def __call__(self, mask, original_frame):
        """Return *mask* unmodified.

        Args:
            mask:           Alpha matte, shape (H, W), dtype float32, range [0, 1].
            original_frame: Original RGB frame, shape (H, W, 3), dtype uint8 (unused).

        Returns:
            The same *mask* array, untouched.
        """
        return mask
