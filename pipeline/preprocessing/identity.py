"""Identity preprocessor — passes the frame through unchanged (no-op baseline)."""

from core.base import Preprocessor
from core.registry import preprocessors


@preprocessors.register
class Identity(Preprocessor):
    name = "identity"
    description = "Passes the frame through unchanged."

    @classmethod
    def parameter_specs(cls):
        """Return the list of tunable parameters for this component."""
        return []

    def __call__(self, frame):
        """Return *frame* unmodified.

        Args:
            frame: RGB image, shape (H, W, 3), dtype uint8.

        Returns:
            The same array, untouched.
        """
        return frame
