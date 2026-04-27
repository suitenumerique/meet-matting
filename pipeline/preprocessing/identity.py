from core.base import Preprocessor
from core.registry import preprocessors


@preprocessors.register
class Identity(Preprocessor):
    name = "identity"
    description = "Passes the frame through unchanged."

    @classmethod
    def parameter_specs(cls):
        return []

    def __call__(self, frame):
        """Return *frame* unmodified.

        Args:
            frame: RGB image, shape (H, W, 3), dtype uint8.

        Returns:
            The same array, untouched.
        """
        return frame
