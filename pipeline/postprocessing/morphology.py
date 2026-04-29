"""
Morphological mask cleanup.

Reference: van Herk (1992); Gil & Werman (1993) -- efficient morphological ops.
OpenCV: cv2.morphologyEx with MORPH_CLOSE and MORPH_OPEN.

Operations:
  Closing  = Dilation then Erosion -> fills small internal holes.
  Opening  = Erosion then Dilation -> removes isolated peripheral noise.

Kernel size of 0 disables the corresponding operation.
Both operations can be applied in either order (close-first or open-first).
"""
from __future__ import annotations

import cv2
import numpy as np
from core.base import Postprocessor
from core.parameters import ParameterSpec
from core.registry import postprocessors

# Map readable names to cv2 morphological shape constants.
_SHAPE_MAP: dict[str, int] = {
    "ELLIPSE": cv2.MORPH_ELLIPSE,
    "RECT":    cv2.MORPH_RECT,
    "CROSS":   cv2.MORPH_CROSS,
}


def _make_kernel(size: int, shape: str) -> np.ndarray:
    """Return a structuring element of the given odd *size* and *shape*."""
    size = max(1, size | 1)   # force odd value; minimum 1
    return cv2.getStructuringElement(_SHAPE_MAP[shape], (size, size))


@postprocessors.register
class MorphologyCleanup(Postprocessor):
    name = "morphology"
    description = (
        "Closes internal holes then removes peripheral noise (configurable kernel sizes)."
    )
    details = (
        "Reference: van Herk (1992) / Gil & Werman (1993).\n"
        "Closing (dilation then erosion): fills small gaps inside the mask.\n"
        "Opening (erosion then dilation): removes small isolated blobs outside the mask.\n"
        "Set either kernel size to 0 to disable that operation.\n"
        "Operation order: 'close_open' applies closing first, 'open_close' applies opening first."
    )

    @classmethod
    def parameter_specs(cls) -> list[ParameterSpec]:
        return [
            ParameterSpec(
                name="close_size",
                type="int",
                default=5,
                label="Closing kernel size",
                min_value=0,
                max_value=21,
                step=2,
                help="Size of the structuring element for Closing. 0 = disabled. Use odd values.",
            ),
            ParameterSpec(
                name="open_size",
                type="int",
                default=3,
                label="Opening kernel size",
                min_value=0,
                max_value=21,
                step=2,
                help="Size of the structuring element for Opening. 0 = disabled. Use odd values.",
            ),
            ParameterSpec(
                name="kernel_shape",
                type="choice",
                default="ELLIPSE",
                label="Kernel shape",
                choices=["ELLIPSE", "RECT", "CROSS"],
                help="Shape of the structuring element used for both operations.",
            ),
            ParameterSpec(
                name="iterations",
                type="int",
                default=1,
                label="Iterations",
                min_value=1,
                max_value=5,
                step=1,
                help="Number of times each morphological operation is applied.",
            ),
            ParameterSpec(
                name="order",
                type="choice",
                default="close_open",
                label="Operation order",
                choices=["close_open", "open_close"],
                help="'close_open': closing first then opening. 'open_close': opening first.",
            ),
        ]

    def reset(self) -> None:
        pass

    def __call__(self, mask: np.ndarray, original_frame: np.ndarray) -> np.ndarray:
        close_size = int(self.params["close_size"])
        open_size  = int(self.params["open_size"])
        shape      = self.params["kernel_shape"]
        iters      = int(self.params["iterations"])
        order      = self.params["order"]

        if close_size == 0 and open_size == 0:
            return mask

        # Convert float32 [0,1] -> uint8 [0,255] for OpenCV morphology ops.
        m_u8 = (mask * 255.0).astype(np.uint8)

        def close_op(img: np.ndarray) -> np.ndarray:
            k = _make_kernel(close_size, shape)
            return cv2.morphologyEx(img, cv2.MORPH_CLOSE, k, iterations=iters)

        def open_op(img: np.ndarray) -> np.ndarray:
            k = _make_kernel(open_size, shape)
            return cv2.morphologyEx(img, cv2.MORPH_OPEN, k, iterations=iters)

        if order == "close_open":
            if close_size > 0:
                m_u8 = close_op(m_u8)
            if open_size > 0:
                m_u8 = open_op(m_u8)
        else:
            if open_size > 0:
                m_u8 = open_op(m_u8)
            if close_size > 0:
                m_u8 = close_op(m_u8)

        return m_u8.astype(np.float32) * (1.0 / 255.0)
