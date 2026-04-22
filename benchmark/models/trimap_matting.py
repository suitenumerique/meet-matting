"""
Wrapper for the Trimap-based Matting approach (classical / hybrid).

Two-stage hybrid pipeline:
  1. Automatic trimap generation from a lightweight segmenter.
  2. Trimap-guided alpha matting via the classical algorithm of Levin et al.
     or a matting network (if available).

This implementation uses OpenCV's GrabCut as a robust approximation of
trimap-guided matting, which avoids extra dependencies while providing
a solid baseline.
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from .base import BaseModelWrapper

logger = logging.getLogger(__name__)


class TrimapMattingWrapper(BaseModelWrapper):
    """
    Hybrid matting based on an auto-generated trimap + GrabCut.

    The trimap is built by:
      1. Otsu thresholding on the luminance channel → initial mask.
      2. Erosion → "definite foreground" region.
      3. Dilation → the difference yields the "probably foreground" region.
      4. GrabCut refines the mask.
    """

    def __init__(self, erosion_size: int = 10, dilation_size: int = 20):
        self._erosion_size = erosion_size
        self._dilation_size = dilation_size

    @property
    def name(self) -> str:
        return "Trimap Matting (GrabCut)"

    @property
    def input_size(self) -> Optional[Tuple[int, int]]:
        return None  # Native size

    def load(self) -> None:
        # No model to load — classical OpenCV algorithm
        logger.info("TrimapMatting: initialised (OpenCV GrabCut).")

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]

        # ── Step 1: Coarse segmentation by thresholding ──
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        # Bilateral filter to smooth while preserving edges
        gray_filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        _, binary = cv2.threshold(
            gray_filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # ── Step 2: Trimap generation ──
        kernel_erode = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self._erosion_size, self._erosion_size),
        )
        kernel_dilate = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self._dilation_size, self._dilation_size),
        )

        fg_sure = cv2.erode(binary, kernel_erode)
        bg_sure = cv2.bitwise_not(cv2.dilate(binary, kernel_dilate))

        # Build the GrabCut mask
        # 0 = definite BG, 1 = definite FG, 2 = probable BG, 3 = probable FG
        grabcut_mask = np.full((h, w), cv2.GC_PR_FGD, dtype=np.uint8)
        grabcut_mask[bg_sure > 0] = cv2.GC_BGD
        grabcut_mask[fg_sure > 0] = cv2.GC_FGD

        # ── Step 3: GrabCut ──
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)

        try:
            cv2.grabCut(
                frame_bgr,
                grabcut_mask,
                None,
                bgd_model,
                fgd_model,
                iterCount=3,
                mode=cv2.GC_INIT_WITH_MASK,
            )
        except cv2.error as e:
            logger.warning("GrabCut failed: %s. Falling back to the binary mask.", e)
            return (binary / 255.0).astype(np.float32)

        # Final mask: definite FG or probable FG
        result_mask = np.where(
            (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
            1.0,
            0.0,
        ).astype(np.float32)

        return result_mask

    def get_flops(self, input_shape: Tuple[int, int, int] = (3, 256, 256)) -> float:
        # GrabCut has no conventional DNN FLOPs
        # Return -1 to indicate "not applicable"
        return -1.0

    def cleanup(self) -> None:
        logger.info("TrimapMatting: no resources to release.")
