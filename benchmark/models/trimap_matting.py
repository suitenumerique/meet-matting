"""
Wrapper pour l'approche Trimap-based Matting (classique/hybride).

Pipeline hybride en 2 étapes :
  1. Génération automatique d'un trimap à partir d'un segmenteur léger.
  2. Alpha matting guidé par le trimap via l'algorithme classique de Levin et al.
     ou un réseau de matting (si disponible).

Cette implémentation utilise GrabCut d'OpenCV comme approximation
robuste du matting guidé par trimap, ce qui évite des dépendances
supplémentaires tout en fournissant un baseline solide.
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from .base import BaseModelWrapper

logger = logging.getLogger(__name__)


class TrimapMattingWrapper(BaseModelWrapper):
    """
    Matting hybride basé sur un trimap auto-généré + GrabCut.

    Le trimap est construit par :
      1. Seuillage Otsu sur le canal de luminance → masque initial.
      2. Érosion → zone « definite foreground ».
      3. Dilatation → la différence donne la zone « probably foreground ».
      4. GrabCut affine le masque.
    """

    def __init__(self, erosion_size: int = 10, dilation_size: int = 20):
        self._erosion_size = erosion_size
        self._dilation_size = dilation_size

    @property
    def name(self) -> str:
        return "Trimap Matting (GrabCut)"

    @property
    def input_size(self) -> Optional[Tuple[int, int]]:
        return None  # Taille native

    def load(self) -> None:
        # Pas de modèle à charger — algorithme classique OpenCV
        logger.info("TrimapMatting: initialisé (OpenCV GrabCut).")

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]

        # ── Étape 1 : Segmentation grossière par seuillage ──
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        # Filtre bilatéral pour lisser tout en préservant les bords
        gray_filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        _, binary = cv2.threshold(
            gray_filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # ── Étape 2 : Génération du trimap ──
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

        # Construire le masque GrabCut
        # 0 = BG défini, 1 = FG défini, 2 = BG probable, 3 = FG probable
        grabcut_mask = np.full((h, w), cv2.GC_PR_FGD, dtype=np.uint8)
        grabcut_mask[bg_sure > 0] = cv2.GC_BGD
        grabcut_mask[fg_sure > 0] = cv2.GC_FGD

        # ── Étape 3 : GrabCut ──
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
            logger.warning("GrabCut a échoué: %s. Retour au masque binaire.", e)
            return (binary / 255.0).astype(np.float32)

        # Masque final : FG défini ou FG probable
        result_mask = np.where(
            (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
            1.0,
            0.0,
        ).astype(np.float32)

        return result_mask

    def get_flops(self, input_shape: Tuple[int, int, int] = (3, 256, 256)) -> float:
        # GrabCut n'a pas de FLOPs DNN conventionnels
        # On retourne -1 pour indiquer "non applicable"
        return -1.0

    def cleanup(self) -> None:
        logger.info("TrimapMatting: aucune ressource à libérer.")
