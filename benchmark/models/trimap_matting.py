import logging
import time
from typing import Optional, Tuple
from pathlib import Path

import cv2
import numpy as np

from .base import BaseModelWrapper
from .mediapipe_selfie import MediapipeLandscapeWrapper
from .mediapipe_pose import MediapipePoseWrapper

logger = logging.getLogger(__name__)

class TrimapMattingWrapper(BaseModelWrapper):
    """
    Trimap Matting V3 (Limb-Lock) - Optimisé pour la latence (240p).
    """

    def __init__(self, 
                 work_size: int = 240,  # Passage en 240p pur pour latence minimale
                 erosion_size: int = 3,  # Ajusté pour 240p
                 dilation_size: int = 8,  # Ajusté pour 240p
                 limb_thickness: int = 8,
                 refine_radius: int = 4,   # Réduit pour la vitesse
                 refine_eps: float = 0.05, # Plus stable et rapide
                 ema_alpha: float = 0.7):
        self._work_size = work_size
        self._erosion_size = erosion_size
        self._dilation_size = dilation_size
        self._limb_thickness = limb_thickness
        self._refine_radius = refine_radius
        self._refine_eps = refine_eps
        self._ema_alpha = ema_alpha

        self._semantic_model = MediapipeLandscapeWrapper()
        self._pose_model = MediapipePoseWrapper()
        self._prev_mask = None

    @property
    def name(self) -> str:
        return "Trimap Matting V3 (Limb-Lock) - 240p"

    @property
    def input_size(self) -> Optional[Tuple[int, int]]:
        return None

    def load(self) -> None:
        self._semantic_model.load()
        try:
            self._pose_model.load()
        except Exception as e:
            logger.warning("Pose model failed to load: %s. Limb-Lock disabled.", e)
            self._pose_model = None

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        if frame_bgr is None: return None
        h_orig, w_orig = frame_bgr.shape[:2]

        # ── Étape 1 : Préparation Low-Res & RGB (Vitesse) ──
        scaling = self._work_size / max(h_orig, w_orig)
        w_work, h_work = int(w_orig * scaling), int(h_orig * scaling)
        frame_work = cv2.resize(frame_bgr, (w_work, h_work), interpolation=cv2.INTER_AREA)
        frame_work_rgb = cv2.cvtColor(frame_work, cv2.COLOR_BGR2RGB)

        # ── Étape 2 : Inférence Parallèle (Sémantique + Pose) ──
        # On passe frame_work_rgb pour éviter une conversion interne redondante
        mask_semantic = self._semantic_model.predict(frame_work, frame_rgb=frame_work_rgb)
        
        limb_mask = 0
        if self._pose_model:
            limb_mask = self._pose_model.get_limb_mask(frame_work, thickness=self._limb_thickness, frame_rgb=frame_work_rgb)

        # ── Étape 3 : Trimap & GrabCut (Low-Res) ──
        binary_mask = (mask_semantic > 0.5).astype(np.uint8) * 255
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self._erosion_size, self._erosion_size))
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self._dilation_size, self._dilation_size))

        fg_sure = cv2.erode(binary_mask, kernel_erode)
        bg_sure = cv2.bitwise_not(cv2.dilate(binary_mask, kernel_dilate))

        grabcut_mask = np.full((h_work, w_work), cv2.GC_PR_FGD, dtype=np.uint8)
        grabcut_mask[bg_sure > 0] = cv2.GC_BGD
        grabcut_mask[fg_sure > 0] = cv2.GC_FGD
        
        # Injection Limb-Lock (Foreground Certain)
        if isinstance(limb_mask, np.ndarray):
            grabcut_mask[limb_mask > 0] = cv2.GC_FGD

        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)
        
        try:
            cv2.grabCut(frame_work, grabcut_mask, None, bgd_model, fgd_model, iterCount=1, mode=cv2.GC_INIT_WITH_MASK)
            low_res_refined = np.where((grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD), 1.0, 0.0).astype(np.float32)
            
            # Garde-fou anti-vide
            if low_res_refined.mean() < 0.01: low_res_refined = mask_semantic
        except:
            low_res_refined = mask_semantic

        # ── Étape 4 : Guided Filter Ultra-Fast ──
        # On limite le guidage à 320px max. Au-delà, le coût CPU explose pour peu de gain visuel en 240p.
        limit_size = 320 
        if max(h_orig, w_orig) > limit_size:
            s_opt = limit_size / max(h_orig, w_orig)
            w_opt, h_opt = int(w_orig * s_opt), int(h_orig * s_opt)
            guide = cv2.cvtColor(cv2.resize(frame_bgr, (w_opt, h_opt), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            src = cv2.resize(low_res_refined, (w_opt, h_opt), interpolation=cv2.INTER_LINEAR)
            mask_final = cv2.ximgproc.guidedFilter(guide, src, self._refine_radius, self._refine_eps)
            mask_final = cv2.resize(mask_final, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        else:
            mask_refined = cv2.resize(low_res_refined, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
            guide = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            mask_final = cv2.ximgproc.guidedFilter(guide, mask_refined, self._refine_radius, self._refine_eps)

        # ── Étape 5 : Lissage & Clipping ──
        mask_final = np.nan_to_num(mask_final, nan=0.0)
        mask_final = np.clip(mask_final, 0.0, 1.0)

        if self._prev_mask is not None and self._prev_mask.shape == mask_final.shape:
            mask_final = cv2.addWeighted(mask_final, self._ema_alpha, self._prev_mask, 1.0 - self._ema_alpha, 0)
        
        self._prev_mask = mask_final.copy()
        return mask_final

    def get_flops(self, input_shape: Tuple[int, int, int] = (3, 256, 256)) -> float:
        return 22.0e6 # Optimisé par rapport à la V2

    def cleanup(self) -> None:
        self._semantic_model.cleanup()
        if self._pose_model: self._pose_model.cleanup()
        self._prev_mask = None
