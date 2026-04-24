"""
Wrapper pour le modèle MediaPipe Selfie Segmenter (Portrait et Landscape).
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import urllib.request

from .base import BaseModelWrapper

logger = logging.getLogger(__name__)

os.environ['GLOG_minloglevel'] = '2'

_MODEL_URLS = {
    "portrait": (
        "https://storage.googleapis.com/mediapipe-models/"
        "image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"
    ),
    "landscape": (
        "https://storage.googleapis.com/mediapipe-models/"
        "image_segmenter/selfie_segmenter_landscape/float16/latest/"
        "selfie_segmenter_landscape.tflite"
    ),
}

class BaseMediapipeWrapper(BaseModelWrapper):
    _variant: str = ""
    _segmenter = None
    _frame_count = 0
    _prev_mask = None
    use_ema = False
    ema_alpha = 0.6
    use_morphology = True

    @property
    def input_size(self) -> Optional[Tuple[int, int]]:
        return (256, 256)

    def load(self) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import ImageSegmenter, ImageSegmenterOptions, RunningMode
        except ImportError as e:
            raise ImportError("mediapipe est requis. Installe-le via : pip install mediapipe") from e

        weights_dir = Path(__file__).parent.parent / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        
        model_filename = _MODEL_URLS[self._variant].split("/")[-1]
        local_path = weights_dir / model_filename

        if not local_path.exists() or local_path.stat().st_size == 0:
            urllib.request.urlretrieve(_MODEL_URLS[self._variant], str(local_path))

        options = ImageSegmenterOptions(
            base_options=BaseOptions(
                model_asset_path=str(local_path),
                delegate=BaseOptions.Delegate.GPU
            ),
            running_mode=RunningMode.VIDEO, 
            output_category_mask=False,
            output_confidence_masks=True,
        )
        self._segmenter = ImageSegmenter.create_from_options(options)
        self._mp = mp
        self._frame_count = 0

    def reset_state(self) -> None:
        self._frame_count = 0
        self._prev_mask = None
        if self._segmenter is not None:
            try: self._segmenter.close()
            except: pass
            self._segmenter = None
            self.load()

    def get_flops(self, input_shape: Tuple[int, int, int] = (3, 256, 256)) -> float:
        estimates = {"portrait": 7.5e6, "selfie_multiclass": 9.2e6, "landscape": 8.1e6}
        return estimates.get(self._variant, 8.0e6)

    def predict(self, frame_bgr: np.ndarray, frame_rgb: Optional[np.ndarray] = None) -> np.ndarray:
        if self._segmenter is None: return None
        try:
            h_orig, w_orig = frame_bgr.shape[:2]
            # 1. Pre-processing: Utilisation de INTER_AREA (meilleure qualité pour la réduction)
            frame_small = cv2.resize(frame_bgr, (256, 256), interpolation=cv2.INTER_AREA)
            frame_small_rgba = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGBA)
            
            mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGBA, data=frame_small_rgba)
            ts = int(self._frame_count * (1000 / 30))
            
            result = self._segmenter.segment_for_video(mp_image, ts)
            self._frame_count += 1

            if result and result.confidence_masks:
                # Pour selfie_multiclass, on somme précisément les canaux humains :
                # 1: Hair, 2: Body, 3: Face, 4: Clothes.
                # On ignore Background(0) et Others(5) pour maximiser l'IoU.
                if self._variant == "selfie_multiclass" and len(result.confidence_masks) >= 5:
                    masks = [result.confidence_masks[i].numpy_view() for i in range(1, 5)]
                    mask_small = np.sum(masks, axis=0)
                    mask_small = np.clip(mask_small, 0.0, 1.0)
                else:
                    mask_small = result.confidence_masks[0].numpy_view()

                # 2. Nettoyage Spatial
                # 2a. Morphologie (Fermeture) pour boucher les trous (logos, reflets)
                if getattr(self, 'use_morphology', True):
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_CLOSE, kernel)

                # 2b. CCA (Connected Component Analysis)
                if getattr(self, 'use_cca', False):
                    binary_mask = (mask_small > 0.5).astype(np.uint8)
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
                    
                    if num_labels > 1: # S'il y a des composants détectés en plus du fond
                        # Conserver les composants qui font au moins 5% de la surface du masque (ex: plusieurs personnes)
                        min_area = (256 * 256) * 0.05
                        valid_labels = [i for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= min_area]
                        
                        # Si aucun composant n'atteint 5%, on garde forcémenent le plus gros
                        if not valid_labels:
                            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                            valid_labels = [largest_label]

                        valid_mask = np.zeros_like(mask_small)
                        for label in valid_labels:
                            valid_mask[labels == label] = 1.0
                            
                        # Appliquer le filtrage CCA sur le masque doux
                        mask_small = mask_small * valid_mask
                
                # Application de l'EMA pour stabiliser le masque
                if self.use_ema:
                    if self._prev_mask is None or self._prev_mask.shape != mask_small.shape:
                        self._prev_mask = mask_small.copy()
                    else:
                        mask_small = self.ema_alpha * mask_small + (1 - self.ema_alpha) * self._prev_mask
                        self._prev_mask = mask_small.copy()

                # 3. Affinement et Upscale (Fast Guided Filter Optimisé)
                if getattr(self, 'use_guided_filter', False):
                    try:
                        # OPTIMISATION MAJEURE: Le Guided Filter sur l'image originale (1080p) détruit les FPS.
                        # Technique du "Fast Guided Filter" : on fait le calcul sur une échelle réduite (max 512px)
                        max_dim = 512
                        scale = min(1.0, max_dim / max(h_orig, w_orig))
                        
                        if scale < 1.0:
                            w_gf, h_gf = int(w_orig * scale), int(h_orig * scale)
                            # On réduit l'image HD pour faire un guide léger
                            guide_small = cv2.resize(frame_bgr, (w_gf, h_gf), interpolation=cv2.INTER_AREA)
                            # On redimensionne le masque à la taille du guide
                            mask_gf = cv2.resize(mask_small, (w_gf, h_gf), interpolation=cv2.INTER_LINEAR).astype(np.float32)
                            
                            # Filtre guidé sur la petite résolution (très rapide, ~2-5ms)
                            gf = cv2.ximgproc.createGuidedFilter(guide_small, radius=5, eps=1e-4)
                            mask_gf = gf.filter(mask_gf)
                            
                            # Upscale final vers la taille originale
                            mask_large = cv2.resize(mask_gf, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR).astype(np.float32)
                        else:
                            # Si l'image est déjà petite, on le fait direct
                            mask_large = cv2.resize(mask_small, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR).astype(np.float32)
                            gf = cv2.ximgproc.createGuidedFilter(frame_bgr, radius=5, eps=1e-4)
                            mask_large = gf.filter(mask_large)
                            
                    except AttributeError:
                        # Fallback si ximgproc absent
                        mask_large = cv2.resize(mask_small, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR).astype(np.float32)
                else:
                    # Upscale classique si pas de Guided Filter
                    mask_large = cv2.resize(mask_small, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR).astype(np.float32)
                
                return np.clip(mask_large, 0.0, 1.0)
            
            # Fallback en cas d'absence de résultat
            return np.zeros((h_orig, w_orig), dtype=np.float32)
        except Exception as e:
            logger.error(f"Erreur interne MediaPipe ({self._variant}): {e}")
            return None

    def cleanup(self) -> None:
        if self._segmenter: self._segmenter.close()
        self._segmenter = None


class MediapipePortraitWrapper(BaseMediapipeWrapper):
    _variant = "portrait"
    @property
    def name(self) -> str: return "MediaPipe Portrait"

class MediapipeLandscapeWrapper(BaseMediapipeWrapper):
    _variant = "landscape"
    @property
    def name(self) -> str: return "MediaPipe Landscape"
