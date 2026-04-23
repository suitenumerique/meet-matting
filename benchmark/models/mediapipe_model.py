"""
Wrappers pour les modèles MediaPipe de segmentation et pose.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .base import BaseModelWrapper

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
#  URLs des modèles MediaPipe
# ──────────────────────────────────────────────
_MODEL_URLS = {
    "portrait": (
        "https://storage.googleapis.com/mediapipe-models/"
        "image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"
    ),
    "selfie_multiclass": (
        "https://storage.googleapis.com/mediapipe-models/"
        "image_segmenter/selfie_multiclass_256x256/float32/latest/"
        "selfie_multiclass_256x256.tflite"
    ),
    "landscape": (
        "https://storage.googleapis.com/mediapipe-models/"
        "image_segmenter/selfie_segmenter_landscape/float16/latest/"
        "selfie_segmenter_landscape.tflite"
    ),
    "pose_lite": (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    ),
}

# Connexions anatomiques bras + corps (indices MediaPipe Pose Landmarker)
_LIMB_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 25), (25, 27), (24, 26), (26, 28),
]


class _BaseMediapipeWrapper(BaseModelWrapper):
    _variant: str = ""
    _segmenter = None

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
            import urllib.request
            urllib.request.urlretrieve(_MODEL_URLS[self._variant], str(local_path))

        options = ImageSegmenterOptions(
            base_options=BaseOptions(
                model_asset_path=str(local_path),
                delegate=BaseOptions.Delegate.GPU # On retente le GPU avec le fix RGBA
            ),
            running_mode=RunningMode.VIDEO, 
            output_category_mask=False,
            output_confidence_masks=True,
        )
        self._segmenter = ImageSegmenter.create_from_options(options)
        self._mp = mp
        self.reset_state()

    def reset_state(self) -> None:
        """Remet à zéro le compteur pour le mode VIDEO."""
        self._frame_count = 0

    def get_flops(self, input_shape: Tuple[int, int, int] = (3, 256, 256)) -> float:
        estimates = {"portrait": 7.5e6, "selfie_multiclass": 9.2e6, "landscape": 8.1e6}
        return estimates.get(self._variant, 8.0e6)

    def predict(self, frame_bgr: np.ndarray, frame_rgb: Optional[np.ndarray] = None) -> np.ndarray:
        if self._segmenter is None: return None
        h_orig, w_orig = frame_bgr.shape[:2]
        
        # FIX GPU : MediaPipe GPU sur Mac exige souvent du RGBA (4 canaux)
        frame_small = cv2.resize(frame_bgr, (256, 256), interpolation=cv2.INTER_NEAREST)
        frame_small_rgba = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGBA)
            
        # Utilisation du format SRGBA pour correspondre au delegate GPU
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGBA, data=frame_small_rgba)
        
        # On utilise le timestamp pour le RunningMode.VIDEO (indispensable en GPU)
        timestamp_ms = int(self._frame_count * (1000 / 30))
        result = self._segmenter.segment_for_video(mp_image, timestamp_ms)
        self._frame_count += 1

        if result.confidence_masks:
            if self._variant == "selfie_multiclass" and len(result.confidence_masks) > 1:
                mask_small = 1.0 - result.confidence_masks[0].numpy_view()
            else:
                mask_small = result.confidence_masks[0].numpy_view()
        else:
            mask_small = np.zeros((256, 256), dtype=np.float32)

        if mask_small.shape[:2] != (h_orig, w_orig):
            mask = cv2.resize(mask_small, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        else:
            mask = mask_small
            
        return mask.astype(np.float32)

    def cleanup(self) -> None:
        if self._segmenter: self._segmenter.close()
        self._segmenter = None


class MediapipePortraitWrapper(_BaseMediapipeWrapper):
    _variant = "portrait"
    @property
    def name(self) -> str: return "MediaPipe Portrait"

class MediapipeLandscapeWrapper(_BaseMediapipeWrapper):
    _variant = "landscape"
    @property
    def name(self) -> str: return "MediaPipe Landscape"

class MediapipeSelfieMulticlassWrapper(_BaseMediapipeWrapper):
    _variant = "selfie_multiclass"
    @property
    def name(self) -> str: return "MediaPipe Selfie Multiclass"


class MediapipePoseWrapper(BaseModelWrapper):
    """Extraction rapide des membres pour le Limb-Lock."""
    def __init__(self):
        self._landmarker = None

    @property
    def name(self) -> str: return "MediaPipe Pose Lite"

    def load(self) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
        
        weights_dir = Path(__file__).parent.parent / "weights"
        local_path = weights_dir / "pose_landmarker_lite.task"
        if not local_path.exists():
            import urllib.request
            urllib.request.urlretrieve(_MODEL_URLS["pose_lite"], str(local_path))

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(local_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE
        )
        self._landmarker = PoseLandmarker.create_from_options(options)
        self._mp = mp

    def get_limb_mask(self, frame_bgr: np.ndarray, thickness=8, frame_rgb: Optional[np.ndarray] = None) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        if frame_rgb is None:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect(mp_image)
        
        mask = np.zeros((h, w), dtype=np.uint8)
        if result.pose_landmarks:
            for landmarker in result.pose_landmarks:
                for connection in _LIMB_CONNECTIONS:
                    start = landmarker[connection[0]]
                    end = landmarker[connection[1]]
                    if start.visibility > 0.5 and end.visibility > 0.5:
                        pt1 = (int(start.x * w), int(start.y * h))
                        pt2 = (int(end.x * w), int(end.y * h))
                        cv2.line(mask, pt1, pt2, 255, thickness)
        return mask

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        # On délègue à get_limb_mask pour la compatibilité
        return self.get_limb_mask(frame_bgr).astype(np.float32) / 255.0

    def get_flops(self, input_shape: Tuple[int, int, int] = (3, 256, 256)) -> float:
        return 5.0e6 # Estimation pose landmarker lite

    def cleanup(self) -> None:
        if self._landmarker: self._landmarker.close()
        self._landmarker = None
