"""
Wrapper pour le modèle MediaPipe Pose (Landmarks).
"""

import logging
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from .base import BaseModelWrapper

logger = logging.getLogger(__name__)

_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"

# Connexions anatomiques bras + corps (indices MediaPipe Pose Landmarker)
_LIMB_CONNECTIONS = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
]


class MediapipePoseWrapper(BaseModelWrapper):
    """Extraction rapide des membres pour le Limb-Lock."""

    def __init__(self):
        self._landmarker = None

    @property
    def name(self) -> str:
        return "MediaPipe Pose Lite"

    def load(self) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions

        weights_dir = Path(__file__).parent.parent / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        local_path = weights_dir / "pose_landmarker_lite.task"

        if not local_path.exists():
            urllib.request.urlretrieve(_MODEL_URL, str(local_path))

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(local_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)
        self._mp = mp

    def get_limb_mask(
        self, frame_bgr: np.ndarray, thickness=8, frame_rgb: np.ndarray | None = None
    ) -> np.ndarray:
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
        return self.get_limb_mask(frame_bgr).astype(np.float32) / 255.0

    def get_flops(self, input_shape: tuple[int, int, int] = (3, 256, 256)) -> float:
        return 5.0e6  # Estimation pose landmarker lite

    def cleanup(self) -> None:
        if self._landmarker:
            self._landmarker.close()
        self._landmarker = None
