from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from core.base import MattingModel
from core.parameters import ParameterSpec
from core.registry import models

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


@models.register
class MediapipePose(MattingModel):
    """Extraction rapide des membres pour le Limb-Lock."""
    name = "mediapipe_pose"
    description = "MediaPipe Pose Lite - Détecte les membres (utile pour le post-traitement)."

    def __init__(self, **params):
        super().__init__(**params)
        self._landmarker = None

    @classmethod
    def parameter_specs(cls):
        return [
            ParameterSpec(
                name="thickness",
                type="int",
                default=8,
                label="Limb thickness",
                min_value=1,
                max_value=50,
                help="Thickness of the limb lines in the output mask.",
            ),
            ParameterSpec(
                name="min_visibility",
                type="float",
                default=0.5,
                label="Min visibility",
                min_value=0.0,
                max_value=1.0,
                help="Minimum visibility confidence to draw a limb.",
            ),
        ]

    def load(self, weights_path: str | None = None):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions

        weights_dir = Path(__file__).parent.parent / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        local_path = weights_dir / "pose_landmarker_lite.task"

        if not local_path.exists():
            logger.info("Downloading MediaPipe Pose model...")
            urllib.request.urlretrieve(_MODEL_URL, str(local_path))

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(local_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)
        self._mp = mp

    def infer(self, frame: np.ndarray) -> np.ndarray:
        if self._landmarker is None:
            self.load()

        h, w = frame.shape[:2]
        
        # Metal delegate on macOS prefers SRGBA
        frame_rgba = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGBA, data=frame_rgba)
        result = self._landmarker.detect(mp_image)

        mask = np.zeros((h, w), dtype=np.uint8)
        thickness = self.params.get("thickness", 8)
        min_vis = self.params.get("min_visibility", 0.5)

        if result.pose_landmarks:
            for landmarker in result.pose_landmarks:
                for connection in _LIMB_CONNECTIONS:
                    start = landmarker[connection[0]]
                    end = landmarker[connection[1]]
                    if start.visibility > min_vis and end.visibility > min_vis:
                        pt1 = (int(start.x * w), int(start.y * h))
                        pt2 = (int(end.x * w), int(end.y * h))
                        cv2.line(mask, pt1, pt2, 255, thickness)
        
        return mask.astype(np.float32) / 255.0

    def cleanup(self):
        if self._landmarker:
            self._landmarker.close()
            self._landmarker = None
