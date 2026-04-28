"""
Utility for person detection using EfficientDet-Lite0.
Runs on CPU to avoid GPU resource contention.
"""

import logging
import urllib.request
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_DETECTOR_URL = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/latest/efficientdet_lite0.tflite"

class PersonDetector:
    def __init__(self, score_threshold: float = 0.25):
        self._detector = None
        self._score_threshold = score_threshold

    def load(self):
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import ObjectDetector, ObjectDetectorOptions, RunningMode
        except ImportError as e:
            raise ImportError("mediapipe is required for PersonDetector.") from e

        weights_dir = Path(__file__).parent.parent / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        local_path = weights_dir / "efficientdet_lite0.tflite"

        if not local_path.exists():
            logger.info("Downloading EfficientDet-Lite0 detector...")
            urllib.request.urlretrieve(_DETECTOR_URL, str(local_path))

        options = ObjectDetectorOptions(
            base_options=BaseOptions(
                model_asset_path=str(local_path), 
                delegate=BaseOptions.Delegate.CPU
            ),
            running_mode=RunningMode.IMAGE,
            score_threshold=self._score_threshold,
            max_results=5
        )
        self._detector = ObjectDetector.create_from_options(options)
        self._mp = mp

    def detect(self, frame_rgb: np.ndarray, padding: float = 0.05) -> List[Tuple[int, int, int, int]]:
        """Detect persons and return bboxes (x1, y1, x2, y2)."""
        if self._detector is None:
            self.load()

        h, w = frame_rgb.shape[:2]
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._detector.detect(mp_image)
        
        bboxes = []
        for detection in result.detections:
            if detection.categories[0].category_name != "person":
                continue
            
            bbox = detection.bounding_box
            x, y, bw, bh = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
            
            # Adjustable padding
            pad_w, pad_h = int(bw * padding), int(bh * padding)
            x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
            x2, y2 = min(w, x + bw + pad_w), min(h, y + bh + pad_h)
            
            bboxes.append((int(x1), int(y1), int(x2), int(y2)))
        
        return bboxes
