"""
Utility for person detection using EfficientDet-Lite0.
Runs on CPU to avoid GPU resource contention.
"""

import logging
import urllib.request
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_DETECTOR_URL = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/latest/efficientdet_lite0.tflite"
_POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"


class PersonDetector:
    def __init__(self, score_threshold: float = 0.25):
        self._detector = None
        self._score_threshold = score_threshold

    def load(self):
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import (
                ObjectDetector,
                ObjectDetectorOptions,
                RunningMode,
            )
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
                model_asset_path=str(local_path), delegate=BaseOptions.Delegate.CPU
            ),
            running_mode=RunningMode.IMAGE,
            score_threshold=self._score_threshold,
            max_results=5,
        )
        self._detector = ObjectDetector.create_from_options(options)
        self._mp = mp

    def detect(
        self, frame_rgb: np.ndarray, padding: float = 0.05
    ) -> list[tuple[int, int, int, int]]:
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


class PoseDetector:
    """Uses MediaPipe Pose to find person bounding boxes based on keypoints."""

    def __init__(self, min_pose_presence_confidence: float = 0.5):
        self._landmarker = None
        self._conf = min_pose_presence_confidence

    def load(self):
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import (
                PoseLandmarker,
                PoseLandmarkerOptions,
                RunningMode,
            )
        except ImportError as e:
            raise ImportError("mediapipe is required for PoseDetector.") from e

        weights_dir = Path(__file__).parent.parent / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        local_path = weights_dir / "pose_landmarker_lite.task"

        if not local_path.exists():
            logger.info("Downloading Pose Landmarker model...")
            urllib.request.urlretrieve(_POSE_MODEL_URL, str(local_path))

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(local_path), delegate=BaseOptions.Delegate.CPU
            ),
            running_mode=RunningMode.IMAGE,
            min_pose_presence_confidence=self._conf,
            num_poses=5,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)
        self._mp = mp

    def detect(
        self, frame_rgb: np.ndarray, padding: float = 0.15
    ) -> list[tuple[int, int, int, int]]:
        """Detect persons using keypoints and return bboxes (x1, y1, x2, y2)."""
        if self._landmarker is None:
            self.load()

        h, w = frame_rgb.shape[:2]
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect(mp_image)

        # Store for debug/visualization
        self.last_result = result

        bboxes = []
        for pose_landmarks in result.pose_landmarks:
            # Extract coordinates
            xs = [lm.x * w for lm in pose_landmarks]
            ys = [lm.y * h for lm in pose_landmarks]

            # Simple bbox from landmarks
            x1, y1 = min(xs), min(ys)
            x2, y2 = max(xs), max(ys)

            bw, bh = x2 - x1, y2 - y1

            # Add padding
            pad_w, pad_h = bw * padding, bh * padding

            # Clamp to frame
            fx1 = max(0, x1 - pad_w)
            fy1 = max(0, y1 - pad_h)
            fx2 = min(w, x2 + pad_w)
            # Always extend to the bottom for visio/webcam scenarios
            fy2 = h

            bboxes.append((int(fx1), int(fy1), int(fx2), int(fy2)))

        return bboxes


class YoloDetector:
    """Uses Ultralytics YOLOv8 for high-performance person detection."""

    def __init__(self, model_size: str = "n", score_threshold: float = 0.25):
        self._model = None
        self._size = model_size
        self._conf = score_threshold

    def load(self):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics is required for YoloDetector. Please run: uv pip install ultralytics"
            ) from e

        weights_dir = Path(__file__).parent.parent / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        # Using nano version (yolov8n.pt) for speed
        model_path = weights_dir / f"yolov8{self._size}.pt"

        # YOLO will handle download automatically if not found
        self._model = YOLO(str(model_path))

    def detect(
        self, frame_rgb: np.ndarray, padding: float = 0.05
    ) -> list[tuple[int, int, int, int]]:
        """Detect persons using YOLO and return bboxes (x1, y1, x2, y2)."""
        if self._model is None:
            self.load()

        h, w = frame_rgb.shape[:2]

        # Inference
        results = self._model.predict(
            source=frame_rgb,
            conf=self._conf,
            classes=[0],  # 0 is 'person' in COCO
            verbose=False,
        )

        bboxes = []
        for result in results:
            for box in result.boxes:
                # Get xyxy tensor and convert to list
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                bw, bh = x2 - x1, y2 - y1

                # Add padding
                pad_w, pad_h = bw * padding, bh * padding

                fx1 = max(0, x1 - pad_w)
                fy1 = max(0, y1 - pad_h)
                fx2 = min(w, x2 + pad_w)
                fy2 = min(h, y2 + pad_h)

                bboxes.append((int(fx1), int(fy1), int(fx2), int(fy2)))

        return bboxes


class FaceDetector:
    """Uses MediaPipe Face Detector (BlazeFace) to find faces."""

    def __init__(self, min_detection_confidence: float = 0.5):
        self._detector = None
        self._conf = min_detection_confidence

    def load(self):
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions, RunningMode
        except ImportError as e:
            raise ImportError("mediapipe is required for FaceDetector.") from e

        weights_dir = Path(__file__).parent.parent / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        local_path = weights_dir / "face_detector.tflite"

        if not local_path.exists():
            logger.info("Downloading Face Detector model...")
            url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
            urllib.request.urlretrieve(url, str(local_path))

        options = FaceDetectorOptions(
            base_options=BaseOptions(
                model_asset_path=str(local_path), delegate=BaseOptions.Delegate.CPU
            ),
            running_mode=RunningMode.IMAGE,
            min_detection_confidence=self._conf,
        )
        self._detector = FaceDetector.create_from_options(options)
        self._mp = mp

    def detect(self, frame_rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Detect faces and return bboxes (x1, y1, x2, y2)."""
        if self._detector is None:
            self.load()

        h, w = frame_rgb.shape[:2]
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._detector.detect(mp_image)

        bboxes = []
        for detection in result.detections:
            bbox = detection.bounding_box
            x1, y1 = bbox.origin_x, bbox.origin_y
            x2, y2 = x1 + bbox.width, y1 + bbox.height
            bboxes.append((int(x1), int(y1), int(x2), int(y2)))

        return bboxes
