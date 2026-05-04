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
        assert self._detector is not None

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
        assert self._landmarker is not None

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
    """Uses Ultralytics YOLO11 for high-performance person detection."""

    # Class-level cache to avoid reloading the model
    _MODEL_CACHE: dict[tuple[str, float], tuple[object, str]] = {}

    def __init__(self, model_size: str = "n", score_threshold: float = 0.25):
        self._model = None
        self._size = model_size
        self._conf = score_threshold
        self._device = "cpu"

    def load(self):
        cache_key = (self._size, self._conf)
        if cache_key in self._MODEL_CACHE:
            self._model, self._device = self._MODEL_CACHE[cache_key]
            return

        try:
            from ultralytics import YOLO

            # CRITICAL FIX: DO NOT use MPS for YOLO on Mac.
            # PyTorch's NMS (Non-Maximum Suppression) operation on MPS
            # currently causes massive hangs (> 8 seconds) and freezes the app.
            # We strictly force CPU, which is actually very fast for YOLO Nano.
            self._device = "cpu"

            logger.info(f"YOLO11 loading on device: {self._device} (MPS disabled for stability)")
        except ImportError as e:
            raise ImportError(
                "ultralytics is required for YoloDetector. Please run: uv pip install ultralytics"
            ) from e

        weights_dir = Path(__file__).parent.parent / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        # Using YOLO11 prefix
        model_path = weights_dir / f"yolo11{self._size}.pt"

        model = YOLO(str(model_path))
        model.to(self._device)
        # Note: Keeping half=False for stability against 'gray screen' issues

        self._model = model
        self._MODEL_CACHE[cache_key] = (self._model, self._device)

    def detect(
        self, frame_rgb: np.ndarray, padding: float = 0.05
    ) -> list[tuple[int, int, int, int]]:
        """Detect persons using YOLO and return bboxes (x1, y1, x2, y2)."""
        if self._model is None:
            self.load()
        assert self._model is not None

        h, w = frame_rgb.shape[:2]

        # Inference
        # Using imgsz=320 for speed, half=False for stability
        results = self._model.predict(
            source=frame_rgb,
            conf=self._conf,
            classes=[0],  # 0 is 'person' in COCO
            verbose=False,
            device=self._device,
            imgsz=320,
            half=False,
        )

        bboxes = []
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                # Get boxes in xyxy format
                boxes = result.boxes.xyxy.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = box
                    bw, bh = x2 - x1, y2 - y1

                    # Add padding
                    pad_w, pad_h = bw * padding, bh * padding

                    fx1 = int(max(0, x1 - pad_w))
                    fy1 = int(max(0, y1 - pad_h))
                    fx2 = int(min(w, x2 + pad_w))
                    fy2 = int(min(h, y2 + pad_h))

                    if fx2 > fx1 and fy2 > fy1:
                        bboxes.append((fx1, fy1, fx2, fy2))

        if not bboxes:
            logger.debug(f"YOLO: No person detected (device={self._device})")
        else:
            logger.debug(f"YOLO: Detected {len(bboxes)} persons")

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
        assert self._detector is not None

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
