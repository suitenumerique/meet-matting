"""
Preprocessor for Person Zoom.
Detects people and populates the shared context with bounding boxes.
Includes an update interval to save CPU/GPU cycles.
"""
import logging
import cv2
import numpy as np
from core.base import Preprocessor
from core.parameters import ParameterSpec
from core.registry import preprocessors
from core.detector import PersonDetector, PoseDetector, YoloDetector, FaceDetector
from core import context

logger = logging.getLogger(__name__)

@preprocessors.register
class PersonZoom(Preprocessor):
    name = "person_zoom"
    description = "Zoom sur les personnes détectées pour une segmentation haute résolution."

    def __init__(self, **params):
        super().__init__(**params)
        self.detector = PersonDetector()
        self.pose_detector = PoseDetector()
        self.yolo_detector = YoloDetector()
        self.face_detector = FaceDetector()
        self.last_bboxes = []
        self._smoothed_state = []  # List of [x1, y1, x2, y2] as floats
        self.frame_count = 0

    @classmethod
    def parameter_specs(cls):
        return [
            ParameterSpec(
                name="detection_mode",
                type="choice",
                default="object",
                choices=["object", "pose", "yolo", "face"],
                label="Mode de détection",
                help="Object: MediaPipe. Pose: Points clés. Yolo: YOLOv8. Face: Ancrage visage (Visio).",
            ),
            ParameterSpec(
                name="padding",
                type="float",
                default=0.2,
                min_value=0.0,
                max_value=1.0,
                label="Rembourrage (Padding)",
                help="Espace supplémentaire autour de la personne (0.0 à 1.0).",
            ),
            ParameterSpec(
                name="update_interval",
                type="int",
                default=1,
                min_value=1,
                max_value=60,
                label="Intervalle de rafraîchissement",
                help="Nombre de frames entre chaque détection (1 = à chaque frame).",
            ),
            ParameterSpec(
                name="smoothing",
                type="float",
                default=0.3,
                min_value=0.05,
                max_value=1.0,
                label="Lissage Temporel (EMA)",
                help="Facteur de lissage (Alpha). Plus bas = plus stable mais plus lent. 1.0 = désactivé.",
            ),
            ParameterSpec(
                name="hysteresis",
                type="float",
                default=0.1,
                min_value=0.0,
                max_value=0.5,
                label="Hystérésis de Taille",
                help="Seuil de changement de taille (0.1 = 10%). Empêche les micro-changements de taille.",
            ),
            ParameterSpec(
                name="show_landmarks",
                type="bool",
                default=False,
                label="Afficher les points clés (33)",
                help="Affiche le squelette et les points clés MediaPipe sur la vue debug.",
            ),
        ]

    def reset(self):
        self.last_bboxes = []
        self._smoothed_state = []
        self.frame_count = 0

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        # Signal to the pipeline that zoom is active
        context.set_val("person_zoom_active", True)
        
        interval = self.params.get("update_interval", 1)
        alpha = self.params.get("smoothing", 0.3)
        hysteresis = self.params.get("hysteresis", 0.1)
        
        # Only run detection every N frames
        if self.frame_count % interval == 0:
            padding = self.params.get("padding", 0.2)
            mode = self.params.get("detection_mode", "object")
            
            if mode == "pose":
                raw_bboxes = self.pose_detector.detect(frame, padding=padding)
            elif mode == "yolo":
                h_img = frame.shape[0]
                raw_bboxes = self.yolo_detector.detect(frame, padding=padding)
                # Extend box to bottom if person is significant (> 5% height)
                extended = []
                for (x1, y1, x2, y2) in raw_bboxes:
                    box_h = y2 - y1
                    # FALLBACK: If person covers > 70% of frame, don't crop (use full width)
                    if box_h / h_img > 0.7:
                        extended.append((0, 0, frame.shape[1], h_img))
                        continue
                    if box_h / h_img > 0.05:
                        y2 = h_img
                    extended.append((x1, y1, x2, y2))
                raw_bboxes = extended
            elif mode == "face":
                face_bboxes = self.face_detector.detect(frame)
                raw_bboxes = []
                h_img, w_img = frame.shape[:2]
                for (fx1, fy1, fx2, fy2) in face_bboxes:
                    fw, fh = fx2 - fx1, fy2 - fy1
                    fcx = (fx1 + fx2) / 2
                    
                    # More generous expansion for close-up
                    rel_h = fh / h_img
                    pad_top = 1.5 if rel_h < 0.3 else 2.0 # More air when close
                    
                    bx1 = max(0, fcx - fw * 3.0)
                    bx2 = min(w_img, fcx + fw * 3.0)
                    by1 = max(0, fy1 - fh * pad_top)
                    by2 = h_img
                    
                    # Fallback if box is too big
                    if (by2 - by1) / h_img > 0.8:
                        raw_bboxes.append((0, 0, w_img, h_img))
                    else:
                        raw_bboxes.append((int(bx1), int(by1), int(bx2), int(by2)))
            else:
                raw_bboxes = self.detector.detect(frame, padding=padding)
                
            self.last_bboxes = self._update_smoothed_boxes(raw_bboxes, alpha, hysteresis)
        
        self.frame_count += 1
        
        # Store in shared context for the pipeline/model to use
        context.set_val("person_bboxes", self.last_bboxes)
        
        # Add landmarks if needed and available
        if self.params.get("show_landmarks") and self.params.get("detection_mode") == "pose":
            context.set_val("show_landmarks", True)
            if hasattr(self.pose_detector, "last_result"):
                context.set_val("pose_landmarks", self.pose_detector.last_result.pose_landmarks)

        return frame

    def _update_smoothed_boxes(self, raw_bboxes, alpha, hysteresis):
        """Apply EMA and Hysteresis to raw detections."""
        if not self._smoothed_state or not raw_bboxes:
            self._smoothed_state = [[float(c) for c in b] for b in raw_bboxes]
            return [list(b) for b in raw_bboxes]

        new_state = []
        used_raw = set()

        # Match existing smoothed boxes to new detections
        for old_box in self._smoothed_state:
            old_cx = (old_box[0] + old_box[2]) / 2
            old_cy = (old_box[1] + old_box[3]) / 2
            
            best_idx = -1
            min_dist = float("inf")
            
            for i, raw_box in enumerate(raw_bboxes):
                if i in used_raw:
                    continue
                raw_cx = (raw_box[0] + raw_box[2]) / 2
                raw_cy = (raw_box[1] + raw_box[3]) / 2
                dist = ((old_cx - raw_cx)**2 + (old_cy - raw_cy)**2)**0.5
                
                # Threshold for matching: 30% of the diagonal of the box
                diag = ((raw_box[2]-raw_box[0])**2 + (raw_box[3]-raw_box[1])**2)**0.5
                if dist < min_dist and dist < diag * 0.5:
                    min_dist = dist
                    best_idx = i
            
            if best_idx != -1:
                raw_box = raw_bboxes[best_idx]
                used_raw.add(best_idx)
                
                # 1. Asymmetric EMA Smoothing
                # Growth is faster than shrinkage to avoid clipping
                for i in range(4):
                    target = raw_box[i]
                    current = old_box[i]
                    
                    # If expanding (x1 decreasing, x2 increasing, etc.)
                    is_expanding = False
                    if i == 0 and target < current: is_expanding = True # x1
                    if i == 1 and target < current: is_expanding = True # y1
                    if i == 2 and target > current: is_expanding = True # x2
                    if i == 3 and target > current: is_expanding = True # y2
                    
                    # Use higher alpha for expansion
                    effective_alpha = min(1.0, alpha * 2.5) if is_expanding else alpha
                    old_box[i] = current * (1 - effective_alpha) + target * effective_alpha
                
                smoothed = old_box
                
                # 2. Hysteresis on size to prevent "pumping"
                old_w = old_box[2] - old_box[0]
                old_h = old_box[3] - old_box[1]
                new_w = smoothed[2] - smoothed[0]
                new_h = smoothed[3] - smoothed[1]
                
                # Only apply hysteresis if shrinking
                if new_w < old_w and (old_w - new_w) / max(old_w, 1) < hysteresis:
                    cx = (smoothed[0] + smoothed[2]) / 2
                    smoothed[0] = cx - old_w / 2
                    smoothed[2] = cx + old_w / 2
                
                if new_h < old_h and (old_h - new_h) / max(old_h, 1) < hysteresis:
                    cy = (smoothed[1] + smoothed[3]) / 2
                    smoothed[1] = cy - old_h / 2
                    smoothed[3] = cy + old_h / 2
                    
                new_state.append(smoothed)
            # If not matched, the person probably left the frame (we drop the box)

        # New detections that weren't matched
        for i, raw_box in enumerate(raw_bboxes):
            if i not in used_raw:
                new_state.append([float(c) for c in raw_box])

        self._smoothed_state = new_state
        return [[int(c) for c in b] for b in new_state]
