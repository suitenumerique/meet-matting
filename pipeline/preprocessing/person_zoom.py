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
from core.detector import PersonDetector
from core import context

logger = logging.getLogger(__name__)

@preprocessors.register
class PersonZoom(Preprocessor):
    name = "person_zoom"
    description = "Zoom sur les personnes détectées pour une segmentation haute résolution."

    def __init__(self, **params):
        super().__init__(**params)
        self.detector = PersonDetector()
        self.last_bboxes = []
        self.frame_count = 0

    @classmethod
    def parameter_specs(cls):
        return [
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
        ]

    def reset(self):
        self.last_bboxes = []
        self.frame_count = 0

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        # Signal to the pipeline that zoom is active
        context.set_val("person_zoom_active", True)
        
        interval = self.params.get("update_interval", 1)
        
        # Only run detection every N frames
        if self.frame_count % interval == 0:
            padding = self.params.get("padding", 0.2)
            # PersonDetector returns [x1, y1, x2, y2]
            self.last_bboxes = self.detector.detect(frame, padding=padding)
        
        self.frame_count += 1
        
        # Store in shared context for the pipeline/model to use
        context.set_val("person_bboxes", self.last_bboxes)

        # Draw debug boxes on the frame returned (debug_frame)
        debug_frame = frame.copy()
        for (x1, y1, x2, y2) in self.last_bboxes:
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(debug_frame, "ZOOM ZONE", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return debug_frame
