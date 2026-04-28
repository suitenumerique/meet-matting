"""
Preprocessor that detects persons and stores their bboxes in the shared context.
This enables the 'Person Zoom' feature in compatible models.
"""

import logging
from core.base import Preprocessor
from core.parameters import ParameterSpec
from core.registry import preprocessors
from core.detector import PersonDetector
from core import context

logger = logging.getLogger(__name__)

@preprocessors.register
class PersonZoom(Preprocessor):
    name = "person_zoom"
    description = "Detects people to enable high-resolution crops in matting models."

    def __init__(self, **params):
        super().__init__(**params)
        self._detector = PersonDetector(score_threshold=self.params["threshold"])

    @classmethod
    def parameter_specs(cls):
        return [
            ParameterSpec(
                name="threshold",
                type="float",
                default=0.25,
                label="Detection Threshold",
                min_value=0.1,
                max_value=0.9,
                step=0.05,
                help="Confidence threshold for person detection.",
            ),
            ParameterSpec(
                name="padding",
                type="float",
                default=0.10,
                label="Box Padding",
                min_value=0.0,
                max_value=0.5,
                step=0.05,
                help="Extra margin around the detected person.",
            ),
            ParameterSpec(
                name="debug_draw",
                type="bool",
                default=True,
                label="Debug: Draw BBoxes",
                help="Draw bboxes on the frame for visualization.",
            ),
        ]

    def __call__(self, frame):
        """Detect people and store bboxes in context."""
        try:
            bboxes = self._detector.detect(frame, padding=self.params["padding"])
        except Exception as e:
            logger.error(f"PersonZoom detection error: {e}")
            bboxes = []

        # Store bboxes in context
        context.set_val("person_bboxes", bboxes)
        # Indicate that PersonZoom is ACTIVE, so the model should not fallback to full frame
        context.set_val("person_zoom_active", True)
        
        if self.params["debug_draw"] and bboxes:
            import cv2
            debug_frame = frame.copy()
            for (x1, y1, x2, y2) in bboxes:
                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            return debug_frame

        return frame
