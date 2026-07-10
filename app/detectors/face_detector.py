import logging
import os

import cv2

logger = logging.getLogger(__name__)

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


class FaceDetector:
    def __init__(self):
        self.cascade = None

    def load_model(self):
        if not os.path.exists(CASCADE_PATH):
            raise RuntimeError(f"Haar cascade not found: {CASCADE_PATH}")
        self.cascade = cv2.CascadeClassifier(CASCADE_PATH)
        logger.info("Face detector loaded")

    def detect(self, frame):
        if self.cascade is None:
            raise RuntimeError("Face detector not loaded")
        h, w = frame.shape[:2]
        # Downsample 2x for ~4x faster detection; minimum 320px wide
        scale = min(1.0, 640.0 / w)
        if scale < 1.0:
            small = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        else:
            small = frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        inv = 1.0 / scale
        return [
            {"bbox": [int(x * inv), int(y * inv), int((x + w) * inv), int((y + h) * inv)],
             "confidence": 1.0, "label": "Face"}
            for (x, y, w, h) in faces
        ]
