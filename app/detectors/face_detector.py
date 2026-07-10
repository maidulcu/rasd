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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return [
            {"bbox": [int(x), int(y), int(x + w), int(y + h)], "confidence": 1.0, "label": "Face"}
            for (x, y, w, h) in faces
        ]
