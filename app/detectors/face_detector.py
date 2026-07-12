import logging
import os

import cv2

logger = logging.getLogger(__name__)

CASCADE_PATH = os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml")
if not os.path.exists(CASCADE_PATH):
    CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


class FaceDetector:
    def __init__(self):
        self.cascade = None

    def load_model(self):
        if not os.path.exists(CASCADE_PATH):
            logger.warning("Haar cascade not found: %s — face detection disabled", CASCADE_PATH)
            return
        try:
            self.cascade = cv2.CascadeClassifier(CASCADE_PATH)
            logger.info("Face detector loaded")
        except AttributeError:
            logger.warning("OpenCV %s missing CascadeClassifier — face detection disabled", cv2.__version__)

    def detect(self, frame):
        if self.cascade is None:
            return []
        h, w = frame.shape[:2]
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
