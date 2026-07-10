import logging

import cv2
from ultralytics import YOLO

from app.core.config import settings

logger = logging.getLogger(__name__)

PERSON_CLASS_ID = 0


class YOLODetector:
    def __init__(self):
        self.model = None

    def load_model(self):
        logger.info("Loading YOLO model: %s", settings.YOLO_MODEL)
        self.model = YOLO(settings.YOLO_MODEL)
        logger.info("YOLO model loaded successfully")

    def detect_frame(self, frame):
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        results = self.model(frame, verbose=False, device="cpu")

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                detections.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": conf,
                    "class_id": cls,
                    "label": result.names[cls] if result.names else str(cls),
                })

        return detections

    def track_frame(self, frame):
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        results = self.model.track(frame, verbose=False, device="cpu", persist=True, conf=0.3)

        tracked = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                tid = int(box.id[0]) if box.id is not None else None
                tracked.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": conf,
                    "class_id": cls,
                    "track_id": tid,
                    "label": result.names[cls] if result.names else str(cls),
                })

        return tracked
