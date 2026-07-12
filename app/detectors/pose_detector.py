import logging

import cv2
import numpy as np
from ultralytics import YOLO

from rasd.app.core.config import settings

logger = logging.getLogger(__name__)

SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
]

SKELETON_COLORS = [
    (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 0, 255), (255, 0, 255),
    (0, 255, 255), (128, 255, 0), (255, 128, 0), (0, 128, 255), (128, 0, 255),
    (255, 0, 128), (0, 255, 128),
]

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

HAND_TO_POCKET_THRESHOLD = 40
BENDING_RATIO_THRESHOLD = 0.60


class PoseDetector:
    def __init__(self):
        self.model = None

    def load_model(self):
        path = settings.pose_model_path
        logger.info("Loading YOLOv8-pose model: %s", path)
        kwargs = {"task": "pose"} if path.endswith(".onnx") else {}
        self.model = YOLO(path, **kwargs)
        logger.info("YOLOv8-pose model loaded successfully")

    def detect(self, frame):
        if self.model is None:
            raise RuntimeError("Pose model not loaded. Call load_model() first.")

        results = self.model(frame, verbose=False, device="cpu")
        persons = []

        for result in results:
            if result.keypoints is None:
                continue
            boxes = result.boxes
            kps = result.keypoints

            for i in range(len(kps)):
                xy = kps[i].xy[0].cpu().numpy() if hasattr(kps[i], 'xy') else kps.xy[0].cpu().numpy()
                conf = kps[i].conf[0].cpu().numpy() if hasattr(kps[i], 'conf') else np.ones(len(xy))

                person_kps = {}
                for j in range(min(len(xy), 17)):
                    x, y = xy[j]
                    c = float(conf[j]) if j < len(conf) else 0.0
                    person_kps[j] = {"x": int(x), "y": int(y), "confidence": c}

                bbox = None
                if boxes is not None and i < len(boxes):
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                    bbox = [int(x1), int(y1), int(x2), int(y2)]

                behaviors = self._classify_behaviors(person_kps)

                persons.append({
                    "keypoints": person_kps,
                    "bbox": bbox,
                    "behaviors": behaviors,
                })

        return persons

    def _classify_behaviors(self, kps):
        behaviors = []

        ls = kps.get(5)
        rs = kps.get(6)
        lh = kps.get(11)
        rh = kps.get(12)
        lw = kps.get(9)
        rw = kps.get(10)

        # Hand-to-pocket: wrist near same-side hip
        if lw and lh and lw["confidence"] > 0.5 and lh["confidence"] > 0.5:
            dist = abs(lw["y"] - lh["y"])
            if dist < HAND_TO_POCKET_THRESHOLD:
                behaviors.append({
                    "type": "hand_to_pocket",
                    "side": "left",
                    "label": "HAND IN POCKET (L)",
                    "confidence": max(0.0, 1.0 - dist / HAND_TO_POCKET_THRESHOLD),
                })

        if rw and rh and rw["confidence"] > 0.5 and rh["confidence"] > 0.5:
            dist = abs(rw["y"] - rh["y"])
            if dist < HAND_TO_POCKET_THRESHOLD:
                behaviors.append({
                    "type": "hand_to_pocket",
                    "side": "right",
                    "label": "HAND IN POCKET (R)",
                    "confidence": max(0.0, 1.0 - dist / HAND_TO_POCKET_THRESHOLD),
                })

        # Bending: shoulder-to-hip distance vs person height
        if ls and rs and lh and rh:
            shoulder_y = (ls["y"] + rs["y"]) / 2
            hip_y = (lh["y"] + rh["y"]) / 2
            sh_dist = abs(hip_y - shoulder_y)

            nose = kps.get(0)
            ankle = kps.get(15)
            if nose and ankle and nose["confidence"] > 0.3 and ankle["confidence"] > 0.3:
                height = abs(ankle["y"] - nose["y"])
                if height > 0 and (sh_dist / height) < BENDING_RATIO_THRESHOLD:
                    behaviors.append({
                        "type": "bending",
                        "label": "BENDING",
                        "confidence": max(0.0, 1.0 - (sh_dist / height) / BENDING_RATIO_THRESHOLD),
                    })

        return behaviors

    def draw_skeleton(self, frame, persons):
        for person in persons:
            kps = person["keypoints"]

            # Draw skeleton lines
            for (i, (j1, j2)) in enumerate(SKELETON):
                p1 = kps.get(j1)
                p2 = kps.get(j2)
                if p1 and p2 and p1["confidence"] > 0.5 and p2["confidence"] > 0.5:
                    cv2.line(frame, (p1["x"], p1["y"]), (p2["x"], p2["y"]),
                             SKELETON_COLORS[i % len(SKELETON_COLORS)], 2)

            # Draw keypoints
            for j, p in kps.items():
                if p["confidence"] > 0.5:
                    cv2.circle(frame, (p["x"], p["y"]), 4, (0, 255, 255), -1)

            # Draw behavior labels
            bbox = person["bbox"]
            if bbox and person["behaviors"]:
                y_offset = bbox[1] - 40
                for bh in person["behaviors"]:
                    color = (0, 0, 255) if bh["type"] == "hand_to_pocket" else (0, 165, 255)
                    label = bh["label"]
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (bbox[0], y_offset - th - 6),
                                  (bbox[0] + tw, y_offset + 4), color, -1)
                    cv2.putText(frame, label, (bbox[0], y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                    y_offset -= th + 10
