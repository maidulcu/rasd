import logging
import os
import time

import cv2

from app.core.config import settings
from app.detectors.face_detector import FaceDetector
from app.detectors.pose_detector import PoseDetector
from app.detectors.theft_detector import TheftDetector
from app.detectors.yolo_detector import YOLODetector

logger = logging.getLogger(__name__)


class VideoProcessor:
    def __init__(self):
        os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
        self.detector = YOLODetector()
        self.detector.load_model()
        self.face_detector = FaceDetector()
        self.face_detector.load_model()
        self.pose_detector = PoseDetector()
        self.pose_detector.load_model()

    def process(self, video_path: str) -> dict:
        logger.info("Analysis started for: %s", video_path)
        start_time = time.time()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        scale = min(settings.MAX_WIDTH / orig_width, 1.0)
        out_width = int(orig_width * scale)
        out_height = int(orig_height * scale)

        logger.info(
            "Video info: %dx%d @ %.1f fps, %d frames (output: %dx%d, skip: %d)",
            orig_width, orig_height, fps, total_frames, out_width, out_height, settings.FRAME_SKIP,
        )

        output_filename = f"annotated_{os.path.basename(video_path)}"
        output_path = os.path.join(settings.OUTPUT_DIR, output_filename)

        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

        theft = TheftDetector()
        frame_idx = 0
        seen_ids = set()
        max_concurrent = 0
        total_faces = 0
        pose_alerts = []
        hand_to_pocket_count = 0
        bending_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if scale < 1.0:
                frame = cv2.resize(frame, (out_width, out_height), interpolation=cv2.INTER_AREA)

            if frame_idx % (settings.FRAME_SKIP + 1) == 0:
                tracked = self.detector.track_frame(frame)
                all_detections = self.detector.detect_frame(frame)
                faces = self.face_detector.detect(frame)
                pose_persons = self.pose_detector.detect(frame)

                person_boxes = []
                person_ids = []
                for d in tracked:
                    if d.get("class_id") == 0:
                        person_boxes.append(d["bbox"])
                        person_ids.append(d.get("track_id"))
                theft.update(all_detections, person_boxes, person_ids)

                for p in pose_persons:
                    for bh in p.get("behaviors", []):
                        if bh["type"] == "hand_to_pocket":
                            hand_to_pocket_count += 1
                            pose_alerts.append({
                                "type": "hand_to_pocket",
                                "frame": frame_idx,
                                "label": bh["label"],
                            })
                        elif bh["type"] == "bending":
                            bending_count += 1
                            pose_alerts.append({
                                "type": "bending",
                                "frame": frame_idx,
                                "label": bh["label"],
                            })
            else:
                tracked = self.detector.detect_frame(frame)
                for t in tracked:
                    t["track_id"] = None
                faces = []
                all_detections = []
                pose_persons = []

            active_ids = set()
            colors = [
                (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
                (255, 0, 255), (0, 255, 255), (128, 0, 128), (0, 128, 128),
            ]

            for det in tracked:
                cid = det.get("class_id", -1)
                x1, y1, x2, y2 = det["bbox"]
                tid = det.get("track_id")
                if tid is not None and cid == 0:
                    active_ids.add(tid)
                    seen_ids.add(tid)

                color = colors[(tid or 0) % len(colors)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"Person #{tid}" if tid is not None else det.get("label", "?")
                if tid is None:
                    label = f"{det.get('label', '?')} {det['confidence']:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
                cv2.putText(
                    frame, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
                )

            for face in faces:
                x1, y1, x2, y2 = face["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(
                    frame, "Face", (x1, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA,
                )
                total_faces += 1

            self.pose_detector.draw_skeleton(frame, pose_persons)

            for interaction in theft.current_interactions:
                x1, y1, x2, y2 = interaction["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 3)
                cv2.putText(
                    frame, f"PICKED UP: {interaction['label']}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA,
                )

            for alert in theft.unattended_alerts:
                x1, y1, x2, y2 = alert["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(
                    frame, f"UNATTENDED: {alert['label']}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA,
                )

            for alert in theft.theft_alerts:
                x1, y1, x2, y2 = alert.get("person_location", alert["bbox"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
                cv2.putText(
                    frame, f"THEFT: {alert['label']} concealed!", (x1, y2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2, cv2.LINE_AA,
                )

            max_concurrent = max(max_concurrent, len(active_ids))
            writer.write(frame)
            frame_idx += 1

            if frame_idx % 100 == 0:
                logger.info("Processed %d frames...", frame_idx)

        cap.release()
        writer.release()

        theft_results = theft.get_results()
        unique_people = len(seen_ids)
        processing_time = round(time.time() - start_time, 2)
        logger.info(
            "Analysis completed: %d people, %d faces, %d unattended, %d theft alerts in %.2fs",
            unique_people, total_faces, len(theft_results["unattended_objects"]),
            len(theft_results["theft_alerts"]), processing_time,
        )

        return {
            "output_file": output_path,
            "processing_time": processing_time,
            "human_count": unique_people,
            "max_concurrent": max_concurrent,
            "face_count": total_faces,
            "unattended_count": len(theft_results["unattended_objects"]),
            "theft_alerts": len(theft_results["theft_alerts"]),
            "hand_to_pocket_count": hand_to_pocket_count,
            "bending_count": bending_count,
        }
