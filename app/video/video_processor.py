import logging
import os
import time

import cv2
import supervision as sv
from ultralytics import YOLO

from app.core.config import settings
from app.detectors.face_detector import FaceDetector
from app.detectors.pose_detector import PoseDetector
from app.detectors.theft_detector import TheftDetector
from app.zones.zone_counter import ZoneCounter

logger = logging.getLogger(__name__)


class VideoProcessor:
    def __init__(self):
        os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
        self.model = YOLO(settings.detection_model_path)
        self.face_detector = FaceDetector()
        self.face_detector.load_model()
        self.pose_detector = PoseDetector()
        self.pose_detector.load_model()

        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
        self.tracker = sv.ByteTrack()
        self.zone_counter = None

    def process(self, video_path: str) -> dict:
        logger.info("Analysis started for: %s", video_path)
        start_time = time.time()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        scale = min(settings.MAX_WIDTH / orig_width, 1.0)
        out_width = int(orig_width * scale)
        out_height = int(orig_height * scale)

        logger.info(
            "Video info: %dx%d @ %.1f fps (output: %dx%d, skip: %d)",
            orig_width, orig_height, fps, out_width, out_height, settings.FRAME_SKIP,
        )

        output_filename = f"annotated_{os.path.basename(video_path)}"
        output_path = os.path.join(settings.OUTPUT_DIR, output_filename)

        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

        self.zone_counter = ZoneCounter(out_width, out_height)
        theft = TheftDetector()
        frame_idx = 0
        seen_ids = set()
        max_concurrent = 0
        total_faces = 0
        hand_to_pocket_count = 0
        bending_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if scale < 1.0:
                frame = cv2.resize(frame, (out_width, out_height), interpolation=cv2.INTER_AREA)

            if frame_idx % (settings.FRAME_SKIP + 1) == 0:
                results = self.model.track(frame, verbose=False, device="cpu", persist=True, conf=0.3)[0]

                detections = sv.Detections.from_ultralytics(results)

                detections = self.tracker.update_with_detections(detections)

                faces = self.face_detector.detect(frame)
                pose_persons = self.pose_detector.detect(frame)

                person_mask = detections.class_id == 0
                person_detections = detections[person_mask]

                for tid in person_detections.tracker_id:
                    if tid is not None:
                        seen_ids.add(tid)

                active_ids = set(person_detections.tracker_id) if person_detections.tracker_id is not None else set()

                theft.update(
                    detections.xyxy.tolist(),
                    person_detections.xyxy.tolist() if len(person_detections) > 0 else [],
                    person_detections.tracker_id.tolist() if person_detections.tracker_id is not None else [],
                )

                labels = []
                for i in range(len(detections)):
                    class_name = detections.data["class_name"][i] if "class_name" in detections.data else str(detections.class_id[i])
                    tid = detections.tracker_id[i] if detections.tracker_id is not None else None
                    conf = detections.confidence[i] if detections.confidence is not None else 0
                    if tid is not None:
                        labels.append(f"{class_name} #{tid} {conf:.2f}")
                    else:
                        labels.append(f"{class_name} {conf:.2f}")

                frame = self.box_annotator.annotate(scene=frame, detections=detections)
                frame = self.label_annotator.annotate(scene=frame, detections=detections, labels=labels)

                for p in pose_persons:
                    for bh in p.get("behaviors", []):
                        if bh["type"] == "hand_to_pocket":
                            hand_to_pocket_count += 1
                        elif bh["type"] == "bending":
                            bending_count += 1

                self.pose_detector.draw_skeleton(frame, pose_persons)

                for face in faces:
                    x1, y1, x2, y2 = face["bbox"]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    cv2.putText(
                        frame, "Face", (x1, y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA,
                    )
                    total_faces += 1

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

                zone_stats = self.zone_counter.update(detections)
                frame = self.zone_counter.annotate(frame, detections)

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

        zone_stats = self.zone_counter.get_stats() if self.zone_counter else {}

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
            "zone_stats": zone_stats,
        }
