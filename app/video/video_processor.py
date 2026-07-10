import logging
import os
import time

import cv2

from app.core.config import settings
from app.detectors.yolo_detector import YOLODetector

logger = logging.getLogger(__name__)


class VideoProcessor:
    def __init__(self):
        os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
        self.detector = YOLODetector()
        self.detector.load_model()

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

        frame_idx = 0
        seen_ids = set()
        max_concurrent = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if scale < 1.0:
                frame = cv2.resize(frame, (out_width, out_height), interpolation=cv2.INTER_AREA)

            if frame_idx % (settings.FRAME_SKIP + 1) == 0:
                tracked = self.detector.track_frame(frame)
            else:
                tracked = self.detector.detect_frame(frame)
                for t in tracked:
                    t["track_id"] = None

            active_ids = set()
            colors = [
                (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
                (255, 0, 255), (0, 255, 255), (128, 0, 128), (0, 128, 128),
            ]

            for det in tracked:
                x1, y1, x2, y2 = det["bbox"]
                tid = det.get("track_id")
                if tid is not None:
                    active_ids.add(tid)
                    seen_ids.add(tid)

                color = colors[(tid or 0) % len(colors)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"Person #{tid}" if tid is not None else f"Person {det['confidence']:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
                cv2.putText(
                    frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
                )

            max_concurrent = max(max_concurrent, len(active_ids))
            writer.write(frame)
            frame_idx += 1

            if frame_idx % 100 == 0:
                logger.info("Processed %d frames...", frame_idx)

        cap.release()
        writer.release()

        unique_people = len(seen_ids)
        processing_time = round(time.time() - start_time, 2)
        logger.info(
            "Analysis completed: %d unique people (max %d concurrent) in %.2f seconds",
            unique_people, max_concurrent, processing_time,
        )

        return {
            "output_file": output_path,
            "processing_time": processing_time,
            "human_count": unique_people,
            "max_concurrent": max_concurrent,
        }
