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
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(
            "Video info: %dx%d @ %.1f fps, %d frames", width, height, fps, total_frames
        )

        output_filename = f"annotated_{os.path.basename(video_path)}"
        output_path = os.path.join(settings.OUTPUT_DIR, output_filename)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        human_count = 0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = self.detector.detect_frame(frame)

            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                conf = det["confidence"]
                label = det["label"]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                text = f"{label} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), (0, 255, 0), -1)
                cv2.putText(
                    frame,
                    text,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

            human_count += len(detections)
            writer.write(frame)
            frame_idx += 1

            if frame_idx % 100 == 0:
                logger.info("Processed %d frames...", frame_idx)

        cap.release()
        writer.release()

        processing_time = round(time.time() - start_time, 2)
        logger.info(
            "Analysis completed: %d human detections in %.2f seconds",
            human_count,
            processing_time,
        )

        return {
            "output_file": output_path,
            "processing_time": processing_time,
            "human_count": human_count,
        }
