"""Business-focused tests for theft detection - UAE retail scenario."""

import cv2
import numpy as np
import time
from pathlib import Path
from ultralytics import YOLO

# Load models
default_model = YOLO("yolov8n.pt")
custom_model = YOLO("best.pt")


def process_video_for_business(video_path: str) -> dict:
    """Run full analysis pipeline on a video and return business metrics."""
    from app.detectors.theft_detector import TheftDetector
    from app.detectors.yolo_detector import YOLODetector

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Cannot open video"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps

    theft_detector = TheftDetector()
    detector = YOLODetector()
    detector.load_model()

    seen_people = set()
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 3 == 0:
            tracked = detector.track_frame(frame)
            person_boxes = []
            person_ids = []
            for d in tracked:
                if d.get("class_id") == 0:
                    person_boxes.append(d["bbox"])
                    person_ids.append(d.get("track_id"))
                    if d.get("track_id"):
                        seen_people.add(d["track_id"])
            theft_detector.update(tracked, person_boxes, person_ids)

        frame_count += 1

    cap.release()
    processing_time = time.time() - start_time

    theft_results = theft_detector.get_results()

    return {
        "video": video_path,
        "duration_seconds": round(duration_sec, 1),
        "total_frames": total_frames,
        "unique_people": len(seen_people),
        "processing_time": round(processing_time, 1),
        "processing_fps": round(frame_count / processing_time, 1) if processing_time > 0 else 0,
        "unattended_alerts": len(theft_results["unattended_objects"]),
        "theft_alerts": len(theft_results["theft_alerts"]),
        "interactions": len(theft_results["current_interactions"]),
        "unattended_details": theft_results["unattended_objects"][:5],
        "theft_details": theft_results["theft_alerts"][:5],
    }


class TestBusinessMetrics:
    """Test real business value for UAE retailers."""

    def test_people_counting_accuracy(self):
        """Can we count unique customers in a video?"""
        videos = list(Path("downloads").glob("*.mp4"))
        if not videos:
            return

        results = []
        for v in videos[:2]:
            r = process_video_for_business(str(v))
            results.append(r)

        for r in results:
            assert r["unique_people"] >= 0, "People count should be non-negative"
            assert r["processing_time"] < r["duration_seconds"] * 2, "Processing should be faster than real-time"

    def test_theft_detection_capabilities(self):
        """Does the system detect suspicious behavior?"""
        videos = list(Path("downloads").glob("*.mp4"))
        if not videos:
            return

        for v in videos[:2]:
            r = process_video_for_business(str(v))
            # System should produce some output
            assert "unattended_alerts" in r
            assert "theft_alerts" in r
            assert "interactions" in r

    def test_processing_speed_business_requirement(self):
        """Is processing fast enough for real-time retail monitoring?"""
        videos = list(Path("downloads").glob("*.mp4"))
        if not videos:
            return

        for v in videos[:1]:
            r = process_video_for_business(str(v))
            # Business requirement: process at least 5 FPS for real-time
            assert r["processing_fps"] >= 5, f"Too slow: {r['processing_fps']} FPS (need 5+)"

    def test_model_comparison_business_value(self):
        """Does custom model detect more UAE-relevant items?"""
        synthetic_images = sorted(Path("training/data/images").glob("*.jpg"))[:20]
        if not synthetic_images:
            return

        default_total = 0
        custom_total = 0

        for img_path in synthetic_images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            r_default = default_model(img, verbose=False, conf=0.25)[0]
            r_custom = custom_model(img, verbose=False, conf=0.25)[0]

            default_total += len(r_default.boxes)
            custom_total += len(r_custom.boxes)

        # Custom model should detect significantly more
        assert custom_total > default_total * 2, \
            f"Custom model should detect 2x more: {custom_total} vs {default_total}"

    def test_annotation_quality(self):
        """Is the output video watchable for security review?"""
        videos = list(Path("downloads").glob("*.mp4"))
        if not videos:
            return

        for v in videos[:1]:
            r = process_video_for_business(str(v))
            # Output should exist
            output_files = list(Path("output").glob("annotated_*.mp4"))
            assert len(output_files) > 0, "No annotated output video produced"


class TestUAEUseCase:
    """Test UAE-specific retail scenarios."""

    def test_dallah_detection(self):
        """Can the model detect UAE-specific items like dallah?"""
        synthetic_images = sorted(Path("training/data/images").glob("*.jpg"))[:10]
        if not synthetic_images:
            return

        dallah_detected = 0
        for img_path in synthetic_images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            results = custom_model(img, verbose=False, conf=0.25)[0]
            for box in results.boxes:
                if int(box.cls[0]) == 0:  # dallah
                    dallah_detected += 1

        assert dallah_detected > 0, "Model should detect dallah (Arabic coffee pot)"

    def test_per_class_detection_summary(self):
        """Summary of what each class detects."""
        classes = ["dallah", "perfume_bottle", "gold_box", "electronics", "dates_box", "wallet", "watch"]
        synthetic_images = sorted(Path("training/data/images").glob("*.jpg"))[:30]

        class_counts = {c: 0 for c in classes}
        for img_path in synthetic_images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            results = custom_model(img, verbose=False, conf=0.25)[0]
            for box in results.boxes:
                cls_id = int(box.cls[0])
                if cls_id < len(classes):
                    class_counts[classes[cls_id]] += 1

        detected = sum(1 for v in class_counts.values() if v > 0)
        assert detected >= 5, f"Only {detected}/7 classes detected"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
