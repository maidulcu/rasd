"""Tests for the fine-tuned UAE retail YOLO model."""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent.parent))

MODEL_PATH = Path("best.pt")
TRAINING_DATA = Path("training/data")
CLASSES = ["dallah", "perfume_bottle", "gold_box", "electronics", "dates_box", "wallet", "watch"]


@pytest.fixture(scope="module")
def model():
    if not MODEL_PATH.exists():
        pytest.skip("best.pt not found — run training first")
    return YOLO(str(MODEL_PATH))


@pytest.fixture(scope="module")
def val_images():
    img_dir = TRAINING_DATA / "images"
    if not img_dir.exists():
        pytest.skip("Training data not found")
    images = sorted(img_dir.glob("*.jpg"))
    assert len(images) > 0, "No validation images found"
    return images


class TestModelLoading:
    def test_model_file_exists(self):
        assert MODEL_PATH.exists(), "best.pt not found in project root"

    def test_model_loads(self, model):
        assert model is not None

    def test_model_has_correct_classes(self, model):
        assert len(model.names) == len(CLASSES)
        for i, name in enumerate(CLASSES):
            assert model.names[i] == name


class TestInference:
    def test_inference_runs(self, model):
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        results = model(img, verbose=False)
        assert len(results) == 1

    def test_returns_detection_boxes(self, model):
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        results = model(img, verbose=False)[0]
        assert hasattr(results, "boxes")

    def test_confidence_scores_valid(self, model):
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        results = model(img, verbose=False)[0]
        if len(results.boxes) > 0:
            confs = results.boxes.conf.numpy()
            assert all(0 <= c <= 1 for c in confs)


class TestSyntheticValidation:
    def test_detects_objects_in_synthetic_images(self, model, val_images):
        total_detections = 0
        images_with_detections = 0
        sample_size = min(20, len(val_images))

        for img_path in val_images[:sample_size]:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            results = model(img, verbose=False, conf=0.25)[0]
            n_det = len(results.boxes)
            total_detections += n_det
            if n_det > 0:
                images_with_detections += 1

        assert images_with_detections > 0, "Model detected nothing in any test image"
        assert total_detections > 0, "Zero total detections"

    def test_per_class_detection_counts(self, model, val_images):
        class_counts = {c: 0 for c in CLASSES}
        sample_size = min(50, len(val_images))

        for img_path in val_images[:sample_size]:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            results = model(img, verbose=False, conf=0.25)[0]
            for box in results.boxes:
                cls_id = int(box.cls[0])
                if cls_id < len(CLASSES):
                    class_counts[CLASSES[cls_id]] += 1

        detected_classes = [c for c, n in class_counts.items() if n > 0]
        assert len(detected_classes) >= 5, f"Only {len(detected_classes)}/7 classes detected: {class_counts}"

    def test_mAP_threshold(self, model):
        results = model.val(data=str(TRAINING_DATA.parent / "config.yaml"), verbose=False)
        assert results.box.map50 > 0.5, f"mAP@0.5 too low: {results.box.map50:.3f}"
        assert results.box.map > 0.3, f"mAP@0.5:0.95 too low: {results.box.map:.3f}"


class TestDetectionQuality:
    def test_high_confidence_detections(self, model, val_images):
        high_conf = 0
        total = 0
        sample_size = min(30, len(val_images))

        for img_path in val_images[:sample_size]:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            results = model(img, verbose=False, conf=0.25)[0]
            for box in results.boxes:
                total += 1
                if float(box.conf[0]) > 0.7:
                    high_conf += 1

        if total == 0:
            pytest.skip("No detections in sample")

        ratio = high_conf / total
        assert ratio > 0.5, f"Only {ratio:.1%} of detections have conf > 0.7 ({high_conf}/{total})"

    def test_bounding_box_dimensions(self, model, val_images):
        sample_size = min(20, len(val_images))
        valid_boxes = 0
        total_boxes = 0

        for img_path in val_images[:sample_size]:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            results = model(img, verbose=False, conf=0.25)[0]
            for box in results.boxes:
                total_boxes += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w = x2 - x1
                h = y2 - y1
                if 10 < w < 500 and 10 < h < 500:
                    valid_boxes += 1

        if total_boxes == 0:
            pytest.skip("No boxes to validate")

        ratio = valid_boxes / total_boxes
        assert ratio > 0.8, f"Only {ratio:.1%} of boxes have valid dimensions"


class TestSpeed:
    def test_inference_speed(self, model):
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        import time
        times = []
        for _ in range(5):
            start = time.perf_counter()
            model(img, verbose=False)
            times.append(time.perf_counter() - start)

        avg_ms = np.mean(times) * 1000
        assert avg_ms < 500, f"Inference too slow: {avg_ms:.0f}ms avg (target < 500ms)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
