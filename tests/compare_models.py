"""Compare default YOLO vs fine-tuned UAE retail model on synthetic images."""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

CLASSES = ["dallah", "perfume_bottle", "gold_box", "electronics", "dates_box", "wallet", "watch"]

default_model = YOLO("yolov8n.pt")
custom_model = YOLO("best.pt")

images = sorted(Path("training/data/images").glob("*.jpg"))[:30]

print(f"Testing on {len(images)} images\n")
print(f"{'Image':<20} {'Default':<15} {'Fine-tuned':<15} {'Difference'}")
print("-" * 60)

total_default = 0
total_custom = 0
correct_default = 0
correct_custom = 0

for img_path in images:
    img = cv2.imread(str(img_path))
    if img is None:
        continue

    r_default = default_model(img, verbose=False, conf=0.25)[0]
    r_custom = custom_model(img, verbose=False, conf=0.25)[0]

    n_def = len(r_default.boxes)
    n_cust = len(r_custom.boxes)

    total_default += n_def
    total_custom += n_cust

    # Check if detections match expected classes
    for box in r_default.boxes:
        if int(box.cls[0]) < len(CLASSES):
            correct_default += 1
    for box in r_custom.boxes:
        if int(box.cls[0]) < len(CLASSES):
            correct_custom += 1

    diff = n_cust - n_def
    sign = "+" if diff > 0 else ""
    print(f"{img_path.name:<20} {n_def:<15} {n_cust:<15} {sign}{diff}")

print("-" * 60)
print(f"{'TOTAL':<20} {total_default:<15} {total_custom:<15} +{total_custom - total_default}")
print()

print("=== Per-Class Detection ===")
print(f"{'Class':<20} {'Default':<15} {'Fine-tuned':<15}")
print("-" * 50)

for cls_idx, cls_name in enumerate(CLASSES):
    def_count = 0
    cust_count = 0
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        r_def = default_model(img, verbose=False, conf=0.25)[0]
        r_cust = custom_model(img, verbose=False, conf=0.25)[0]
        for box in r_def.boxes:
            if int(box.cls[0]) == cls_idx:
                def_count += 1
        for box in r_cust.boxes:
            if int(box.cls[0]) == cls_idx:
                cust_count += 1
    print(f"{cls_name:<20} {def_count:<15} {cust_count:<15}")

print()
print("=== Speed Comparison ===")
import time
dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

for name, m in [("Default YOLO", default_model), ("Fine-tuned", custom_model)]:
    times = []
    for _ in range(10):
        start = time.perf_counter()
        m(dummy, verbose=False)
        times.append(time.perf_counter() - start)
    avg = np.mean(times) * 1000
    print(f"  {name:<15} {avg:.0f}ms avg")
