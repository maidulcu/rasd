# Fine-Tuning YOLOv8 for UAE Retail — Complete Guide

## Overview

This document covers the full pipeline for fine-tuning a custom object detection model
for UAE retail surveillance. You'll learn to:

1. Generate a synthetic training dataset (UAE-specific items)
2. Fine-tune YOLOv8n on that dataset (or real data)
3. Export and integrate the custom model into the video analyzer
4. Collect real-world data for production-quality results

---

## Project Structure

```
training/
├── config.yaml               # Dataset config (auto-generated)
├── generate_data.py           # Synthetic data generator
├── train.py                   # Fine-tuning pipeline
└── data/
    ├── images/                # 600 synthetic images (640x640)
    │   ├── img_000000.jpg
    │   └── ...
    └── labels/                # YOLO-format labels
        ├── img_000000.txt
        └── ...
```

### 7 Custom UAE Retail Classes

| Class ID | Name | Description |
|---|---|---|
| 0 | `dallah` | Arabic coffee pot |
| 1 | `perfume_bottle` | Perfume/attar bottle |
| 2 | `gold_box` | Jewelry/gift box |
| 3 | `electronics` | Phone/tablet |
| 4 | `dates_box` | Box of dates |
| 5 | `wallet` | Wallet/purse |
| 6 | `watch` | Wristwatch |

---

## 1. Transfer to GPU PC

### Option A: Copy the whole project via USB/SD card

```bash
# On your Mac (source)
tar czf video-analyzer.tar.gz \
    --exclude=.venv \
    --exclude=__pycache__ \
    --exclude='*.db' \
    --exclude='yolov8n.pt' \
    --exclude='yolov8n-pose.pt' \
    --exclude='yolov8n.onnx' \
    --exclude='yolov8n-pose.onnx' \
    --exclude=downloads \
    --exclude=output \
    /Users/maidul/Downloads/video-analyzer-starter

# Copy to your GPU PC
# (USB drive, SCP, Nextcloud, etc.)
```

### Option B: Git clone on GPU PC

```bash
git clone <your-repo-url> video-analyzer
cd video-analyzer
```

---

## 2. GPU PC Setup

### Recommended Hardware

| Component | Minimum | Recommended |
|---|---|---|
| GPU | NVIDIA GTX 1060 6GB | NVIDIA RTX 3060+ or A4000 |
| RAM | 16GB | 32GB |
| Storage | 20GB free | SSD |
| OS | Ubuntu 22.04 / Windows 10 | Ubuntu 22.04 |
| CUDA | 11.8+ | 12.1+ |

### Install Dependencies

```bash
# Ubuntu / WSL2
sudo apt update && sudo apt install -y python3 python3-pip python3-venv ffmpeg

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install everything else
pip install -r requirements.txt

# Verify GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}') if torch.cuda.is_available() else print('CPU only')"
```

Expected output: `GPU: NVIDIA GeForce RTX 3060`

### Windows Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## 3. Generate Synthetic Dataset

Already done — 600 images are generated and ready in `training/data/`.

To regenerate or create a larger dataset:

```bash
# Regenerate with 2000 images (1000 train + 1000 val)
python -c "
import training.generate_data as gd
import os
os.makedirs('training/data/images', exist_ok=True)
os.makedirs('training/data/labels', exist_ok=True)
gd.N_TRAIN = 1000
gd.N_VAL = 1000
gd.generate_split(1000, 0)
gd.generate_split(1000, 1000)
gd.write_dataset_yaml()
"
```

### To add new classes

Edit `training/generate_data.py`:

1. Add a new `make_<item>()` function that returns an RGBA image (item + alpha channel)
2. Add the function to `ITEM_MAKERS` list
3. Add the class name to `CLASSES`
4. Regenerate dataset

Example — adding "saffron":

```python
def make_saffron():
    """Saffron container."""
    img = np.zeros((60, 40, 4), dtype=np.uint8)
    cv2.rectangle(img, (2, 2), (38, 58), (180, 50, 50, 255), -1)  # Red box
    cv2.rectangle(img, (5, 5), (35, 20), (200, 180, 100, 255), -1)  # Gold label
    cv2.putText(img, "SAFFRON", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0, 200), 1)
    return img

CLASSES = ["dallah", "perfume_bottle", "gold_box", "electronics",
           "dates_box", "wallet", "watch", "saffron"]

ITEM_MAKERS = [make_dallah, make_perfume, make_gold_box, make_electronics,
               make_dates_box, make_wallet, make_watch, make_saffron]
```

---

## 4. Training

### Quick Start (CPU — slow, for testing only)

```bash
python training/train.py --epochs 10
```

### Full Training (GPU — fast)

The `train.py` script auto-detects GPU. On a GPU PC, training is ~50× faster.

```bash
# Train with defaults (100 epochs, batch=8, lr=0.001)
python training/train.py

# Or customize
python training/train.py \
    --epochs 200 \
    --batch 32 \
    --imgsz 640 \
    --lr 0.001
```

### Flags

| Flag | Default | Description |
|---|---|---|
| `--epochs` | 100 | Number of training epochs |
| `--batch` | 8 | Batch size (higher on GPU = faster) |
| `--imgsz` | 640 | Training image size |
| `--lr` | 0.001 | Learning rate |
| `--model` | `yolov8n.pt` | Pretrained model to start from |
| `--resume` | false | Resume from checkpoint |
| `--export-only` | false | Just export to ONNX (skip training) |

### Expected Training Times

| Hardware | Epochs | Batch | Time |
|---|---|---|---|
| MacBook CPU (M3) | 100 | 8 | ~4 hours |
| NVIDIA RTX 3060 | 100 | 32 | ~8 minutes |
| NVIDIA RTX 4090 | 100 | 64 | ~3 minutes |
| NVIDIA A100 | 100 | 128 | ~1 minute |

### Training Output

```
training/runs/uae_retail/
├── weights/
│   ├── best.pt      ← Best model (by mAP)
│   └── last.pt      ← Final epoch
├── confusion_matrix.png
├── results.csv
├── results.png      ← Loss + metric curves
├── val_batch0_pred.jpg  ← Sample predictions
└── ...
```

### Monitoring Training

Watch loss curves:

```bash
# In real-time
tail -f training/runs/uae_retail/results.csv

# Or view results.png after training
```

Expected final metrics (synthetic data):

| Metric | Expected Value |
|---|---|
| mAP@0.5 | >0.90 |
| mAP@0.5:0.95 | >0.65 |
| Precision | >0.85 |
| Recall | >0.88 |

---

## 5. Export Model

Training auto-exports the model. To manually export:

```bash
# Export best model to ONNX
python training/train.py --export-only

# This creates:
#   uae_retail_n.pt       ← Fine-tuned PyTorch model
#   uae_retail_n.onnx     ← Fine-tuned ONNX model (for faster CPU inference)
```

---

## 6. Integrate into Video Analyzer

### Option A: Replace YOLO model path

Edit `.env` or set environment variable:

```bash
YOLO_MODEL=uae_retail_n.pt uvicorn app.main:app --port 8000
```

Or for ONNX:

```bash
USE_ONNX=true YOLO_MODEL=uae_retail_n.pt uvicorn app.main:app --port 8000
```

### Option B: Run alongside COCO model

Modify `app/detectors/yolo_detector.py` to load a second model:

```python
class YOLODetector:
    def __init__(self):
        self.model = None
        self.uae_model = None

    def load_model(self):
        self.model = YOLO("yolov8n.pt")           # General detection (person, bag, etc.)
        self.uae_model = YOLO("uae_retail_n.pt")   # UAE-specific retail items
```

Then in `video_processor.py`:

```python
# Run both models
tracked = self.detector.track_frame(frame)         # People + general objects
uae_dets = self.detector.uae_model(frame)           # UAE retail items
```

This gives you the best of both: YOLOv8n detects people/bags/phones (COCO), and
the fine-tuned model detects UAE-specific items like dallah, dates, perfumes.

### Option C: Full custom model

If you want to detect ONLY your custom classes (no COCO):

```bash
YOLO_MODEL=uae_retail_n.pt uvicorn app.main:app --port 8000
```

The model will detect dallah, perfume_bottle, gold_box, electronics,
dates_box, wallet, watch — ignoring generic COCO classes.

---

## 7. Collect Real Data (For Production)

Synthetic data is great for prototyping, but for a production system you need
real images from UAE retail stores. Here's how:

### Method 1: Phone Photos

1. Visit Carrefour, Lulu, Union Coop, or any UAE supermarket
2. Take 30–50 photos per item class on your phone
3. Include variations: different angles, lighting, positions on shelf, in hand
4. Transfer photos to the project: `training/real_data/{class_name}/*.jpg`

### Method 2: CCTV Frame Extraction

```bash
# Extract frames from existing surveillance footage
python -c "
import cv2, os
cap = cv2.VideoCapture('store_footage.mp4')
os.makedirs('training/real_frames', exist_ok=True)
count = 0
while count < 200:
    ret, frame = cap.read()
    if not ret: break
    if count % 30 == 0:  # Every 30th frame
        cv2.imwrite(f'training/real_frames/frame_{count:04d}.jpg', frame)
    count += 1
"
```

### Method 3: Label with Roboflow

1. Upload images to [Roboflow](https://roboflow.com) (free tier = 1000 images)
2. Draw bounding boxes with their web annotation tool
3. Export in YOLOv8 format → download ZIP
4. Unzip into `training/data/` (replacing the synthetic data)

### Recommended Dataset Size

| Quality | Images per class | Total images | Expected mAP |
|---|---|---|---|
| Toy demo (synthetic) | 85 | 600 | >0.90 (on synthetic) |
| Good | 200 | 1,400 | >0.85 (on real) |
| Production | 500+ | 3,500+ | >0.93 (on real) |

### Label Format

YOLO format: one `.txt` file per image, one line per object:

```
<class_id> <cx> <cy> <width> <height>
```

Each value normalized 0–1. Example:

```
0 0.45 0.52 0.12 0.18
3 0.78 0.63 0.08 0.14
```

Where `cx, cy` = center, `width, height` = box size, all relative to image dimensions.

---

## 8. Full Training Pipeline (Real Data)

```bash
# 1. Activate GPU environment
source .venv/bin/activate

# 2. Replace synthetic data with real data
#    - Put images in training/data/images/
#    - Put labels in training/data/labels/
#    - Update training/config.yaml if needed

# 3. Train on real data (faster convergence)
python training/train.py \
    --epochs 300 \
    --batch 64 \
    --lr 0.01 \
    --model yolov8n.pt

# 4. Evaluate on a test video
python -c "
from ultralytics import YOLO
import cv2
model = YOLO('training/runs/uae_retail/weights/best.pt')
cap = cv2.VideoCapture('test_video.mp4')
while True:
    ret, frame = cap.read()
    if not ret: break
    results = model(frame)[0]
    # Draw and display predictions
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = box.conf[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f'{model.names[cls]} {conf:.2f}', (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    cv2.imshow('Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
"

# 5. Export for deployment
python training/train.py --export-only
```

---

## 9. Before/After Comparison

### Demo Script

```bash
python -c "
from ultralytics import YOLO
import cv2, time

# Load both models
coco = YOLO('yolov8n.pt')
finetuned = YOLO('uae_retail_n.pt')

cap = cv2.VideoCapture('test_uae_items.mp4')
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret: break
    if frame_count % 30 != 0:
        frame_count += 1
        continue
    frame_count += 1

    # COCO detects generic 'bottle', 'cell phone', etc.
    r1 = coco(frame)[0]
    # Fine-tuned detects dallah, perfume, dates, etc.
    r2 = finetuned(frame)[0]

    print(f'Frame {frame_count}:')
    print(f'  COCO: {len(r1.boxes)} objects')
    for b in r1.boxes:
        print(f'    {r1.names[int(b.cls[0])]}: {b.conf[0]:.2f}')
    print(f'  UAE model: {len(r2.boxes)} objects')
    for b in r2.boxes:
        print(f'    {r2.names[int(b.cls[0])]}: {b.conf[0]:.2f}')
    print()

    if frame_count > 50:  # Check first 50 frames
        break

cap.release()
" 2>&1 | head -30
```

Expected result: COCO model sees "bottle" or nothing for a dallah (Arabic coffee
pot). Fine-tuned model correctly identifies it as "dallah".

---

## 10. Troubleshooting

### Training is slow on GPU

```bash
# Check GPU utilization
nvidia-smi -l 1

# If GPU < 80%, increase batch size
python training/train.py --batch 64

# If still slow, check for CPU bottleneck in data loading
# Add workers (not supported in ultralytics directly, but you can
# pre-cache the dataset to RAM):
python -c "
import numpy as np, pickle, cv2, os
from pathlib import Path
data = {}
for f in Path('training/data/images').glob('*.jpg'):
    img = cv2.imread(str(f))
    lab = Path('training/data/labels') / (f.stem + '.txt')
    with open(lab) as lf:
        labels = lf.read().strip()
    data[f.name] = (img, labels)
pickle.dump(data, open('training/data/cache.pkl', 'wb'))
print('Dataset cached to RAM')
"
```

### Out of memory (CUDA OOM)

```bash
# Reduce batch size
python training/train.py --batch 16

# Or use a smaller model
python training/train.py --model yolov8n.pt

# Or reduce image size
python training/train.py --imgsz 416
```

### Synthetic data doesn't generalize to real video

Normal. Synthetic data is for prototyping the pipeline.
For production, collect 200+ real images per class (Section 7).

### NaN loss during training

```bash
# Reduce learning rate
python training/train.py --lr 0.0001

# Or use gradient clipping (edit train.py):
# model.train(... clip_grad=10.0 ...)
```

---

## 11. Quick Reference

```bash
# === SETUP ===
git clone <repo> && cd video-analyzer
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# === TRAIN ===
python training/train.py                                    # 100 epochs, batch 8
python training/train.py --epochs 200 --batch 64           # Custom

# === EXPORT ===
python training/train.py --export-only                     # Creates uae_retail_n.pt + .onnx

# === DEPLOY ===
YOLO_MODEL=uae_retail_n.pt uvicorn app.main:app --port 8000
# Or with ONNX: USE_ONNX=true YOLO_MODEL=uae_retail_n.pt ...

# === ADD NEW CLASS ===
# 1. Add make_<item>() to training/generate_data.py
# 2. Add name to CLASSES
# 3. Regenerate: python training/generate_data.py
# 4. Retrain:   python training/train.py --epochs 50

# === REAL DATA ===
# Put images in training/data/images/
# Put YOLO labels in training/data/labels/
# Run: python training/train.py --epochs 300
```
