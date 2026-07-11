# Training Guide — Small Milestones Approach

## Philosophy

**One object at a time. Each detection is a milestone.**

Don't try to detect all 7 items at once. Train on one, verify it works, then add the next. This way you always have a working model and can measure progress.

```
Milestone 1: Detect "watch"           → working model
Milestone 2: Add "wallet"             → model detects 2 items
Milestone 3: Add "gold_box"           → model detects 3 items
...
Milestone 7: All 7 items detected     → full system
```

---

## How YOLO Training Works (Simplified)

```
Your images (shelves with items)
         ↓
    YOLOv8n (pretrained on 80 generic objects)
         ↓
    Fine-tune: retrain last layers on YOUR items
         ↓
    Model learns: "this shape = dallah", "this shape = watch"
         ↓
    Export: custom .pt file for your video analyzer
```

**Key insight:** YOLOv8n already knows how to detect objects (people, bottles, phones).
Fine-tuning teaches it YOUR specific objects. It's like teaching someone who
already speaks a language to recognize new words.

---

## Your Hardware: M3 MacBook Pro 36GB

| | Value |
|--|--|
| **GPU** | Apple Silicon (Metal Performance Shaders) |
| **RAM** | 36GB unified (GPU shares this) |
| **Training speed** | ~15-30 sec/epoch (600 images) |
| **Batch size** | 16-32 (use more = faster + better) |
| **50 epochs** | ~15-30 minutes total |

**This is excellent hardware for this task.** You'll train ~8x faster than the Linux server.

---

## Step-by-Step: First Milestone (Watch Detection)

### 1. Setup on your Mac

```bash
# Clone/copy the project
cd /path/to/Video-Analyzer-Starter

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (MPS is built into PyTorch, no CUDA needed)
pip install -r requirements.txt

# Verify MPS works
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
# Should print: MPS: True
```

### 2. Generate Data for ONE Class

```bash
# Generate 600 images of watches only
python -c "
from training.generate_data import *
import os, shutil

# Clean old data
os.makedirs('training/data/images', exist_ok=True)
os.makedirs('training/data/labels', exist_ok=True)

# Generate 500 train + 100 val for 'watch' only
generate_class_split(['watch'], 500, 0)
generate_class_split(['watch'], 100, 500)

# Write config for 1 class
write_dataset_yaml(classes=['watch'])

print('Done! 600 images of watches generated.')
"
```

**What this creates:**
```
training/data/
├── images/
│   ├── img_000000.jpg    ← shelf with 1-2 watches
│   ├── img_000001.jpg
│   └── ... (600 total)
└── labels/
    ├── img_000000.txt    ← YOLO format: "0 0.45 0.52 0.12 0.18"
    └── ...

training/config.yaml      ← tells YOLO there's 1 class: "watch"
```

### 3. Train on Your Mac

```bash
# Train: 50 epochs, batch 16, Apple GPU
python training/train.py \
    --epochs 50 \
    --batch 16 \
    --device mps \
    --name milestone_1_watch
```

**Expected time: ~15-25 minutes on M3**

### 4. Check Results

After training, you'll see:

```
training/runs/milestone_1_watch/
├── weights/
│   ├── best.pt           ← USE THIS ONE
│   └── last.pt
├── results.csv           ← loss per epoch
├── results.png           ← loss curve graph
├── confusion_matrix.png  ← how well it detects
└── val_batch0_pred.jpg   ← sample predictions
```

**What to look for in `results.png`:**
- `box_loss` should decrease (model learning bounding boxes)
- `mAP@0.5` should go above 0.90 (good detection rate)
- No wild spikes = training is stable

### 5. Test It

```bash
python -c "
from ultralytics import YOLO
import cv2

model = YOLO('training/runs/milestone_1_watch/weights/best.pt')

# Test on a generated image
img = cv2.imread('training/data/images/img_000050.jpg')
results = model(img, verbose=False)[0]

print(f'Detections: {len(results.boxes)}')
for box in results.boxes:
    cls = int(box.cls[0])
    conf = float(box.conf[0])
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    print(f'  {model.names[cls]}: {conf:.2f} at [{x1},{y1},{x2},{y2}]')

cv2.imwrite('milestone_1_test.jpg', results.plot())
print('Saved: milestone_1_test.jpg')
"
```

**Success = it detects "watch" in the image.**

### 6. Export the Model

```bash
python training/train.py --export-only --name milestone_1_watch
```

This creates:
- `milestone_1_watch.pt` — PyTorch model
- `milestone_1_watch.onnx` — ONNX model (faster inference)

---

## Adding More Objects (Next Milestones)

### Milestone 2: Add Wallet

```bash
# Generate data for 2 classes
python -c "
from training.generate_data import *
import os

os.makedirs('training/data/images', exist_ok=True)
os.makedirs('training/data/labels', exist_ok=True)

# 2 classes: watch=0, wallet=1
generate_class_split(['watch', 'wallet'], 500, 0)
generate_class_split(['watch', 'wallet'], 100, 500)
write_dataset_yaml(classes=['watch', 'wallet'])
"

# Train
python training/train.py \
    --epochs 50 \
    --batch 16 \
    --device mps \
    --name milestone_2_watch_wallet
```

### Milestone 3: Add Gold Box

```bash
python -c "
from training.generate_data import *
import os

os.makedirs('training/data/images', exist_ok=True)
os.makedirs('training/data/labels', exist_ok=True)

generate_class_split(['watch', 'wallet', 'gold_box'], 500, 0)
generate_class_split(['watch', 'wallet', 'gold_box'], 100, 500)
write_dataset_yaml(classes=['watch', 'wallet', 'gold_box'])
"

python training/train.py \
    --epochs 50 \
    --batch 16 \
    --device mps \
    --name milestone_3_watch_wallet_gold
```

### Continue Until All 7:

```
Milestone 4: + electronics
Milestone 5: + perfume_bottle
Milestone 6: + dates_box
Milestone 7: + dallah
```

---

## Real Data (When You're Ready)

Synthetic data proves the pipeline works. Real data makes it production-ready.

### Collecting Real Photos

**Minimum viable dataset:** 50 photos per item

```
training/real_data/
├── watch/
│   ├── img_001.jpg       ← photo from Carrefour shelf
│   ├── img_002.jpg
│   └── ... (50+ photos)
├── wallet/
│   ├── img_001.jpg
│   └── ... (50+ photos)
└── gold_box/
    └── ...
```

**Photo tips:**
- Multiple angles (front, side, 45-degree)
- Different lighting (bright store, dim corner)
- On shelf vs. in hand vs. on counter
- Various backgrounds (other items nearby)

### Label with Roboflow (Free)

1. Upload photos to [roboflow.com](https://roboflow.com) (free = 1000 images)
2. Draw bounding boxes around each item
3. Export as YOLOv8 format
4. Replace `training/data/images/` and `training/data/labels/`
5. Retrain

### Train on Real Data

```bash
python training/train.py \
    --epochs 100 \
    --batch 16 \
    --device mps \
    --name production_v1
```

---

## Integrating Into the Video Analyzer

Once you have a trained model:

### Option 1: Replace default model

```bash
# .env
YOLO_MODEL=milestone_1_watch.pt
```

### Option 2: Run both models (recommended)

Edit `app/detectors/yolo_detector.py`:

```python
class YOLODetector:
    def __init__(self):
        self.model = None
        self.custom_model = None

    def load_model(self):
        self.model = YOLO("yolov8n.pt")              # People, bags, phones (COCO)
        self.custom_model = YOLO("milestone_7.pt")    # UAE retail items

    def detect_frame(self, frame):
        # General detections
        results = self.model(frame, conf=CONFIDENCE_THRESHOLD)[0]
        # Custom item detections
        custom_results = self.custom_model(frame, conf=CONFIDENCE_THRESHOLD)[0]
        return results, custom_results
```

This way you get:
- **People detection** (from COCO model) → for theft tracking
- **Custom item detection** (from fine-tuned model) → for valuable item alerts

---

## Troubleshooting

### Training is slow
```bash
# Check if MPS is being used (should see GPU usage in Activity Monitor)
# If not, try:
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7 python training/train.py --device mps
```

### Low accuracy (< 0.70 mAP)
- Need more data: generate 1000+ images or collect real photos
- Reduce augment: edit train.py, set `mosaic=0.0, mixup=0.0`
- Lower learning rate: `--lr 0.0001`

### Model doesn't detect anything
- Check `val_batch0_pred.jpg` — are labels correct?
- Reduce confidence: `model(frame, conf=0.25)`
- Train longer: `--epochs 100`

### Out of memory
```bash
# Reduce batch size
python training/train.py --batch 8 --device mps
```

---

## Quick Reference

```bash
# === FIRST MILESTONE ===
python training/train.py --epochs 50 --batch 16 --device mps --name milestone_1_watch

# === ADD ANOTHER CLASS ===
# 1. Generate data with --classes flag
# 2. Train with new --name

# === CHECK RESULTS ===
# Open training/runs/<name>/results.png

# === EXPORT ===
python training/train.py --export-only --name milestone_1_watch

# === USE IN SERVER ===
YOLO_MODEL=milestone_1_watch.pt uvicorn app.main:app --port 8000
```

---

## Training Time Cheat Sheet (M3 MacBook Pro)

| Images | Epochs | Batch | Time |
|--------|--------|-------|------|
| 600 | 10 | 16 | ~3 min |
| 600 | 50 | 16 | ~15 min |
| 600 | 100 | 16 | ~30 min |
| 2000 | 50 | 16 | ~50 min |
| 2000 | 100 | 16 | ~1.5 hr |
