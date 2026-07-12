# Training — Rasd Free

Fine-tune YOLOv8 on your own data to improve detection for your specific environment.

## What You Can Train

| Task | Example | Base Model |
|------|---------|------------|
| Person detection | Store entrance, warehouse | yolov8n.pt |
| Custom objects | Any items not in COCO | yolov8n.pt |
| Pose estimation | Specific actions | yolov8n-pose.pt |

## How It Works

The free version uses **pretrained YOLOv8 models** (COCO — 80 classes). Training lets you fine-tune on your own images to detect things that matter to you.

## Quick Start

### 1. Prepare Data

Organize images in YOLO format:

```
dataset/
├── train/
│   ├── images/   # .jpg files
│   └── labels/   # .txt files (YOLO format)
├── val/
│   ├── images/
│   └── labels/
└── dataset.yaml  # config file
```

Create `dataset.yaml`:
```yaml
path: /path/to/dataset
train: train/images
val: val/images
nc: 2  # number of classes
names: ["person", "product"]  # your class names
```

### 2. Train

```bash
python training/train.py --data dataset.yaml --epochs 100
```

### 3. Use Your Model

Set in `.env`:
```
YOLO_MODEL=custom_model.pt
```

Or for edge deployment:
```
YOLO_MODEL=custom_model.onnx
USE_ONNX=True
```

## Tips

- **50+ images per class** minimum for meaningful results
- **500+ images per class** recommended for production
- Start with **yolov8n.pt** (fastest), upgrade to yolov8s.pt for better accuracy
- Use `--device mps` on Apple Silicon, `--device cuda` on NVIDIA

## Free vs Pro Training

| Free | Pro |
|------|-----|
| Train any YOLO-format dataset | Train any YOLO-format dataset |
| Generic base models (yolov8n.pt) | + UAE retail pretrained models |
| User provides own data | + Synthetic data generators |
| Manual data labeling | + Dataset download tools |
| — | + 7 UAE retail classes ready |
| — | + Real training data (5.4GB) |
| — | + Queue, dwell, flow datasets |
