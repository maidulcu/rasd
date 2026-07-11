#!/usr/bin/env python3
"""
Fine-tune YOLOv8 on a custom UAE retail dataset.

Usage:
  python training/train.py                     # Full training
  python training/train.py --epochs 50         # Custom epochs
  python training/train.py --resume            # Resume from checkpoint
  python training/train.py --export-only       # Just export to ONNX
"""

import argparse, os, sys, shutil
from pathlib import Path

os.environ["YOLO_VERBOSE"] = "False"

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO

DEFAULT_EPOCHS = 100
DEFAULT_BATCH = 8
DEFAULT_IMGSZ = 640
DEFAULT_LR = 0.001


def get_dataset_yaml() -> str:
    """Return path to dataset config."""
    cfg = Path(__file__).parent / "config.yaml"
    if not cfg.exists():
        # Generate dataset first
        print("Dataset not found. Generating synthetic dataset...")
        from training.generate_data import generate_split, write_dataset_yaml, \
            IMG_DIR, LAB_DIR, N_TRAIN, N_VAL
        os.makedirs(IMG_DIR, exist_ok=True)
        os.makedirs(LAB_DIR, exist_ok=True)
        print("  Generating training set...")
        generate_split(N_TRAIN, 0)
        print("  Generating validation set...")
        generate_split(N_VAL, N_TRAIN)
        write_dataset_yaml()
    return str(cfg)


def train(args):
    """Fine-tune YOLOv8n on custom dataset."""
    dataset_yaml = get_dataset_yaml()

    # Load pretrained model
    print(f"Loading pretrained YOLOv8n...")
    model = YOLO(args.model if args.model else "yolov8n.pt")

    print(f"\n{'='*60}")
    print(f"Starting fine-tuning on custom UAE retail dataset")
    print(f"  Dataset: {dataset_yaml}")
    print(f"  Epochs:  {args.epochs}")
    print(f"  Batch:   {args.batch}")
    print(f"  ImgSz:   {args.imgsz}")
    print(f"  LR:      {args.lr}")
    print(f"  Resume:  {args.resume}")
    print(f"{'='*60}\n")

    results = model.train(
        data=dataset_yaml,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr,
        device="cpu",
        patience=20,
        seed=42,
        deterministic=True,
        pretrained=True,
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.3,
        mixup=0.1,
        copy_paste=0.1,
        project="training/runs",
        name="uae_retail",
        exist_ok=True,
        resume=args.resume,
        verbose=False,
    )

    print(f"\nTraining complete! Results saved to training/runs/uae_retail")
    return results


def export_model(args):
    """Export trained model to ONNX and copy to project root."""
    # Find best model
    best_pt = Path("training/runs/uae_retail/weights/best.pt")
    last_pt = Path("training/runs/uae_retail/weights/last.pt")

    model_path = best_pt if best_pt.exists() else last_pt
    if not model_path.exists():
        print(f"No trained model found at {best_pt} or {last_pt}")
        print("Run training first: python training/train.py")
        return

    print(f"Loading trained model: {model_path}")
    model = YOLO(str(model_path))

    # Export to ONNX
    print("Exporting to ONNX...")
    onnx_path = model.export(format="onnx", imgsz=args.imgsz)
    print(f"  ONNX model: {onnx_path}")

    # Also export the model with a UAE-specific name
    dst = Path("uae_retail_n.pt")
    shutil.copy(str(model_path), str(dst))
    print(f"  Copied to: {dst}")

    dst_onnx = Path("uae_retail_n.onnx")
    if Path(onnx_path).exists():
        shutil.copy(str(onnx_path), str(dst_onnx))
        print(f"  Copied to: {dst_onnx}")

    # Test inference
    print("\nTesting inference with fine-tuned model...")
    import cv2, numpy as np, time
    from training.generate_data import make_background, ITEM_MAKERS, composite

    # Create a test scene
    bg = make_background(640, 640)
    test_img = bg.copy()
    item_img = ITEM_MAKERS[0]()  # dallah
    test_img, _ = composite(item_img, test_img, max_instances=2)
    item_img2 = ITEM_MAKERS[1]()  # perfume
    test_img, _ = composite(item_img2, test_img, max_instances=1)

    # Test with PyTorch
    t0 = time.perf_counter()
    r = model(test_img, verbose=False)[0]
    pt_time = (time.perf_counter() - t0) * 1000
    print(f"  PyTorch inference: {pt_time:.1f}ms, {len(r.boxes)} detections")

    # Test with ONNX if available
    if Path(onnx_path).exists():
        model_onnx = YOLO(str(onnx_path), task="detect")
        t0 = time.perf_counter()
        r = model_onnx(test_img, verbose=False)[0]
        onnx_time = (time.perf_counter() - t0) * 1000
        print(f"  ONNX inference:   {onnx_time:.1f}ms, {len(r.boxes)} detections")

    print(f"\nFine-tuned model ready: {dst}")

    # Show sample detections
    if len(r.boxes) > 0:
        print(f"\nSample detections:")
        for i, box in enumerate(r.boxes[:5]):
            cls_id = int(box.cls[0])
            conf = box.conf[0]
            print(f"  {i+1}. {CLASSES[cls_id] if cls_id < len(CLASSES) else '?'} ({conf:.2f})")

    return model


CLASSES = ["dallah", "perfume_bottle", "gold_box",
           "electronics", "dates_box", "wallet", "watch"]


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on UAE retail dataset")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="Pretrained model path")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()

    if args.export_only:
        export_model(args)
    else:
        train(args)
        export_model(args)


if __name__ == "__main__":
    main()
