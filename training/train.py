#!/usr/bin/env python3
"""
Fine-tune YOLOv8 on a custom UAE retail dataset.

Usage:
  python training/train.py                                    # Auto-detect device, 100 epochs
  python training/train.py --epochs 50 --batch 16             # Custom
  python training/train.py --device mps                       # Force Apple GPU
  python training/train.py --device cuda                      # Force NVIDIA GPU
  python training/train.py --name dallah                      # Name the run
  python training/train.py --resume                           # Resume from checkpoint
  python training/train.py --export-only                      # Just export to ONNX
"""

import argparse, os, sys, shutil
from pathlib import Path

os.environ["YOLO_VERBOSE"] = "False"

sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO

DEFAULT_EPOCHS = 100
DEFAULT_BATCH = 8
DEFAULT_IMGSZ = 640
DEFAULT_LR = 0.001


def detect_device():
    """Auto-detect best available device."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_dataset_yaml(classes=None) -> str:
    """Return path to dataset config. Optionally filter to specific classes."""
    cfg = Path(__file__).parent / "config.yaml"
    if not cfg.exists():
        print("Dataset not found. Generating synthetic dataset...")
        from training.generate_data import generate_split, write_dataset_yaml, \
            IMG_DIR, LAB_DIR
        os.makedirs(IMG_DIR, exist_ok=True)
        os.makedirs(LAB_DIR, exist_ok=True)

        n_train = 500
        n_val = 100

        if classes:
            print(f"  Generating for classes: {classes}")
            from training.generate_data import generate_class_split
            generate_class_split(classes, n_train, 0)
            generate_class_split(classes, n_val, n_train)
        else:
            print("  Generating full training set...")
            from training.generate_data import generate_split as gs, N_TRAIN, N_VAL
            gs(N_TRAIN, 0)
            print("  Generating full validation set...")
            gs(N_VAL, N_TRAIN)

        write_dataset_yaml(classes=classes)
    return str(cfg)


def train(args):
    """Fine-tune YOLOv8n on custom dataset."""
    device = args.device or detect_device()
    dataset_yaml = get_dataset_yaml(classes=args.classes)

    model = YOLO(args.model if args.model else "yolov8n.pt")

    print(f"\n{'='*60}")
    print(f"Fine-tuning YOLOv8n")
    print(f"  Dataset:  {dataset_yaml}")
    print(f"  Device:   {device}")
    print(f"  Epochs:   {args.epochs}")
    print(f"  Batch:    {args.batch}")
    print(f"  ImgSz:    {args.imgsz}")
    print(f"  LR:       {args.lr}")
    print(f"  Resume:   {args.resume}")
    if args.classes:
        print(f"  Classes:  {', '.join(args.classes)}")
    print(f"{'='*60}\n")

    results = model.train(
        data=dataset_yaml,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr,
        device=device,
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
        name=args.name,
        exist_ok=True,
        resume=args.resume,
        verbose=True,
    )

    print(f"\nTraining complete! Results saved to training/runs/{args.name}")
    return results


def export_model(args):
    """Export trained model to ONNX and copy to project root."""
    run_dir = Path("training/runs") / args.name
    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"

    model_path = best_pt if best_pt.exists() else last_pt
    if not model_path.exists():
        print(f"No trained model found at {best_pt} or {last_pt}")
        print("Run training first: python training/train.py")
        return

    print(f"Loading trained model: {model_path}")
    model = YOLO(str(model_path))

    print("Exporting to ONNX...")
    onnx_path = model.export(format="onnx", imgsz=args.imgsz)
    print(f"  ONNX model: {onnx_path}")

    dst = Path(f"{args.name}.pt")
    shutil.copy(str(model_path), str(dst))
    print(f"  Copied to: {dst}")

    dst_onnx = Path(f"{args.name}.onnx")
    if Path(onnx_path).exists():
        shutil.copy(str(onnx_path), str(dst_onnx))
        print(f"  Copied to: {dst_onnx}")

    print(f"\nFine-tuned model ready: {dst}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on custom dataset")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device: cpu, cuda, mps")
    parser.add_argument("--name", type=str, default="uae_retail",
                        help="Run name (used for output directory)")
    parser.add_argument("--classes", type=str, nargs="+", default=None,
                        help="Train on specific classes only (e.g. --classes dallah)")
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
