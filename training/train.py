#!/usr/bin/env python3
"""
Train YOLOv8 on your own dataset.

Usage:
  python training/train.py --data dataset.yaml     # Train with your data
  python training/train.py --epochs 50 --batch 16   # Custom settings
  python training/train.py --device mps             # Apple GPU
  python training/train.py --export-only            # Export to ONNX
"""

import argparse, os, shutil
from pathlib import Path

os.environ["YOLO_VERBOSE"] = "False"

from ultralytics import YOLO


def detect_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train(args):
    device = args.device or detect_device()

    model = YOLO(args.model)

    print(f"\nTraining YOLOv8")
    print(f"  Data:    {args.data}")
    print(f"  Device:  {device}")
    print(f"  Epochs:  {args.epochs}")
    print(f"  Batch:   {args.batch}")
    print(f"  ImgSz:   {args.imgsz}")
    print(f"  Resume:  {args.resume}\n")

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=0.001,
        device=device,
        patience=20,
        seed=42,
        pretrained=True,
        augment=True,
        project="training/runs",
        name=args.name,
        exist_ok=True,
        resume=args.resume,
        verbose=True,
    )

    print(f"\nTraining complete. Results saved to training/runs/{args.name}")
    return results


def export_model(args):
    run_dir = Path("training/runs") / args.name
    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"

    model_path = best_pt if best_pt.exists() else last_pt
    if not model_path.exists():
        print(f"No trained model found at {best_pt} or {last_pt}")
        return

    print(f"Exporting {model_path} to ONNX...")
    model = YOLO(str(model_path))
    onnx_path = model.export(format="onnx", imgsz=args.imgsz)

    dst = Path(f"{args.name}.pt")
    shutil.copy(str(model_path), str(dst))
    print(f"  Model:  {dst}")

    dst_onnx = Path(f"{args.name}.onnx")
    if Path(onnx_path).exists():
        shutil.copy(str(onnx_path), str(dst_onnx))
        print(f"  ONNX:   {dst_onnx}")

    print(f"\nUse your model by setting YOLO_MODEL={dst} in .env")


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on your own dataset")
    parser.add_argument("--data", type=str, default="dataset.yaml",
                        help="Path to dataset YAML (YOLO format)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="Base model (yolov8n.pt, yolov8s.pt, etc.)")
    parser.add_argument("--device", type=str, default=None,
                        help="cpu, cuda, mps")
    parser.add_argument("--name", type=str, default="custom_model",
                        help="Run name (output directory)")
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
