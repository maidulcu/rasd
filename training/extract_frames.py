#!/usr/bin/env python3
"""
Extract frames from sample videos for training.
Uses existing model to auto-label frames.

Usage:
    python training/extract_frames.py                    # Extract from all categories
    python training/extract_frames.py --category retail  # Specific category
    python training/extract_frames.py --skip-label       # Don't auto-label
"""

import os
import argparse
import random
import cv2
import numpy as np
from pathlib import Path

SAMPLE_DIR = Path(__file__).parent.parent / "tests" / "sample"
REAL_DATA_DIR = Path(__file__).parent / "real_data"
REAL_IMG_DIR = REAL_DATA_DIR / "images"
REAL_LAB_DIR = REAL_DATA_DIR / "labels"

# We'll use the existing best.pt model for auto-labeling
MODEL_PATH = Path(__file__).parent.parent / "best.pt"

CATEGORIES = [
    "retail_cctv",
    "shoplifting",
    "normal_shopping",
    "employee_activity",
    "customer_flow",
]


def extract_frames(video_path, max_frames=50, skip_frames=2):
    """Extract frames from video, skipping every N frames."""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames = []
    frame_idx = 0
    
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % skip_frames == 0:
            frames.append(frame)
        
        frame_idx += 1
    
    cap.release()
    return frames


def auto_label(frame, model):
    """Use existing model to auto-label a frame."""
    results = model(frame, conf=0.3, verbose=False)
    
    labels = []
    h, w = frame.shape[:2]
    
    for r in results:
        if r.boxes:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Convert to YOLO format (center x, center y, width, height) normalized
                cx = (x1 + x2) / 2 / w
                cy = (y1 + y2) / 2 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                
                labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    
    return labels


def process_category(category, model=None, max_frames=30, skip_frames=3):
    """Process all videos in a category."""
    cat_dir = SAMPLE_DIR / category
    if not cat_dir.exists():
        print(f"Category not found: {cat_dir}")
        return 0
    
    videos = list(cat_dir.glob("*.mp4"))
    if not videos:
        print(f"No videos in {cat_dir}")
        return 0
    
    print(f"\nCategory: {category} ({len(videos)} videos)")
    
    extracted = 0
    for video_path in videos:
        print(f"  Processing: {video_path.name}")
        
        frames = extract_frames(video_path, max_frames=max_frames, skip_frames=skip_frames)
        print(f"    Extracted {len(frames)} frames")
        
        for i, frame in enumerate(frames):
            img_name = f"{category}_{video_path.stem}_{i:04d}.jpg"
            img_path = REAL_IMG_DIR / img_name
            lab_path = REAL_LAB_DIR / img_name.replace(".jpg", ".txt")
            
            # Save image
            cv2.imwrite(str(img_path), frame)
            
            # Auto-label if model available
            if model is not None:
                labels = auto_label(frame, model)
                if labels:
                    lab_path.write_text("\n".join(labels))
            
            extracted += 1
    
    return extracted


def main():
    parser = argparse.ArgumentParser(description="Extract frames from sample videos")
    parser.add_argument("--category", choices=CATEGORIES,
                        help="Extract from specific category")
    parser.add_argument("--skip-label", action="store_true",
                        help="Don't auto-label (just extract frames)")
    parser.add_argument("--max-frames", type=int, default=30,
                        help="Max frames per video")
    parser.add_argument("--skip-frames", type=int, default=3,
                        help="Skip N frames between extractions")
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(REAL_IMG_DIR, exist_ok=True)
    os.makedirs(REAL_LAB_DIR, exist_ok=True)
    
    # Load model for auto-labeling
    model = None
    if not args.skip_label and MODEL_PATH.exists():
        print("Loading model for auto-labeling...")
        from ultralytics import YOLO
        model = YOLO(str(MODEL_PATH))
        print(f"  Model loaded: {MODEL_PATH}")
    
    # Process categories
    categories = [args.category] if args.category else CATEGORIES
    
    total = 0
    for cat in categories:
        total += process_category(
            cat,
            model=model,
            max_frames=args.max_frames,
            skip_frames=args.skip_frames
        )
    
    print(f"\nExtracted {total} frames total")
    print(f"  Images: {REAL_IMG_DIR}")
    print(f"  Labels: {REAL_LAB_DIR}")
    
    # Update config.yaml
    update_config()


def update_config():
    """Update training config.yaml for real data."""
    config_path = Path(__file__).parent / "config.yaml"
    
    config_content = f"""# UAE Retail Dataset - Real Data
path: {REAL_DATA_DIR.resolve()}
train: images
val: images

nc: 7
names: ["dallah", "perfume_bottle", "gold_box", "electronics", "dates_box", "wallet", "watch"]
"""
    config_path.write_text(config_content)
    print(f"\nUpdated config: {config_path}")


if __name__ == "__main__":
    main()
