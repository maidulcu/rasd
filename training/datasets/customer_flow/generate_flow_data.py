#!/usr/bin/env python3
"""
Synthetic Customer Flow Data Generator
Generates YOLO-format annotations for people counting.
"""
import random
import json
from pathlib import Path

def generate_flow_frame(frame_id, num_people, width=1920, height=1080):
    """Generate synthetic people detections for a frame."""
    detections = []
    for i in range(num_people):
        # Random position with bias toward entry/exit
        x = random.randint(0, width - 100)
        y = random.randint(0, height - 200)
        w = random.randint(60, 120)
        h = random.randint(150, 250)
        conf = random.uniform(0.7, 0.99)
        
        detections.append({
            "class": 0,  # person
            "bbox": [x, y, x + w, y + h],
            "confidence": conf,
            "track_id": random.randint(1, num_people),
            "zone": "entry" if x < width // 3 else ("exit" if x > 2 * width // 3 else "floor"),
        })
    return {
        "frame_id": frame_id,
        "width": width,
        "height": height,
        "people_count": len(detections),
        "detections": detections,
    }

def generate_day_data(num_frames=3000, output_dir="output"):
    """Generate a full day of flow data."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    frames = []
    for i in range(num_frames):
        # Simulate varying traffic (busy during lunch, quiet at night)
        hour = (i // 125) % 24  # 25 fps * 120s = 3000 frames = 2 min
        if 11 <= hour <= 13:
            num_people = random.randint(8, 20)  # lunch rush
        elif 17 <= hour <= 19:
            num_people = random.randint(5, 15)  # evening
        else:
            num_people = random.randint(1, 8)   # normal
        
        frame = generate_flow_frame(i, num_people)
        frames.append(frame)
    
    # Save
    with open(output_path / "flow_data.json", "w") as f:
        json.dump(frames, f, indent=2)
    
    print(f"Generated {len(frames)} frames in {output_path}")
    return frames

if __name__ == "__main__":
    generate_day_data()
