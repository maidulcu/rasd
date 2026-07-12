#!/usr/bin/env python3
"""
Download retail analytics training datasets.
Categories:
1. Customer Flow & Traffic
2. Dwell Time & Engagement
3. Queue & Checkout
"""

import os
import sys
import urllib.request
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent / "datasets"


def download_file(url: str, dest: Path, desc: str = ""):
    """Download a file with progress."""
    if dest.exists():
        print(f"  [SKIP] {dest.name} already exists")
        return True
    
    print(f"  [DOWN] {desc or dest.name}...")
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
        print(f"  [OK] {dest.name} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


# ============================================================
# CATEGORY 1: Customer Flow & Traffic
# ============================================================
def download_customer_flow():
    print("\n=== CATEGORY 1: Customer Flow & Traffic ===")
    out = BASE_DIR / "customer_flow"
    out.mkdir(parents=True, exist_ok=True)
    
    # 1. RetailS Dataset (shoplifting but has person tracking with pose)
    # Good for: people counting, tracking, flow analysis
    print("\n--- RetailS Dataset (Person Tracking + Pose) ---")
    print("  Source: https://github.com/TeCSAR-UNCC/RetailS")
    print("  Contains: ~20M normal frames, 6 camera views, pose data")
    print("  Use for: people counting, flow patterns, tracking")
    print("  License: Research use - request access")
    
    # 2. Standard AI Tracks Dataset
    print("\n--- Standard AI Tracks Dataset ---")
    print("  Source: https://huggingface.co/datasets/standard-cognition/Tracks")
    print("  Contains: 60K hours, 2.3M shoppers, 3D trajectories")
    print("  Use for: customer flow, path analysis, dwell patterns")
    print("  License: Evaluation subset free, full commercial")
    
    # 3. MALL dataset (people counting)
    print("\n--- Mall Dataset (People Counting) ---")
    mall_url = "https://raw.githubusercontent.com/checkoutfree/checkoutfree/main/README.md"
    download_file(mall_url, out / "README_checkoutfree.md", "Checkout-free grocery info")
    
    # 4. Generate synthetic flow data script
    print("\n--- Creating synthetic flow data generator ---")
    create_flow_generator(out)
    
    return out


# ============================================================
# CATEGORY 2: Dwell Time & Engagement
# ============================================================
def download_dwell_engagement():
    print("\n=== CATEGORY 2: Dwell Time & Engagement ===")
    out = BASE_DIR / "dwell_engagement"
    out.mkdir(parents=True, exist_ok=True)
    
    # 1. MERL Shopping Dataset
    print("\n--- MERL Shopping Dataset ---")
    print("  Source: https://huggingface.co/datasets/Voxel51/MERL_Shopping_Dataset")
    print("  Contains: 106 videos, 5377 action intervals, 5 action classes")
    print("  Actions: Reach, Retract, Hand-in-Shelf, Inspect Product, Inspect Shelf")
    print("  Use for: product interaction, dwell time, browsing behavior")
    print("  License: Non-commercial")
    
    # Download MERL metadata
    merl_url = "https://huggingface.co/datasets/Voxel51/MERL_Shopping_Dataset/raw/main/README.md"
    download_file(merl_url, out / "README_merl.md", "MERL Shopping Dataset README")
    
    # 2. RetailAction Dataset
    print("\n--- RetailAction Dataset ---")
    print("  Source: https://huggingface.co/datasets/Voxel51/RetailAction")
    print("  Contains: 21,000 samples, 10 real stores, multi-view")
    print("  Actions: take, put, touch (product interactions)")
    print("  Use for: product engagement, interaction detection")
    print("  License: Standard AI proprietary")
    
    # 3. VRAI Datasets (HaDa for interactions)
    print("\n--- VRAI/HaDa Dataset ---")
    print("  Source: Request from paper authors")
    print("  Contains: 13,856 labeled frames, shopper-shelf interactions")
    print("  Use for: dwell time measurement, shelf engagement")
    
    # 4. RCA-TVGender Dataset
    print("\n--- RCA-TVGender Dataset ---")
    print("  Source: Paper supplementary materials")
    print("  Contains: 5,930 images, top-view retail, gender-labeled")
    print("  Use for: customer demographics, heatmap generation")
    
    # 5. Create dwell time annotation tool
    print("\n--- Creating dwell time annotation helper ---")
    create_dwell_annotator(out)
    
    return out


# ============================================================
# CATEGORY 3: Queue & Checkout
# ============================================================
def download_queue_checkout():
    print("\n=== CATEGORY 3: Queue & Checkout ===")
    out = BASE_DIR / "queue_checkout"
    out.mkdir(parents=True, exist_ok=True)
    
    # 1. QueueIQ Dataset
    print("\n--- QueueIQ Dataset ---")
    print("  Source: https://universe.roboflow.com/queueiq/queueiq")
    print("  Contains: Labeled queue length images, counter regions")
    print("  Use for: queue counting, line detection")
    print("  License: Roboflow (free tier available)")
    
    # 2. Intel Line Monitoring
    print("\n--- Intel Line Monitoring ---")
    intel_url = "https://raw.githubusercontent.com/intel-iot-devkit/line-monitoring/master/README.md"
    download_file(intel_url, out / "README_intel_line.md", "Intel line monitoring README")
    
    # 3. Collective Activity Dataset
    print("\n--- Collective Activity Dataset ---")
    print("  Source: https://sites.google.com/site/chalearnlaps/")
    print("  Contains: 44 videos, queue detection subset (7 videos)")
    print("  Use for: queue length estimation, waiting detection")
    
    # 4. Create queue zone annotation tool
    print("\n--- Creating queue zone annotator ---")
    create_queue_annotator(out)
    
    return out


# ============================================================
# HELPER: Synthetic Data Generators
# ============================================================
def create_flow_generator(out_dir: Path):
    """Create a script to generate synthetic customer flow data."""
    script = '''#!/usr/bin/env python3
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
'''
    (out_dir / "generate_flow_data.py").write_text(script)
    print(f"  Created: {out_dir / 'generate_flow_data.py'}")


def create_dwell_annotator(out_dir: Path):
    """Create a dwell time annotation helper."""
    script = '''#!/usr/bin/env python3
"""
Dwell Time Annotation Helper
Annotate video frames with shelf interaction regions.
"""
import cv2
import json
from pathlib import Path

class DwellAnnotator:
    def __init__(self, video_path):
        self.video_path = video_path
        self.regions = []
        self.current_region = None
        self.drawing = False
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.current_region = [(x, y)]
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.current_region.append((x, y))
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.current_region:
                self.regions.append({
                    "points": self.current_region,
                    "label": "shelf",
                    "dwell_threshold_sec": 3.0,
                })
                self.current_region = None
                
    def annotate(self, num_frames=100):
        """Annotate shelf regions in video."""
        cap = cv2.VideoCapture(self.video_path)
        cv2.namedWindow("Annotate Shelf Regions")
        cv2.setMouseCallback("Annotate Shelf Regions", self.mouse_callback)
        
        frame_count = 0
        while cap.isOpened() and frame_count < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Draw existing regions
            for region in self.regions:
                pts = region["points"]
                if len(pts) > 2:
                    cv2.polylines(frame, [np.array(pts)], True, (0, 255, 0), 2)
                    
            cv2.imshow("Annotate Shelf Regions", frame)
            key = cv2.waitKey(30)
            if key == 27:  # ESC
                break
            elif key == ord('s'):
                self.save()
                
            frame_count += 1
            
        cap.release()
        cv2.destroyAllWindows()
        
    def save(self, output_path="dwell_regions.json"):
        with open(output_path, "w") as f:
            json.dump(self.regions, f, indent=2)
        print(f"Saved {len(self.regions)} regions to {output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python dwell_annotator.py <video_path>")
        sys.exit(1)
    
    annotator = DwellAnnotator(sys.argv[1])
    annotator.annotate()
'''
    (out_dir / "dwell_annotator.py").write_text(script)
    print(f"  Created: {out_dir / 'dwell_annotator.py'}")


def create_queue_annotator(out_dir: Path):
    """Create a queue zone annotation helper."""
    script = '''#!/usr/bin/env python3
"""
Queue Zone Annotation Helper
Define checkout queue regions for queue length detection.
"""
import cv2
import json
import numpy as np
from pathlib import Path

class QueueAnnotator:
    def __init__(self, video_path):
        self.video_path = video_path
        self.queues = {}
        self.current_queue = None
        self.points = []
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            
    def annotate(self):
        """Annotate queue zones in first frame."""
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        if not ret:
            print("Cannot read video")
            return
            
        cv2.namedWindow("Define Queue Zones")
        cv2.setMouseCallback("Define Queue Zones", self.mouse_callback)
        
        queue_name = input("Enter queue name (e.g., 'checkout_1'): ")
        print("Click to define polygon, press 'q' when done, 'n' for new queue, ESC to finish")
        
        while True:
            display = frame.copy()
            
            # Draw existing queues
            for name, q in self.queues.items():
                pts = np.array(q["points"], np.int32)
                cv2.polylines(display, [pts], True, (0, 255, 0), 2)
                cv2.putText(display, name, tuple(q["points"][0]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
            # Draw current points
            if self.points:
                for p in self.points:
                    cv2.circle(display, p, 5, (0, 0, 255), -1)
                    
            cv2.imshow("Define Queue Zones", display)
            key = cv2.waitKey(30)
            
            if key == 27:  # ESC
                break
            elif key == ord('q') and self.points:
                self.queues[queue_name] = {
                    "points": self.points,
                    "max_capacity": 10,
                    "avg_service_time_sec": 45,
                }
                self.points = []
                queue_name = input("Enter next queue name (or empty to finish): ")
                if not queue_name:
                    break
            elif key == ord('c'):
                self.points = []
                
        cap.release()
        cv2.destroyAllWindows()
        self.save()
        
    def save(self, output_path="queue_zones.json"):
        with open(output_path, "w") as f:
            json.dump(self.queues, f, indent=2)
        print(f"Saved {len(self.queues)} queue zones to {output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python queue_annotator.py <video_path>")
        sys.exit(1)
    
    annotator = QueueAnnotator(sys.argv[1])
    annotator.annotate()
'''
    (out_dir / "queue_annotator.py").write_text(script)
    print(f"  Created: {out_dir / 'queue_annotator.py'}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("RASD - Retail Analytics Training Dataset Downloader")
    print("=" * 60)
    
    flow_dir = download_customer_flow()
    dwell_dir = download_dwell_engagement()
    queue_dir = download_queue_checkout()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print("""
CATEGORY 1: Customer Flow & Traffic
  Dataset: RetailS (request access)
  - 20M normal frames, 6 cameras, pose data
  - Best for: people counting, flow patterns
  
  Dataset: Standard AI Tracks (evaluation available)
  - 60K hours, 2.3M shoppers, 3D trajectories
  - Best for: path analysis, dwell patterns

CATEGORY 2: Dwell Time & Engagement
  Dataset: MERL Shopping (request access)
  - 106 videos, 5 action classes
  - Best for: product interaction, shelf engagement
  
  Dataset: RetailAction (request access)
  - 21K samples, 10 real stores, multi-view
  - Best for: take/put/touch detection

CATEGORY 3: Queue & Checkout
  Dataset: QueueIQ (Roboflow - free tier)
  - Labeled queue images
  - Best for: queue counting, line detection
  
  Dataset: Intel Line Monitoring (code provided)
  - Queue detection reference implementation
  - Best for: virtual line queue counting

TOOLS CREATED:
  - generate_flow_data.py (synthetic data generator)
  - dwell_annotator.py (shelf region annotation)
  - queue_annotator.py (queue zone definition)
""")
    
    print("\nNEXT STEPS:")
    print("1. Request access to RetailS and MERL datasets")
    print("2. Download QueueIQ from Roboflow (free)")
    print("3. Use annotators to label your own CCTV footage")
    print("4. Generate synthetic data for augmentation")
    print("5. Run dry test with: python dry_run_test.py")


if __name__ == "__main__":
    main()
