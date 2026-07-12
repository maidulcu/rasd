#!/usr/bin/env python3
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
