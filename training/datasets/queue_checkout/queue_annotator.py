#!/usr/bin/env python3
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
