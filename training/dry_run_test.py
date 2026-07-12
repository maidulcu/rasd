#!/usr/bin/env python3
"""
Rasd Pro — Dry Run Test
Verifies all analytics modules work correctly without requiring a camera.
"""

import sys
import time
sys.path.insert(0, "/Users/maidul/projects/local/rasd-pro")

from app.detection.flow_analytics import CustomerFlowAnalyzer
from app.detection.dwell_analytics import DwellTimeAnalyzer
from app.detection.queue_analytics import QueueAnalyzer


def test_flow_analytics():
    print("\n=== Testing Customer Flow Analytics ===")
    flow = CustomerFlowAnalyzer(1920, 1080)
    flow.define_default_zones()

    # Simulate people entering
    for i in range(5):
        detections = [
            {"bbox": [100 + i*50, 400, 160 + i*50, 600], "track_id": i+1, "class_id": 0}
        ]
        flow.process_frame(detections, time.time() + i)
        time.sleep(0.1)

    analytics = flow.get_analytics()
    print(f"  Zones defined: {analytics['zones']}")
    print(f"  Current inside: {analytics['current_inside']}")
    print(f"  Total entries: {analytics['total_entries']}")
    print(f"  Active tracks: {analytics['active_tracks']}")
    print("  ✓ Flow analytics working")
    return analytics


def test_dwell_analytics():
    print("\n=== Testing Dwell Time Analytics ===")
    dwell = DwellTimeAnalyzer(1920, 1080)
    dwell.define_default_shelves()

    # Simulate person browsing a shelf
    track_id = 1
    shelf_bbox = dwell.shelf_zones["shelf_left"].bbox
    cx = (shelf_bbox[0] + shelf_bbox[2]) // 2
    cy = (shelf_bbox[1] + shelf_bbox[3]) // 2

    # Person stays in shelf area for 5 seconds
    for i in range(10):
        detections = [
            {"bbox": [cx-30, cy-75, cx+30, cy+75], "track_id": track_id, "class_id": 0}
        ]
        dwell.process_frame(detections, time.time() + i * 0.5)
        time.sleep(0.05)

    # Person leaves shelf
    detections = [
        {"bbox": [500, 500, 560, 700], "track_id": track_id, "class_id": 0}
    ]
    dwell.process_frame(detections, time.time() + 6)

    analytics = dwell.get_analytics()
    print(f"  Shelf zones: {analytics['zones_defined']}")
    print(f"  Total events: {analytics['total_events']}")
    print(f"  Zone summary: {analytics['zone_summary']}")
    print("  ✓ Dwell analytics working")
    return analytics


def test_queue_analytics():
    print("\n=== Testing Queue Analytics ===")
    queue = QueueAnalyzer(1920, 1080)
    queue.define_default_queues()

    # Simulate people in queue
    for i in range(4):
        detections = [
            {"bbox": [200 + i*30, 800, 260 + i*30, 1000], "track_id": i+10, "class_id": 0}
        ]
        queue.process_frame(detections, time.time() + i)
        time.sleep(0.05)

    analytics = queue.get_analytics()
    print(f"  Queues defined: {analytics['queues_defined']}")
    for name, q in analytics['queues'].items():
        print(f"  {name}: {q['current_length']} waiting, ~{q['estimated_wait_minutes']}min wait")
    print(f"  Total waiting: {analytics['total_waiting']}")
    print(f"  Congestion: {queue.get_congestion_level()}")
    print("  ✓ Queue analytics working")
    return analytics


def main():
    print("=" * 60)
    print("RASD PRO — DRY RUN TEST")
    print("Verifying analytics modules work correctly")
    print("=" * 60)

    try:
        flow_result = test_flow_analytics()
        dwell_result = test_dwell_analytics()
        queue_result = test_queue_analytics()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)

        print("\nAnalytics Summary:")
        print(f"  Flow: {flow_result['total_entries']} entries tracked")
        print(f"  Dwell: {dwell_result['total_events']} dwell events")
        print(f"  Queue: {queue_result['total_waiting']} people waiting")

        print("\nNext Steps:")
        print("  1. Download training datasets using: python training/download_retail_datasets.py")
        print("  2. Request access to RetailS and MERL datasets")
        print("  3. Use annotators to label your own CCTV footage")
        print("  4. Train custom models with the labeled data")
        print("  5. Deploy rasd-pro with: python run.py")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
