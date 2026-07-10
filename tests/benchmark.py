#!/usr/bin/env python3
"""
Performance benchmark: times every component in the video pipeline.
Run: python tests/benchmark.py [video_path] [num_frames]
"""

import os, sys, time, statistics, tracemalloc

os.environ["YOLO_VERBOSE"] = "False"
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.core.config import settings
from app.detectors.yolo_detector import YOLODetector
from app.detectors.face_detector import FaceDetector
from app.detectors.pose_detector import PoseDetector
from app.detectors.theft_detector import TheftDetector

FRAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 300
VIDEO = sys.argv[1] if len(sys.argv) > 1 else "downloads/19ba1e3ebd20421780a183d18b517251.mp4"

OUT_W, OUT_H = 1280, 720
SKIP = settings.FRAME_SKIP  # 2


def benchmark():
    cap = cv2.VideoCapture(VIDEO)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    scale = min(OUT_W / orig_w, 1.0)
    out_w, out_h = int(orig_w * scale), int(orig_h * scale)

    print(f"{'Component':<30s} {'Count':>6s} {'Avg (ms)':>10s} {'Min (ms)':>10s} {'Max (ms)':>10s} {'Total (s)':>10s}")
    print("-" * 76)

    # Init models
    t0 = time.perf_counter()
    yolo = YOLODetector()
    yolo.load_model()
    face = FaceDetector()
    face.load_model()
    pose = PoseDetector()
    pose.load_model()
    model_load = time.perf_counter() - t0
    print(f"{'Model load time':<30s} {'':>6s} {'':>10s} {'':>10s} {'':>10s} {model_load:>9.2f}s")

    theft = TheftDetector()
    fourcc = cv2.VideoWriter_fourcc(*"avc1")

    # Timers
    timers = {
        "read+resize": [],
        "track_frame": [],
        "detect_frame": [],
        "face_detect": [],
        "pose_detect": [],
        "theft_update": [],
        "draw_boxes": [],
        "draw_faces": [],
        "draw_skeleton": [],
        "draw_theft": [],
        "video_write": [],
        "total_frame": [],
    }

    f_idx = 0
    frames_processed = 0

    # Pre-allocate writer so we can time it
    writer = cv2.VideoWriter("/dev/null", fourcc, src_fps, (out_w, out_h))

    # Warmup — one silent frame
    ret, warmup = cap.read()
    if ret:
        if scale < 1.0:
            warmup = cv2.resize(warmup, (out_w, out_h))
        yolo.track_frame(warmup)
        yolo.detect_frame(warmup)
        face.detect(warmup)
        pose.detect(warmup)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # reset

    tracemalloc.start()

    while f_idx < tot_frames and frames_processed < FRAMES:
        t_start = time.perf_counter()

        t_read = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break
        if scale < 1.0:
            frame = cv2.resize(frame, (out_w, out_h))
        timers["read+resize"].append((time.perf_counter() - t_read) * 1000)

        if f_idx % (SKIP + 1) == 0:
            frames_processed += 1

            t0 = time.perf_counter()
            tracked = yolo.track_frame(frame)
            timers["track_frame"].append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            faces = face.detect(frame)
            timers["face_detect"].append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            pose_persons = pose.detect(frame)
            timers["pose_detect"].append((time.perf_counter() - t0) * 1000)

            person_boxes = [d["bbox"] for d in tracked if d.get("class_id") == 0]
            person_ids = [d.get("track_id") for d in tracked if d.get("class_id") == 0]

            t0 = time.perf_counter()
            # track_frame already returns all COCO classes; reuse for theft
            theft.update(tracked, person_boxes, person_ids)
            timers["theft_update"].append((time.perf_counter() - t0) * 1000)
        else:
            t0 = time.perf_counter()
            tracked = yolo.detect_frame(frame)
            for t in tracked:
                t["track_id"] = None
            timers["track_frame"].append((time.perf_counter() - t0) * 1000)
            faces = []
            pose_persons = []

        # Drawing
        t0 = time.perf_counter()
        active_ids = set()
        colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (0, 128, 128),
        ]
        for det in tracked:
            cid = det.get("class_id", -1)
            x1, y1, x2, y2 = det["bbox"]
            tid = det.get("track_id")
            if tid is not None and cid == 0:
                active_ids.add(tid)
            color = colors[(tid or 0) % len(colors)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"Person #{tid}" if tid is not None else f"{det.get('label', '?')} {det['confidence']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        timers["draw_boxes"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        for face_d in faces:
            x1, y1, x2, y2 = face_d["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, "Face", (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        timers["draw_faces"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        pose.draw_skeleton(frame, pose_persons)
        timers["draw_skeleton"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        for interaction in theft.current_interactions:
            x1, y1, x2, y2 = interaction["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 3)
            cv2.putText(frame, f"PICKED UP: {interaction['label']}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)
        for alert in theft.unattended_alerts:
            x1, y1, x2, y2 = alert["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, f"UNATTENDED: {alert['label']}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        for alert in theft.theft_alerts:
            x1, y1, x2, y2 = alert.get("person_location", alert["bbox"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
            cv2.putText(frame, f"THEFT: {alert['label']} concealed!", (x1, y2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2, cv2.LINE_AA)
        timers["draw_theft"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        writer.write(frame)
        timers["video_write"].append((time.perf_counter() - t0) * 1000)

        timers["total_frame"].append((time.perf_counter() - t_start) * 1000)
        f_idx += 1

        if f_idx % 100 == 0:
            print(f"  Progress: {f_idx}/{min(tot_frames, FRAMES)} frames ({(f_idx/min(tot_frames, FRAMES))*100:.0f}%)", file=sys.stderr)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    cap.release()
    writer.release()

    # Report
    print()
    for key, vals in timers.items():
        if not vals:
            continue
        label = key.replace("_", " ").title()
        print(f"{label:<30s} {len(vals):>6d} {statistics.mean(vals):>9.2f}ms {min(vals):>9.2f}ms {max(vals):>9.2f}ms {sum(vals)/1000:>9.2f}s")

    # Summary
    total_process = sum(timers["total_frame"])
    det_frames = frames_processed
    print()
    print(f"{'---- SUMMARY':-^76s}")
    print(f"{'Frames processed (tracked):':<40s} {det_frames}")
    print(f"{'Total frames read:':<40s} {f_idx}")
    print(f"{'Source video':<40s} {VIDEO}")
    print(f"{'Source resolution':<40s} {orig_w}x{orig_h}")
    print(f"{'Output resolution':<40s} {out_w}x{out_h}")
    print(f"{'Source FPS':<40s} {src_fps}")
    print(f"{'FRAME_SKIP':<40s} {SKIP}")
    print(f"{'Total pipeline time:':<40s} {total_process/1000:.2f}s")
    print(f"{'Effective processing rate:':<40s} {det_frames / (total_process/1000):.1f} fps")
    print(f"{'Real-time factor (vs source):':<40s} {(total_process/1000) / (f_idx/src_fps):.2f}x")
    print(f"{'Memory peak (tracemalloc):':<40s} {peak/1e6:.1f} MB")
    print(f"{'Memory current:':<40s} {current/1e6:.1f} MB")


if __name__ == "__main__":
    benchmark()
