# AI Video Intelligence Platform - System Architecture

## High Level Architecture

```text
IP Camera / CCTV
       │
       │ RTSP / ONVIF
       ▼
Stream Manager (FFmpeg/GStreamer)
       ▼
YOLO Detector (ONNX Runtime)
       ▼
ByteTrack Tracker
       ▼
Trajectory Store
       ▼
Event Engine
       ├── PostgreSQL
       ├── Redis
       ▼
FastAPI Backend
       ▼
React Dashboard
```

## Detection Layer
- Person, vehicle, bicycle, motorcycle detection
- YOLOv11 (future YOLOv12)
- PyTorch (dev), ONNX Runtime (prod), TensorRT (NVIDIA)

## Tracking Layer
- ByteTrack (preferred)
- StrongSORT, DeepSORT, OCSORT alternatives

## Event Engine
- Entry/Exit
- Line crossing
- Zone events
- Restricted area
- Loitering
- Queue detection
- Abandoned object

## Hardware Strategy
- Mac Mini M3 (development)
- Orange Pi 5 / Radxa Rock 5B (edge)
- Intel N100 / Jetson Orin Nano (mid-tier)
- RTX 3060 / Tesla T4 (enterprise)
