# Rasd — AI Video Analytics

<div align="center">

**AI-powered CCTV & surveillance video analysis for smart cities, retail security, and public safety.**

[![Python](https://img.shields.io/badge/Python-3.12+-blue)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)](https://ultralytics.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-teal)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**[Docs](#quick-start)** · **[Report Bug](../../issues)** · **[Request Feature](../../issues)**

</div>

---

## What is Rasd?

**Rasd** (رصد — Arabic for "surveillance") is an open-source AI video intelligence platform that analyzes CCTV and surveillance footage in real-time. Built for **smart cities, retail security, and public safety** in the Gulf region and beyond.

Rasd uses **YOLOv8** for real-time object detection, tracking, and behavior analysis. It can detect people, faces, and objects, track unique individuals across frames, and identify suspicious behavior like theft, concealment, and loitering.

### Key Features

- **Person Detection & Counting** — Track unique people across video frames with YOLOv8 + ByteTrack
- **Object Classification** — Detect 80 COCO classes: bags, phones, laptops, weapons, and more
- **Theft Detection** — Unattended object alerts and concealment detection for retail security
- **Pose Estimation** — Detect suspicious behavior: hand-in-pocket, bending, and unusual poses
- **Face Detection** — Real-time face counting and recognition hooks
- **Zone Monitoring** — Entry/exit zones, shelf engagement, line crossing counts
- **Edge Deployment** — Optimized for Orange Pi 5, Jetson Orin Nano, Raspberry Pi
- **ONNX Export** — Export trained models to ONNX for production deployment

### Use Cases

- **Retail Security** — Prevent shoplifting, monitor customer flow, analyze dwell time
- **Smart Cities** — Traffic monitoring, pedestrian counting, public safety
- **Public Safety** — Suspicious behavior detection, unattended luggage alerts
- **Building Security** — Access control, intrusion detection, face recognition
- **Analytics** — Customer behavior, zone occupancy, traffic patterns

---

## Quick Start

### Prerequisites

- Python 3.12+
- SQLite (included) or PostgreSQL

### Installation

```bash
git clone https://github.com/maidulcu/rasd.git
cd rasd
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open **http://localhost:8000** — upload a video or paste a URL.

### Try the Dashboard

```bash
# Dashboard is at http://localhost:8000/dashboard
```

### Test with Public RTSP Stream

```bash
# Big Buck Bunny test stream (public)
# Add in dashboard: rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov
```

---

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Upload page |
| `POST` | `/analyze` | Analyze video |
| `GET` | `/results/{id}` | View results |
| `GET` | `/dashboard` | Dashboard |
| `GET` | `/api/cameras` | List cameras |
| `POST` | `/api/cameras` | Add camera |
| `GET` | `/api/stats` | Stats |
| `GET` | `/health` | Health check |

---

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Video In   │────▶│  YOLOv8n     │────▶│  ByteTrack  │
│  (RTSP/MP4) │     │  Detection   │     │  Tracking   │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
                    ┌──────────────┐     ┌──────▼──────┐
                    │  Pose Est.   │◀────│  Person     │
                    │  (Actions)   │     │  Re-ID      │
                    └──────────────┘     └──────┬──────┘
                                                │
                    ┌──────────────┐     ┌──────▼──────┐
                    │  Face Det.   │◀────│  Theft      │
                    │  (Haar)      │     │  Detection  │
                    └──────────────┘     └──────┬──────┘
                                                │
                    ┌──────────────┐     ┌──────▼──────┐
                    │  Zone Counter│◀────│  Supervision│
                    │  (Entry/Exit)│     │  Library    │
                    └──────────────┘     └──────┬──────┘
                                                │
                                        ┌──────▼──────┐
                                        │  Results    │
                                        │  (JSON/DB)  │
                                        └─────────────┘
```

---

## Project Structure

```
rasd/
├── app/
│   ├── api/routes.py              # FastAPI endpoints
│   ├── core/config.py             # Settings (pydantic-settings)
│   ├── core/database.py           # SQLAlchemy engine
│   ├── detectors/
│   │   ├── yolo_detector.py       # YOLOv8 object detection
│   │   ├── pose_detector.py       # YOLOv8-pose behavior analysis
│   │   └── face_detector.py       # Haar cascade face detection
│   │   └── theft_detector.py      # Rule-based theft detection
│   ├── zones/
│   │   └── zone_counter.py        # Entry/exit/shelf zone counting
│   ├── storage/models.py          # SQLAlchemy models
│   ├── video/
│   │   ├── downloader.py          # Video download service
│   │   └── video_processor.py     # Processing pipeline
│   ├── dashboard/                 # Web dashboard
│   ├── templates/                 # Jinja2 HTML templates
│   └── main.py                    # App entry point
├── training/
│   ├── train.py                   # Fine-tuning pipeline
│   ├── generate_data.py           # Synthetic data generation
│   └── config.yaml                # Training configuration
├── requirements.txt
└── .env.example
```

---

## Custom Training

Fine-tune for your use case:

```bash
python training/train.py --epochs 50 --img-size 640
python training/train.py --export onnx
```

---

## Edge Deployment

Rasd is optimized for edge deployment. Export to ONNX and run on low-power devices for real-time processing without cloud dependency.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FRAME_SKIP` | `2` | Process every Nth frame |
| `MAX_WIDTH` | `1280` | Downscale frames wider than this |
| `DATABASE_URL` | `sqlite:///./rasd.db` | Database connection |
| `YOLO_MODEL` | `yolov8n.pt` | YOLO detection model |
| `POSE_MODEL` | `yolov8n-pose.pt` | YOLO pose model |

---

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

1. Fork the repo
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open Pull Request

---

## License

MIT License — see [LICENSE](LICENSE).

---

## Credits

Built by **[Dynamic Web Lab](https://dynamicweblab.com)**

<p align="center">
  <a href="https://dynamicweblab.com">Website</a> ·
  <a href="https://github.com/maidulcu/rasd">GitHub</a>
</p>
