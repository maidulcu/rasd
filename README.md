# Rasd — AI Video Intelligence Platform

<div align="center">

**AI-powered CCTV & surveillance video analysis for smart cities and retail security.**

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)](https://ultralytics.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-teal)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**[Live Demo](#)** · **[Report Bug](../../issues)** · **[Request Feature](../../issues)**

</div>

---

## What is Rasd?

**Rasd** (رصد — Arabic for "surveillance") is an open-source AI video intelligence platform that analyzes CCTV and surveillance footage in real-time. Built for **smart cities, retail security, and public safety** in the Gulf region and beyond.

### Key Features

- **Person Detection & Counting** — Unique people tracking across frames
- **Face Detection** — Real-time face counting and recognition hooks
- **Theft Detection** — Unattended object alerts and concealment detection
- **Pose Estimation** — Suspicious behavior: hand-in-pocket, bending detection
- **Object Classification** — 80 COCO classes (bags, phones, laptops, weapons)
- **UAE Retail Classes** — Custom model for dallah, perfume, gold boxes, watches
- **Edge Deployment** — Runs on Orange Pi 5, Jetson Orin Nano, Raspberry Pi
- **Export Ready** — ONNX format for production deployment

---

## Demo

| Input | Output |
|---|---|
| Upload CCTV footage | Annotated video with bounding boxes, counts, alerts |

**Results include:**
- Unique people count
- Face count
- Unattended object alerts
- Theft / concealment alerts
- Hand-to-pocket alerts
- Bending alerts

---

## Quick Start

### Prerequisites

- Python 3.12
- PostgreSQL (or SQLite for development)

### Installation

```bash
# Clone the repository
git clone https://github.com/dynamicweblab/rasd.git
cd rasd

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env with your database credentials

# Create database
createdb video_intelligence

# Run migrations
alembic upgrade head

# Start the server
uvicorn app.main:app --reload
```

Open **http://localhost:8000** and upload a video.

---

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web interface |
| `POST` | `/analyze` | Start video analysis |
| `GET` | `/results/{id}` | View results |
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
│   │   └── pose_detector.py       # YOLOv8-pose behavior analysis
│   ├── detectors/theft_detector.py # Rule-based theft detection
│   ├── storage/models.py          # SQLAlchemy models
│   ├── video/
│   │   ├── downloader.py          # Video download service
│   │   └── video_processor.py     # Processing pipeline
│   ├── templates/                 # Jinja2 HTML templates
│   └── main.py                    # App entry point
├── training/
│   ├── train.py                   # Fine-tuning pipeline
│   ├── generate_data.py           # Synthetic data generation
│   └── config.yaml                # Training configuration
├── alembic/                       # Database migrations
├── requirements.txt
└── .env.example
```

---

## Custom Training (UAE Retail)

Fine-tune for Gulf-specific retail items:

```bash
# Generate synthetic training data
python training/generate_data.py

# Train custom model
python training/train.py --epochs 50 --img-size 640

# Export to ONNX
python training/train.py --export onnx
```

**Custom classes:** `dallah`, `perfume_bottle`, `gold_box`, `electronics`, `dates_box`, `wallet`, `watch`

---

## Edge Deployment

Optimized for low-power edge devices:

| Device | FPS | Power | Use Case |
|--------|-----|-------|----------|
| Orange Pi 5 | 8-12 | 5W | Retail stores |
| Jetson Orin Nano | 15-25 | 15W | Smart cities |
| Mac Mini M3 | 20-30 | 6W | Development |
| RTX 3060 | 40-60 | 170W | Production server |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FRAME_SKIP` | `2` | Process every Nth frame |
| `MAX_WIDTH` | `1280` | Downscale frames wider than this |
| `DATABASE_URL` | `sqlite:///./rasd.db` | Database connection |

---

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repo
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open Pull Request

---

## Roadmap

- [ ] Real-time RTSP stream processing
- [ ] Multi-camera support
- [ ] Arabic/English bilingual dashboard
- [ ] Mobile app (React Native)
- [ ] AWS/GCP/Azure deployment guides
- [ ] API rate limiting and auth
- [ ] WebSocket live streaming

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Credits

Built by **[Dynamic Web Lab](https://dynamicweblab.com)** — AI & Web Development

<p align="center">
  <a href="https://dynamicweblab.com">Website</a> ·
  <a href="https://github.com/dynamicweblab">GitHub</a> ·
  <a href="https://linkedin.com/company/dynamicweblab">LinkedIn</a>
</p>

---

<div align="center">

**Made for the Gulf 🌙 — Open Source for Everyone**

</div>
