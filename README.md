# Rasd — AI Video Analytics

<div align="center">

**AI-powered CCTV & surveillance video analysis for smart cities and retail security.**

[![Python](https://img.shields.io/badge/Python-3.12+-blue)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)](https://ultralytics.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-teal)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**[Docs](#quick-start)** · **[Report Bug](../../issues)** · **[Request Feature](../../issues)**

</div>

---

## What is Rasd?

**Rasd** (رصد — Arabic for "surveillance") is an open-source AI video intelligence platform. Analyze CCTV and surveillance footage in real-time with YOLOv8.

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

## Custom Training

Fine-tune for your use case:

```bash
python training/train.py --epochs 50 --img-size 640
python training/train.py --export onnx
```

---

## Edge Deployment

| Device | FPS | Power | Use Case |
|--------|-----|-------|----------|
| Orange Pi 5 | 8-12 | 5W | Retail stores |
| Jetson Orin Nano | 15-25 | 15W | Smart cities |
| Mac Mini M3 | 20-30 | 6W | Development |

---

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

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
