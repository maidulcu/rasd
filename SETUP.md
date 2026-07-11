# Video Intelligence Platform — Complete Setup Guide

## Prerequisites

- **Python 3.10+** (tested with 3.12)
- **pip** (Python package manager)
- **Git**
- **FFmpeg** (for video processing)
- **~2GB free disk** (models + dependencies)
- **macOS/Linux** (Windows untested but should work)

### Install FFmpeg

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows (Chocolatey)
choco install ffmpeg
```

---

## 1. Clone & Setup

```bash
# Clone the repository
git clone <your-repo-url> video-analyzer
cd video-analyzer

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### Requirements file

Create `requirements.txt`:

```txt
fastapi==0.115.6
uvicorn[standard]==0.34.0
sqlalchemy==2.0.36
pydantic-settings==2.7.0
opencv-python-headless==4.10.0.84
numpy==1.26.4
ultralytics==8.3.62
scipy==1.12.0
httpx
pytest
jinja2
aiosqlite
```

> **Note:** NumPy is pinned to 1.26.4 for torch/ultralytics compatibility.
> SciPy is pinned to <1.13 to avoid numpy ABI conflicts.

---

## 2. Configuration

All settings live in `app/core/config.py`. Set via environment variables or `.env` file:

```env
# .env
DATABASE_URL=sqlite:///./video_intelligence.db
DOWNLOAD_DIR=downloads
OUTPUT_DIR=output
YOLO_MODEL=yolov8n.pt
CONFIDENCE_THRESHOLD=0.5
FRAME_SKIP=2
MAX_WIDTH=1280
```

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./video_intelligence.db` | Database connection string |
| `DOWNLOAD_DIR` | `downloads` | Where downloaded/uploaded videos are stored |
| `OUTPUT_DIR` | `output` | Where annotated output videos are saved |
| `YOLO_MODEL` | `yolov8n.pt` | YOLO model for detection (auto-downloads if missing) |
| `POSE_MODEL` | `yolov8n-pose.pt` | Pose estimation model |
| `USE_ONNX` | `false` | Use ONNX Runtime (set `true` for ~5% faster inference) |
| `CONFIDENCE_THRESHOLD` | `0.5` | Minimum confidence for detections |
| `FRAME_SKIP` | `2` | Process every (N+1)th frame (2 = every 3rd frame) |
| `MAX_WIDTH` | `1280` | Scale videos wider than this down to this width |

---

## 3. Run the Server

```bash
# Activate venv (if not already)
source .venv/bin/activate

# Start the server
uvicorn app.main:app --port 8000 --host 0.0.0.0
```

The server will:

1. Auto-download **yolov8n.pt** (~6MB) on first YOLO call
2. Auto-download **yolov8n-pose.pt** (~6MB) on first pose call
3. Create the SQLite database file (`video_intelligence.db`) on first request
4. Create `downloads/` and `output/` directories as needed

Open **http://localhost:8000** in your browser.

---

## 4. Usage

### Upload a Video

1. Visit **http://localhost:8000**
2. Either:
   - Paste a video URL (MP4, MOV, AVI, MKV) and click **Analyze URL**
   - Upload a local video file and click **Upload & Analyze**
3. Wait for processing (shows progress every 100 frames in the server log)
4. Results page displays:
   - **Unique People Detected** — total distinct people (ByteTrack IDs)
   - **Faces Detected** — total face detections (Haar cascade)
   - **Unattended Objects** — valuable items left alone >30 frames
   - **Theft (Concealed Items)** — items that disappeared while near a person
   - **Hand-to-Pocket Alerts** — wrist near hip (potential concealment via pose)
   - **Bending Alerts** — person bent over (hiding items via pose)
5. Play the annotated video with all detections overlaid

### Server Logs

```bash
# Tail the log
tail -f /tmp/video-analyzer.log
```

Sample output:
```
INFO: Analysis started for: downloads/abc123.mp4
INFO: Video info: 1280x720 @ 30.0 fps, 3510 frames (output: 1280x720, skip: 2)
INFO: Processed 100 frames...
INFO: Processed 200 frames...
INFO: Analysis completed: 48 people, 12 faces, 2 unattended, 1 theft alerts in 117.32s
```

---

## 5. Detectors (How It Works)

### YOLO + ByteTrack (`app/detectors/yolo_detector.py`)
- Uses **YOLOv8n** (nano) for object detection (80 COCO classes)
- Uses **ByteTrack** (`model.track(persist=True)`) for person tracking across frames
- Every `FRAME_SKIP+1` frames: tracking + full detection
- Intermediate frames: detection only (no tracking, no IDs)

### Face Detection (`app/detectors/face_detector.py`)
- OpenCV **Haar Cascade** (`haarcascade_frontalface_default.xml`)
- Fast CPU detection, cyan bounding boxes
- Counts total faces across all frames

### Theft Detection (`app/detectors/theft_detector.py`)
- Tracks **valuable objects**: backpack, handbag, suitcase, cell phone, laptop, bottle, book, knife, scissors, etc.
- **Unattended detection**: object not near any person for 30+ frames → red "UNATTENDED"
- **Concealment detection**: person overlaps valuable object, then object disappears → red "THEFT: ... concealed!"
- **Interaction display**: person currently holding/picking up an object → orange "PICKED UP"

### Pose Estimation (`app/detectors/pose_detector.py`)
- Uses **YOLOv8n-pose** for 17-keypoint human pose estimation
- **Skeleton overlay**: colored lines between connected joints, yellow keypoint dots
- **Hand-to-pocket**: wrist Y near same-side hip Y → red "HAND IN POCKET (L/R)" banner
- **Bending**: shoulder-to-hip distance < 60% of total height → orange "BENDING" banner
- Runs on every tracked frame alongside other detectors

### Object Interaction Markers

| Marker | Color | Meaning |
|---|---|---|
| Person #N | Varies (rainbow) | Unique tracked person |
| Face | Cyan | Face detected |
| PICKED UP | Orange | Person holding/overlapping a valuable item |
| UNATTENDED | Red | Valuable item left alone >30 frames |
| THEFT: concealed! | Dark Red | Item disappeared while near a person |
| HAND IN POCKET | Red banner (on skeleton) | Wrist at pocket level |
| BENDING | Orange banner (on skeleton) | Person bent over |

---

## 6. API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Home page (upload form) |
| POST | `/analyze` | Submit video URL or file for analysis |
| GET | `/results/{job_id}` | View analysis results |
| GET | `/output/{filename}` | Serve annotated video file |
| GET | `/health` | Health check → `{"status":"ok"}` |

### POST /analyze

Form parameters:
- `video_url` — URL to a video file (MP4, MOV, AVI, MKV)
- `video_file` — Upload a video file directly

Returns: Redirect to `/results/{job_id}`

---

## 7. Database

SQLite file: `video_intelligence.db`

### Tables

**videos**
| Column | Type | Description |
|---|---|---|
| id | INTEGER | Primary key |
| source_url | TEXT | Original URL or "(uploaded) filename.mp4" |
| local_path | TEXT | Local file path |
| status | TEXT | uploaded / processing / completed / failed |
| created_at | DATETIME | Upload timestamp |

**analysis_jobs**
| Column | Type | Description |
|---|---|---|
| id | INTEGER | Primary key |
| video_id | INTEGER | FK → videos.id |
| started_at | DATETIME | Analysis start time |
| completed_at | DATETIME | Analysis end time |
| processing_time | FLOAT | Duration in seconds |
| status | TEXT | pending / running / completed / failed |

**detection_summaries**
| Column | Type | Description |
|---|---|---|
| id | INTEGER | Primary key |
| job_id | INTEGER | FK → analysis_jobs.id |
| human_count | INTEGER | Unique people detected |
| face_count | INTEGER | Total face detections |
| unattended_count | INTEGER | Unattended object alerts |
| theft_alert_count | INTEGER | Concealed item alerts |
| hand_to_pocket_count | INTEGER | Hand-to-pocket pose alerts |
| bending_count | INTEGER | Bending pose alerts |

Reset the database:
```bash
rm video_intelligence.db
```

---

## 8. Testing

```bash
# Run all tests (from project root)
source .venv/bin/activate
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_api.py -v

# Run with coverage
pip install pytest-cov
python -m pytest tests/ --cov=app
```

### Test structure

| File | What it tests |
|---|---|
| `tests/test_api.py` | HTTP endpoints, status codes, HTML content |
| `tests/test_downloader.py` | URL validation (invalid, ftp, valid http/https) |
| `tests/test_models.py` | DB models, relationships, defaults |
| `tests/conftest.py` | Test fixtures (TestClient, test DB) |

---

## 9. Project Structure

```
video-analyzer-starter/
├── .env                   # Environment variables (optional)
├── .gitignore
├── requirements.txt
├── SETUP.md               # This file
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI app creation
│   ├── api/
│   │   └── routes.py      # All HTTP endpoints
│   ├── core/
│   │   ├── config.py      # Settings via pydantic-settings
│   │   └── database.py    # SQLAlchemy engine + session
│   ├── detectors/
│   │   ├── face_detector.py    # Haar cascade face detection
│   │   ├── pose_detector.py    # YOLOv8-pose + behavior classification
│   │   ├── theft_detector.py   # Object tracking + unattended/concealment
│   │   └── yolo_detector.py    # YOLO detection + ByteTrack tracking
│   ├── storage/
│   │   └── models.py      # SQLAlchemy ORM models
│   ├── templates/
│   │   ├── base.html       # Bootstrap layout
│   │   ├── home.html       # Upload form
│   │   └── results.html    # Results + annotated video
│   └── video/
│       ├── downloader.py   # Video URL downloader
│       └── video_processor.py  # Main processing pipeline
├── docs/
│   └── ROADMAP.md          # Full product roadmap
├── downloads/              # Uploaded/downloaded videos (gitignored)
├── output/                 # Annotated output videos (gitignored)
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_api.py
    ├── test_downloader.py
    └── test_models.py
```

---

## 10. Performance Notes

| Setting | Effect | Recommended |
|---|---|---|
| `FRAME_SKIP=2` | Processes every 3rd frame | Good for 30fps video (10 effective fps) |
| `FRAME_SKIP=0` | Processes every frame | Slow but precise |
| `MAX_WIDTH=640` | Scales video down | Faster but lower resolution output |
| `CONFIDENCE_THRESHOLD=0.5` | Ignores low-confidence detections | Balanced |

### Typical processing time (CPU only)

| Video length | Resolution | FRAME_SKIP | Time |
|---|---|---|---|
| 30 seconds | 1280x720 | 2 | ~35s |
| 60 seconds | 1280x720 | 2 | ~70s |
| 2 minutes | 1920x1080 | 2 | ~4min |

### YOLO model size vs speed

| Model | Size | Speed | Accuracy |
|---|---|---|---|
| `yolov8n.pt` (nano) | 6MB | Fastest | Lowest |
| `yolov8s.pt` (small) | 22MB | Fast | Medium |
| `yolov8m.pt` (medium) | 52MB | Medium | High |

Change in `app/core/config.py` or `.env`:
```env
YOLO_MODEL=yolov8s.pt
```

---

## 11. Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### OpenCV crashes or shows GUI windows

The code uses `cv2.VideoWriter` and `cv2.imread` only — no GUI calls.
If you see windows popping up, ensure no `cv2.imshow()` calls remain.

### "Cannot open video file"

- Verify the file exists: `ls -lh downloads/`
- Check the file is not corrupted: `ffprobe downloads/your-file.mp4`
- Ensure FFmpeg is installed: `ffmpeg -version`

### "avc1" codec not found

The output uses `avc1` (H.264) for browser compatibility.
If unavailable, change to `mp4v` in `video_processor.py`:
```python
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
```

### Pose model not downloading

If `yolov8n-pose.pt` fails to download:
```bash
# Manual download
wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n-pose.pt
# Or
curl -L -o yolov8n-pose.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n-pose.pt
```

### Port 8000 already in use

```bash
# Find what's using port 8000
lsof -i :8000

# Kill it
kill -9 <PID>

# Or use a different port
uvicorn app.main:app --port 8001
```

### Database migration errors

If you add columns to the SQLAlchemy models, delete the old DB:
```bash
rm video_intelligence.db
```

---

## 12. Quick Start (TL;DR)

```bash
# One-time setup
git clone <repo-url> video-analyzer
cd video-analyzer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run
uvicorn app.main:app --port 8000 --host 0.0.0.0

# Open
open http://localhost:8000
```
