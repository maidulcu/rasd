# Video Analyzer Starter

Person detection for video files using YOLO on CPU.

## Prerequisites

- Python 3.12
- PostgreSQL

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create a `.env` file from the example:

```bash
cp .env.example .env
```

Edit `.env` with your PostgreSQL credentials.

3. Create the database:

```bash
createdb video_intelligence
```

4. Run migrations:

```bash
alembic upgrade head
```

5. Start the server:

```bash
uvicorn app.main:app --reload
```

6. Open http://localhost:8000

## Usage

1. Enter a video URL (direct MP4 link)
2. Click **Analyze**
3. View results: annotated video with bounding boxes, human detection count, and processing time

## Performance Optimizations

Optimized for CPU-only inference on older hardware:

| Variable      | Default | Description                        |
|---------------|---------|------------------------------------|
| `FRAME_SKIP`  | `2`     | Process every Nth frame (1 = all)  |
| `MAX_WIDTH`   | `1280`  | Downscale frames wider than this   |

Set via `.env` file.

## API Endpoints

| Method | Path            | Description                |
|--------|-----------------|----------------------------|
| GET    | /               | Home page                  |
| POST   | /analyze        | Start analysis             |
| GET    | /results/{id}   | View analysis results      |
| GET    | /health         | Health check               |

## Project Structure

```
app/
  api/routes.py          - FastAPI endpoints
  core/config.py         - Pydantic settings
  core/database.py       - SQLAlchemy engine/session
  detectors/yolo_detector.py  - YOLO person detection (CPU only)
  storage/models.py      - SQLAlchemy models
  video/downloader.py    - Video download service
  video/video_processor.py - Video processing pipeline (with frame skip + resize)
  templates/             - Jinja2 HTML templates
  main.py                - Application entry point
alembic/                 - Database migrations
```
