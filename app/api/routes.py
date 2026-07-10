import logging
import os
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from app.core.config import settings
from app.core.database import SessionLocal
from app.storage.models import AnalysisJob, DetectionSummary, Video
from app.video.downloader import VideoDownloader
from app.video.video_processor import VideoProcessor

logger = logging.getLogger(__name__)

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

downloader = VideoDownloader()
processor = VideoProcessor()


@router.get("/")
def home(request: Request):
    return templates.TemplateResponse(request, "home.html")


@router.post("/analyze")
def analyze(request: Request, video_url: str = Form(...)):
    db = SessionLocal()
    video = None
    job = None
    try:
        video = Video(source_url=video_url, status="uploaded")
        db.add(video)
        db.commit()
        db.refresh(video)

        local_path = downloader.download(video_url)

        video.local_path = local_path
        video.status = "processing"
        db.commit()

        job = AnalysisJob(
            video_id=video.id,
            started_at=datetime.now(timezone.utc),
            status="running",
        )
        db.add(job)
        db.commit()
        db.refresh(job)

        result = processor.process(local_path)

        job.completed_at = datetime.now(timezone.utc)
        job.processing_time = result["processing_time"]
        job.status = "completed"

        summary = DetectionSummary(job_id=job.id, human_count=result["human_count"])
        db.add(summary)

        video.status = "completed"
        db.commit()

        logger.info(
            "Job %d completed: %d humans detected", job.id, result["human_count"]
        )

        return RedirectResponse(url=f"/results/{job.id}", status_code=303)

    except Exception as e:
        logger.error("Analysis failed: %s", e)
        if video is not None:
            video.status = "failed"
            db.commit()
        if job is not None:
            job.status = "failed"
            job.completed_at = datetime.now(timezone.utc)
            db.commit()
        return templates.TemplateResponse(
            request,
            "home.html",
            context={"error": str(e)},
            status_code=400,
        )
    finally:
        db.close()


@router.get("/results/{job_id}")
def results(request: Request, job_id: int):
    db = SessionLocal()
    try:
        job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
        if not job:
            return templates.TemplateResponse(
                request,
                "home.html",
                context={"error": "Job not found"},
                status_code=404,
            )

        video = db.query(Video).filter(Video.id == job.video_id).first()
        summary = db.query(DetectionSummary).filter(DetectionSummary.job_id == job.id).first()

        output_file = None
        output_filename = None
        if video and video.local_path:
            output_filename = f"annotated_{os.path.basename(video.local_path)}"
            output_path = os.path.join(settings.OUTPUT_DIR, output_filename)
            if os.path.exists(output_path):
                output_file = output_path

        return templates.TemplateResponse(
            request,
            "results.html",
            context={
                "video_url": video.source_url if video else "",
                "human_count": summary.human_count if summary else 0,
                "processing_time": job.processing_time or 0,
                "output_file": output_file,
                "output_filename": output_filename,
            },
        )
    finally:
        db.close()


@router.get("/output/{filename}")
def serve_output(filename: str):
    from fastapi.responses import FileResponse

    file_path = os.path.join(settings.OUTPUT_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="video/mp4")
    return {"error": "File not found"}


@router.get("/health")
def health():
    return {"status": "ok"}
