from app.storage.models import AnalysisJob, DetectionSummary, Video


def test_video_model(db_session):
    video = Video(source_url="https://example.com/video.mp4", status="uploaded")
    db_session.add(video)
    db_session.commit()
    db_session.refresh(video)

    assert video.id is not None
    assert video.source_url == "https://example.com/video.mp4"
    assert video.status == "uploaded"
    assert video.created_at is not None


def test_analysis_job_model(db_session):
    video = Video(source_url="https://example.com/video.mp4", status="completed")
    db_session.add(video)
    db_session.commit()
    db_session.refresh(video)

    job = AnalysisJob(video_id=video.id, status="completed", processing_time=12.5)
    db_session.add(job)
    db_session.commit()
    db_session.refresh(job)

    assert job.id is not None
    assert job.video_id == video.id
    assert job.processing_time == 12.5
    assert job.status == "completed"


def test_detection_summary_model(db_session):
    video = Video(source_url="https://example.com/video.mp4", status="completed")
    db_session.add(video)
    db_session.commit()
    db_session.refresh(video)

    job = AnalysisJob(video_id=video.id, status="completed")
    db_session.add(job)
    db_session.commit()
    db_session.refresh(job)

    summary = DetectionSummary(job_id=job.id, human_count=42)
    db_session.add(summary)
    db_session.commit()
    db_session.refresh(summary)

    assert summary.id is not None
    assert summary.job_id == job.id
    assert summary.human_count == 42


def test_video_job_relationship(db_session):
    video = Video(source_url="https://example.com/video.mp4", status="completed")
    db_session.add(video)
    db_session.commit()
    db_session.refresh(video)

    job = AnalysisJob(video_id=video.id, status="completed")
    db_session.add(job)
    db_session.commit()
    db_session.refresh(job)

    assert job in video.jobs
    assert job.video == video


def test_job_summary_relationship(db_session):
    video = Video(source_url="https://example.com/video.mp4", status="completed")
    db_session.add(video)
    db_session.commit()
    db_session.refresh(video)

    job = AnalysisJob(video_id=video.id, status="completed")
    db_session.add(job)
    db_session.commit()
    db_session.refresh(job)

    summary = DetectionSummary(job_id=job.id, human_count=5)
    db_session.add(summary)
    db_session.commit()
    db_session.refresh(summary)

    assert summary.job == job
    assert job.summary == summary
