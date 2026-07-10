import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class Video(Base):
    __tablename__ = "videos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_url: Mapped[str] = mapped_column(String, nullable=False)
    local_path: Mapped[str] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, default="uploaded")
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    jobs: Mapped[list["AnalysisJob"]] = relationship(back_populates="video")


class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[int] = mapped_column(Integer, ForeignKey("videos.id"))
    started_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    processing_time: Mapped[float] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String, default="pending")

    video: Mapped["Video"] = relationship(back_populates="jobs")
    summary: Mapped["DetectionSummary | None"] = relationship(
        back_populates="job", uselist=False
    )


class DetectionSummary(Base):
    __tablename__ = "detection_summaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[int] = mapped_column(Integer, ForeignKey("analysis_jobs.id"))
    human_count: Mapped[int] = mapped_column(Integer, default=0)

    job: Mapped["AnalysisJob"] = relationship(back_populates="summary")
