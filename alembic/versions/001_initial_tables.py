"""initial tables

Revision ID: 001
Revises: 
Create Date: 2026-07-09

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "videos",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("source_url", sa.String(), nullable=False),
        sa.Column("local_path", sa.String(), nullable=True),
        sa.Column("status", sa.String(), server_default="uploaded", nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "analysis_jobs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("video_id", sa.Integer(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("processing_time", sa.Float(), nullable=True),
        sa.Column("status", sa.String(), server_default="pending", nullable=True),
        sa.ForeignKeyConstraint(["video_id"], ["videos.id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "detection_summaries",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("job_id", sa.Integer(), nullable=False),
        sa.Column("human_count", sa.Integer(), server_default="0", nullable=True),
        sa.Column("face_count", sa.Integer(), server_default="0", nullable=True),
        sa.Column("unattended_count", sa.Integer(), server_default="0", nullable=True),
        sa.Column("theft_alert_count", sa.Integer(), server_default="0", nullable=True),
        sa.Column("hand_to_pocket_count", sa.Integer(), server_default="0", nullable=True),
        sa.Column("bending_count", sa.Integer(), server_default="0", nullable=True),
        sa.ForeignKeyConstraint(["job_id"], ["analysis_jobs.id"]),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("detection_summaries")
    op.drop_table("analysis_jobs")
    op.drop_table("videos")
