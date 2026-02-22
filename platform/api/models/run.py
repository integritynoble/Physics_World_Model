import enum
from datetime import datetime
from sqlalchemy import String, Enum, DateTime, Text, JSON, Float
from sqlalchemy.orm import Mapped, mapped_column
from api.models.base import Base

class RunStatus(str, enum.Enum):
    queued = "queued"
    running = "running"
    done = "done"
    failed = "failed"

class RunRecord(Base):
    __tablename__ = "runs"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="queued")
    compute_mode: Mapped[str] = mapped_column(String(16), default="auto")
    prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    spec_json: Mapped[dict] = mapped_column(JSON, default=dict)
    modality: Mapped[str | None] = mapped_column(String(64), nullable=True)
    celery_task_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    local_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    diagnosis_verdict: Mapped[str | None] = mapped_column(String(64), nullable=True)
    diagnosis_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    modal_cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
