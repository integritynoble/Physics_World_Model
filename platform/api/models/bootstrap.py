import enum
from datetime import datetime
from sqlalchemy import String, Text, JSON, DateTime, Integer
from sqlalchemy.orm import Mapped, mapped_column
from api.models.base import Base

class BootstrapStatus(str, enum.Enum):
    draft = "draft"
    in_review = "in_review"
    approved = "approved"
    rejected = "rejected"

class BootstrapProposal(Base):
    __tablename__ = "bootstrap_proposals"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="draft")
    name: Mapped[str] = mapped_column(String(128))
    description: Mapped[str] = mapped_column(Text)
    physics_class: Mapped[str | None] = mapped_column(String(64), nullable=True)
    sensor_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    geometry: Mapped[str | None] = mapped_column(String(64), nullable=True)
    similar_modalities: Mapped[list] = mapped_column(JSON, default=list)
    operator_graph_template: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    experiment_spec_template: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    sim_dataset_plan: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    real_data_checklist: Mapped[list] = mapped_column(JSON, default=list)
    calibration_modes: Mapped[list] = mapped_column(JSON, default=list)
    benchmark_metrics: Mapped[list] = mapped_column(JSON, default=list)
    uncertainty_notes: Mapped[list] = mapped_column(JSON, default=list)
    viability_checklist: Mapped[list] = mapped_column(JSON, default=list)
    reviewer_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    local_proposal_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
