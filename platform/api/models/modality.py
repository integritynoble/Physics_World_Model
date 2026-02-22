from datetime import datetime
from sqlalchemy import String, Text, JSON, DateTime, Boolean
from sqlalchemy.orm import Mapped, mapped_column
from api.models.base import Base

class ModalityRecord(Base):
    __tablename__ = "modalities"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    physics_class: Mapped[str] = mapped_column(String(64), default="unknown")
    forward_model_family: Mapped[str] = mapped_column(String(64), default="unknown")
    sensor_type: Mapped[str] = mapped_column(String(64), default="unknown")
    geometry: Mapped[str] = mapped_column(String(64), default="unknown")
    task_types: Mapped[list] = mapped_column(JSON, default=list)
    noise_models: Mapped[list] = mapped_column(JSON, default=list)
    primitives: Mapped[list] = mapped_column(JSON, default=list)
    graph_template_ids: Mapped[list] = mapped_column(JSON, default=list)
    default_metrics: Mapped[list] = mapped_column(JSON, default=list)
    calibration_params: Mapped[dict] = mapped_column(JSON, default=dict)
    embedding: Mapped[list | None] = mapped_column(JSON, nullable=True)
    is_supported: Mapped[bool] = mapped_column(Boolean, default=True)
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
