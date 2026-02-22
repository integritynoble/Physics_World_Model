from datetime import datetime
from sqlalchemy import String, Text, JSON, DateTime, Boolean, BigInteger
from sqlalchemy.orm import Mapped, mapped_column
from api.models.base import Base

class DatasetRecord(Base):
    __tablename__ = "datasets"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    version: Mapped[str] = mapped_column(String(32), default="1.0.0")
    modality: Mapped[str] = mapped_column(String(64))
    kind: Mapped[str] = mapped_column(String(32))
    local_path: Mapped[str] = mapped_column(String(512))
    manifest_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    num_samples: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    sha256_manifest: Mapped[str | None] = mapped_column(String(64), nullable=True)
    is_public: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    tags: Mapped[dict] = mapped_column(JSON, default=dict)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
