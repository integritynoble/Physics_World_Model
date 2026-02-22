"""
SQLAlchemy ORM models for PWM Platform.

User model is CompareGPT-compatible (sso_user_id, sso_token, api_key)
with extensions for local login (email, password_hash) and role-based access.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from pwm_platform.db.database import Base


def _utcnow():
    return datetime.now(timezone.utc)


# ═══════════════════════════════════════════════════════════════════════════
#  Auth / Users
# ═══════════════════════════════════════════════════════════════════════════


class User(Base):
    """User account — supports both SSO (CompareGPT) and local login."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), unique=True, nullable=True)
    username = Column(String(100), nullable=False)
    password_hash = Column(String(255), nullable=True)       # bcrypt; NULL for SSO-only

    # SSO fields (CompareGPT compatibility)
    sso_user_id = Column(Integer, unique=True, nullable=True)
    sso_token = Column(String(512), nullable=True)
    api_key = Column(String(255), nullable=True)

    # Access control
    role = Column(String(20), default="user")                # user / admin / reviewer
    is_active = Column(Boolean, default=True)

    created_at = Column(DateTime(timezone=True), default=_utcnow)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    # Relationships
    runs = relationship("Run", back_populates="user", lazy="selectin")

    def __repr__(self):
        return f"<User id={self.id} username={self.username!r} role={self.role!r}>"


# ═══════════════════════════════════════════════════════════════════════════
#  Runs
# ═══════════════════════════════════════════════════════════════════════════


class Run(Base):
    """A PWM pipeline run (simulate / reconstruct / calibrate / etc.)."""

    __tablename__ = "runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(100), unique=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    modality = Column(String(100), nullable=False)
    task_kind = Column(String(50), nullable=False)
    status = Column(String(20), default="pending", index=True)
    compute_mode = Column(String(10), default="auto")

    # Input
    input_mode = Column(String(20))                          # prompt / spec / measured
    experiment_spec = Column(JSONB, default=dict)
    dataset_id = Column(String(255), nullable=True)

    # Execution
    submitted_at = Column(DateTime(timezone=True), default=_utcnow)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    duration_seconds = Column(Float, nullable=True)
    modal_job_id = Column(String(255), nullable=True)
    worker_type = Column(String(50), nullable=True)

    # Outputs
    gcs_bundle_path = Column(String(1024), nullable=True)
    error_message = Column(Text, nullable=True)

    # Provenance
    pwm_version = Column(String(50), default="")
    git_hash = Column(String(50), default="")

    created_at = Column(DateTime(timezone=True), default=_utcnow)

    # Relationships
    user = relationship("User", back_populates="runs")
    triad_report = relationship("TriadReport", back_populates="run", uselist=False)


# ═══════════════════════════════════════════════════════════════════════════
#  Triad Reports
# ═══════════════════════════════════════════════════════════════════════════


class TriadReport(Base):
    """Reconstruction quality / diagnosis report for a run."""

    __tablename__ = "triad_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(100), ForeignKey("runs.run_id"), unique=True)
    modality = Column(String(100))

    # Quality metrics
    psnr = Column(Float, nullable=True)
    ssim = Column(Float, nullable=True)
    lpips = Column(Float, nullable=True)
    sam = Column(Float, nullable=True)
    custom_metrics = Column(JSONB, default=dict)

    # Diagnosis
    diagnosis_severity = Column(String(20), nullable=True)
    diagnosis_codes = Column(JSONB, default=list)
    recommended_actions = Column(JSONB, default=list)

    # Uncertainty
    uncertainty_mean = Column(Float, nullable=True)
    uncertainty_max = Column(Float, nullable=True)
    confidence_interval = Column(JSONB, nullable=True)

    # Operator fidelity
    operator_mismatch_detected = Column(Boolean, default=False)
    mismatch_type = Column(String(100), nullable=True)
    theta_fitted = Column(JSONB, nullable=True)

    # Solver info
    reconstruction_method = Column(String(100), default="")
    solver_iterations = Column(Integer, nullable=True)
    convergence_residual = Column(Float, nullable=True)

    created_at = Column(DateTime(timezone=True), default=_utcnow)

    # Relationships
    run = relationship("Run", back_populates="triad_report")


# ═══════════════════════════════════════════════════════════════════════════
#  Datasets
# ═══════════════════════════════════════════════════════════════════════════


class Dataset(Base):
    """Registered dataset (simulation / real / benchmark)."""

    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(String(255), unique=True, nullable=False, index=True)
    version = Column(String(20), default="1.0.0")
    modality = Column(String(100), nullable=False, index=True)
    data_type = Column(String(20), nullable=False)           # simulation / real / benchmark
    description = Column(Text, default="")
    source = Column(String(255), default="")
    license = Column(String(100), default="internal")
    num_samples = Column(Integer, default=0)
    x_shape = Column(JSONB, nullable=True)
    y_shape = Column(JSONB, nullable=True)
    gcs_prefix = Column(String(1024), default="")
    experiment_spec = Column(JSONB, default=dict)
    calibration_ref = Column(String(1024), default="")
    manifest_ref = Column(String(1024), default="")
    tags = Column(JSONB, default=list)
    is_public = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=_utcnow)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Bootstrap Proposals
# ═══════════════════════════════════════════════════════════════════════════


class BootstrapProposal(Base):
    """Proposal for bootstrapping a new imaging modality."""

    __tablename__ = "bootstrap_proposals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    proposal_id = Column(String(100), unique=True, nullable=False, index=True)
    modality_key = Column(String(100), nullable=False)
    display_name = Column(String(255), default="")
    submitted_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    submitted_at = Column(DateTime(timezone=True), default=_utcnow)
    status = Column(String(30), default="draft", index=True)
    version = Column(Integer, default=1)

    # Modality basics
    physics_class = Column(String(100), default="")
    forward_model_family = Column(String(100), default="")
    sensor_type = Column(String(100), default="")
    source_type = Column(String(100), default="")
    geometry = Column(String(100), default="")
    noise_model = Column(String(50), default="")

    # Generated outputs
    operator_graph_template = Column(JSONB, default=dict)
    experiment_spec_template = Column(JSONB, default=dict)
    simulation_plan = Column(JSONB, default=dict)
    collection_checklist = Column(JSONB, default=list)
    calibration_modes = Column(JSONB, default=list)
    recommended_metrics = Column(JSONB, default=list)
    benchmark_tasks = Column(JSONB, default=list)
    uncertainty_notes = Column(JSONB, default=list)
    viability_checklist = Column(JSONB, default=dict)

    # Similarity
    similar_modalities = Column(JSONB, default=list)

    # Review
    reviewer_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    review_notes = Column(Text, default="")
    review_history = Column(JSONB, default=list)


# ═══════════════════════════════════════════════════════════════════════════
#  Modality Knowledge Base
# ═══════════════════════════════════════════════════════════════════════════


class ModalityBasics(Base):
    """Structured knowledge about an imaging modality (for bootstrap engine)."""

    __tablename__ = "modality_basics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    modality_key = Column(String(100), unique=True, nullable=False, index=True)
    display_name = Column(String(255), default="")
    category = Column(String(100), default="")
    physics_class = Column(String(100), default="")
    forward_model_family = Column(String(100), default="")
    primitive_gates = Column(JSONB, default=list)
    wave_model = Column(String(50), default="")
    sensor_type = Column(String(100), default="")
    source_type = Column(String(100), default="")
    geometry = Column(String(100), default="")
    typical_x_dims = Column(JSONB, nullable=True)
    typical_y_dims = Column(JSONB, nullable=True)
    typical_snr_range = Column(JSONB, nullable=True)
    calibration_params = Column(JSONB, default=list)
    mismatch_modes = Column(JSONB, default=list)
    noise_model = Column(String(50), default="")
    noise_params = Column(JSONB, default=dict)
    reconstruction_task_types = Column(JSONB, default=list)
    default_solver = Column(String(100), default="")
    evaluation_metrics = Column(JSONB, default=list)
    default_experiment_spec = Column(JSONB, default=dict)
    default_operator_graph = Column(JSONB, default=dict)
    canonical_references = Column(JSONB, default=list)
    canonical_datasets = Column(JSONB, default=list)
    feature_vector = Column(JSONB, default=list)
    tags = Column(JSONB, default=list)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)
