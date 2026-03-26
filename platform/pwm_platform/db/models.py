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

    # Credits (ComparGPT-style flex credits)
    credit_balance = Column(Float, default=100.0)            # initial free credits

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
    is_public = Column(Boolean, default=True, server_default="true", index=True)

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


# ═══════════════════════════════════════════════════════════════════════════
#  Challenge Submissions
# ═══════════════════════════════════════════════════════════════════════════


class ChallengeSubmission(Base):
    """Community submission for a blind reconstruction challenge tier."""

    __tablename__ = "challenge_submissions"

    id                 = Column(Integer, primary_key=True, autoincrement=True)
    submission_id      = Column(String(100), unique=True, nullable=False, index=True)
    submission_type    = Column(String(30), nullable=False)       # reconstruction / algorithm / dataset
    category           = Column(String(30), default="competition")  # competition / contribution
    variant_key        = Column(String(100), nullable=False, index=True)
    tier_name          = Column(String(20), nullable=False)       # public / dev / hidden
    credit_cost        = Column(Float, default=0.0)               # credits charged (hidden tier)
    submitted_by       = Column(Integer, ForeignKey("users.id"), nullable=False)
    submitted_at       = Column(DateTime(timezone=True), default=_utcnow)
    method_name        = Column(String(255), nullable=False)
    method_description = Column(Text, default="")
    paper_url          = Column(String(1024), default="")
    code_url           = Column(String(1024), default="")
    file_path          = Column(String(1024), nullable=False)
    original_filename  = Column(String(255), nullable=False)
    file_size_bytes    = Column(Integer, default=0)
    gcs_path           = Column(String(1024), default="")
    corrected_spec     = Column(JSONB, nullable=True)
    status             = Column(String(30), default="pending", index=True)  # pending/approved/rejected
    reviewer_id        = Column(Integer, ForeignKey("users.id"), nullable=True)
    review_notes       = Column(Text, default="")
    reviewed_at        = Column(DateTime(timezone=True), nullable=True)
    scores             = Column(JSONB, nullable=True)
    # Trust ratchet (Dyson Swarm P0)
    trust_tier         = Column(String(30), default="draft", nullable=False, index=True)
    gate_verdicts      = Column(JSONB, nullable=True)   # {s1: {verdict, message}, ...}
    created_at         = Column(DateTime(timezone=True), default=_utcnow)
    updated_at         = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    submitter = relationship("User", foreign_keys=[submitted_by], lazy="selectin")
    reviewer  = relationship("User", foreign_keys=[reviewer_id], lazy="selectin")


class SpecChatSession(Base):
    """Persistent multi-turn spec builder chat session."""

    __tablename__ = "spec_chat_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), unique=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    variant_key = Column(String(100), default="sd_cassi")
    history = Column(JSONB, default=list)     # [{role, content}, ...]
    created_at = Column(DateTime(timezone=True), default=_utcnow)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    # Dataset mode — uploaded measurement/matrix/ground-truth metadata
    dataset_meta = Column(JSONB, nullable=True)
    matrix_meta = Column(JSONB, nullable=True)
    ground_truth_meta = Column(JSONB, nullable=True)

    # Relationships
    user = relationship("User")


# ═══════════════════════════════════════════════════════════════════════════
#  Billing / Subscriptions / Credits
# ═══════════════════════════════════════════════════════════════════════════


class CreditAccount(Base):
    """
    User's credit account — tracks subscription tier, run/report credits,
    overage balances, and expiry (for WeChat credit packs).
    Adapted from CompareGPT's PaymentPointsManagement model.
    """

    __tablename__ = "credit_accounts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False, index=True)

    # Subscription state
    plan_tier = Column(String(30), default="free", nullable=False)  # free/researcher/pro/team/enterprise
    payment_status = Column(String(30), default="free", index=True)  # free/active/cancelled/expired
    subscription_id = Column(String(255), nullable=True)  # Stripe subscription ID

    # Monthly plan credits (reset on renewal)
    run_credits = Column(Integer, default=3, nullable=False)
    report_credits = Column(Integer, default=0, nullable=False)

    # Overage / add-on credits (purchased separately, never reset)
    overage_run_credits = Column(Integer, default=0, nullable=False)
    overage_report_credits = Column(Integer, default=0, nullable=False)

    # For WeChat credit packs: credits expire after validity period
    credits_expire_at = Column(DateTime(timezone=True), nullable=True)

    # Legacy compatibility — maps to User.credit_balance for hidden tier
    legacy_credit_balance = Column(Float, default=100.0)

    created_at = Column(DateTime(timezone=True), default=_utcnow)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    # Relationships
    user = relationship("User", backref="credit_account", uselist=False)


class CreditTransaction(Base):
    """
    Immutable ledger of credit changes — consumption, provisioning, purchases.
    Adapted from CompareGPT's PaymentConsumption model.
    """

    __tablename__ = "credit_transactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String(64), unique=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    transaction_type = Column(String(30), nullable=False)   # consume / provision / purchase / refund
    credit_kind = Column(String(20), nullable=False)         # run / report / legacy / plan
    amount = Column(Float, nullable=False)                   # positive=add, negative=deduct
    description = Column(String(500), default="")

    # Snapshot of remaining credits after this transaction
    remaining_run_credits = Column(Integer, default=0)
    remaining_report_credits = Column(Integer, default=0)

    created_at = Column(DateTime(timezone=True), default=_utcnow)


class Subscription(Base):
    """
    Subscription record — tracks Stripe or WeChat payment lifecycle.
    """

    __tablename__ = "subscriptions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    plan_tier = Column(String(30), nullable=False)
    billing_period = Column(String(20), nullable=False)  # monthly / yearly / one_time
    payment_method = Column(String(30), nullable=False)  # stripe / wechat / manual

    # Stripe-specific
    stripe_subscription_id = Column(String(255), nullable=True)
    stripe_customer_id = Column(String(255), nullable=True)

    # State
    status = Column(String(30), default="pending", index=True)  # pending/active/cancelled/expired
    started_at = Column(DateTime(timezone=True), nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    cancelled_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime(timezone=True), default=_utcnow)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)


class PaymentOrder(Base):
    """
    Payment order — created before redirect to Stripe/WeChat, completed on webhook.
    Adapted from CompareGPT's PaymentOrder model.
    """

    __tablename__ = "payment_orders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(String(64), unique=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    order_type = Column(String(30), nullable=False)   # subscription / credit_pack / overage_run / overage_report
    plan_tier = Column(String(30), default="free")
    pack_key = Column(String(50), nullable=True)       # WeChat credit pack key
    amount_usd = Column(Float, default=0.0)
    amount_cny = Column(Float, default=0.0)
    credit_amount = Column(Integer, nullable=True)     # for overage purchases

    payment_method = Column(String(30), nullable=False)  # stripe / wechat
    payment_ref = Column(String(255), nullable=True)     # Stripe session ID or WeChat order ID
    status = Column(String(30), default="pending", index=True)  # pending / completed / failed / refunded

    created_at = Column(DateTime(timezone=True), default=_utcnow)
    completed_at = Column(DateTime(timezone=True), nullable=True)


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


# ═══════════════════════════════════════════════════════════════════════════
#  Auth — Password Reset
# ═══════════════════════════════════════════════════════════════════════════


class PasswordResetToken(Base):
    """One-time password reset tokens (expire in 1 hour)."""

    __tablename__ = "password_reset_tokens"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    token = Column(String(64), unique=True, nullable=False, index=True)
    used = Column(Boolean, default=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), default=_utcnow)

    user = relationship("User")


# ═══════════════════════════════════════════════════════════════════════════
#  Contributor Economy
# ═══════════════════════════════════════════════════════════════════════════


class ContributorProfile(Base):
    """Public contributor profile — tracks contribution stats and badge tier."""

    __tablename__ = "contributor_profiles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)

    # Contribution stats
    modalities_contributed = Column(JSONB, default=list)   # list of modality keys
    solver_count = Column(Integer, default=0)
    dataset_count = Column(Integer, default=0)
    claim_count = Column(Integer, default=0)
    verified_claim_count = Column(Integer, default=0)

    # Badge tier: none / bronze / silver / gold / platinum
    badge_tier = Column(String(20), default="none", nullable=False)

    # Role assignment and badge tracking (Dyson Swarm contributor economy)
    roles = Column(JSONB, default=list)                    # ["modality_maintainer", "benchmark_reviewer", ...]
    badges = Column(JSONB, default=list)                   # [{"badge": "first_certified", "earned_at": "..."}, ...]
    maintained_modalities = Column(JSONB, default=list)    # ["ct", "mri"] — for modality maintainers
    contribution_history = Column(JSONB, default=list)     # [{"action": "approved_claim", "timestamp": "..."}]
    total_reproductions = Column(Integer, default=0)
    total_certifications = Column(Integer, default=0)
    total_claims_reviewed = Column(Integer, default=0)

    # Public profile
    bio = Column(Text, default="")
    orcid = Column(String(50), nullable=True)
    github_handle = Column(String(100), nullable=True)
    website_url = Column(String(512), nullable=True)

    created_at = Column(DateTime(timezone=True), default=_utcnow)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    user = relationship("User", backref="contributor_profile", uselist=False)


# ═══════════════════════════════════════════════════════════════════════════
#  Instrument Registry
# ═══════════════════════════════════════════════════════════════════════════


class Instrument(Base):
    """Registered instrument / InstrumentCard for calibration tracking."""

    __tablename__ = "instruments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    instrument_id = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    modality = Column(String(100), nullable=False, index=True)
    lab = Column(String(255), default="")
    institution = Column(String(255), default="")
    manufacturer = Column(String(255), default="")
    model_number = Column(String(255), default="")
    serial_number = Column(String(255), default="")
    description = Column(Text, default="")
    calibration_date = Column(DateTime(timezone=True), nullable=True)
    drift_budget_pct = Column(Float, default=0.0)   # allowed drift in %
    contact_email = Column(String(255), default="")
    card_data = Column(JSONB, default=dict)           # full InstrumentCard JSON
    uploaded_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    is_public = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    uploader = relationship("User", foreign_keys=[uploaded_by], lazy="selectin")
