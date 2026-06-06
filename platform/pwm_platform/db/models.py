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
    LargeBinary,
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

    # SIWE / Web3 identity (lowercase 0x… EIP-55 not enforced at DB layer).
    # Unique when present; nullable so traditional email accounts can exist.
    wallet_address = Column(String(42), unique=True, nullable=True, index=True)

    # How this account was created. Values: "local", "sso", "siwe", "google"
    auth_method = Column(String(20), default="local", nullable=False, server_default="local")

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


class SpecChatSession(Base):
    """Persistent multi-turn spec builder chat session."""

    __tablename__ = "spec_chat_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), unique=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    variant_key = Column(String(100), default="cassi")
    history = Column(JSONB, default=list)     # [{role, content}, ...]
    created_at = Column(DateTime(timezone=True), default=_utcnow)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    # Dataset mode — uploaded measurement/matrix/ground-truth metadata
    dataset_meta = Column(JSONB, nullable=True)
    matrix_meta = Column(JSONB, nullable=True)
    ground_truth_meta = Column(JSONB, nullable=True)

    # Spec.md — generated spec file and usage type
    spec_md = Column(Text, nullable=True)
    usage_type = Column(String(20), nullable=True)  # reconstruct / mismatch / design / simulate

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
#  PWM Token (protocol reward token)
# ═══════════════════════════════════════════════════════════════════════════


class PWMTokenAccount(Base):
    """Off-chain PWM token balance for each user.

    Tokens are awarded when a contributor's submission is promoted from
    testnet (any trust_tier) to mainnet (trust_tier == 'certified') by a founder.
    """

    __tablename__ = "pwm_token_accounts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False, index=True)
    balance = Column(Float, default=0.0, nullable=False)
    lifetime_earned = Column(Float, default=0.0, nullable=False)
    on_chain_address = Column(String(255), nullable=True)
    # Fernet-encrypted private key for platform-managed wallets.
    # Null for users who supplied their own external wallet address.
    wallet_private_key_enc = Column(String(512), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    user = relationship("User", backref="pwm_token_account", uselist=False)


class PWMTokenTransaction(Base):
    """Immutable ledger of PWM token movements.

    transaction_type values:
      - 'award_promotion'  : auto-awarded when submission promoted to certified
      - 'award_manual'     : founder/admin manual award
      - 'spend'            : user-initiated spend (future)
      - 'adjust'           : admin balance adjustment (refund / clawback)
    """

    __tablename__ = "pwm_token_transactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String(64), unique=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    amount = Column(Float, nullable=False)
    transaction_type = Column(String(30), nullable=False, index=True)
    description = Column(String(500), default="")
    submission_id = Column(String(100), nullable=True, index=True)
    artifact_type = Column(String(30), nullable=True)
    awarded_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    balance_after = Column(Float, nullable=False)
    # ── 90/10 split (record-only; M6 settles these credits on-chain) ──────
    # On a `spend` row these capture the intended on-chain distribution of the
    # spent amount: provider_amount (provider_split_bps of the spend) to
    # provider_wallet, pool_amount (the remainder) to the mining pool. They are
    # NOT off-chain credits (the ledger is user-keyed and has no pool account) —
    # the promotion/settlement relayer reads them to credit provider + pool on
    # Base. NULL on non-spend rows (awards / adjustments).
    provider_wallet = Column(String(64), nullable=True)
    provider_amount = Column(Float, nullable=True)
    pool_amount = Column(Float, nullable=True)
    provider_split_bps = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, index=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Contributor Profiles
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
#  Instrument
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
    drift_budget_pct = Column(Float, default=0.0)
    contact_email = Column(String(255), default="")
    card_data = Column(JSONB, default=dict)
    uploaded_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    is_public = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    uploader = relationship("User", foreign_keys=[uploaded_by], lazy="selectin")


# ═══════════════════════════════════════════════════════════════════════════
#  Audit Log
# ═══════════════════════════════════════════════════════════════════════════


class AuditLog(Base):
    """Platform audit log — records significant contributor and admin actions."""

    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    action = Column(String(100), nullable=False, index=True)
    actor_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    actor_name = Column(String(100), default="")
    resource_type = Column(String(50), default="")
    resource_id = Column(String(255), default="")
    detail = Column(Text, default="")
    ip_address = Column(String(45), default="")
    created_at = Column(DateTime(timezone=True), default=_utcnow, index=True)

    actor = relationship("User", foreign_keys=[actor_id], lazy="selectin")


# ═══════════════════════════════════════════════════════════════════════════
#  PWM Protocol — Principles, Specs, Benchmarks, Activity, Solutions
# ═══════════════════════════════════════════════════════════════════════════


class Principle(Base):
    """L1 Principle — physics fingerprint and observable profile."""

    __tablename__ = "principles"

    principle_id        = Column(String(16),  primary_key=True)   # L1-xxx artifact ID
    document_hash       = Column(String(64),  nullable=False)
    physics_fingerprint = Column(JSONB,        nullable=False)     # carrier, mechanism, primitives …
    observable_profile  = Column(JSONB,        nullable=False)     # {obs: {floor, sota_reference}}
    size_tiers          = Column(JSONB,        nullable=False)     # {dim: [tier_values]}
    hardness_fn         = Column(JSONB,        nullable=False)     # {type, baseline, metric}
    initiator_dataset   = Column(JSONB,        nullable=False)     # [{name, ipfs_cid, license_hash, weight}]
    status              = Column(String(20),   nullable=False, default="draft", index=True)
    staked_pwm          = Column(Float,        nullable=False, default=0.0)
    created_at          = Column(DateTime(timezone=True), default=_utcnow)
    chain_hash          = Column(String(66),   nullable=True)      # 0x + keccak256 of canonical JSON
    chain_tx_hash       = Column(String(66),   nullable=True)      # Base Sepolia registration tx
    chain_block         = Column(Integer,      nullable=True)      # block number of registration

    digital_twins       = relationship("ProtocolDigitalTwin", back_populates="principle", lazy="selectin")


class ProtocolDigitalTwin(Base):
    """L2 Digital Twin — six-tuple (Omega, E, B, I, O, epsilon).

    Formerly named "Spec" / table `protocol_digital_twins`; renamed to "Digital Twin"
    (table `protocol_digital_twins`, PK `digital_twin_id`) in the 2026-06 rename.
    """

    __tablename__ = "protocol_digital_twins"

    digital_twin_id     = Column(String(16),  primary_key=True)   # L2-xxx-xxx artifact ID
    principle_id        = Column(String(16),  ForeignKey("principles.principle_id"), nullable=False, index=True)
    digital_twin_case   = Column(String(100), nullable=False)
    omega_type          = Column(String(10),  nullable=False)      # 'range' | 'point'
    omega               = Column(JSONB,        nullable=False)
    epsilon             = Column(JSONB,        nullable=False)
    difficulty_score    = Column(Float,        nullable=True)      # d(Ω_i) or d(P)
    delta_tier          = Column(Integer,      nullable=True)      # 1 | 3 | 5 | 10 | 50
    fingerprint         = Column(String(64),   nullable=True, index=True)  # for I-benchmark dupe check
    status              = Column(String(20),   nullable=False, default="pending", index=True)
    submitted_by        = Column(String(255),  nullable=False)
    created_at          = Column(DateTime(timezone=True), default=_utcnow)
    chain_hash          = Column(String(66),   nullable=True)
    chain_tx_hash       = Column(String(66),   nullable=True)
    chain_block         = Column(Integer,      nullable=True)

    principle           = relationship("Principle", back_populates="digital_twins")
    benchmarks          = relationship("ProtocolBenchmark", back_populates="digital_twin", lazy="selectin")


class DigitalTwinSetupImage(Base):
    """Experimental-setup figure for an L2 digital twin (best-effort).

    Stored in a side-table — NOT a column on `protocol_digital_twins` — so the
    hot catalog/list queries never drag image blobs through TOAST. One image per
    digital twin. Images are sourced ONLY from open-access CC BY / CC0 papers;
    `license` + `attribution` + `source_url` must always be recorded.
    Formerly `DigitalTwinSetupImage` / table `digital_twin_setup_images`.
    """

    __tablename__ = "digital_twin_setup_images"

    digital_twin_id = Column(String(16), ForeignKey("protocol_digital_twins.digital_twin_id"), primary_key=True)
    image_bytes  = Column(LargeBinary, nullable=False)
    mime         = Column(String(32),  nullable=False, default="image/jpeg")
    width        = Column(Integer,     nullable=True)
    height       = Column(Integer,     nullable=True)
    source_url   = Column(Text,        nullable=True)   # paper / page the figure came from
    caption      = Column(Text,        nullable=True)   # figure caption
    attribution  = Column(Text,        nullable=True)   # authors / journal
    license      = Column(String(32),  nullable=True)   # must be CC BY / CC0 / public-domain
    uploaded_by  = Column(String(255), nullable=True)   # "auto-fetch" or admin username
    created_at   = Column(DateTime(timezone=True), default=_utcnow)


class CliAuthoringSession(Base):
    """In-progress artifact bundle being authored in the browser CLI (/cli).

    One active session per (user, layer): holds the field values + the
    Haiku-drafted MD/JSON across requests so `show`/`validate`/`submit` work.
    """

    __tablename__ = "cli_authoring_sessions"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    user_id      = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    layer        = Column(String(20), nullable=False)   # principle | digital-twin | benchmark | solution
    fields       = Column(JSONB, nullable=False, default=dict)   # {field: value}
    draft_md     = Column(Text, nullable=True)
    draft_json   = Column(JSONB, nullable=True)
    is_active    = Column(Boolean, nullable=False, default=True, index=True)
    created_at   = Column(DateTime(timezone=True), default=_utcnow)
    updated_at   = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)


class CliHaikuUsage(Base):
    """Per-user/day Haiku call counter for the CLI rate limit (100/day)."""

    __tablename__ = "cli_haiku_usage"

    user_id      = Column(Integer, ForeignKey("users.id"), primary_key=True)
    day          = Column(String(10), primary_key=True)   # YYYY-MM-DD (UTC)
    count        = Column(Integer, nullable=False, default=0)
    updated_at   = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)


class ProtocolBenchmark(Base):
    """L3 Benchmark — parametric (P-benchmark) or fixed I-benchmark tier."""

    __tablename__ = "protocol_benchmarks"

    benchmark_id        = Column(String(20),  primary_key=True)   # L3-xxx-xxx-xxx artifact ID
    digital_twin_id     = Column(String(16),  ForeignKey("protocol_digital_twins.digital_twin_id"), nullable=False, index=True)
    benchmark_type      = Column(String(20),  nullable=False)      # 'parametric' | 'fixed'
    tier                = Column(String(5),   nullable=True)       # 'XS' | 'M' | 'XL' | NULL
    manifest            = Column(JSONB,        nullable=False)
    ipfs_cid            = Column(String(255),  nullable=True)
    benchmark_hash      = Column(String(64),   nullable=False)
    quality_tier        = Column(String(20),   nullable=False, default="standard")  # standard | verified
    builder             = Column(String(255),  nullable=False)
    royalty_bps         = Column(Integer,      nullable=False, default=1500)         # 15% = 1500 bps
    created_at          = Column(DateTime(timezone=True), default=_utcnow)
    chain_hash          = Column(String(66),   nullable=True)
    chain_tx_hash       = Column(String(66),   nullable=True)   # parent L3 tx — shared across T1/T2/T3/T4 tier rows
    chain_block         = Column(Integer,      nullable=True)

    digital_twin        = relationship("ProtocolDigitalTwin", back_populates="benchmarks")
    activity            = relationship("BenchmarkActivity", back_populates="benchmark", uselist=False)
    solutions           = relationship("ProtocolSolution", back_populates="benchmark", lazy="selectin")


class BenchmarkActivity(Base):
    """Activity tracking for pool weight calculation (Σ d(Ω_s) last 90 days)."""

    __tablename__ = "benchmark_activity"

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    principle_id        = Column(String(16),  ForeignKey("principles.principle_id"), nullable=False, index=True)
    digital_twin_id     = Column(String(16),  ForeignKey("protocol_digital_twins.digital_twin_id"), nullable=False, index=True)
    benchmark_id        = Column(String(20),  ForeignKey("protocol_benchmarks.benchmark_id"), nullable=False, unique=True, index=True)
    activity_score      = Column(Float,        nullable=False, default=0.0)  # Σ d(Ω_s) last 90 days
    last_updated        = Column(DateTime(timezone=True), default=_utcnow)

    benchmark           = relationship("ProtocolBenchmark", back_populates="activity")


class ProtocolSolution(Base):
    """L4 Solution — verified S4 certificate submission."""

    __tablename__ = "protocol_solutions"

    solution_id         = Column(String(100), primary_key=True)
    benchmark_id        = Column(String(100), ForeignKey("protocol_benchmarks.benchmark_id"), nullable=False, index=True)
    solver_id           = Column(String(255), nullable=False, index=True)
    certificate_hash    = Column(String(64),  nullable=False)
    omega_s             = Column(JSONB,        nullable=True)       # Ω at which solution was evaluated
    difficulty_score    = Column(Float,        nullable=False)      # d(Ω_s)
    quality_ratio       = Column(Float,        nullable=False)      # q ∈ [0.75, 1.0]
    status              = Column(String(20),   nullable=False, default="pending", index=True)
    reward_pwm          = Column(Float,        nullable=True)
    submitted_at        = Column(DateTime(timezone=True), default=_utcnow)
    finalized_at        = Column(DateTime(timezone=True), nullable=True)

    benchmark           = relationship("ProtocolBenchmark", back_populates="solutions")


# ═══════════════════════════════════════════════════════════════════════════
#  PWM Protocol — Governance and Challenges
# ═══════════════════════════════════════════════════════════════════════════


class GovernanceProposal(Base):
    """Sub-DAO governance proposal (conviction voting, no hard deadline)."""

    __tablename__ = "governance_proposals"

    proposal_id     = Column(String(64),  primary_key=True)
    principle_id    = Column(String(16),  ForeignKey("principles.principle_id"), nullable=False, index=True)
    proposal_type   = Column(String(40),  nullable=False)      # dataset_add | dataset_replace | sota_update | …
    payload         = Column(JSONB,        nullable=False)      # type-specific content
    conviction      = Column(Float,        nullable=False, default=0.0)
    threshold_pct   = Column(Float,        nullable=False)      # 0.50 or 0.667
    status          = Column(String(20),   nullable=False, default="open", index=True)
    created_by      = Column(String(255),  nullable=False)
    created_at      = Column(DateTime(timezone=True), default=_utcnow)
    resolved_at     = Column(DateTime(timezone=True), nullable=True)

    votes           = relationship("GovernanceVote", back_populates="proposal", lazy="selectin")


class GovernanceVote(Base):
    """A single conviction vote cast by a contributor."""

    __tablename__ = "governance_votes"

    vote_id         = Column(String(64),  primary_key=True)
    proposal_id     = Column(String(64),  ForeignKey("governance_proposals.proposal_id"), nullable=False, index=True)
    voter_id        = Column(String(255), nullable=False, index=True)
    weight          = Column(Float,        nullable=False)
    voted_at        = Column(DateTime(timezone=True), default=_utcnow)

    proposal        = relationship("GovernanceProposal", back_populates="votes")


class GovernanceDelegation(Base):
    """Liquid democracy delegation for a specific Principle's sub-DAO."""

    __tablename__ = "governance_delegations"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    delegator_id    = Column(String(255), nullable=False, index=True)
    delegate_id     = Column(String(255), nullable=False, index=True)
    principle_id    = Column(String(16),  ForeignKey("principles.principle_id"), nullable=False, index=True)
    created_at      = Column(DateTime(timezone=True), default=_utcnow)
    revoked_at      = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        __import__("sqlalchemy").UniqueConstraint("delegator_id", "principle_id", name="uq_delegation_delegator_principle"),
    )


class ProtocolChallenge(Base):
    """On-chain challenge record (mirrored off-chain for platform UI)."""

    __tablename__ = "protocol_challenges"

    challenge_id    = Column(String(64),  primary_key=True)
    cert_hash       = Column(String(64),  nullable=False, index=True)
    solution_id     = Column(String(100), ForeignKey("protocol_solutions.solution_id"), nullable=True, index=True)
    challenger      = Column(String(255), nullable=False)
    evidence_type   = Column(String(40),  nullable=False)   # REPLAY_MISMATCH | METRIC_MISMATCH | GATE_FAILURE | …
    evidence_hash   = Column(String(64),  nullable=False)
    disputed_step   = Column(Integer,     nullable=True)     # 0-5 for bisection
    bond_amount     = Column(Float,        nullable=False)
    status          = Column(String(20),   nullable=False, default="open", index=True)
    verdict         = Column(String(20),   nullable=True)    # upheld | dismissed
    verdict_log     = Column(JSONB,        nullable=True)    # [{verifier_id, verdict, message}, ...]
    submitted_at    = Column(DateTime(timezone=True), default=_utcnow)
    created_at      = Column(DateTime(timezone=True), default=_utcnow)
    resolved_at     = Column(DateTime(timezone=True), nullable=True)


# ═══════════════════════════════════════════════════════════════════════════
#  PWM Submission System
# ═══════════════════════════════════════════════════════════════════════════


class PWMSubmission(Base):
    """User-submitted PWM protocol artifact — Principle, Spec, I-Benchmark, or Solution."""

    __tablename__ = "pwm_submissions"

    id               = Column(Integer,  primary_key=True, autoincrement=True)
    submission_id    = Column(String(64), unique=True, nullable=False, index=True)
    submission_type  = Column(String(20), nullable=False, index=True)
    # "principle" | "spec" | "benchmark" | "solution"

    # Submitter identity
    submitted_by_id  = Column(Integer, ForeignKey("users.id"), nullable=True)
    submitter_wallet = Column(String(42), nullable=True)
    submitter_label  = Column(String(255), nullable=False, server_default="anonymous")

    # Status lifecycle
    status           = Column(String(20), nullable=False, default="testnet", index=True)
    # "testnet" | "reviewing" | "mainnet" | "rejected" | "needs_revision"

    # Parent reference for hierarchy (spec→principle, benchmark→spec, solution→principle)
    parent_id        = Column(String(255), nullable=True)
    parent_type      = Column(String(20),  nullable=True)

    # All simplified form fields as JSONB
    form_data        = Column(JSONB, nullable=False, default=dict)

    # Verification output
    gate_results     = Column(JSONB, nullable=True)
    # {S1: {pass: bool, reason: str}, S2: {...}, S3: {...}, S4: {...}}
    verdict          = Column(String(20), nullable=True)   # ACCEPT | NEEDS_REVISION | REJECT
    verdict_reason   = Column(Text, nullable=True)
    verified_at      = Column(DateTime(timezone=True), nullable=True)

    # Revision tracking
    revision_count   = Column(Integer, nullable=False, default=0)
    original_id      = Column(String(64), nullable=True)   # original submission_id for revisions

    # PWM reward (set on mainnet promotion)
    pwm_awarded      = Column(Float, nullable=True)

    # Timestamps
    submitted_at     = Column(DateTime(timezone=True), default=_utcnow)
    updated_at       = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)
