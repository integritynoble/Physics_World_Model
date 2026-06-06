"""
Async SQLAlchemy engine + session factory for PostgreSQL.
"""

from __future__ import annotations

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from pwm_platform.config import settings

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    pool_size=5,
    max_overflow=10,
)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Shared declarative base for all models."""
    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency — yields an async DB session."""
    async with async_session_factory() as session:
        yield session


async def init_db() -> None:
    """Create all tables if they don't exist (safe with multiple workers)."""
    from sqlalchemy import text

    async with engine.begin() as conn:
        # ── Pre-create_all rename: Spec → Digital Twin (2026-06) ──────────
        # MUST run BEFORE create_all, otherwise create_all would make fresh
        # EMPTY protocol_digital_twins / digital_twin_setup_images tables and
        # orphan the existing rows. Idempotent + guarded on the OLD names so it
        # is a no-op once applied (and on brand-new installs where neither
        # exists yet — create_all then makes the new tables directly).
        # NOTE: the left-hand (OLD) names below are intentionally the original
        # `spec*` identifiers. Do NOT search-replace them to the new names — the
        # whole point is to rename FROM old TO new on already-deployed databases.
        # Each statement is GUARDED (DO $$ IF … $$) so it can NEVER raise — a
        # single raised DDL would abort this whole transaction and fail startup.
        _OLD = "spec"  # guard token so a future blanket rename can't silently neuter this block

        def _rename_table(old: str, new: str) -> str:
            return (
                f"DO $$ BEGIN IF EXISTS (SELECT 1 FROM information_schema.tables "
                f"WHERE table_name='{old}') AND NOT EXISTS (SELECT 1 FROM "
                f"information_schema.tables WHERE table_name='{new}') THEN "
                f"ALTER TABLE {old} RENAME TO {new}; END IF; END $$"
            )

        def _rename_col(table: str, old: str, new: str) -> str:
            return (
                f"DO $$ BEGIN IF EXISTS (SELECT 1 FROM information_schema.columns "
                f"WHERE table_name='{table}' AND column_name='{old}') AND NOT EXISTS "
                f"(SELECT 1 FROM information_schema.columns WHERE table_name='{table}' "
                f"AND column_name='{new}') THEN "
                f"ALTER TABLE {table} RENAME COLUMN {old} TO {new}; END IF; END $$"
            )

        for rename_sql in [
            # tables: protocol_specs → protocol_digital_twins, spec_setup_images → digital_twin_setup_images
            _rename_table(f"protocol_{_OLD}s", "protocol_digital_twins"),
            _rename_table(f"{_OLD}_setup_images", "digital_twin_setup_images"),
            # primary/foreign-key columns: spec_id → digital_twin_id, spec_case → digital_twin_case
            _rename_col("protocol_digital_twins", f"{_OLD}_id", "digital_twin_id"),
            _rename_col("protocol_digital_twins", f"{_OLD}_case", "digital_twin_case"),
            _rename_col("protocol_benchmarks", f"{_OLD}_id", "digital_twin_id"),
            _rename_col("benchmark_activity", f"{_OLD}_id", "digital_twin_id"),
            _rename_col("digital_twin_setup_images", f"{_OLD}_id", "digital_twin_id"),
        ]:
            await conn.execute(text(rename_sql))

        # Use checkfirst=True (the default) and catch race conditions
        try:
            await conn.run_sync(Base.metadata.create_all)
        except Exception:
            # Tables likely created by another worker — verify connectivity
            await conn.execute(text("SELECT 1"))

        # Idempotent migration: add is_public column to runs table
        await conn.execute(text(
            "ALTER TABLE runs ADD COLUMN IF NOT EXISTS is_public BOOLEAN DEFAULT TRUE"
        ))

        # Idempotent migration: dataset-mode columns on spec_chat_sessions
        for col in ("dataset_meta", "matrix_meta", "ground_truth_meta"):
            await conn.execute(text(
                f"ALTER TABLE spec_chat_sessions ADD COLUMN IF NOT EXISTS {col} JSONB"
            ))

        # Idempotent migration: credit_balance on users
        await conn.execute(text(
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS credit_balance DOUBLE PRECISION DEFAULT 100.0"
        ))

        # Idempotent migration: challenge_submissions new columns
        await conn.execute(text(
            "ALTER TABLE challenge_submissions ADD COLUMN IF NOT EXISTS category VARCHAR(30) DEFAULT 'competition'"
        ))
        await conn.execute(text(
            "ALTER TABLE challenge_submissions ADD COLUMN IF NOT EXISTS credit_cost DOUBLE PRECISION DEFAULT 0.0"
        ))

        # Idempotent migration: billing tables (credit_accounts, etc.)
        # The tables are created by Base.metadata.create_all above;
        # here we just add any columns that may be missing on existing installs.
        for col_def in [
            "ALTER TABLE credit_accounts ADD COLUMN IF NOT EXISTS overage_run_credits INTEGER DEFAULT 0",
            "ALTER TABLE credit_accounts ADD COLUMN IF NOT EXISTS overage_report_credits INTEGER DEFAULT 0",
            "ALTER TABLE credit_accounts ADD COLUMN IF NOT EXISTS legacy_credit_balance DOUBLE PRECISION DEFAULT 100.0",
            "ALTER TABLE credit_accounts ADD COLUMN IF NOT EXISTS credits_expire_at TIMESTAMPTZ",
            "ALTER TABLE payment_orders ADD COLUMN IF NOT EXISTS amount_cny DOUBLE PRECISION DEFAULT 0.0",
            "ALTER TABLE payment_orders ADD COLUMN IF NOT EXISTS credit_amount INTEGER",
        ]:
            try:
                await conn.execute(text(col_def))
            except Exception:
                pass  # table may not exist yet on first run

        # Note: platformaigpt@gmail.com role is managed via /admin/users
        # Do NOT auto-promote here — role changes should persist across restarts

        # Idempotent migration: spec_md and usage_type on spec_chat_sessions
        for col_def in [
            "ALTER TABLE spec_chat_sessions ADD COLUMN IF NOT EXISTS spec_md TEXT",
            "ALTER TABLE spec_chat_sessions ADD COLUMN IF NOT EXISTS usage_type VARCHAR(20)",
        ]:
            await conn.execute(text(col_def))

        # Idempotent migration: 90/10 split columns on pwm_token_transactions.
        # On a `spend` row these record how the spent PWM is to be credited
        # on-chain (provider_amount → provider_wallet, pool_amount → mining pool);
        # the settlement relayer (M6) reads them. NULL on award/adjust rows.
        for col_def in [
            "ALTER TABLE pwm_token_transactions ADD COLUMN IF NOT EXISTS provider_wallet VARCHAR(64)",
            "ALTER TABLE pwm_token_transactions ADD COLUMN IF NOT EXISTS provider_amount DOUBLE PRECISION",
            "ALTER TABLE pwm_token_transactions ADD COLUMN IF NOT EXISTS pool_amount DOUBLE PRECISION",
            "ALTER TABLE pwm_token_transactions ADD COLUMN IF NOT EXISTS provider_split_bps INTEGER",
        ]:
            try:
                await conn.execute(text(col_def))
            except Exception:
                pass  # table may not exist yet on first run

        # Idempotent migration: trust-tier columns on challenge_submissions
        for col_def in [
            "ALTER TABLE challenge_submissions ADD COLUMN IF NOT EXISTS trust_tier VARCHAR(30) NOT NULL DEFAULT 'draft'",
            "ALTER TABLE challenge_submissions ADD COLUMN IF NOT EXISTS gate_verdicts JSONB",
        ]:
            try:
                await conn.execute(text(col_def))
            except Exception:
                pass  # table may not exist yet on first run

        # Index on trust_tier for leaderboard filtering
        try:
            await conn.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_challenge_submissions_trust_tier "
                "ON challenge_submissions (trust_tier)"
            ))
        except Exception:
            pass

        # Idempotent migration: contributor economy columns on contributor_profiles
        for col_def in [
            "ALTER TABLE contributor_profiles ADD COLUMN IF NOT EXISTS roles JSONB DEFAULT '[]'::jsonb",
            "ALTER TABLE contributor_profiles ADD COLUMN IF NOT EXISTS badges JSONB DEFAULT '[]'::jsonb",
            "ALTER TABLE contributor_profiles ADD COLUMN IF NOT EXISTS maintained_modalities JSONB DEFAULT '[]'::jsonb",
            "ALTER TABLE contributor_profiles ADD COLUMN IF NOT EXISTS contribution_history JSONB DEFAULT '[]'::jsonb",
            "ALTER TABLE contributor_profiles ADD COLUMN IF NOT EXISTS total_reproductions INTEGER DEFAULT 0",
            "ALTER TABLE contributor_profiles ADD COLUMN IF NOT EXISTS total_certifications INTEGER DEFAULT 0",
            "ALTER TABLE contributor_profiles ADD COLUMN IF NOT EXISTS total_claims_reviewed INTEGER DEFAULT 0",
        ]:
            try:
                await conn.execute(text(col_def))
            except Exception:
                pass  # table may not exist yet on first run

        # Idempotent migration: audit_log ip_address column
        try:
            await conn.execute(text(
                "ALTER TABLE audit_log ADD COLUMN IF NOT EXISTS ip_address VARCHAR(45) DEFAULT ''"
            ))
        except Exception:
            pass

        # Idempotent migration: PWM protocol tables (added post-initial-deploy)
        for col_def in [
            # protocol_challenges: verdict_log + created_at added in Phase 9 fix
            "ALTER TABLE protocol_challenges ADD COLUMN IF NOT EXISTS verdict_log JSONB",
            "ALTER TABLE protocol_challenges ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW()",
        ]:
            try:
                await conn.execute(text(col_def))
            except Exception:
                pass

        # Idempotent column renames: spec_hash → spec_id (L2-xxx-xxx naming scheme)
        for rename_sql in [
            """
            DO $$ BEGIN
              IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name='protocol_digital_twins' AND column_name='spec_hash'
              ) THEN
                ALTER TABLE protocol_digital_twins RENAME COLUMN spec_hash TO digital_twin_id;
              END IF;
            END $$
            """,
            """
            DO $$ BEGIN
              IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name='protocol_benchmarks' AND column_name='spec_hash'
              ) THEN
                ALTER TABLE protocol_benchmarks RENAME COLUMN spec_hash TO digital_twin_id;
              END IF;
            END $$
            """,
            """
            DO $$ BEGIN
              IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name='benchmark_activity' AND column_name='spec_hash'
              ) THEN
                ALTER TABLE benchmark_activity RENAME COLUMN spec_hash TO digital_twin_id;
              END IF;
            END $$
            """,
        ]:
            try:
                await conn.execute(text(rename_sql))
            except Exception:
                pass

        # Idempotent migration: pwm_token_accounts / pwm_token_transactions
        # (tables created by create_all above; this guards against drift)
        for col_def in [
            "ALTER TABLE pwm_token_accounts ADD COLUMN IF NOT EXISTS on_chain_address VARCHAR(255)",
            "ALTER TABLE pwm_token_accounts ADD COLUMN IF NOT EXISTS lifetime_earned DOUBLE PRECISION DEFAULT 0.0",
            "ALTER TABLE pwm_token_accounts ADD COLUMN IF NOT EXISTS wallet_private_key_enc VARCHAR(512)",
            "ALTER TABLE pwm_token_transactions ADD COLUMN IF NOT EXISTS artifact_type VARCHAR(30)",
            "ALTER TABLE pwm_token_transactions ADD COLUMN IF NOT EXISTS submission_id VARCHAR(100)",
            "ALTER TABLE pwm_token_transactions ADD COLUMN IF NOT EXISTS awarded_by INTEGER",
        ]:
            try:
                await conn.execute(text(col_def))
            except Exception:
                pass
