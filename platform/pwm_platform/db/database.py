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

        # Idempotent migration: trust ratchet columns on challenge_submissions
        for col_def in [
            "ALTER TABLE challenge_submissions ADD COLUMN IF NOT EXISTS trust_tier VARCHAR(30) DEFAULT 'draft'",
            "ALTER TABLE challenge_submissions ADD COLUMN IF NOT EXISTS gate_verdicts JSONB",
        ]:
            try:
                await conn.execute(text(col_def))
            except Exception:
                pass

        # Idempotent migration: password_reset_tokens table columns
        for col_def in [
            "ALTER TABLE password_reset_tokens ADD COLUMN IF NOT EXISTS used BOOLEAN DEFAULT FALSE",
            "ALTER TABLE password_reset_tokens ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW()",
        ]:
            try:
                await conn.execute(text(col_def))
            except Exception:
                pass  # table may not exist yet on first run; create_all handles it

        # Idempotent migration: instruments table (create_all handles table creation;
        # add any missing columns for existing installs)
        for col_def in [
            "ALTER TABLE instruments ADD COLUMN IF NOT EXISTS is_public BOOLEAN DEFAULT TRUE",
            "ALTER TABLE instruments ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW()",
        ]:
            try:
                await conn.execute(text(col_def))
            except Exception:
                pass

        # Ensure platformaigpt@gmail.com is admin
        await conn.execute(text(
            "UPDATE users SET role = 'admin' WHERE email = 'platformaigpt@gmail.com' AND role != 'admin'"
        ))
