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

        # Ensure platformaigpt@gmail.com is admin
        await conn.execute(text(
            "UPDATE users SET role = 'admin' WHERE email = 'platformaigpt@gmail.com' AND role != 'admin'"
        ))
