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
