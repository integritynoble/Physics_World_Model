"""Async multi-turn Gemini client for the spec chat feature.

Uses the CompareGPT OpenAI-compatible API (https://comparegpt.io/api) to call
Gemini 2.5 Flash with multi-turn conversation history.

Conversations are persisted to PostgreSQL (spec_chat_sessions table) so that
user history survives server restarts.  A lightweight in-memory cache avoids
DB reads on every turn within the same session.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from pwm_platform.db.models import SpecChatSession

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

_MODEL = "gemini-2.5-flash"
_COMPAREGPT_BASE_URL = "https://comparegpt.io/api"
_COMPAREGPT_API_KEY_ENV = "COMPAREGPT_API_KEY"
_MAX_OUTPUT_TOKENS = 4096
_TEMPERATURE = 0.3
_REQUEST_TIMEOUT_S = 120

# ── In-memory cache (write-through to DB) ────────────────────────────────────

_cache: dict[str, list[dict]] = {}


# ── Conversation CRUD (DB-backed) ────────────────────────────────────────────


async def create_conversation(
    db: AsyncSession,
    user_id: int | None = None,
    variant_key: str = "sd_cassi",
) -> str:
    """Create a new conversation session in the DB and return its ID."""
    session_id = uuid.uuid4().hex
    row = SpecChatSession(
        session_id=session_id,
        user_id=user_id,
        variant_key=variant_key,
        history=[],
    )
    db.add(row)
    await db.commit()
    _cache[session_id] = []
    return session_id


async def get_conversation(db: AsyncSession, session_id: str) -> list[dict] | None:
    """Return conversation history or None if not found."""
    # Check cache first
    if session_id in _cache:
        return _cache[session_id]

    # Fall back to DB
    result = await db.execute(
        select(SpecChatSession).where(SpecChatSession.session_id == session_id)
    )
    row = result.scalar_one_or_none()
    if row is None:
        return None

    history = row.history or []
    _cache[session_id] = history
    return history


async def append_to_conversation(
    db: AsyncSession, session_id: str, role: str, text: str,
) -> None:
    """Append a turn and persist to DB.

    Parameters
    ----------
    role : str
        ``"user"`` or ``"model"`` (mapped to ``"assistant"`` for the API).
    """
    api_role = "assistant" if role == "model" else role
    turn = {"role": api_role, "content": text}

    # Update cache
    if session_id not in _cache:
        await get_conversation(db, session_id)
    if session_id in _cache:
        _cache[session_id].append(turn)

    # Persist to DB
    result = await db.execute(
        select(SpecChatSession).where(SpecChatSession.session_id == session_id)
    )
    row = result.scalar_one_or_none()
    if row is not None:
        history = list(row.history or [])
        history.append(turn)
        row.history = history
        await db.commit()


async def list_user_sessions(
    db: AsyncSession, user_id: int, limit: int = 20,
) -> list[dict]:
    """Return recent sessions for a user (for history sidebar)."""
    result = await db.execute(
        select(SpecChatSession)
        .where(SpecChatSession.user_id == user_id)
        .order_by(SpecChatSession.updated_at.desc())
        .limit(limit)
    )
    rows = result.scalars().all()
    return [
        {
            "session_id": r.session_id,
            "variant_key": r.variant_key,
            "turns": len(r.history or []),
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "updated_at": r.updated_at.isoformat() if r.updated_at else None,
        }
        for r in rows
    ]


# ── Dataset metadata helpers ────────────────────────────────────────────


async def update_session_dataset_meta(
    db: AsyncSession,
    session_id: str,
    dataset_meta: dict | None = None,
    matrix_meta: dict | None = None,
    gt_meta: dict | None = None,
) -> None:
    """Store uploaded dataset metadata on a session."""
    result = await db.execute(
        select(SpecChatSession).where(SpecChatSession.session_id == session_id)
    )
    row = result.scalar_one_or_none()
    if row is None:
        return
    if dataset_meta is not None:
        row.dataset_meta = dataset_meta
    if matrix_meta is not None:
        row.matrix_meta = matrix_meta
    if gt_meta is not None:
        row.ground_truth_meta = gt_meta
    await db.commit()


async def get_session_dataset_meta(
    db: AsyncSession,
    session_id: str,
) -> tuple[dict | None, dict | None, dict | None]:
    """Retrieve dataset metadata from a session.

    Returns (dataset_meta, matrix_meta, ground_truth_meta).
    """
    result = await db.execute(
        select(SpecChatSession).where(SpecChatSession.session_id == session_id)
    )
    row = result.scalar_one_or_none()
    if row is None:
        return None, None, None
    return row.dataset_meta, row.matrix_meta, row.ground_truth_meta


# ── LLM API call (via CompareGPT) ───────────────────────────────────────────


async def call_gemini(system_prompt: str, history: list[dict]) -> str:
    """Call Gemini 2.5 Flash via the CompareGPT OpenAI-compatible API.

    Parameters
    ----------
    system_prompt : str
        System instruction for the model.
    history : list[dict]
        Conversation turns in OpenAI format:
        ``[{"role": "user"|"assistant", "content": "..."}]``.

    Returns
    -------
    str
        Model response text.

    Raises
    ------
    RuntimeError
        If the API key is missing or the response is malformed.
    httpx.HTTPStatusError
        If the API returns a non-2xx status.
    """
    api_key = os.environ.get(_COMPAREGPT_API_KEY_ENV, "")
    if not api_key:
        raise RuntimeError(
            f"{_COMPAREGPT_API_KEY_ENV} not set. Cannot call CompareGPT API."
        )

    url = f"{_COMPAREGPT_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    messages = [{"role": "system", "content": system_prompt}] + list(history)
    payload: dict[str, Any] = {
        "model": _MODEL,
        "messages": messages,
        "max_tokens": _MAX_OUTPUT_TOKENS,
        "temperature": _TEMPERATURE,
    }

    logger.debug(
        "CompareGPT request: model=%s, turns=%d, url=%s",
        _MODEL, len(history), url,
    )

    async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_S) as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        body = resp.json()

    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(
            f"Unexpected CompareGPT response structure: {body}"
        ) from exc
