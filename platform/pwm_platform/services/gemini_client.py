"""Async multi-turn Gemini client for the spec chat feature.

Uses the CompareGPT OpenAI-compatible API (https://comparegpt.io/api) to call
Gemini 2.5 Flash with multi-turn conversation history.  Conversations are
stored in-memory with a 1-hour TTL and lazy cleanup.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

_MODEL = "gemini-2.5-flash"
_COMPAREGPT_BASE_URL = "https://comparegpt.io/api"
_COMPAREGPT_API_KEY_ENV = "COMPAREGPT_API_KEY"
_MAX_OUTPUT_TOKENS = 4096
_TEMPERATURE = 0.3
_REQUEST_TIMEOUT_S = 120
_SESSION_TTL_S = 3600  # 1 hour


# ── In-memory conversation store ─────────────────────────────────────────────

# {session_id: {"history": [{"role": ..., "content": ...}], "created_at": float}}
_conversations: dict[str, dict[str, Any]] = {}


def _cleanup_stale() -> None:
    """Remove conversations older than TTL (lazy, called before each op)."""
    now = time.time()
    stale = [
        sid for sid, conv in _conversations.items()
        if now - conv["created_at"] > _SESSION_TTL_S
    ]
    for sid in stale:
        del _conversations[sid]


def create_conversation() -> str:
    """Create a new conversation session and return its ID."""
    _cleanup_stale()
    session_id = uuid.uuid4().hex
    _conversations[session_id] = {"history": [], "created_at": time.time()}
    return session_id


def get_conversation(session_id: str) -> dict[str, Any] | None:
    """Return conversation dict or None if expired / not found."""
    _cleanup_stale()
    return _conversations.get(session_id)


def append_to_conversation(session_id: str, role: str, text: str) -> None:
    """Append a turn to the conversation history.

    Parameters
    ----------
    role : str
        ``"user"`` or ``"model"`` (will be mapped to ``"assistant"`` for the
        OpenAI-compatible API).
    """
    conv = _conversations.get(session_id)
    if conv is None:
        return
    api_role = "assistant" if role == "model" else role
    conv["history"].append({"role": api_role, "content": text})


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
