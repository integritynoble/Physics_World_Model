"""Async multi-turn Gemini client for the spec chat feature.

Uses httpx (already a platform dependency) to call the Gemini 2.5 Flash REST
API with multi-turn conversation history.  Conversations are stored in-memory
with a 1-hour TTL and lazy cleanup.
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

_GEMINI_MODEL = "gemini-2.5-flash"
_GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
_MAX_OUTPUT_TOKENS = 4096
_TEMPERATURE = 0.3
_REQUEST_TIMEOUT_S = 120
_SESSION_TTL_S = 3600  # 1 hour


# ── In-memory conversation store ─────────────────────────────────────────────

# {session_id: {"history": [{"role": ..., "parts": [...]}], "created_at": float}}
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
    """Append a turn to the conversation history."""
    conv = _conversations.get(session_id)
    if conv is None:
        return
    conv["history"].append({"role": role, "parts": [{"text": text}]})


# ── Gemini API call ──────────────────────────────────────────────────────────


async def call_gemini(system_prompt: str, history: list[dict]) -> str:
    """Call Gemini 2.5 Flash with a system prompt and multi-turn history.

    Parameters
    ----------
    system_prompt : str
        System instruction for the model.
    history : list[dict]
        Conversation turns, each ``{"role": "user"|"model", "parts": [{"text": ...}]}``.

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
    api_key = os.environ.get("PWM_GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "PWM_GEMINI_API_KEY not set. Cannot call Gemini API."
        )

    url = _GEMINI_URL.format(model=_GEMINI_MODEL)
    payload: dict[str, Any] = {
        "systemInstruction": {
            "parts": [{"text": system_prompt}],
        },
        "contents": history,
        "generationConfig": {
            "maxOutputTokens": _MAX_OUTPUT_TOKENS,
            "temperature": _TEMPERATURE,
        },
    }

    logger.debug("Gemini request: model=%s, turns=%d", _GEMINI_MODEL, len(history))

    async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_S) as client:
        resp = await client.post(url, params={"key": api_key}, json=payload)
        resp.raise_for_status()
        body = resp.json()

    try:
        return body["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(
            f"Unexpected Gemini response structure: {body}"
        ) from exc
