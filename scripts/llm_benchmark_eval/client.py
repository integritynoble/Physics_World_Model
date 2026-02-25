"""Async OpenAI-compatible client for comparegpt.io."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from openai import AsyncOpenAI, APIError, APIStatusError, APITimeoutError, RateLimitError

from .config import (
    COMPAREGPT_BASE_URL,
    MAX_CONCURRENCY,
    MAX_RETRIES,
    REQUEST_TIMEOUT,
    RETRY_BASE_DELAY,
    get_api_key,
)

logger = logging.getLogger(__name__)


def _parse_json(text: str) -> dict[str, Any]:
    """Extract and parse a JSON object from LLM output.

    Handles markdown fences, leading prose, etc.
    """
    cleaned = text.strip()

    # Strip markdown code fences
    if cleaned.startswith("```"):
        first_newline = cleaned.index("\n") if "\n" in cleaned else 3
        cleaned = cleaned[first_newline + 1:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    # Direct parse
    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Find embedded JSON object
    brace_start = cleaned.find("{")
    brace_end = cleaned.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        candidate = cleaned[brace_start: brace_end + 1]
        try:
            result = json.loads(candidate)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"Could not parse JSON from LLM response: {text[:200]}..."
        if len(text) > 200
        else f"Could not parse JSON from LLM response: {text}"
    )


class CompareGPTClient:
    """Async client wrapping comparegpt.io's OpenAI-compatible endpoint."""

    def __init__(self) -> None:
        self._client = AsyncOpenAI(
            base_url=COMPAREGPT_BASE_URL,
            api_key=get_api_key(),
            timeout=REQUEST_TIMEOUT,
        )
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async def chat(
        self,
        model_id: str,
        system: str,
        user: str,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Send a chat completion request and return parsed JSON + metadata.

        Returns
        -------
        dict with keys:
            raw_text: str — full assistant reply
            parsed: dict | None — parsed JSON (None if parsing failed)
            latency_s: float — wall-clock seconds
            error: str | None — error message if the call failed entirely
        """
        async with self._semaphore:
            return await self._chat_with_retry(model_id, system, user, temperature)

    async def _chat_with_retry(
        self,
        model_id: str,
        system: str,
        user: str,
        temperature: float,
    ) -> dict[str, Any]:
        last_error: str | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            t0 = time.monotonic()
            try:
                response = await self._client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                )
                latency = time.monotonic() - t0
                raw_text = response.choices[0].message.content or ""
                try:
                    parsed = _parse_json(raw_text)
                except ValueError:
                    parsed = None
                return {
                    "raw_text": raw_text,
                    "parsed": parsed,
                    "latency_s": round(latency, 3),
                    "error": None,
                }
            except RateLimitError as exc:
                last_error = f"RateLimitError: {exc}"
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "Rate limited (attempt %d/%d), retrying in %.1fs: %s",
                    attempt, MAX_RETRIES, delay, exc,
                )
                await asyncio.sleep(delay)
            except APIStatusError as exc:
                # Don't retry auth errors (401/403)
                if exc.status_code in (401, 403):
                    latency = time.monotonic() - t0
                    return {
                        "raw_text": "",
                        "parsed": None,
                        "latency_s": round(latency, 3),
                        "error": f"AuthError ({exc.status_code}): {exc}",
                    }
                last_error = f"{type(exc).__name__}: {exc}"
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "API error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt, MAX_RETRIES, delay, exc,
                )
                await asyncio.sleep(delay)
            except (APIError, APITimeoutError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "API error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt, MAX_RETRIES, delay, exc,
                )
                await asyncio.sleep(delay)
            except Exception as exc:
                latency = time.monotonic() - t0
                return {
                    "raw_text": "",
                    "parsed": None,
                    "latency_s": round(latency, 3),
                    "error": f"{type(exc).__name__}: {exc}",
                }

        # All retries exhausted
        return {
            "raw_text": "",
            "parsed": None,
            "latency_s": 0.0,
            "error": f"Max retries ({MAX_RETRIES}) exhausted. Last: {last_error}",
        }

    async def close(self) -> None:
        await self._client.close()
