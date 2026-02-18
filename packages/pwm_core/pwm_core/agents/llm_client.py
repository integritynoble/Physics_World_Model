"""pwm_core.agents.llm_client

Multi-LLM client with automatic provider fallback for the PWM agent layer.

Supports Gemini, Claude (Anthropic), and OpenAI via their REST APIs using
the ``requests`` library (no provider-specific SDKs required).

Fallback chain
--------------
1. Gemini   -- env var ``PWM_GEMINI_API_KEY``,    model ``gemini-2.5-pro``
2. Claude   -- env var ``PWM_ANTHROPIC_API_KEY``,  model ``claude-sonnet-4-5-20250929``
3. OpenAI   -- env var ``PWM_OPENAI_API_KEY``,     model ``gpt-4o-mini``

The client walks the chain until it finds a provider whose API key is
available in the environment (or was passed explicitly).
"""

from __future__ import annotations

import json
import logging
import os
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import requests

# ---------------------------------------------------------------------------
# Optional dotenv support -- load .env if python-dotenv is installed
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider enum
# ---------------------------------------------------------------------------


class LLMProvider(Enum):
    """Supported LLM providers."""

    GEMINI = "gemini"
    CLAUDE = "claude"
    OPENAI = "openai"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROVIDER_DEFAULTS: Dict[LLMProvider, Dict[str, str]] = {
    LLMProvider.GEMINI: {
        "env_key": "PWM_GEMINI_API_KEY",
        "model": "gemini-2.5-pro",
    },
    LLMProvider.CLAUDE: {
        "env_key": "PWM_ANTHROPIC_API_KEY",
        "model": "claude-sonnet-4-5-20250929",
    },
    LLMProvider.OPENAI: {
        "env_key": "PWM_OPENAI_API_KEY",
        "model": "gpt-4o-mini",
    },
}

_FALLBACK_CHAIN: Tuple[LLMProvider, ...] = (
    LLMProvider.GEMINI,
    LLMProvider.CLAUDE,
    LLMProvider.OPENAI,
)

# API endpoints
_GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
_CLAUDE_URL = "https://api.anthropic.com/v1/messages"
_OPENAI_URL = "https://api.openai.com/v1/chat/completions"

# Anthropic API version header
_ANTHROPIC_VERSION = "2023-06-01"

# Reasonable defaults for HTTP requests
_REQUEST_TIMEOUT_S = 120
_MAX_TOKENS = 4096


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------


class LLMClient:
    """Multi-provider LLM client with automatic fallback.

    Parameters
    ----------
    provider : LLMProvider or str, optional
        Explicit provider to use.  When *None* the client walks the
        ``FALLBACK_CHAIN`` and picks the first provider whose API key is
        available in the environment.
    api_key : str, optional
        Explicit API key.  When *None* the key is read from the
        corresponding environment variable (see ``_PROVIDER_DEFAULTS``).
    model : str, optional
        Override the default model for the selected provider.

    Raises
    ------
    RuntimeError
        If no provider can be configured (no API keys found anywhere).
    """

    FALLBACK_CHAIN: Tuple[LLMProvider, ...] = _FALLBACK_CHAIN

    def __init__(
        self,
        provider: Optional[LLMProvider | str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        resolved_provider, resolved_key, resolved_model = self._resolve(
            provider, api_key, model
        )
        self.provider: LLMProvider = resolved_provider
        self.api_key: str = resolved_key
        self.model: str = resolved_model

        logger.info(
            "LLMClient initialised: provider=%s  model=%s",
            self.provider.value,
            self.model,
        )

    # ------------------------------------------------------------------
    # Resolution helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_provider(provider: LLMProvider | str) -> LLMProvider:
        """Convert a string to ``LLMProvider`` if necessary."""
        if isinstance(provider, LLMProvider):
            return provider
        try:
            return LLMProvider(provider.lower())
        except ValueError:
            raise ValueError(
                f"Unknown LLM provider '{provider}'. "
                f"Choose from: {[p.value for p in LLMProvider]}"
            ) from None

    @classmethod
    def _resolve(
        cls,
        provider: Optional[LLMProvider | str],
        api_key: Optional[str],
        model: Optional[str],
    ) -> Tuple[LLMProvider, str, str]:
        """Determine (provider, api_key, model) using explicit args + env.

        Returns
        -------
        tuple
            ``(provider, api_key, model)`` ready for use.

        Raises
        ------
        RuntimeError
            If no usable provider can be found.
        """
        # --- Explicit provider requested -----------------------------------
        if provider is not None:
            prov = cls._coerce_provider(provider)
            defaults = _PROVIDER_DEFAULTS[prov]
            key = api_key or os.environ.get(defaults["env_key"], "")
            if not key:
                raise RuntimeError(
                    f"API key for {prov.value} not found.  "
                    f"Set the {defaults['env_key']} environment variable or "
                    f"pass api_key= explicitly."
                )
            mdl = model or defaults["model"]
            return prov, key, mdl

        # --- Auto-detect via fallback chain --------------------------------
        for prov in cls.FALLBACK_CHAIN:
            defaults = _PROVIDER_DEFAULTS[prov]
            key = api_key or os.environ.get(defaults["env_key"], "")
            if key:
                mdl = model or defaults["model"]
                logger.info(
                    "Auto-selected provider %s via env %s",
                    prov.value,
                    defaults["env_key"],
                )
                return prov, key, mdl

        raise RuntimeError(
            "No LLM API key found.  Set one of: "
            + ", ".join(
                _PROVIDER_DEFAULTS[p]["env_key"] for p in cls.FALLBACK_CHAIN
            )
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        """Send a simple single-turn prompt and return the response text.

        This is a convenience wrapper used by ``BaseAgent.explain()`` and
        other call-sites that need a quick one-shot generation without
        structured output.

        Parameters
        ----------
        prompt : str
            The user-facing prompt text.

        Returns
        -------
        str
            Raw text response from the LLM.
        """
        system_msg = (
            "You are a helpful scientific-imaging assistant.  "
            "Respond concisely and accurately."
        )
        return self._call(system_msg, prompt)

    def select(
        self,
        prompt: str,
        available_keys: List[str],
    ) -> Dict[str, Any]:
        """Ask the LLM to choose from *available_keys* given a user prompt.

        The LLM is instructed to return valid JSON with at least a
        ``"selected"`` key whose value is one of *available_keys*.

        Parameters
        ----------
        prompt : str
            Natural-language description of the task / imaging scenario.
        available_keys : list[str]
            Registry keys the LLM may choose from (e.g. solver ids,
            modality names).

        Returns
        -------
        dict
            Parsed JSON from the LLM, guaranteed to contain ``"selected"``.
            On failure a dict with ``"selected": available_keys[0]`` and an
            ``"error"`` key is returned so callers always get a usable value.
        """
        system_msg = (
            "You are an expert computational-imaging assistant.  "
            "Given the user's task description and the list of available "
            "registry keys, select the single best key.\n\n"
            "Reply ONLY with a JSON object:\n"
            '  {"selected": "<key>", "reason": "<one-line explanation>"}\n\n'
            f"Available keys: {json.dumps(available_keys)}"
        )

        fallback = {"selected": available_keys[0] if available_keys else ""}

        try:
            raw = self._call(system_msg, prompt)
            result = _parse_json(raw)
            if "selected" not in result:
                logger.warning(
                    "LLM response missing 'selected' key; using fallback."
                )
                result["selected"] = fallback["selected"]
            # Validate the selection is actually in the available list
            if result["selected"] not in available_keys and available_keys:
                logger.warning(
                    "LLM selected '%s' which is not in available_keys; "
                    "falling back to '%s'.",
                    result["selected"],
                    fallback["selected"],
                )
                result["selected"] = fallback["selected"]
            return result
        except Exception as exc:
            logger.error("LLM select() failed: %s", exc, exc_info=True)
            fallback["error"] = str(exc)
            return fallback

    def narrate(self, data: Any) -> str:
        """Generate a human-readable narrative explanation of *data*.

        Parameters
        ----------
        data : Any
            Arbitrary data (dict, list, metrics, etc.) to be described.

        Returns
        -------
        str
            Plain-English explanation.  On failure a generic error string
            is returned.
        """
        system_msg = (
            "You are a scientific imaging analyst.  "
            "Given the following reconstruction result data, provide a "
            "clear, concise, human-readable explanation (2-4 sentences) "
            "that a researcher could paste into a lab notebook."
        )
        user_msg = json.dumps(data, indent=2, default=str)

        try:
            return self._call(system_msg, user_msg)
        except Exception as exc:
            logger.error("LLM narrate() failed: %s", exc, exc_info=True)
            return f"[LLM narration unavailable: {exc}]"

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _call(self, system: str, user: str) -> str:
        """Dispatch to the appropriate provider backend.

        Parameters
        ----------
        system : str
            System-level instruction for the LLM.
        user : str
            User-level message / prompt.

        Returns
        -------
        str
            Raw text response from the LLM.
        """
        dispatch = {
            LLMProvider.GEMINI: self._call_gemini,
            LLMProvider.CLAUDE: self._call_claude,
            LLMProvider.OPENAI: self._call_openai,
        }
        handler = dispatch.get(self.provider)
        if handler is None:
            raise ValueError(f"No handler for provider {self.provider}")
        return handler(system, user)

    # ------------------------------------------------------------------
    # Provider backends (all use ``requests``)
    # ------------------------------------------------------------------

    def _call_gemini(self, system: str, user: str) -> str:
        """Call the Google Gemini REST API.

        Endpoint
        --------
        ``generativelanguage.googleapis.com/v1beta/models/{model}:generateContent``
        """
        url = _GEMINI_URL.format(model=self.model)
        params = {"key": self.api_key}
        payload: Dict[str, Any] = {
            "systemInstruction": {
                "parts": [{"text": system}],
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user}],
                }
            ],
            "generationConfig": {
                "maxOutputTokens": _MAX_TOKENS,
                "temperature": 0.2,
            },
        }

        logger.debug("Gemini request: model=%s, url=%s", self.model, url)

        response = requests.post(
            url,
            params=params,
            json=payload,
            timeout=_REQUEST_TIMEOUT_S,
        )
        response.raise_for_status()
        body = response.json()

        try:
            return body["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(
                f"Unexpected Gemini response structure: {body}"
            ) from exc

    def _call_claude(self, system: str, user: str) -> str:
        """Call the Anthropic Messages API.

        Endpoint
        --------
        ``api.anthropic.com/v1/messages``
        """
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": _MAX_TOKENS,
            "temperature": 0.2,
            "system": system,
            "messages": [
                {"role": "user", "content": user},
            ],
        }

        logger.debug("Claude request: model=%s", self.model)

        response = requests.post(
            _CLAUDE_URL,
            headers=headers,
            json=payload,
            timeout=_REQUEST_TIMEOUT_S,
        )
        response.raise_for_status()
        body = response.json()

        try:
            # Anthropic returns content as a list of blocks
            return body["content"][0]["text"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(
                f"Unexpected Claude response structure: {body}"
            ) from exc

    def _call_openai(self, system: str, user: str) -> str:
        """Call the OpenAI Chat Completions API.

        Endpoint
        --------
        ``api.openai.com/v1/chat/completions``
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": _MAX_TOKENS,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        logger.debug("OpenAI request: model=%s", self.model)

        response = requests.post(
            _OPENAI_URL,
            headers=headers,
            json=payload,
            timeout=_REQUEST_TIMEOUT_S,
        )
        response.raise_for_status()
        body = response.json()

        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(
                f"Unexpected OpenAI response structure: {body}"
            ) from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_json(text: str) -> Dict[str, Any]:
    """Extract and parse a JSON object from *text*.

    Handles common LLM quirks:
    - Leading/trailing whitespace or markdown fences
    - JSON embedded inside prose

    Parameters
    ----------
    text : str
        Raw LLM output that should contain a JSON object.

    Returns
    -------
    dict
        Parsed JSON object.

    Raises
    ------
    ValueError
        If no valid JSON object can be extracted.
    """
    cleaned = text.strip()

    # Strip markdown code fences if present
    if cleaned.startswith("```"):
        # Remove opening fence (```json or ```)
        first_newline = cleaned.index("\n") if "\n" in cleaned else 3
        cleaned = cleaned[first_newline + 1 :]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    # Try direct parse first
    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Try to find a JSON object within the text
    brace_start = cleaned.find("{")
    brace_end = cleaned.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        candidate = cleaned[brace_start : brace_end + 1]
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
