"""Base Claude API wrapper shared by all three agents."""
from __future__ import annotations
import json
import anthropic
from typing import Any


class BaseAgent:
    """Thin wrapper around anthropic.Anthropic with streaming + structured output."""

    def __init__(self, model: str = "claude-opus-4-6"):
        self.model  = model
        self.client = anthropic.Anthropic()

    def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 8192,
        thinking: bool = True,
        stream_to_stdout: bool = False,
    ) -> str:
        """Call the API and return the full text response.

        Uses streaming to prevent timeout on long generations.
        Adaptive thinking is enabled by default (Opus 4.6).
        """
        kwargs: dict[str, Any] = {
            "model":      self.model,
            "max_tokens": max_tokens,
            "system":     system,
            "messages":   messages,
        }
        if thinking:
            kwargs["thinking"] = {"type": "adaptive"}

        with self.client.messages.stream(**kwargs) as stream:
            text_parts: list[str] = []
            for event in stream:
                if event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        text_parts.append(event.delta.text)
                        if stream_to_stdout:
                            print(event.delta.text, end="", flush=True)
            if stream_to_stdout:
                print()  # newline after streaming
            return "".join(text_parts)

    def complete_json(
        self,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """Call the API expecting a JSON response wrapped in ```json...``` or plain."""
        text = self.complete(system, messages, max_tokens=max_tokens, thinking=True)
        # Strip markdown code fences if present
        stripped = text.strip()
        if stripped.startswith("```json"):
            stripped = stripped[7:]
        elif stripped.startswith("```"):
            stripped = stripped[3:]
        if stripped.endswith("```"):
            stripped = stripped[:-3]
        return json.loads(stripped.strip())
