"""The PWM model proxy: an OpenAI-compatible provider for AI4Science's models, paid in PWM.

OpenCode adds it as one provider (baseURL http://127.0.0.1:8795/v1). Each chat completion
is one run on the test ledger: a quote whose cap is the turn's ceiling, a reservation,
the upstream call, then settlement at metered usage (never above the cap, the shortfall
recorded as over_cap, as the platform's F02 proxy does) to the model provider with 3% to
the treasury. An upstream error, or a turn with no reported usage, is released: 0 PWM.

A model the catalogue does not list is refused, never substituted. The user's own LLM is
configured directly in OpenCode and never passes through here, so it costs 0 PWM.

Upstreams:
  anthropic   Claude (e.g. claude-opus-5-5) through the official Anthropic SDK; the
              OpenAI request is translated to the Messages API and back.
  fleet-qwen  the fleet's own OpenAI-compatible Qwen endpoint, passed through.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from . import catalog
from .ledger import Ledger, LedgerError, micro, pwm

DEFAULT_TURN_CAP = os.environ.get("PWM_MODE_TURN_CAP_PWM", "0.5")
KEYS_FILE = Path(os.environ.get("PWM_MODE_KEYS", catalog.ROOT / "config" / "test_keys.json"))

_ledger: Optional[Ledger] = None


def ledger() -> Ledger:
    global _ledger
    if _ledger is None:
        _ledger = Ledger(catalog.ledger_path())
    return _ledger


def payer_for(request: Request) -> Optional[str]:
    auth = request.headers.get("authorization", "")
    key = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    if not key or not KEYS_FILE.exists():
        return None
    keys = json.loads(KEYS_FILE.read_text())
    return keys.get(hashlib.sha256(key.encode()).hexdigest())


def err(status: int, message: str, **extra) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": "pwm_mode", **extra}}, status)


# ---- upstream: fleet Qwen (OpenAI-compatible passthrough) -------------------------------

def _qwen_conf() -> Tuple[str, str]:
    base = os.environ.get("PWM_MODE_QWEN_BASE", "https://physicsworldmodel.org/qwen/v1")
    key = Path(os.environ.get("PWM_MODE_QWEN_KEY_FILE", "/srv/fleet/qwen-workers-key")).read_text().strip()
    return base, key


def qwen_call(body: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, int]]:
    base, key = _qwen_conf()
    r = httpx.post(base + "/chat/completions", json=dict(body, stream=False),
                   headers={"Authorization": "Bearer " + key}, timeout=600)
    if r.status_code >= 400:
        raise RuntimeError("upstream HTTP %d: %s" % (r.status_code, r.text[:300]))
    data = r.json()
    u = data.get("usage") or {}
    return data, {"input": int(u.get("prompt_tokens") or 0), "output": int(u.get("completion_tokens") or 0),
                  "reported": bool(u)}


def qwen_stream(body: Dict[str, Any], usage: Dict[str, Any]) -> Iterator[bytes]:
    base, key = _qwen_conf()
    body = dict(body, stream=True, stream_options={"include_usage": True})
    with httpx.stream("POST", base + "/chat/completions", json=body,
                      headers={"Authorization": "Bearer " + key}, timeout=600) as r:
        if r.status_code >= 400:
            raise RuntimeError("upstream HTTP %d: %s" % (r.status_code, r.read()[:300]))
        for line in r.iter_lines():
            if line.startswith("data: ") and line[6:].strip() not in ("", "[DONE]"):
                try:
                    u = json.loads(line[6:]).get("usage")
                except ValueError:
                    u = None
                if u:
                    usage.update(input=int(u.get("prompt_tokens") or 0),
                                 output=int(u.get("completion_tokens") or 0), reported=True)
            if line.strip() == "data: [DONE]":
                continue                      # the proxy sends its own [DONE] after the receipt
            if line:
                yield (line + "\n\n").encode()


# ---- upstream: Anthropic (OpenAI <-> Messages translation) ------------------------------

# Opus 5.5 ties thinking blocks to the conversation. OpenAI-shaped history drops them, so the
# proxy remembers each assistant turn's original content blocks, keyed by what the client
# will send back (its tool-call ids, or its text), and restores them verbatim.
_TURNS: Dict[str, List[Dict[str, Any]]] = {}


def _turn_key(text: str, tool_ids: List[str]) -> str:
    return hashlib.sha256(json.dumps([text or "", sorted(tool_ids)]).encode()).hexdigest()


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content or [] if isinstance(p, dict)
                   and p.get("type") in ("text", "input_text"))


def to_anthropic(body: Dict[str, Any]) -> Dict[str, Any]:
    system: List[str] = []
    msgs: List[Dict[str, Any]] = []

    def push(role, blocks):
        if msgs and msgs[-1]["role"] == role:
            msgs[-1]["content"].extend(blocks)
        else:
            msgs.append({"role": role, "content": list(blocks)})

    for m in body.get("messages", []):
        role = m.get("role")
        if role in ("system", "developer"):
            system.append(_text_of(m.get("content")))
        elif role == "user":
            c = m.get("content")
            blocks = [{"type": "text", "text": c}] if isinstance(c, str) else [
                {"type": "text", "text": p["text"]} for p in c or [] if p.get("type") == "text"]
            push("user", [b for b in blocks if b["text"]] or [{"type": "text", "text": "."}])
        elif role == "assistant":
            calls = m.get("tool_calls") or []
            text = _text_of(m.get("content"))
            saved = _TURNS.get(_turn_key(text, [c["id"] for c in calls]))
            if saved:
                push("assistant", saved)
                continue
            blocks = [{"type": "text", "text": text}] if text else []
            for c in calls:
                try:
                    args = json.loads(c["function"].get("arguments") or "{}")
                except ValueError:
                    args = {}
                blocks.append({"type": "tool_use", "id": c["id"], "name": c["function"]["name"],
                               "input": args})
            if blocks:
                push("assistant", blocks)
        elif role == "tool":
            push("user", [{"type": "tool_result", "tool_use_id": m["tool_call_id"],
                           "content": _text_of(m.get("content")) or "(empty)"}])
    req: Dict[str, Any] = {"model": body["model"], "messages": msgs,
                           "max_tokens": int(body.get("max_tokens") or body.get("max_completion_tokens")
                                             or 32000)}
    if system:
        req["system"] = "\n\n".join(s for s in system if s)
    tools = [{"name": t["function"]["name"], "description": t["function"].get("description", ""),
              "input_schema": t["function"].get("parameters") or {"type": "object", "properties": {}}}
             for t in body.get("tools") or [] if t.get("type") == "function"]
    if tools:
        req["tools"] = tools
        # Opus 5.5 refuses forced tool choice; "required"/named choices degrade to auto.
        req["tool_choice"] = {"type": "none"} if body.get("tool_choice") == "none" else {"type": "auto"}
    effort = body.get("reasoning_effort")
    if effort in ("low", "medium", "high", "xhigh", "max"):
        req["output_config"] = {"effort": effort}
    return req


_STOP = {"end_turn": "stop", "tool_use": "tool_calls", "max_tokens": "length",
         "stop_sequence": "stop", "refusal": "content_filter", "pause_turn": "stop"}


def _remember(final) -> Tuple[str, List[Dict[str, Any]]]:
    blocks = [b.model_dump(exclude_none=True) for b in final.content]
    text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    calls = [b for b in blocks if b.get("type") == "tool_use"]
    _TURNS[_turn_key(text, [c["id"] for c in calls])] = blocks
    return text, calls


def _client():
    import anthropic
    return anthropic.Anthropic()


def anthropic_call(body: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, int]]:
    with _client().messages.stream(**to_anthropic(body)) as s:
        final = s.get_final_message()
    text, calls = _remember(final)
    msg: Dict[str, Any] = {"role": "assistant", "content": text or None}
    if calls:
        msg["tool_calls"] = [{"id": c["id"], "type": "function",
                              "function": {"name": c["name"], "arguments": json.dumps(c["input"])}}
                             for c in calls]
    u = final.usage
    usage = {"input": u.input_tokens + (u.cache_read_input_tokens or 0)
             + (u.cache_creation_input_tokens or 0), "output": u.output_tokens, "reported": True}
    return {"id": "chatcmpl-" + final.id, "object": "chat.completion", "created": int(time.time()),
            "model": body["model"],
            "choices": [{"index": 0, "message": msg,
                         "finish_reason": _STOP.get(final.stop_reason, "stop")}],
            "usage": {"prompt_tokens": usage["input"], "completion_tokens": usage["output"],
                      "total_tokens": usage["input"] + usage["output"]}}, usage


def anthropic_stream(body: Dict[str, Any], usage: Dict[str, Any]) -> Iterator[bytes]:
    cid, created, model = "chatcmpl-" + uuid.uuid4().hex, int(time.time()), body["model"]

    def chunk(delta, finish=None):
        return ("data: " + json.dumps({"id": cid, "object": "chat.completion.chunk", "created": created,
                                       "model": model, "choices": [{"index": 0, "delta": delta,
                                                                    "finish_reason": finish}]})
                + "\n\n").encode()

    yield chunk({"role": "assistant"})
    tool_index: Dict[int, int] = {}
    with _client().messages.stream(**to_anthropic(body)) as s:
        for ev in s:
            if ev.type == "content_block_start" and ev.content_block.type == "tool_use":
                tool_index[ev.index] = len(tool_index)
                yield chunk({"tool_calls": [{"index": tool_index[ev.index], "id": ev.content_block.id,
                                             "type": "function",
                                             "function": {"name": ev.content_block.name, "arguments": ""}}]})
            elif ev.type == "content_block_delta":
                if ev.delta.type == "text_delta":
                    yield chunk({"content": ev.delta.text})
                elif ev.delta.type == "input_json_delta" and ev.index in tool_index:
                    yield chunk({"tool_calls": [{"index": tool_index[ev.index],
                                                 "function": {"arguments": ev.delta.partial_json}}]})
        final = s.get_final_message()
    _remember(final)
    u = final.usage
    usage.update(input=u.input_tokens + (u.cache_read_input_tokens or 0)
                 + (u.cache_creation_input_tokens or 0), output=u.output_tokens, reported=True)
    yield chunk({}, _STOP.get(final.stop_reason, "stop"))
    yield ("data: " + json.dumps({"id": cid, "object": "chat.completion.chunk", "created": created,
                                  "model": model, "choices": [],
                                  "usage": {"prompt_tokens": usage["input"],
                                            "completion_tokens": usage["output"],
                                            "total_tokens": usage["input"] + usage["output"]}})
           + "\n\n").encode()


UPSTREAMS = {"anthropic": (anthropic_call, anthropic_stream), "fleet-qwen": (qwen_call, qwen_stream)}


# ---- metering ---------------------------------------------------------------------------

def _close(run_id: str, m: Dict[str, Any], usage: Dict[str, Any], error: Optional[str]) -> Dict[str, Any]:
    L = ledger()
    if error or not usage.get("reported"):
        return L.release(run_id, "failed: %s" % (error or "no usage reported by the upstream"))
    cost = catalog.model_cost_pwm(m, usage["input"], usage["output"])
    return L.settle(run_id, gross=str(cost), usage={"input_tokens": usage["input"],
                                                    "output_tokens": usage["output"],
                                                    "model": m["id"]})


async def chat(request: Request):
    payer = payer_for(request)
    if payer is None:
        return err(401, "PWM mode needs a PWM key (test ledger)")
    body = await request.json()
    try:
        m = catalog.model(body.get("model", ""))
    except KeyError:
        return err(404, "model %r is not offered through AI4Science; use your own provider for "
                        "it (free), or pick one from /v1/models" % body.get("model"))
    cap = request.headers.get("x-pwm-cap") or DEFAULT_TURN_CAP
    run_id = request.headers.get("x-request-id") or "t_" + uuid.uuid4().hex[:16]
    L = ledger()
    try:
        prior = L.db.execute("SELECT status FROM runs WHERE id=?", (run_id,)).fetchone()
        if prior:                                   # idempotent: a replay is never re-dispatched
            return JSONResponse({"replay": True, "receipt": L.receipt(run_id)}, 409)
        q = L.quote(payer, catalog.model_service(m, cap), cap=cap)
        L.reserve(run_id, q["quote_id"])
    except LedgerError as exc:
        return err(402 if exc.code == "insufficient_pwm" else 409, str(exc), code=exc.code)
    call, stream = UPSTREAMS[m["upstream"]]
    headers = {"X-PWM-Run-Id": run_id, "X-PWM-Cap": str(cap), "X-PWM-Network": L.network}

    if not body.get("stream"):
        usage: Dict[str, Any] = {}
        try:
            data, usage = call(body)
        except Exception as exc:
            receipt = _close(run_id, m, usage, str(exc)[:500])
            return err(502, "upstream failed; not charged", receipt=receipt)
        data["pwm_receipt"] = _close(run_id, m, usage, None)
        return JSONResponse(data, headers=headers)

    def gen():
        usage: Dict[str, Any] = {}
        error = None
        try:
            yield from stream(body, usage)
        except GeneratorExit:
            error = "client disconnected before the turn completed"
            raise
        except Exception as exc:
            error = str(exc)[:500]
            yield ("data: " + json.dumps({"error": {"message": "upstream failed; not charged: "
                                                   + error}}) + "\n\n").encode()
        finally:
            receipt = _close(run_id, m, usage, error)
        # SSE comment: ignored by OpenAI clients, readable by anyone auditing the stream.
        yield (": pwm-receipt " + json.dumps(receipt) + "\n\n").encode()
        yield b"data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)


async def models(request: Request):
    return JSONResponse({"object": "list", "data": [
        {"id": m["id"], "object": "model", "owned_by": "ai4science-pwm",
         "pwm_per_million_input": m["pwm_per_million_input"],
         "pwm_per_million_output": m["pwm_per_million_output"]} for m in catalog.load()["models"]]})


async def receipt(request: Request):
    payer = payer_for(request)
    try:
        r = ledger().receipt(request.path_params["run_id"])
    except LedgerError:
        return err(404, "unknown run")
    if payer is None or r.get("payer", payer) != payer:
        return err(404, "unknown run")
    return JSONResponse(r)


async def balance(request: Request):
    payer = payer_for(request)
    if payer is None:
        return err(401, "PWM key required")
    return JSONResponse(ledger().balance(payer))


app = Starlette(routes=[
    Route("/v1/chat/completions", chat, methods=["POST"]),
    Route("/v1/models", models),
    Route("/pwm/receipts/{run_id}", receipt),
    Route("/pwm/balance", balance),
])


def main():
    import uvicorn
    uvicorn.run(app, host=os.environ.get("PWM_MODE_PROXY_HOST", "127.0.0.1"),
                port=int(os.environ.get("PWM_MODE_PROXY_PORT", "8795")))


if __name__ == "__main__":
    main()
