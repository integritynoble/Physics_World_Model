import hashlib
import json

import pytest
from starlette.testclient import TestClient

KEY = "pwmt_testkey"


@pytest.fixture
def px(tmp_path, monkeypatch):
    monkeypatch.setenv("PWM_MODE_LEDGER", str(tmp_path / "ledger.sqlite"))
    from pwm_mode import proxy as p
    keys = tmp_path / "keys.json"
    keys.write_text(json.dumps({hashlib.sha256(KEY.encode()).hexdigest(): "user:u"}))
    monkeypatch.setattr(p, "KEYS_FILE", keys)
    monkeypatch.setattr(p, "_ledger", None)
    p.ledger().fund("user:u", "10")
    return p


def fake(usage=(1000, 500), fail=False):
    def call(body):
        if fail:
            raise RuntimeError("503 overloaded")
        return ({"id": "x", "object": "chat.completion", "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}]},
                {"input": usage[0], "output": usage[1], "reported": True})

    def stream(body, u):
        if fail:
            raise RuntimeError("stream broke")
        yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        u.update(input=usage[0], output=usage[1], reported=True)
    return call, stream


H = {"Authorization": "Bearer " + KEY}
BODY = {"model": "claude-opus-5-5", "messages": [{"role": "user", "content": "hello"}]}


def test_opus_turn_settles_at_metered_usage(px, monkeypatch):
    monkeypatch.setitem(px.UPSTREAMS, "anthropic", fake((1000, 500)))
    r = TestClient(px.app).post("/v1/chat/completions", json=BODY, headers=H)
    assert r.status_code == 200
    rec = r.json()["pwm_receipt"]
    # 1000 * 0.8/M + 500 * 4.0/M = 0.0028 PWM
    assert rec["status"] == "settled" and rec["charged_pwm"] == "0.0028"
    assert {l["account"] for l in rec["lines"]} == {"provider:test-anthropic-route", "treasury"}
    assert px.ledger().balance("user:u")["pwm"] == "9.9972"
    assert px.ledger().books_balance() == 0


def test_failed_turn_not_charged(px, monkeypatch):
    monkeypatch.setitem(px.UPSTREAMS, "anthropic", fake(fail=True))
    r = TestClient(px.app).post("/v1/chat/completions", json=BODY, headers=H)
    assert r.status_code == 502
    assert r.json()["error"]["receipt"]["charged_pwm"] == "0"
    assert px.ledger().balance("user:u")["pwm"] == "10"


def test_stream_settles_and_emits_receipt(px, monkeypatch):
    monkeypatch.setitem(px.UPSTREAMS, "fleet-qwen", fake((2_000_000, 1_000_000)))
    r = TestClient(px.app).post("/v1/chat/completions", headers=dict(H, **{"X-PWM-Cap": "1"}),
                                json=dict(BODY, model="qwen3.8:27b", stream=True))
    text = r.text
    assert ": pwm-receipt " in text and text.rstrip().endswith("data: [DONE]")
    rec = json.loads(text.split(": pwm-receipt ", 1)[1].split("\n", 1)[0])
    assert rec["charged_pwm"] == "0.24"          # 2M*0.04 + 1M*0.16


def test_stream_failure_not_charged(px, monkeypatch):
    monkeypatch.setitem(px.UPSTREAMS, "fleet-qwen", fake(fail=True))
    r = TestClient(px.app).post("/v1/chat/completions", headers=H,
                                json=dict(BODY, model="qwen3.8:27b", stream=True))
    rec = json.loads(r.text.split(": pwm-receipt ", 1)[1].split("\n", 1)[0])
    assert rec["status"] == "released" and rec["charged_pwm"] == "0"


def test_cap_bounds_the_charge(px, monkeypatch):
    monkeypatch.setitem(px.UPSTREAMS, "anthropic", fake((10_000_000, 1_000_000)))   # 12 PWM of usage
    r = TestClient(px.app).post("/v1/chat/completions", json=BODY,
                                headers=dict(H, **{"X-PWM-Cap": "0.5"}))
    rec = r.json()["pwm_receipt"]
    assert rec["charged_pwm"] == "0.5" and rec["over_cap_pwm"] == "11.5"


def test_unknown_model_refused_no_key_refused_replay_not_redispatched(px, monkeypatch):
    c = TestClient(px.app)
    assert c.post("/v1/chat/completions", json=dict(BODY, model="gpt-own"), headers=H).status_code == 404
    assert c.post("/v1/chat/completions", json=BODY).status_code == 401
    calls = []
    call, stream = fake()
    monkeypatch.setitem(px.UPSTREAMS, "anthropic", (lambda b: (calls.append(1), call(b))[1], stream))
    h = dict(H, **{"X-Request-Id": "same-1"})
    assert c.post("/v1/chat/completions", json=BODY, headers=h).status_code == 200
    again = c.post("/v1/chat/completions", json=BODY, headers=h)
    assert again.status_code == 409 and again.json()["replay"] and len(calls) == 1


def test_insufficient_balance_is_402(px):
    r = TestClient(px.app).post("/v1/chat/completions", json=BODY,
                                headers=dict(H, **{"X-PWM-Cap": "50"}))
    assert r.status_code == 402


def test_translation_to_messages_api(px):
    body = {"model": "claude-opus-5-5", "tool_choice": "required", "messages": [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "list files"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "toolu_1", "type": "function", "function": {"name": "ls", "arguments": "{\"p\":\".\"}"}}]},
        {"role": "tool", "tool_call_id": "toolu_1", "content": "a.py"},
        {"role": "user", "content": "thanks"}],
        "tools": [{"type": "function", "function": {"name": "ls", "parameters": {"type": "object"}}}]}
    a = px.to_anthropic(body)
    assert a["system"] == "be brief" and a["tool_choice"] == {"type": "auto"}
    assert [m["role"] for m in a["messages"]] == ["user", "assistant", "user"]
    assert a["messages"][1]["content"][0] == {"type": "tool_use", "id": "toolu_1", "name": "ls",
                                              "input": {"p": "."}}
    assert a["messages"][2]["content"][0]["type"] == "tool_result"   # tool result + next user merged
    assert a["messages"][2]["content"][1] == {"type": "text", "text": "thanks"}
    # a remembered assistant turn is restored verbatim (thinking blocks included)
    px._TURNS[px._turn_key("", ["toolu_1"])] = [{"type": "thinking", "thinking": "", "signature": "s"},
                                                 {"type": "tool_use", "id": "toolu_1", "name": "ls",
                                                  "input": {"p": "."}}]
    assert px.to_anthropic(body)["messages"][1]["content"][0]["type"] == "thinking"
