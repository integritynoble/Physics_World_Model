"""Harness mechanics only (no model): workspaces, configs, outside scoring, the blocked record."""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine import agent_loop as al

RUN = Path.home() / ".ai4science-pwm-mode" / "run1-repro"
pytestmark = pytest.mark.skipif(not (RUN / "split.json").exists(), reason="needs a local engine run")


def _args(tmp_path, **kw):
    return SimpleNamespace(run=str(RUN), out=str(tmp_path), state=8, model="m", base="http://x/v1",
                           key_file="/dev/null", **kw)


def test_arms_differ_only_by_the_pwm_server(tmp_path):
    a = _args(tmp_path)
    pid = "ldct-c052-s0-c150-r3"
    c = json.loads((al.workspace("common", pid, a, tmp_path / "l.sqlite") / "opencode.json").read_text())
    p = json.loads((al.workspace("pwm", pid, a, tmp_path / "l.sqlite") / "opencode.json").read_text())
    assert "mcp" not in c
    env = p.pop("mcp")["ai4science-pwm"]["environment"]
    assert c == p                                              # same model, provider, permissions
    assert env["PWM_LIBRARY_STATE"] == "8" and env["PWM_MODE_LEDGER"] == str(tmp_path / "l.sqlite")
    ws = tmp_path / "common" / pid
    assert not any((ws / "data").glob("full_dose*")) and not (ws / "lesion_mask.npy").exists()   # key withheld


def test_outside_scoring_matches_run1_for_the_defaults(tmp_path):
    a = _args(tmp_path)
    results = json.loads((RUN / "results.json").read_text())["common"]
    for row in results[:4]:
        ws = al.workspace("common", row["problem"], a, tmp_path / "l.sqlite")
        subprocess.run([sys.executable, "code/run_solver.py", "--workspace", "."], cwd=ws, check=True)
        s = al.score(row["problem"], ws, a)
        assert s["passed"] == row["passed"]
        assert abs(s["metrics"]["lesion_cnr"] - row["metrics"]["lesion_cnr"]) < 1e-6


def test_missing_deliverable_is_a_fail(tmp_path):
    a = _args(tmp_path)
    ws = al.workspace("common", "ldct-c052-s0-c150-r3", a, tmp_path / "l.sqlite")
    assert al.score("ldct-c052-s0-c150-r3", ws, a) == {"passed": False, "why": "no results/reconstruction.npy"}


def test_unreachable_model_writes_blocked_and_claims_nothing(tmp_path):
    code = al.main(["--run", str(RUN), "--out", str(tmp_path), "--base", "http://127.0.0.1:9/v1",
                    "--key-file", "/dev/null"])
    rec = [json.loads(l) for l in (tmp_path / "agent_loop.jsonl").read_text().splitlines()]
    assert code == 2 and rec[-1]["status"] == "blocked" and len(rec) == 1
    assert not (tmp_path / "common").exists()


def test_builtin_login_model_has_no_provider_block_and_arms_differ_only_by_the_mcp_server(tmp_path):
    from types import SimpleNamespace
    from engine import agent_loop as al
    a = SimpleNamespace(model="openai/gpt-6.1-sol", base="unused", key_file="unused", run=str(tmp_path), state=3)
    common = al.config("common", tmp_path, a, tmp_path / "l.sqlite")
    pwm = al.config("pwm", tmp_path, a, tmp_path / "l.sqlite")
    assert common["model"] == "openai/gpt-6.1-sol" and "provider" not in common
    assert {k: v for k, v in pwm.items() if k != "mcp"} == common and "ai4science-pwm" in pwm["mcp"]
    assert al.run_model("qwen3.8:27b") == "own/qwen3.8:27b" and al.run_model(a.model) == a.model
    own = al.config("common", tmp_path, SimpleNamespace(**{**vars(a), "model": "qwen3.8:27b"}), tmp_path / "l")
    assert own["model"] == "own/qwen3.8:27b" and "own" in own["provider"]
