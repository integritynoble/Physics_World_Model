"""The real agent loop: OpenCode + a language model on the held-out LDCT problems, common mode vs PWM mode.

    python -m engine.agent_loop --run <engine run dir> --out <dir> \
        [--arms common,pwm] [--state N] [--limit 54] [--timeout 600]

RUN1 (engine/run_engine.py) answered "does the library help?" with a script. This asks it of an agent.
Both arms get the same model, the same prompt core, the same workspace (public data, the runtime's solver,
its adopted defaults in params.json) and the same time limit. The only difference:

  common  OpenCode with the user's own model. No PWM MCP server, no library, 0 PWM.
  pwm     the same, plus the ai4science-pwm MCP server reading the library pinned at --state, on a TEST ledger
          (PWM_MODE_LEDGER under --out; never the production ledger), with a session cap.

Scoring happens outside the agent: validator-ldct-1 reads results/reconstruction.npy against the withheld key.
Nothing the agent says about its own result counts. A missing or malformed file is a fail.

Preflight: the model endpoint must answer. If it does not, the run stops and writes a "blocked" record, and
claims nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pwm_mode import ldct_family as fam          # noqa: E402
from pwm_mode import validator_ldct as val       # noqa: E402
from pwm_mode.ledger import Ledger               # noqa: E402

DEFAULTS = {"sigma_s": 3.0, "sigma_r_scale": 0.3, "iters": 3, "radius": 4}
SOLVER = Path(fam.RUNTIME_SRC) / "ai4science/harness/agents/research_agents/runners/payload/ldct/run_solver.py"
PAYER = "user:agent-loop"
PY = str(ROOT / ".venv" / "bin" / "python")

CORE = """You are solving one real low-dose CT restoration problem. Everything is in this directory.

- problem.json: the question, the lesion conditions (lesion_hu, lesion_r in pixels) and the acceptance criteria.
- data/low_dose.npy: the low-dose slice (HU). The full-dose answer key is withheld; you cannot score yourself.
- code/run_solver.py: a guided bilateral restoration. Run it with `python code/run_solver.py --workspace .`
  It reads params.json (sigma_s, sigma_r_scale, iters, radius) and writes results/reconstruction.npy.
- params.json currently holds the method's adopted defaults.

Deliver results/reconstruction.npy (same shape as data/low_dose.npy, finite, HU). It will be judged afterwards
on all three criteria. You may change params.json and rerun the solver. Finish by stating the parameters you used."""

PWM_EXTRA = """
You also have the ai4science-pwm MCP tools (PWM mode, test ledger). The PWM library holds validated solutions
for problems like this one, each with the conditions it was validated for. Authorize a session cap of 1 PWM with
pwm_authorize, then pwm_search with this problem's conditions, and pwm_get the closest accepted solution to read
its parameters before you choose yours."""


def builtin(model: str) -> bool:
    """A model named "<provider>/<id>" (e.g. openai/gpt-6.1-sol) uses OpenCode's own login for that provider (the
    owner's ChatGPT account, 1 Oct 2026); a bare id uses the OpenAI-compatible --base and --key-file."""
    return "/" in model


def preflight(base: str, key: str, model: str) -> tuple[bool, str]:
    if builtin(model):
        try:
            r = subprocess.run(["opencode", "run", "-m", model, "Reply with exactly: ok"], capture_output=True,
                               text=True, timeout=180, stdin=subprocess.DEVNULL, cwd="/tmp")
            out = (r.stdout or "") + (r.stderr or "")
            return r.returncode == 0 and "ok" in r.stdout.lower(), "exit %s %s" % (r.returncode, out[-120:].replace("\n", " "))
        except Exception as exc:
            return False, "unreachable: %s" % exc
    try:
        r = httpx.post(base.rstrip("/") + "/chat/completions", timeout=120,
                       headers={"Authorization": "Bearer " + key},
                       json={"model": model, "messages": [{"role": "user", "content": "say ok"}], "max_tokens": 50})
        return r.status_code == 200, "HTTP %d %s" % (r.status_code, r.text[:120].replace("\n", " "))
    except Exception as exc:
        return False, "unreachable: %s" % exc


def config(arm: str, ws: Path, a, ledger_path: Path) -> dict:
    c = {"$schema": "https://opencode.ai/config.json", "model": run_model(a.model),
         "permission": {"edit": "allow", "bash": "allow", "webfetch": "deny"}, "share": "disabled"}
    if not builtin(a.model):
        c["provider"] = {"own": {"npm": "@ai-sdk/openai-compatible", "name": "own model",
                                 "options": {"baseURL": a.base, "apiKey": "{file:%s}" % a.key_file},
                                 "models": {a.model: {"name": a.model}}}}
    if arm == "pwm":
        c["mcp"] = {"ai4science-pwm": {
            "type": "local", "command": [PY, "-m", "pwm_mode.mcp_server"], "enabled": True,
            "environment": {"PYTHONPATH": str(ROOT), "PWM_MODE_PAYER": PAYER, "PWM_MODE_WORKSPACE": str(ws),
                            "PWM_MODE_HOME": str(Path(a.run)), "PWM_LIBRARY_STATE": str(a.state),
                            "PWM_MODE_LEDGER": str(ledger_path),
                            "AI4SCIENCE_DATA": os.environ.get("AI4SCIENCE_DATA", "/home/spiritai/.ai4science/data")}}}
    return c


def run_model(model: str) -> str:
    return model if builtin(model) else "own/" + model


def workspace(arm: str, pid: str, a, ledger_path: Path) -> Path:
    ws = Path(a.out) / arm / pid
    if ws.exists():
        shutil.rmtree(ws)
    shutil.copytree(Path(a.run) / "problems" / "public" / pid, ws)
    (ws / "code").mkdir()
    shutil.copy2(SOLVER, ws / "code" / "run_solver.py")
    (ws / "params.json").write_text(json.dumps(DEFAULTS, indent=1))
    (ws / "opencode.json").write_text(json.dumps(config(arm, ws, a, ledger_path), indent=1))
    return ws


def tool_calls(events_path: Path) -> list:
    calls = []
    for line in events_path.read_text(errors="replace").splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        part = ev.get("part") or {}
        if part.get("type") == "tool":
            calls.append({"tool": part.get("tool"), "status": (part.get("state") or {}).get("status")})
    return calls


def score(pid: str, ws: Path, a) -> dict:
    f = ws / "results" / "reconstruction.npy"
    if not f.exists():
        return {"passed": False, "why": "no results/reconstruction.npy"}
    try:
        rec = np.load(f, allow_pickle=False)
        v = val.validate(Path(a.run) / "problems" / "keys" / pid, rec)
    except Exception as exc:
        return {"passed": False, "why": "invalid deliverable: %s" % exc}
    params = json.loads((ws / "params.json").read_text()) if (ws / "params.json").exists() else None
    return {"passed": v["passed"], "metrics": v["metrics"], "params_file": params}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="engine run dir (problems/, library/, split.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="common,pwm")
    ap.add_argument("--state", type=int, default=None, help="library state the PWM arm reads (default: latest)")
    ap.add_argument("--limit", type=int, default=54)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--base", default="https://physicsworldmodel.org/qwen/v1")
    ap.add_argument("--key-file", default="/srv/fleet/qwen-workers-key")
    ap.add_argument("--model", default="qwen3.8:27b",
                    help='a bare id for --base, or "<provider>/<id>" for OpenCode\'s own login (openai/gpt-6.1-sol)')
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec_path = out / "agent_loop.jsonl"
    ok, why = preflight(a.base, "" if builtin(a.model) else Path(a.key_file).read_text().strip(), a.model)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if not ok:
        rec = {"at": stamp, "status": "blocked", "reason": "model endpoint not available", "probe": why,
               "model": a.model, "base": a.base}
        with open(rec_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps(rec))
        return 2

    if a.state is None:
        from pwm_mode.library import Library
        a.state = Library(Path(a.run) / "library").head()
    ledger_path = out / "test-ledger.sqlite"          # a test ledger for this run only
    L = Ledger(ledger_path)
    held = json.loads((Path(a.run) / "split.json").read_text())["held_out"]
    pids = sorted(p.name for p in (Path(a.run) / "problems" / "public").iterdir()
                  if json.loads((p / "problem.json").read_text())["patient"] in held)[: a.limit]
    for pid in pids:
        for arm in a.arms.split(","):
            if arm == "pwm":
                L.fund(PAYER, "1", memo="agent loop: one problem's cap")
            before = L.balance_micro(PAYER)
            ws = workspace(arm, pid, a, ledger_path)
            prompt = CORE + (PWM_EXTRA if arm == "pwm" else "")
            t0 = time.time()
            events = ws / "opencode-events.jsonl"
            with open(events, "w") as fh:
                try:
                    code = subprocess.run(["opencode", "run", "--format", "json", "-m", run_model(a.model), prompt],
                                          cwd=ws, stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT,
                                          timeout=a.timeout, env=dict(os.environ)).returncode
                except subprocess.TimeoutExpired:
                    code = "timeout"          # scored as delivered so far; usually no file, so a fail
            row = {"at": stamp, "problem": pid, "arm": arm, "model": a.model,
                   "library_state": f"L{a.state}" if arm == "pwm" else None,
                   "exit": code, "seconds": round(time.time() - t0, 1),
                   "tool_calls": tool_calls(events), **score(pid, ws, a),
                   "pwm_spent": (before - L.balance_micro(PAYER)) / 1e6}
            with open(rec_path, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(json.dumps({k: row[k] for k in ("problem", "arm", "passed", "seconds", "pwm_spent")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
