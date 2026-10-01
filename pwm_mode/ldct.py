"""Low-dose CT services, built on the AI4Science runtime's own benchmark.

Reused unchanged from the runtime (imported, not copied):
  runners/domains.py  LDCT benchmark, _score_ldct, _judge_ldct (fidelity + Rose-criterion
                      detectability, reported together)
  runners/common.py   seed_workspace, run_domain_task (seed -> stage minus the key ->
                      execute -> score from outside)
The only new piece is LocalSandbox, a stand-in for the control-plane client that runs
the solver in a fresh directory holding everything except the answer key.

The answer key lives under the server's own store and never enters a user workspace.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

RUNTIME_SRC = os.environ.get("AI4SCIENCE_SRC", "/home/spiritai/pwm/Physics_World_Model/AI4Science")
os.environ.setdefault("AI4SCIENCE_DATA", "/home/spiritai/.ai4science/data")
if RUNTIME_SRC not in sys.path:
    sys.path.insert(0, RUNTIME_SRC)

from ai4science.harness.agents.research_agents.runners import common, domains  # noqa: E402

BENCH = domains.LDCT


def store_root() -> Path:
    root = Path(os.environ.get("PWM_MODE_HOME", Path.home() / ".ai4science-pwm-mode")) / "ldct"
    root.mkdir(parents=True, exist_ok=True)
    return root


class LocalSandbox:
    """Control-plane stand-in: open_run / stage_input / sandbox_execute."""

    def __init__(self, root: Path, timeout_s: int = 900):
        self.root, self.timeout_s, self.runs = Path(root), timeout_s, {}

    def open_run(self, goal, profile, budget, **kw):
        rid = uuid.uuid4().hex[:12]
        ws = self.root / rid
        ws.mkdir(parents=True)
        self.runs[rid] = ws
        return {"run_id": rid, "workspace_path": str(ws)}

    def stage_input(self, rid, rel, data: bytes):
        p = self.runs[rid] / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def sandbox_execute(self, rid, argv):
        argv = [sys.executable] + list(argv[1:]) if argv and argv[0] == "python3" else argv
        try:
            out = subprocess.run(argv, cwd=self.runs[rid], capture_output=True, text=True,
                                 timeout=self.timeout_s, env=common.child_env())
        except subprocess.TimeoutExpired:
            return {"is_error": True, "stderr": "solver timed out after %ds" % self.timeout_s}
        return {"is_error": out.returncode != 0, "stdout": out.stdout, "stderr": out.stderr}


def describe() -> Dict[str, Any]:
    return {"agent": BENCH.agent, "goal": BENCH.goal, "criteria": list(BENCH.criteria),
            "parameters": [{"name": p.name, "low": p.low, "high": p.high, "default": p.default,
                            "means": p.means} for p in BENCH.parameters],
            "seeds": list(BENCH.usable_seeds), "real_data": BENCH.provenance()}


def _check_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in BENCH.usable_seeds:
        raise ValueError("seed must be one of %s (four distinct patients)" % (BENCH.usable_seeds,))
    return seed


def solve(*, seed: int = 0, params: Optional[Dict[str, float]] = None,
          fail_injection: bool = False) -> Dict[str, Any]:
    """The LDCT agent's end-to-end run: the runtime's run_domain_task.
    Returns status delivered (judge passed) | rejected (judge failed) | failed."""
    seed = _check_seed(seed)
    job = store_root() / "solve" / uuid.uuid4().hex[:12]
    client = LocalSandbox(job / "runs")
    if fail_injection:        # test hook: a solver that crashes, to prove failures cost 0
        client.sandbox_execute = lambda rid, argv: {"is_error": True,
                                                    "stderr": "injected solver failure (test)"}
    r = common.run_domain_task(BENCH, client=client, workspace=job / "seed", seed=seed,
                               params=params, agent_id="low-dose-ct")
    v = r.get("verdict")
    out = {k: r.get(k) for k in ("status", "seed", "params", "metrics", "withheld", "why",
                                 "criteria")}
    out["job"] = job.name
    if v is not None:
        out["verdict"] = {"passed": bool(v.passed), "reasons": list(v.reasons),
                          "report": v.report()}
    out["artifact"] = str(Path(r["run_workspace"]) / BENCH.deliverables[0]) \
        if r.get("status") in ("delivered", "rejected") else None
    return out


def problem(*, seed: int = 0, dest: Path) -> Dict[str, Any]:
    """Give the user a real problem to solve themselves: data minus the answer key.
    The key stays in the server store under the returned problem_id."""
    seed = _check_seed(seed)
    pid = "ldct-%d-%s" % (seed, uuid.uuid4().hex[:10])
    key_ws = store_root() / "problems" / pid
    meta = common.seed_workspace(BENCH, key_ws, seed=seed)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for p in sorted(key_ws.rglob("*")):
        rel = p.relative_to(key_ws).as_posix()
        if p.is_file() and rel not in BENCH.answer_key:
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest / rel)
            copied.append(rel)
    (key_ws / "problem.json").write_text(json.dumps({"seed": seed, "meta": meta}))
    return {"problem_id": pid, "seed": seed, "workspace": str(dest), "files": copied,
            "withheld": list(BENCH.answer_key),
            "deliverable": "write your restored slice to results/reconstruction.npy "
                           "(same shape as data/low_dose.npy, HU), then call ldct_judge"}


def judge(*, problem_id: str, reconstruction: Path) -> Dict[str, Any]:
    """Score a user's reconstruction against the withheld key with the runtime's judge.
    Raises ValueError (no verdict, so nothing to charge) on a malformed submission."""
    key_ws = store_root() / "problems" / problem_id
    if not (key_ws / "problem.json").exists() or "/" in problem_id:
        raise ValueError("unknown problem_id %r" % problem_id)
    low = np.load(key_ws / "data" / "low_dose.npy")
    try:
        rec = np.load(Path(reconstruction), allow_pickle=False)
    except Exception as exc:
        raise ValueError("cannot read reconstruction: %s" % exc)
    if rec.shape != low.shape or not np.isfinite(rec).all():
        raise ValueError("reconstruction must be finite with shape %s (got %s)"
                         % (low.shape, rec.shape))
    run_ws = store_root() / "judged" / uuid.uuid4().hex[:12]
    (run_ws / "data").mkdir(parents=True)
    (run_ws / "results").mkdir()
    np.save(run_ws / "data" / "low_dose.npy", low)
    np.save(run_ws / "results" / "reconstruction.npy", rec.astype(np.float32))
    m = BENCH.score(key_ws, run_ws)
    v = BENCH.judge(m)
    return {"problem_id": problem_id, "passed": bool(v.passed), "metrics": m,
            "reasons": list(v.reasons), "criteria": list(BENCH.criteria)}
