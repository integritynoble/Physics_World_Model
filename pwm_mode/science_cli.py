"""Command-line tools for science tasks in PWM mode: make a problem, validate a result.

    python -m pwm_mode.science_cli make --field ldct --patient C052 --slice 1 --hu 150 --r 5 --into <work folder>
    python -m pwm_mode.science_cli validate --field ldct --problem <problem_id> --reconstruction <path> [--json <out>]

``make`` writes the answer key into the protected store (PWM_MODE_HOME/science/<field>/keys) and only the public part
(the low-dose slice and problem.json) into the worker's folder. ``validate`` is the done check a science task's goal
names as ``command:``: it runs the field's independent validator against the protected key and exits 0 on a pass,
1 on a fail, 2 when there is nothing valid to judge. It prints one JSON line, the evidence.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np


def store(field: str) -> Path:
    root = Path(os.environ.get("PWM_MODE_HOME", Path.home() / ".ai4science-pwm-mode")) / "science" / field
    root.mkdir(parents=True, exist_ok=True)
    return root


def make(a) -> int:
    if a.field != "ldct":
        print(json.dumps({"error": f"no problem maker for field {a.field!r}"}))
        return 2
    from . import ldct_family as fam
    spec = fam.make(store(a.field), a.patient, a.slice, float(a.hu), int(a.r))
    pub = store(a.field) / "public" / spec["problem_id"]
    into = Path(a.into)
    (into / "data").mkdir(parents=True, exist_ok=True)
    shutil.copy2(pub / "data" / "low_dose.npy", into / "data" / "low_dose.npy")
    shutil.copy2(pub / "problem.json", into / "problem.json")
    print(json.dumps({"problem_id": spec["problem_id"], "conditions": spec["conditions"], "into": str(into)}))
    return 0


def validate(a) -> int:
    if a.field != "ldct":
        print(json.dumps({"error": f"no validator for field {a.field!r}"}))
        return 2
    from . import validator_ldct as val
    key = store(a.field) / "keys" / a.problem
    if "/" in a.problem or not (key / "full_dose.npy").exists():
        print(json.dumps({"error": f"unknown problem {a.problem!r}"}))
        return 2
    try:
        rec = np.load(a.reconstruction, allow_pickle=False)
        v = val.validate(key, rec)
    except Exception as exc:                         # nothing valid to judge: not a fail, not a pass
        print(json.dumps({"error": f"cannot judge: {exc}"}))
        return 2
    out = {"field": a.field, "problem": a.problem, "validator": v["validator"], "passed": v["passed"],
           "reasons": v["reasons"], "metrics": v["metrics"]}
    line = json.dumps(out)
    if a.json:
        Path(a.json).write_text(line + "\n")
    print(line)
    return 0 if v["passed"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pwm_mode.science_cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--field", default="ldct")
    m.add_argument("--patient", required=True)
    m.add_argument("--slice", type=int, default=1)
    m.add_argument("--hu", type=float, default=150.0)
    m.add_argument("--r", type=int, default=5)
    m.add_argument("--into", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--field", default="ldct")
    v.add_argument("--problem", required=True)
    v.add_argument("--reconstruction", required=True)
    v.add_argument("--json", default="")
    a = ap.parse_args(argv)
    return make(a) if a.cmd == "make" else validate(a)


if __name__ == "__main__":
    sys.exit(main())
