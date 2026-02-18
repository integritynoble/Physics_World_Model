"""pwm_core.cli.reproduce

Reproduce a prior runbundle using generated scripts.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def reproduce(runbundle_dir: str):
    rb = Path(runbundle_dir)
    sim = rb / "reproduce" / "simulate.py"
    rec = rb / "reproduce" / "recon.py"
    if sim.exists():
        subprocess.check_call(["python", str(sim)], cwd=str(sim.parent))
    if rec.exists():
        subprocess.check_call(["python", str(rec)], cwd=str(rec.parent))
