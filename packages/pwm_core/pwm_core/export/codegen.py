"""pwm_core.export.codegen

Generate artifact-loaded scripts that reproduce the exact run.
Key idea:
- scripts should load RunBundle/internal_state/* and manifest data pointers
- avoid re-sampling randomness during reproduction

This starter provides minimal generators for simulate.py and recon.py.
"""

from __future__ import annotations

from typing import Any, Dict


def generate_simulate_py(manifest: Dict[str, Any]) -> str:
    return f'''#!/usr/bin/env python3
"""Reproduce simulation for RunBundle: {manifest.get("run_id","")}"""

import json
from pathlib import Path

import numpy as np

def main():
    root = Path(__file__).resolve().parent
    m = json.loads((root / "runbundle_manifest.json").read_text())
    # Load internal_state (e.g., perturbations) if present
    state_path = root / "internal_state" / "perturbations.npy"
    if state_path.exists():
        perturb = np.load(state_path, allow_pickle=True).item()
        print("Loaded perturbations:", list(perturb.keys()))
    else:
        perturb = None

    print("This is a starter stub. In full PWM, instantiate PhysicsTrue from manifest/spec, then simulate y.")
    print("Spec version:", m.get("spec_version"))

if __name__ == "__main__":
    main()
'''


def generate_recon_py(manifest: Dict[str, Any]) -> str:
    return f'''#!/usr/bin/env python3
"""Reproduce reconstruction for RunBundle: {manifest.get("run_id","")}"""

import json
from pathlib import Path

def main():
    root = Path(__file__).resolve().parent
    m = json.loads((root / "runbundle_manifest.json").read_text())
    print("This is a starter stub. In full PWM, load y and operator artifacts then run solver recipe.")
    print("Run id:", m.get("run_id"))
    print("Recon outputs:", m.get("artifacts",{{}}).get("recon",{{}}))

if __name__ == "__main__":
    main()
'''
