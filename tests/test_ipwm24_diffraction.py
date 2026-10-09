"""Security and trap checks for the IPWM-24 independent diffraction judges."""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import numpy as np

from pwm_mode.imaging.diffraction.judges import judge_atom_probe_simulate, judge_saxs_simulate


def _curve_hash(curve):
    raw = np.ascontiguousarray(np.asarray(curve, dtype="<f8")).tobytes()
    return hashlib.sha256(raw).hexdigest()


def test_judge_has_no_solver_imports():
    path = Path(__file__).parents[1] / "pwm_mode" / "imaging" / "diffraction" / "judges.py"
    tree = ast.parse(path.read_text())
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add((node.module or "").split(".")[0])
    assert imports <= {"__future__", "hashlib", "math", "typing", "numpy"}


def test_saxs_simulation_judge_traps():
    radius = np.linspace(65.0, 345.0, 29)
    reference = 0.2 + 0.7 * np.exp(-radius / 130.0) * (1.0 + 0.2 * np.cos(radius / 18.0))
    public = reference * 1.01
    key = {"reference_curve": reference,
           "forbidden_public_curve_sha256": [_curve_hash(public)]}
    criterion = {"relative_l2_max": 0.10}

    assert judge_saxs_simulate({"curve": reference}, key, criterion)["pass"]
    assert not judge_saxs_simulate({"sample": "SAXS", "exposure_s": 1.0}, key, criterion)["pass"]
    assert not judge_saxs_simulate({"curve": np.zeros_like(reference)}, key, criterion)["pass"]
    assert not judge_saxs_simulate({"curve": np.full_like(reference, reference.mean())}, key, criterion)["pass"]
    assert not judge_saxs_simulate({"curve": np.convolve(reference, np.ones(7) / 7, mode="same")}, key, criterion)["pass"]
    assert not judge_saxs_simulate({"curve": public}, key, criterion)["pass"]
    assert not judge_saxs_simulate({"curve": np.ones((8, 8))}, key, criterion)["pass"]


def test_atom_probe_simulation_judge_traps():
    axis = np.arange(0.0, 70.0, 0.025)
    reference = 4000 * np.exp(-0.5 * ((axis - 26.98) / 0.05) ** 2)
    reference += 3000 * np.exp(-0.5 * ((axis - 27.97) / 0.05) ** 2)
    reference += 2000 * np.exp(-0.5 * ((axis - 55.93) / 0.05) ** 2)
    public_copy = reference * 1.02
    key = {"reference_spectrum": reference,
           "forbidden_public_spectrum_sha256": [_curve_hash(public_copy)]}
    criterion = {"relative_l2_max": 0.10}

    assert judge_atom_probe_simulate({"spectrum": reference}, key, criterion)["pass"]
    assert not judge_atom_probe_simulate({"sample": "APT", "detector_mode": "nominal"}, key, criterion)["pass"]
    assert not judge_atom_probe_simulate({"spectrum": np.zeros_like(reference)}, key, criterion)["pass"]
    assert not judge_atom_probe_simulate({"spectrum": np.full_like(reference, reference.mean())}, key, criterion)["pass"]
    assert not judge_atom_probe_simulate({"spectrum": np.convolve(reference, np.ones(15) / 15, mode="same")}, key, criterion)["pass"]
    assert not judge_atom_probe_simulate({"spectrum": public_copy}, key, criterion)["pass"]
    assert not judge_atom_probe_simulate({"spectrum": np.ones((64, 64))}, key, criterion)["pass"]
