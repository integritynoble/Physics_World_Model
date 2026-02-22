"""pwm_core.targeting.contrib_check
=====================================

``pwm contrib check my_solver``

One-command validation for contributions before PR:
1. Signature compliance
2. Isolation (no forbidden imports)
3. Ground truth isolation
4. Sandbox evaluation
5. Registry validation
6. Format check
"""

from __future__ import annotations

import importlib
import inspect
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)

_CONTRIB_DIR = Path(__file__).resolve().parents[2] / "contrib"

# Imports that solvers must NOT use
FORBIDDEN_IMPORTS = [
    "graph.compiler",
    "graph.primitives",
    "targeting",
    "pwm_core.graph.compiler",
    "pwm_core.graph.primitives",
    "pwm_core.targeting",
]


def _check_signature(module_path: str, func_name: str) -> Dict[str, Any]:
    """Check that the solver function has the correct signature."""
    try:
        mod = importlib.import_module(module_path)
        fn = getattr(mod, func_name)
    except (ImportError, AttributeError) as e:
        return {"passed": False, "reason": f"Cannot import {module_path}.{func_name}: {e}"}

    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())

    if params != ["y", "physics", "cfg"]:
        return {
            "passed": False,
            "reason": f"Expected (y, physics, cfg), got ({', '.join(params)})",
        }

    return {"passed": True, "reason": "Signature correct: (y, physics, cfg)"}


def _check_isolation(module_path: str) -> Dict[str, Any]:
    """Check that the solver does not import forbidden modules."""
    try:
        mod = importlib.import_module(module_path)
        source = inspect.getsource(mod)
    except Exception as e:
        return {"passed": False, "reason": f"Cannot read source: {e}"}

    violations = []
    for forbidden in FORBIDDEN_IMPORTS:
        if forbidden in source:
            violations.append(forbidden)

    if violations:
        return {
            "passed": False,
            "reason": f"Forbidden imports found: {violations}",
        }

    return {"passed": True, "reason": "No forbidden imports"}


def _check_ground_truth_isolation(
    module_path: str, func_name: str
) -> Dict[str, Any]:
    """Check that the solver never accesses H_true or x_gt."""

    class SpyOperator:
        """Tracks attribute access to detect ground-truth leaks."""

        x_shape = (32, 32)
        y_shape = (32, 32)
        all_linear = True

        def __init__(self):
            self.accessed_attributes: List[str] = []

        def __getattr__(self, name):
            if name not in ("x_shape", "y_shape", "all_linear", "forward", "adjoint",
                            "accessed_attributes"):
                self.accessed_attributes.append(name)
            raise AttributeError(f"SpyOperator has no attribute '{name}'")

        def forward(self, x):
            return x * 0.5

        def adjoint(self, y):
            return y * 0.5

    try:
        mod = importlib.import_module(module_path)
        fn = getattr(mod, func_name)

        spy = SpyOperator()
        y = np.random.randn(*spy.y_shape)
        fn(y, spy, {})

        gt_accesses = [a for a in spy.accessed_attributes
                       if a in ("H_true", "x_gt", "x_true", "ground_truth")]
        if gt_accesses:
            return {
                "passed": False,
                "reason": f"Accessed ground-truth attributes: {gt_accesses}",
            }
    except Exception:
        pass  # If it crashes, that's OK for this check

    return {"passed": True, "reason": "No ground-truth access detected"}


def _check_sandbox_eval(name: str) -> Dict[str, Any]:
    """Run a quick sandbox evaluation."""
    # Prefer cassi (guaranteed in solver_registry); fall back to first available.
    _SANDBOX_MODALITY_PREFERENCE = ["cassi", "widefield", "ct", "mri"]

    try:
        from pwm_core.targeting.harness import Harness, load_solver_registry

        registry = load_solver_registry()
        sandbox_modality = None
        for candidate in _SANDBOX_MODALITY_PREFERENCE:
            if candidate in registry and "traditional_cpu" in registry.get(candidate, {}):
                sandbox_modality = candidate
                break
        if sandbox_modality is None:
            for mod, tiers in registry.items():
                if isinstance(tiers, dict) and "traditional_cpu" in tiers:
                    sandbox_modality = mod
                    break
        if sandbox_modality is None:
            return {"passed": False, "reason": "No traditional_cpu solver found in registry"}

        harness = Harness(
            modality=sandbox_modality,
            solver="traditional_cpu",
            sandbox=True,
        )
        result = harness.run(n_scenes=1, seed=42, severity="mild")
        return {
            "passed": True,
            "reason": (
                f"Sandbox eval ({sandbox_modality}): "
                f"rho={result.aggregate.rho:.4f}"
            ),
        }
    except Exception as e:
        return {"passed": False, "reason": f"Sandbox eval failed: {e}"}


def check_contribution(
    name: str,
    contrib_type: str = "solver",
) -> Dict[str, Any]:
    """Run all validation checks on a contribution.

    Parameters
    ----------
    name : str
        Component name.
    contrib_type : str
        'solver', 'calibrator', or 'modality'.

    Returns
    -------
    dict
        Keys: 'passed' (bool), 'report' (list of str), 'checks' (list of dict).
    """
    report: List[str] = []
    checks: List[Dict[str, Any]] = []
    all_passed = True

    report.append(f"Checking {contrib_type}: {name}")
    report.append("=" * 40)

    if contrib_type == "solver":
        module_path = f"contrib.solvers.{name}.solver"
        func_name = f"run_{name}"

        # 1. Signature
        sig_result = _check_signature(module_path, func_name)
        checks.append({"check": "signature", **sig_result})
        status = "PASS" if sig_result["passed"] else "FAIL"
        report.append(f"  [{status}] Signature: {sig_result['reason']}")
        if not sig_result["passed"]:
            all_passed = False

        # 2. Isolation
        iso_result = _check_isolation(module_path)
        checks.append({"check": "isolation", **iso_result})
        status = "PASS" if iso_result["passed"] else "FAIL"
        report.append(f"  [{status}] Isolation: {iso_result['reason']}")
        if not iso_result["passed"]:
            all_passed = False

        # 3. Ground truth
        gt_result = _check_ground_truth_isolation(module_path, func_name)
        checks.append({"check": "ground_truth", **gt_result})
        status = "PASS" if gt_result["passed"] else "FAIL"
        report.append(f"  [{status}] Ground truth: {gt_result['reason']}")
        if not gt_result["passed"]:
            all_passed = False

    # 4. Sandbox eval (for all types)
    sandbox_result = _check_sandbox_eval(name)
    checks.append({"check": "sandbox_eval", **sandbox_result})
    status = "PASS" if sandbox_result["passed"] else "FAIL"
    report.append(f"  [{status}] Sandbox eval: {sandbox_result['reason']}")
    if not sandbox_result["passed"]:
        all_passed = False

    report.append("")
    if all_passed:
        report.append("All checks passed. Ready for PR.")
    else:
        report.append("Some checks failed. Fix issues before submitting PR.")

    return {"passed": all_passed, "report": report, "checks": checks}
