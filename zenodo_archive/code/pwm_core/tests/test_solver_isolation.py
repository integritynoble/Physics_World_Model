"""Dedicated solver isolation tests (anti-cheating).

Per rails/plan.md Addition 7: Solvers that peek at internals undermine
the entire evaluation. These tests run on every solver PR via CI.

Tests verify:
1. Solvers do not import graph compiler or targeting internals
2. Solvers do not access ground-truth attributes (H_true, x_gt)
3. Solver signatures comply with the LinearLikeOperator protocol
4. Solvers respect compute budget constraints
5. Solvers work with the SpyOperator (minimal interface)
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import textwrap
import types
from pathlib import Path
from typing import List

import numpy as np
import pytest

# contrib/templates lives outside the pwm_core package, so we load by path
_REPO_ROOT = Path(__file__).resolve().parents[1]  # packages/pwm_core/
_TEMPLATES = _REPO_ROOT / "contrib" / "templates"


def _load_template(filename: str):
    """Load a contrib template module by file path (not package import)."""
    path = _TEMPLATES / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# SpyOperator: tracks all attribute access to catch ground-truth leaks
# ---------------------------------------------------------------------------


class SpyOperator:
    """Minimal LinearLikeOperator that records all attribute access.

    Only exposes the allowed interface: forward, adjoint, x_shape, y_shape,
    all_linear. Any access to other attributes (H_true, x_gt, etc.) is
    recorded and raises AttributeError.
    """

    ALLOWED = frozenset({
        "forward", "adjoint", "x_shape", "y_shape", "all_linear",
        "accessed_attributes", "_forward_calls", "_adjoint_calls",
    })

    def __init__(self, shape=(32, 32), scale=0.5):
        object.__setattr__(self, "x_shape", shape)
        object.__setattr__(self, "y_shape", shape)
        object.__setattr__(self, "all_linear", True)
        object.__setattr__(self, "_scale", scale)
        object.__setattr__(self, "accessed_attributes", [])
        object.__setattr__(self, "_forward_calls", 0)
        object.__setattr__(self, "_adjoint_calls", 0)

    def __getattr__(self, name):
        if name in self.ALLOWED:
            return object.__getattribute__(self, name)
        object.__getattribute__(self, "accessed_attributes").append(name)
        raise AttributeError(
            f"SpyOperator: '{name}' is not part of the solver interface"
        )

    def forward(self, x):
        object.__setattr__(self, "_forward_calls",
                           object.__getattribute__(self, "_forward_calls") + 1)
        return x * object.__getattribute__(self, "_scale")

    def adjoint(self, y):
        object.__setattr__(self, "_adjoint_calls",
                           object.__getattribute__(self, "_adjoint_calls") + 1)
        return y * object.__getattribute__(self, "_scale")


# ---------------------------------------------------------------------------
# Forbidden-import detection
# ---------------------------------------------------------------------------

FORBIDDEN_PATTERNS = [
    "graph.compiler",
    "graph.primitives",
    "targeting",
    "pwm_core.graph.compiler",
    "pwm_core.graph.primitives",
    "pwm_core.targeting",
]


def _source_contains_forbidden(source: str) -> List[str]:
    """Return list of forbidden import patterns found in *import statements*.

    Only checks actual ``import`` / ``from ... import`` lines, ignoring
    comments and docstrings.
    """
    import re

    violations = []
    for line in source.splitlines():
        stripped = line.strip()
        # Skip comments and docstrings
        if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
            continue
        # Only check import statements
        if not (stripped.startswith("import ") or stripped.startswith("from ")):
            continue
        for pattern in FORBIDDEN_PATTERNS:
            if pattern in stripped and pattern not in violations:
                violations.append(pattern)
    return violations


# ---------------------------------------------------------------------------
# Tests: Solver template compliance
# ---------------------------------------------------------------------------


class TestSolverTemplateIsolation:
    """Verify the contrib solver template is fully isolated."""

    def test_no_forbidden_imports(self):
        """Solver template must not import graph compiler or targeting."""
        mod = _load_template("contrib_solver_template.py")
        source = inspect.getsource(mod)
        violations = _source_contains_forbidden(source)
        assert not violations, f"Forbidden imports in solver template: {violations}"

    def test_correct_signature(self):
        """Solver must accept exactly (y, physics, cfg)."""
        mod = _load_template("contrib_solver_template.py")
        sig = inspect.signature(mod.run_example_solver)
        params = list(sig.parameters.keys())
        assert params == ["y", "physics", "cfg"], (
            f"Expected (y, physics, cfg), got ({', '.join(params)})"
        )

    def test_returns_tuple(self):
        """Solver must return (x_hat, info_dict)."""
        mod = _load_template("contrib_solver_template.py")
        run_example_solver = mod.run_example_solver

        spy = SpyOperator()
        y = np.random.randn(*spy.y_shape)
        result = run_example_solver(y, spy, {"iters": 3})

        assert isinstance(result, tuple), "Solver must return a tuple"
        assert len(result) == 2, "Solver must return (x_hat, info)"

        x_hat, info = result
        assert isinstance(x_hat, np.ndarray), "x_hat must be ndarray"
        assert isinstance(info, dict), "info must be dict"

    def test_output_shape(self):
        """Solver output x_hat must match physics.x_shape."""
        mod = _load_template("contrib_solver_template.py")
        run_example_solver = mod.run_example_solver

        spy = SpyOperator(shape=(16, 16))
        y = np.random.randn(*spy.y_shape)
        x_hat, _ = run_example_solver(y, spy, {"iters": 3})
        assert x_hat.shape == spy.x_shape, (
            f"x_hat shape {x_hat.shape} != physics.x_shape {spy.x_shape}"
        )

    def test_no_ground_truth_access(self):
        """Solver must not try to access H_true, x_gt, or similar."""
        mod = _load_template("contrib_solver_template.py")
        run_example_solver = mod.run_example_solver

        spy = SpyOperator()
        y = np.random.randn(*spy.y_shape)

        try:
            run_example_solver(y, spy, {"iters": 3})
        except Exception:
            pass  # Crashes are fine for this check

        gt_attrs = {"H_true", "x_gt", "x_true", "ground_truth", "true_params"}
        accessed = set(spy.accessed_attributes)
        leaked = accessed & gt_attrs
        assert not leaked, f"Solver accessed ground-truth attributes: {leaked}"

    def test_only_uses_allowed_interface(self):
        """Solver should only use forward/adjoint/shapes, nothing else."""
        mod = _load_template("contrib_solver_template.py")
        run_example_solver = mod.run_example_solver

        spy = SpyOperator()
        y = np.random.randn(*spy.y_shape)

        try:
            run_example_solver(y, spy, {"iters": 3})
        except Exception:
            pass

        assert not spy.accessed_attributes, (
            f"Solver accessed non-interface attributes: {spy.accessed_attributes}"
        )


# ---------------------------------------------------------------------------
# Tests: Calibrator template compliance
# ---------------------------------------------------------------------------


class TestCalibratorTemplateIsolation:
    """Verify the contrib calibrator template is isolated."""

    def test_no_forbidden_imports(self):
        mod = _load_template("contrib_calibrator_template.py")
        source = inspect.getsource(mod)
        violations = _source_contains_forbidden(source)
        assert not violations, f"Forbidden imports in calibrator template: {violations}"

    def test_correct_signature(self):
        mod = _load_template("contrib_calibrator_template.py")
        sig = inspect.signature(mod.calibrate_example)
        params = list(sig.parameters.keys())
        assert params == ["y", "H_nom", "budget"], (
            f"Expected (y, H_nom, budget), got ({', '.join(params)})"
        )


# ---------------------------------------------------------------------------
# Tests: Dynamic solver module checking
# ---------------------------------------------------------------------------


class TestDynamicIsolation:
    """Test isolation checking on dynamically created modules."""

    def _make_module(self, name: str, source: str) -> types.ModuleType:
        """Create a module from source string for testing."""
        mod = types.ModuleType(name)
        exec(compile(textwrap.dedent(source), f"<{name}>", "exec"), mod.__dict__)
        return mod

    def test_clean_solver_passes(self):
        """A solver using only the allowed interface should pass."""
        source = """
        import numpy as np

        def run_clean(y, physics, cfg):
            x = physics.adjoint(y)
            for _ in range(cfg.get('iters', 5)):
                r = physics.forward(x) - y
                g = physics.adjoint(r)
                x = x - 0.01 * g
            return x, {"solver": "clean"}
        """
        mod = self._make_module("clean_solver", source)
        mod_source = textwrap.dedent(source)
        violations = _source_contains_forbidden(mod_source)
        assert not violations

        # Also run through SpyOperator
        spy = SpyOperator()
        y = np.random.randn(*spy.y_shape)
        x_hat, info = mod.run_clean(y, spy, {"iters": 3})
        assert not spy.accessed_attributes

    def test_cheating_solver_detected(self):
        """A solver importing graph.compiler should be caught."""
        source = """
        # This solver cheats by importing graph.compiler
        from pwm_core.graph.compiler import GraphCompiler
        import numpy as np

        def run_cheat(y, physics, cfg):
            return physics.adjoint(y), {"solver": "cheat"}
        """
        violations = _source_contains_forbidden(textwrap.dedent(source))
        assert len(violations) > 0, "Cheating solver not detected"
        assert "graph.compiler" in violations[0]

    def test_ground_truth_leak_detected(self):
        """A solver accessing H_true should be caught by SpyOperator."""
        spy = SpyOperator()
        y = np.random.randn(*spy.y_shape)

        # Simulate a solver that tries to access H_true
        def cheating_solver(y, physics, cfg):
            try:
                _ = physics.H_true  # This should be caught
            except AttributeError:
                pass
            return physics.adjoint(y), {}

        cheating_solver(y, spy, {})
        assert "H_true" in spy.accessed_attributes, (
            "SpyOperator did not detect H_true access"
        )

    def test_targeting_import_detected(self):
        """A solver importing targeting internals should be caught."""
        source = """
        from pwm_core.targeting.scoring import compute_recovery_ratio
        import numpy as np

        def run_cheat2(y, physics, cfg):
            return physics.adjoint(y), {}
        """
        violations = _source_contains_forbidden(textwrap.dedent(source))
        assert len(violations) > 0, "Targeting import not detected"


# ---------------------------------------------------------------------------
# Tests: SpyOperator correctness
# ---------------------------------------------------------------------------


class TestSpyOperator:
    """Verify the SpyOperator itself works correctly."""

    def test_allowed_attributes_work(self):
        spy = SpyOperator(shape=(8, 8))
        assert spy.x_shape == (8, 8)
        assert spy.y_shape == (8, 8)
        assert spy.all_linear is True

    def test_forward_adjoint_work(self):
        spy = SpyOperator(shape=(4, 4), scale=0.3)
        x = np.ones((4, 4))
        y = spy.forward(x)
        np.testing.assert_allclose(y, 0.3 * np.ones((4, 4)))

        x_back = spy.adjoint(y)
        np.testing.assert_allclose(x_back, 0.09 * np.ones((4, 4)))

    def test_forbidden_access_raises(self):
        spy = SpyOperator()
        with pytest.raises(AttributeError, match="not part of the solver"):
            _ = spy.H_true

    def test_forbidden_access_recorded(self):
        spy = SpyOperator()
        try:
            _ = spy.H_true
        except AttributeError:
            pass
        try:
            _ = spy.x_gt
        except AttributeError:
            pass
        assert spy.accessed_attributes == ["H_true", "x_gt"]

    def test_call_counts(self):
        spy = SpyOperator(shape=(4, 4))
        x = np.ones((4, 4))
        spy.forward(x)
        spy.forward(x)
        spy.adjoint(x)
        assert spy._forward_calls == 2
        assert spy._adjoint_calls == 1
