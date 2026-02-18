"""Level 4 Example: Adjoint correctness test for custom primitives.

Every new primitive MUST pass this test before acceptance.

The test verifies the fundamental identity:
    <A(x), y> == <x, A^T(y)>

for random vectors x, y. This guarantees that iterative solvers
(FISTA, ADMM, PnP, etc.) can safely use the adjoint.

Usage:
    python examples/level4_primitive/adjoint_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Import the example primitive
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from level4_primitive.primitive import GaussianBlurPrimitive


def test_adjoint_dot_product(
    primitive,
    shape: tuple,
    n_trials: int = 10,
    tol: float = 1e-6,
    seed: int = 42,
) -> dict:
    """Verify adjoint correctness via the dot-product test.

    For a linear operator A:
        <A(x), y> should equal <x, A^T(y)> for all x, y

    Returns a dict with test results.
    """
    rng = np.random.default_rng(seed)
    errors = []

    for trial in range(n_trials):
        x = rng.standard_normal(shape)
        y = rng.standard_normal(shape)

        Ax = primitive.forward(x)
        ATy = primitive.adjoint(y)

        inner_left = float(np.sum(Ax * y))
        inner_right = float(np.sum(x * ATy))

        denom = max(abs(inner_left), abs(inner_right), 1e-30)
        rel_err = abs(inner_left - inner_right) / denom
        errors.append(rel_err)

    max_err = max(errors)
    mean_err = sum(errors) / len(errors)
    passed = max_err < tol

    return {
        "passed": passed,
        "max_relative_error": max_err,
        "mean_relative_error": mean_err,
        "n_trials": n_trials,
        "tolerance": tol,
        "shape": list(shape),
    }


def test_adjoint_energy_conservation(
    primitive,
    shape: tuple,
    n_trials: int = 10,
    tol: float = 0.1,
    seed: int = 42,
) -> dict:
    """Check that A does not amplify energy unboundedly.

    For well-behaved linear operators, ||A(x)|| should not vastly
    exceed ||x||. This is a sanity check, not a strict requirement.
    """
    rng = np.random.default_rng(seed)
    ratios = []

    for trial in range(n_trials):
        x = rng.standard_normal(shape)
        Ax = primitive.forward(x)
        ratio = np.linalg.norm(Ax) / max(np.linalg.norm(x), 1e-30)
        ratios.append(float(ratio))

    max_ratio = max(ratios)
    passed = max_ratio < 100.0  # very generous bound

    return {
        "passed": passed,
        "max_energy_ratio": max_ratio,
        "mean_energy_ratio": sum(ratios) / len(ratios),
        "note": "Ratio > 100 suggests an implementation error",
    }


def test_adjoint_linearity(
    primitive,
    shape: tuple,
    n_trials: int = 5,
    tol: float = 1e-6,
    seed: int = 42,
) -> dict:
    """Verify that forward is linear: A(ax + by) == a*A(x) + b*A(y)."""
    if not primitive.is_linear:
        return {"passed": True, "note": "Skipped (primitive is nonlinear)"}

    rng = np.random.default_rng(seed)
    errors = []

    for trial in range(n_trials):
        x1 = rng.standard_normal(shape)
        x2 = rng.standard_normal(shape)
        a, b = rng.standard_normal(2)

        # A(ax1 + bx2)
        lhs = primitive.forward(a * x1 + b * x2)
        # aA(x1) + bA(x2)
        rhs = a * primitive.forward(x1) + b * primitive.forward(x2)

        denom = max(np.linalg.norm(lhs), np.linalg.norm(rhs), 1e-30)
        rel_err = float(np.linalg.norm(lhs - rhs) / denom)
        errors.append(rel_err)

    max_err = max(errors)
    passed = max_err < tol

    return {
        "passed": passed,
        "max_relative_error": max_err,
        "n_trials": n_trials,
        "tolerance": tol,
    }


def run_all_tests():
    """Run the complete adjoint test suite for the example primitive."""
    prim = GaussianBlurPrimitive(sigma=1.5)
    shape = (32, 32)

    print(f"Testing: {prim.primitive_id}")
    print(f"  Tier: {prim._physics_tier}")
    print(f"  Linear: {prim.is_linear}")
    print()

    # Test 1: Dot-product test
    result = test_adjoint_dot_product(prim, shape)
    status = "PASS" if result["passed"] else "FAIL"
    print(f"[{status}] Dot-product test")
    print(f"  Max relative error: {result['max_relative_error']:.2e}")
    print(f"  Tolerance: {result['tolerance']:.2e}")
    print()

    # Test 2: Energy conservation
    result2 = test_adjoint_energy_conservation(prim, shape)
    status2 = "PASS" if result2["passed"] else "FAIL"
    print(f"[{status2}] Energy conservation")
    print(f"  Max energy ratio: {result2['max_energy_ratio']:.4f}")
    print()

    # Test 3: Linearity
    result3 = test_adjoint_linearity(prim, shape)
    status3 = "PASS" if result3["passed"] else "FAIL"
    print(f"[{status3}] Linearity test")
    if "max_relative_error" in result3:
        print(f"  Max relative error: {result3['max_relative_error']:.2e}")
    print()

    all_passed = result["passed"] and result2["passed"] and result3["passed"]
    print("=" * 40)
    print(f"Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
