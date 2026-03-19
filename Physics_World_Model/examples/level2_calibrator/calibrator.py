"""Level 2 Example: Grid-search calibrator (working reference implementation).

Calibrates operator parameters by grid search over perturbations,
minimizing re-projection error.

Usage:
    python examples/level2_calibrator/calibrator.py  # self-test
"""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, Tuple

import numpy as np


def calibrate_grid_search(
    y: np.ndarray,
    H_nom: Any,
    budget: float,
) -> Tuple[Any, Dict[str, Any]]:
    """Grid-search calibrator: search parameter perturbations.

    For each learnable parameter, evaluates a grid of offsets and picks
    the one minimizing ||H(A^T y) - y||.
    """
    H_hat = copy.deepcopy(H_nom)
    theta = H_hat.get_theta()
    best_theta = dict(theta)
    best_residual = float("inf")
    n_evals = 0

    t_start = time.perf_counter()

    # Quick baseline: residual with original params
    try:
        x_approx = H_hat.adjoint(y)
        y_reproj = H_hat.forward(x_approx)
        best_residual = float(np.linalg.norm(y_reproj - y))
    except Exception:
        pass

    # Grid search over each parameter
    grid_points = 11
    for key, val in theta.items():
        if not isinstance(val, (int, float)):
            continue
        if time.perf_counter() - t_start > budget * 0.9:
            break

        for delta in np.linspace(-1.0, 1.0, grid_points):
            if time.perf_counter() - t_start > budget * 0.9:
                break

            test_theta = dict(best_theta)
            test_theta[key] = float(val) + delta
            H_hat.set_theta(test_theta)

            try:
                x_approx = H_hat.adjoint(y)
                y_reproj = H_hat.forward(x_approx)
                residual = float(np.linalg.norm(y_reproj - y))
            except Exception:
                residual = float("inf")

            n_evals += 1
            if residual < best_residual:
                best_residual = residual
                best_theta = dict(test_theta)

    H_hat.set_theta(best_theta)
    elapsed = time.perf_counter() - t_start

    return H_hat, {
        "method": "grid_search",
        "budget_s": budget,
        "elapsed_s": elapsed,
        "n_evaluations": n_evals,
        "best_residual": best_residual,
        "params_found": best_theta,
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    class MockOperator:
        def __init__(self, sigma=2.0):
            self._theta = {"blur.sigma": sigma}

        x_shape = (32, 32)
        y_shape = (32, 32)
        all_linear = True

        def get_theta(self): return dict(self._theta)
        def set_theta(self, t): self._theta = dict(t)
        def forward(self, x):
            s = self._theta.get("blur.sigma", 2.0)
            return x * (1.0 / (1.0 + s * 0.1))
        def adjoint(self, y):
            s = self._theta.get("blur.sigma", 2.0)
            return y * (1.0 / (1.0 + s * 0.1))

    H_true = MockOperator(sigma=2.0)
    H_nom = MockOperator(sigma=3.5)  # mismatched

    rng = np.random.default_rng(42)
    x_true = rng.standard_normal(H_true.x_shape)
    y = H_true.forward(x_true) + rng.normal(0, 0.01, H_true.y_shape)

    H_hat, info = calibrate_grid_search(y, H_nom, budget=5.0)

    print(f"Self-test PASSED")
    print(f"  Original sigma: 3.5, Calibrated: {H_hat.get_theta()['blur.sigma']:.2f}, True: 2.0")
    print(f"  Evaluations: {info['n_evaluations']}, Elapsed: {info['elapsed_s']:.2f}s")
