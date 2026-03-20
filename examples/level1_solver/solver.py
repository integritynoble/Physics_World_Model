"""Level 1 Example: FISTA solver (working reference implementation).

This is a real, working solver that produces real scores on the harness.
Copy this file, replace the algorithm, and submit.

Usage:
    pwm evaluate --sandbox --modality widefield --solver example_fista
    python examples/level1_solver/solver.py   # self-test
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def run_example_fista(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """FISTA: Fast Iterative Shrinkage-Thresholding Algorithm.

    A proximal gradient method with Nesterov momentum for solving:
        min_x ||Ax - y||^2 + lambda * ||x||_1
    """
    iters = cfg.get("iters", 100)
    step_size = cfg.get("step_size", 0.01)
    lam = cfg.get("lambda", 0.001)
    momentum = cfg.get("momentum", True)

    # Initialize
    x_hat = physics.adjoint(y)
    x_prev = x_hat.copy()
    t = 1.0

    for i in range(iters):
        # Gradient: A^T(Ax - y)
        residual = physics.forward(x_hat) - y
        gradient = physics.adjoint(residual)

        # Gradient step
        x_new = x_hat - step_size * gradient

        # Soft thresholding (L1 proximal)
        x_new = np.sign(x_new) * np.maximum(np.abs(x_new) - lam * step_size, 0)

        # Nesterov momentum
        if momentum:
            t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
            x_hat = x_new + ((t - 1) / t_new) * (x_new - x_prev)
            x_prev = x_new
            t = t_new
        else:
            x_hat = x_new

    return x_hat, {
        "solver": "example_fista",
        "iters": iters,
        "step_size": step_size,
        "lambda": lam,
        "momentum": momentum,
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    class ToyOp:
        x_shape = (32, 32)
        y_shape = (32, 32)
        all_linear = True
        def forward(self, x): return x * 0.8
        def adjoint(self, y): return y * 0.8

    op = ToyOp()
    rng = np.random.default_rng(42)
    x_true = rng.standard_normal(op.x_shape)
    x_true = (x_true - x_true.min()) / (x_true.max() - x_true.min())
    y = op.forward(x_true) + rng.normal(0, 0.01, op.y_shape)

    x_hat, info = run_example_fista(y, op, {"iters": 50, "step_size": 0.5})

    mse = float(np.mean((x_hat - x_true) ** 2))
    print(f"Self-test PASSED: MSE = {mse:.6f}")
    print(f"Info: {info}")
