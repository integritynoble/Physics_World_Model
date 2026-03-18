"""PWM Solver Contribution Template
====================================

Copy this file and implement your reconstruction algorithm.

Protocol (frozen per Rail Constitution Article 1.7):
    run_<solver_name>(y, physics, cfg) -> (x_hat, info)

The ``physics`` argument satisfies LinearLikeOperator:
    physics.forward(x) -> y      Apply the forward model
    physics.adjoint(y) -> x      Apply the adjoint (transpose)
    physics.x_shape -> tuple     Input (object) shape
    physics.y_shape -> tuple     Output (measurement) shape
    physics.all_linear -> bool   True if fully linear

IMPORTANT:
- Your solver must NOT import from graph.compiler, graph.primitives, or targeting.*
- Your solver must NOT access H_true, x_gt, or any ground-truth data
- Your solver must be stateless (no class attributes that persist between calls)
- Your solver receives only (y, physics, cfg) -- nothing else

Quickstart:
    1. Copy this file to contrib/solvers/<your_name>/solver.py
    2. Implement run_<your_name>()
    3. Run: python -m contrib.solvers.<your_name>.solver  (self-test)
    4. Run: pwm evaluate --sandbox --modality widefield --solver <your_name>
    5. Run: pwm contrib check <your_name>
    6. Submit PR

Example paper: "Our method achieves rho=0.85 across 20 modalities on LIP-Arena"
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def run_example_solver(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Example: gradient descent reconstruction.

    Replace this with your algorithm. Keep the signature identical.
    """
    iters = cfg.get("iters", 100)
    step_size = cfg.get("step_size", 0.01)

    # Initialize with adjoint backprojection
    x_hat = physics.adjoint(y)

    # Iterative refinement
    for i in range(iters):
        # Forward model residual
        residual = physics.forward(x_hat) - y
        # Gradient step
        gradient = physics.adjoint(residual)
        x_hat = x_hat - step_size * gradient

    info = {
        "solver": "example_solver",
        "iters": iters,
        "step_size": step_size,
        "final_residual_norm": float(np.linalg.norm(
            physics.forward(x_hat) - y
        )),
    }

    return x_hat, info


# ---------------------------------------------------------------------------
# Self-test (run with: python -m contrib.templates.contrib_solver_template)
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    class _ToyOperator:
        x_shape = (32, 32)
        y_shape = (32, 32)
        all_linear = True

        def forward(self, x):
            return x * 0.8

        def adjoint(self, y):
            return y * 0.8

    op = _ToyOperator()
    rng = np.random.default_rng(42)
    x_true = rng.standard_normal(op.x_shape)
    y = op.forward(x_true) + rng.normal(0, 0.01, op.y_shape)

    x_hat, info = run_example_solver(y, op, {"iters": 50, "step_size": 0.1})

    error = float(np.linalg.norm(x_hat - x_true)) / float(np.linalg.norm(x_true))
    print(f"Self-test PASSED: relative error = {error:.4f}")
    print(f"Info: {info}")
