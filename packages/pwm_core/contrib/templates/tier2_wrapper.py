"""Tier-2 Physics Kernel Wrapper Template
==========================================

You implement: forward_kernel() and adjoint_kernel()
We handle: validation, shape checking, tier tagging, serialization

Example: Mie scattering kernel for microscopy

Usage:
    1. Copy this file
    2. Implement forward_kernel() and adjoint_kernel()
    3. Run the self-test at the bottom
    4. Submit RFC + PR for steward review
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


class Tier2Adapter:
    """Wraps raw physics kernels into valid PrimitiveOps.

    Contributors implement only:
    - forward_kernel(x, params) -> y
    - adjoint_kernel(y, params) -> x  (or auto-derived for linear ops)

    The adapter handles:
    - Adjoint correctness validation (dot-product test)
    - Shape checking and dtype normalization
    - Physics tier tag injection
    - Serialization boilerplate
    """

    def __init__(
        self,
        forward_kernel,
        adjoint_kernel,
        params: Dict[str, Any],
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        name: str = "tier2_custom",
    ):
        self.forward_kernel = forward_kernel
        self.adjoint_kernel = adjoint_kernel
        self._params = dict(params)
        self._input_shape = input_shape
        self._output_shape = output_shape
        self.primitive_id = name
        self._is_linear = True
        self._physics_tier = "tier2_full"
        self._physics_subrole = "interaction"

    @property
    def is_linear(self) -> bool:
        return self._is_linear

    @property
    def is_stochastic(self) -> bool:
        return False

    @property
    def is_differentiable(self) -> bool:
        return True

    @property
    def is_stateful(self) -> bool:
        return False

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        y = self.forward_kernel(x, self._params)
        return np.asarray(y, dtype=np.float64)

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        x = self.adjoint_kernel(y, self._params)
        return np.asarray(x, dtype=np.float64)

    def serialize(self) -> Dict[str, Any]:
        return {
            "primitive_id": self.primitive_id,
            "params": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                       for k, v in self._params.items()},
            "physics_tier": self._physics_tier,
        }

    def check_adjoint(self, n_trials: int = 5, tol: float = 1e-6, seed: int = 42) -> bool:
        """Verify <Ax, y> == <x, A^T y> for random vectors."""
        rng = np.random.default_rng(seed)
        max_err = 0.0

        for _ in range(n_trials):
            x = rng.standard_normal(self._input_shape)
            y = rng.standard_normal(self._output_shape)

            Ax = self.forward(x)
            ATy = self.adjoint(y)

            inner1 = float(np.sum(Ax.ravel() * y.ravel()))
            inner2 = float(np.sum(x.ravel() * ATy.ravel()))

            denom = max(abs(inner1), abs(inner2), 1e-30)
            rel_err = abs(inner1 - inner2) / denom
            max_err = max(max_err, rel_err)

        passed = max_err < tol
        print(f"Adjoint check: max_rel_err={max_err:.2e}, tol={tol:.2e}, "
              f"{'PASS' if passed else 'FAIL'}")
        return passed


# ===================================================================
# EXAMPLE: Simple scaling operator (replace with your physics kernel)
# ===================================================================

def example_forward_kernel(x: np.ndarray, params: dict) -> np.ndarray:
    """YOUR PHYSICS HERE: apply forward model."""
    scale = params.get("scale", 0.8)
    return x * scale


def example_adjoint_kernel(y: np.ndarray, params: dict) -> np.ndarray:
    """YOUR ADJOINT HERE: transpose of Jacobian."""
    scale = params.get("scale", 0.8)
    return y * scale  # Self-adjoint for scaling


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    adapter = Tier2Adapter(
        forward_kernel=example_forward_kernel,
        adjoint_kernel=example_adjoint_kernel,
        params={"scale": 0.8},
        input_shape=(64, 64),
        output_shape=(64, 64),
        name="example_tier2",
    )

    # 1. Adjoint correctness
    passed = adapter.check_adjoint()
    assert passed, "Adjoint check failed!"

    # 2. Forward/adjoint shape check
    x = np.random.randn(64, 64)
    y = adapter.forward(x)
    x_back = adapter.adjoint(y)
    assert y.shape == (64, 64), f"Forward shape: {y.shape}"
    assert x_back.shape == (64, 64), f"Adjoint shape: {x_back.shape}"

    # 3. Serialization
    s = adapter.serialize()
    assert s["physics_tier"] == "tier2_full"

    print("All self-tests PASSED")
