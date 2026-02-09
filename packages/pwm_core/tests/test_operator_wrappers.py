"""
test_operator_wrappers.py

Unit tests for PWM operator wrappers:
- MatrixOperator wrapper (dense/sparse) basic forward/adjoint consistency
- CallableOperator wrapper interface checks
- DeepInv bridge smoke test (optional if deepinv installed)

These tests are intentionally lightweight and do NOT require large datasets.
They focus on:
- shape correctness
- adjoint consistency (inner product test)
- safe error messages for mismatch shapes

Run:
    pytest -q packages/pwm_core/tests/test_operator_wrappers.py
"""

from __future__ import annotations

import math
import numpy as np
import pytest

try:
    import torch
except Exception as e:  # pragma: no cover
    torch = None

# Import wrappers (expected paths in the proposed scheme)
from pwm_core.physics.adapters.matrix_operator import MatrixOperator
from pwm_core.physics.adapters.callable_operator import CallableOperator

# Optional deepinv bridge
try:
    from pwm_core.physics.adapters.deepinv_bridge import to_deepinv_physics
    _HAS_DEEPINV_BRIDGE = True
except Exception:
    _HAS_DEEPINV_BRIDGE = False


pytestmark = pytest.mark.skipif(torch is None, reason="torch not installed")


def _inner_product_test(A, At, x, y, rtol=1e-3, atol=1e-3):
    """Check <Ax, y> == <x, A^T y>."""
    Ax = A(x)
    Aty = At(y)
    lhs = torch.sum(Ax * y).item()
    rhs = torch.sum(x * Aty).item()
    denom = max(1e-8, abs(lhs), abs(rhs))
    rel_err = abs(lhs - rhs) / denom
    assert rel_err <= rtol or abs(lhs - rhs) <= atol


def test_matrix_operator_dense_forward_shapes():
    M, N = 32, 64
    A = torch.randn(M, N)
    op = MatrixOperator.from_dense(A)

    x = torch.randn(N)
    y = op.forward(x)

    assert y.shape == (M,)
    x2 = op.adjoint(y)
    assert x2.shape == (N,)


def test_matrix_operator_dense_adjoint_consistency():
    M, N = 25, 40
    A = torch.randn(M, N)
    op = MatrixOperator.from_dense(A)

    x = torch.randn(N)
    y = torch.randn(M)

    _inner_product_test(op.forward, op.adjoint, x, y, rtol=2e-3, atol=2e-3)


def test_matrix_operator_sparse_adjoint_consistency():
    # Build a random sparse matrix in CSR using scipy-style indices, then convert to torch sparse
    M, N = 50, 80
    density = 0.08
    nnz = int(M * N * density)

    rng = np.random.default_rng(0)
    rows = rng.integers(0, M, size=nnz)
    cols = rng.integers(0, N, size=nnz)
    vals = rng.standard_normal(size=nnz).astype(np.float32)

    indices = torch.tensor(np.vstack([rows, cols]), dtype=torch.int64)
    values = torch.tensor(vals, dtype=torch.float32)
    A_sp = torch.sparse_coo_tensor(indices, values, size=(M, N)).coalesce()

    op = MatrixOperator.from_sparse(A_sp)

    x = torch.randn(N)
    y = torch.randn(M)

    _inner_product_test(op.forward, op.adjoint, x, y, rtol=3e-3, atol=3e-3)


def test_matrix_operator_shape_mismatch_raises():
    M, N = 10, 20
    A = torch.randn(M, N)
    op = MatrixOperator.from_dense(A)

    bad_x = torch.randn(N + 1)
    with pytest.raises(ValueError):
        _ = op.forward(bad_x)

    bad_y = torch.randn(M + 1)
    with pytest.raises(ValueError):
        _ = op.adjoint(bad_y)


def test_callable_operator_forward_adjoint_consistency():
    # A simple linear operator: y = Bx where B is dense
    M, N = 16, 24
    B = torch.randn(M, N)

    def fwd(x: torch.Tensor) -> torch.Tensor:
        return B @ x

    def adj(y: torch.Tensor) -> torch.Tensor:
        return B.t() @ y

    op = CallableOperator(
        name="dense_linear_callable",
        forward_fn=fwd,
        adjoint_fn=adj,
        x_shape=(N,),
        y_shape=(M,),
    )

    x = torch.randn(N)
    y = torch.randn(M)

    assert op.forward(x).shape == (M,)
    assert op.adjoint(y).shape == (N,)
    _inner_product_test(op.forward, op.adjoint, x, y, rtol=2e-3, atol=2e-3)


def test_callable_operator_requires_shapes():
    with pytest.raises(ValueError):
        _ = CallableOperator(
            name="bad",
            forward_fn=lambda x: x,
            adjoint_fn=lambda y: y,
            x_shape=None,  # type: ignore
            y_shape=(10,),
        )


def test_deepinv_bridge_smoke():
    # This is a smoke test: ensure conversion does not crash for a simple MatrixOperator.
    M, N = 8, 12
    A = torch.randn(M, N)
    op = MatrixOperator.from_dense(A)

    physics = to_deepinv_physics(op)
    # deepinv Physics typically is callable: physics(x)->y
    x = torch.randn(N)
    y = physics(x)
    assert y.shape[-1] == M or y.numel() == M  # allow flexibility in deepinv wrapper conventions
