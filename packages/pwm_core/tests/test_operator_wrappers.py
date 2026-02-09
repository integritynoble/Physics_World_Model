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


# ============================================================================
# OCT and Light Field operator tests (numpy-based, no torch dependency)
# ============================================================================

from pwm_core.physics.oct.oct_operator import OCTOperator
from pwm_core.physics.light_field.lf_operator import LightFieldOperator


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
class TestOCTOperator:
    """Tests for OCTOperator adjoint and shape consistency."""

    def test_oct_adjoint_check(self):
        op = OCTOperator(n_alines=16, n_depth=32, n_spectral=64)
        report = op.check_adjoint(n_trials=3, tol=1e-4)
        assert report.passed, report.summary()

    def test_oct_forward_adjoint_shapes(self):
        op = OCTOperator(n_alines=8, n_depth=16, n_spectral=32)
        rng = np.random.default_rng(0)

        x = rng.standard_normal(op.x_shape).astype(np.float32)
        y = op.forward(x)
        assert y.shape == op.y_shape, f"forward shape {y.shape} != {op.y_shape}"

        y2 = rng.standard_normal(op.y_shape).astype(np.float32)
        x2 = op.adjoint(y2)
        assert x2.shape == op.x_shape, f"adjoint shape {x2.shape} != {op.x_shape}"

    def test_oct_metadata(self):
        op = OCTOperator(n_alines=128, n_depth=256, n_spectral=512)
        meta = op.metadata()
        assert meta.modality == "oct"
        assert meta.is_linear is True
        assert meta.x_shape == [128, 256]
        assert meta.y_shape == [128, 512]

    def test_oct_serialize_roundtrip(self):
        op = OCTOperator(n_alines=8, n_depth=16, n_spectral=32)
        data = op.serialize()
        assert data["operator_id"] == "oct"
        assert data["x_shape"] == [8, 16]
        assert data["y_shape"] == [8, 32]


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
class TestLightFieldOperator:
    """Tests for LightFieldOperator adjoint and shape consistency."""

    def test_light_field_adjoint_check(self):
        op = LightFieldOperator(sx=16, sy=16, nu=3, nv=3, disparity=0.5)
        report = op.check_adjoint(n_trials=3, tol=1e-4)
        assert report.passed, report.summary()

    def test_light_field_forward_adjoint_shapes(self):
        op = LightFieldOperator(sx=8, sy=8, nu=3, nv=3)
        rng = np.random.default_rng(0)

        x = rng.standard_normal(op.x_shape).astype(np.float32)
        y = op.forward(x)
        assert y.shape == op.y_shape, f"forward shape {y.shape} != {op.y_shape}"

        y2 = rng.standard_normal(op.y_shape).astype(np.float32)
        x2 = op.adjoint(y2)
        assert x2.shape == op.x_shape, f"adjoint shape {x2.shape} != {op.x_shape}"

    def test_light_field_metadata(self):
        op = LightFieldOperator(sx=64, sy=64, nu=5, nv=5)
        meta = op.metadata()
        assert meta.modality == "light_field"
        assert meta.is_linear is True
        assert meta.x_shape == [64, 64, 5, 5]
        assert meta.y_shape == [64, 64]

    def test_light_field_serialize_roundtrip(self):
        op = LightFieldOperator(sx=8, sy=8, nu=3, nv=3)
        data = op.serialize()
        assert data["operator_id"] == "light_field"
        assert data["x_shape"] == [8, 8, 3, 3]
        assert data["y_shape"] == [8, 8]
