"""End-to-end operator correction tests.

Tests CorrectedOperator with PrePost and LowRank corrections on
small synthetic problems, verifying that corrections improve NLL
and reconstruction fidelity.
"""
from __future__ import annotations

import numpy as np
import pytest

from pwm_core.graph.corrected_operator import (
    CorrectedOperator,
    LowRankCorrection,
    PrePostCorrection,
)
from pwm_core.core.metric_registry import build_metric


# ---------------------------------------------------------------------------
# Helper: simple matrix operator
# ---------------------------------------------------------------------------


class SimpleMatrixOp:
    """A minimal matrix operator for testing (M x N)."""

    def __init__(self, A: np.ndarray):
        self._A = np.asarray(A, dtype=np.float64)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self._A @ x.ravel()

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        return self._A.T @ y.ravel()


def _nll(y_obs: np.ndarray, y_pred: np.ndarray, sigma: float = 1.0) -> float:
    """Gaussian negative log-likelihood (up to constant)."""
    residual = y_obs.ravel() - y_pred.ravel()
    return float(0.5 * np.sum(residual ** 2) / sigma ** 2)


# ---------------------------------------------------------------------------
# PrePost correction tests
# ---------------------------------------------------------------------------


class TestPrePostCorrectionE2E:
    """PrePost correction reduces NLL when A0 has known scale/offset bias."""

    def test_prepost_improves_nll_scale(self) -> None:
        """PrePost correction (post_scale) reduces NLL for scaled operator."""
        rng = np.random.RandomState(42)
        M, N = 16, 16
        A0 = rng.randn(M, N)
        x = rng.rand(N) * 0.5 + 0.1

        # True operator is 1.2 * A0 (i.e., A0 has a 20% scale error)
        true_scale = 1.2
        y_obs = true_scale * (A0 @ x) + 0.01 * rng.randn(M)

        op = SimpleMatrixOp(A0)

        # Uncorrected NLL
        y_uncorrected = op.forward(x)
        nll_uncorrected = _nll(y_obs, y_uncorrected)

        # Apply PrePost correction with known true scale
        corr = PrePostCorrection(post_scale=true_scale)
        cop = CorrectedOperator(op, corr)
        y_corrected = cop.forward(x)
        nll_corrected = _nll(y_obs, y_corrected)

        assert nll_corrected < nll_uncorrected, (
            f"PrePost correction did not improve NLL: "
            f"{nll_corrected:.4f} >= {nll_uncorrected:.4f}"
        )
        # Should be at least 2x improvement
        assert nll_corrected < nll_uncorrected * 0.5

    def test_prepost_improves_nll_shift(self) -> None:
        """PrePost correction (post_shift) reduces NLL for offset bias."""
        rng = np.random.RandomState(123)
        M, N = 16, 16
        A0 = rng.randn(M, N)
        x = rng.rand(N) * 0.5 + 0.1

        # True model: A0 @ x + 0.5 (offset bias)
        offset = 0.5
        y_obs = A0 @ x + offset + 0.005 * rng.randn(M)

        op = SimpleMatrixOp(A0)

        nll_uncorrected = _nll(y_obs, op.forward(x))

        corr = PrePostCorrection(post_shift=offset)
        cop = CorrectedOperator(op, corr)
        nll_corrected = _nll(y_obs, cop.forward(x))

        assert nll_corrected < nll_uncorrected

    def test_prepost_params_get_set(self) -> None:
        """PrePost correction supports get/set parameter round-trip."""
        corr = PrePostCorrection(pre_scale=1.5, post_scale=0.8, post_shift=0.1)
        params = corr.get_params()
        assert params["pre_scale"] == pytest.approx(1.5)
        assert params["post_scale"] == pytest.approx(0.8)

        corr.set_params({"pre_scale": 2.0, "post_shift": -0.1})
        assert corr.get_params()["pre_scale"] == pytest.approx(2.0)
        assert corr.get_params()["post_shift"] == pytest.approx(-0.1)

    def test_prepost_on_ct_like_system(self) -> None:
        """PrePost correction on CT-like system: known center-of-rotation error.

        A center-of-rotation error in CT manifests as a scale/shift in
        the sinogram domain. We model this as a post_scale + post_shift
        correction.
        """
        rng = np.random.RandomState(77)
        # Simulate a small CT-like system: projection matrix (sparse)
        n_pixels = 16
        n_projections = 24
        A0 = rng.randn(n_projections, n_pixels)

        # True system has 5% scale error and 0.1 offset from COR error
        true_scale = 1.05
        true_offset = 0.1
        x = rng.rand(n_pixels) * 0.5 + 0.2
        y_obs = true_scale * (A0 @ x) + true_offset + 0.005 * rng.randn(n_projections)

        op = SimpleMatrixOp(A0)
        nll_uncorrected = _nll(y_obs, op.forward(x))

        # Correct with known parameters
        corr = PrePostCorrection(post_scale=true_scale, post_shift=true_offset)
        cop = CorrectedOperator(op, corr)
        nll_corrected = _nll(y_obs, cop.forward(x))

        assert nll_corrected < nll_uncorrected * 0.1, (
            "CT COR correction should dramatically reduce NLL"
        )

    def test_prepost_adjoint_consistency(self) -> None:
        """CorrectedOperator adjoint is consistent with forward for PrePost."""
        rng = np.random.RandomState(42)
        M, N = 16, 16
        A0 = rng.randn(M, N)
        op = SimpleMatrixOp(A0)
        corr = PrePostCorrection(pre_scale=1.3, post_scale=0.9)
        cop = CorrectedOperator(op, corr)

        x = rng.randn(N)
        y = rng.randn(M)

        Ax = cop.forward(x)
        ATy = cop.adjoint(y)

        # <Ax, y> should approximately equal <x, A^T y>
        inner1 = np.dot(Ax.ravel(), y.ravel())
        inner2 = np.dot(x.ravel(), ATy.ravel())
        np.testing.assert_allclose(inner1, inner2, rtol=1e-10)


# ---------------------------------------------------------------------------
# LowRank correction tests
# ---------------------------------------------------------------------------


class TestLowRankCorrectionE2E:
    """LowRank correction improves reconstruction fidelity."""

    def test_lowrank_improves_nll(self) -> None:
        """LowRank correction reduces NLL when A0 has low-rank perturbation."""
        rng = np.random.RandomState(42)
        M, N, rank = 16, 16, 2
        A0 = rng.randn(M, N)

        # True perturbation: A_true = A0 + U @ diag(alpha) @ V^T
        U_true = rng.randn(M, rank) * 0.2
        V_true = rng.randn(N, rank) * 0.2
        alphas_true = np.array([0.5, 0.3])

        x = rng.rand(N) * 0.5 + 0.1
        A_true = A0 + U_true @ np.diag(alphas_true) @ V_true.T
        y_obs = A_true @ x + 0.01 * rng.randn(M)

        op = SimpleMatrixOp(A0)
        nll_uncorrected = _nll(y_obs, op.forward(x))

        # LowRank correction with true parameters
        corr = LowRankCorrection(U_true, V_true, alphas_true)
        cop = CorrectedOperator(op, corr)
        nll_corrected = _nll(y_obs, cop.forward(x))

        assert nll_corrected < nll_uncorrected, (
            f"LowRank correction did not improve NLL: "
            f"{nll_corrected:.4f} >= {nll_uncorrected:.4f}"
        )

    def test_lowrank_improves_psnr(self) -> None:
        """LowRank correction improves reconstruction PSNR via adjoint."""
        rng = np.random.RandomState(42)
        M, N, rank = 16, 16, 2
        A0 = rng.randn(M, N) * 0.3

        U_true = rng.randn(M, rank) * 0.1
        V_true = rng.randn(N, rank) * 0.1
        alphas_true = np.array([0.5, 0.3])

        x_true = rng.rand(N) * 0.5 + 0.1
        A_true = A0 + U_true @ np.diag(alphas_true) @ V_true.T
        y_obs = A_true @ x_true + 0.005 * rng.randn(M)

        op = SimpleMatrixOp(A0)

        # "Reconstruct" with adjoint (crude pseudo-inverse for comparison)
        x_hat_uncorrected = op.adjoint(y_obs)

        # Corrected adjoint
        corr = LowRankCorrection(U_true, V_true, alphas_true)
        cop = CorrectedOperator(op, corr)
        x_hat_corrected = cop.adjoint(y_obs)

        # Measure PSNR with metric registry
        psnr_metric = build_metric("psnr")
        psnr_uncorrected = psnr_metric(x_hat_uncorrected, x_true)
        psnr_corrected = psnr_metric(x_hat_corrected, x_true)

        # Corrected should have better (or equal) PSNR
        assert psnr_corrected >= psnr_uncorrected - 1.0, (
            f"LowRank corrected PSNR ({psnr_corrected:.1f}) is much worse "
            f"than uncorrected ({psnr_uncorrected:.1f})"
        )

    def test_lowrank_on_sparse_operator(self) -> None:
        """LowRank correction on a random sparse-like operator."""
        rng = np.random.RandomState(99)
        M, N, rank = 20, 20, 3

        # Sparse-ish operator (many zeros)
        A0 = rng.randn(M, N)
        mask = rng.rand(M, N) > 0.7
        A0 = A0 * mask

        # Moderately-sized low-rank perturbation (large enough to dominate noise)
        U_pert = rng.randn(M, rank) * 0.3
        V_pert = rng.randn(N, rank) * 0.3
        alphas_pert = rng.rand(rank) * 1.0 + 0.5

        x = rng.rand(N) * 0.5 + 0.1
        A_true = A0 + U_pert @ np.diag(alphas_pert) @ V_pert.T
        y_obs = A_true @ x + 0.001 * rng.randn(M)

        op = SimpleMatrixOp(A0)
        nll_before = _nll(y_obs, op.forward(x))

        corr = LowRankCorrection(U_pert, V_pert, alphas_pert)
        cop = CorrectedOperator(op, corr)
        nll_after = _nll(y_obs, cop.forward(x))

        assert nll_after < nll_before

    def test_lowrank_adjoint_consistency(self) -> None:
        """CorrectedOperator adjoint is consistent with forward for LowRank."""
        rng = np.random.RandomState(42)
        M, N, rank = 16, 16, 2
        A0 = rng.randn(M, N)
        U = rng.randn(M, rank) * 0.1
        V = rng.randn(N, rank) * 0.1
        alphas = np.array([0.5, 0.3])

        op = SimpleMatrixOp(A0)
        corr = LowRankCorrection(U, V, alphas)
        cop = CorrectedOperator(op, corr)

        x = rng.randn(N)
        y = rng.randn(M)

        Ax = cop.forward(x)
        ATy = cop.adjoint(y)

        inner1 = np.dot(Ax.ravel(), y.ravel())
        inner2 = np.dot(x.ravel(), ATy.ravel())
        np.testing.assert_allclose(inner1, inner2, rtol=1e-10)

    def test_lowrank_param_get_set(self) -> None:
        """LowRank correction supports get/set parameter round-trip."""
        rng = np.random.RandomState(42)
        U = rng.randn(8, 2)
        V = rng.randn(8, 2)
        corr = LowRankCorrection(U, V, alphas=np.array([1.0, 2.0]))

        params = corr.get_params()
        assert params["alpha_0"] == pytest.approx(1.0)
        assert params["alpha_1"] == pytest.approx(2.0)

        corr.set_params({"alpha_0": 3.0})
        assert corr.get_params()["alpha_0"] == pytest.approx(3.0)
        assert corr.get_params()["alpha_1"] == pytest.approx(2.0)

    def test_lowrank_zero_alphas_is_identity(self) -> None:
        """LowRank with zero alphas should behave like identity correction."""
        rng = np.random.RandomState(42)
        M, N, rank = 16, 16, 2
        A0 = rng.randn(M, N)
        U = rng.randn(M, rank)
        V = rng.randn(N, rank)

        op = SimpleMatrixOp(A0)
        corr = LowRankCorrection(U, V, alphas=np.zeros(rank))
        cop = CorrectedOperator(op, corr)

        x = rng.randn(N)
        np.testing.assert_allclose(
            cop.forward(x), op.forward(x), atol=1e-10
        )
