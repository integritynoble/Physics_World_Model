"""Tests for pwm_core.objectives: NLL + prior."""

import numpy as np
import pytest

from pwm_core.objectives.base import (
    ObjectiveSpec,
    NegLogLikelihood,
    PoissonNLL,
    GaussianNLL,
    ComplexGaussianNLL,
    MixedPoissonGaussianNLL,
    HuberNLL,
    TukeyBiweightNLL,
    OBJECTIVE_REGISTRY,
    build_objective,
)
from pwm_core.objectives.prior import PriorSpec


# ---------------------------------------------------------------------------
# NLL correctness
# ---------------------------------------------------------------------------


class TestPoissonNLL:
    def test_nonneg(self):
        nll = PoissonNLL()
        y = np.abs(np.random.randn(64))
        yhat = np.abs(np.random.randn(64)) + 0.1
        assert nll(y, yhat) >= 0

    def test_zero_residual(self):
        nll = PoissonNLL()
        y = np.ones(10) * 5.0
        assert nll(y, y) < nll(y, y + 1.0)

    def test_gradient_shape(self):
        nll = PoissonNLL()
        y = np.abs(np.random.randn(8, 8)) + 0.1
        yhat = np.abs(np.random.randn(8, 8)) + 0.1
        grad = nll.gradient(y, yhat)
        assert grad.shape == y.shape


class TestGaussianNLL:
    def test_minimum_at_match(self):
        nll = GaussianNLL(sigma=1.0)
        y = np.random.randn(64)
        assert nll(y, y) < nll(y, y + 0.5)

    def test_gradient_finite_diff(self):
        nll = GaussianNLL(sigma=0.5)
        y = np.random.randn(16)
        yhat = np.random.randn(16)
        grad = nll.gradient(y, yhat)

        eps = 1e-5
        for i in range(min(5, len(yhat))):
            yhat_p = yhat.copy()
            yhat_p[i] += eps
            yhat_m = yhat.copy()
            yhat_m[i] -= eps
            fd = (nll(y, yhat_p) - nll(y, yhat_m)) / (2 * eps)
            assert abs(grad[i] - fd) < 1e-4, f"idx {i}: grad={grad[i]:.6f}, fd={fd:.6f}"


class TestComplexGaussianNLL:
    def test_complex_input(self):
        nll = ComplexGaussianNLL(sigma=0.1)
        y = np.random.randn(32) + 1j * np.random.randn(32)
        yhat = y + 0.01 * (np.random.randn(32) + 1j * np.random.randn(32))
        val = nll(y, yhat)
        assert isinstance(val, float)

    def test_minimum_at_match(self):
        nll = ComplexGaussianNLL(sigma=1.0)
        y = np.random.randn(16) + 1j * np.random.randn(16)
        assert nll(y, y) < nll(y, y + 0.5 + 0.5j)


class TestMixedPoissonGaussianNLL:
    def test_convex_combination(self):
        nll = MixedPoissonGaussianNLL(alpha=0.5, sigma=1.0)
        y = np.abs(np.random.randn(64)) + 0.1
        yhat = np.abs(np.random.randn(64)) + 0.1
        val = nll(y, yhat)
        assert isinstance(val, float)

    def test_gradient_shape(self):
        nll = MixedPoissonGaussianNLL(alpha=0.3, sigma=0.5)
        y = np.abs(np.random.randn(8, 8)) + 0.1
        yhat = np.abs(np.random.randn(8, 8)) + 0.1
        grad = nll.gradient(y, yhat)
        assert grad.shape == y.shape


class TestHuberNLL:
    def test_small_residuals_quadratic(self):
        nll = HuberNLL(delta=10.0)
        y = np.ones(10)
        yhat = y + 0.1
        val = nll(y, yhat)
        expected = 0.5 * 10 * 0.01  # 0.5 * n * eps^2
        assert abs(val - expected) < 1e-8

    def test_gradient_shape(self):
        nll = HuberNLL(delta=1.0)
        y = np.random.randn(16)
        yhat = np.random.randn(16)
        assert nll.gradient(y, yhat).shape == y.shape


class TestTukeyBiweightNLL:
    def test_outlier_rejection(self):
        nll = TukeyBiweightNLL(c=1.0)
        y = np.zeros(10)
        # Normal residual
        yhat1 = np.ones(10) * 0.5
        # Outlier
        yhat2 = np.ones(10) * 100.0
        grad1 = nll.gradient(y, yhat1)
        grad2 = nll.gradient(y, yhat2)
        # Tukey should zero-out gradient for outliers
        assert np.all(grad2 == 0.0)
        assert np.any(grad1 != 0.0)


# ---------------------------------------------------------------------------
# Registry + builder
# ---------------------------------------------------------------------------


class TestObjectiveRegistry:
    def test_all_six_registered(self):
        expected = {"poisson", "gaussian", "complex_gaussian",
                    "mixed_poisson_gaussian", "huber", "tukey_biweight"}
        assert expected.issubset(set(OBJECTIVE_REGISTRY.keys()))

    def test_build_gaussian(self):
        spec = ObjectiveSpec(kind="gaussian", params={"sigma": 0.5})
        obj = build_objective(spec)
        assert isinstance(obj, GaussianNLL)

    def test_build_unknown_raises(self):
        spec = ObjectiveSpec(kind="nonexistent")
        with pytest.raises(KeyError):
            build_objective(spec)

    def test_round_trip(self):
        for kind in OBJECTIVE_REGISTRY:
            spec = ObjectiveSpec(kind=kind)
            obj = build_objective(spec)
            assert isinstance(obj, NegLogLikelihood)


# ---------------------------------------------------------------------------
# PriorSpec
# ---------------------------------------------------------------------------


class TestPriorSpec:
    def test_valid_kinds(self):
        for kind in ["tv", "l1_wavelet", "low_rank", "deep_prior", "l2", "none"]:
            ps = PriorSpec(kind=kind, weight=0.1)
            assert ps.kind == kind

    def test_invalid_kind_raises(self):
        with pytest.raises(Exception):
            PriorSpec(kind="invalid_prior")

    def test_negative_weight_raises(self):
        with pytest.raises(Exception):
            PriorSpec(kind="tv", weight=-1.0)

    def test_default_none(self):
        ps = PriorSpec()
        assert ps.kind == "none"
        assert ps.weight == 0.0
