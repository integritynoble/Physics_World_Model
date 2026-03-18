"""test_mode2_calibration.py

Tests for Track C: Mode 2 Calibration Pipeline.

Tests
-----
* Poisson NLL, Gaussian NLL, mixed NLL
* Likelihood-aware scoring dispatch
* Sensitivity probe (identifiability)
* Filter space
* Beam search calibration
* Stop criteria
* Pipeline e2e (fit_operator_only + calibrate_and_reconstruct)
"""

from __future__ import annotations

import numpy as np
import pytest

from pwm_core.mismatch.scoring import (
    poisson_nll,
    gaussian_nll,
    mixed_nll,
    score_theta_likelihood,
)
from pwm_core.mismatch.identifiability import (
    IdentifiabilityReport,
    sensitivity_probe,
    filter_space,
)
from pwm_core.mismatch.parameterizations import ThetaSpace
from pwm_core.mismatch.calibrators import (
    CalibConfig,
    CalibResult,
    calibrate,
    random_theta,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_space():
    return ThetaSpace(
        name="test_space",
        params={
            "a": {"type": "float", "low": -5.0, "high": 5.0},
            "b": {"type": "float", "low": -5.0, "high": 5.0},
        },
    )


@pytest.fixture
def quadratic_forward():
    """Forward fn where y = (x-a)^2 + (x-b)^2, minimum at a=1, b=2."""
    x = np.linspace(0, 1, 64)
    def forward_fn(theta):
        a = theta.get("a", 0.0)
        b = theta.get("b", 0.0)
        return (x - a) ** 2 + (x - b) ** 2
    return forward_fn, x


# ---------------------------------------------------------------------------
# C.1: Likelihood-aware scoring
# ---------------------------------------------------------------------------

class TestLikelihoodScoring:

    def test_poisson_nll(self):
        y = np.array([1.0, 2.0, 3.0])
        yhat = np.array([1.0, 2.0, 3.0])
        nll = poisson_nll(y, yhat)
        # At y == yhat: sum(yhat - y*log(yhat)) = sum(y - y*log(y))
        expected = float(np.sum(y - y * np.log(y)))
        assert abs(nll - expected) < 1e-6

    def test_poisson_nll_zero_safe(self):
        y = np.array([1.0, 0.0, 2.0])
        yhat = np.array([0.0, 0.0, 2.0])
        nll = poisson_nll(y, yhat)
        assert np.isfinite(nll)

    def test_gaussian_nll(self):
        y = np.array([1.0, 2.0, 3.0])
        yhat = np.array([1.0, 2.0, 3.0])
        nll = gaussian_nll(y, yhat, sigma=1.0)
        # At y == yhat, sigma=1: 0.5*sum(0 + log(1)) = 0
        assert abs(nll) < 1e-6

    def test_gaussian_nll_with_sigma(self):
        y = np.array([0.0])
        yhat = np.array([1.0])
        sigma = 2.0
        nll = gaussian_nll(y, yhat, sigma=sigma)
        expected = 0.5 * (1.0 / 4.0 + np.log(4.0))
        assert abs(nll - expected) < 1e-6

    def test_mixed_nll(self):
        y = np.array([1.0, 2.0])
        yhat = np.array([1.5, 2.5])
        alpha = 0.3
        mixed = mixed_nll(y, yhat, alpha=alpha, sigma=1.0)
        expected = alpha * poisson_nll(y, yhat) + (1 - alpha) * gaussian_nll(y, yhat, 1.0)
        assert abs(mixed - expected) < 1e-6

    def test_score_theta_likelihood_gaussian(self):
        y = np.array([1.0, 2.0])
        yhat = np.array([1.0, 2.0])
        result = score_theta_likelihood(y, yhat, {"a": 0.0}, noise_model="gaussian")
        assert result.terms["noise_model"] == "gaussian"
        assert np.isfinite(result.total)

    def test_score_theta_likelihood_poisson(self):
        y = np.array([1.0, 2.0])
        yhat = np.array([1.0, 2.0])
        result = score_theta_likelihood(y, yhat, {"a": 0.0}, noise_model="poisson")
        assert result.terms["noise_model"] == "poisson"
        assert np.isfinite(result.total)

    def test_score_theta_likelihood_mixed(self):
        y = np.array([1.0, 2.0])
        yhat = np.array([1.0, 2.0])
        result = score_theta_likelihood(y, yhat, {"a": 0.0}, noise_model="mixed")
        assert result.terms["noise_model"] == "mixed"
        assert np.isfinite(result.total)


# ---------------------------------------------------------------------------
# C.2: Identifiability
# ---------------------------------------------------------------------------

class TestIdentifiability:

    def test_sensitivity_probe(self, simple_space):
        """Test that sensitivity probe identifies sensitive params."""
        # Forward: y = a * x + b. Changing a or b changes output.
        x = np.linspace(0, 1, 32)
        y_target = 2.0 * x + 1.0

        def fwd(theta):
            return theta["a"] * x + theta["b"]

        # Test AWAY from optimum so finite differences are non-zero
        theta = {"a": 1.0, "b": 0.0}
        report = sensitivity_probe(y_target, fwd, theta, simple_space, threshold=1e-6)

        assert "a" in report.param_sensitivities
        assert "b" in report.param_sensitivities
        assert report.param_sensitivities["a"] > 0
        assert report.param_sensitivities["b"] > 0
        assert len(report.identifiable_params) == 2

    def test_sensitivity_probe_insensitive(self):
        """Test that insensitive param gets frozen."""
        space = ThetaSpace(
            name="test",
            params={
                "active": {"type": "float", "low": -5.0, "high": 5.0},
                "dead": {"type": "float", "low": -5.0, "high": 5.0},
            },
        )
        x = np.ones(16)

        def fwd(theta):
            # Output depends only on 'active', not 'dead'
            return x * theta["active"]

        y_target = x * 3.0
        # Test away from optimum so 'active' has nonzero gradient
        theta = {"active": 1.0, "dead": 0.0}
        report = sensitivity_probe(y_target, fwd, theta, space, threshold=1e-4)

        assert "dead" in report.frozen_params
        assert "active" in report.identifiable_params

    def test_filter_space(self, simple_space):
        report = IdentifiabilityReport(
            param_sensitivities={"a": 1.0, "b": 0.0},
            frozen_params=["b"],
            identifiable_params=["a"],
        )
        filtered = filter_space(simple_space, report)
        assert "a" in filtered.params
        assert "b" not in filtered.params


# ---------------------------------------------------------------------------
# C.4/C.5: Calibration loop + stop criteria
# ---------------------------------------------------------------------------

class TestCalibrate:

    def test_calibrate_converges(self, simple_space):
        """Beam search should find approximately correct theta on simple problem."""
        x = np.linspace(0, 1, 32)
        true_a, true_b = 1.5, -0.5
        y_target = true_a * x + true_b

        def fwd(theta):
            return theta["a"] * x + theta["b"]

        cfg = CalibConfig(
            num_candidates=16,
            num_refine_steps=5,
            max_evals=200,
            seed=42,
            run_identifiability=False,
        )
        result = calibrate(y_target, fwd, simple_space, cfg)

        assert result.status in ("converged", "budget_exhausted")
        assert result.best_score < 1.0  # Should find something reasonable
        assert len(result.logs) > 0

    def test_calibrate_budget_exhausted(self, simple_space):
        """Test that calibration stops at max_evals."""
        x = np.ones(8)

        def fwd(theta):
            return x * theta.get("a", 0)

        cfg = CalibConfig(
            num_candidates=5,
            num_refine_steps=10,
            max_evals=5,
            seed=0,
            run_identifiability=False,
        )
        result = calibrate(x, fwd, simple_space, cfg)
        assert result.num_evals <= cfg.max_evals + 1  # Allow small overrun

    def test_calibrate_flat_landscape(self):
        """Test detection of flat landscape."""
        space = ThetaSpace(
            name="flat",
            params={"a": {"type": "float", "low": 0.0, "high": 1.0}},
        )

        def fwd(theta):
            return np.ones(8)  # constant output regardless of theta

        y = np.ones(8)
        cfg = CalibConfig(
            num_candidates=8,
            num_refine_steps=5,
            max_evals=200,
            seed=0,
            run_identifiability=False,
        )
        result = calibrate(y, fwd, space, cfg)
        assert result.status in ("flat_landscape", "converged")

    def test_calibrate_with_identifiability(self, simple_space):
        """Test that identifiability probe runs and freezes insensitive params."""
        x = np.linspace(0, 1, 16)

        def fwd(theta):
            return x * theta["a"]  # Only depends on 'a', not 'b'

        y = x * 2.0
        cfg = CalibConfig(
            num_candidates=8,
            num_refine_steps=3,
            max_evals=100,
            seed=42,
            run_identifiability=True,
        )
        result = calibrate(y, fwd, simple_space, cfg)
        assert result.identifiability is not None
        assert "b" in result.identifiability.frozen_params


# ---------------------------------------------------------------------------
# C.7: Pipeline e2e
# ---------------------------------------------------------------------------

class TestPipelineMode2:

    def test_pipeline_mode2_e2e(self, tmp_path):
        """Full runner.py calibrate_and_reconstruct path."""
        from pwm_core.api.types import (
            ExperimentSpec,
            ExperimentInput,
            ExperimentStates,
            InputMode,
            PhysicsState,
            TaskState,
            TaskKind,
            MismatchSpec,
            MismatchFitOperator,
        )
        from pwm_core.core.runner import run_pipeline

        # Create synthetic measurement matching operator shape (64x64 for widefield)
        rng = np.random.default_rng(42)
        x_true = rng.random((64, 64)).astype(np.float32)
        y = x_true + rng.normal(0, 0.01, x_true.shape).astype(np.float32)

        # Save to files
        y_path = str(tmp_path / "y.npy")
        x_path = str(tmp_path / "x.npy")
        np.save(y_path, y)
        np.save(x_path, x_true)

        spec = ExperimentSpec(
            id="test_mode2_e2e",
            input=ExperimentInput(
                mode=InputMode.measured,
                y_source=y_path,
                x_source=x_path,
            ),
            states=ExperimentStates(
                physics=PhysicsState(modality="widefield"),
                task=TaskState(kind=TaskKind.calibrate_and_reconstruct),
            ),
            mismatch=MismatchSpec(
                enabled=True,
                fit_operator=MismatchFitOperator(
                    enabled=True,
                    search={"num_candidates": 4, "max_evals": 10},
                    scoring={"noise_model": "gaussian"},
                ),
            ),
        )

        result = run_pipeline(spec, out_dir=str(tmp_path))
        assert result.calib is not None
        assert result.calib.num_evals > 0

    def test_fit_operator_only_e2e(self, tmp_path):
        """fit_operator_only task kind."""
        from pwm_core.api.types import (
            ExperimentSpec,
            ExperimentInput,
            ExperimentStates,
            InputMode,
            PhysicsState,
            TaskState,
            TaskKind,
            MismatchSpec,
            MismatchFitOperator,
        )
        from pwm_core.core.runner import run_pipeline

        rng = np.random.default_rng(42)
        y = rng.random((64, 64)).astype(np.float32)
        y_path = str(tmp_path / "y.npy")
        np.save(y_path, y)

        spec = ExperimentSpec(
            id="test_fit_only",
            input=ExperimentInput(
                mode=InputMode.measured,
                y_source=y_path,
            ),
            states=ExperimentStates(
                physics=PhysicsState(modality="widefield"),
                task=TaskState(kind=TaskKind.fit_operator_only),
            ),
            mismatch=MismatchSpec(
                enabled=True,
                fit_operator=MismatchFitOperator(
                    enabled=True,
                    search={"num_candidates": 4, "max_evals": 8},
                ),
            ),
        )

        result = run_pipeline(spec, out_dir=str(tmp_path))
        assert result.calib is not None
        assert result.diagnosis is not None
        assert result.diagnosis.verdict == "calibration_only"
        assert result.recon == []
