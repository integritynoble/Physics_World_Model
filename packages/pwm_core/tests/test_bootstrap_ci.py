"""test_bootstrap_ci.py

Synthetic test verifying that bootstrap_correction() achieves >= 90%
coverage of the true parameter values over 100 independent trials.

Coverage definition: in at least 90 of the 100 trials, the 95% CI
produced by bootstrap_correction() contains the true parameter value.

The test uses a simple linear model:
    y = gain * x + bias + noise
where gain and bias are the "true" parameters, and the correction function
estimates them via least-squares on bootstrap-resampled data.

Run:
    pytest -q packages/pwm_core/tests/test_bootstrap_ci.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure pwm_core is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pwm_core.mismatch.uncertainty import bootstrap_correction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_linear_problem(
    n_samples: int = 50,
    gain_true: float = 1.0,
    bias_true: float = 0.0,
    noise_sigma: float = 0.05,
    seed: int = 0,
):
    """Create a synthetic calibration dataset.

    Returns
    -------
    data : np.ndarray, shape (n_samples, 2)
        Column 0 = x (inputs), column 1 = y (noisy outputs).
    gain_true, bias_true : float
        Ground-truth parameters.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.5, 2.0, size=n_samples)
    y = gain_true * x + bias_true + rng.normal(0, noise_sigma, size=n_samples)
    data = np.column_stack([x, y])
    return data, gain_true, bias_true


def _correction_fn(data: np.ndarray) -> dict:
    """Estimate gain and bias from data via least-squares.

    Parameters
    ----------
    data : np.ndarray, shape (n, 2)
        Column 0 = x, column 1 = y.

    Returns
    -------
    dict
        Keys: "gain", "bias", "psnr" (proxy quality metric).
    """
    x = data[:, 0]
    y = data[:, 1]
    n = len(x)
    if n < 2:
        return {"gain": 1.0, "bias": 0.0, "psnr": 0.0}

    # Least-squares: y = gain * x + bias
    A = np.column_stack([x, np.ones(n)])
    params, residuals, _, _ = np.linalg.lstsq(A, y, rcond=None)
    gain_est, bias_est = float(params[0]), float(params[1])

    # Proxy PSNR: inverse of residual energy (higher is better)
    y_pred = gain_est * x + bias_est
    mse = np.mean((y - y_pred) ** 2)
    psnr = float(10 * np.log10(1.0 / max(mse, 1e-12)))

    return {"gain": gain_est, "bias": bias_est, "psnr": psnr}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBootstrapCoverage:
    """Verify that bootstrap 95% CI achieves >= 90% coverage."""

    N_TRIALS = 100
    TARGET_COVERAGE = 0.90  # must achieve at least 90%
    K_BOOTSTRAP = 100      # enough resamples for accurate 2.5/97.5 percentiles
    N_SAMPLES = 100        # enough samples for stable least-squares
    NOISE_SIGMA = 0.05

    def test_coverage_gain_and_bias(self):
        """Run N_TRIALS independent synthetic experiments and check
        that the 95% CI covers the true gain and bias in >= 90% of
        trials."""
        gain_true = 1.3
        bias_true = -0.15

        gain_covered = 0
        bias_covered = 0

        for trial in range(self.N_TRIALS):
            data, _, _ = _make_linear_problem(
                n_samples=self.N_SAMPLES,
                gain_true=gain_true,
                bias_true=bias_true,
                noise_sigma=self.NOISE_SIGMA,
                seed=trial * 1000,  # distinct data per trial
            )

            result = bootstrap_correction(
                correction_fn=_correction_fn,
                data=data,
                K=self.K_BOOTSTRAP,
                seed=trial,  # distinct bootstrap seeds per trial
            )

            # Check coverage for gain
            gain_ci = result.theta_uncertainty["gain"]
            if gain_ci[0] <= gain_true <= gain_ci[1]:
                gain_covered += 1

            # Check coverage for bias
            bias_ci = result.theta_uncertainty["bias"]
            if bias_ci[0] <= bias_true <= bias_ci[1]:
                bias_covered += 1

        gain_coverage = gain_covered / self.N_TRIALS
        bias_coverage = bias_covered / self.N_TRIALS

        # Report
        print(
            f"\nBootstrap coverage over {self.N_TRIALS} trials: "
            f"gain={gain_coverage:.2%}, bias={bias_coverage:.2%}"
        )

        assert gain_coverage >= self.TARGET_COVERAGE, (
            f"Gain coverage {gain_coverage:.2%} < {self.TARGET_COVERAGE:.0%}"
        )
        assert bias_coverage >= self.TARGET_COVERAGE, (
            f"Bias coverage {bias_coverage:.2%} < {self.TARGET_COVERAGE:.0%}"
        )

    def test_correction_result_schema(self):
        """Verify that CorrectionResult has all required fields and
        passes validation."""
        data, _, _ = _make_linear_problem(seed=999)
        result = bootstrap_correction(
            correction_fn=_correction_fn,
            data=data,
            K=self.K_BOOTSTRAP,
            seed=42,
        )

        # Required fields exist
        assert "gain" in result.theta_corrected
        assert "bias" in result.theta_corrected
        assert "gain" in result.theta_uncertainty
        assert "bias" in result.theta_uncertainty

        # CI structure: [lower, upper]
        for param, ci in result.theta_uncertainty.items():
            assert len(ci) == 2, f"CI for {param} has {len(ci)} elements"
            assert ci[0] <= ci[1], f"CI for {param}: lower > upper"

        # Convergence curve non-empty
        assert len(result.convergence_curve) == self.K_BOOTSTRAP

        # Bootstrap metadata
        assert len(result.bootstrap_seeds) == self.K_BOOTSTRAP
        assert len(result.resampling_indices) == self.K_BOOTSTRAP
        for indices in result.resampling_indices:
            assert len(indices) > 0
            assert all(0 <= i < self.N_SAMPLES for i in indices)

        # n_evaluations == K
        assert result.n_evaluations == self.K_BOOTSTRAP

    def test_deterministic_reproducibility(self):
        """Two runs with same seed and data must produce identical results."""
        data, _, _ = _make_linear_problem(seed=123)

        r1 = bootstrap_correction(_correction_fn, data, K=10, seed=0)
        r2 = bootstrap_correction(_correction_fn, data, K=10, seed=0)

        assert r1.theta_corrected == r2.theta_corrected
        assert r1.theta_uncertainty == r2.theta_uncertainty
        assert r1.convergence_curve == r2.convergence_curve
        assert r1.bootstrap_seeds == r2.bootstrap_seeds
        assert r1.resampling_indices == r2.resampling_indices

    def test_empty_data_raises(self):
        """Passing empty data must raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            bootstrap_correction(
                _correction_fn, np.array([]).reshape(0, 2), K=5
            )

    def test_improvement_db_with_baseline(self):
        """Verify improvement_db is computed relative to baseline when
        provided."""
        data, _, _ = _make_linear_problem(seed=42)
        baseline_psnr = 10.0

        result = bootstrap_correction(
            _correction_fn, data, K=5, seed=0,
            psnr_without_correction=baseline_psnr,
        )

        median_psnr = float(np.median(result.convergence_curve))
        expected_improvement = median_psnr - baseline_psnr
        assert abs(result.improvement_db - expected_improvement) < 1e-6
