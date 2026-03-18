"""Tier-3 Learned Surrogate Wrapper Template
=============================================

You implement: forward_model() and (optionally) adjoint_model()
We handle: uncertainty calibration, validation, tier tagging, serialization

Tier-3 primitives wrap neural network forward models (learned surrogates)
that approximate expensive physics simulations. They MUST provide calibrated
uncertainty estimates.

Example: A neural network trained to approximate Mie scattering

Usage:
    1. Copy this file
    2. Implement forward_model() (and optionally adjoint_model())
    3. Run the self-test at the bottom
    4. Submit RFC + PR for steward review (governance lane)

IMPORTANT:
    Tier-3 primitives go through the governance lane (90-day RFC) because
    they introduce learned approximations that affect all downstream results.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


class Tier3Adapter:
    """Wraps learned surrogate models into valid PrimitiveOps.

    Contributors implement only:
    - forward_model(x, params) -> y  (the neural forward model)
    - adjoint_model(y, params) -> x  (optional; auto-derived if linear)

    The adapter handles:
    - Uncertainty calibration (residual-based confidence intervals)
    - Agreement check with Tier-2 reference (when available)
    - Shape checking and dtype normalization
    - Physics tier tag injection
    - Serialization boilerplate
    """

    def __init__(
        self,
        forward_model,
        adjoint_model=None,
        params: Dict[str, Any] = None,
        input_shape: Tuple[int, ...] = (64, 64),
        output_shape: Tuple[int, ...] = (64, 64),
        name: str = "tier3_custom",
        uncertainty_model=None,
    ):
        self.forward_model_fn = forward_model
        self.adjoint_model_fn = adjoint_model
        self.uncertainty_model_fn = uncertainty_model
        self._params = dict(params or {})
        self._input_shape = input_shape
        self._output_shape = output_shape
        self.primitive_id = name
        self._is_linear = False  # Learned models are generally nonlinear
        self._physics_tier = "tier3_learned"
        self._physics_subrole = "surrogate"
        self._calibration_data: Optional[Dict[str, float]] = None

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
        """Apply the learned forward model."""
        x = np.asarray(x, dtype=np.float64)
        y = self.forward_model_fn(x, self._params)
        return np.asarray(y, dtype=np.float64)

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        """Apply the adjoint (transpose of Jacobian).

        If no adjoint_model is provided, uses a simple transpose
        approximation (only valid for near-linear models).
        """
        y = np.asarray(y, dtype=np.float64)
        if self.adjoint_model_fn is not None:
            x = self.adjoint_model_fn(y, self._params)
        else:
            # Fallback: identity mapping (user should override)
            x = y
        return np.asarray(x, dtype=np.float64)

    def forward_with_uncertainty(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply forward model and return (prediction, uncertainty).

        Uncertainty is per-pixel standard deviation.
        """
        y = self.forward(x)

        if self.uncertainty_model_fn is not None:
            sigma = self.uncertainty_model_fn(x, self._params)
            sigma = np.asarray(sigma, dtype=np.float64)
        elif self._calibration_data is not None:
            # Use calibrated residual-based uncertainty
            sigma = np.full_like(y, self._calibration_data["mean_residual_std"])
        else:
            # No uncertainty model: report NaN
            sigma = np.full_like(y, np.nan)

        return y, sigma

    def calibrate_uncertainty(
        self,
        x_samples: np.ndarray,
        y_reference: np.ndarray,
        n_samples: int = 100,
    ) -> Dict[str, float]:
        """Calibrate uncertainty estimates against reference data.

        Computes residual statistics between the learned model and
        reference (Tier-2) outputs to establish confidence intervals.

        Parameters
        ----------
        x_samples : ndarray
            Input samples, shape (n_samples, *input_shape).
        y_reference : ndarray
            Reference outputs (from Tier-2), shape (n_samples, *output_shape).
        n_samples : int
            Number of samples to use.

        Returns
        -------
        dict
            Calibration statistics.
        """
        residuals = []
        n = min(n_samples, len(x_samples))

        for i in range(n):
            y_pred = self.forward(x_samples[i])
            residual = y_pred - y_reference[i]
            residuals.append(residual)

        residuals = np.array(residuals)
        self._calibration_data = {
            "mean_residual_std": float(np.std(residuals)),
            "max_residual": float(np.max(np.abs(residuals))),
            "mean_residual_norm": float(np.mean(np.linalg.norm(
                residuals.reshape(n, -1), axis=1
            ))),
            "n_calibration_samples": n,
        }

        return self._calibration_data

    def check_agreement(
        self,
        tier2_forward,
        n_trials: int = 10,
        tol: float = 0.1,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Check agreement between this Tier-3 model and a Tier-2 reference.

        Parameters
        ----------
        tier2_forward : callable
            Tier-2 forward model: (x, params) -> y
        n_trials : int
            Number of random inputs to test.
        tol : float
            Maximum acceptable relative error.

        Returns
        -------
        dict
            Keys: passed (bool), max_rel_err (float), mean_rel_err (float).
        """
        rng = np.random.default_rng(seed)
        rel_errors = []

        for _ in range(n_trials):
            x = rng.standard_normal(self._input_shape)
            y_t3 = self.forward(x)
            y_t2 = tier2_forward(x, self._params)

            denom = max(np.linalg.norm(y_t2), 1e-30)
            rel_err = np.linalg.norm(y_t3 - y_t2) / denom
            rel_errors.append(float(rel_err))

        max_err = max(rel_errors)
        mean_err = float(np.mean(rel_errors))
        passed = max_err < tol

        print(
            f"Tier-3 vs Tier-2 agreement: max_rel_err={max_err:.2e}, "
            f"mean_rel_err={mean_err:.2e}, tol={tol:.2e}, "
            f"{'PASS' if passed else 'FAIL'}"
        )

        return {
            "passed": passed,
            "max_rel_err": max_err,
            "mean_rel_err": mean_err,
            "n_trials": n_trials,
            "tol": tol,
        }

    def serialize(self) -> Dict[str, Any]:
        """Serialize the adapter configuration (not the model weights)."""
        result = {
            "primitive_id": self.primitive_id,
            "params": {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in self._params.items()
            },
            "physics_tier": self._physics_tier,
            "physics_subrole": self._physics_subrole,
            "input_shape": list(self._input_shape),
            "output_shape": list(self._output_shape),
            "is_linear": self._is_linear,
        }
        if self._calibration_data:
            result["uncertainty_calibration"] = self._calibration_data
        return result


# ===================================================================
# EXAMPLE: Simple learned scaling (replace with your neural model)
# ===================================================================


def example_forward_model(x: np.ndarray, params: dict) -> np.ndarray:
    """YOUR NEURAL FORWARD MODEL HERE.

    Replace with your trained model inference.
    Example: a simple nonlinear scaling (placeholder).
    """
    scale = params.get("scale", 0.8)
    bias = params.get("bias", 0.01)
    return x * scale + bias


def example_adjoint_model(y: np.ndarray, params: dict) -> np.ndarray:
    """YOUR ADJOINT MODEL HERE (optional).

    Replace with the transpose of your model's Jacobian, or leave None
    for identity fallback.
    """
    scale = params.get("scale", 0.8)
    return y * scale  # Adjoint of linear part (ignoring bias)


def example_uncertainty_model(x: np.ndarray, params: dict) -> np.ndarray:
    """YOUR UNCERTAINTY MODEL HERE (optional).

    Returns per-pixel standard deviation.
    Replace with ensemble disagreement, MC dropout, etc.
    """
    return np.full_like(x, 0.05)  # Constant uncertainty placeholder


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Create adapter
    adapter = Tier3Adapter(
        forward_model=example_forward_model,
        adjoint_model=example_adjoint_model,
        uncertainty_model=example_uncertainty_model,
        params={"scale": 0.8, "bias": 0.01},
        input_shape=(64, 64),
        output_shape=(64, 64),
        name="example_tier3",
    )

    # 2. Forward + uncertainty
    rng = np.random.default_rng(42)
    x = rng.standard_normal((64, 64))
    y, sigma = adapter.forward_with_uncertainty(x)
    assert y.shape == (64, 64), f"Forward shape: {y.shape}"
    assert sigma.shape == (64, 64), f"Uncertainty shape: {sigma.shape}"
    assert np.all(sigma > 0), "Uncertainty must be positive"

    # 3. Adjoint shape check
    x_back = adapter.adjoint(y)
    assert x_back.shape == (64, 64), f"Adjoint shape: {x_back.shape}"

    # 4. Agreement with Tier-2 reference
    def tier2_reference(x, params):
        """Simulated Tier-2 'ground truth' forward model."""
        return x * params.get("scale", 0.8)

    agreement = adapter.check_agreement(tier2_reference, n_trials=10, tol=0.2)
    assert agreement["passed"], f"Agreement check failed: {agreement}"

    # 5. Uncertainty calibration
    n_cal = 50
    x_cal = rng.standard_normal((n_cal, 64, 64))
    y_cal = np.array([tier2_reference(x_cal[i], {"scale": 0.8}) for i in range(n_cal)])
    cal_stats = adapter.calibrate_uncertainty(x_cal, y_cal, n_samples=n_cal)
    assert cal_stats["mean_residual_std"] >= 0
    print(f"  Calibration: mean_residual_std={cal_stats['mean_residual_std']:.4f}")

    # 6. Serialization
    s = adapter.serialize()
    assert s["physics_tier"] == "tier3_learned"
    assert "uncertainty_calibration" in s

    print("All self-tests PASSED")
