"""Tier-3 Adapter: Learned Surrogates
======================================

Wraps neural network forward models (learned surrogates) that approximate
expensive physics simulations. Tier-3 primitives MUST provide calibrated
uncertainty estimates.

Examples:
- Neural radiance fields (NeRF) forward model
- 3D Gaussian splatting renderer
- Physics-informed neural networks (PINNs)
- Learned Mie scattering surrogates
- Deep image prior forward models

IMPORTANT:
    Tier-3 primitives go through the governance lane (90-day RFC) because
    they introduce learned approximations that affect all downstream results.

Usage:
    from pwm_core.graph.adapters import Tier3Adapter

    adapter = Tier3Adapter(
        forward_kernel=my_neural_forward,
        adjoint_kernel=my_neural_adjoint,
        params={"model_path": "weights.pth"},
        input_shape=(128, 128),
        output_shape=(128, 128),
        name="neural_scatter",
        uncertainty_kernel=my_uncertainty_fn,
    )
    report = adapter.validate()
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from pwm_core.graph.adapters.tier_adapter import TierAdapter


class Tier3Adapter(TierAdapter):
    """Adapter for Tier-3 learned surrogate primitives.

    Extends TierAdapter with:
    - Uncertainty estimation (required for Tier-3)
    - Uncertainty calibration against Tier-2 reference
    - Agreement check with higher-fidelity reference
    - Calibration data persistence
    """

    _physics_tier = "tier3_learned"
    _physics_subrole = "surrogate"

    def __init__(
        self,
        forward_kernel: Callable,
        adjoint_kernel: Optional[Callable] = None,
        params: Optional[Dict[str, Any]] = None,
        input_shape: Tuple[int, ...] = (64, 64),
        output_shape: Tuple[int, ...] = (64, 64),
        name: str = "tier3_learned",
        is_linear: bool = False,
        uncertainty_kernel: Optional[Callable] = None,
    ):
        super().__init__(
            forward_kernel=forward_kernel,
            adjoint_kernel=adjoint_kernel,
            params=params,
            input_shape=input_shape,
            output_shape=output_shape,
            name=name,
            is_linear=is_linear,
        )
        self._uncertainty_kernel = uncertainty_kernel
        self._calibration_data: Optional[Dict[str, float]] = None

    def forward_with_uncertainty(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply forward model and return (prediction, uncertainty).

        Uncertainty is per-element standard deviation. If no uncertainty
        kernel is provided and calibration has been performed, uses
        calibrated residual statistics.
        """
        x = np.asarray(x, dtype=np.float64)
        y = self._forward_kernel(x, self._params)
        y = np.asarray(y, dtype=np.float64)

        if self._uncertainty_kernel is not None:
            sigma = np.asarray(
                self._uncertainty_kernel(x, self._params), dtype=np.float64
            )
        elif self._calibration_data is not None:
            sigma = np.full_like(y, self._calibration_data["mean_residual_std"])
        else:
            sigma = np.full_like(y, np.nan)

        return y, sigma

    def calibrate_uncertainty(
        self,
        x_samples: np.ndarray,
        y_reference: np.ndarray,
        n_samples: Optional[int] = None,
    ) -> Dict[str, float]:
        """Calibrate uncertainty against reference (Tier-2) outputs.

        Computes residual statistics to establish confidence intervals.

        Parameters
        ----------
        x_samples : ndarray
            Shape (N, *input_shape).
        y_reference : ndarray
            Shape (N, *output_shape). Reference outputs from Tier-2.
        n_samples : int or None
            Max samples to use. None = use all.

        Returns
        -------
        dict with calibration statistics.
        """
        n = len(x_samples)
        if n_samples is not None:
            n = min(n, n_samples)

        residuals = []
        for i in range(n):
            y_pred = self._forward_kernel(x_samples[i], self._params)
            residual = np.asarray(y_pred) - y_reference[i]
            residuals.append(residual)

        residuals = np.array(residuals)
        self._calibration_data = {
            "mean_residual_std": float(np.std(residuals)),
            "max_residual": float(np.max(np.abs(residuals))),
            "mean_residual_norm": float(
                np.mean(np.linalg.norm(residuals.reshape(n, -1), axis=1))
            ),
            "n_calibration_samples": n,
        }
        return self._calibration_data

    def check_agreement(
        self,
        reference_kernel: Callable,
        n_trials: int = 10,
        tol: float = 0.1,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Check agreement between this Tier-3 model and a reference.

        Parameters
        ----------
        reference_kernel : callable
            Higher-fidelity forward model: (x, params) -> y
        n_trials : int
            Number of random inputs.
        tol : float
            Maximum acceptable relative error.
        """
        rng = np.random.default_rng(seed)
        rel_errors = []

        for _ in range(n_trials):
            x = rng.standard_normal(self._input_shape)
            y_t3 = self._forward_kernel(x, self._params)
            y_ref = reference_kernel(x, self._params)

            denom = max(np.linalg.norm(y_ref), 1e-30)
            rel_err = float(np.linalg.norm(np.asarray(y_t3) - np.asarray(y_ref)) / denom)
            rel_errors.append(rel_err)

        max_err = max(rel_errors)
        return {
            "passed": max_err < tol,
            "max_relative_error": max_err,
            "mean_relative_error": float(np.mean(rel_errors)),
            "n_trials": n_trials,
            "tolerance": tol,
        }

    def check_uncertainty_calibration(
        self,
        x_samples: np.ndarray,
        y_reference: np.ndarray,
        target_coverage: float = 0.90,
        n_sigma: float = 1.645,
    ) -> Dict[str, Any]:
        """Check if uncertainty estimates are well-calibrated.

        At the 90% confidence level (1.645 sigma), ~90% of reference
        outputs should fall within the predicted uncertainty bounds.
        """
        n = len(x_samples)
        in_bounds = 0

        for i in range(n):
            y_pred, sigma = self.forward_with_uncertainty(x_samples[i])
            if np.any(np.isnan(sigma)):
                return {
                    "passed": False,
                    "error": "uncertainty model returns NaN",
                }

            diff = np.abs(np.asarray(y_reference[i]) - y_pred)
            bound = n_sigma * sigma
            if np.all(diff <= bound):
                in_bounds += 1

        coverage = in_bounds / max(n, 1)
        return {
            "passed": abs(coverage - target_coverage) < 0.15,
            "actual_coverage": coverage,
            "target_coverage": target_coverage,
            "n_samples": n,
            "note": "Coverage should be within 15% of target",
        }

    def serialize(self) -> Dict[str, Any]:
        result = super().serialize()
        if self._calibration_data:
            result["uncertainty_calibration"] = self._calibration_data
        return result

    def validate(
        self,
        adjoint_tol: float = 1e-4,
        linearity_tol: float = 1e-4,
        energy_max: float = 1000.0,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Run all validation checks including Tier-3 specific ones.

        Note: Tier-3 uses relaxed tolerances since learned models are
        approximate by nature.
        """
        report = super().validate(
            adjoint_tol=adjoint_tol,
            linearity_tol=linearity_tol,
            energy_max=energy_max,
            seed=seed,
        )

        # Tier-3 specific: check uncertainty is available
        report["uncertainty_available"] = (
            self._uncertainty_kernel is not None
            or self._calibration_data is not None
        )

        return report
