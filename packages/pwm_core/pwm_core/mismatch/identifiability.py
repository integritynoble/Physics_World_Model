"""pwm_core.mismatch.identifiability

Sensitivity-based identifiability analysis for calibration parameters.

Computes finite-difference sensitivity for each parameter in theta space,
identifies insensitive (unidentifiable) parameters, and provides a filtered
search space for the calibration loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from pwm_core.mismatch.parameterizations import ThetaSpace
from pwm_core.mismatch.scoring import residual_energy


@dataclass
class IdentifiabilityReport:
    """Report from sensitivity probe analysis."""
    param_sensitivities: Dict[str, float]  # param_name -> |d(score)/d(theta)|
    frozen_params: List[str]               # params below threshold
    identifiable_params: List[str]         # params above threshold
    threshold: float = 1e-4


def sensitivity_probe(
    y: np.ndarray,
    forward_fn: Callable[[Dict[str, Any]], np.ndarray],
    theta: Dict[str, Any],
    space: ThetaSpace,
    eps: float = 1e-3,
    score_fn: Optional[Callable] = None,
    threshold: float = 1e-4,
) -> IdentifiabilityReport:
    """Finite-difference sensitivity: perturb each param by eps*range_width.

    Parameters
    ----------
    y : np.ndarray
        Measurement data.
    forward_fn : callable
        Maps theta dict -> predicted measurement.
    theta : dict
        Current parameter values.
    space : ThetaSpace
        Parameter search space with bounds.
    eps : float
        Perturbation fraction of range width.
    score_fn : callable, optional
        Score function (y, yhat) -> float. Defaults to residual_energy.
    threshold : float
        Sensitivity threshold below which params are frozen.

    Returns
    -------
    IdentifiabilityReport
    """
    if score_fn is None:
        score_fn = residual_energy

    # Base score
    yhat_base = forward_fn(theta)
    score_base = score_fn(y, yhat_base)

    sensitivities: Dict[str, float] = {}
    frozen: List[str] = []
    identifiable: List[str] = []

    for param_name, spec in space.params.items():
        if param_name not in theta:
            continue

        ptype = spec.get("type", "float")
        if ptype not in ("float", "int"):
            # Skip enum params
            sensitivities[param_name] = 0.0
            frozen.append(param_name)
            continue

        lo, hi = float(spec["low"]), float(spec["high"])
        range_width = hi - lo
        if range_width < 1e-12:
            sensitivities[param_name] = 0.0
            frozen.append(param_name)
            continue

        delta = eps * range_width
        val = float(theta[param_name])

        # Forward perturbation (clamped)
        theta_plus = dict(theta)
        theta_plus[param_name] = min(val + delta, hi)

        # Backward perturbation (clamped)
        theta_minus = dict(theta)
        theta_minus[param_name] = max(val - delta, lo)

        actual_delta = float(theta_plus[param_name]) - float(theta_minus[param_name])
        if actual_delta < 1e-12:
            sensitivities[param_name] = 0.0
            frozen.append(param_name)
            continue

        yhat_plus = forward_fn(theta_plus)
        yhat_minus = forward_fn(theta_minus)

        score_plus = score_fn(y, yhat_plus)
        score_minus = score_fn(y, yhat_minus)

        sensitivity = abs(score_plus - score_minus) / actual_delta
        sensitivities[param_name] = sensitivity

        if sensitivity < threshold:
            frozen.append(param_name)
        else:
            identifiable.append(param_name)

    return IdentifiabilityReport(
        param_sensitivities=sensitivities,
        frozen_params=frozen,
        identifiable_params=identifiable,
        threshold=threshold,
    )


def filter_space(space: ThetaSpace, report: IdentifiabilityReport) -> ThetaSpace:
    """Return a ThetaSpace with only identifiable params."""
    filtered_params = {
        k: v for k, v in space.params.items()
        if k in report.identifiable_params
    }
    return ThetaSpace(
        name=f"{space.name}_filtered",
        params=filtered_params,
    )
