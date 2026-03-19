"""pwm_core.targeting.scenarios
================================

4-Scenario evaluation protocol from ``docs/targeting_system.md``.

Scenario I   (Ideal):     measure with H_true, reconstruct with H_true
Scenario II  (Assumed):   measure with H_true, reconstruct with H_nom
Scenario III (Corrected): calibrate H_nom -> H_hat, reconstruct with H_hat
Scenario IV  (Oracle):    reconstruct with partial oracle operator
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from pwm_core.analysis.metrics import psnr
from pwm_core.targeting.budget import BudgetGuard

logger = logging.getLogger(__name__)


@dataclass
class ScenarioResult:
    """Result of running one scenario on one scene."""

    scenario_id: str
    psnr: float
    ssim: float
    x_hat: np.ndarray
    runtime_s: float
    budget_used_s: float
    info: Dict[str, Any] = field(default_factory=dict)


def _compute_ssim(x_true: np.ndarray, x_hat: np.ndarray) -> float:
    """Structural similarity (simplified, channel-averaged)."""
    x = x_true.astype(np.float64).ravel()
    y = x_hat.astype(np.float64).ravel()
    mu_x, mu_y = x.mean(), y.mean()
    var_x = float(np.mean((x - mu_x) ** 2))
    var_y = float(np.mean((y - mu_y) ** 2))
    cov_xy = float(np.mean((x - mu_x) * (y - mu_y)))
    c1 = (0.01 * 1.0) ** 2
    c2 = (0.03 * 1.0) ** 2
    num = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
    den = (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    return float(num / den) if den > 0 else 0.0


def _run_solver(
    solver_fn: Callable,
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
    budget_guard: BudgetGuard,
) -> Tuple[np.ndarray, Dict[str, Any], float]:
    """Run a solver under budget enforcement, return (x_hat, info, elapsed)."""
    t0 = time.perf_counter()
    with budget_guard:
        x_hat, info = solver_fn(y, physics, cfg)
    elapsed = time.perf_counter() - t0
    return x_hat, info, elapsed


def run_scenario_I(
    x_true: np.ndarray,
    y: np.ndarray,
    H_true: Any,
    solver_fn: Callable,
    cfg: Dict[str, Any],
    budget_guard: BudgetGuard,
) -> ScenarioResult:
    """Ideal: measure with H_true, reconstruct with H_true."""
    x_hat, info, elapsed = _run_solver(solver_fn, y, H_true, cfg, budget_guard)
    data_range = float(np.max(x_true) - np.min(x_true)) or 1.0
    return ScenarioResult(
        scenario_id="I",
        psnr=psnr(x_true, x_hat, data_range=data_range),
        ssim=_compute_ssim(x_true, x_hat),
        x_hat=x_hat,
        runtime_s=elapsed,
        budget_used_s=elapsed,
        info=info,
    )


def run_scenario_II(
    x_true: np.ndarray,
    y: np.ndarray,
    H_nom: Any,
    solver_fn: Callable,
    cfg: Dict[str, Any],
    budget_guard: BudgetGuard,
) -> ScenarioResult:
    """Assumed: measure with H_true, reconstruct with H_nom (mismatched)."""
    x_hat, info, elapsed = _run_solver(solver_fn, y, H_nom, cfg, budget_guard)
    data_range = float(np.max(x_true) - np.min(x_true)) or 1.0
    return ScenarioResult(
        scenario_id="II",
        psnr=psnr(x_true, x_hat, data_range=data_range),
        ssim=_compute_ssim(x_true, x_hat),
        x_hat=x_hat,
        runtime_s=elapsed,
        budget_used_s=elapsed,
        info=info,
    )


def run_scenario_III(
    x_true: np.ndarray,
    y: np.ndarray,
    H_nom: Any,
    calibrator_fn: Optional[Callable],
    solver_fn: Callable,
    cfg: Dict[str, Any],
    budget_guard: BudgetGuard,
    calibration_budget_s: float = 300.0,
) -> ScenarioResult:
    """Corrected: calibrate H_nom -> H_hat, then reconstruct with H_hat."""
    total_elapsed = 0.0
    info: Dict[str, Any] = {}

    if calibrator_fn is not None:
        t0 = time.perf_counter()
        H_hat, cal_info = calibrator_fn(y, H_nom, calibration_budget_s)
        cal_elapsed = time.perf_counter() - t0
        total_elapsed += cal_elapsed
        info["calibration"] = cal_info
        info["calibration_time_s"] = cal_elapsed
    else:
        H_hat = H_nom
        info["calibration"] = "none"

    x_hat, solver_info, solve_elapsed = _run_solver(
        solver_fn, y, H_hat, cfg, budget_guard
    )
    total_elapsed += solve_elapsed
    info.update(solver_info)

    data_range = float(np.max(x_true) - np.min(x_true)) or 1.0
    return ScenarioResult(
        scenario_id="III",
        psnr=psnr(x_true, x_hat, data_range=data_range),
        ssim=_compute_ssim(x_true, x_hat),
        x_hat=x_hat,
        runtime_s=total_elapsed,
        budget_used_s=total_elapsed,
        info=info,
    )


def run_scenario_IV(
    x_true: np.ndarray,
    y: np.ndarray,
    H_true: Any,
    H_nom: Any,
    solver_fn: Callable,
    cfg: Dict[str, Any],
    budget_guard: BudgetGuard,
) -> ScenarioResult:
    """Oracle Mask: reconstruct with partial oracle (H_true)."""
    x_hat, info, elapsed = _run_solver(solver_fn, y, H_true, cfg, budget_guard)
    data_range = float(np.max(x_true) - np.min(x_true)) or 1.0
    return ScenarioResult(
        scenario_id="IV",
        psnr=psnr(x_true, x_hat, data_range=data_range),
        ssim=_compute_ssim(x_true, x_hat),
        x_hat=x_hat,
        runtime_s=elapsed,
        budget_used_s=elapsed,
        info=info,
    )
