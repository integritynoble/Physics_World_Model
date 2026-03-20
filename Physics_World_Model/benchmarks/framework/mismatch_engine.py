"""MismatchEngine – parameterised mismatch injection and grid-search correction.

Supports B2 (forward + reconstruct under mismatch) and B4 (correction via
grid search) benchmark levels.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from benchmarks.framework.benchmark_config import MismatchParam
from benchmarks.framework.metrics import MetricSet


# ---------------------------------------------------------------------------
# MismatchScenario – a single perturbation setting
# ---------------------------------------------------------------------------

@dataclass
class MismatchScenario:
    """One set of perturbed operator parameters."""
    name: str
    params: Dict[str, float]  # param_name → perturbed value
    description: str = ""


# ---------------------------------------------------------------------------
# MismatchResult – outcome of running one scenario
# ---------------------------------------------------------------------------

@dataclass
class MismatchResult:
    """Metrics for one mismatch scenario."""
    scenario: MismatchScenario
    metrics: Dict[str, float]
    psnr_drop: float = 0.0  # Relative to nominal


# ---------------------------------------------------------------------------
# GridSearchResult – outcome of B4 correction
# ---------------------------------------------------------------------------

@dataclass
class GridSearchResult:
    """Result of grid-search parameter correction."""
    best_params: Dict[str, float]
    best_metric: float
    metric_name: str
    grid_shape: Tuple[int, ...]
    all_metrics: Optional[np.ndarray] = None  # Full grid of metric values
    rho: float = 0.0  # Recovery fraction: (corrected - mismatched) / (nominal - mismatched)


# ---------------------------------------------------------------------------
# MismatchEngine
# ---------------------------------------------------------------------------

class MismatchEngine:
    """Inject mismatch and perform grid-search correction.

    Usage::

        engine = MismatchEngine(mismatch_params, metric_set)

        # B2: single-param mismatch scenarios
        scenarios = engine.generate_scenarios(level="M1")

        # B4: grid search correction
        result = engine.grid_search(
            run_fn=lambda theta: (x_hat, metrics),
            nominal_theta=...,
        )
    """

    def __init__(
        self,
        mismatch_params: List[MismatchParam],
        metric_set: MetricSet = None,
    ):
        self.params = mismatch_params
        self.metric_set = metric_set or MetricSet()

    # ------------------------------------------------------------------
    # Scenario generation
    # ------------------------------------------------------------------

    def generate_scenarios(self, level: str = "M1") -> List[MismatchScenario]:
        """Generate mismatch scenarios for the given maturity level.

        Args:
            level: ``"M0"`` (nominal only), ``"M1"`` (single-param),
                   ``"M2"`` (compound), ``"M4"`` (adversarial extremes).

        Returns:
            List of ``MismatchScenario`` objects.
        """
        if level == "M0":
            # Nominal only
            return [MismatchScenario(
                name="nominal",
                params={p.name: p.nominal for p in self.params},
                description="Nominal parameters",
            )]

        if level == "M1":
            # Single-parameter perturbations at boundary values
            scenarios = [MismatchScenario(
                name="nominal",
                params={p.name: p.nominal for p in self.params},
            )]
            for param in self.params:
                base = {p.name: p.nominal for p in self.params}
                # Low end
                base_lo = dict(base)
                base_lo[param.name] = param.range[0]
                scenarios.append(MismatchScenario(
                    name=f"{param.name}_lo",
                    params=base_lo,
                    description=f"{param.name} at lower bound {param.range[0]} {param.unit}",
                ))
                # High end
                base_hi = dict(base)
                base_hi[param.name] = param.range[1]
                scenarios.append(MismatchScenario(
                    name=f"{param.name}_hi",
                    params=base_hi,
                    description=f"{param.name} at upper bound {param.range[1]} {param.unit}",
                ))
            return scenarios

        if level == "M2":
            # Compound: random samples from the parameter hypercube
            scenarios = [MismatchScenario(
                name="nominal",
                params={p.name: p.nominal for p in self.params},
            )]
            rng = np.random.RandomState(42)
            n_compound = min(10, 2 ** len(self.params))
            for i in range(n_compound):
                params = {}
                parts = []
                for p in self.params:
                    val = rng.uniform(p.range[0], p.range[1])
                    params[p.name] = float(val)
                    parts.append(f"{p.name}={val:.3f}")
                scenarios.append(MismatchScenario(
                    name=f"compound_{i:02d}",
                    params=params,
                    description=", ".join(parts),
                ))
            return scenarios

        if level in ("M3", "M4"):
            # Adversarial: worst-case corners of the hypercube
            scenarios = self.generate_scenarios("M2")
            # Add extreme corners
            for corner_idx in range(min(8, 2 ** len(self.params))):
                params = {}
                for j, p in enumerate(self.params):
                    bit = (corner_idx >> j) & 1
                    params[p.name] = p.range[1] if bit else p.range[0]
                scenarios.append(MismatchScenario(
                    name=f"corner_{corner_idx:02d}",
                    params=params,
                    description="Hypercube corner",
                ))
            return scenarios

        return self.generate_scenarios("M1")  # fallback

    # ------------------------------------------------------------------
    # Mismatch evaluation
    # ------------------------------------------------------------------

    def evaluate_scenarios(
        self,
        scenarios: List[MismatchScenario],
        run_fn: Callable[[Dict[str, float]], Tuple[np.ndarray, np.ndarray]],
        x_true: np.ndarray,
    ) -> List[MismatchResult]:
        """Run reconstruction under each scenario and measure degradation.

        Args:
            scenarios: Mismatch scenarios to evaluate.
            run_fn: ``fn(theta_dict) -> (x_hat, y)`` that rebuilds operator
                     with perturbed params and reconstructs.
            x_true: Ground truth for metric computation.

        Returns:
            List of ``MismatchResult``.
        """
        results: List[MismatchResult] = []
        nominal_psnr = None

        for sc in scenarios:
            x_hat, _ = run_fn(sc.params)
            metrics = self.metric_set.evaluate(x_true, x_hat)
            psnr = metrics.get("psnr", 0.0)

            if sc.name == "nominal":
                nominal_psnr = psnr

            drop = (nominal_psnr or psnr) - psnr
            results.append(MismatchResult(
                scenario=sc,
                metrics=metrics,
                psnr_drop=drop,
            ))
        return results

    # ------------------------------------------------------------------
    # Grid search (B4 correction)
    # ------------------------------------------------------------------

    def grid_search(
        self,
        run_fn: Callable[[Dict[str, float]], Tuple[np.ndarray, np.ndarray]],
        x_true: np.ndarray,
        n_grid: int = 5,
        nominal_metric: Optional[float] = None,
        mismatched_metric: Optional[float] = None,
    ) -> GridSearchResult:
        """Grid search over mismatch parameter space to find best correction.

        Args:
            run_fn: ``fn(theta_dict) -> (x_hat, y)``.
            x_true: Ground truth.
            n_grid: Number of grid points per parameter.
            nominal_metric: Metric value at nominal (for rho computation).
            mismatched_metric: Metric value under mismatch (for rho).

        Returns:
            ``GridSearchResult`` with best parameters and rho.
        """
        metric_name = self.metric_set.primary

        # Build grid axes
        axes = []
        for p in self.params:
            if p.grid_values:
                axes.append(np.array(p.grid_values))
            else:
                axes.append(np.linspace(p.range[0], p.range[1], n_grid))

        grid_shape = tuple(len(a) for a in axes)
        all_metrics = np.zeros(grid_shape, dtype=np.float64)
        best_val = -np.inf
        best_params: Dict[str, float] = {}

        for idx in itertools.product(*(range(len(a)) for a in axes)):
            params = {}
            for j, p in enumerate(self.params):
                params[p.name] = float(axes[j][idx[j]])
            x_hat, _ = run_fn(params)
            metrics = self.metric_set.evaluate(x_true, x_hat)
            val = metrics.get(metric_name, 0.0)
            all_metrics[idx] = val

            if val > best_val:
                best_val = val
                best_params = dict(params)

        # Compute rho (recovery fraction)
        rho = 0.0
        if nominal_metric is not None and mismatched_metric is not None:
            denom = nominal_metric - mismatched_metric
            if abs(denom) > 1e-6:
                rho = (best_val - mismatched_metric) / denom

        return GridSearchResult(
            best_params=best_params,
            best_metric=float(best_val),
            metric_name=metric_name,
            grid_shape=grid_shape,
            all_metrics=all_metrics,
            rho=float(rho),
        )
