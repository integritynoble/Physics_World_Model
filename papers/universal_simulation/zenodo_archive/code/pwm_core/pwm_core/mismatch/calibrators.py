"""pwm_core.mismatch.calibrators

Operator fitting / theta search engines:
- coarse sampling + local refine (bounded) - legacy fit_theta
- beam search calibration with likelihood scoring - calibrate()

This is a minimal engine intended to be called from core.runner in "y + A(theta)" mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from pwm_core.mismatch.parameterizations import ThetaSpace, cassi_theta_space, generic_gain_shift_space
from pwm_core.mismatch.scoring import ScoreResult, score_theta, score_theta_likelihood
from pwm_core.mismatch.identifiability import (
    IdentifiabilityReport,
    sensitivity_probe,
    filter_space,
)


@dataclass
class FitConfig:
    num_candidates: int = 8
    seed: int = 0
    theta_reg: float = 1e-4


@dataclass
class FitResult:
    best_theta: Dict[str, Any]
    best_score: float
    scores: List[Tuple[Dict[str, Any], float]]
    best_terms: Dict[str, float]


@dataclass
class CalibConfig:
    num_candidates: int = 16       # beam width
    num_refine_steps: int = 5      # local refinement rounds
    max_evals: int = 200           # budget
    seed: int = 0
    noise_model: str = "gaussian"  # for likelihood scoring
    shrink_factor: float = 0.5     # range shrink per refine step
    convergence_tol: float = 1e-4  # stop if score improves < tol
    patience: int = 3              # stop after N steps without improvement
    theta_reg: float = 1e-4
    run_identifiability: bool = True


@dataclass
class CalibLog:
    step: int
    best_score: float
    num_evals: int
    best_theta: Dict[str, Any]


@dataclass
class CalibResult:
    best_theta: Dict[str, Any]
    best_score: float
    num_evals: int
    logs: List[CalibLog]
    status: str  # "converged", "budget_exhausted", "flat_landscape", "diverged"
    identifiability: Optional[IdentifiabilityReport] = None


def random_theta(space: ThetaSpace, rng: np.random.Generator) -> Dict[str, Any]:
    theta = {}
    for k, spec in space.params.items():
        t = spec.get("type", "float")
        lo, hi = float(spec["low"]), float(spec["high"])
        if t == "int":
            theta[k] = int(rng.integers(int(lo), int(hi) + 1))
        else:
            theta[k] = float(rng.uniform(lo, hi))
    return theta


def _shrink_space(space: ThetaSpace, center: Dict[str, Any], factor: float) -> ThetaSpace:
    """Shrink parameter ranges around center by factor."""
    new_params = {}
    for k, spec in space.params.items():
        new_spec = dict(spec)
        lo, hi = float(spec["low"]), float(spec["high"])
        c = float(center.get(k, (lo + hi) / 2))
        half_range = (hi - lo) * factor / 2
        new_spec["low"] = max(lo, c - half_range)
        new_spec["high"] = min(hi, c + half_range)
        new_params[k] = new_spec
    return ThetaSpace(name=space.name, params=new_params)


def calibrate(
    y: np.ndarray,
    forward_fn: Callable[[Dict[str, Any]], np.ndarray],
    space: ThetaSpace,
    cfg: CalibConfig,
) -> CalibResult:
    """Beam search calibration loop.

    1. Run identifiability probe -> freeze insensitive params
    2. Random sample num_candidates initial thetas
    3. Score all candidates (using likelihood-aware scoring)
    4. Keep top-K beam
    5. Local refine: shrink ranges around best, resample, score
    6. Check stop criteria each step
    7. Return best theta + logs + status
    """
    rng = np.random.default_rng(cfg.seed)
    logs: List[CalibLog] = []
    total_evals = 0

    # 1. Identifiability probe
    ident_report: Optional[IdentifiabilityReport] = None
    active_space = space

    if cfg.run_identifiability and len(space.params) > 0:
        # Use midpoint as probe theta
        mid_theta = {}
        for k, spec in space.params.items():
            lo, hi = float(spec["low"]), float(spec["high"])
            mid_theta[k] = (lo + hi) / 2
            if spec.get("type") == "int":
                mid_theta[k] = int(round(mid_theta[k]))

        try:
            ident_report = sensitivity_probe(
                y, forward_fn, mid_theta, space, eps=1e-3
            )
            if ident_report.identifiable_params:
                active_space = filter_space(space, ident_report)
            total_evals += 2 * len(space.params)  # forward + backward per param
        except Exception:
            pass  # If probe fails, use full space

    # 2. Initial random candidates
    beam_k = min(cfg.num_candidates, max(4, cfg.num_candidates // 2))
    candidates: List[Tuple[Dict[str, Any], float]] = []

    for _ in range(cfg.num_candidates):
        if total_evals >= cfg.max_evals:
            break
        theta = random_theta(active_space, rng)
        # Merge frozen params from full space midpoints
        full_theta = {}
        for k, spec in space.params.items():
            if k in theta:
                full_theta[k] = theta[k]
            else:
                lo, hi = float(spec["low"]), float(spec["high"])
                full_theta[k] = (lo + hi) / 2
                if spec.get("type") == "int":
                    full_theta[k] = int(round(full_theta[k]))

        yhat = forward_fn(full_theta)
        s = score_theta_likelihood(
            y, yhat, full_theta,
            noise_model=cfg.noise_model,
            theta_reg=cfg.theta_reg,
        )
        candidates.append((full_theta, s.total))
        total_evals += 1

    if not candidates:
        return CalibResult(
            best_theta={},
            best_score=float("inf"),
            num_evals=total_evals,
            logs=logs,
            status="budget_exhausted",
            identifiability=ident_report,
        )

    candidates.sort(key=lambda t: t[1])
    best_theta, best_score = candidates[0]

    logs.append(CalibLog(
        step=0, best_score=best_score,
        num_evals=total_evals, best_theta=dict(best_theta),
    ))

    # 3-6. Refine steps with stop criteria
    no_improvement_count = 0
    diverge_count = 0
    prev_score = best_score

    for step in range(1, cfg.num_refine_steps + 1):
        if total_evals >= cfg.max_evals:
            return CalibResult(
                best_theta=best_theta,
                best_score=best_score,
                num_evals=total_evals,
                logs=logs,
                status="budget_exhausted",
                identifiability=ident_report,
            )

        # Keep top-K beam
        beam = candidates[:beam_k]

        # Shrink space around best
        shrunk = _shrink_space(active_space, best_theta, cfg.shrink_factor ** step)

        # Resample around beam centers
        new_candidates: List[Tuple[Dict[str, Any], float]] = list(beam)

        per_beam = max(1, (cfg.num_candidates - beam_k) // beam_k)
        for center_theta, _ in beam:
            local_space = _shrink_space(active_space, center_theta, cfg.shrink_factor ** step)
            for _ in range(per_beam):
                if total_evals >= cfg.max_evals:
                    break
                theta = random_theta(local_space, rng)
                full_theta = {}
                for k, spec in space.params.items():
                    if k in theta:
                        full_theta[k] = theta[k]
                    elif k in center_theta:
                        full_theta[k] = center_theta[k]
                    else:
                        lo, hi = float(spec["low"]), float(spec["high"])
                        full_theta[k] = (lo + hi) / 2
                yhat = forward_fn(full_theta)
                s = score_theta_likelihood(
                    y, yhat, full_theta,
                    noise_model=cfg.noise_model,
                    theta_reg=cfg.theta_reg,
                )
                new_candidates.append((full_theta, s.total))
                total_evals += 1

        # Sort and update best
        new_candidates.sort(key=lambda t: t[1])
        candidates = new_candidates

        step_best_theta, step_best_score = candidates[0]

        # Check flat landscape
        if len(candidates) > 1:
            scores_arr = np.array([s for _, s in candidates])
            if np.std(scores_arr) < 1e-6:
                return CalibResult(
                    best_theta=step_best_theta,
                    best_score=step_best_score,
                    num_evals=total_evals,
                    logs=logs,
                    status="flat_landscape",
                    identifiability=ident_report,
                )

        # Update best
        if step_best_score < best_score:
            best_theta = step_best_theta
            best_score = step_best_score

        logs.append(CalibLog(
            step=step, best_score=best_score,
            num_evals=total_evals, best_theta=dict(best_theta),
        ))

        # Check convergence
        improvement = prev_score - best_score
        if improvement < cfg.convergence_tol:
            no_improvement_count += 1
        else:
            no_improvement_count = 0

        # Check divergence
        if best_score > prev_score:
            diverge_count += 1
        else:
            diverge_count = 0

        if no_improvement_count >= cfg.patience:
            return CalibResult(
                best_theta=best_theta,
                best_score=best_score,
                num_evals=total_evals,
                logs=logs,
                status="converged",
                identifiability=ident_report,
            )

        if diverge_count >= cfg.patience:
            return CalibResult(
                best_theta=best_theta,
                best_score=best_score,
                num_evals=total_evals,
                logs=logs,
                status="diverged",
                identifiability=ident_report,
            )

        prev_score = best_score

    return CalibResult(
        best_theta=best_theta,
        best_score=best_score,
        num_evals=total_evals,
        logs=logs,
        status="converged",
        identifiability=ident_report,
    )


def fit_theta(
    y: np.ndarray,
    forward_fn: Callable[[Dict[str, Any]], np.ndarray],
    space: ThetaSpace,
    cfg: FitConfig,
) -> FitResult:
    """Legacy random-search fit. Delegates to calibrate() internally."""
    calib_cfg = CalibConfig(
        num_candidates=cfg.num_candidates,
        num_refine_steps=0,
        max_evals=cfg.num_candidates,
        seed=cfg.seed,
        theta_reg=cfg.theta_reg,
        run_identifiability=False,
    )
    result = calibrate(y, forward_fn, space, calib_cfg)
    return FitResult(
        best_theta=result.best_theta,
        best_score=result.best_score,
        scores=[(log.best_theta, log.best_score) for log in result.logs],
        best_terms={},
    )


def get_theta_space(operator_id: str) -> ThetaSpace:
    if operator_id.startswith("cassi"):
        return cassi_theta_space()
    if operator_id.startswith("matrix") or operator_id.startswith("generic"):
        return generic_gain_shift_space()
    # default: treat as gain/bias
    return generic_gain_shift_space()
