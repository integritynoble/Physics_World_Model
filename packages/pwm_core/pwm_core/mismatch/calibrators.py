"""pwm_core.mismatch.calibrators

Operator fitting / theta search engines:
- coarse sampling + local refine (bounded)
- scoring based on residual / proxy recon stability

This is a minimal engine intended to be called from core.runner in "y + A(theta)" mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from pwm_core.mismatch.parameterizations import ThetaSpace, cassi_theta_space, generic_gain_shift_space
from pwm_core.mismatch.scoring import ScoreResult, score_theta


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


def fit_theta(
    y: np.ndarray,
    forward_fn: Callable[[Dict[str, Any]], np.ndarray],
    space: ThetaSpace,
    cfg: FitConfig,
) -> FitResult:
    rng = np.random.default_rng(cfg.seed)
    scored: List[Tuple[Dict[str, Any], float, Dict[str, float]]] = []
    for _ in range(cfg.num_candidates):
        theta = random_theta(space, rng)
        yhat = forward_fn(theta)
        s = score_theta(y, yhat, theta, theta_reg=cfg.theta_reg)
        scored.append((theta, s.total, s.terms))
    scored.sort(key=lambda t: t[1])
    best_theta, best_score, best_terms = scored[0]
    return FitResult(
        best_theta=best_theta,
        best_score=float(best_score),
        scores=[(t, float(sc), terms) for (t, sc, terms) in scored],
        best_terms=best_terms,
    )


def get_theta_space(operator_id: str) -> ThetaSpace:
    if operator_id.startswith("cassi"):
        return cassi_theta_space()
    if operator_id.startswith("matrix") or operator_id.startswith("generic"):
        return generic_gain_shift_space()
    # default: treat as gain/bias
    return generic_gain_shift_space()
