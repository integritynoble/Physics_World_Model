"""pwm_core.mismatch.scoring

Objective functions and residual-based scoring for operator fitting (theta search).

A good score should correlate with:
- residual whiteness / lack of structure
- stable recon (conditioning proxy)
- avoids trivial solutions (regularization on theta)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class ScoreResult:
    total: float
    terms: Dict[str, float]
    diagnostics: Dict[str, Any]


def residual_energy(y: np.ndarray, yhat: np.ndarray) -> float:
    r = (y - yhat).reshape(-1)
    return float(np.mean(r * r))


def l2_norm(x: np.ndarray) -> float:
    v = x.reshape(-1)
    return float(np.sqrt(np.mean(v * v)))


def score_theta(y: np.ndarray, yhat: np.ndarray, theta: Dict[str, Any], theta_reg: float = 1e-4) -> ScoreResult:
    e = residual_energy(y, yhat)
    # simple regularizer: sum of squares of theta values
    reg = 0.0
    for k, v in theta.items():
        try:
            reg += float(v) * float(v)
        except Exception:
            continue
    total = e + theta_reg * reg
    return ScoreResult(
        total=float(total),
        terms={"residual_mse": float(e), "theta_l2": float(reg), "theta_reg": float(theta_reg)},
        diagnostics={"residual_rms": float(np.sqrt(e))},
    )
