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


def poisson_nll(y: np.ndarray, yhat: np.ndarray) -> float:
    """Poisson negative log-likelihood: sum(yhat - y*log(yhat+eps))."""
    eps = 1e-10
    yhat_safe = np.maximum(yhat.reshape(-1), eps)
    y_flat = y.reshape(-1)
    return float(np.sum(yhat_safe - y_flat * np.log(yhat_safe)))


def gaussian_nll(y: np.ndarray, yhat: np.ndarray, sigma: float = 1.0) -> float:
    """Gaussian NLL: 0.5*sum((y-yhat)^2/sigma^2 + log(sigma^2))."""
    r = (y.reshape(-1) - yhat.reshape(-1))
    sigma2 = sigma * sigma
    return float(0.5 * np.sum(r * r / sigma2 + np.log(sigma2)))


def mixed_nll(y: np.ndarray, yhat: np.ndarray, alpha: float = 0.5, sigma: float = 1.0) -> float:
    """Convex combo: alpha*poisson_nll + (1-alpha)*gaussian_nll."""
    return alpha * poisson_nll(y, yhat) + (1.0 - alpha) * gaussian_nll(y, yhat, sigma)


def score_theta_likelihood(
    y: np.ndarray,
    yhat: np.ndarray,
    theta: Dict[str, Any],
    noise_model: str = "gaussian",
    theta_reg: float = 1e-4,
    **kwargs,
) -> ScoreResult:
    """Route to appropriate NLL scorer based on noise_model string."""
    if noise_model == "poisson":
        nll = poisson_nll(y, yhat)
    elif noise_model == "mixed":
        alpha = kwargs.get("alpha", 0.5)
        sigma = kwargs.get("sigma", 1.0)
        nll = mixed_nll(y, yhat, alpha=alpha, sigma=sigma)
    else:  # gaussian (default)
        sigma = kwargs.get("sigma", 1.0)
        nll = gaussian_nll(y, yhat, sigma=sigma)

    # Regularizer
    reg = 0.0
    for k, v in theta.items():
        try:
            reg += float(v) * float(v)
        except Exception:
            continue

    total = nll + theta_reg * reg
    return ScoreResult(
        total=float(total),
        terms={"nll": float(nll), "theta_l2": float(reg), "noise_model": noise_model},
        diagnostics={"nll_raw": float(nll)},
    )
