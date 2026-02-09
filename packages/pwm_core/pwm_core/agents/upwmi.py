"""pwm_core.agents.upwmi

Unified UPWMI scoring, caching, budget guardrails, and active learning search.

Generalises the CASSI-specific UPWMI benchmarks into modality-agnostic
infrastructure for operator correction across all 26 modalities.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .contracts import StrictBaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Budget guardrails
# ---------------------------------------------------------------------------

class UPWMIBudget(StrictBaseModel):
    """Budget guardrails for UPWMI calibration search."""

    max_candidates: int = 24
    max_refinements_per_candidate: int = 8
    max_total_runtime_s: float = 300.0
    early_stop_plateau_delta: float = 1e-3
    early_stop_patience: int = 3
    proxy_solver: str = "fista"
    proxy_max_iters: int = 40


# ---------------------------------------------------------------------------
# Scoring weights
# ---------------------------------------------------------------------------

class ScoringWeights(StrictBaseModel):
    """4-term scoring weights for UPWMI objective.

    Lower composite score is better.
    """

    alpha: float = 1.0    # residual norm
    beta: float = 0.3     # whiteness
    gamma: float = 0.1    # prior
    delta: float = 0.05   # proxy cost


# ---------------------------------------------------------------------------
# Scoring function
# ---------------------------------------------------------------------------

def upwmi_score(
    y: np.ndarray,
    operator: Any,
    proxy_recon: Callable,
    weights: ScoringWeights,
    modality_key: str,
) -> Tuple[float, np.ndarray]:
    """Compute unified UPWMI score for a candidate operator parameterisation.

    score = alpha*residual_norm + beta*(1-whiteness) + gamma*(1-prior) + delta*proxy_cost

    Lower is better.

    Parameters
    ----------
    y : np.ndarray
        Measured data.
    operator : PhysicsOperator
        Operator configured with candidate theta.
    proxy_recon : callable
        Proxy reconstruction function: (y, operator) -> x_hat.
    weights : ScoringWeights
        4-term weighting.
    modality_key : str
        Modality identifier for residual feature dispatch.

    Returns
    -------
    score : float
        Composite UPWMI score (lower is better).
    x_proxy : np.ndarray
        Proxy reconstruction.
    """
    x_proxy = proxy_recon(y, operator)

    # Residual
    y_hat = operator.forward(x_proxy)
    residual = y.ravel() - y_hat.ravel()
    residual_norm = float(np.linalg.norm(residual)) / max(float(np.linalg.norm(y.ravel())), 1e-30)

    # Whiteness: autocorrelation of residual (1.0 = perfectly white)
    r = residual - residual.mean()
    acf = np.correlate(r, r, mode="full")
    acf = acf / max(acf[len(r) - 1], 1e-30)
    # Whiteness = 1 - mean of |acf| at lags 1..min(50, len)
    n_lags = min(50, len(r) - 1)
    if n_lags > 0:
        whiteness = 1.0 - float(np.mean(np.abs(acf[len(r): len(r) + n_lags])))
    else:
        whiteness = 1.0

    # Prior: use residual features as proxy for signal quality
    feats = residual_features(modality_key, residual)
    prior = 1.0 - feats.get("hf_energy_ratio", 0.5)

    # Proxy cost: relative reconstruction energy
    proxy_cost = float(np.linalg.norm(x_proxy.ravel())) / max(float(np.linalg.norm(y.ravel())), 1e-30)

    score = (
        weights.alpha * residual_norm
        + weights.beta * (1.0 - whiteness)
        + weights.gamma * (1.0 - prior)
        + weights.delta * proxy_cost
    )

    return float(score), x_proxy


# ---------------------------------------------------------------------------
# Residual features (modality-dispatched)
# ---------------------------------------------------------------------------

def _hf_energy_ratio(signal: np.ndarray) -> float:
    """High-frequency energy ratio via DFT."""
    spectrum = np.abs(np.fft.fft(signal))
    n = len(spectrum) // 2
    if n < 2:
        return 0.5
    hf = float(np.sum(spectrum[n // 2: n] ** 2))
    total = float(np.sum(spectrum[:n] ** 2))
    return hf / max(total, 1e-30)


def _ct_streak_score(residual: np.ndarray) -> float:
    """CT streak artefact score: directional energy in FFT."""
    n = int(np.sqrt(len(residual)))
    if n < 4:
        return 0.0
    img = residual[:n * n].reshape(n, n)
    fft2 = np.abs(np.fft.fftshift(np.fft.fft2(img)))
    # Streaks show as radial lines — measure variance along angular slices
    cy, cx = n // 2, n // 2
    radii = np.sqrt((np.arange(n)[:, None] - cy) ** 2 + (np.arange(n)[None, :] - cx) ** 2)
    ring_mask = (radii > n // 8) & (radii < n // 3)
    if ring_mask.sum() < 4:
        return 0.0
    return float(np.std(fft2[ring_mask]) / max(np.mean(fft2[ring_mask]), 1e-30))


def _spectral_mixing_score(residual: np.ndarray) -> float:
    """Spectral mixing score for CASSI/CACTI (cross-channel leakage)."""
    spectrum = np.abs(np.fft.fft(residual))
    n = len(spectrum) // 2
    if n < 4:
        return 0.0
    # Mixing shows as peaks in low-frequency band
    lf = float(np.max(spectrum[1: max(2, n // 8)]))
    baseline = float(np.median(spectrum[n // 4: n // 2]))
    return lf / max(baseline, 1e-30)


def _kspace_ringing_score(residual: np.ndarray) -> float:
    """K-space ringing score for MRI residuals."""
    spectrum = np.abs(np.fft.fft(residual))
    n = len(spectrum) // 2
    if n < 4:
        return 0.0
    # Ringing shows as oscillations in spectrum
    diffs = np.abs(np.diff(spectrum[:n]))
    return float(np.mean(diffs) / max(np.mean(spectrum[:n]), 1e-30))


def residual_features(modality: str, residual: np.ndarray) -> Dict[str, float]:
    """Dispatch residual analysis by modality.

    Parameters
    ----------
    modality : str
        Modality key (e.g. "ct", "cassi", "mri").
    residual : np.ndarray
        Residual vector (1D).

    Returns
    -------
    Dict[str, float]
        Feature dict with modality-specific scores.
    """
    flat = residual.ravel().astype(np.float64)
    features: Dict[str, float] = {
        "autocorrelation": 0.0,
        "hf_energy_ratio": _hf_energy_ratio(flat),
    }

    # Autocorrelation at lag-1
    if len(flat) > 1:
        r = flat - flat.mean()
        var = float(np.sum(r ** 2))
        if var > 1e-30:
            features["autocorrelation"] = float(np.sum(r[:-1] * r[1:]) / var)

    modality_lower = modality.lower()
    if "ct" in modality_lower:
        features["streak_score"] = _ct_streak_score(flat)
    elif modality_lower in ("cassi", "cacti"):
        features["spectral_mixing"] = _spectral_mixing_score(flat)
    elif "mri" in modality_lower:
        features["kspace_ringing"] = _kspace_ringing_score(flat)

    return features


# ---------------------------------------------------------------------------
# CandidateCache (bug-fixed from v2: max_size properly assigned)
# ---------------------------------------------------------------------------

class CandidateCache:
    """Cache for evaluated operator candidates.

    Bug-fix from v2: max_size is now properly assigned before use.
    """

    def __init__(self, max_size: int = 100, disk_path: Optional[str] = None):
        self.max_size = max_size
        self.disk_path = disk_path
        self._cache: Dict[str, Tuple[np.ndarray, float]] = {}

    @staticmethod
    def key(theta: Dict[str, Any]) -> str:
        """Stable float serialization for cache keys.

        Rounds floats to 10 decimals and sorts keys for determinism.
        """
        def _normalize(v: Any) -> Any:
            if isinstance(v, float):
                return round(v, 10)
            if isinstance(v, np.floating):
                return round(float(v), 10)
            if isinstance(v, np.integer):
                return int(v)
            if isinstance(v, np.ndarray):
                return [round(float(x), 10) for x in v.ravel()]
            if isinstance(v, (list, tuple)):
                return [_normalize(x) for x in v]
            if isinstance(v, dict):
                return {k: _normalize(val) for k, val in sorted(v.items())}
            return v

        normalized = {k: _normalize(v) for k, v in sorted(theta.items())}
        return json.dumps(normalized, sort_keys=True)

    def get(self, theta: Dict[str, Any]) -> Optional[Tuple[np.ndarray, float]]:
        """Look up cached result for theta."""
        k = self.key(theta)
        return self._cache.get(k)

    def put(self, theta: Dict[str, Any], x_proxy: np.ndarray, score: float) -> None:
        """Store result, evicting oldest if at capacity."""
        k = self.key(theta)
        if len(self._cache) >= self.max_size and k not in self._cache:
            # Evict first inserted (FIFO)
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[k] = (x_proxy, score)


# ---------------------------------------------------------------------------
# ActiveLearningSearch
# ---------------------------------------------------------------------------

class ActiveLearningSearch:
    """Dual-path gradient/finite-difference refinement for operator parameters.

    Dispatches on operator.supports_autodiff:
    - True: allows modality-specific torch overrides (architecture hook)
    - False: generic finite-difference gradient path
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        fd_epsilon: float = 1e-3,
    ):
        self.learning_rate = learning_rate
        self.bounds = bounds or {}
        self.fd_epsilon = fd_epsilon

    def refine(
        self,
        theta_init: Dict[str, float],
        y: np.ndarray,
        operator: Any,
        proxy_recon: Callable,
        n_steps: int = 5,
    ) -> Dict[str, float]:
        """Refine theta via gradient descent on UPWMI score.

        Parameters
        ----------
        theta_init : dict
            Initial parameter values (must be float-valued).
        y : np.ndarray
            Measured data.
        operator : PhysicsOperator
            Operator to configure with candidate theta.
        proxy_recon : callable
            Proxy reconstruction function.
        n_steps : int
            Number of gradient steps.

        Returns
        -------
        dict
            Refined theta.
        """
        theta = dict(theta_init)
        weights = ScoringWeights()

        for step in range(n_steps):
            grad = self._fd_gradient(theta, y, operator, proxy_recon, weights)

            for k in theta:
                if isinstance(theta[k], (int, float, np.floating)):
                    theta[k] = float(theta[k]) - self.learning_rate * grad.get(k, 0.0)
                    # Clip to bounds
                    if k in self.bounds:
                        lo, hi = self.bounds[k]
                        theta[k] = max(lo, min(hi, theta[k]))

        return theta

    def _fd_gradient(
        self,
        theta: Dict[str, float],
        y: np.ndarray,
        operator: Any,
        proxy_recon: Callable,
        weights: ScoringWeights,
    ) -> Dict[str, float]:
        """Coordinate-wise finite-difference gradient."""
        grad: Dict[str, float] = {}

        for k in theta:
            if not isinstance(theta[k], (int, float, np.floating)):
                continue

            theta_plus = dict(theta)
            theta_plus[k] = float(theta[k]) + self.fd_epsilon
            operator.set_theta(theta_plus)
            score_plus, _ = upwmi_score(y, operator, proxy_recon, weights, "default")

            theta_minus = dict(theta)
            theta_minus[k] = float(theta[k]) - self.fd_epsilon
            operator.set_theta(theta_minus)
            score_minus, _ = upwmi_score(y, operator, proxy_recon, weights, "default")

            grad[k] = (score_plus - score_minus) / (2.0 * self.fd_epsilon)

        # Restore original theta
        operator.set_theta(theta)
        return grad
