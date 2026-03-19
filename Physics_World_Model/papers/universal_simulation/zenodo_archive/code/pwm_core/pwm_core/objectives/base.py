"""pwm_core.objectives.base
============================

NegLogLikelihood ABC + 6 concrete implementations + OBJECTIVE_REGISTRY.

Each NLL class implements ``__call__(y, yhat) -> float`` and
``gradient(y, yhat) -> ndarray`` for use in reconstruction and calibration.

Implementations
---------------
PoissonNLL              Shot-noise-limited detection
GaussianNLL             Additive white Gaussian noise
ComplexGaussianNLL      Complex-valued Gaussian (MRI k-space)
MixedPoissonGaussianNLL Combined shot + read noise
HuberNLL                Robust Huber loss (outlier-resistant)
TukeyBiweightNLL        Robust Tukey biweight loss (heavy outlier rejection)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator


# ---------------------------------------------------------------------------
# StrictBaseModel (local copy)
# ---------------------------------------------------------------------------


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        ser_json_inf_nan="constants",
    )

    @model_validator(mode="after")
    def _reject_nan_inf(self) -> "StrictBaseModel":
        for field_name in self.__class__.model_fields:
            val = getattr(self, field_name)
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                raise ValueError(
                    f"Field '{field_name}' contains {val!r}, which is not allowed."
                )
        return self


# ---------------------------------------------------------------------------
# ObjectiveSpec
# ---------------------------------------------------------------------------


class ObjectiveSpec(StrictBaseModel):
    """Specification for constructing a NLL objective.

    Attributes
    ----------
    kind : str
        Key in OBJECTIVE_REGISTRY (e.g. ``"poisson"``, ``"gaussian"``).
    params : dict
        Parameters forwarded to the NLL constructor (e.g. ``sigma``, ``delta``).
    """

    kind: str = "gaussian"
    params: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# NegLogLikelihood ABC
# ---------------------------------------------------------------------------


class NegLogLikelihood(ABC):
    """Abstract base class for negative log-likelihood objectives."""

    def __init__(self, **params: Any) -> None:
        self._params = params

    @abstractmethod
    def __call__(self, y: np.ndarray, yhat: np.ndarray) -> float:
        """Compute NLL value."""
        ...

    @abstractmethod
    def gradient(self, y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
        """Gradient of NLL w.r.t. yhat."""
        ...


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------

_EPS = 1e-10


class PoissonNLL(NegLogLikelihood):
    """Poisson NLL: sum(yhat - y*log(yhat + eps))."""

    def __call__(self, y: np.ndarray, yhat: np.ndarray) -> float:
        yhat_safe = np.maximum(yhat.ravel().astype(np.float64), _EPS)
        y_flat = y.ravel().astype(np.float64)
        return float(np.sum(yhat_safe - y_flat * np.log(yhat_safe)))

    def gradient(self, y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
        yhat_safe = np.maximum(yhat.astype(np.float64), _EPS)
        return (1.0 - y.astype(np.float64) / yhat_safe)


class GaussianNLL(NegLogLikelihood):
    """Gaussian NLL: 0.5 * sum((y - yhat)^2 / sigma^2 + log(sigma^2)).

    Parameters
    ----------
    sigma : float
        Noise standard deviation (default 1.0).
    """

    def __init__(self, sigma: float = 1.0, **params: Any) -> None:
        super().__init__(sigma=sigma, **params)
        self._sigma = sigma

    def __call__(self, y: np.ndarray, yhat: np.ndarray) -> float:
        r = (y.ravel() - yhat.ravel()).astype(np.float64)
        s2 = self._sigma ** 2
        return float(0.5 * np.sum(r * r / s2 + np.log(s2)))

    def gradient(self, y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
        s2 = self._sigma ** 2
        return (yhat.astype(np.float64) - y.astype(np.float64)) / s2


class ComplexGaussianNLL(NegLogLikelihood):
    """Complex Gaussian NLL for k-space data.

    NLL = sum(|y - yhat|^2 / sigma^2 + 2*log(sigma)).

    Parameters
    ----------
    sigma : float
        Noise standard deviation per real/imag component.
    """

    def __init__(self, sigma: float = 1.0, **params: Any) -> None:
        super().__init__(sigma=sigma, **params)
        self._sigma = sigma

    def __call__(self, y: np.ndarray, yhat: np.ndarray) -> float:
        r = (y.ravel() - yhat.ravel()).astype(np.complex128)
        s2 = self._sigma ** 2
        return float(np.sum(np.abs(r) ** 2 / s2 + 2 * np.log(self._sigma)).real)

    def gradient(self, y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
        s2 = self._sigma ** 2
        return (yhat.astype(np.complex128) - y.astype(np.complex128)) / s2


class MixedPoissonGaussianNLL(NegLogLikelihood):
    """Mixed Poisson-Gaussian NLL: alpha * Poisson + (1-alpha) * Gaussian.

    Parameters
    ----------
    alpha : float
        Mixing weight for Poisson component (default 0.5).
    sigma : float
        Gaussian component noise std (default 1.0).
    """

    def __init__(self, alpha: float = 0.5, sigma: float = 1.0, **params: Any) -> None:
        super().__init__(alpha=alpha, sigma=sigma, **params)
        self._alpha = alpha
        self._poisson = PoissonNLL()
        self._gaussian = GaussianNLL(sigma=sigma)

    def __call__(self, y: np.ndarray, yhat: np.ndarray) -> float:
        return (
            self._alpha * self._poisson(y, yhat)
            + (1.0 - self._alpha) * self._gaussian(y, yhat)
        )

    def gradient(self, y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
        return (
            self._alpha * self._poisson.gradient(y, yhat)
            + (1.0 - self._alpha) * self._gaussian.gradient(y, yhat)
        )


class HuberNLL(NegLogLikelihood):
    """Huber loss (robust, differentiable approximation to L1).

    L(r) = 0.5*r^2              if |r| <= delta
           delta*(|r| - 0.5*d)  otherwise

    Parameters
    ----------
    delta : float
        Transition point between quadratic and linear regions (default 1.0).
    """

    def __init__(self, delta: float = 1.0, **params: Any) -> None:
        super().__init__(delta=delta, **params)
        self._delta = delta

    def __call__(self, y: np.ndarray, yhat: np.ndarray) -> float:
        r = (y.ravel() - yhat.ravel()).astype(np.float64)
        d = self._delta
        abs_r = np.abs(r)
        loss = np.where(abs_r <= d, 0.5 * r * r, d * (abs_r - 0.5 * d))
        return float(np.sum(loss))

    def gradient(self, y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
        r = (yhat.astype(np.float64) - y.astype(np.float64))
        d = self._delta
        return np.clip(r, -d, d)


class TukeyBiweightNLL(NegLogLikelihood):
    """Tukey biweight loss (strong outlier rejection).

    rho(r) = (c^2/6)*(1 - (1-(r/c)^2)^3)  if |r| <= c
             c^2/6                            otherwise

    Parameters
    ----------
    c : float
        Tuning constant; outliers beyond c are fully rejected (default 4.685).
    """

    def __init__(self, c: float = 4.685, **params: Any) -> None:
        super().__init__(c=c, **params)
        self._c = c

    def __call__(self, y: np.ndarray, yhat: np.ndarray) -> float:
        r = (y.ravel() - yhat.ravel()).astype(np.float64)
        c = self._c
        c2_6 = c * c / 6.0
        u = r / c
        mask = np.abs(r) <= c
        loss = np.where(mask, c2_6 * (1.0 - (1.0 - u * u) ** 3), c2_6)
        return float(np.sum(loss))

    def gradient(self, y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
        r = (yhat.astype(np.float64) - y.astype(np.float64))
        c = self._c
        u = r / c
        mask = np.abs(r) <= c
        # gradient of rho w.r.t. yhat: r * (1 - (r/c)^2)^2
        return np.where(mask, r * (1.0 - u * u) ** 2, 0.0)


# ---------------------------------------------------------------------------
# Registry + builder
# ---------------------------------------------------------------------------

OBJECTIVE_REGISTRY: Dict[str, Type[NegLogLikelihood]] = {
    "poisson": PoissonNLL,
    "gaussian": GaussianNLL,
    "complex_gaussian": ComplexGaussianNLL,
    "mixed_poisson_gaussian": MixedPoissonGaussianNLL,
    "huber": HuberNLL,
    "tukey_biweight": TukeyBiweightNLL,
}


def build_objective(spec: ObjectiveSpec) -> NegLogLikelihood:
    """Construct a NLL objective from an ObjectiveSpec.

    Raises
    ------
    KeyError
        If ``spec.kind`` is not in OBJECTIVE_REGISTRY.
    """
    if spec.kind not in OBJECTIVE_REGISTRY:
        raise KeyError(
            f"Unknown objective kind '{spec.kind}'. "
            f"Available: {sorted(OBJECTIVE_REGISTRY.keys())}"
        )
    cls = OBJECTIVE_REGISTRY[spec.kind]
    return cls(**spec.params)
