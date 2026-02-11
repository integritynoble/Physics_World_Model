"""pwm_core.objectives.noise_model
====================================

NoiseModel ABC and concrete implementations.

Separates noise sampling (Mode S) from objective/loss function (Mode I).
Each NoiseModel knows how to:
  1. Sample noisy measurements from clean data
  2. Compute log-likelihood
  3. Suggest a default objective (NLL) for reconstruction
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from pwm_core.objectives.base import ObjectiveSpec


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
                raise ValueError(f"Field '{field_name}' contains {val!r}")
        return self


# ---------------------------------------------------------------------------
# NoiseModelSpec
# ---------------------------------------------------------------------------

class NoiseModelSpec(StrictBaseModel):
    """Specification for constructing a NoiseModel."""
    kind: str
    params: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# NoiseModel ABC
# ---------------------------------------------------------------------------

class NoiseModel(ABC):
    """Abstract base for noise models."""

    @abstractmethod
    def sample(self, y_clean: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Draw a noisy measurement from clean data."""
        ...

    @abstractmethod
    def log_likelihood(self, y: np.ndarray, y_clean: np.ndarray) -> float:
        """Compute log-likelihood of observed y given clean y_clean."""
        ...

    @abstractmethod
    def default_objective(self) -> ObjectiveSpec:
        """Return the default NLL objective for this noise model."""
        ...


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------

class PoissonGaussianNoiseModel(NoiseModel):
    """Mixed Poisson-Gaussian noise: shot noise + read noise."""

    def __init__(self, peak_photons: float = 10000.0, read_sigma: float = 0.01):
        self.peak_photons = peak_photons
        self.read_sigma = read_sigma

    def sample(self, y_clean: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        scaled = np.clip(y_clean, 0, None) * self.peak_photons
        shot = rng.poisson(np.clip(scaled, 0, 1e8)).astype(np.float64)
        read = rng.normal(0, self.read_sigma, size=y_clean.shape)
        return shot / max(self.peak_photons, 1e-30) + read

    def log_likelihood(self, y: np.ndarray, y_clean: np.ndarray) -> float:
        # Approximate: Gaussian with variance = shot + read^2
        var = np.clip(y_clean, 1e-10, None) / self.peak_photons + self.read_sigma ** 2
        residual = y - y_clean
        return float(-0.5 * np.sum(residual ** 2 / var + np.log(2 * np.pi * var)))

    def default_objective(self) -> ObjectiveSpec:
        return ObjectiveSpec(kind="mixed_poisson_gaussian",
                           params={"alpha": 0.5, "sigma": self.read_sigma})


class GaussianNoiseModel(NoiseModel):
    """Additive white Gaussian noise."""

    def __init__(self, sigma: float = 0.01):
        self.sigma = sigma

    def sample(self, y_clean: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return y_clean + rng.normal(0, self.sigma, size=y_clean.shape)

    def log_likelihood(self, y: np.ndarray, y_clean: np.ndarray) -> float:
        n = y.size
        residual = y - y_clean
        return float(-0.5 * n * np.log(2 * np.pi * self.sigma ** 2)
                     - 0.5 * np.sum(residual ** 2) / self.sigma ** 2)

    def default_objective(self) -> ObjectiveSpec:
        return ObjectiveSpec(kind="gaussian", params={"sigma": self.sigma})


class ComplexGaussianNoiseModel(NoiseModel):
    """Complex Gaussian noise for k-space / coherent imaging."""

    def __init__(self, sigma: float = 0.01):
        self.sigma = sigma

    def sample(self, y_clean: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        noise = (rng.normal(0, self.sigma / np.sqrt(2), size=y_clean.shape)
                + 1j * rng.normal(0, self.sigma / np.sqrt(2), size=y_clean.shape))
        return y_clean + noise

    def log_likelihood(self, y: np.ndarray, y_clean: np.ndarray) -> float:
        residual = y - y_clean
        return float(-np.sum(np.abs(residual) ** 2) / self.sigma ** 2)

    def default_objective(self) -> ObjectiveSpec:
        return ObjectiveSpec(kind="complex_gaussian", params={"sigma": self.sigma})


class PoissonOnlyNoiseModel(NoiseModel):
    """Pure Poisson noise (e.g., photon-counting CT)."""

    def __init__(self, peak_photons: float = 10000.0):
        self.peak_photons = peak_photons

    def sample(self, y_clean: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        scaled = np.clip(y_clean, 0, None) * self.peak_photons
        counts = rng.poisson(np.clip(scaled, 0, 1e8)).astype(np.float64)
        return counts / max(self.peak_photons, 1e-30)

    def log_likelihood(self, y: np.ndarray, y_clean: np.ndarray) -> float:
        lam = np.clip(y_clean * self.peak_photons, 1e-10, None)
        k = np.clip(y * self.peak_photons, 0, None)
        return float(np.sum(k * np.log(lam) - lam))

    def default_objective(self) -> ObjectiveSpec:
        return ObjectiveSpec(kind="poisson")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

NOISE_MODEL_REGISTRY: Dict[str, Type[NoiseModel]] = {
    "poisson_gaussian": PoissonGaussianNoiseModel,
    "gaussian": GaussianNoiseModel,
    "complex_gaussian": ComplexGaussianNoiseModel,
    "poisson_only": PoissonOnlyNoiseModel,
    # Aliases matching primitive_ids
    "poisson_gaussian_sensor": PoissonGaussianNoiseModel,
    "complex_gaussian_sensor": ComplexGaussianNoiseModel,
    "poisson_only_sensor": PoissonOnlyNoiseModel,
}


def build_noise_model(spec: NoiseModelSpec) -> NoiseModel:
    """Build a NoiseModel from a spec."""
    if spec.kind not in NOISE_MODEL_REGISTRY:
        raise KeyError(f"Unknown noise model kind '{spec.kind}'. "
                      f"Available: {sorted(NOISE_MODEL_REGISTRY.keys())}")
    cls = NOISE_MODEL_REGISTRY[spec.kind]
    return cls(**spec.params)


# Map primitive_id -> noise model constructor kwargs
_PRIMITIVE_TO_NOISE_MODEL: Dict[str, str] = {
    "poisson_gaussian_sensor": "poisson_gaussian",
    "complex_gaussian_sensor": "complex_gaussian",
    "poisson_only_sensor": "poisson_only",
    "poisson_gaussian": "poisson_gaussian",
    "poisson": "poisson_only",
    "gaussian": "gaussian",
}


def noise_model_from_primitive(primitive_id: str, params: Dict[str, Any]) -> NoiseModel:
    """Build a NoiseModel from a noise primitive's ID and params."""
    kind = _PRIMITIVE_TO_NOISE_MODEL.get(primitive_id, "gaussian")
    # Extract relevant params
    nm_params: Dict[str, Any] = {}
    if "peak_photons" in params:
        nm_params["peak_photons"] = float(params["peak_photons"])
    if "read_sigma" in params:
        nm_params["read_sigma"] = float(params["read_sigma"])
    if "sigma" in params:
        nm_params["sigma"] = float(params["sigma"])
    return build_noise_model(NoiseModelSpec(kind=kind, params=nm_params))
