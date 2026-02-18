"""pwm_core.mismatch.priors

Priors over mismatch parameters:
- used to sample theta_true vs theta_model
- used to define search ranges for calibration

This file contains simple, extensible parameter prior definitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np


@dataclass
class UniformPrior:
    low: float
    high: float

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.low, self.high))


@dataclass
class NormalPrior:
    mean: float
    sigma: float
    clip: Optional[Tuple[float, float]] = None

    def sample(self, rng: np.random.Generator) -> float:
        x = float(rng.normal(self.mean, self.sigma))
        if self.clip:
            x = float(np.clip(x, self.clip[0], self.clip[1]))
        return x


def sample_theta(prior_spec: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
    """Sample theta dict from a prior specification.

    prior_spec example:
      {"dx": {"type":"uniform", "low":-2, "high":2},
       "sigma": {"type":"normal", "mean":1.2, "sigma":0.2, "clip":[0.5,2.0]}}
    """
    theta: Dict[str, Any] = {}
    for k, v in prior_spec.items():
        t = v.get("type", "uniform")
        if t == "uniform":
            theta[k] = UniformPrior(float(v["low"]), float(v["high"])).sample(rng)
        elif t == "normal":
            clip = tuple(v["clip"]) if "clip" in v else None
            theta[k] = NormalPrior(float(v["mean"]), float(v["sigma"]), clip=clip).sample(rng)
        else:
            raise ValueError(f"Unknown prior type: {t}")
    return theta
