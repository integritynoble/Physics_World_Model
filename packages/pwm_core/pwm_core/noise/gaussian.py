"""pwm_core.noise.gaussian

Additive Gaussian noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from pwm_core.noise.base import BaseNoise


@dataclass
class GaussianNoise(BaseNoise):
    def apply(self, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        sigma = float(self.params.get("sigma", 0.01))
        return (y + rng.normal(0.0, sigma, size=y.shape)).astype(np.float32)
