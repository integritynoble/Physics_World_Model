"""pwm_core.noise.poisson_gaussian

Poisson + Gaussian mixture (common sensor model).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from pwm_core.noise.base import BaseNoise
from pwm_core.noise.poisson import PoissonNoise
from pwm_core.noise.gaussian import GaussianNoise


@dataclass
class PoissonGaussianNoise(BaseNoise):
    def apply(self, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        gain = float(self.params.get("gain", 1.0))
        sigma = float(self.params.get("sigma", 0.01))
        y1 = PoissonNoise(noise_id="poisson", params={"gain": gain}).apply(y, rng)
        y2 = GaussianNoise(noise_id="gaussian", params={"sigma": sigma}).apply(y1, rng)
        return y2
