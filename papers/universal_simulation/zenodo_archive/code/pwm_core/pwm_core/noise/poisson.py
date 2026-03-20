"""pwm_core.noise.poisson

Poisson (shot) noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from pwm_core.noise.base import BaseNoise


@dataclass
class PoissonNoise(BaseNoise):
    def apply(self, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        gain = float(self.params.get("gain", 1.0))
        y_scaled = np.clip(y * gain, 0.0, None)
        return rng.poisson(y_scaled).astype(np.float32) / gain
