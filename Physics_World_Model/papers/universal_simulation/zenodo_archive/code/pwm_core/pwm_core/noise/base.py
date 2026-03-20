"""pwm_core.noise.base

Noise models applied to measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol

import numpy as np


class NoiseModel(Protocol):
    def apply(self, y: np.ndarray, rng: np.random.Generator) -> np.ndarray: ...
    def info(self) -> Dict[str, Any]: ...


@dataclass
class BaseNoise:
    noise_id: str
    params: Dict[str, Any]

    def info(self) -> Dict[str, Any]:
        return {"noise_id": self.noise_id, "params": self.params}
