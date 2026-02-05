"""pwm_core.noise.sensor_pipeline

Sensor effects beyond stochastic noise:
- saturation / full-well
- quantization
- fixed-pattern noise (FPN)
- nonlinearity (gamma)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class SensorPipeline:
    saturation_full_well: Optional[float] = None
    quantization_bits: Optional[int] = None
    fpn_sigma: float = 0.0
    gamma: Optional[float] = None  # simple power law

    def apply(self, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        out = y.astype(np.float32)

        if self.fpn_sigma > 0:
            out = out + rng.normal(0.0, self.fpn_sigma, size=out.shape).astype(np.float32)

        if self.gamma is not None:
            g = float(self.gamma)
            out = np.clip(out, 0.0, None)
            out = np.power(out, g).astype(np.float32)

        if self.saturation_full_well is not None:
            fw = float(self.saturation_full_well)
            out = np.clip(out, 0.0, fw)

        if self.quantization_bits is not None:
            b = int(self.quantization_bits)
            levels = 2 ** b
            mn, mx = float(out.min()), float(out.max())
            if mx > mn:
                out_n = (out - mn) / (mx - mn)
                out_q = np.round(out_n * (levels - 1)) / (levels - 1)
                out = (out_q * (mx - mn) + mn).astype(np.float32)

        return out

    def info(self) -> Dict[str, Any]:
        return {
            "saturation_full_well": self.saturation_full_well,
            "quantization_bits": self.quantization_bits,
            "fpn_sigma": self.fpn_sigma,
            "gamma": self.gamma,
        }
