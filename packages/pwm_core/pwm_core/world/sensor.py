"""pwm_core.world.sensor

SensorState:
- saturation, quantization bits
- read noise, fixed-pattern noise
- nonlinearity / gain
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class SensorState:
    read_noise: Dict[str, Any] = field(default_factory=dict)
    fpn: Dict[str, Any] = field(default_factory=dict)
    nonlinearity: Dict[str, Any] = field(default_factory=dict)
    gain: Dict[str, Any] = field(default_factory=dict)
    quantization_bits: Optional[int] = 12
    saturation_full_well: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "read_noise": self.read_noise,
            "fpn": self.fpn,
            "nonlinearity": self.nonlinearity,
            "gain": self.gain,
            "quantization_bits": self.quantization_bits,
            "saturation_full_well": self.saturation_full_well,
        }
