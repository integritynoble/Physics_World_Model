"""pwm_core.world.calibration

CalibrationState (theta):
- alignment (dx, dy, rotation)
- PSF params
- aberrations, dispersion
- gain drift, timing jitter
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class CalibrationState:
    alignment: Dict[str, Any] = field(default_factory=dict)
    psf: Dict[str, Any] = field(default_factory=dict)
    aberrations: Dict[str, Any] = field(default_factory=dict)
    dispersion: Dict[str, Any] = field(default_factory=dict)
    gain_drift: Dict[str, Any] = field(default_factory=dict)
    timing: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "alignment": self.alignment,
            "psf": self.psf,
            "aberrations": self.aberrations,
            "dispersion": self.dispersion,
            "gain_drift": self.gain_drift,
            "timing": self.timing,
        }
