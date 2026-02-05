"""pwm_core.world.environment

EnvironmentState:
- background / autofluorescence
- scattering / attenuation vs depth
- ambient light (photography) etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class EnvironmentState:
    background: Dict[str, Any] = field(default_factory=dict)
    scattering: Dict[str, Any] = field(default_factory=dict)
    attenuation: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {"background": self.background, "scattering": self.scattering, "attenuation": self.attenuation}
