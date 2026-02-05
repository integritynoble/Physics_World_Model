"""pwm_core.world.sample

SampleState:
- motion/drift
- labeling density / blinking kinetics (fluorescence)
- structural priors (optional)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class SampleState:
    motion: Dict[str, Any] = field(default_factory=dict)
    labeling: Dict[str, Any] = field(default_factory=dict)
    structure: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {"motion": self.motion, "labeling": self.labeling, "structure": self.structure}
