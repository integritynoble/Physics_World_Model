"""pwm_core.world.budget

BudgetState:
- photon_budget: max_photons, exposure, bleaching params (optional)
- measurement_budget: sampling_rate, num_measurements, scan speed, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class BudgetState:
    photon_budget: Dict[str, Any] = field(default_factory=dict)
    measurement_budget: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {"photon_budget": self.photon_budget, "measurement_budget": self.measurement_budget}
