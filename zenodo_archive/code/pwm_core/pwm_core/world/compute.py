"""pwm_core.world.compute

ComputeState (optional):
- runtime / memory budgets
- streaming constraints
- device preferences (cpu/gpu)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ComputeState:
    max_seconds: Optional[float] = None
    max_memory_gb: Optional[float] = None
    device: str = "auto"  # cpu|cuda|auto
    streaming: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "max_seconds": self.max_seconds,
            "max_memory_gb": self.max_memory_gb,
            "device": self.device,
            "streaming": self.streaming,
        }
