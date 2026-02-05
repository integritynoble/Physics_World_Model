"""pwm_core.world.task

TaskState (optional but recommended):
- what the user wants PWM to do (simulate, reconstruct, fit operator, etc.)
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class TaskKind(str, Enum):
    simulate_only = "simulate_only"
    reconstruct_only = "reconstruct_only"
    simulate_and_reconstruct = "simulate_and_reconstruct"

    fit_operator_only = "fit_operator_only"                 # NEW
    calibrate_and_reconstruct = "calibrate_and_reconstruct" # NEW (y + A(theta) -> fit theta -> recon)


@dataclass
class TaskState:
    kind: TaskKind = TaskKind.simulate_and_reconstruct
    notes: Optional[str] = None
