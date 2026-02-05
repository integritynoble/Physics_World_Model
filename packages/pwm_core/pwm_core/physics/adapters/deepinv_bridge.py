"""pwm_core.physics.adapters.deepinv_bridge

Bridge between PWM PhysicsOperator and deepinv Physics objects.

This module is a thin adapter:
- PWM operators can be wrapped as deepinv.physics.Physics if deepinv is installed.
- deepinv operators can be wrapped back to PWM PhysicsOperator.

Kept optional to avoid forcing deepinv dependency in all environments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from pwm_core.physics.base import BaseOperator, PhysicsOperator


def has_deepinv() -> bool:
    try:
        import deepinv  # noqa: F401
        return True
    except Exception:
        return False


@dataclass
class PWMtoDeepInvPhysics:
    """Wrap a PWM operator into something deepinv-like."""
    op: PhysicsOperator

    def A(self, x):
        # deepinv expects torch tensors typically; keep this as a placeholder.
        raise NotImplementedError("Implement torch bridge when wiring deepinv runtime.")


def wrap_deepinv_physics(physics_obj: Any) -> PhysicsOperator:
    """Wrap an existing deepinv Physics into PWM operator interface."""
    class _Wrapped(BaseOperator):
        def forward(self, x: np.ndarray) -> np.ndarray:
            raise NotImplementedError("Implement torch conversion in integration layer.")

        def adjoint(self, y: np.ndarray) -> np.ndarray:
            raise NotImplementedError("Implement torch conversion in integration layer.")

    return _Wrapped(operator_id=getattr(physics_obj, "name", "deepinv_physics"), theta={})
