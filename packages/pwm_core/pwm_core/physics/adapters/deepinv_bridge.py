"""pwm_core.physics.adapters.deepinv_bridge

Bridge between PWM PhysicsOperator and deepinv Physics objects.

This module is a thin adapter:
- PWM operators can be wrapped as deepinv-compatible objects via to_deepinv_physics().
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


class PWMtoDeepInvPhysics:
    """Wrap a PWM operator into a callable deepinv-compatible physics object.

    Usage::

        physics = to_deepinv_physics(pwm_op)
        y = physics(x)  # calls forward
    """

    def __init__(self, op: PhysicsOperator):
        self.op = op

    def __call__(self, x: Any) -> Any:
        """Forward pass: x -> y, handling torch tensors."""
        if hasattr(x, "detach"):
            import torch
            x_np = x.detach().cpu().numpy()
            y_np = self.op.forward(x_np)
            return torch.from_numpy(y_np).to(x.device)
        return self.op.forward(np.asarray(x))

    def A(self, x: Any) -> Any:
        """deepinv-style forward."""
        return self(x)

    def A_adjoint(self, y: Any) -> Any:
        """deepinv-style adjoint."""
        if hasattr(y, "detach"):
            import torch
            y_np = y.detach().cpu().numpy()
            x_np = self.op.adjoint(y_np)
            return torch.from_numpy(x_np).to(y.device)
        return self.op.adjoint(np.asarray(y))


def to_deepinv_physics(op: PhysicsOperator) -> PWMtoDeepInvPhysics:
    """Convert a PWM operator to a deepinv-compatible physics object.

    Parameters
    ----------
    op : PhysicsOperator
        Any PWM-compatible physics operator.

    Returns
    -------
    PWMtoDeepInvPhysics
        Callable wrapper with ``__call__``, ``A``, and ``A_adjoint`` methods.
    """
    return PWMtoDeepInvPhysics(op)


def wrap_deepinv_physics(physics_obj: Any) -> PhysicsOperator:
    """Wrap an existing deepinv Physics into PWM operator interface."""
    class _Wrapped(BaseOperator):
        def forward(self, x: np.ndarray) -> np.ndarray:
            raise NotImplementedError("Implement torch conversion in integration layer.")

        def adjoint(self, y: np.ndarray) -> np.ndarray:
            raise NotImplementedError("Implement torch conversion in integration layer.")

    return _Wrapped(operator_id=getattr(physics_obj, "name", "deepinv_physics"), theta={})
