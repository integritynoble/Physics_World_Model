"""
new_operator_template.py

Contributor template: add a new operator (forward model) to PWM.

How to use:
1) Copy this file to:
     packages/pwm_core/pwm_core/contrib/templates/my_operator.py
2) Implement the TODOs.
3) Register your operator via the plugin registry (entrypoints) OR by adding to
   pwm_core/core/registry.py in a PR.

Design goals:
- Clear forward(x, theta) and adjoint(y, theta)
- Deterministic + reproducible
- Safe parameter validation (bounds, units, dtype)
- Support "PhysicsTrue" vs "PhysicsModel" mismatch by allowing theta perturbations

Notes:
- Prefer torch tensors (CPU/GPU). Avoid heavy deps.
- If your operator is nonlinear, still implement:
    forward(x, theta) -> y
    linearize_jvp(x, v, theta) optional
    linearize_vjp(x, w, theta) optional
  (PWM can then use gradient-based refinement if available.)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from pwm_core.physics.base import PhysicsOperator, OperatorContext, OperatorMeta


@dataclass
class MyTheta:
    """Parameterization for your operator.

    Keep this dataclass small and JSON-serializable.
    """
    # TODO: define your parameters
    param_a: float = 1.0
    param_b: float = 0.0


class MyOperator(PhysicsOperator):
    """Example operator implementation.

    Replace 'MyOperator' and implement required methods.
    """

    OPERATOR_ID = "my_operator"

    def __init__(self, meta: OperatorMeta, ctx: Optional[OperatorContext] = None):
        super().__init__(meta=meta, ctx=ctx)

    @classmethod
    def from_assets(cls, assets: Dict[str, Any], **kwargs) -> "MyOperator":
        """Construct from assets (masks, PSFs, calibration files, etc)."""
        # TODO: parse assets and set OperatorMeta (dims, dtype, device)
        meta = OperatorMeta(
            operator_id=cls.OPERATOR_ID,
            x_shape=tuple(assets.get("x_shape", ())),
            y_shape=tuple(assets.get("y_shape", ())),
            is_linear=True,
            is_differentiable=True,
        )
        return cls(meta=meta, ctx=None)

    def validate_theta(self, theta: Dict[str, Any]) -> Dict[str, Any]:
        """Clamp/repair/validate theta dict and return sanitized copy."""
        # TODO: enforce bounds / units
        t = dict(theta)
        if "param_a" in t:
            t["param_a"] = float(t["param_a"])
        if "param_b" in t:
            t["param_b"] = float(t["param_b"])
        return t

    def forward(self, x: torch.Tensor, theta: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Compute y = A_theta(x)."""
        theta = self.validate_theta(theta or {})
        # TODO: implement forward
        # Example: identity
        y = x
        return y

    def adjoint(self, y: torch.Tensor, theta: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Compute x = A_theta^T(y) (adjoint)."""
        theta = self.validate_theta(theta or {})
        # TODO: implement adjoint
        x = y
        return x

    # Optional: for nonlinear operators and gradient-based refinement
    def jvp(self, x: torch.Tensor, v: torch.Tensor, theta: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Jacobian-vector product J(x) v."""
        raise NotImplementedError

    def vjp(self, x: torch.Tensor, w: torch.Tensor, theta: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Vector-Jacobian product J(x)^T w."""
        raise NotImplementedError


def register(registry) -> None:
    """Entry-point style registration.

    PWM can import this module and call register(registry).
    """
    registry.register_operator(MyOperator.OPERATOR_ID, MyOperator)
