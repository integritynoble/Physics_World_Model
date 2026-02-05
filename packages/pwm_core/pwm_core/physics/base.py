"""pwm_core.physics.base

Core PhysicsOperator abstraction used across PWM.

Supports:
- Parametric operators: A(theta) with explicit forward/adjoint
- Matrix-backed operators (dense/sparse/LinearOperator)
- Callable-backed operators (user provides forward/adjoint)

This is intentionally minimal so contributors can implement new modalities quickly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol, Tuple

import numpy as np


class PhysicsOperator(Protocol):
    """Minimal operator API."""

    def forward(self, x: np.ndarray) -> np.ndarray: ...
    def adjoint(self, y: np.ndarray) -> np.ndarray: ...

    def set_theta(self, theta: Dict[str, Any]) -> None: ...
    def get_theta(self) -> Dict[str, Any]: ...

    def info(self) -> Dict[str, Any]: ...


@dataclass
class BaseOperator:
    """A convenience base class for operators."""
    operator_id: str
    theta: Dict[str, Any]

    def set_theta(self, theta: Dict[str, Any]) -> None:
        self.theta = dict(theta)

    def get_theta(self) -> Dict[str, Any]:
        return dict(self.theta)

    def info(self) -> Dict[str, Any]:
        return {"operator_id": self.operator_id, "theta": self.theta}
