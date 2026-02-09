"""pwm_core.physics.adapters.callable_operator

Wrap user-provided python callables:
- forward_fn(x) -> y
- adjoint_fn(y) -> x

Useful for rapid prototyping or binding to existing codebases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from pwm_core.physics.base import BaseOperator


@dataclass
class CallableOperator(BaseOperator):
    """Operator backed by user-provided forward/adjoint callables.

    Parameters
    ----------
    name : str, optional
        Human-readable name (mapped to operator_id).
    forward_fn : callable
        Forward function: x -> y.
    adjoint_fn : callable
        Adjoint function: y -> x.
    x_shape : tuple[int, ...]
        Expected input shape.
    y_shape : tuple[int, ...]
        Expected output shape.
    """

    fwd: Callable | None = None
    adj: Callable | None = None

    def __init__(
        self,
        name: str = "callable_op",
        forward_fn: Callable | None = None,
        adjoint_fn: Callable | None = None,
        x_shape: Tuple[int, ...] | None = None,
        y_shape: Tuple[int, ...] | None = None,
        # Also support original interface
        fwd: Callable | None = None,
        adj: Callable | None = None,
        operator_id: str | None = None,
        theta: Dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        if x_shape is None:
            raise ValueError("x_shape is required for CallableOperator")
        if y_shape is None:
            raise ValueError("y_shape is required for CallableOperator")

        # Resolve callables: prefer forward_fn/adjoint_fn, fall back to fwd/adj
        resolved_fwd = forward_fn or fwd
        resolved_adj = adjoint_fn or adj

        if resolved_fwd is None:
            raise ValueError("forward_fn (or fwd) is required")
        if resolved_adj is None:
            raise ValueError("adjoint_fn (or adj) is required")

        super().__init__(
            operator_id=operator_id or name,
            theta=theta or {},
            _x_shape=tuple(x_shape),
            _y_shape=tuple(y_shape),
        )
        self.fwd = resolved_fwd
        self.adj = resolved_adj

    def forward(self, x: Any) -> Any:
        if self.fwd is None:
            raise ValueError("CallableOperator.fwd is None")
        return self.fwd(x)

    def adjoint(self, y: Any) -> Any:
        if self.adj is None:
            raise ValueError("CallableOperator.adj is None")
        return self.adj(y)
