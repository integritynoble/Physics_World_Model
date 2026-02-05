"""pwm_core.physics.adapters.callable_operator

Wrap user-provided python callables:
- fwd(x) -> y
- adj(y) -> x

Useful for rapid prototyping or binding to existing codebases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

import numpy as np

from pwm_core.physics.base import BaseOperator


@dataclass
class CallableOperator(BaseOperator):
    fwd: Callable[[np.ndarray], np.ndarray] | None = None
    adj: Callable[[np.ndarray], np.ndarray] | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.fwd is None:
            raise ValueError("CallableOperator.fwd is None")
        return self.fwd(x)

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        if self.adj is None:
            raise ValueError("CallableOperator.adj is None")
        return self.adj(y)
