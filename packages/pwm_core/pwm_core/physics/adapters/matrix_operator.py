"""pwm_core.physics.adapters.matrix_operator

Wrap an explicit matrix A as an operator with forward/adjoint.
Supports dense numpy arrays; can be extended to scipy.sparse or LinearOperator.

For large A, store as a reference (RunBundle data manifest) and load lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from pwm_core.physics.base import BaseOperator


@dataclass
class MatrixOperator(BaseOperator):
    A: np.ndarray | None = None  # shape (m, n)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.A is None:
            raise ValueError("MatrixOperator.A is None")
        return self.A @ x.reshape(-1)

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        if self.A is None:
            raise ValueError("MatrixOperator.A is None")
        return (self.A.T @ y.reshape(-1))

    def info(self) -> Dict[str, Any]:
        info = super().info()
        if self.A is not None:
            info.update({"shape": list(self.A.shape), "dtype": str(self.A.dtype)})
        return info
