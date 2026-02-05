"""Single-Pixel Camera (SPC) operator.

Implements compressed sensing measurements: y = Ax where A is a measurement matrix.
Output is 1D measurements from 2D input.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from pwm_core.physics.base import BaseOperator


class SPCOperator(BaseOperator):
    """Single-pixel camera operator with Hadamard-like measurements.

    Forward: y = A @ x.flatten()  (2D -> 1D)
    Adjoint: x = A.T @ y reshaped to 2D
    """

    def __init__(
        self,
        operator_id: str = "spc",
        theta: Optional[Dict[str, Any]] = None,
        x_shape: Tuple[int, int] = (64, 64),
        sampling_rate: float = 0.15,
        seed: int = 42,
    ):
        self.operator_id = operator_id
        self.theta = theta or {}
        self.x_shape = x_shape
        self.sampling_rate = sampling_rate

        # Compute measurement dimensions
        N = x_shape[0] * x_shape[1]
        self.n_measurements = int(N * sampling_rate)

        # Generate measurement matrix (Hadamard-like random)
        rng = np.random.default_rng(seed)
        self.A = (rng.random((self.n_measurements, N)) > 0.5).astype(np.float32) * 2 - 1
        self.A /= np.sqrt(N)  # Normalize

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply SPC measurement: y = A @ x.flatten()"""
        x_flat = x.flatten().astype(np.float32)
        y = self.A @ x_flat
        return y.astype(np.float32)

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        """Apply adjoint: x = A.T @ y reshaped"""
        x_flat = self.A.T @ y.flatten().astype(np.float32)
        return x_flat.reshape(self.x_shape).astype(np.float32)

    def info(self) -> Dict[str, Any]:
        return {
            "operator_id": self.operator_id,
            "x_shape": self.x_shape,
            "n_measurements": self.n_measurements,
            "sampling_rate": self.sampling_rate,
        }
