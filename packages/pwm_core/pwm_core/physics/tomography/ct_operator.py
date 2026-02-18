"""CT (Computed Tomography) operator.

Implements Radon transform (sinogram projection).
Output is 2D sinogram (n_angles, width) from 2D input.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import ndimage

from pwm_core.physics.base import BaseOperator


class CTOperator(BaseOperator):
    """CT/Radon transform operator.

    Forward: Compute sinogram projections
    Adjoint: Back-projection (transpose of Radon)
    """

    def __init__(
        self,
        operator_id: str = "ct",
        theta: Optional[Dict[str, Any]] = None,
        x_shape: Tuple[int, int] = (64, 64),
        n_angles: int = 180,
    ):
        self.operator_id = operator_id
        self.theta = theta or {}
        self.x_shape = x_shape
        self.n_angles = n_angles
        self.angles = np.linspace(0, 180, n_angles, endpoint=False)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute sinogram (Radon transform)."""
        H, W = self.x_shape
        sinogram = np.zeros((self.n_angles, W), dtype=np.float32)

        for i, angle in enumerate(self.angles):
            rotated = ndimage.rotate(x, angle, reshape=False, mode='constant', order=1)
            sinogram[i, :] = rotated.sum(axis=0)

        return sinogram.astype(np.float32)

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        """Back-projection (adjoint of Radon)."""
        H, W = self.x_shape
        x_bp = np.zeros((H, W), dtype=np.float32)

        for i, angle in enumerate(self.angles):
            # Create image from projection (smear along rows)
            projection = y[i, :]
            smeared = np.tile(projection, (H, 1))
            # Rotate back
            rotated = ndimage.rotate(smeared, -angle, reshape=False, mode='constant', order=1)
            x_bp += rotated

        return (x_bp / self.n_angles).astype(np.float32)

    def info(self) -> Dict[str, Any]:
        return {
            "operator_id": self.operator_id,
            "x_shape": self.x_shape,
            "n_angles": self.n_angles,
        }
