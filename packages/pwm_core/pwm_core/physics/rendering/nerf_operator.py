"""NeRF (Neural Radiance Fields) operator.

Implements multi-view rendering from a 3D volume.
Input is 3D volume (H, W, D), output is stack of 2D views.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import ndimage

from pwm_core.physics.base import BaseOperator


class NeRFOperator(BaseOperator):
    """NeRF-style multi-view rendering operator.

    Forward: Render volume from multiple viewing angles
    Adjoint: Back-project views into volume
    """

    def __init__(
        self,
        operator_id: str = "nerf",
        theta: Optional[Dict[str, Any]] = None,
        x_shape: Tuple[int, int, int] = (64, 64, 32),
        n_views: int = 10,
        seed: int = 42,
    ):
        self.operator_id = operator_id
        self.theta = theta or {}
        self.x_shape = x_shape
        self.n_views = n_views

        # Generate viewing angles (azimuth angles around the object)
        self.angles = np.linspace(0, 360, n_views, endpoint=False)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Render volume from multiple views using simple projection."""
        H, W, D = self.x_shape

        # Handle 2D input by expanding to 3D
        if x.ndim == 2:
            x_3d = np.tile(x[:, :, np.newaxis], (1, 1, D))
        else:
            x_3d = x

        y = np.zeros((self.n_views, H, W), dtype=np.float32)

        for i, angle in enumerate(self.angles):
            # Rotate volume around vertical axis (simple approximation)
            rotated = ndimage.rotate(x_3d, angle, axes=(0, 1), reshape=False, mode='constant', order=1)

            # Project along depth axis (sum projection)
            projection = rotated.sum(axis=2)

            # Normalize
            y[i] = (projection / D).astype(np.float32)

        return y

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        """Back-project views into volume."""
        H, W, D = self.x_shape
        x_bp = np.zeros((H, W, D), dtype=np.float32)

        for i, angle in enumerate(self.angles):
            # Expand 2D projection to 3D (smear along depth)
            projection_3d = np.tile(y[i][:, :, np.newaxis], (1, 1, D))

            # Rotate back
            rotated = ndimage.rotate(projection_3d, -angle, axes=(0, 1), reshape=False, mode='constant', order=1)

            x_bp += rotated

        return (x_bp / self.n_views).astype(np.float32)

    def info(self) -> Dict[str, Any]:
        return {
            "operator_id": self.operator_id,
            "x_shape": self.x_shape,
            "n_views": self.n_views,
        }
