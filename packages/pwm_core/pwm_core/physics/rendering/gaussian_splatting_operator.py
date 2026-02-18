"""Gaussian Splatting operator.

Implements 3D Gaussian Splatting rendering from multiple views.
Similar to NeRF but uses Gaussian splatting style rendering.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import ndimage

from pwm_core.physics.base import BaseOperator


class GaussianSplattingOperator(BaseOperator):
    """3D Gaussian Splatting rendering operator.

    Forward: Render Gaussians from multiple views
    Adjoint: Back-project views to Gaussian space
    """

    def __init__(
        self,
        operator_id: str = "gaussian_splatting",
        theta: Optional[Dict[str, Any]] = None,
        x_shape: Tuple[int, int, int] = (64, 64, 32),
        n_views: int = 10,
        splat_sigma: float = 2.0,
        seed: int = 42,
    ):
        self.operator_id = operator_id
        self.theta = theta or {}
        self.x_shape = x_shape
        self.n_views = n_views
        self.splat_sigma = splat_sigma

        # Generate viewing angles
        self.angles = np.linspace(0, 360, n_views, endpoint=False)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Render with Gaussian splatting."""
        H, W, D = self.x_shape

        # Handle 2D input
        if x.ndim == 2:
            x_3d = np.tile(x[:, :, np.newaxis], (1, 1, D))
        else:
            x_3d = x

        # Apply Gaussian blur to simulate splatting
        x_splatted = ndimage.gaussian_filter(x_3d, sigma=self.splat_sigma)

        y = np.zeros((self.n_views, H, W), dtype=np.float32)

        for i, angle in enumerate(self.angles):
            # Rotate volume
            rotated = ndimage.rotate(x_splatted, angle, axes=(0, 1), reshape=False, mode='constant', order=1)

            # Alpha compositing along depth (front-to-back)
            # Simplified: weighted sum with depth-based weights
            weights = np.exp(-0.1 * np.arange(D))
            weights = weights / weights.sum()

            projection = np.zeros((H, W), dtype=np.float32)
            for d in range(D):
                projection += rotated[:, :, d] * weights[d]

            y[i] = projection

        return y

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        """Back-project and unsplat."""
        H, W, D = self.x_shape
        x_bp = np.zeros((H, W, D), dtype=np.float32)

        weights = np.exp(-0.1 * np.arange(D))
        weights = weights / weights.sum()

        for i, angle in enumerate(self.angles):
            # Distribute projection along depth
            projection_3d = np.zeros((H, W, D), dtype=np.float32)
            for d in range(D):
                projection_3d[:, :, d] = y[i] * weights[d]

            # Rotate back
            rotated = ndimage.rotate(projection_3d, -angle, axes=(0, 1), reshape=False, mode='constant', order=1)

            x_bp += rotated

        # Apply adjoint of Gaussian (same as forward blur for symmetric kernel)
        x_bp = ndimage.gaussian_filter(x_bp, sigma=self.splat_sigma)

        return (x_bp / self.n_views).astype(np.float32)

    def info(self) -> Dict[str, Any]:
        return {
            "operator_id": self.operator_id,
            "x_shape": self.x_shape,
            "n_views": self.n_views,
            "splat_sigma": self.splat_sigma,
        }
