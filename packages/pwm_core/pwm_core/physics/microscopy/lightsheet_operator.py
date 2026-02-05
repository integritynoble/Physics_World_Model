"""Light-Sheet Microscopy operator.

Implements light-sheet imaging with stripe artifacts and attenuation.
Input/output are 3D volumes.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import ndimage

from pwm_core.physics.base import BaseOperator


class LightsheetOperator(BaseOperator):
    """Light-sheet microscopy operator.

    Forward: Apply PSF blur + stripe artifacts + depth attenuation
    Adjoint: Approximate transpose
    """

    def __init__(
        self,
        operator_id: str = "lightsheet",
        theta: Optional[Dict[str, Any]] = None,
        x_shape: Tuple[int, int, int] = (64, 64, 32),
        psf_sigma: Tuple[float, float, float] = (1.5, 1.5, 1.0),
        stripe_strength: float = 0.2,
        attenuation_coef: float = 0.02,
    ):
        self.operator_id = operator_id
        self.theta = theta or {}
        self.x_shape = x_shape
        self.psf_sigma = psf_sigma
        self.stripe_strength = stripe_strength
        self.attenuation_coef = attenuation_coef

        # Pre-compute stripe pattern
        H, W, D = x_shape
        self.stripes = 1.0 - self.stripe_strength * (
            0.5 + 0.5 * np.sin(2 * np.pi * np.arange(H) / 10)
        )[:, None, None]

        # Pre-compute attenuation
        self.attenuation = np.exp(-self.attenuation_coef * np.arange(D))[None, None, :]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply light-sheet forward model."""
        H, W, D = self.x_shape

        # Handle 2D input by expanding to 3D
        if x.ndim == 2:
            x_3d = np.tile(x[:, :, np.newaxis], (1, 1, D))
        else:
            x_3d = x

        # Apply anisotropic PSF blur
        y = ndimage.gaussian_filter(x_3d, sigma=self.psf_sigma)

        # Apply stripe artifacts
        y = y * self.stripes

        # Apply depth attenuation
        y = y * self.attenuation

        return y.astype(np.float32)

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        """Adjoint of light-sheet operator."""
        # Reverse attenuation weighting
        x = y * self.attenuation

        # Reverse stripe weighting
        x = x * self.stripes

        # Apply adjoint of blur (same as blur for Gaussian)
        x = ndimage.gaussian_filter(x, sigma=self.psf_sigma)

        return x.astype(np.float32)

    def info(self) -> Dict[str, Any]:
        return {
            "operator_id": self.operator_id,
            "x_shape": self.x_shape,
            "psf_sigma": self.psf_sigma,
        }
