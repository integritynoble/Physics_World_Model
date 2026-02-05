"""pwm_core.physics.microscopy.widefield

Simple widefield (blur) operator for testing.

Forward: Gaussian convolution (blur)
Adjoint: Same Gaussian convolution (self-adjoint for symmetric PSF)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
from scipy.ndimage import gaussian_filter

from pwm_core.physics.base import BaseOperator


@dataclass
class WidefieldOperator(BaseOperator):
    """Widefield microscopy operator using Gaussian PSF blur.

    This is a simple self-adjoint operator where forward and adjoint
    are both Gaussian convolution (since a symmetric PSF is self-adjoint).

    Theta parameters:
        sigma: float - Gaussian blur sigma in pixels (default 2.0)
        mode: str - boundary handling mode (default 'reflect')
    """
    x_shape: tuple = field(default_factory=lambda: (64, 64))

    def __post_init__(self):
        if not hasattr(self, 'theta') or self.theta is None:
            self.theta = {}
        self.theta.setdefault('sigma', 2.0)
        self.theta.setdefault('mode', 'reflect')

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur (forward model of widefield imaging)."""
        sigma = float(self.theta.get('sigma', 2.0))
        mode = str(self.theta.get('mode', 'reflect'))
        return gaussian_filter(x.astype(np.float32), sigma=sigma, mode=mode)

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        """Adjoint is the same as forward for symmetric Gaussian PSF."""
        sigma = float(self.theta.get('sigma', 2.0))
        mode = str(self.theta.get('mode', 'reflect'))
        return gaussian_filter(y.astype(np.float32), sigma=sigma, mode=mode)

    def info(self) -> Dict[str, Any]:
        info = super().info()
        info['x_shape'] = self.x_shape
        return info
