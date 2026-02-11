"""pwm_core.physics.spectral.cassi_operator

Parametric CASSI operator A(theta).

This is a *skeleton* that encodes the interface + theta parameterization.
A full implementation will:
- build coded aperture / mask
- apply dispersion shift per wavelength
- integrate along spectral dimension into 2D measurement (or coded 3D, depending on setup)
- provide adjoint for reconstruction/gradient methods

Theta examples:
- dx, dy (alignment)
- disp_poly (dispersion polynomial coefficients)
- mask_shift / rotation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

from pwm_core.mismatch.subpixel import subpixel_shift_2d
from pwm_core.physics.base import BaseOperator
from pwm_core.physics.spectral.dispersion_models import dispersion_shift


@dataclass
class CASSIOperator(BaseOperator):
    mask: np.ndarray | None = None  # (H, W) binary/float mask

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (H, W, L) hyperspectral cube or (H, W) 2D -> y: (H, W) measurement."""
        if self.mask is None:
            raise ValueError("CASSIOperator.mask is None")
        H, W = self.mask.shape
        L = int(self.theta.get("L", 8))

        # Handle 2D input by creating synthetic spectral bands
        if x.ndim == 2:
            x_3d = np.zeros((H, W, L), dtype=np.float32)
            for l in range(L):
                # Simulate spectral variation with wavelength-dependent intensity
                weight = 1.0 - 0.05 * abs(l - L // 2)
                x_3d[:, :, l] = x * weight
        else:
            x_3d = x
            L = x_3d.shape[2]

        y = np.zeros((H, W), dtype=np.float32)
        # Sum masked bands shifted by dispersion
        for l in range(L):
            dx, dy = dispersion_shift(self.theta, band=l)
            band = x_3d[:, :, l]
            band_s = subpixel_shift_2d(band, dx, dy)
            y += band_s * self.mask
        return y

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        """Adjoint maps 2D y back to a cube (H,W,L) (placeholder)."""
        if self.mask is None:
            raise ValueError("CASSIOperator.mask is None")
        H, W = self.mask.shape
        L = int(self.theta.get("L", 8))
        x = np.zeros((H, W, L), dtype=np.float32)
        for l in range(L):
            dx, dy = dispersion_shift(self.theta, band=l)
            back = y * self.mask
            back_s = subpixel_shift_2d(back, -dx, -dy)
            x[:, :, l] = back_s
        return x
