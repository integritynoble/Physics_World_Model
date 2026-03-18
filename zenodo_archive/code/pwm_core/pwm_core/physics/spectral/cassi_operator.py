"""pwm_core.physics.spectral.cassi_operator

Parametric SD-CASSI operator A(theta) following ECCV-2020 formulation.

Provides two operator classes:
- CASSIOperator: legacy operator, shifts within (H, W) grid, output (H, W).
- SDCASSIOperator: paper-correct extended output (Nx, Ny + (L-1)*step),
  configurable dispersion axis, quadratic dispersion support.

Theta examples:
- dx, dy (alignment)
- disp_poly (dispersion polynomial coefficients)
- mask_shift / rotation
- disp_a2 (quadratic dispersion curvature for SD distortion)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import numpy as np

from pwm_core.mismatch.subpixel import subpixel_shift_2d
from pwm_core.physics.base import BaseOperator
from pwm_core.physics.spectral.dispersion_models import dispersion_shift


@dataclass
class CASSIOperator(BaseOperator):
    """Legacy CASSI operator: shifts within (H, W) grid, output shape (H, W)."""
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
        """Adjoint maps 2D y back to a cube (H,W,L)."""
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


@dataclass
class SDCASSIOperator(BaseOperator):
    """SD-CASSI operator following ECCV-2020 formulation.

    Forward: Y(x,y) = Σ_l shift( X[:,:,l] ⊙ M, d_l )
    Output shape: (Nx, Ny + (L-1)*step) when disp_axis=1.

    Parameters in theta:
        L: number of spectral bands
        dispersion_step: pixels per band (default 2.0, 54-pixel total for 28 ch)
        disp_axis: 0=rows, 1=columns (default 1, ECCV-2020 convention)
        disp_a2: quadratic dispersion curvature (default 0.0)
    """
    mask: np.ndarray | None = None  # (Nx, Ny) binary/float mask

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (Nx, Ny, L) spectral cube -> y: (Nx, Ny_ext) measurement."""
        if self.mask is None:
            raise ValueError("SDCASSIOperator.mask is None")
        Nx, Ny = self.mask.shape
        L = int(self.theta.get("L", x.shape[2] if x.ndim == 3 else 8))
        step = float(self.theta.get("dispersion_step", 2.0))
        disp_axis = int(self.theta.get("disp_axis", 1))
        a2 = float(self.theta.get("disp_a2", 0.0))

        if x.ndim == 2:
            x = np.stack([x] * L, axis=-1)

        ext = int(round(step * (L - 1)))
        l_c = (L - 1) / 2.0

        if disp_axis == 1:
            Y = np.zeros((Nx, Ny + ext), dtype=np.float64)
            for l in range(L):
                dl = l - l_c
                d_l = int(round(step * l + a2 * dl * dl))
                coded_slice = x[:, :, l].astype(np.float64) * self.mask
                y_start = max(0, d_l)
                y_end = min(Ny + ext, Ny + d_l)
                src_start = max(0, -d_l)
                src_end = src_start + (y_end - y_start)
                if y_end > y_start and src_end > src_start:
                    Y[:, y_start:y_end] += coded_slice[:, src_start:src_end]
        else:
            Y = np.zeros((Nx + ext, Ny), dtype=np.float64)
            for l in range(L):
                dl = l - l_c
                d_l = int(round(step * l + a2 * dl * dl))
                coded_slice = x[:, :, l].astype(np.float64) * self.mask
                x_start = max(0, d_l)
                x_end = min(Nx + ext, Nx + d_l)
                src_start = max(0, -d_l)
                src_end = src_start + (x_end - x_start)
                if x_end > x_start and src_end > src_start:
                    Y[x_start:x_end, :] += coded_slice[src_start:src_end, :]
        return Y

    def adjoint(self, Y: np.ndarray) -> np.ndarray:
        """Y: (Nx, Ny_ext) measurement -> x: (Nx, Ny, L) spectral cube."""
        if self.mask is None:
            raise ValueError("SDCASSIOperator.mask is None")
        Nx, Ny = self.mask.shape
        L = int(self.theta.get("L", 8))
        step = float(self.theta.get("dispersion_step", 2.0))
        disp_axis = int(self.theta.get("disp_axis", 1))
        a2 = float(self.theta.get("disp_a2", 0.0))
        ext = int(round(step * (L - 1)))
        l_c = (L - 1) / 2.0

        X = np.zeros((Nx, Ny, L), dtype=np.float64)
        if disp_axis == 1:
            for l in range(L):
                dl = l - l_c
                d_l = int(round(step * l + a2 * dl * dl))
                y_start = max(0, d_l)
                y_end = min(Ny + ext, Ny + d_l)
                src_start = max(0, -d_l)
                src_end = src_start + (y_end - y_start)
                if y_end > y_start and src_end > src_start:
                    X[:, src_start:src_end, l] = (
                        self.mask[:, src_start:src_end] * Y[:, y_start:y_end]
                    )
        else:
            for l in range(L):
                dl = l - l_c
                d_l = int(round(step * l + a2 * dl * dl))
                x_start = max(0, d_l)
                x_end = min(Nx + ext, Nx + d_l)
                src_start = max(0, -d_l)
                src_end = src_start + (x_end - x_start)
                if x_end > x_start and src_end > src_start:
                    X[src_start:src_end, :, l] = (
                        self.mask[src_start:src_end, :] * Y[x_start:x_end, :]
                    )
        return X
