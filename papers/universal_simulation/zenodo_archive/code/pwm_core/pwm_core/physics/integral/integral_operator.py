"""Integral Photography operator.

Depth-dependent PSF convolution forward model for integral (plenoptic) imaging.
Forward: multi-depth volume -> 2D image via per-depth PSF convolution
Adjoint: 2D image -> multi-depth volume via conjugate filtering

    y = sum_d  w_d * conv(x[:, :, d], PSF_d)

where PSF_d is a Gaussian with sigma increasing with depth, modelling
the defocus blur inherent to plenoptic cameras.

References:
- Ng, R. et al. (2005). "Light Field Photography with a Hand-held
  Plenoptic Camera", Stanford Tech Report.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pwm_core.physics.base import BaseOperator, OperatorMetadata


class IntegralOperator(BaseOperator):
    """Integral Photography depth-PSF convolution operator.

    Forward: x (ny, nx, n_depths) -> y (ny, nx)
        y = sum_d IFFT2(OTF_d * FFT2(x[:, :, d]))
        where OTF_d = w_d * exp(-2 * pi^2 * sigma_d^2 * (fy^2 + fx^2))

    Adjoint: y (ny, nx) -> x (ny, nx, n_depths)
        x[:, :, d] = IFFT2(conj(OTF_d) * FFT2(y))  for each d
    """

    def __init__(
        self,
        operator_id: str = "integral",
        theta: Optional[Dict[str, Any]] = None,
        ny: int = 64,
        nx: int = 64,
        n_depths: int = 8,
        psf_sigmas: Optional[np.ndarray] = None,
        depth_weights: Optional[np.ndarray] = None,
    ):
        self.operator_id = operator_id
        self.theta = theta or {}
        self.ny = ny
        self.nx = nx
        self.n_depths = n_depths
        self._x_shape = (ny, nx, n_depths)
        self._y_shape = (ny, nx)
        self._is_linear = True
        self._supports_autodiff = False

        # Per-depth PSF widths (increasing blur with depth)
        if psf_sigmas is None:
            self.psf_sigmas = np.linspace(0.5, 4.0, n_depths)
        else:
            self.psf_sigmas = np.asarray(psf_sigmas, dtype=np.float64)

        # Per-depth weights (uniform by default)
        if depth_weights is None:
            self.depth_weights = np.ones(n_depths, dtype=np.float64) / n_depths
        else:
            self.depth_weights = np.asarray(depth_weights, dtype=np.float64)

        # Build frequency grids (matching FFT convention: 0..N-1 mapped by fftfreq)
        fy = np.fft.fftfreq(ny).astype(np.float64)  # (ny,)
        fx = np.fft.fftfreq(nx).astype(np.float64)  # (nx,)
        FY, FX = np.meshgrid(fy, fx, indexing="ij")  # (ny, nx)
        freq_sq = FY ** 2 + FX ** 2  # (ny, nx)

        # Precompute per-depth OTFs: OTF_d = w_d * exp(-2*pi^2*sigma_d^2*(fy^2+fx^2))
        # Shape: (n_depths, ny, nx)
        self._otfs = np.zeros((n_depths, ny, nx), dtype=np.float64)
        for d in range(n_depths):
            sigma_d = self.psf_sigmas[d]
            w_d = self.depth_weights[d]
            self._otfs[d] = w_d * np.exp(
                -2.0 * np.pi ** 2 * sigma_d ** 2 * freq_sq
            )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute 2D integral image from multi-depth volume.

        Args:
            x: Multi-depth volume (ny, nx, n_depths).

        Returns:
            Integral image (ny, nx).
        """
        x64 = x.astype(np.float64)
        y = np.zeros((self.ny, self.nx), dtype=np.float64)

        for d in range(self.n_depths):
            X_d = np.fft.fft2(x64[:, :, d])  # (ny, nx) complex
            y += np.real(np.fft.ifft2(self._otfs[d] * X_d))

        return y.astype(np.float32)

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        """Back-project 2D image to multi-depth volume.

        Args:
            y: 2D image (ny, nx).

        Returns:
            Multi-depth volume (ny, nx, n_depths).
        """
        y64 = y.astype(np.float64)
        Y = np.fft.fft2(y64)  # (ny, nx) complex
        x = np.zeros((self.ny, self.nx, self.n_depths), dtype=np.float64)

        for d in range(self.n_depths):
            # Adjoint of (OTF * FFT) is IFFT(conj(OTF) * .)
            # Since OTFs are real, conj(OTF) = OTF
            x[:, :, d] = np.real(np.fft.ifft2(np.conj(self._otfs[d]) * Y))

        return x.astype(np.float32)

    @property
    def x_shape(self) -> Tuple[int, ...]:
        return (self.ny, self.nx, self.n_depths)

    @property
    def y_shape(self) -> Tuple[int, ...]:
        return (self.ny, self.nx)

    @property
    def is_linear(self) -> bool:
        return True

    @property
    def supports_autodiff(self) -> bool:
        return False

    def info(self) -> Dict[str, Any]:
        return {
            "operator_id": self.operator_id,
            "ny": self.ny,
            "nx": self.nx,
            "n_depths": self.n_depths,
            "psf_sigmas": self.psf_sigmas.tolist(),
            "depth_weights": self.depth_weights.tolist(),
        }

    def metadata(self) -> OperatorMetadata:
        return OperatorMetadata(
            modality="integral",
            operator_id=self.operator_id,
            x_shape=list(self.x_shape),
            y_shape=list(self.y_shape),
            is_linear=True,
            supports_autodiff=False,
            axes={
                "x_dim0": "y_spatial",
                "x_dim1": "x_spatial",
                "x_dim2": "depth",
                "y_dim0": "y_spatial",
                "y_dim1": "x_spatial",
            },
            units={"spatial": "pixels", "depth": "depth_planes"},
        )
