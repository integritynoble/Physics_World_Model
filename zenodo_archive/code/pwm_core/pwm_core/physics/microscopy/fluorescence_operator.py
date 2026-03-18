"""Fluorescence Microscopy operator.

Dual-PSF model with Stokes shift: excitation and emission wavelengths
differ, producing different PSF widths.

Forward: y = h_em ** (eta * h_ex ** x) + b
Adjoint: x_hat = h_ex ** (eta * h_em ** (y - b))
  (both PSFs are symmetric Gaussians → self-adjoint)

Mismatch ThetaSpace:
    psf_sigma_ex: [0.5, 3.0] pixels
    psf_sigma_em: [0.8, 4.0] pixels
    quantum_yield: [0.1, 1.0]
    background: [0, 0.15]

References:
- Lichtman, J.W. & Conchello, J.A. (2005). "Fluorescence microscopy",
  Nature Methods.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from pwm_core.physics.base import BaseOperator, OperatorMetadata


class FluorescenceMicroscopyOperator(BaseOperator):
    """Fluorescence microscopy operator (dual-PSF Stokes shift model).

    Forward: x (ny, nx) -> y (ny, nx)
        y = G_em ** (eta * G_ex ** x) + b

    Adjoint: y (ny, nx) -> x (ny, nx)
        x_hat = G_ex ** (eta * G_em ** (y - b))

    Both Gaussian PSFs are symmetric → self-adjoint convolution.

    Primitives: C (excitation PSF), M (fluorophore response),
                C (emission PSF), D (sCMOS detection)
    """

    def __init__(
        self,
        operator_id: str = "fluorescence_microscopy",
        theta: Optional[Dict[str, Any]] = None,
        ny: int = 64,
        nx: int = 64,
        psf_sigma_ex: float = 1.5,
        psf_sigma_em: float = 2.0,
        quantum_yield: float = 0.7,
        background: float = 0.02,
    ):
        self.operator_id = operator_id
        self.theta = theta or {}
        self.ny = ny
        self.nx = nx
        self.psf_sigma_ex = psf_sigma_ex
        self.psf_sigma_em = psf_sigma_em
        self.quantum_yield = quantum_yield
        self.background = background

        self._x_shape = (ny, nx)
        self._y_shape = (ny, nx)
        self._is_linear = True
        self._supports_autodiff = False

    def set_theta(self, **kwargs: Any) -> None:
        """Update mismatch parameters."""
        for key in ("psf_sigma_ex", "psf_sigma_em", "quantum_yield", "background"):
            if key in kwargs:
                setattr(self, key, float(kwargs[key]))

    def get_theta(self) -> Dict[str, float]:
        """Return current mismatch parameters."""
        return {
            "psf_sigma_ex": self.psf_sigma_ex,
            "psf_sigma_em": self.psf_sigma_em,
            "quantum_yield": self.quantum_yield,
            "background": self.background,
        }

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply fluorescence microscopy forward model.

        Args:
            x: Fluorophore concentration (ny, nx).

        Returns:
            Fluorescence image (ny, nx).
        """
        x64 = np.asarray(x, dtype=np.float64)

        # Excitation PSF blur
        excited = gaussian_filter(x64, sigma=self.psf_sigma_ex, mode="reflect")

        # Fluorophore response (quantum yield)
        emitted = self.quantum_yield * excited

        # Emission PSF blur
        detected = gaussian_filter(emitted, sigma=self.psf_sigma_em, mode="reflect")

        # Add background
        y = detected + self.background

        return y.astype(np.float32)

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        """Adjoint: reverse PSF convolutions (symmetric → self-adjoint).

        Args:
            y: Fluorescence image (ny, nx).

        Returns:
            Estimated concentration (ny, nx).
        """
        y64 = np.asarray(y, dtype=np.float64)

        # Subtract background
        y_nobg = y64 - self.background

        # Emission PSF adjoint (= same convolution for symmetric Gaussian)
        em_adj = gaussian_filter(y_nobg, sigma=self.psf_sigma_em, mode="reflect")

        # Quantum yield scaling
        scaled = self.quantum_yield * em_adj

        # Excitation PSF adjoint
        x_hat = gaussian_filter(scaled, sigma=self.psf_sigma_ex, mode="reflect")

        return x_hat.astype(np.float32)

    @property
    def x_shape(self) -> Tuple[int, ...]:
        return self._x_shape

    @property
    def y_shape(self) -> Tuple[int, ...]:
        return self._y_shape

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
            "psf_sigma_ex": self.psf_sigma_ex,
            "psf_sigma_em": self.psf_sigma_em,
            "quantum_yield": self.quantum_yield,
            "background": self.background,
        }

    def metadata(self) -> OperatorMetadata:
        return OperatorMetadata(
            modality="fluorescence_microscopy",
            operator_id=self.operator_id,
            x_shape=list(self.x_shape),
            y_shape=list(self.y_shape),
            is_linear=True,
            supports_autodiff=False,
            axes={
                "x_dim0": "y_spatial",
                "x_dim1": "x_spatial",
                "y_dim0": "y_spatial",
                "y_dim1": "x_spatial",
            },
            units={
                "psf_sigma_ex": "pixels",
                "psf_sigma_em": "pixels",
                "quantum_yield": "unitless",
                "background": "normalized",
            },
        )
