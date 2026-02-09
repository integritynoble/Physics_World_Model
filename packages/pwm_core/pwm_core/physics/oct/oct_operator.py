"""OCT (Optical Coherence Tomography) operator.

Balanced-detection spectral interferometry forward model.
Forward: depth-reflectivity -> spectral interferogram via DFT
Adjoint: transpose of forward DFT matrix

References:
- Fercher, A.F. et al. (2003). "Optical coherence tomography - principles
  and applications", Reports on Progress in Physics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pwm_core.physics.base import BaseOperator, OperatorMetadata


class OCTOperator(BaseOperator):
    """OCT balanced-detection spectral interferometry operator.

    Forward: x (n_alines, n_depth) -> y (n_alines, n_spectral)
        y = 2 * Re(x @ exp(2j*k*z).T)

    Adjoint: y (n_alines, n_spectral) -> x (n_alines, n_depth)
        x = Re(y @ exp(-2j*k*z)) * (2/n_spectral)
    """

    def __init__(
        self,
        operator_id: str = "oct",
        theta: Optional[Dict[str, Any]] = None,
        n_alines: int = 128,
        n_depth: int = 256,
        n_spectral: int = 512,
        dispersion_coeffs: Optional[List[float]] = None,
    ):
        self.operator_id = operator_id
        self.theta = theta or {}
        self.n_alines = n_alines
        self.n_depth = n_depth
        self.n_spectral = n_spectral
        self.dispersion_coeffs = dispersion_coeffs
        self._x_shape = (n_alines, n_depth)
        self._y_shape = (n_alines, n_spectral)
        self._is_linear = True
        self._supports_autodiff = False

        # Build wavenumber and depth arrays (matching benchmark convention)
        self._k = np.arange(n_spectral, dtype=np.float64) * np.pi / n_spectral
        self._z = np.arange(n_depth, dtype=np.float64)

        # Precompute the DFT kernel: exp(2j * k * z) -> (n_spectral, n_depth)
        self._exp_fwd = np.exp(2j * np.outer(self._k, self._z))

        # Apply dispersion compensation if coefficients provided
        if dispersion_coeffs is not None:
            phase_correction = np.zeros(n_spectral, dtype=np.float64)
            for order, coeff in enumerate(dispersion_coeffs):
                phase_correction += coeff * self._k ** (order + 2)
            self._exp_fwd *= np.exp(-1j * phase_correction)[:, np.newaxis]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute spectral interferogram from depth-reflectivity.

        Args:
            x: Depth-reflectivity (n_alines, n_depth).

        Returns:
            Spectral interferogram (n_alines, n_spectral).
        """
        x64 = x.astype(np.float64)
        # y = 2 * Re(x @ exp(2j*k*z).T)
        y = 2.0 * np.real(x64 @ self._exp_fwd.T)
        return y.astype(np.float32)

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        """Back-project spectral interferogram to depth-reflectivity.

        Args:
            y: Spectral interferogram (n_alines, n_spectral).

        Returns:
            Depth-reflectivity (n_alines, n_depth).
        """
        y64 = y.astype(np.float64)
        # Adjoint of forward: x = Re(y @ conj(exp_fwd)) * (2/n_spectral)
        # Since forward is y = 2*Re(x @ exp_fwd.T),
        # the adjoint is x = 2*Re(y @ conj(exp_fwd.T).T) / ... but we need
        # the exact transpose. For real x and real y:
        # <Ax, y> = <2*Re(x @ E^T), y> = 2*Re(tr(x @ E^T @ y^T))
        #         = 2*Re(tr(E^T @ y^T @ x)) = <x, 2*Re(y @ conj(E^T)^T)>
        #         = <x, 2*Re(y @ conj(E))>
        # So A^T y = 2 * Re(y @ conj(E))
        x = 2.0 * np.real(y64 @ np.conj(self._exp_fwd))
        return x.astype(np.float32)

    @property
    def x_shape(self) -> Tuple[int, ...]:
        return (self.n_alines, self.n_depth)

    @property
    def y_shape(self) -> Tuple[int, ...]:
        return (self.n_alines, self.n_spectral)

    @property
    def is_linear(self) -> bool:
        return True

    @property
    def supports_autodiff(self) -> bool:
        return False

    def info(self) -> Dict[str, Any]:
        return {
            "operator_id": self.operator_id,
            "n_alines": self.n_alines,
            "n_depth": self.n_depth,
            "n_spectral": self.n_spectral,
            "dispersion_coeffs": self.dispersion_coeffs,
        }

    def metadata(self) -> OperatorMetadata:
        return OperatorMetadata(
            modality="oct",
            operator_id=self.operator_id,
            x_shape=list(self.x_shape),
            y_shape=list(self.y_shape),
            is_linear=True,
            supports_autodiff=False,
            axes={"x_dim0": "alines", "x_dim1": "depth",
                  "y_dim0": "alines", "y_dim1": "spectral"},
            wavelength_range_nm=[800.0, 900.0],
            units={"depth": "pixels", "spectral": "wavenumber_bins"},
        )
