"""TEM (Transmission Electron Microscopy) operator.

Forward model: CTF filtering in Fourier space.
The projected potential is convolved with the Contrast Transfer Function (CTF).

Forward: projected potential -> TEM image (CTF-filtered)
Adjoint: multiplication by same CTF in Fourier space (CTF is real and symmetric)

References:
- Williams, D.B. & Carter, C.B. (2009). "Transmission Electron Microscopy",
  Springer.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from pwm_core.physics.base import BaseOperator, OperatorMetadata


class TEMOperator(BaseOperator):
    """TEM imaging operator (CTF filtering).

    Forward: x (ny, nx) -> y (ny, nx)
        Y(f) = CTF(f) * X(f)  (multiplication in Fourier space)

    Adjoint: y (ny, nx) -> x (ny, nx)
        X(f) = CTF(f) * Y(f)  (CTF is real, so adjoint = same multiplication)
    """

    def __init__(
        self,
        operator_id: str = "tem",
        theta: Optional[Dict[str, Any]] = None,
        ny: int = 64,
        nx: int = 64,
        defocus_nm: float = -50.0,
        Cs_mm: float = 1.0,
        wavelength_pm: float = 2.51,
    ):
        self.operator_id = operator_id
        self.theta = theta or {}
        self.ny = ny
        self.nx = nx
        self.defocus_nm = defocus_nm
        self.Cs_mm = Cs_mm
        self.wavelength_pm = wavelength_pm

        self._x_shape = (ny, nx)
        self._y_shape = (ny, nx)
        self._is_linear = True
        self._supports_autodiff = False

        # Precompute CTF
        freqs_y = np.fft.fftfreq(ny)
        freqs_x = np.fft.fftfreq(nx)
        FY, FX = np.meshgrid(freqs_y, freqs_x, indexing="ij")
        q2 = FX ** 2 + FY ** 2

        wl_nm = wavelength_pm * 1e-3
        Cs_nm = Cs_mm * 1e6

        chi = (
            np.pi * wl_nm * defocus_nm * q2
            - 0.5 * np.pi * Cs_nm * wl_nm ** 3 * q2 ** 2
        )
        self._ctf = np.sin(chi).astype(np.float64)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply CTF filtering to projected potential.

        Args:
            x: Projected potential (ny, nx).

        Returns:
            TEM image (ny, nx).
        """
        x64 = np.asarray(x, dtype=np.float64)
        X_f = np.fft.fft2(x64)
        Y_f = self._ctf * X_f
        y = np.real(np.fft.ifft2(Y_f))
        return y.astype(np.float32)

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        """Adjoint of CTF filtering (same as forward since CTF is real).

        Args:
            y: TEM image (ny, nx).

        Returns:
            Back-projected potential (ny, nx).
        """
        y64 = np.asarray(y, dtype=np.float64)
        Y_f = np.fft.fft2(y64)
        X_f = self._ctf * Y_f
        x = np.real(np.fft.ifft2(X_f))
        return x.astype(np.float32)

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
            "defocus_nm": self.defocus_nm,
            "Cs_mm": self.Cs_mm,
            "wavelength_pm": self.wavelength_pm,
        }

    def metadata(self) -> OperatorMetadata:
        return OperatorMetadata(
            modality="tem",
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
                "defocus": "nm",
                "Cs": "mm",
                "wavelength": "pm",
                "potential": "a.u.",
            },
        )
