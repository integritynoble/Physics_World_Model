"""Cryo-EM (Cryo-Electron Microscopy) operator.

Single-particle cryo-EM forward model: CTF filtering with B-factor
envelope and ice thickness attenuation.

Forward: projected potential -> cryo-EM micrograph
    y = D * IFFT{ CTF(f) * E(f;B) * FFT{x} }

Adjoint: same multiplication (CTF and E are real → self-adjoint in Fourier domain)

Mismatch ThetaSpace:
    defocus_nm: [-2000, 2000]
    Cs_mm: [0.5, 2.5]
    B_factor: [0, 200]
    ice_thickness_nm: [20, 100]

References:
- Frank, J. (2006). "Three-Dimensional Electron Microscopy of Macromolecular
  Assemblies", Oxford University Press.
- Penczek, P.A. (2010). "Fundamentals of Three-Dimensional Reconstruction
  from Projections", Methods in Enzymology.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from pwm_core.physics.base import BaseOperator, OperatorMetadata


class CryoEMOperator(BaseOperator):
    """Cryo-EM single-particle imaging operator.

    Forward: x (ny, nx) -> y (ny, nx)
        Y(f) = CTF(f) * E(f;B) * A_ice * X(f)

    Adjoint: y (ny, nx) -> x (ny, nx)
        Same multiplication (CTF*E are real, so self-adjoint in Fourier domain)

    Primitives: C (CTF convolution), M (B-factor envelope, ice attenuation), D (electron detection)
    """

    def __init__(
        self,
        operator_id: str = "cryo_em",
        theta: Optional[Dict[str, Any]] = None,
        ny: int = 64,
        nx: int = 64,
        defocus_nm: float = -500.0,
        Cs_mm: float = 2.0,
        wavelength_pm: float = 2.51,
        B_factor: float = 50.0,
        ice_thickness_nm: float = 50.0,
        pixel_size_nm: float = 1.0,
    ):
        self.operator_id = operator_id
        self.theta = theta or {}
        self.ny = ny
        self.nx = nx
        self.defocus_nm = defocus_nm
        self.Cs_mm = Cs_mm
        self.wavelength_pm = wavelength_pm
        self.B_factor = B_factor
        self.ice_thickness_nm = ice_thickness_nm
        self.pixel_size_nm = pixel_size_nm

        self._x_shape = (ny, nx)
        self._y_shape = (ny, nx)
        self._is_linear = True
        self._supports_autodiff = False

        self._precompute()

    def _precompute(self) -> None:
        """Precompute CTF, B-factor envelope, and ice attenuation."""
        ny, nx = self.ny, self.nx

        # Spatial frequency grid (1/nm)
        freqs_y = np.fft.fftfreq(ny, d=self.pixel_size_nm)
        freqs_x = np.fft.fftfreq(nx, d=self.pixel_size_nm)
        FY, FX = np.meshgrid(freqs_y, freqs_x, indexing="ij")
        q2 = FX ** 2 + FY ** 2  # |f|^2 in (1/nm)^2

        # CTF: sin(pi*lambda*Df*|f|^2 - 0.5*pi*Cs*lambda^3*|f|^4)
        wl_nm = self.wavelength_pm * 1e-3
        Cs_nm = self.Cs_mm * 1e6

        chi = (
            np.pi * wl_nm * self.defocus_nm * q2
            - 0.5 * np.pi * Cs_nm * wl_nm ** 3 * q2 ** 2
        )
        self._ctf = np.sin(chi).astype(np.float64)

        # B-factor envelope: exp(-B*|f|^2/4)
        self._envelope = np.exp(-self.B_factor * q2 / 4.0).astype(np.float64)

        # Ice thickness attenuation (exponential decay)
        # Mean free path ~300-400 nm for 300 keV electrons in vitreous ice
        mean_free_path_nm = 350.0
        self._ice_atten = float(np.exp(-self.ice_thickness_nm / mean_free_path_nm))

        # Combined transfer function (all real-valued)
        self._transfer = self._ctf * self._envelope * self._ice_atten

    def set_theta(self, **kwargs: Any) -> None:
        """Update mismatch parameters and recompute."""
        changed = False
        for key in ("defocus_nm", "Cs_mm", "B_factor", "ice_thickness_nm"):
            if key in kwargs:
                setattr(self, key, float(kwargs[key]))
                changed = True
        if changed:
            self._precompute()

    def get_theta(self) -> Dict[str, float]:
        """Return current mismatch parameters."""
        return {
            "defocus_nm": self.defocus_nm,
            "Cs_mm": self.Cs_mm,
            "B_factor": self.B_factor,
            "ice_thickness_nm": self.ice_thickness_nm,
        }

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply cryo-EM forward model (CTF + B-factor + ice).

        Args:
            x: Projected potential (ny, nx).

        Returns:
            Cryo-EM micrograph (ny, nx).
        """
        x64 = np.asarray(x, dtype=np.float64)
        X_f = np.fft.fft2(x64)
        Y_f = self._transfer * X_f
        y = np.real(np.fft.ifft2(Y_f))
        return y.astype(np.float32)

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        """Adjoint: same multiplication (transfer function is real).

        Args:
            y: Cryo-EM micrograph (ny, nx).

        Returns:
            Back-projected potential (ny, nx).
        """
        y64 = np.asarray(y, dtype=np.float64)
        Y_f = np.fft.fft2(y64)
        X_f = self._transfer * Y_f
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
            "B_factor": self.B_factor,
            "ice_thickness_nm": self.ice_thickness_nm,
        }

    def metadata(self) -> OperatorMetadata:
        return OperatorMetadata(
            modality="cryo_em",
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
                "B_factor": "A^2",
                "ice_thickness": "nm",
                "wavelength": "pm",
            },
        )
