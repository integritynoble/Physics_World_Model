"""Compressive Holography operator.

Multi-depth Fresnel propagation — 3D object encoded into single 2D hologram.
Compressive: (K, ny, nx) -> (ny, nx).

Forward: y_linear = sum_k Real{ conj(R) * P(z_k) x_k }
Adjoint: x_hat_k = Real{ R * P(-z_k) y } per depth k

Mismatch ThetaSpace:
    propagation_distance_error: [-100, 100] um
    carrier_freq_error: [-0.05, 0.05]
    wavelength_error: [-5, 5] nm

References:
- Brady, D.J. et al. (2009). "Compressive Holography", Optics Express.
- Denis, L. et al. (2009). "Inline hologram reconstruction with sparsity
  constraints", Optics Letters.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pwm_core.physics.base import BaseOperator, OperatorMetadata


class CompressiveHolographyOperator(BaseOperator):
    """Compressive holography operator (multi-depth Fresnel → single hologram).

    Forward: x (K, ny, nx) -> y (ny, nx)
        y = sum_k Real{ conj(R) * P(z_k) x_k }

    Adjoint: y (ny, nx) -> x (K, ny, nx)
        x_hat_k = Real{ R * P(-z_k) y }

    Primitives: P (Fresnel propagation), M (reference interference),
                Sigma (depth accumulation), D (intensity detection)
    """

    def __init__(
        self,
        operator_id: str = "compressive_holography",
        theta: Optional[Dict[str, Any]] = None,
        ny: int = 64,
        nx: int = 64,
        n_depths: int = 4,
        depth_spacing_um: float = 100.0,
        wavelength_nm: float = 532.0,
        pixel_size_um: float = 5.0,
        carrier_freq: float = 0.15,
        prop_distance_error_um: float = 0.0,
        carrier_freq_error: float = 0.0,
        wavelength_error_nm: float = 0.0,
    ):
        self.operator_id = operator_id
        self.theta = theta or {}
        self.ny = ny
        self.nx = nx
        self.n_depths = n_depths
        self.depth_spacing_um = depth_spacing_um
        self.wavelength_nm = wavelength_nm
        self.pixel_size_um = pixel_size_um
        self.carrier_freq = carrier_freq
        self.prop_distance_error_um = prop_distance_error_um
        self.carrier_freq_error = carrier_freq_error
        self.wavelength_error_nm = wavelength_error_nm

        self._x_shape = (n_depths, ny, nx)
        self._y_shape = (ny, nx)
        self._is_linear = True
        self._supports_autodiff = False

        self._precompute()

    def _precompute(self) -> None:
        """Precompute Fresnel kernels and reference wave."""
        ny, nx = self.ny, self.nx

        # Effective wavelength with error
        wl_um = (self.wavelength_nm + self.wavelength_error_nm) * 1e-3  # nm -> um

        # Spatial frequency grid
        fy = np.fft.fftfreq(ny, d=self.pixel_size_um)
        fx = np.fft.fftfreq(nx, d=self.pixel_size_um)
        FY, FX = np.meshgrid(fy, fx, indexing="ij")
        f2 = FX ** 2 + FY ** 2

        # Fresnel propagation kernels for each depth plane
        self._kernels_fwd: List[np.ndarray] = []
        self._kernels_adj: List[np.ndarray] = []
        for k in range(self.n_depths):
            z_k = (k + 1) * self.depth_spacing_um + self.prop_distance_error_um
            # H(f;z) = exp(j*pi*lambda*z*(fx^2+fy^2))
            kernel = np.exp(1j * np.pi * wl_um * z_k * f2)
            self._kernels_fwd.append(kernel.astype(np.complex128))
            self._kernels_adj.append(np.conj(kernel).astype(np.complex128))

        # Off-axis reference wave
        yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        eff_carrier = self.carrier_freq + self.carrier_freq_error
        self._reference = np.exp(
            1j * 2 * np.pi * eff_carrier * (xx + yy) / max(ny, nx)
        ).astype(np.complex128)
        self._reference_conj = np.conj(self._reference)

    def set_theta(self, **kwargs: Any) -> None:
        """Update mismatch parameters and recompute."""
        changed = False
        for key in ("prop_distance_error_um", "carrier_freq_error", "wavelength_error_nm"):
            if key in kwargs:
                setattr(self, key, float(kwargs[key]))
                changed = True
        if changed:
            self._precompute()

    def get_theta(self) -> Dict[str, float]:
        """Return current mismatch parameters."""
        return {
            "prop_distance_error_um": self.prop_distance_error_um,
            "carrier_freq_error": self.carrier_freq_error,
            "wavelength_error_nm": self.wavelength_error_nm,
        }

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Multi-depth Fresnel propagation to single hologram.

        Args:
            x: Multi-depth object (n_depths, ny, nx).

        Returns:
            Hologram (ny, nx).
        """
        x64 = np.asarray(x, dtype=np.float64)
        hologram = np.zeros((self.ny, self.nx), dtype=np.float64)

        for k in range(self.n_depths):
            # Fresnel propagation of depth plane k
            X_f = np.fft.fft2(x64[k])
            propagated = np.fft.ifft2(self._kernels_fwd[k] * X_f)
            # Interference with reference: Real{ conj(R) * propagated }
            hologram += np.real(self._reference_conj * propagated)

        return hologram.astype(np.float32)

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        """Back-propagate hologram to multi-depth object.

        Args:
            y: Hologram (ny, nx).

        Returns:
            Multi-depth estimate (n_depths, ny, nx).
        """
        y64 = np.asarray(y, dtype=np.float64)
        x_hat = np.zeros(self._x_shape, dtype=np.float64)

        for k in range(self.n_depths):
            # Multiply by reference, then back-propagate
            modulated = self._reference * y64
            Y_f = np.fft.fft2(modulated)
            back_prop = np.fft.ifft2(self._kernels_adj[k] * Y_f)
            x_hat[k] = np.real(back_prop)

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
            "n_depths": self.n_depths,
            "depth_spacing_um": self.depth_spacing_um,
            "wavelength_nm": self.wavelength_nm,
            "carrier_freq": self.carrier_freq,
        }

    def metadata(self) -> OperatorMetadata:
        return OperatorMetadata(
            modality="compressive_holography",
            operator_id=self.operator_id,
            x_shape=list(self.x_shape),
            y_shape=list(self.y_shape),
            is_linear=True,
            supports_autodiff=False,
            axes={
                "x_dim0": "depth",
                "x_dim1": "y_spatial",
                "x_dim2": "x_spatial",
                "y_dim0": "y_spatial",
                "y_dim1": "x_spatial",
            },
            units={
                "depth_spacing": "um",
                "wavelength": "nm",
                "pixel_size": "um",
            },
        )
