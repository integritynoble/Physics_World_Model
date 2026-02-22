"""X-ray Radiography operator.

Forward model: linear Beer-Lambert attenuation (log-domain).
In the linearised model, the measurement is the line integral of the
attenuation coefficient, making this a linear operator.

Forward: attenuation map -> log-attenuation projection
Adjoint: transpose (identity-like since projection is planar)

References:
- Bushberg, J. et al. (2011). "The Essential Physics of Medical Imaging",
  Lippincott Williams & Wilkins.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import ndimage

from pwm_core.physics.base import BaseOperator, OperatorMetadata


class XRayRadiographyOperator(BaseOperator):
    """X-ray radiography operator (linear Beer-Lambert).

    Works in log-attenuation domain where the forward model is linear:
    Forward: x (ny, nx) -> y (ny, nx)
        y = mu * blur(x)  (log-attenuation with optional PSF)

    Adjoint: y (ny, nx) -> x (ny, nx)
        x = mu * blur^T(y)  (Gaussian blur is self-adjoint)
    """

    def __init__(
        self,
        operator_id: str = "xray_radiography",
        theta: Optional[Dict[str, Any]] = None,
        ny: int = 64,
        nx: int = 64,
        mu: float = 1.0,
        psf_sigma: float = 0.5,
    ):
        self.operator_id = operator_id
        self.theta = theta or {}
        self.ny = ny
        self.nx = nx
        self.mu = mu
        self.psf_sigma = psf_sigma

        self._x_shape = (ny, nx)
        self._y_shape = (ny, nx)
        self._is_linear = True
        self._supports_autodiff = False

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute log-attenuation radiograph.

        Args:
            x: Attenuation/thickness map (ny, nx).

        Returns:
            Log-attenuation image (ny, nx).
        """
        x64 = np.asarray(x, dtype=np.float64)
        blurred = ndimage.gaussian_filter(x64, sigma=self.psf_sigma, mode="reflect")
        y = self.mu * blurred
        return y.astype(np.float32)

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        """Adjoint of log-attenuation model.

        Args:
            y: Log-attenuation image (ny, nx).

        Returns:
            Back-projected attenuation map (ny, nx).
        """
        y64 = np.asarray(y, dtype=np.float64)
        blurred = ndimage.gaussian_filter(y64, sigma=self.psf_sigma, mode="reflect")
        x = self.mu * blurred
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
            "mu": self.mu,
            "psf_sigma": self.psf_sigma,
        }

    def metadata(self) -> OperatorMetadata:
        return OperatorMetadata(
            modality="xray_radiography",
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
                "mu": "1/pixel",
                "attenuation": "a.u.",
                "signal": "log-attenuation",
            },
        )
