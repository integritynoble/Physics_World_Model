"""SEM (Scanning Electron Microscopy) operator.

Forward model: secondary electron yield scaling + PSF blur.
The SE yield is approximately linear in material density for a fixed voltage,
making this a linear operator.

Forward: material density map -> SE image (blurred yield)
Adjoint: transpose of the blur + scale operation

References:
- Goldstein, J. et al. (2017). "Scanning Electron Microscopy and X-Ray
  Microanalysis", Springer.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import ndimage

from pwm_core.physics.base import BaseOperator, OperatorMetadata


class SEMOperator(BaseOperator):
    """SEM imaging operator (SE yield + PSF blur).

    Forward: x (ny, nx) -> y (ny, nx)
        y = blur(se_scale * x)
        se_scale = 1 / sqrt(voltage_kv)

    Adjoint: y (ny, nx) -> x (ny, nx)
        x = se_scale * blur^T(y)
        (Gaussian blur is self-adjoint, so blur^T = blur)
    """

    def __init__(
        self,
        operator_id: str = "sem",
        theta: Optional[Dict[str, Any]] = None,
        ny: int = 64,
        nx: int = 64,
        voltage_kv: float = 15.0,
        psf_sigma: float = 1.0,
    ):
        self.operator_id = operator_id
        self.theta = theta or {}
        self.ny = ny
        self.nx = nx
        self.voltage_kv = voltage_kv
        self.psf_sigma = psf_sigma

        self._x_shape = (ny, nx)
        self._y_shape = (ny, nx)
        self._is_linear = True
        self._supports_autodiff = False

        # SE yield scale factor
        self._se_scale = 1.0 / max(np.sqrt(voltage_kv), 1e-12)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute SEM SE image from material density map.

        Args:
            x: Material density map (ny, nx).

        Returns:
            SE image (ny, nx).
        """
        x64 = np.asarray(x, dtype=np.float64)
        scaled = self._se_scale * x64
        blurred = ndimage.gaussian_filter(scaled, sigma=self.psf_sigma, mode="reflect")
        return blurred.astype(np.float32)

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        """Adjoint: blur is self-adjoint for symmetric Gaussian kernel.

        Args:
            y: SE image (ny, nx).

        Returns:
            Back-projected material map (ny, nx).
        """
        y64 = np.asarray(y, dtype=np.float64)
        blurred = ndimage.gaussian_filter(y64, sigma=self.psf_sigma, mode="reflect")
        scaled = self._se_scale * blurred
        return scaled.astype(np.float32)

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
            "voltage_kv": self.voltage_kv,
            "psf_sigma": self.psf_sigma,
        }

    def metadata(self) -> OperatorMetadata:
        return OperatorMetadata(
            modality="sem",
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
                "voltage": "kV",
                "density": "a.u.",
                "signal": "SE counts",
            },
        )
