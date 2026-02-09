"""Light Field microlens array integration operator.

Forward: 4D light field (sx, sy, nu, nv) -> 2D sensor image (sx, sy)
    Applies per-view disparity shift then averages across angular dims.
Adjoint: 2D sensor image -> 4D light field (back-projection with inverse shifts)

Uses FFT-based subpixel shifting for exact adjoint consistency.

References:
- Levoy, M. & Hanrahan, P. (1996). "Light Field Rendering", SIGGRAPH.
- Ng, R. (2005). "Fourier Slice Photography", SIGGRAPH.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from pwm_core.physics.base import BaseOperator, OperatorMetadata


def _fft_shift_2d(img: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Subpixel circular shift via Fourier-domain phase ramp.

    Adjoint-exact: shift(dy, dx)^T = shift(-dy, -dx).
    """
    H, W = img.shape
    fy = np.fft.fftfreq(H).reshape(-1, 1)
    fx = np.fft.fftfreq(W).reshape(1, -1)
    phase = np.exp(-2j * np.pi * (fy * dy + fx * dx))
    return np.real(np.fft.ifft2(np.fft.fft2(img) * phase))


class LightFieldOperator(BaseOperator):
    """Light field microlens array integration operator.

    Forward: x (sx, sy, nu, nv) -> y (sx, sy)
        For each angular view (u, v), shift x[:,:,u,v] by
        disparity * (u - cu, v - cv), then average all views.

    Adjoint: y (sx, sy) -> x (sx, sy, nu, nv)
        Replicate 2D image into each angular view with inverse shifts.
    """

    def __init__(
        self,
        operator_id: str = "light_field",
        theta: Optional[Dict[str, Any]] = None,
        sx: int = 64,
        sy: int = 64,
        nu: int = 5,
        nv: int = 5,
        disparity: float = 0.5,
    ):
        self.operator_id = operator_id
        self.theta = theta or {}
        self.sx = sx
        self.sy = sy
        self.nu = nu
        self.nv = nv
        self.disparity = disparity
        self._cu = nu // 2
        self._cv = nv // 2
        self._x_shape = (sx, sy, nu, nv)
        self._y_shape = (sx, sy)
        self._is_linear = True
        self._supports_autodiff = False

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Integrate 4D light field to 2D sensor image.

        Args:
            x: 4D light field (sx, sy, nu, nv).

        Returns:
            2D sensor image (sx, sy).
        """
        x64 = x.astype(np.float64)
        result = np.zeros((self.sx, self.sy), dtype=np.float64)

        for u in range(self.nu):
            for v in range(self.nv):
                du = (u - self._cu) * self.disparity
                dv = (v - self._cv) * self.disparity
                result += _fft_shift_2d(x64[:, :, u, v], du, dv)

        result /= (self.nu * self.nv)
        return result.astype(np.float32)

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        """Back-project 2D image into 4D light field.

        Args:
            y: 2D sensor image (sx, sy).

        Returns:
            4D light field (sx, sy, nu, nv).
        """
        y64 = y.astype(np.float64)
        x = np.zeros((self.sx, self.sy, self.nu, self.nv), dtype=np.float64)

        for u in range(self.nu):
            for v in range(self.nv):
                du = (u - self._cu) * self.disparity
                dv = (v - self._cv) * self.disparity
                x[:, :, u, v] = _fft_shift_2d(y64, -du, -dv)

        x /= (self.nu * self.nv)
        return x.astype(np.float32)

    @property
    def x_shape(self) -> Tuple[int, ...]:
        return (self.sx, self.sy, self.nu, self.nv)

    @property
    def y_shape(self) -> Tuple[int, ...]:
        return (self.sx, self.sy)

    @property
    def is_linear(self) -> bool:
        return True

    @property
    def supports_autodiff(self) -> bool:
        return False

    def info(self) -> Dict[str, Any]:
        return {
            "operator_id": self.operator_id,
            "sx": self.sx,
            "sy": self.sy,
            "nu": self.nu,
            "nv": self.nv,
            "disparity": self.disparity,
        }

    def metadata(self) -> OperatorMetadata:
        return OperatorMetadata(
            modality="light_field",
            operator_id=self.operator_id,
            x_shape=list(self.x_shape),
            y_shape=list(self.y_shape),
            is_linear=True,
            supports_autodiff=False,
            axes={"x_dim0": "spatial_y", "x_dim1": "spatial_x",
                  "x_dim2": "angular_u", "x_dim3": "angular_v",
                  "y_dim0": "spatial_y", "y_dim1": "spatial_x"},
            units={"spatial": "pixels", "angular": "subaperture_index"},
        )
