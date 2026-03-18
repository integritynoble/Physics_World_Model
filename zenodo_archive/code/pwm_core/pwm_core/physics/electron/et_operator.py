"""Electron Tomography (ET) operator.

Forward model: tilt-series projection (rotate 3D volume + sum along depth).
Linear operator analogous to CT but with limited tilt angles.

Forward: 3D volume -> stack of 2D projections at different tilt angles
Adjoint: back-projection (smear projections back through volume)

References:
- Frank, J. (2006). "Electron Tomography", Springer.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import ndimage

from pwm_core.physics.base import BaseOperator, OperatorMetadata


class ETOperator(BaseOperator):
    """Electron Tomography operator (tilt-series projection).

    Forward: x (D, H, W) -> y (n_tilts, H, W)
        Each projection: rotate volume by tilt angle, sum along depth axis.

    Adjoint: y (n_tilts, H, W) -> x (D, H, W)
        Back-project: smear each projection along depth, rotate back.
    """

    def __init__(
        self,
        operator_id: str = "electron_tomography",
        theta: Optional[Dict[str, Any]] = None,
        D: int = 32,
        H: int = 64,
        W: int = 64,
        n_tilts: int = 16,
        tilt_range: Tuple[float, float] = (-60.0, 60.0),
    ):
        self.operator_id = operator_id
        self.theta = theta or {}
        self.D = D
        self.H = H
        self.W = W
        self.n_tilts = n_tilts

        self._x_shape = (D, H, W)
        self._y_shape = (n_tilts, H, W)
        self._is_linear = True
        self._supports_autodiff = False

        # Tilt angles in degrees
        self._tilt_angles = np.linspace(
            tilt_range[0], tilt_range[1], n_tilts, endpoint=True
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Project 3D volume at all tilt angles.

        Args:
            x: 3D volume (D, H, W).

        Returns:
            Tilt-series projections (n_tilts, H, W).
        """
        x64 = np.asarray(x, dtype=np.float64)
        y = np.zeros(self._y_shape, dtype=np.float64)

        for i, angle in enumerate(self._tilt_angles):
            rotated = ndimage.rotate(
                x64, angle, axes=(0, 2), reshape=False, order=1, mode="constant"
            )
            y[i] = rotated.sum(axis=0)

        return y.astype(np.float32)

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        """Back-project tilt-series to 3D volume.

        Args:
            y: Tilt-series projections (n_tilts, H, W).

        Returns:
            Back-projected volume (D, H, W).
        """
        y64 = np.asarray(y, dtype=np.float64)
        x = np.zeros(self._x_shape, dtype=np.float64)

        for i, angle in enumerate(self._tilt_angles):
            # Smear projection along depth
            smeared = np.stack([y64[i]] * self.D, axis=0)  # (D, H, W)
            # Rotate back by negative angle
            back_rotated = ndimage.rotate(
                smeared, -angle, axes=(0, 2), reshape=False, order=1, mode="constant"
            )
            x += back_rotated

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
            "D": self.D,
            "H": self.H,
            "W": self.W,
            "n_tilts": self.n_tilts,
            "tilt_angles": self._tilt_angles.tolist(),
        }

    def metadata(self) -> OperatorMetadata:
        return OperatorMetadata(
            modality="electron_tomography",
            operator_id=self.operator_id,
            x_shape=list(self.x_shape),
            y_shape=list(self.y_shape),
            is_linear=True,
            supports_autodiff=False,
            axes={
                "x_dim0": "depth",
                "x_dim1": "height",
                "x_dim2": "width",
                "y_dim0": "tilt",
                "y_dim1": "height",
                "y_dim2": "width",
            },
            units={
                "tilt": "degrees",
                "density": "a.u.",
            },
        )
