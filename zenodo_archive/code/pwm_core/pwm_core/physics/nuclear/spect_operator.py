"""SPECT (Single Photon Emission Computed Tomography) operator.

Forward model: collimator PSF blur + rotation-based Radon projection.
Linear operator mapping emission activity map to sinogram.

Forward: emission map -> sinogram (blurred angular projections)
Adjoint: back-projection with collimator PSF

References:
- Cherry, S.R. et al. (2012). "Physics in Nuclear Medicine", Elsevier.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import ndimage

from pwm_core.physics.base import BaseOperator, OperatorMetadata


class SPECTOperator(BaseOperator):
    """SPECT imaging operator (collimator + Radon projection).

    Forward: x (ny, nx) -> y (n_angles, n_detectors)
        y = Radon(blur(x)), where blur is the collimator response.

    Adjoint: y (n_angles, n_detectors) -> x (ny, nx)
        Back-project then apply collimator blur (self-adjoint Gaussian).
    """

    def __init__(
        self,
        operator_id: str = "spect",
        theta: Optional[Dict[str, Any]] = None,
        ny: int = 64,
        nx: int = 64,
        n_angles: int = 32,
        n_detectors: Optional[int] = None,
        collimator_sigma: float = 2.0,
    ):
        self.operator_id = operator_id
        self.theta = theta or {}
        self.ny = ny
        self.nx = nx
        self.n_angles = n_angles
        self.n_detectors = n_detectors if n_detectors is not None else nx
        self.collimator_sigma = collimator_sigma

        self._x_shape = (ny, nx)
        self._y_shape = (n_angles, self.n_detectors)
        self._is_linear = True
        self._supports_autodiff = False

        self._angles = np.linspace(0.0, 180.0, n_angles, endpoint=False)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward project with collimator blur.

        Args:
            x: Emission activity map (ny, nx).

        Returns:
            Sinogram (n_angles, n_detectors).
        """
        x64 = np.asarray(x, dtype=np.float64)
        # Apply collimator PSF
        blurred = ndimage.gaussian_filter(x64, sigma=self.collimator_sigma)

        sinogram = np.zeros(self._y_shape, dtype=np.float64)
        for i, angle in enumerate(self._angles):
            rotated = ndimage.rotate(
                blurred, angle, reshape=False, order=1, mode="constant"
            )
            projection = np.sum(rotated, axis=0)
            if len(projection) != self.n_detectors:
                indices = np.linspace(0, len(projection) - 1, self.n_detectors)
                sinogram[i] = np.interp(
                    indices, np.arange(len(projection)), projection
                )
            else:
                sinogram[i] = projection

        return sinogram.astype(np.float32)

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        """Back-project sinogram with collimator blur.

        Args:
            y: Sinogram (n_angles, n_detectors).

        Returns:
            Back-projected image (ny, nx).
        """
        y64 = np.asarray(y, dtype=np.float64)
        x = np.zeros(self._x_shape, dtype=np.float64)

        for i, angle in enumerate(self._angles):
            proj = y64[i]
            if len(proj) != self.nx:
                indices = np.linspace(0, len(proj) - 1, self.nx)
                proj = np.interp(indices, np.arange(len(proj)), proj)
            smeared = np.tile(proj, (self.ny, 1))
            back_rotated = ndimage.rotate(
                smeared, -angle, reshape=False, order=1, mode="constant"
            )
            x += back_rotated

        # Apply collimator PSF (self-adjoint for symmetric Gaussian)
        x = ndimage.gaussian_filter(x, sigma=self.collimator_sigma)
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
            "n_angles": self.n_angles,
            "n_detectors": self.n_detectors,
            "collimator_sigma": self.collimator_sigma,
        }

    def metadata(self) -> OperatorMetadata:
        return OperatorMetadata(
            modality="spect",
            operator_id=self.operator_id,
            x_shape=list(self.x_shape),
            y_shape=list(self.y_shape),
            is_linear=True,
            supports_autodiff=False,
            axes={
                "x_dim0": "y_spatial",
                "x_dim1": "x_spatial",
                "y_dim0": "angle",
                "y_dim1": "detector",
            },
            units={
                "activity": "Bq/voxel",
                "sinogram": "counts",
                "collimator_sigma": "pixels",
            },
        )
