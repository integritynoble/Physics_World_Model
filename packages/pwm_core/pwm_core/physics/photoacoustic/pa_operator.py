"""Photoacoustic (PA) operator.

Circular Radon transform forward model for photoacoustic imaging.
Forward: initial pressure distribution -> sinogram (time-of-flight)
Adjoint: back-projection of sinogram to image space

Transducers are placed uniformly on a circle surrounding the image.
The forward model bins pixel contributions by their integer time-of-flight
to each transducer.

References:
- Xu, M. & Wang, L.V. (2006). "Photoacoustic imaging in biomedicine",
  Review of Scientific Instruments.
- Kuchment, P. & Kunyansky, L. (2011). "Mathematics of photoacoustic and
  thermoacoustic tomography", European Journal of Applied Mathematics.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from pwm_core.physics.base import BaseOperator, OperatorMetadata


class PAOperator(BaseOperator):
    """Photoacoustic imaging operator (circular Radon transform).

    Forward: x (ny, nx) -> y (n_transducers, n_times)
        sinogram[i, t] = sum of x[pixels] where round(dist(pixel, transducer_i) / speed_of_sound) == t

    Adjoint: y (n_transducers, n_times) -> x (ny, nx)
        x[pixel] = sum over transducers of y[i, time_index(pixel, i)]
    """

    def __init__(
        self,
        operator_id: str = "photoacoustic",
        theta: Optional[Dict[str, Any]] = None,
        ny: int = 64,
        nx: int = 64,
        n_transducers: int = 32,
        n_times: Optional[int] = None,
        speed_of_sound: float = 1.0,
    ):
        self.operator_id = operator_id
        self.theta = theta or {}
        self.ny = ny
        self.nx = nx
        self.n_transducers = n_transducers
        self.speed_of_sound = speed_of_sound
        self.grid_shape = (ny, nx)

        # Default n_times: maximum possible distance across the image
        if n_times is None:
            n_times = int(np.ceil(np.sqrt(ny ** 2 + nx ** 2)))
        self.n_times = n_times

        self._x_shape = (ny, nx)
        self._y_shape = (n_transducers, n_times)
        self._is_linear = True
        self._supports_autodiff = False

        # Transducer positions on a circle of radius max(ny, nx) // 2
        # centered at the image center
        radius = max(ny, nx) // 2
        center_y = (ny - 1) / 2.0
        center_x = (nx - 1) / 2.0
        angles = np.linspace(0.0, 2.0 * np.pi, n_transducers, endpoint=False)
        self.transducer_positions = np.zeros(
            (n_transducers, 2), dtype=np.float64
        )
        self.transducer_positions[:, 0] = center_y + radius * np.sin(angles)
        self.transducer_positions[:, 1] = center_x + radius * np.cos(angles)

        # Build pixel grid
        py, px = np.meshgrid(
            np.arange(ny, dtype=np.float64),
            np.arange(nx, dtype=np.float64),
            indexing="ij",
        )
        # (ny * nx, 2) pixel positions
        pixel_pos = np.stack([py.ravel(), px.ravel()], axis=-1)
        n_pixels = pixel_pos.shape[0]

        # Precompute distance matrix and integer time indices
        # distance_matrix: (n_transducers, n_pixels)
        # time_indices: (n_transducers, n_pixels) -- integer time bin
        self._distance_matrix = np.zeros(
            (n_transducers, n_pixels), dtype=np.float64
        )
        for i in range(n_transducers):
            diff = pixel_pos - self.transducer_positions[i]  # (n_pixels, 2)
            self._distance_matrix[i, :] = np.sqrt(np.sum(diff ** 2, axis=-1))

        # Convert distance to time index (integer bins)
        self._time_indices = np.rint(
            self._distance_matrix / speed_of_sound
        ).astype(np.int64)

        # Clamp to valid range [0, n_times - 1]; out-of-range are clipped
        self._time_indices = np.clip(self._time_indices, 0, n_times - 1)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute sinogram from initial pressure distribution.

        Args:
            x: Initial pressure (ny, nx).

        Returns:
            Sinogram (n_transducers, n_times).
        """
        x64 = x.astype(np.float64).ravel()
        n_pixels = x64.shape[0]
        y = np.zeros((self.n_transducers, self.n_times), dtype=np.float64)

        for i in range(self.n_transducers):
            # Accumulate pixel values into time bins
            np.add.at(y[i], self._time_indices[i], x64)

        return y.astype(np.float32)

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        """Back-project sinogram to image space.

        True adjoint (no normalization): each pixel accumulates the sinogram
        values from all transducers at its corresponding time delay.

        Args:
            y: Sinogram (n_transducers, n_times).

        Returns:
            Back-projected image (ny, nx).
        """
        y64 = y.astype(np.float64)
        n_pixels = self.ny * self.nx
        x = np.zeros(n_pixels, dtype=np.float64)

        for i in range(self.n_transducers):
            # Each pixel gets the sinogram value at its time index
            x += y64[i, self._time_indices[i]]

        return x.reshape(self.ny, self.nx).astype(np.float32)

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
            "n_transducers": self.n_transducers,
            "n_times": self.n_times,
            "speed_of_sound": self.speed_of_sound,
        }

    def metadata(self) -> OperatorMetadata:
        return OperatorMetadata(
            modality="photoacoustic",
            operator_id=self.operator_id,
            x_shape=list(self.x_shape),
            y_shape=list(self.y_shape),
            is_linear=True,
            supports_autodiff=False,
            axes={
                "x_dim0": "y_spatial",
                "x_dim1": "x_spatial",
                "y_dim0": "transducer",
                "y_dim1": "time",
            },
            units={
                "speed_of_sound": "pixels/sample",
                "pressure": "a.u.",
                "time": "samples",
            },
        )
