"""Ultrasound operator.

Pulse-echo forward model for B-mode ultrasound imaging.
Forward: tissue reflectivity map -> RF channel data (sinogram-like)
Adjoint: delay-and-sum back-projection

Includes frequency-dependent attenuation and mismatch ThetaSpace
for speed_of_sound, element_pitch, and attenuation_coeff.

Uses propagate_rf from ultrasound_helpers for the physics model.

References:
- Jensen, J.A. (1996). "Field: A Program for Simulating Ultrasound Systems",
  Medical & Biological Engineering & Computing.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from pwm_core.physics.base import BaseOperator, OperatorMetadata


class UltrasoundOperator(BaseOperator):
    """Ultrasound pulse-echo imaging operator.

    Forward: x (nz, nx) -> y (n_elements, n_samples)
        RF channel data via round-trip delay model with frequency-dependent
        attenuation: y[e,t] = sum_{z,x} x[z,x] * exp(-alpha*f_c*d) * delta(t - 2d/c)

    Adjoint: y (n_elements, n_samples) -> x (nz, nx)
        Delay-and-sum back-projection with matched attenuation weighting.

    Mismatch ThetaSpace:
        speed_of_sound: [1400, 1600] m/s
        element_pitch: [0.1e-3, 0.5e-3] m
        attenuation_coeff: [0.0, 1.5] dB/(MHz*cm)
    """

    def __init__(
        self,
        operator_id: str = "ultrasound",
        theta: Optional[Dict[str, Any]] = None,
        nz: int = 64,
        nx: int = 64,
        n_elements: int = 32,
        n_samples: int = 128,
        speed_of_sound: float = 1540.0,
        element_pitch: float = 0.3e-3,
        fs: float = 40e6,
        attenuation_coeff: float = 0.5,
        center_freq_hz: float = 5e6,
    ):
        self.operator_id = operator_id
        self.theta = theta or {}
        self.nz = nz
        self.nx = nx
        self.n_elements = n_elements
        self.n_samples = n_samples
        self.speed_of_sound = speed_of_sound
        self.element_pitch = element_pitch
        self.fs = fs
        self.attenuation_coeff = attenuation_coeff
        self.center_freq_hz = center_freq_hz

        self._x_shape = (nz, nx)
        self._y_shape = (n_elements, n_samples)
        self._is_linear = True
        self._supports_autodiff = False

        self._precompute()

    def _precompute(self) -> None:
        """Precompute pixel grid, element positions, time indices, and attenuation."""
        nz, nx = self.nz, self.nx
        n_elements = self.n_elements
        n_samples = self.n_samples
        speed_of_sound = self.speed_of_sound
        element_pitch = self.element_pitch
        fs = self.fs

        aperture = n_elements * element_pitch
        self._pixel_size_x = aperture / nx
        self._pixel_size_z = (n_samples / fs * speed_of_sound / 2.0) / nz

        # Element x-positions
        self._elem_x = np.array(
            [(e - n_elements / 2.0) * element_pitch for e in range(n_elements)],
            dtype=np.float64,
        )

        # Pixel positions
        self._z_pos = (np.arange(nz, dtype=np.float64) + 0.5) * self._pixel_size_z
        self._x_pos = (np.arange(nx, dtype=np.float64) - nx / 2.0) * self._pixel_size_x

        # Precompute time indices and distances: (n_elements, nz, nx)
        self._time_indices = np.zeros((n_elements, nz, nx), dtype=np.int64)
        self._distances = np.zeros((n_elements, nz, nx), dtype=np.float64)
        for e in range(n_elements):
            for iz in range(nz):
                dx = self._elem_x[e] - self._x_pos  # (nx,)
                dist = np.sqrt(dx ** 2 + self._z_pos[iz] ** 2)
                self._distances[e, iz, :] = dist
                t_round = 2.0 * dist / speed_of_sound
                self._time_indices[e, iz, :] = np.clip(
                    (t_round * fs).astype(np.int64), 0, n_samples - 1
                )

        # Frequency-dependent attenuation weights
        # alpha in dB/(MHz*cm), f_c in Hz, d in meters
        # Convert: alpha_neper = alpha_dB * ln(10)/20, f_MHz = f/1e6, d_cm = d*100
        # atten = exp(-alpha_dB * ln(10)/20 * f_MHz * d_cm)
        #       = exp(-alpha * f_c/1e6 * d*100 * ln(10)/20)
        alpha = self.attenuation_coeff  # dB/(MHz*cm)
        f_mhz = self.center_freq_hz / 1e6
        d_cm = self._distances * 100.0  # meters to cm
        self._atten_weights = np.exp(
            -alpha * f_mhz * d_cm * np.log(10) / 20.0
        )

    def set_theta(self, **kwargs: Any) -> None:
        """Update mismatch parameters and recompute internal state.

        Supported parameters:
            speed_of_sound: float (m/s)
            element_pitch: float (m)
            attenuation_coeff: float (dB/(MHz*cm))
        """
        changed = False
        if "speed_of_sound" in kwargs:
            self.speed_of_sound = float(kwargs["speed_of_sound"])
            changed = True
        if "element_pitch" in kwargs:
            self.element_pitch = float(kwargs["element_pitch"])
            changed = True
        if "attenuation_coeff" in kwargs:
            self.attenuation_coeff = float(kwargs["attenuation_coeff"])
            changed = True
        if changed:
            self._precompute()

    def get_theta(self) -> Dict[str, float]:
        """Return current mismatch parameters."""
        return {
            "speed_of_sound": self.speed_of_sound,
            "element_pitch": self.element_pitch,
            "attenuation_coeff": self.attenuation_coeff,
        }

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Generate RF channel data from tissue reflectivity.

        Args:
            x: Tissue reflectivity map (nz, nx).

        Returns:
            RF channel data (n_elements, n_samples).
        """
        x64 = np.asarray(x, dtype=np.float64)
        y = np.zeros(self._y_shape, dtype=np.float64)

        for e in range(self.n_elements):
            for iz in range(self.nz):
                weighted = x64[iz] * self._atten_weights[e, iz]
                np.add.at(y[e], self._time_indices[e, iz], weighted)

        return y.astype(np.float32)

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        """Delay-and-sum back-projection with matched attenuation weighting.

        Args:
            y: RF channel data (n_elements, n_samples).

        Returns:
            Back-projected image (nz, nx).
        """
        y64 = np.asarray(y, dtype=np.float64)
        x = np.zeros(self._x_shape, dtype=np.float64)

        for e in range(self.n_elements):
            for iz in range(self.nz):
                x[iz] += y64[e, self._time_indices[e, iz]] * self._atten_weights[e, iz]

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
            "nz": self.nz,
            "nx": self.nx,
            "n_elements": self.n_elements,
            "n_samples": self.n_samples,
            "speed_of_sound": self.speed_of_sound,
            "attenuation_coeff": self.attenuation_coeff,
            "center_freq_hz": self.center_freq_hz,
        }

    def metadata(self) -> OperatorMetadata:
        return OperatorMetadata(
            modality="ultrasound",
            operator_id=self.operator_id,
            x_shape=list(self.x_shape),
            y_shape=list(self.y_shape),
            is_linear=True,
            supports_autodiff=False,
            axes={
                "x_dim0": "depth",
                "x_dim1": "lateral",
                "y_dim0": "element",
                "y_dim1": "time_sample",
            },
            units={
                "speed_of_sound": "m/s",
                "attenuation_coeff": "dB/(MHz*cm)",
                "center_freq_hz": "Hz",
                "reflectivity": "a.u.",
                "time": "samples",
            },
        )
