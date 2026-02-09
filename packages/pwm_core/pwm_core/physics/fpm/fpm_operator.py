"""FPM (Fourier Ptychographic Microscopy) operator.

Coherent imaging forward model with angle-varied illumination.
Forward: high-resolution complex object -> low-resolution intensity images
Adjoint: linearized back-projection from intensities to HR estimate

    y_j = |F^{-1}{ P(k - k_j) * O(k) }|^2   for each LED j

where O(k) is the Fourier transform of the high-resolution object, P is the
pupil function, and k_j is the illumination wavevector for LED j.

References:
- Zheng, G., Horstmeyer, R. & Yang, C. (2013). "Wide-field, high-resolution
  Fourier ptychographic microscopy", Nature Photonics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pwm_core.physics.base import BaseOperator, OperatorMetadata


class FPMOperator(BaseOperator):
    """Fourier Ptychographic Microscopy operator.

    Forward: x (hr_size, hr_size) complex -> y (n_leds, lr_size, lr_size) real
        For each LED j:
            1. FFT of HR object
            2. Crop sub-aperture at LED position k_j
            3. Multiply by pupil
            4. IFFT -> field
            5. |field|^2 -> intensity

    Adjoint (linearized): y (n_leds, lr_size, lr_size) -> x (hr_size, hr_size) real
        Approximate linearised adjoint for gradient-based solvers.
    """

    def __init__(
        self,
        operator_id: str = "fpm",
        theta: Optional[Dict[str, Any]] = None,
        hr_size: int = 128,
        lr_size: int = 32,
        n_leds: Optional[int] = None,
        led_positions: Optional[np.ndarray] = None,
        pupil: Optional[np.ndarray] = None,
        na: float = 0.1,
    ):
        self.operator_id = operator_id
        self.theta = theta or {}
        self.hr_size = hr_size
        self.lr_size = lr_size
        self.na = na
        self._is_linear = False
        self._supports_autodiff = False

        # LED grid configuration
        if n_leds is not None and led_positions is not None:
            self.n_leds = n_leds
            self.led_positions = np.asarray(led_positions, dtype=np.int32)
        elif led_positions is not None:
            self.led_positions = np.asarray(led_positions, dtype=np.int32)
            self.n_leds = self.led_positions.shape[0]
        else:
            # Default: 5x5 LED grid
            n_side = 5
            self.n_leds = n_side * n_side
            span = 2 * lr_size // 3
            offsets = np.linspace(-span, span, n_side, dtype=np.int32)
            gy, gx = np.meshgrid(offsets, offsets, indexing="ij")
            self.led_positions = np.stack(
                [gy.ravel(), gx.ravel()], axis=-1
            )  # (n_leds, 2) as (ky, kx)

        self._x_shape = (hr_size, hr_size)
        self._y_shape = (self.n_leds, lr_size, lr_size)

        # Pupil function (circular aperture)
        if pupil is not None:
            self.pupil = np.asarray(pupil, dtype=np.complex128)
        else:
            radius = lr_size // 2
            cy, cx = lr_size // 2, lr_size // 2
            yy, xx = np.mgrid[:lr_size, :lr_size]
            dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            self.pupil = (dist <= radius).astype(np.complex128)

    def _crop_spectrum(
        self, spectrum: np.ndarray, led_pos: np.ndarray
    ) -> np.ndarray:
        """Extract lr_size x lr_size sub-aperture from HR spectrum.

        The crop is centred at (hr_size//2 + led_pos[0], hr_size//2 + led_pos[1])
        in the shifted spectrum.

        Args:
            spectrum: FFT-shifted HR spectrum (hr_size, hr_size), complex.
            led_pos: (ky, kx) offset from DC in pixels.

        Returns:
            Sub-aperture (lr_size, lr_size), complex.
        """
        half = self.lr_size // 2
        centre_y = self.hr_size // 2 + int(led_pos[0])
        centre_x = self.hr_size // 2 + int(led_pos[1])

        # Compute row/col ranges and handle boundary via periodic wrapping
        rows = (np.arange(centre_y - half, centre_y - half + self.lr_size)
                % self.hr_size)
        cols = (np.arange(centre_x - half, centre_x - half + self.lr_size)
                % self.hr_size)

        return spectrum[np.ix_(rows, cols)]

    def _place_spectrum(
        self,
        sub: np.ndarray,
        led_pos: np.ndarray,
        target: np.ndarray,
    ) -> None:
        """Accumulate a lr_size x lr_size patch into the HR spectrum.

        Inverse of _crop_spectrum (adjoint accumulation).

        Args:
            sub: Sub-aperture (lr_size, lr_size), complex.
            led_pos: (ky, kx) offset from DC in pixels.
            target: HR spectrum (hr_size, hr_size), complex. Modified in-place.
        """
        half = self.lr_size // 2
        centre_y = self.hr_size // 2 + int(led_pos[0])
        centre_x = self.hr_size // 2 + int(led_pos[1])

        rows = (np.arange(centre_y - half, centre_y - half + self.lr_size)
                % self.hr_size)
        cols = (np.arange(centre_x - half, centre_x - half + self.lr_size)
                % self.hr_size)

        target[np.ix_(rows, cols)] += sub

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute low-resolution intensity images from HR complex object.

        Args:
            x: High-resolution complex object (hr_size, hr_size).

        Returns:
            Low-resolution intensity stack (n_leds, lr_size, lr_size).
        """
        x128 = x.astype(np.complex128)
        # Shifted HR spectrum (DC at centre)
        O = np.fft.fftshift(np.fft.fft2(x128))

        y = np.zeros((self.n_leds, self.lr_size, self.lr_size),
                      dtype=np.float64)

        for j in range(self.n_leds):
            sub = self._crop_spectrum(O, self.led_positions[j])
            field = np.fft.ifft2(np.fft.ifftshift(self.pupil * sub))
            y[j] = np.abs(field) ** 2

        return y.astype(np.float32)

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        """Linearised adjoint: back-project intensities to HR estimate.

        Uses the amplitude-based linearisation:
            1. Take sqrt of each intensity image (amplitude estimate)
            2. FFT each amplitude, multiply by conj(pupil)
            3. Place back into HR spectrum, average over LEDs
            4. IFFT to spatial domain, return real part

        Args:
            y: Intensity images (n_leds, lr_size, lr_size).

        Returns:
            HR estimate (hr_size, hr_size), real-valued.
        """
        y64 = y.astype(np.float64)
        O_accum = np.zeros((self.hr_size, self.hr_size), dtype=np.complex128)

        for j in range(self.n_leds):
            # Amplitude estimate
            amp = np.sqrt(np.maximum(y64[j], 0.0))
            # Low-res field in Fourier domain (shifted so DC at centre)
            F_amp = np.fft.fftshift(np.fft.fft2(amp))
            # Multiply by conjugate pupil
            sub = np.conj(self.pupil) * F_amp
            # Place into HR spectrum
            self._place_spectrum(sub, self.led_positions[j], O_accum)

        # Average over LEDs
        O_accum /= self.n_leds

        # Back to spatial domain
        x_est = np.fft.ifft2(np.fft.ifftshift(O_accum))
        return np.real(x_est).astype(np.float32)

    @property
    def x_shape(self) -> Tuple[int, ...]:
        return (self.hr_size, self.hr_size)

    @property
    def y_shape(self) -> Tuple[int, ...]:
        return (self.n_leds, self.lr_size, self.lr_size)

    @property
    def is_linear(self) -> bool:
        return False

    @property
    def supports_autodiff(self) -> bool:
        return False

    def info(self) -> Dict[str, Any]:
        return {
            "operator_id": self.operator_id,
            "hr_size": self.hr_size,
            "lr_size": self.lr_size,
            "n_leds": self.n_leds,
            "na": self.na,
            "led_positions_shape": list(self.led_positions.shape),
            "pupil_shape": list(self.pupil.shape),
        }

    def metadata(self) -> OperatorMetadata:
        return OperatorMetadata(
            modality="fpm",
            operator_id=self.operator_id,
            x_shape=list(self.x_shape),
            y_shape=list(self.y_shape),
            is_linear=False,
            supports_autodiff=False,
            axes={
                "x_dim0": "y_spatial_hr",
                "x_dim1": "x_spatial_hr",
                "y_dim0": "led_index",
                "y_dim1": "y_spatial_lr",
                "y_dim2": "x_spatial_lr",
            },
            units={"spatial": "pixels", "na": "dimensionless"},
            sampling_info={
                "n_leds": self.n_leds,
                "hr_size": self.hr_size,
                "lr_size": self.lr_size,
                "na": self.na,
            },
        )
