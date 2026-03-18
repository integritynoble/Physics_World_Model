"""Holography operator.

Implements off-axis digital holography.
Input is complex field (or real-valued for amplitude only).
Output is 2D hologram (interference pattern).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from pwm_core.physics.base import BaseOperator


class HolographyOperator(BaseOperator):
    """Off-axis holography operator.

    Forward: Interfere object wave with reference wave
    Adjoint: Extract object wave from hologram
    """

    def __init__(
        self,
        operator_id: str = "holography",
        theta: Optional[Dict[str, Any]] = None,
        x_shape: Tuple[int, int] = (64, 64),
        carrier_freq: float = 0.2,
        reference_amplitude: float = 1.0,
    ):
        self.operator_id = operator_id
        self.theta = theta or {}
        self.x_shape = x_shape
        self.carrier_freq = carrier_freq
        self.reference_amplitude = reference_amplitude

        H, W = x_shape

        # Generate off-axis reference wave (tilted plane wave)
        yy, xx = np.meshgrid(np.arange(W), np.arange(H))
        self.reference = reference_amplitude * np.exp(
            1j * 2 * np.pi * carrier_freq * (xx + yy)
        )
        self.reference = self.reference.astype(np.complex64)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute hologram: |object + reference|^2"""
        # Convert to complex if real
        if not np.iscomplexobj(x):
            obj = x.astype(np.complex64)
        else:
            obj = x

        # Interference
        total = obj + self.reference
        hologram = np.abs(total)**2

        return hologram.astype(np.float32)

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        """Extract object wave from hologram using reference conjugate."""
        # Multiply by conjugate of reference to shift carrier
        demodulated = y * np.conj(self.reference)

        # Apply Fourier filtering to extract object term
        # (In full reconstruction, would filter in Fourier domain)
        # Here we use a simple approximation
        fft_demod = np.fft.fft2(demodulated)
        fft_shifted = np.fft.fftshift(fft_demod)

        # Create low-pass filter to remove DC and twin image
        H, W = self.x_shape
        yy, xx = np.meshgrid(np.arange(W), np.arange(H))
        center_x, center_y = H // 2, W // 2

        # Shift to carrier frequency location
        shift_x = int(self.carrier_freq * H)
        shift_y = int(self.carrier_freq * W)

        # Gaussian filter at carrier location
        sigma = H / 8
        filter_mask = np.exp(
            -((xx - center_x - shift_x)**2 + (yy - center_y - shift_y)**2) / (2 * sigma**2)
        )

        # Apply filter and shift back
        filtered = fft_shifted * filter_mask
        filtered_shifted = np.fft.ifftshift(filtered)
        reconstructed = np.fft.ifft2(filtered_shifted)

        return np.abs(reconstructed).astype(np.float32)

    def info(self) -> Dict[str, Any]:
        return {
            "operator_id": self.operator_id,
            "x_shape": self.x_shape,
            "carrier_freq": self.carrier_freq,
        }
