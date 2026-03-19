"""MRI operator.

Implements Fourier encoding with k-space undersampling.
Output is complex k-space (same shape as input, with masked entries).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from pwm_core.physics.base import BaseOperator


class MRIOperator(BaseOperator):
    """MRI operator with k-space undersampling.

    Forward: FFT + mask (simulate k-space acquisition)
    Adjoint: mask + IFFT
    """

    def __init__(
        self,
        operator_id: str = "mri",
        theta: Optional[Dict[str, Any]] = None,
        x_shape: Tuple[int, int] = (64, 64),
        sampling_rate: float = 0.25,
        seed: int = 42,
    ):
        self.operator_id = operator_id
        self.theta = theta or {}
        self.x_shape = x_shape
        self.sampling_rate = sampling_rate

        # Generate undersampling mask
        H, W = x_shape
        rng = np.random.default_rng(seed)
        self.mask = (rng.random((H, W)) < sampling_rate).astype(np.float32)

        # Always keep center region (calibration)
        center_h, center_w = H // 8, W // 8
        self.mask[H//2-center_h:H//2+center_h, W//2-center_w:W//2+center_w] = 1.0

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward: FFT + undersampling mask."""
        # Compute full k-space
        kspace = np.fft.fft2(x)
        kspace = np.fft.fftshift(kspace)

        # Apply undersampling mask
        kspace_under = kspace * self.mask

        return kspace_under.astype(np.complex64)

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        """Adjoint: mask + IFFT."""
        # Apply mask (already applied in forward, but for consistency)
        y_masked = y * self.mask

        # Inverse FFT
        y_shifted = np.fft.ifftshift(y_masked)
        x_recon = np.fft.ifft2(y_shifted)

        return np.abs(x_recon).astype(np.float32)

    def info(self) -> Dict[str, Any]:
        return {
            "operator_id": self.operator_id,
            "x_shape": self.x_shape,
            "sampling_rate": self.sampling_rate,
        }
