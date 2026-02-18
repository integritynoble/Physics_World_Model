"""Lensless (Diffuser Camera) operator.

Implements PSF-based lensless imaging with large diffuser PSF.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import ndimage

from pwm_core.physics.base import BaseOperator


class LenslessOperator(BaseOperator):
    """Lensless imaging operator with diffuser PSF.

    Forward: Convolution with diffuser PSF
    Adjoint: Correlation with PSF (transposed convolution)
    """

    def __init__(
        self,
        operator_id: str = "lensless",
        theta: Optional[Dict[str, Any]] = None,
        x_shape: Tuple[int, int] = (64, 64),
        psf_sigma: float = 10.0,
        seed: int = 42,
    ):
        self.operator_id = operator_id
        self.theta = theta or {}
        self.x_shape = x_shape
        self.psf_sigma = psf_sigma

        # Generate random diffuser PSF
        H, W = x_shape
        rng = np.random.default_rng(seed)
        psf = rng.random((H, W)).astype(np.float32)
        psf = ndimage.gaussian_filter(psf, sigma=psf_sigma)
        self.psf = psf / psf.sum()  # Normalize

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward: Convolve with diffuser PSF."""
        # FFT-based convolution
        x_fft = np.fft.fft2(x)
        psf_fft = np.fft.fft2(self.psf)
        y_fft = x_fft * psf_fft
        y = np.fft.ifft2(y_fft)
        return np.real(y).astype(np.float32)

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        """Adjoint: Correlation with PSF."""
        # FFT-based correlation (convolve with flipped PSF)
        y_fft = np.fft.fft2(y)
        psf_fft_conj = np.conj(np.fft.fft2(self.psf))
        x_fft = y_fft * psf_fft_conj
        x = np.fft.ifft2(x_fft)
        return np.real(x).astype(np.float32)

    def info(self) -> Dict[str, Any]:
        return {
            "operator_id": self.operator_id,
            "x_shape": self.x_shape,
            "psf_sigma": self.psf_sigma,
        }
