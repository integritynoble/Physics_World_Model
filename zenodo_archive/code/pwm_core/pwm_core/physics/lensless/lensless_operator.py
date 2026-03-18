"""Lensless (Diffuser Camera) operator.

Implements PSF-based lensless imaging with a diffuser PSF generated from a
random phase mask model.  The phase mask approach produces a caustic-like PSF
whose Fourier magnitudes are approximately flat, yielding a well-conditioned
forward operator that can be reliably inverted by Tikhonov / ADMM solvers.

Previous implementation used ``gaussian_filter(random_field, sigma)`` which
created a nearly flat (spatially uniform) PSF.  After normalization to unit
sum the non-DC Fourier coefficients were O(1/N), making the deconvolution
problem catastrophically ill-conditioned (condition number > 1e5) and
causing PSNR ~ 9 dB even on noiseless data.
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

    The PSF is generated from a random phase mask model:
        1. Draw random phase phi ~ U[0, 2*pi)
        2. Smooth with Gaussian kernel of width ``psf_sigma``
        3. PSF = |ifft2(exp(j * phi_smooth))|^2, normalized to sum=1

    This produces a caustic pattern with good frequency coverage
    (|H(f)| ~ 1 for all f), ensuring the forward model is invertible.
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

        # Generate diffuser PSF via random phase mask model
        H, W = x_shape
        rng = np.random.default_rng(seed)
        phase_raw = rng.uniform(0, 2 * np.pi, (H, W))
        phase_smooth = ndimage.gaussian_filter(phase_raw, sigma=psf_sigma)
        transfer = np.exp(1j * phase_smooth)
        psf = np.abs(np.fft.ifft2(transfer)) ** 2
        self.psf = (psf / psf.sum()).astype(np.float32)  # Normalize

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
