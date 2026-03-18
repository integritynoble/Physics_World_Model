"""Richardson-Lucy Deconvolution for Widefield Microscopy.

Classical iterative deconvolution algorithm for Poisson noise model.

References:
- Richardson, W.H. (1972). "Bayesian-based iterative method of image restoration"
- Lucy, L.B. (1974). "An iterative technique for the rectification of observed distributions"

Benchmark: DeconvolutionLab2 test images (Ely, CElegans, Sperm)
Expected PSNR: 25-30 dB after 50 iterations
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve


def richardson_lucy_2d(
    image: np.ndarray,
    psf: np.ndarray,
    iterations: int = 50,
    clip: bool = True,
    background: float = 0.0,
    regularization: float = 0.0,
) -> np.ndarray:
    """2D Richardson-Lucy deconvolution.

    Args:
        image: Blurred/noisy input image (H, W)
        psf: Point spread function (must sum to ~1)
        iterations: Number of RL iterations
        clip: Whether to clip negative values
        background: Background level to subtract
        regularization: TV regularization strength (0 = none)

    Returns:
        Deconvolved image
    """
    # Ensure float32
    image = image.astype(np.float32)
    psf = psf.astype(np.float32)

    # Normalize PSF
    psf = psf / (psf.sum() + 1e-12)

    # Flipped PSF for correlation
    psf_flip = psf[::-1, ::-1]

    # Subtract background
    image = np.maximum(image - background, 1e-10)

    # Initialize estimate
    estimate = np.ones_like(image) * image.mean()
    estimate = np.maximum(estimate, 1e-10)

    for i in range(iterations):
        # Forward model: PSF * estimate
        blurred_estimate = fftconvolve(estimate, psf, mode='same')
        blurred_estimate = np.maximum(blurred_estimate, 1e-10)

        # Ratio
        ratio = image / blurred_estimate

        # Correction factor: PSF^T * ratio (correlation)
        correction = fftconvolve(ratio, psf_flip, mode='same')

        # Update
        estimate = estimate * correction

        # Optional TV regularization
        if regularization > 0:
            grad_x = np.roll(estimate, -1, axis=1) - estimate
            grad_y = np.roll(estimate, -1, axis=0) - estimate
            grad_norm = np.sqrt(grad_x**2 + grad_y**2 + 1e-8)

            div_x = grad_x / grad_norm - np.roll(grad_x / grad_norm, 1, axis=1)
            div_y = grad_y / grad_norm - np.roll(grad_y / grad_norm, 1, axis=0)

            estimate = estimate / (1 - regularization * (div_x + div_y))

        # Clip negatives
        if clip:
            estimate = np.maximum(estimate, 1e-10)

    return estimate.astype(np.float32)


def richardson_lucy_3d(
    volume: np.ndarray,
    psf: np.ndarray,
    iterations: int = 50,
    clip: bool = True,
    background: float = 0.0,
) -> np.ndarray:
    """3D Richardson-Lucy deconvolution for confocal/widefield stacks.

    Args:
        volume: Blurred/noisy input volume (H, W, D) or (D, H, W)
        psf: 3D Point spread function
        iterations: Number of RL iterations
        clip: Whether to clip negative values
        background: Background level to subtract

    Returns:
        Deconvolved volume
    """
    from scipy.fft import fftn, ifftn

    volume = volume.astype(np.float32)
    psf = psf.astype(np.float32)

    # Normalize PSF
    psf = psf / (psf.sum() + 1e-12)

    # Pad PSF to match volume size for FFT
    psf_padded = np.zeros_like(volume)
    slices = tuple(slice(0, s) for s in psf.shape)
    psf_padded[slices] = psf

    # Shift PSF center to corner for FFT
    for axis in range(3):
        psf_padded = np.roll(psf_padded, -psf.shape[axis] // 2, axis=axis)

    # Precompute FFTs
    psf_fft = fftn(psf_padded)
    psf_fft_conj = np.conj(psf_fft)

    # Subtract background
    volume = np.maximum(volume - background, 1e-10)

    # Initialize
    estimate = np.ones_like(volume) * volume.mean()
    estimate = np.maximum(estimate, 1e-10)

    for i in range(iterations):
        # Forward: convolve estimate with PSF
        estimate_fft = fftn(estimate)
        blurred = np.real(ifftn(estimate_fft * psf_fft))
        blurred = np.maximum(blurred, 1e-10)

        # Ratio
        ratio = volume / blurred

        # Correction: correlate ratio with PSF
        ratio_fft = fftn(ratio)
        correction = np.real(ifftn(ratio_fft * psf_fft_conj))

        # Update
        estimate = estimate * correction

        if clip:
            estimate = np.maximum(estimate, 1e-10)

    return estimate.astype(np.float32)


def richardson_lucy_operator(
    y: np.ndarray,
    forward: Callable,
    adjoint: Callable,
    x_shape: Tuple[int, ...],
    iterations: int = 50,
    clip: bool = True,
) -> np.ndarray:
    """Richardson-Lucy using forward/adjoint operators.

    More general version that works with any linear operator.

    Args:
        y: Measurements (blurred image)
        forward: Forward operator (blur/convolve)
        adjoint: Adjoint operator (correlation)
        x_shape: Shape of object to reconstruct
        iterations: Number of iterations
        clip: Clip negative values

    Returns:
        Deconvolved image
    """
    y = y.astype(np.float32)
    y = np.maximum(y, 1e-10)

    # Initialize with adjoint
    estimate = adjoint(y).reshape(x_shape).astype(np.float32)
    estimate = np.maximum(estimate, 1e-10)

    # For normalization, compute adjoint of ones
    ones = np.ones_like(y)
    norm = adjoint(ones).reshape(x_shape).astype(np.float32)
    norm = np.maximum(norm, 1e-10)

    for i in range(iterations):
        # Forward model
        blurred = forward(estimate)
        blurred = np.maximum(blurred, 1e-10)

        # Ratio in measurement space
        ratio = y / blurred

        # Back-project ratio
        correction = adjoint(ratio).reshape(x_shape)

        # Update (multiplicative)
        estimate = estimate * (correction / norm)

        if clip:
            estimate = np.maximum(estimate, 1e-10)

    return estimate.astype(np.float32)


def run_richardson_lucy(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run Richardson-Lucy reconstruction.

    Args:
        y: Blurred measurements
        physics: Physics operator (should have PSF or forward/adjoint)
        cfg: Configuration with:
            - iters: Number of iterations (default: 50)
            - background: Background level (default: 0)
            - regularization: TV regularization (default: 0)

    Returns:
        Tuple of (reconstructed, info_dict)
    """
    iters = cfg.get("iters", 50)
    background = cfg.get("background", 0.0)
    regularization = cfg.get("regularization", 0.0)

    info = {
        "solver": "richardson_lucy",
        "iters": iters,
    }

    try:
        # Try to get PSF from physics
        psf = None
        if hasattr(physics, 'psf'):
            psf = physics.psf
        elif hasattr(physics, 'kernel'):
            psf = physics.kernel
        elif hasattr(physics, 'info'):
            op_info = physics.info()
            if 'psf' in op_info:
                psf = op_info['psf']

        if psf is not None:
            # Direct PSF-based RL
            if y.ndim == 2:
                result = richardson_lucy_2d(
                    y, psf, iters, clip=True,
                    background=background,
                    regularization=regularization
                )
            elif y.ndim == 3:
                result = richardson_lucy_3d(
                    y, psf, iters, clip=True,
                    background=background
                )
            else:
                result = y

            return result, info

        # Fall back to operator-based RL
        if hasattr(physics, 'forward') and hasattr(physics, 'adjoint'):
            x_shape = y.shape
            if hasattr(physics, 'x_shape'):
                x_shape = tuple(physics.x_shape)
            elif hasattr(physics, 'info'):
                op_info = physics.info()
                if 'x_shape' in op_info:
                    x_shape = tuple(op_info['x_shape'])

            result = richardson_lucy_operator(
                y, physics.forward, physics.adjoint,
                x_shape, iters, clip=True
            )
            return result, info

        # No PSF or operators, return input
        info["warning"] = "no_psf_or_operators"
        return y.astype(np.float32), info

    except Exception as e:
        info["error"] = str(e)
        return y.astype(np.float32), info
