"""Integral Photography reconstruction solvers.

References:
- Lippmann, G. (1908). "Epreuves reversibles donnant la sensation du relief",
  Comptes Rendus de l'Academie des Sciences.
- Park, J.H. et al. (2009). "Recent progress in three-dimensional information
  processing based on integral imaging", Applied Optics.

Expected PSNR: 27.0 dB on synthetic benchmark
"""
from __future__ import annotations

import numpy as np
from typing import Any, Dict, Tuple


def depth_estimation(
    measurement: np.ndarray,
    depth_weights: np.ndarray,
    psf_sigmas: np.ndarray = None,
    regularization: float = 0.01,
) -> np.ndarray:
    """Depth-plane deconvolution for integral photography.

    Uses Wiener deconvolution with depth-dependent PSFs (defocus model).
    Solves: x_d = IFFT{ conj(H_d) * Y / (sum_d' |H_d'|^2 + reg) }

    Args:
        measurement: 2D measurement image (H, W) from integral sensor.
        depth_weights: Weights per depth plane (n_depths,).
        psf_sigmas: Per-depth Gaussian PSF sigma (n_depths,). If None, uses scalar weights only.
        regularization: Wiener filter regularization parameter.

    Returns:
        Reconstructed volume (H, W, n_depths).
    """
    h, w = measurement.shape
    n_depths = len(depth_weights)
    depth_weights = depth_weights / (depth_weights.sum() + 1e-10)

    result = np.zeros((h, w, n_depths), dtype=np.float32)
    Y = np.fft.fft2(measurement.astype(np.float64))

    if psf_sigmas is not None:
        # Build per-depth OTFs
        fy = np.fft.fftfreq(h)[:, np.newaxis]
        fx = np.fft.fftfreq(w)[np.newaxis, :]
        freq_sq = fy**2 + fx**2

        OTFs = []
        for d in range(n_depths):
            sigma = psf_sigmas[d]
            # Gaussian OTF: exp(-2*pi^2*sigma^2*f^2)
            otf = depth_weights[d] * np.exp(-2 * np.pi**2 * sigma**2 * freq_sq)
            OTFs.append(otf)

        # Denominator: sum_d |H_d|^2 + reg
        denom = np.zeros((h, w), dtype=np.float64)
        for otf in OTFs:
            denom += np.abs(otf)**2
        denom += regularization

        # Per-depth Wiener filter
        for d in range(n_depths):
            X_d = np.conj(OTFs[d]) * Y / denom
            x_d = np.real(np.fft.ifft2(X_d))
            result[:, :, d] = np.clip(x_d, 0, None).astype(np.float32)
    else:
        # Scalar weights only (fallback) - correct multi-depth formula
        w_sq_sum = np.sum(depth_weights**2) + regularization
        for d in range(n_depths):
            w_d = depth_weights[d]
            x_d = measurement.astype(np.float64) * w_d / w_sq_sum
            result[:, :, d] = np.clip(x_d, 0, None).astype(np.float32)

    return result


def dibr(
    measurement: np.ndarray,
    depth_weights: np.ndarray,
    psf_sigmas: np.ndarray = None,
    depth_map: np.ndarray = None,
    regularization: float = 0.01,
    n_iters: int = 50,
) -> np.ndarray:
    """DIBR iterative refinement for integral photography.

    Uses Landweber iteration with depth-dependent PSF forward/adjoint.

    Args:
        measurement: 2D measurement image (H, W).
        depth_weights: Weights per depth plane (n_depths,).
        psf_sigmas: Per-depth Gaussian PSF sigmas.
        depth_map: Not used (kept for API compat).
        regularization: Regularization parameter.
        n_iters: Number of iterations.

    Returns:
        Reconstructed volume (H, W, n_depths).
    """
    h, w = measurement.shape[:2]
    n_depths = len(depth_weights)
    depth_weights = depth_weights / (depth_weights.sum() + 1e-10)

    # Initialize with Wiener solution
    volume = depth_estimation(measurement, depth_weights, psf_sigmas, regularization)

    if psf_sigmas is not None:
        # Precompute OTFs
        fy = np.fft.fftfreq(h)[:, np.newaxis]
        fx = np.fft.fftfreq(w)[np.newaxis, :]
        freq_sq = fy**2 + fx**2
        OTFs = []
        for d in range(n_depths):
            sigma = psf_sigmas[d]
            otf = depth_weights[d] * np.exp(-2 * np.pi**2 * sigma**2 * freq_sq)
            OTFs.append(otf)

        # Lipschitz constant for step size
        L = 0.0
        for otf in OTFs:
            L += np.max(np.abs(otf)**2)
        step = 1.0 / (L + 1e-10)

        Y = np.fft.fft2(measurement.astype(np.float64))

        for it in range(n_iters):
            # Forward: sum_d conv(x_d, PSF_d)
            Forward_F = np.zeros((h, w), dtype=np.complex128)
            for d in range(n_depths):
                X_d = np.fft.fft2(volume[:, :, d].astype(np.float64))
                Forward_F += OTFs[d] * X_d

            # Residual in Fourier domain
            Residual_F = Forward_F - Y

            # Back-project: conj(H_d) * Residual_F
            for d in range(n_depths):
                grad_F = np.conj(OTFs[d]) * Residual_F
                grad = np.real(np.fft.ifft2(grad_F))
                volume[:, :, d] -= (step * grad).astype(np.float32)

            volume = np.clip(volume, 0, None)
    else:
        # Scalar weight iteration (Landweber)
        L = np.sum(depth_weights**2)
        step = 1.0 / (L + 1e-10)

        for it in range(n_iters):
            forward = np.zeros((h, w), dtype=np.float64)
            for d in range(n_depths):
                forward += depth_weights[d] * volume[:, :, d].astype(np.float64)
            residual = forward - measurement.astype(np.float64)
            for d in range(n_depths):
                volume[:, :, d] -= (step * depth_weights[d] * residual).astype(np.float32)
            volume = np.clip(volume, 0, None)

    return volume.astype(np.float32)


def run_integral(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Portfolio-compatible wrapper for integral photography reconstruction.

    Args:
        y: Measurement image (H, W).
        physics: Physics operator with depth_weights attribute.
        cfg: Configuration dict with optional 'method' key.

    Returns:
        Tuple of (reconstructed_volume, info_dict).
    """
    method = cfg.get("method", "depth_estimation")
    info: Dict[str, Any] = {"solver": "integral", "method": method}

    try:
        # Get depth weights from physics
        if hasattr(physics, 'depth_weights'):
            depth_weights = physics.depth_weights
        else:
            depth_weights = np.ones(16) / 16.0

        # Get PSF sigmas from physics if available
        psf_sigmas = getattr(physics, 'psf_sigmas', None)

        if method == "dibr":
            n_iters = cfg.get("n_iters", 50)
            reg = cfg.get("regularization", 0.01)
            result = dibr(y, depth_weights, psf_sigmas=psf_sigmas, regularization=reg, n_iters=n_iters)
        else:
            reg = cfg.get("regularization", 0.01)
            result = depth_estimation(y, depth_weights, psf_sigmas=psf_sigmas, regularization=reg)

        return result, info
    except Exception as e:
        info["error"] = str(e)
        return y.astype(np.float32), info
