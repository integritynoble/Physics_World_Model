"""Compressive mask category module – shared physics for ~20 compressive modalities.

Handles modalities with coded aperture / mask + compressive sensing:
  CASSI, CACTI, SPC, matrix, coded_exposure, ghost_imaging, etc.

DAG patterns: M --> D, M --> Sigma --> D, M --> W --> Sigma --> D, M --> T --> D
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Mask generation
# ---------------------------------------------------------------------------

def binary_random_mask(shape: Tuple[int, ...], density: float = 0.5, seed: int = 42) -> np.ndarray:
    """Generate a binary random mask.

    Args:
        shape: Mask shape (H, W) or (H, W, T).
        density: Fraction of ones.
        seed: Random seed.
    """
    rng = np.random.RandomState(seed)
    return (rng.rand(*shape) < density).astype(np.float32)


def gaussian_random_matrix(n_measurements: int, n_pixels: int, seed: int = 42) -> np.ndarray:
    """Gaussian random sensing matrix for SPC / CS."""
    rng = np.random.RandomState(seed)
    return (rng.randn(n_measurements, n_pixels) / np.sqrt(n_measurements)).astype(np.float32)


def hadamard_matrix(n: int) -> np.ndarray:
    """Generate Hadamard sensing matrix of order *n* (must be power of 2)."""
    from scipy.linalg import hadamard
    H = hadamard(n).astype(np.float32) / np.sqrt(n)
    return H


def spectral_dispersion_matrix(
    n_bands: int,
    n_det: int,
    step: int = 2,
) -> np.ndarray:
    """Spectral dispersion matrix for CASSI-type modalities.

    Models the shift-and-sum operation where each spectral band
    is shifted by `step * band_index` pixels along the detector.
    """
    # Build a sparse shift matrix
    A = np.zeros((n_det, n_bands), dtype=np.float32)
    for b in range(n_bands):
        shift = b * step
        if shift < n_det:
            A[shift, b] = 1.0
    return A


# ---------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------

def forward(operator, x: np.ndarray, **kwargs) -> np.ndarray:
    """Compressive forward model: coded mask + optional spectral dispersion.

    If the operator has a ``forward()`` method, delegates to it.
    """
    try:
        y = operator.forward(x)
        if y is not None:
            return np.asarray(y)
    except Exception:
        pass

    # Fallback: simple mask modulation
    if x.ndim == 2:
        mask = binary_random_mask(x.shape)
        return (x * mask).astype(np.float32)
    elif x.ndim == 3:
        # Spectral: mask each band, then sum (CASSI-like)
        H, W, C = x.shape
        mask = binary_random_mask((H, W))
        y = np.zeros((H, W), dtype=np.float32)
        for c in range(C):
            y += x[..., c] * mask
        return y
    return x


def adjoint(operator, y: np.ndarray, **kwargs) -> np.ndarray:
    """Compressive adjoint: transpose of mask operation."""
    try:
        x_hat = operator.adjoint(y)
        if x_hat is not None:
            return np.asarray(x_hat)
    except Exception:
        pass
    return y


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------

def gap_tv_solve(
    y: np.ndarray,
    mask: np.ndarray,
    n_iter: int = 50,
    tv_weight: float = 0.1,
) -> np.ndarray:
    """Generalised Alternating Projection with TV denoising.

    Simplified implementation for benchmark baseline.
    """
    try:
        from skimage.restoration import denoise_tv_chambolle
    except ImportError:
        def denoise_tv_chambolle(img, weight=0.1, **kw):
            from scipy.ndimage import gaussian_filter
            return gaussian_filter(img, sigma=weight * 10)

    x = np.zeros_like(mask, dtype=np.float32) if mask.ndim == 2 else y.copy()

    for _ in range(n_iter):
        # Projection
        if mask.ndim == 2 and x.ndim == 2:
            residual = y - x * mask
            x = x + residual * mask
        # TV denoising
        x = denoise_tv_chambolle(x, weight=tv_weight).astype(np.float32)

    return np.clip(x, 0, 1)


# ---------------------------------------------------------------------------
# Phantom generation
# ---------------------------------------------------------------------------

def generate_phantom(dims: Tuple[int, ...], seed: int = 42) -> np.ndarray:
    """Generate a spectral scene for compressive modalities."""
    rng = np.random.RandomState(seed)

    if len(dims) < 3:
        dims = (*dims, 28)
    H, W, C = dims[0], dims[1], dims[2]
    x = np.zeros((H, W, C), dtype=np.float32)

    n_regions = max(5, int(H * W / 3000))
    spectra = []
    for _ in range(n_regions):
        center = rng.rand() * C
        width = rng.rand() * C * 0.3 + C * 0.05
        spec = np.exp(-0.5 * ((np.arange(C) - center) / max(width, 0.1)) ** 2)
        spectra.append((spec / (spec.max() + 1e-12)).astype(np.float32))

    for i in range(n_regions):
        cx = rng.randint(int(W * 0.1), int(W * 0.9))
        cy = rng.randint(int(H * 0.1), int(H * 0.9))
        r = rng.randint(max(3, min(H, W) // 15), max(5, min(H, W) // 5))
        intensity = rng.rand() * 0.7 + 0.3
        yy, xx = np.ogrid[:H, :W]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
        x[mask] += intensity * spectra[i % len(spectra)][None, :]

    return np.clip(x, 0, 1).astype(np.float32)
