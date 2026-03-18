"""Microscopy PSF category module – shared physics for ~30 microscopy modalities.

Handles modalities whose forward model is PSF convolution:
  widefield, confocal, TIRF, lightsheet, STED, SIM, expansion, etc.

DAG patterns: C --> D, C --> N --> D, P --> C --> D, C --> C --> D
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.signal import fftconvolve


# ---------------------------------------------------------------------------
# PSF generation
# ---------------------------------------------------------------------------

def gaussian_psf(
    size: int,
    sigma: float = 2.0,
    sigma_y: Optional[float] = None,
) -> np.ndarray:
    """2-D Gaussian PSF.

    Args:
        size: Kernel side length (odd recommended).
        sigma: Standard deviation in x.
        sigma_y: Standard deviation in y (defaults to *sigma*).
    """
    if sigma_y is None:
        sigma_y = sigma
    c = size // 2
    y, x = np.ogrid[:size, :size]
    psf = np.exp(-((x - c) ** 2 / (2 * sigma ** 2) + (y - c) ** 2 / (2 * sigma_y ** 2)))
    return (psf / psf.sum()).astype(np.float32)


def airy_psf(size: int, na: float = 1.4, wavelength_nm: float = 520.0, pixel_nm: float = 100.0) -> np.ndarray:
    """Approximate Airy disk PSF.

    Uses the Gaussian approximation:  sigma ≈ 0.21 * lambda / NA  (in object space).
    """
    sigma_nm = 0.21 * wavelength_nm / na
    sigma_px = sigma_nm / pixel_nm
    return gaussian_psf(size, sigma=sigma_px)


def confocal_psf(size: int, sigma: float = 1.5) -> np.ndarray:
    """Confocal PSF (product of excitation and emission PSFs)."""
    exc = gaussian_psf(size, sigma * 0.9)
    em = gaussian_psf(size, sigma)
    psf = exc * em
    return (psf / psf.sum()).astype(np.float32)


def lightsheet_psf(size: int, sigma_lateral: float = 2.0, sigma_axial: float = 4.0) -> np.ndarray:
    """Light-sheet PSF (anisotropic Gaussian)."""
    return gaussian_psf(size, sigma=sigma_lateral, sigma_y=sigma_axial)


def sim_otf_pattern(shape: Tuple[int, int], n_orientations: int = 3, n_phases: int = 3) -> np.ndarray:
    """Generate SIM illumination patterns (H, W, n_orientations * n_phases)."""
    H, W = shape
    patterns = []
    for ori in range(n_orientations):
        angle = np.pi * ori / n_orientations
        kx = np.cos(angle) * 2 * np.pi / (W / 4)
        ky = np.sin(angle) * 2 * np.pi / (H / 4)
        yy, xx = np.mgrid[:H, :W]
        for phase_idx in range(n_phases):
            phase = 2 * np.pi * phase_idx / n_phases
            pattern = 0.5 * (1 + np.cos(kx * xx + ky * yy + phase))
            patterns.append(pattern.astype(np.float32))
    return np.stack(patterns, axis=-1)


# ---------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------

def forward(operator, x: np.ndarray, **kwargs) -> np.ndarray:
    """PSF convolution forward model.

    If the operator has a ``forward()`` method, delegates to it.
    Otherwise performs direct PSF convolution.
    """
    try:
        y = operator.forward(x)
        if y is not None:
            return np.asarray(y)
    except Exception:
        pass

    # Fallback: direct PSF convolution
    sigma = 2.0
    if hasattr(operator, "theta"):
        sigma = operator.theta.get("sigma", sigma)
    psf = gaussian_psf(15, sigma=sigma)

    if x.ndim == 2:
        return fftconvolve(x, psf, mode="same").astype(np.float32)
    elif x.ndim == 3:
        result = np.zeros_like(x)
        for c in range(x.shape[-1]):
            result[..., c] = fftconvolve(x[..., c], psf, mode="same")
        return result.astype(np.float32)
    return x


def adjoint(operator, y: np.ndarray, **kwargs) -> np.ndarray:
    """PSF convolution adjoint (correlation with flipped PSF)."""
    try:
        x_hat = operator.adjoint(y)
        if x_hat is not None:
            return np.asarray(x_hat)
    except Exception:
        pass
    # Convolution is self-adjoint for symmetric PSFs
    return forward(operator, y, **kwargs)


# ---------------------------------------------------------------------------
# Phantom generation
# ---------------------------------------------------------------------------

def generate_phantom(dims: Tuple[int, ...], seed: int = 42) -> np.ndarray:
    """Generate a fluorescence cell phantom."""
    rng = np.random.RandomState(seed)
    H = dims[0]
    W = dims[1] if len(dims) >= 2 else H
    x = np.zeros((H, W), dtype=np.float32)

    n_cells = max(5, int(H * W / 2000))
    for _ in range(n_cells):
        cx = rng.randint(int(W * 0.08), int(W * 0.92))
        cy = rng.randint(int(H * 0.08), int(H * 0.92))
        rx = rng.randint(max(2, W // 40), max(4, W // 10))
        ry = rng.randint(max(2, H // 40), max(4, H // 10))
        intensity = rng.rand() * 0.7 + 0.3
        yy, xx = np.ogrid[:H, :W]
        mask = ((xx - cx) / max(rx, 1)) ** 2 + ((yy - cy) / max(ry, 1)) ** 2 <= 1.0
        x[mask] = np.maximum(x[mask], intensity)

    if len(dims) >= 3:
        C = dims[2]
        x3d = np.stack([
            np.clip(x * (0.5 + 0.5 * rng.rand()), 0, 1) for _ in range(C)
        ], axis=-1)
        return x3d.astype(np.float32)
    return x


def add_poisson_noise(y: np.ndarray, peak_photons: float = 1000.0, seed: int = 42) -> np.ndarray:
    """Add Poisson noise to measurements."""
    rng = np.random.RandomState(seed)
    y_scaled = np.clip(y, 0, None) * peak_photons
    noisy = rng.poisson(y_scaled).astype(np.float32) / peak_photons
    return noisy


def add_gaussian_noise(y: np.ndarray, sigma: float = 0.01, seed: int = 42) -> np.ndarray:
    """Add additive Gaussian noise."""
    rng = np.random.RandomState(seed)
    return (y + rng.randn(*y.shape).astype(np.float32) * sigma).astype(np.float32)
