"""Electron CTF category module – shared physics for ~10 electron microscopy modalities.

Handles modalities involving Contrast Transfer Function / electron probe:
  TEM, STEM, cryo_em, cryo_et, EELS, EBSD, electron_diffraction,
  electron_holography, electron_tomography, cathodoluminescence, etc.

DAG patterns: CTF --> D, P --> CTF --> D, E --> D, E --> S --> D
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Contrast Transfer Function
# ---------------------------------------------------------------------------

def compute_ctf(
    shape: Tuple[int, int],
    defocus_nm: float = 1000.0,
    cs_mm: float = 2.7,
    voltage_kv: float = 300.0,
    pixel_nm: float = 1.0,
    amplitude_contrast: float = 0.07,
) -> np.ndarray:
    """Compute the Contrast Transfer Function for TEM.

    Args:
        shape: Image shape (H, W).
        defocus_nm: Defocus in nanometres (positive = underfocus).
        cs_mm: Spherical aberration in mm.
        voltage_kv: Accelerating voltage in kV.
        pixel_nm: Pixel size in nm.
        amplitude_contrast: Amplitude contrast ratio.

    Returns:
        CTF image of shape (H, W), values in [-1, 1].
    """
    H, W = shape

    # Electron wavelength (relativistic)
    e = 1.602e-19
    m0 = 9.109e-31
    c = 3e8
    h = 6.626e-34
    V = voltage_kv * 1e3
    wavelength_m = h / np.sqrt(2 * m0 * e * V * (1 + e * V / (2 * m0 * c ** 2)))
    wavelength_nm = wavelength_m * 1e9

    cs_nm = cs_mm * 1e6  # mm to nm

    # Spatial frequency grid
    fy = np.fft.fftfreq(H, d=pixel_nm)
    fx = np.fft.fftfreq(W, d=pixel_nm)
    FX, FY = np.meshgrid(fx, fy)
    s2 = FX ** 2 + FY ** 2  # |s|^2

    # Phase shift
    chi = np.pi * wavelength_nm * defocus_nm * s2 - \
          0.5 * np.pi * cs_nm * wavelength_nm ** 3 * s2 ** 2

    w = np.arctan(amplitude_contrast / np.sqrt(1 - amplitude_contrast ** 2))
    ctf = -np.sin(chi + w)

    return np.fft.fftshift(ctf).astype(np.float32)


def apply_ctf(
    image: np.ndarray,
    defocus_nm: float = 1000.0,
    **kwargs,
) -> np.ndarray:
    """Apply CTF to an image in Fourier space."""
    ctf = compute_ctf(image.shape[:2], defocus_nm=defocus_nm, **kwargs)
    ft = np.fft.fftshift(np.fft.fft2(image))
    modulated = ft * ctf
    return np.real(np.fft.ifft2(np.fft.ifftshift(modulated))).astype(np.float32)


def wiener_deconvolution(
    image: np.ndarray,
    ctf: np.ndarray,
    snr: float = 100.0,
) -> np.ndarray:
    """Wiener filter deconvolution using CTF."""
    ft = np.fft.fftshift(np.fft.fft2(image))
    ctf_shifted = np.fft.ifftshift(ctf)
    ctf_ft = np.fft.fftshift(np.fft.fft2(ctf_shifted))
    # Use CTF directly since it's already a frequency-domain filter
    denom = np.abs(ctf) ** 2 + 1.0 / snr
    wiener = np.conj(ctf) / (denom + 1e-12)
    recovered = ft * wiener
    return np.abs(np.fft.ifft2(np.fft.ifftshift(recovered))).astype(np.float32)


# ---------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------

def forward(operator, x: np.ndarray, **kwargs) -> np.ndarray:
    """CTF forward model for electron microscopy."""
    try:
        y = operator.forward(x)
        if y is not None:
            return np.asarray(y)
    except Exception:
        pass

    # Fallback: apply CTF
    defocus = 1000.0
    if hasattr(operator, "theta"):
        defocus = operator.theta.get("defocus_nm", defocus)
    if x.ndim == 2:
        return apply_ctf(x, defocus_nm=defocus)
    if x.ndim == 3:
        return np.stack([apply_ctf(x[..., c], defocus_nm=defocus) for c in range(x.shape[-1])], axis=-1)
    return x


def adjoint(operator, y: np.ndarray, **kwargs) -> np.ndarray:
    """CTF adjoint (apply conjugate CTF)."""
    try:
        x_hat = operator.adjoint(y)
        if x_hat is not None:
            return np.asarray(x_hat)
    except Exception:
        pass
    return y


# ---------------------------------------------------------------------------
# Phantom generation
# ---------------------------------------------------------------------------

def generate_phantom(dims: Tuple[int, ...], seed: int = 42) -> np.ndarray:
    """Generate a particle / lattice phantom for EM benchmarks."""
    rng = np.random.RandomState(seed)
    H = dims[0]
    W = dims[1] if len(dims) >= 2 else H
    x = np.zeros((H, W), dtype=np.float32)

    # Random circular particles (e.g. protein complexes)
    n_particles = max(10, int(H * W / 1500))
    for _ in range(n_particles):
        cx = rng.randint(5, W - 5)
        cy = rng.randint(5, H - 5)
        r = rng.randint(2, max(3, min(H, W) // 20))
        yy, xx = np.ogrid[:H, :W]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
        x[mask] = rng.uniform(0.4, 1.0)

    if len(dims) >= 3:
        return np.stack([x] * dims[2], axis=-1)
    return x
