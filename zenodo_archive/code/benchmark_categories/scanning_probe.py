"""Scanning probe category module – shared physics for ~8 SPM modalities.

Handles modalities involving tip convolution / surface scanning:
  AFM, STM, NSOM, MFM, etc.

DAG patterns: T --> C --> D, T --> D, T --> F --> D, S --> D, S --> M --> D
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.signal import fftconvolve


# ---------------------------------------------------------------------------
# Tip models
# ---------------------------------------------------------------------------

def parabolic_tip(size: int, radius_px: float = 5.0) -> np.ndarray:
    """Parabolic tip shape for AFM dilation model.

    Args:
        size: Kernel side length.
        radius_px: Tip apex radius in pixels.
    """
    c = size // 2
    yy, xx = np.ogrid[:size, :size]
    r2 = (xx - c) ** 2 + (yy - c) ** 2
    tip = r2 / (2 * radius_px)
    tip = np.clip(tip, 0, size)
    # Normalize to [0, 1]
    tip = (tip.max() - tip) / (tip.max() + 1e-12)
    return tip.astype(np.float32)


def conical_tip(size: int, half_angle_deg: float = 15.0) -> np.ndarray:
    """Conical tip shape."""
    c = size // 2
    yy, xx = np.ogrid[:size, :size]
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)
    half_angle_rad = np.radians(half_angle_deg)
    tip_height = r * np.tan(half_angle_rad)
    max_h = tip_height.max()
    tip = np.clip(max_h - tip_height, 0, max_h) / (max_h + 1e-12)
    return tip.astype(np.float32)


def stm_tunneling_kernel(size: int, decay_px: float = 2.0) -> np.ndarray:
    """STM tunneling current kernel (exponential decay)."""
    c = size // 2
    yy, xx = np.ogrid[:size, :size]
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)
    kernel = np.exp(-r / decay_px)
    return (kernel / kernel.sum()).astype(np.float32)


# ---------------------------------------------------------------------------
# AFM dilation model
# ---------------------------------------------------------------------------

def afm_dilation(surface: np.ndarray, tip: np.ndarray) -> np.ndarray:
    """Morphological dilation of surface with tip (AFM imaging model).

    The measured AFM image is the Minkowski sum of the surface
    and the tip shape.
    """
    from scipy.ndimage import maximum_filter
    # Approximate dilation via local maximum with tip footprint
    tip_mask = tip > (tip.max() * 0.1)
    result = maximum_filter(surface, footprint=tip_mask)
    return result.astype(np.float32)


def afm_erosion(image: np.ndarray, tip: np.ndarray) -> np.ndarray:
    """Morphological erosion – inverse of AFM dilation (reconstruction).

    Simplified tip deconvolution.
    """
    from scipy.ndimage import minimum_filter
    tip_mask = tip > (tip.max() * 0.1)
    result = minimum_filter(image, footprint=tip_mask)
    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Piezo model
# ---------------------------------------------------------------------------

def add_piezo_drift(
    image: np.ndarray,
    drift_rate: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    """Add linear + random piezo drift to scanning image."""
    rng = np.random.RandomState(seed)
    H, W = image.shape[:2]
    # Linear drift along slow scan axis
    drift = np.linspace(0, drift_rate * H, H)[:, None]
    # Random jitter
    jitter = rng.randn(H, 1).astype(np.float32) * drift_rate * 0.1
    shifted = image + (drift + jitter).astype(np.float32)
    return shifted.astype(np.float32)


# ---------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------

def forward(operator, x: np.ndarray, **kwargs) -> np.ndarray:
    """SPM forward model: tip convolution / dilation."""
    try:
        y = operator.forward(x)
        if y is not None:
            return np.asarray(y)
    except Exception:
        pass

    # Fallback: tip convolution
    tip = parabolic_tip(11)
    if x.ndim == 2:
        return fftconvolve(x, tip / tip.sum(), mode="same").astype(np.float32)
    return x


def adjoint(operator, y: np.ndarray, **kwargs) -> np.ndarray:
    """SPM adjoint: tip deconvolution."""
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
    """Generate a surface topography phantom for SPM benchmarks."""
    rng = np.random.RandomState(seed)
    H = dims[0]
    W = dims[1] if len(dims) >= 2 else H
    x = np.zeros((H, W), dtype=np.float32)

    # Stepped surface with random features
    # Background terrace
    for i in range(4):
        y_start = i * H // 4
        y_end = (i + 1) * H // 4
        x[y_start:y_end, :] = 0.2 * i

    # Random bumps (nanoparticles on surface)
    n_bumps = max(5, int(H * W / 3000))
    for _ in range(n_bumps):
        cx = rng.randint(3, W - 3)
        cy = rng.randint(3, H - 3)
        r = rng.randint(2, max(3, min(H, W) // 20))
        height = rng.uniform(0.3, 1.0)
        yy, xx = np.ogrid[:H, :W]
        r2 = (xx - cx) ** 2 + (yy - cy) ** 2
        bump = np.clip(1.0 - r2 / (r ** 2 + 1e-6), 0, 1) * height
        x = np.maximum(x, bump)

    if len(dims) >= 3:
        return np.stack([x] * dims[2], axis=-1)
    return np.clip(x, 0, 1).astype(np.float32)
