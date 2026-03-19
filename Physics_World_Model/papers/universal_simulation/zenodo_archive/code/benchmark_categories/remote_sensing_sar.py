"""Remote sensing SAR category module – shared physics for ~10 remote sensing modalities.

Handles modalities involving SAR, phase history, or radio-frequency imaging:
  SAR, InSAR, PolSAR, hyperspectral_remote, multispectral_sat,
  radio_astronomy, radio_interferometry, ocean_color, weather_radar, etc.

DAG patterns: Phi --> R --> D, Phi --> D, Phi --> I --> D, R --> D, F --> S --> D
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# SAR Phase History Model
# ---------------------------------------------------------------------------

def sar_phase_history(
    scene: np.ndarray,
    n_pulses: int = 128,
    n_range_bins: int = 128,
    wavelength_m: float = 0.03,  # X-band
    bandwidth_hz: float = 150e6,
    velocity_ms: float = 200.0,
    seed: int = 42,
) -> np.ndarray:
    """Simulate SAR raw phase history data.

    Simplified spotlight SAR model for benchmark purposes.

    Args:
        scene: 2-D reflectivity scene (H, W).
        n_pulses: Number of azimuth pulses.
        n_range_bins: Number of range samples.

    Returns:
        Phase history data of shape (n_pulses, n_range_bins), complex-valued
        returned as magnitude.
    """
    H, W = scene.shape
    c = 3e8
    range_res = c / (2 * bandwidth_hz)

    # Simplified: DFT-based model
    # Subsample scene to match output dimensions
    from scipy.ndimage import zoom
    if H != n_pulses or W != n_range_bins:
        scene_resized = zoom(scene, (n_pulses / H, n_range_bins / W), order=1)
    else:
        scene_resized = scene.copy()

    # Apply 2-D FFT (raw data ≈ Fourier of scene in spotlight mode)
    raw = np.fft.fft2(scene_resized)
    # Add phase modulation
    rng = np.random.RandomState(seed)
    phase = rng.uniform(-np.pi, np.pi, raw.shape)
    raw = raw * np.exp(1j * phase)

    return np.abs(raw).astype(np.float32)


def sar_image_formation(
    phase_history: np.ndarray,
    algorithm: str = "rda",
) -> np.ndarray:
    """SAR image formation from phase history.

    Args:
        phase_history: Raw data (n_pulses, n_range_bins).
        algorithm: ``"rda"`` (Range-Doppler) or ``"pfa"`` (Polar Format).

    Returns:
        SAR image (magnitude).
    """
    # Range-Doppler Algorithm (simplified)
    # Range compression (IFFT along range)
    range_compressed = np.fft.ifft(phase_history, axis=1)
    # Azimuth compression (IFFT along azimuth)
    image = np.fft.ifft(range_compressed, axis=0)
    return np.abs(image).astype(np.float32)


# ---------------------------------------------------------------------------
# Interferometric SAR
# ---------------------------------------------------------------------------

def insar_phase(
    scene: np.ndarray,
    baseline_m: float = 100.0,
    wavelength_m: float = 0.03,
) -> np.ndarray:
    """Generate InSAR interferometric phase from a DEM-like scene."""
    # Phase proportional to height
    phase = 4 * np.pi * baseline_m * scene / wavelength_m
    return (phase % (2 * np.pi) - np.pi).astype(np.float32)


def phase_unwrap_2d(wrapped: np.ndarray) -> np.ndarray:
    """Simple 2-D phase unwrapping (quality-guided)."""
    try:
        from skimage.restoration import unwrap_phase
        return unwrap_phase(wrapped).astype(np.float32)
    except ImportError:
        # Row-by-row 1-D unwrapping fallback
        unwrapped = np.zeros_like(wrapped)
        for i in range(wrapped.shape[0]):
            unwrapped[i] = np.unwrap(wrapped[i])
        return unwrapped.astype(np.float32)


# ---------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------

def forward(operator, x: np.ndarray, **kwargs) -> np.ndarray:
    """Remote sensing forward model."""
    try:
        y = operator.forward(x)
        if y is not None:
            return np.asarray(y)
    except Exception:
        pass

    # Fallback: SAR phase history
    if x.ndim == 2:
        return sar_phase_history(x)
    if x.ndim == 3:
        # Multi-band: process each band
        return np.stack([sar_phase_history(x[..., c]) for c in range(x.shape[-1])], axis=-1)
    return x


def adjoint(operator, y: np.ndarray, **kwargs) -> np.ndarray:
    """Remote sensing adjoint: image formation."""
    try:
        x_hat = operator.adjoint(y)
        if x_hat is not None:
            return np.asarray(x_hat)
    except Exception:
        pass

    if y.ndim == 2:
        return sar_image_formation(y)
    if y.ndim == 3:
        return np.stack([sar_image_formation(y[..., c]) for c in range(y.shape[-1])], axis=-1)
    return y


# ---------------------------------------------------------------------------
# Phantom generation
# ---------------------------------------------------------------------------

def generate_phantom(dims: Tuple[int, ...], seed: int = 42) -> np.ndarray:
    """Generate a terrain/urban scene phantom for SAR benchmarks."""
    rng = np.random.RandomState(seed)
    H = dims[0]
    W = dims[1] if len(dims) >= 2 else H
    x = np.zeros((H, W), dtype=np.float32)

    # Background terrain (smooth gradient)
    yy = np.linspace(0, 1, H)[:, None]
    xx = np.linspace(0, 1, W)[None, :]
    x += (0.2 * yy + 0.1 * xx).astype(np.float32)

    # Urban structures (bright rectangular scatterers)
    n_buildings = max(3, int(H * W / 5000))
    for _ in range(n_buildings):
        bx = rng.randint(0, W - 10)
        by = rng.randint(0, H - 10)
        bw = rng.randint(3, max(4, W // 15))
        bh = rng.randint(3, max(4, H // 15))
        intensity = rng.uniform(0.6, 1.0)
        x[by:by + bh, bx:bx + bw] = intensity

    # Point scatterers
    n_points = max(5, int(H * W / 2000))
    for _ in range(n_points):
        py = rng.randint(0, H)
        px = rng.randint(0, W)
        x[py, px] = rng.uniform(0.7, 1.0)

    if len(dims) >= 3:
        C = dims[2]
        return np.stack([
            np.clip(x * rng.uniform(0.7, 1.3), 0, 1) for _ in range(C)
        ], axis=-1).astype(np.float32)
    return np.clip(x, 0, 1).astype(np.float32)
