"""Medical MRI k-space category module – shared physics for ~10 MRI modalities.

Handles modalities based on k-space undersampling:
  MRI, fMRI, diffusion_mri, MRA, MRS, asl_mri, mr_fingerprinting, etc.

DAG patterns: F --> S --> D, F --> D, F --> M --> D, M --> F --> S --> D
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# k-space sampling masks
# ---------------------------------------------------------------------------

def cartesian_undersampling_mask(
    shape: Tuple[int, int],
    acceleration: int = 4,
    acs_lines: int = 24,
    seed: int = 42,
) -> np.ndarray:
    """Random Cartesian undersampling mask for k-space.

    Args:
        shape: k-space shape (kH, kW).
        acceleration: Acceleration factor.
        acs_lines: Number of fully-sampled auto-calibration lines at center.
        seed: Random seed.

    Returns:
        Binary mask of shape (kH, kW).
    """
    rng = np.random.RandomState(seed)
    kH, kW = shape
    mask = np.zeros(shape, dtype=np.float32)

    # ACS region (center lines)
    center = kH // 2
    acs_start = center - acs_lines // 2
    acs_end = acs_start + acs_lines
    mask[acs_start:acs_end, :] = 1.0

    # Random lines outside ACS
    remaining = kH - acs_lines
    n_sample = max(1, remaining // acceleration)
    candidates = list(range(0, acs_start)) + list(range(acs_end, kH))
    if candidates:
        chosen = rng.choice(candidates, size=min(n_sample, len(candidates)), replace=False)
        mask[chosen, :] = 1.0

    return mask


def radial_sampling_mask(
    shape: Tuple[int, int],
    n_spokes: int = 50,
) -> np.ndarray:
    """Radial (golden-angle) sampling mask."""
    kH, kW = shape
    mask = np.zeros(shape, dtype=np.float32)
    cy, cx = kH // 2, kW // 2
    golden = np.pi * (3 - np.sqrt(5))  # golden angle ≈ 137.5°

    for s in range(n_spokes):
        angle = s * golden
        for t in np.linspace(-0.5, 0.5, max(kH, kW)):
            y = int(round(cy + t * kH * np.sin(angle)))
            x = int(round(cx + t * kW * np.cos(angle)))
            if 0 <= y < kH and 0 <= x < kW:
                mask[y, x] = 1.0

    return mask


def poisson_disk_mask(
    shape: Tuple[int, int],
    acceleration: int = 4,
    seed: int = 42,
) -> np.ndarray:
    """Poisson-disc variable-density mask (approximate)."""
    rng = np.random.RandomState(seed)
    kH, kW = shape
    # Variable density: higher sampling near center
    yy = np.linspace(-1, 1, kH)
    xx = np.linspace(-1, 1, kW)
    Y, X = np.meshgrid(xx, yy)
    radius = np.sqrt(X ** 2 + Y ** 2)
    prob = np.clip(1.0 / (acceleration * (0.5 + radius)), 0, 1)
    mask = (rng.rand(kH, kW) < prob).astype(np.float32)
    return mask


# ---------------------------------------------------------------------------
# Forward / inverse Fourier
# ---------------------------------------------------------------------------

def image_to_kspace(image: np.ndarray) -> np.ndarray:
    """2-D FFT to k-space."""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image))).astype(np.complex64)


def kspace_to_image(kspace: np.ndarray) -> np.ndarray:
    """Inverse 2-D FFT from k-space."""
    return np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace)))).astype(np.float32)


# ---------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------

def forward(operator, x: np.ndarray, **kwargs) -> np.ndarray:
    """MRI forward model: FFT + undersampling mask.

    Delegates to operator if available, otherwise applies k-space undersampling.
    """
    try:
        y = operator.forward(x)
        if y is not None:
            return np.asarray(y)
    except Exception:
        pass

    # Fallback: undersample k-space
    if x.ndim == 2:
        kspace = image_to_kspace(x)
        mask = cartesian_undersampling_mask(x.shape)
        return np.abs(kspace * mask).astype(np.float32)
    if x.ndim == 3:
        results = []
        for c in range(x.shape[-1]):
            kspace = image_to_kspace(x[..., c])
            mask = cartesian_undersampling_mask(x[..., c].shape)
            results.append(np.abs(kspace * mask).astype(np.float32))
        return np.stack(results, axis=-1)
    return x


def adjoint(operator, y: np.ndarray, **kwargs) -> np.ndarray:
    """MRI adjoint: zero-filled IFFT."""
    try:
        x_hat = operator.adjoint(y)
        if x_hat is not None:
            return np.asarray(x_hat)
    except Exception:
        pass

    if y.ndim == 2:
        return kspace_to_image(y.astype(np.complex64))
    if y.ndim == 3:
        slices = []
        for c in range(y.shape[-1]):
            slices.append(kspace_to_image(y[..., c].astype(np.complex64)))
        return np.stack(slices, axis=-1)
    return y


# ---------------------------------------------------------------------------
# Phantom generation
# ---------------------------------------------------------------------------

def generate_phantom(dims: Tuple[int, ...], seed: int = 42) -> np.ndarray:
    """Generate a brain-like phantom for MRI benchmarks."""
    rng = np.random.RandomState(seed)
    H = dims[0]
    W = dims[1] if len(dims) >= 2 else H
    x = np.zeros((H, W), dtype=np.float32)

    # Outer skull ellipse
    yy = np.linspace(-1, 1, H)
    xx = np.linspace(-1, 1, W)
    X, Y = np.meshgrid(xx, yy)
    skull = (X / 0.8) ** 2 + (Y / 0.95) ** 2 <= 1.0
    x[skull] = 0.2

    # Brain tissue
    brain = (X / 0.7) ** 2 + (Y / 0.85) ** 2 <= 1.0
    x[brain] = 0.8

    # Ventricles
    vent_l = ((X + 0.15) / 0.08) ** 2 + (Y / 0.25) ** 2 <= 1.0
    vent_r = ((X - 0.15) / 0.08) ** 2 + (Y / 0.25) ** 2 <= 1.0
    x[vent_l] = 1.0
    x[vent_r] = 1.0

    # Gray matter patches
    for _ in range(8):
        cx = rng.uniform(-0.5, 0.5)
        cy = rng.uniform(-0.6, 0.6)
        r = rng.uniform(0.05, 0.12)
        patch = ((X - cx) / r) ** 2 + ((Y - cy) / r) ** 2 <= 1.0
        in_brain = brain & patch
        x[in_brain] = rng.uniform(0.5, 0.9)

    if len(dims) >= 3:
        C = dims[2]
        x3d = np.stack([
            np.clip(x + rng.randn(H, W).astype(np.float32) * 0.02, 0, 1)
            for _ in range(C)
        ], axis=-1)
        return x3d.astype(np.float32)
    return x
