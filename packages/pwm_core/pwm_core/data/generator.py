"""pwm_core.data.generator

Backend for ``pwm synthesize`` — generates synthetic benchmark splits.

Public API
----------
    generate_benchmark_split(
        modality: str,
        n_samples: int = 10,
        tier: str = "public",
        base_seed: int = 0,
        out_dir: str | Path = "data/generated",
    ) -> Path

Writes one .npz per sample to ``<out_dir>/<modality>/<tier>/``.
Each .npz contains:
    x_true   – float32 ground-truth image  (H, W) or (H, W, C)
    y        – float32 measurement         (depends on modality)
    params   – dict serialised as JSON string in params_json key
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Seed offsets per tier (mirrors platform convention)
_TIER_SEED_OFFSET = {"public": 0, "dev": 10_000, "hidden": 20_000}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_benchmark_split(
    modality: str,
    n_samples: int = 10,
    tier: str = "public",
    base_seed: int = 0,
    out_dir: str | Path = "data/generated",
) -> Path:
    """Generate a benchmark split for *modality* and write it to disk.

    Parameters
    ----------
    modality:
        Physics modality string (e.g. "ct", "mri", "cassi", "cacti").
    n_samples:
        Number of (x_true, y) pairs to generate.
    tier:
        "public" | "dev" | "hidden"  — controls seed offset.
    base_seed:
        Additional seed offset for determinism.
    out_dir:
        Root output directory; files land in ``<out_dir>/<modality>/<tier>/``.

    Returns
    -------
    Path
        The directory where .npz files were written.
    """
    out_path = Path(out_dir) / modality / tier
    out_path.mkdir(parents=True, exist_ok=True)

    seed_offset = _TIER_SEED_OFFSET.get(tier, 0) + base_seed
    generator_fn = _MODALITY_GENERATORS.get(modality, _generate_generic)

    for i in range(n_samples):
        rng = np.random.default_rng(seed_offset + i)
        x_true, y, params = generator_fn(rng)
        out_file = out_path / f"sample_{i:04d}.npz"
        np.savez_compressed(
            out_file,
            x_true=x_true.astype(np.float32),
            y=y.astype(np.float32),
            params_json=json.dumps(params),
        )
        logger.debug("wrote %s", out_file)

    logger.info("generated %d samples → %s", n_samples, out_path)
    return out_path


# ---------------------------------------------------------------------------
# Per-modality generators
# ---------------------------------------------------------------------------

def _generate_ct(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """CT: Shepp-Logan-like phantom → parallel-beam Radon sinogram + Poisson."""
    from scipy.ndimage import gaussian_filter

    size = 128
    x_true = _shepp_logan_phantom(size, rng)
    # Simple parallel-beam projection (n_angles=60, n_det=182)
    n_angles, n_det = 60, 182
    angles = np.linspace(0, np.pi, n_angles, endpoint=False)
    y = _radon_transform(x_true, angles, n_det)
    # Add Poisson noise (I0=1e5)
    I0 = 1e5
    y_counts = rng.poisson(I0 * np.exp(-y)).astype(np.float64)
    y_noisy = -np.log(np.clip(y_counts, 1, None) / I0)
    params = {"modality": "ct", "n_angles": n_angles, "n_det": n_det, "I0": I0}
    return x_true, y_noisy.astype(np.float32), params


def _generate_mri(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """MRI: brain phantom → undersampled k-space (30% radial mask) + Gaussian noise."""
    size = 128
    x_true = _brain_phantom(size, rng)
    # Random radial k-space mask ~30% sampling
    kspace_full = np.fft.fft2(x_true)
    mask = _radial_mask(size, n_spokes=20, rng=rng)
    kspace_masked = kspace_full * mask
    noise_std = 0.02 * np.abs(kspace_full).max()
    kspace_masked += (rng.standard_normal(kspace_masked.shape) + 1j * rng.standard_normal(kspace_masked.shape)) * noise_std
    # Stack real + imag as (2, H, W)
    y = np.stack([kspace_masked.real, kspace_masked.imag], axis=0)
    params = {"modality": "mri", "acc_factor": int(1 / 0.30), "mask_type": "radial", "n_spokes": 20}
    return x_true, y.astype(np.float32), params


def _generate_cassi(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """CASSI: spectral cube → coded snapshot measurement."""
    H, W, C = 64, 64, 16
    # Synthetic spectral cube
    x_true = _spectral_cube(H, W, C, rng)  # (H, W, C)
    # Random binary coded aperture
    mask = rng.integers(0, 2, size=(H, W)).astype(np.float32)
    # Shift-and-sum: y[h, w + c] += x[h, w, c] * mask[h, w]
    y = np.zeros((H, W + C - 1), dtype=np.float32)
    for c in range(C):
        y[:, c: c + W] += x_true[:, :, c] * mask
    noise = rng.standard_normal(y.shape).astype(np.float32) * 0.01
    y = y + noise
    params = {"modality": "cassi", "H": H, "W": W, "C": C, "mask_density": float(mask.mean())}
    return x_true.reshape(H, W, C).astype(np.float32), y, params


def _generate_generic(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generic: random natural image → Gaussian blur + noise."""
    size = 128
    x_true = rng.random((size, size), dtype=np.float32)
    # Smooth to make it more image-like
    from scipy.ndimage import gaussian_filter
    x_true = gaussian_filter(x_true, sigma=3.0)
    x_true = (x_true - x_true.min()) / (x_true.max() - x_true.min() + 1e-8)
    # Measurement: blur + noise
    sigma_blur = rng.uniform(1.0, 3.0)
    y = gaussian_filter(x_true, sigma=sigma_blur)
    y = y + rng.standard_normal(y.shape).astype(np.float32) * 0.02
    params = {"modality": "generic", "sigma_blur": float(sigma_blur)}
    return x_true, y, params


# Registry
_MODALITY_GENERATORS = {
    "ct": _generate_ct,
    "mri": _generate_mri,
    "cassi": _generate_cassi,
    "cacti": _generate_cassi,  # same compressed sensing structure
}


# ---------------------------------------------------------------------------
# Phantom helpers
# ---------------------------------------------------------------------------

def _shepp_logan_phantom(size: int, rng: np.random.Generator) -> np.ndarray:
    """Simple Shepp-Logan-like phantom."""
    x = np.zeros((size, size), dtype=np.float32)
    cx, cy = size // 2, size // 2
    # Outer ellipse
    for i in range(size):
        for j in range(size):
            xi = (i - cx) / (size * 0.45)
            yj = (j - cy) / (size * 0.35)
            if xi**2 + yj**2 <= 1:
                x[i, j] = 1.0
    # Inner ellipses (simplified)
    for _ in range(4):
        ox = rng.integers(-size // 6, size // 6)
        oy = rng.integers(-size // 6, size // 6)
        rx = rng.uniform(size * 0.05, size * 0.15)
        ry = rng.uniform(size * 0.05, size * 0.15)
        val = rng.uniform(-0.4, 0.4)
        for i in range(size):
            for j in range(size):
                xi = (i - cx - ox) / rx
                yj = (j - cy - oy) / ry
                if xi**2 + yj**2 <= 1:
                    x[i, j] += val
    return np.clip(x, 0, 1)


def _brain_phantom(size: int, rng: np.random.Generator) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    x = rng.random((size, size), dtype=np.float32) * 0.1
    # Add elliptical brain region
    cx, cy = size // 2, size // 2
    for i in range(size):
        for j in range(size):
            xi = (i - cx) / (size * 0.4)
            yj = (j - cy) / (size * 0.45)
            if xi**2 + yj**2 <= 1:
                x[i, j] += 0.5 + rng.random() * 0.3
    return np.clip(gaussian_filter(x, sigma=2.0), 0, 1)


def _spectral_cube(H: int, W: int, C: int, rng: np.random.Generator) -> np.ndarray:
    """Synthetic hyperspectral cube with smooth spatial and spectral structure."""
    from scipy.ndimage import gaussian_filter
    cube = np.zeros((H, W, C), dtype=np.float32)
    for c in range(C):
        layer = rng.random((H, W), dtype=np.float32)
        cube[:, :, c] = gaussian_filter(layer, sigma=4.0)
    # Spectral smoothing
    for h in range(H):
        for w in range(W):
            cube[h, w, :] = np.convolve(cube[h, w, :], np.ones(3) / 3, mode="same")
    return np.clip(cube, 0, 1)


def _radon_transform(x: np.ndarray, angles: np.ndarray, n_det: int) -> np.ndarray:
    """Approximate parallel-beam Radon transform via bilinear interpolation."""
    size = x.shape[0]
    sino = np.zeros((len(angles), n_det), dtype=np.float64)
    t = np.linspace(-1, 1, n_det)
    for ai, angle in enumerate(angles):
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        for di, ti in enumerate(t):
            # Integrate along the perpendicular ray
            s = np.linspace(-1, 1, size * 2)
            xi = ti * cos_a - s * sin_a
            yi = ti * sin_a + s * cos_a
            # Map to pixel coords
            px = (xi + 1) / 2 * (size - 1)
            py = (yi + 1) / 2 * (size - 1)
            valid = (px >= 0) & (px < size - 1) & (py >= 0) & (py < size - 1)
            if valid.any():
                px_v = px[valid]
                py_v = py[valid]
                fx = np.floor(px_v).astype(int)
                fy = np.floor(py_v).astype(int)
                dx = px_v - fx
                dy = py_v - fy
                val = (
                    x[fx, fy] * (1 - dx) * (1 - dy)
                    + x[fx + 1, fy] * dx * (1 - dy)
                    + x[fx, fy + 1] * (1 - dx) * dy
                    + x[fx + 1, fy + 1] * dx * dy
                )
                sino[ai, di] = val.sum() * 2.0 / (size * 2)
    return sino.astype(np.float32)


def _radial_mask(size: int, n_spokes: int, rng: np.random.Generator) -> np.ndarray:
    """Binary radial k-space mask."""
    mask = np.zeros((size, size), dtype=np.float32)
    cx, cy = size // 2, size // 2
    angles = np.linspace(0, np.pi, n_spokes, endpoint=False)
    spoke_w = max(1, size // 64)
    for angle in angles:
        for r in range(-size // 2, size // 2):
            xi = int(cx + r * np.cos(angle))
            yi = int(cy + r * np.sin(angle))
            for dxi in range(-spoke_w, spoke_w + 1):
                for dyi in range(-spoke_w, spoke_w + 1):
                    if 0 <= xi + dxi < size and 0 <= yi + dyi < size:
                        mask[xi + dxi, yi + dyi] = 1.0
    return mask
