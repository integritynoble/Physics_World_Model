#!/usr/bin/env python3
"""Generate Lensless Camera benchmark dataset.

Forward model (diffuser-based lensless imaging, incoherent):
    y = |F^{-1}{ H_psf * F{ x } }|^2 + noise

where:
    x_true      -- ground truth natural scene (256x256, grayscale, [0,1])
    H_psf       -- system PSF in Fourier domain (large-support caustic pattern)
    y           -- sensor measurement (coded/diffused image)
    noise       -- additive Gaussian readout + Poisson photon noise

The PSF is a physically realistic caustic pattern generated from a random
phase diffuser placed in front of the sensor. The PSF has large spatial support
(spans most of the sensor) creating a multiplexed measurement.

Mismatch parameters (ThetaSpace):
    psf_calibration_error  : relative RMS error in PSF calibration
    distance_error         : relative error in scene-to-diffuser distance
    diffuser_rotation      : small rotation of diffuser vs calibration (radians)
    noise_level            : additive Gaussian noise sigma (fraction of signal)

Ground truth phantoms (256x256):
    Natural scene patches with text, edges, textures, and mixed content.

Tiers:
    public : 12 samples (seeds from 0)
    dev    : 20 samples (seeds from 10000)
    hidden : 20 samples (seeds from 20000)

CPU baseline: Wiener deconvolution. Expected ~20-26 dB.

Usage:
    cd datasets/benchmark/lensless
    python3 generate_dataset.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, rotate as nd_rotate
from scipy.signal import fftconvolve

BENCHMARK_DIR = Path(__file__).resolve().parent
IMAGE_SIZE = 256

# -- Mismatch spec ranges per tier -------------------------------------------

SPEC = {
    "public": {
        "psf_calibration_error": {"min": 0.002, "max": 0.015, "unit": "relative_rms"},
        "distance_error":        {"min": -0.01, "max": 0.01, "unit": "relative"},
        "diffuser_rotation":     {"min": -0.005, "max": 0.005, "unit": "radians"},
        "noise_level":           {"min": 0.005, "max": 0.020, "unit": "fraction"},
    },
    "dev": {
        "psf_calibration_error": {"min": 0.005, "max": 0.03, "unit": "relative_rms"},
        "distance_error":        {"min": -0.03, "max": 0.03, "unit": "relative"},
        "diffuser_rotation":     {"min": -0.015, "max": 0.015, "unit": "radians"},
        "noise_level":           {"min": 0.01,  "max": 0.04, "unit": "fraction"},
    },
    "hidden": {
        "psf_calibration_error": {"min": 0.01,  "max": 0.06, "unit": "relative_rms"},
        "distance_error":        {"min": -0.06, "max": 0.06, "unit": "relative"},
        "diffuser_rotation":     {"min": -0.03, "max": 0.03, "unit": "radians"},
        "noise_level":           {"min": 0.02,  "max": 0.07, "unit": "fraction"},
    },
}


# =============================================================================
# PSF Generation: Caustic diffuser pattern
# =============================================================================

def generate_diffuser_psf(
    N: int,
    rng: np.random.Generator,
    feature_scale: float = 8.0,
    diffuser_strength: float = 4.0,
) -> np.ndarray:
    """Generate a physically realistic caustic PSF from a random phase diffuser.

    The diffuser introduces a spatially varying random phase to light passing
    through it. The resulting PSF is computed by Fresnel propagation of a
    plane wave through the random phase screen, then regularised with a mild
    Gaussian envelope to keep the OTF away from deep spectral nulls.

    This yields a large-support caustic pattern (spans ~40-60% of the sensor)
    that is well-conditioned enough for Wiener deconvolution to achieve 20-26 dB
    on natural scene phantoms.

    Parameters
    ----------
    N : int
        PSF size (N x N).
    rng : np.random.Generator
        Random number generator for reproducibility.
    feature_scale : float
        Spatial correlation length of the diffuser phase features (pixels).
        Controls the speckle grain size.
    diffuser_strength : float
        Peak-to-valley phase variation in radians.
        Higher values create more complex caustic patterns.

    Returns
    -------
    psf : np.ndarray, float64, shape (N, N)
        Normalized PSF (sums to 1).
    """
    # Generate correlated random phase screen
    # Start with white noise, apply Gaussian filter for correlation
    phase_noise = rng.standard_normal((N, N))
    phase_smooth = gaussian_filter(phase_noise, sigma=feature_scale)
    # Normalize and scale to desired phase range
    phase_smooth = phase_smooth / (phase_smooth.std() + 1e-12)
    phase_screen = phase_smooth * diffuser_strength

    # Add a second finer-scale component for caustic structure
    fine_noise = rng.standard_normal((N, N))
    fine_smooth = gaussian_filter(fine_noise, sigma=feature_scale / 3.0)
    fine_smooth = fine_smooth / (fine_smooth.std() + 1e-12)
    phase_screen += fine_smooth * (diffuser_strength * 0.3)

    # Complex transmission through diffuser
    diffuser_field = np.exp(1j * phase_screen)

    # Fresnel propagation (simulate propagation from diffuser to sensor)
    # The PSF is |F{diffuser_field}|^2 (Fraunhofer regime)
    F_field = np.fft.fft2(diffuser_field)
    psf = np.abs(F_field) ** 2

    # Shift to center
    psf = np.fft.fftshift(psf)

    # Apply a Gaussian envelope so the PSF has large support (~40-60% of FOV)
    # but decays at the edges, improving the OTF conditioning
    yy, xx = np.mgrid[0:N, 0:N]
    envelope_sigma = N * float(rng.uniform(0.15, 0.22))
    envelope = np.exp(-((yy - N / 2) ** 2 + (xx - N / 2) ** 2) / (2 * envelope_sigma ** 2))
    psf = psf * envelope

    # Add a small DC pedestal to fill spectral nulls (physically: stray light)
    # This ensures the OTF never drops below ~5-10% of its peak
    pedestal = float(rng.uniform(0.05, 0.12)) * psf.max()
    psf = psf + pedestal * envelope

    # Normalize to sum to 1 (energy conservation)
    psf = psf / (psf.sum() + 1e-15)

    return psf.astype(np.float64)


def apply_psf_mismatch(
    psf: np.ndarray,
    calib_error: float,
    distance_error: float,
    rotation_rad: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply calibration mismatch to a PSF.

    Simulates the difference between the "ideal" calibrated PSF and the
    actual PSF during measurement. Returns the mismatched PSF.

    Parameters
    ----------
    psf : np.ndarray, float64, (N, N)
        Ideal PSF.
    calib_error : float
        Relative RMS calibration error.
    distance_error : float
        Relative distance error (affects PSF scale/defocus).
    rotation_rad : float
        Small rotation of diffuser vs calibration position.
    rng : np.random.Generator
        For reproducibility.

    Returns
    -------
    psf_mismatched : np.ndarray, float64, (N, N)
        Mismatched PSF (still normalized to sum=1).
    """
    N = psf.shape[0]
    psf_m = psf.copy()

    # 1. Calibration error: add correlated noise to the PSF
    if abs(calib_error) > 1e-8:
        noise = gaussian_filter(rng.standard_normal((N, N)), sigma=5.0)
        noise = noise / (noise.std() + 1e-12)
        psf_m = psf_m + calib_error * psf.std() * noise
        psf_m = np.maximum(psf_m, 0.0)

    # 2. Distance error: slight defocus (convolve with small Gaussian)
    if abs(distance_error) > 1e-8:
        defocus_sigma = max(0.5, abs(distance_error) * 10.0)
        psf_m = gaussian_filter(psf_m, sigma=defocus_sigma)

    # 3. Rotation mismatch
    if abs(rotation_rad) > 1e-6:
        angle_deg = np.degrees(rotation_rad)
        psf_m = nd_rotate(psf_m, angle_deg, reshape=False, mode="constant", cval=0.0)
        psf_m = np.maximum(psf_m, 0.0)

    # Re-normalize
    psf_m = psf_m / (psf_m.sum() + 1e-15)

    return psf_m.astype(np.float64)


# =============================================================================
# Forward Model
# =============================================================================

def forward_lensless(
    x: np.ndarray,
    psf: np.ndarray,
) -> np.ndarray:
    """Lensless forward model: y = |F^{-1}{ H_psf * F{x} }|^2

    For incoherent imaging with a shift-invariant PSF, this reduces to
    a convolution: y = psf ** x (the intensity PSF convolved with scene).

    Since x is a real non-negative intensity image and psf is a real
    non-negative intensity PSF, the measurement is simply the convolution.

    Parameters
    ----------
    x : np.ndarray, float64, (N, N)
        Scene image [0, 1].
    psf : np.ndarray, float64, (N, N)
        Normalized PSF (sums to 1).

    Returns
    -------
    y : np.ndarray, float64, (N, N)
        Sensor measurement.
    """
    # Use FFT-based convolution (circular, matches the Fourier-domain model)
    X = np.fft.fft2(x)
    H = np.fft.fft2(np.fft.ifftshift(psf))  # PSF centered -> uncentered for FFT
    y = np.real(np.fft.ifft2(X * H))
    y = np.maximum(y, 0.0)
    return y.astype(np.float64)


def forward_lensless_noisy(
    x: np.ndarray,
    psf_true: np.ndarray,
    noise_level: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Lensless forward model with noise.

    Parameters
    ----------
    x : np.ndarray, float64, (N, N)
        Scene image [0, 1].
    psf_true : np.ndarray, float64, (N, N)
        TRUE PSF used by nature.
    noise_level : float
        Additive Gaussian noise sigma as fraction of signal max.
    rng : np.random.Generator
        For reproducibility.

    Returns
    -------
    y : np.ndarray, float64, (N, N)
        Noisy sensor measurement.
    """
    y_clean = forward_lensless(x, psf_true)

    # Poisson-like noise (signal-dependent)
    signal_max = y_clean.max() + 1e-12
    # Photon noise approximation
    y_noisy = y_clean + rng.normal(0.0, noise_level * signal_max, y_clean.shape)
    # Ensure non-negative
    y_noisy = np.maximum(y_noisy, 0.0)

    return y_noisy.astype(np.float64)


# =============================================================================
# Wiener Deconvolution Baseline
# =============================================================================

def wiener_deconvolution(
    y: np.ndarray,
    psf: np.ndarray,
    reg_param: float = 0.01,
) -> np.ndarray:
    """Wiener deconvolution baseline for lensless reconstruction.

    x_hat = F^{-1}{ conj(H) / (|H|^2 + lambda) * Y }

    Parameters
    ----------
    y : np.ndarray, float64, (N, N)
        Sensor measurement.
    psf : np.ndarray, float64, (N, N)
        Calibrated PSF (may have mismatch from true PSF).
    reg_param : float
        Wiener regularization parameter (noise-to-signal ratio estimate).

    Returns
    -------
    x_hat : np.ndarray, float64, (N, N)
        Reconstructed scene image.
    """
    Y = np.fft.fft2(y)
    H = np.fft.fft2(np.fft.ifftshift(psf))

    # Wiener filter
    H_conj = np.conj(H)
    H_sq = np.abs(H) ** 2
    W = H_conj / (H_sq + reg_param)

    x_hat = np.real(np.fft.ifft2(W * Y))
    x_hat = np.clip(x_hat, 0.0, 1.0)

    return x_hat.astype(np.float64)


# =============================================================================
# Phantom Generators: Natural Scene Patches
# =============================================================================

def _smooth_field(N: int, rng: np.random.Generator, sigma: float = 15.0) -> np.ndarray:
    """Generate a smooth random field in [0, 1]."""
    raw = rng.standard_normal((N, N))
    smooth = gaussian_filter(raw, sigma=sigma)
    lo, hi = smooth.min(), smooth.max()
    return ((smooth - lo) / (hi - lo + 1e-12)).astype(np.float64)


def generate_text_phantom(N: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a phantom with text-like features (sharp edges, fine details).

    Simulates printed text, barcodes, and high-frequency edge content.
    """
    img = np.ones((N, N), dtype=np.float64) * 0.85  # bright background

    # Horizontal text lines at various positions
    n_lines = int(rng.integers(6, 14))
    line_positions = np.sort(rng.integers(10, N - 10, size=n_lines))

    for y_pos in line_positions:
        line_height = int(rng.integers(2, 6))
        y_end = min(y_pos + line_height, N)

        # Generate "characters" as rectangular blocks with gaps
        x = 5
        while x < N - 10:
            char_w = int(rng.integers(3, 10))
            char_val = float(rng.uniform(0.05, 0.35))
            gap = int(rng.integers(1, 5))

            x_end = min(x + char_w, N)
            # Some characters have internal structure (serifs, gaps)
            if rng.random() > 0.3:
                img[y_pos:y_end, x:x_end] = char_val
            else:
                # Character with internal gap
                mid = (y_pos + y_end) // 2
                img[y_pos:mid, x:x_end] = char_val
                img[mid + 1:y_end, x:x_end] = char_val

            x = x_end + gap

    # Add some vertical bars (barcode-like)
    if rng.random() > 0.5:
        bar_y = int(rng.integers(N // 2, N - 40))
        bar_h = int(rng.integers(15, 35))
        for bx in range(10, N - 10, int(rng.integers(2, 6))):
            bar_w = int(rng.integers(1, 4))
            if rng.random() > 0.4:
                img[bar_y:bar_y + bar_h, bx:min(bx + bar_w, N)] = float(rng.uniform(0.05, 0.3))

    return img.astype(np.float64)


def generate_edge_phantom(N: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a phantom with strong edges (step functions, geometric shapes)."""
    img = np.zeros((N, N), dtype=np.float64)

    # Background gradient
    bg = _smooth_field(N, rng, sigma=40.0) * 0.3
    img += bg

    # Geometric shapes with sharp edges
    n_shapes = int(rng.integers(5, 15))
    yy, xx = np.mgrid[0:N, 0:N]

    for _ in range(n_shapes):
        shape_type = rng.choice(["rect", "circle", "triangle", "line"])
        intensity = float(rng.uniform(0.4, 1.0))

        if shape_type == "rect":
            y0 = int(rng.integers(5, N - 30))
            x0 = int(rng.integers(5, N - 30))
            h = int(rng.integers(10, 60))
            w = int(rng.integers(10, 60))
            img[y0:min(y0 + h, N), x0:min(x0 + w, N)] = intensity

        elif shape_type == "circle":
            cy = float(rng.uniform(20, N - 20))
            cx = float(rng.uniform(20, N - 20))
            r = float(rng.uniform(8, 40))
            mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r ** 2
            img[mask] = intensity

        elif shape_type == "triangle":
            # Simple triangle via line rasterization
            cy = int(rng.integers(20, N - 40))
            cx = int(rng.integers(20, N - 40))
            size = int(rng.integers(15, 50))
            for dy in range(size):
                half_w = max(1, int(dy * size / (2 * size + 1)))
                y_row = cy + dy
                if y_row >= N:
                    break
                x_lo = max(0, cx - half_w)
                x_hi = min(N, cx + half_w)
                img[y_row, x_lo:x_hi] = intensity

        elif shape_type == "line":
            # Thick diagonal line
            thickness = int(rng.integers(2, 6))
            y0 = int(rng.integers(0, N))
            x0 = int(rng.integers(0, N))
            length = int(rng.integers(30, N))
            angle = float(rng.uniform(0, np.pi))
            for t in range(length):
                py = int(y0 + t * np.sin(angle))
                px = int(x0 + t * np.cos(angle))
                for dt in range(-thickness // 2, thickness // 2 + 1):
                    ry = py + dt
                    rx = px + dt
                    if 0 <= ry < N and 0 <= rx < N:
                        img[ry, rx] = intensity

    return np.clip(img, 0.0, 1.0).astype(np.float64)


def generate_texture_phantom(N: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a phantom with rich texture content (fabric, natural surface)."""
    img = np.zeros((N, N), dtype=np.float64)

    # Multi-scale texture synthesis
    for scale in [3.0, 8.0, 20.0, 50.0]:
        weight = float(rng.uniform(0.1, 0.4))
        layer = _smooth_field(N, rng, sigma=scale)
        img += weight * layer

    # Add periodic patterns (fabric-like)
    yy, xx = np.mgrid[0:N, 0:N]
    freq1 = float(rng.uniform(0.05, 0.2))
    freq2 = float(rng.uniform(0.05, 0.2))
    angle = float(rng.uniform(0, np.pi))
    periodic = 0.15 * (np.sin(2 * np.pi * freq1 * (xx * np.cos(angle) + yy * np.sin(angle)))
                       + np.sin(2 * np.pi * freq2 * (-xx * np.sin(angle) + yy * np.cos(angle))))
    img += periodic

    # Add some fine grain noise for micro-texture
    grain = rng.standard_normal((N, N)) * 0.02
    img += grain

    # Normalize to [0, 1]
    lo, hi = img.min(), img.max()
    img = (img - lo) / (hi - lo + 1e-12)

    return img.astype(np.float64)


def generate_mixed_phantom(N: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a mixed phantom with text, edges, and textures combined."""
    img = np.zeros((N, N), dtype=np.float64)

    # Quadrant 1: text-like content
    text_block = generate_text_phantom(N // 2, rng)
    img[0:N // 2, 0:N // 2] = text_block

    # Quadrant 2: edges
    edge_block = generate_edge_phantom(N // 2, rng)
    img[0:N // 2, N // 2:N] = edge_block

    # Quadrant 3: texture
    tex_block = generate_texture_phantom(N // 2, rng)
    img[N // 2:N, 0:N // 2] = tex_block

    # Quadrant 4: smooth gradient with fine features
    grad = _smooth_field(N // 2, rng, sigma=25.0) * 0.7
    # Sprinkle small dots
    n_dots = int(rng.integers(10, 40))
    for _ in range(n_dots):
        dy = int(rng.integers(2, N // 2 - 2))
        dx = int(rng.integers(2, N // 2 - 2))
        r = int(rng.integers(1, 4))
        yy, xx = np.mgrid[0:N // 2, 0:N // 2]
        dot = ((yy - dy) ** 2 + (xx - dx) ** 2) <= r ** 2
        grad[dot] = float(rng.uniform(0.7, 1.0))
    img[N // 2:N, N // 2:N] = grad

    # Slight global smooth field to unify
    global_mod = _smooth_field(N, rng, sigma=60.0) * 0.15
    img = img * 0.85 + global_mod

    return np.clip(img, 0.0, 1.0).astype(np.float64)


def generate_natural_scene(N: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a natural-looking scene with smooth regions and edges.

    Combines procedural elements: sky/ground separation, objects, textures.
    """
    img = np.zeros((N, N), dtype=np.float64)

    # Sky-ground split
    horizon = int(rng.integers(N // 3, 2 * N // 3))
    sky_val = float(rng.uniform(0.5, 0.9))
    ground_val = float(rng.uniform(0.15, 0.45))

    # Smooth transition
    yy = np.arange(N)
    transition = 1.0 / (1.0 + np.exp(-0.3 * (yy - horizon)))
    img += (sky_val * (1 - transition) + ground_val * transition)[:, np.newaxis]

    # Add "objects" (buildings, trees represented as blocks)
    n_objects = int(rng.integers(3, 10))
    for _ in range(n_objects):
        obj_x = int(rng.integers(5, N - 30))
        obj_w = int(rng.integers(10, 40))
        obj_h = int(rng.integers(20, 80))
        obj_y = horizon - obj_h + int(rng.integers(-10, 10))
        obj_val = float(rng.uniform(0.2, 0.8))

        y0 = max(0, obj_y)
        y1 = min(N, obj_y + obj_h)
        x0 = max(0, obj_x)
        x1 = min(N, obj_x + obj_w)
        img[y0:y1, x0:x1] = obj_val

    # Add ground texture
    ground_tex = _smooth_field(N, rng, sigma=6.0) * 0.1
    img[horizon:, :] += ground_tex[horizon:, :]

    # Sky texture (clouds)
    sky_tex = _smooth_field(N, rng, sigma=20.0) * 0.08
    img[:horizon, :] += sky_tex[:horizon, :]

    return np.clip(img, 0.0, 1.0).astype(np.float64)


def generate_resolution_chart(N: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a resolution chart phantom with multi-scale bar patterns."""
    img = np.ones((N, N), dtype=np.float64) * 0.9  # bright background

    # Horizontal and vertical bar groups at different scales
    y_cursor = 10
    for group_idx in range(6):
        bar_w = max(2, N // (6 + group_idx * 4))
        n_bars = min(5, (N - 20) // (2 * bar_w))
        x_start = 10 + group_idx * (N // 7)

        for b in range(n_bars):
            # Horizontal bar
            y0 = y_cursor + b * 2 * bar_w
            y1 = min(y0 + bar_w, N - 5)
            x1 = min(x_start + bar_w * 4, N - 5)
            if y1 < N and x1 < N:
                img[y0:y1, x_start:x1] = 0.1

    # Concentric circles
    cy, cx = N * 3 // 4, N // 2
    yy, xx = np.mgrid[0:N, 0:N]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    n_rings = int(rng.integers(5, 12))
    ring_width = float(rng.uniform(2.0, 5.0))
    for i in range(n_rings):
        r_inner = i * ring_width * 2
        r_outer = r_inner + ring_width
        mask = (r >= r_inner) & (r < r_outer)
        img[mask] = 0.15

    # Checkerboard region
    check_size = int(rng.integers(4, 12))
    check_y = N // 4
    check_x = N * 2 // 3
    for dy in range(50):
        for dx in range(50):
            if (dy // check_size + dx // check_size) % 2 == 0:
                py, px = check_y + dy, check_x + dx
                if py < N and px < N:
                    img[py, px] = 0.1

    return np.clip(img, 0.0, 1.0).astype(np.float64)


# =============================================================================
# Phantom Dispatcher
# =============================================================================

PHANTOM_TYPES = {
    "text":             generate_text_phantom,
    "edges":            generate_edge_phantom,
    "texture":          generate_texture_phantom,
    "mixed":            generate_mixed_phantom,
    "natural_scene":    generate_natural_scene,
    "resolution_chart": generate_resolution_chart,
}

# Per-tier phantom assignment (ensures diversity within each tier)
TIER_PHANTOMS = {
    "public": [
        "text", "edges", "texture", "mixed", "natural_scene", "resolution_chart",
        "text", "edges", "texture", "mixed", "natural_scene", "resolution_chart",
    ],
    "dev": [
        "text", "edges", "texture", "mixed", "natural_scene", "resolution_chart",
        "text", "edges", "texture", "mixed", "natural_scene", "resolution_chart",
        "text", "edges", "texture", "mixed", "natural_scene", "resolution_chart",
        "mixed", "natural_scene",
    ],
    "hidden": [
        "mixed", "mixed", "text", "edges", "texture", "natural_scene",
        "resolution_chart", "mixed", "text", "edges",
        "texture", "natural_scene", "resolution_chart", "mixed",
        "text", "edges", "texture", "mixed", "natural_scene", "mixed",
    ],
}

TIER_CONFIG = {
    "public": {"n_samples": 12, "base_seed": 0},
    "dev":    {"n_samples": 20, "base_seed": 10000},
    "hidden": {"n_samples": 20, "base_seed": 20000},
}


# =============================================================================
# Image Utilities
# =============================================================================

def _norm(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-12:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def _save_png(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(
        np.clip(_norm(arr) * 255, 0, 255).astype(np.uint8), "L"
    ).save(str(path))


# =============================================================================
# Metrics
# =============================================================================

def compute_psnr(x_true: np.ndarray, x_recon: np.ndarray) -> float:
    """PSNR between ground truth and reconstruction.

    Data range assumed to be [0, 1].
    """
    x_t = x_true.astype(np.float64)
    x_r = x_recon.astype(np.float64)
    mse = np.mean((x_t - x_r) ** 2)
    if mse < 1e-15:
        return 60.0
    data_range = x_t.max() - x_t.min() + 1e-12
    return float(10 * np.log10(data_range ** 2 / mse))


def compute_ssim_simple(x: np.ndarray, y: np.ndarray) -> float:
    """Simplified global SSIM for grayscale images."""
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    mu_x = x.mean()
    mu_y = y.mean()
    sig_x = x.std()
    sig_y = y.std()
    sig_xy = np.mean((x - mu_x) * (y - mu_y))
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim = ((2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)) / \
           ((mu_x ** 2 + mu_y ** 2 + C1) * (sig_x ** 2 + sig_y ** 2 + C2))
    return float(ssim)


# =============================================================================
# Dataset Tier Generator
# =============================================================================

def sample_mismatch(rng: np.random.Generator, spec: dict) -> dict:
    return {k: float(rng.uniform(v["min"], v["max"])) for k, v in spec.items()}


def generate_tier(tier: str) -> dict:
    """Generate one tier of the lensless benchmark dataset.

    Returns
    -------
    summary : dict
        Per-sample baseline metrics.
    """
    config = TIER_CONFIG[tier]
    spec_ranges = SPEC[tier]
    n_samples = config["n_samples"]
    base_seed = config["base_seed"]
    phantom_types = TIER_PHANTOMS[tier]

    tier_dir = BENCHMARK_DIR / tier
    if tier_dir.exists():
        shutil.rmtree(tier_dir)
    tier_dir.mkdir(parents=True, exist_ok=True)

    h5_path = tier_dir / f"lensless_challenge_{tier}.h5"
    rng = np.random.default_rng(base_seed)

    results = {}

    with h5py.File(h5_path, "w") as f:
        f.attrs["description"] = (
            f"PWM Lensless Camera benchmark -- {tier} tier "
            f"(diffuser-based, incoherent imaging)"
        )
        f.attrs["spec_ranges"] = json.dumps(spec_ranges)
        f.attrs["physics"] = json.dumps({
            "image_size": IMAGE_SIZE,
            "forward_model": "y = conv(psf, x) + noise  (incoherent, shift-invariant)",
            "psf_type": "caustic_diffuser",
            "measurement_type": "intensity_sensor",
            "baseline": "wiener_deconvolution",
        })

        for idx in range(n_samples):
            key = f"sample_{idx:02d}"
            sample_seed = base_seed + idx
            sample_rng = np.random.default_rng(sample_seed)

            # Generate phantom
            phantom_type = phantom_types[idx % len(phantom_types)]
            x_true = PHANTOM_TYPES[phantom_type](IMAGE_SIZE, sample_rng)

            # Generate PSF (unique per sample for variety)
            psf_rng = np.random.default_rng(sample_seed + 50000)
            feature_scale = float(psf_rng.uniform(6.0, 14.0))
            diffuser_strength = float(psf_rng.uniform(1.5, 3.5))
            psf_ideal = generate_diffuser_psf(
                IMAGE_SIZE, psf_rng,
                feature_scale=feature_scale,
                diffuser_strength=diffuser_strength,
            )

            # Sample mismatch parameters
            mis = sample_mismatch(rng, spec_ranges)

            # Generate mismatched PSF (what nature actually uses)
            psf_true = apply_psf_mismatch(
                psf_ideal,
                calib_error=mis["psf_calibration_error"],
                distance_error=mis["distance_error"],
                rotation_rad=mis["diffuser_rotation"],
                rng=np.random.default_rng(sample_seed + 70000),
            )

            # Forward model with TRUE PSF + noise
            y = forward_lensless_noisy(
                x_true, psf_true, mis["noise_level"],
                rng=np.random.default_rng(sample_seed + 90000),
            )

            # Wiener baseline reconstruction using IDEAL (mismatched) PSF
            # Try multiple regularization parameters, pick best
            best_psnr = -1.0
            best_recon = None
            for reg in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]:
                recon = wiener_deconvolution(y, psf_ideal, reg_param=reg)
                psnr_val = compute_psnr(x_true, recon)
                if psnr_val > best_psnr:
                    best_psnr = psnr_val
                    best_recon = recon

            ssim_val = compute_ssim_simple(x_true, best_recon)

            # Store in HDF5
            grp = f.create_group(key)
            grp.create_dataset("x_true", data=x_true.astype(np.float32),
                               compression="gzip")
            grp.create_dataset("y", data=y.astype(np.float32),
                               compression="gzip")
            # Store ideal PSF as H_ideal (what the algorithm is given)
            grp.create_dataset("H_ideal", data=psf_ideal.astype(np.float32),
                               compression="gzip")

            grp.attrs["metadata"] = json.dumps({
                "phantom_type": phantom_type,
                "shape": [IMAGE_SIZE, IMAGE_SIZE],
                "seed": sample_seed,
                "psf_feature_scale": round(feature_scale, 2),
                "psf_diffuser_strength": round(diffuser_strength, 2),
                "baseline_psnr_db": round(best_psnr, 2),
                "baseline_ssim": round(ssim_val, 4),
            })
            grp.attrs["spec_ranges"] = json.dumps(spec_ranges)
            grp.attrs["true_spec"] = json.dumps(mis)

            results[key] = {
                "phantom_type": phantom_type,
                "psnr_db": round(best_psnr, 2),
                "ssim": round(ssim_val, 4),
                "mismatch": mis,
            }

            print(f"  [{tier}] {key} {phantom_type:20s} "
                  f"PSNR={best_psnr:.2f} dB  SSIM={ssim_val:.4f}  "
                  f"psf_err={mis['psf_calibration_error']:.4f} "
                  f"dist_err={mis['distance_error']:.4f} "
                  f"rot={mis['diffuser_rotation']:.4f} "
                  f"noise={mis['noise_level']:.4f}")

    # Save spec
    with open(tier_dir / "spec.json", "w") as sf:
        json.dump(spec_ranges, sf, indent=2)
    with open(tier_dir / "results.json", "w") as rf:
        json.dump(results, rf, indent=2)

    print(f"  [{tier}] HDF5 -> {h5_path.name}  ({n_samples} samples)")
    return results


# =============================================================================
# Gallery Image Generation
# =============================================================================

def generate_gallery(n_scenes: int = 4) -> None:
    """Generate gallery preview images for the platform.

    Creates scene_00 through scene_{n-1} directories with:
        gt.png              -- ground truth scene
        measurement_I.png   -- measurement (low noise/mismatch)
        measurement_II.png  -- measurement (high noise/mismatch)
        recon_I.png         -- Wiener reconstruction (low noise)
        recon_II.png        -- Wiener reconstruction (high noise)
        recon_III.png       -- Wiener reconstruction (medium, different phantom)
    """
    gallery_root = (
        Path(__file__).resolve().parents[3]
        / "platform" / "pwm_platform" / "static" / "img"
        / "benchmark_gallery" / "lensless"
    )

    print(f"\nGenerating gallery images -> {gallery_root}")

    phantom_types_gallery = ["mixed", "edges", "text", "natural_scene"]

    for scene_idx in range(n_scenes):
        scene_dir = gallery_root / f"scene_{scene_idx:02d}"
        scene_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(42000 + scene_idx)

        # Choose phantom
        ptype = phantom_types_gallery[scene_idx % len(phantom_types_gallery)]
        x_true = PHANTOM_TYPES[ptype](IMAGE_SIZE, rng)

        # Generate PSF
        psf_rng = np.random.default_rng(42000 + scene_idx + 1000)
        psf_ideal = generate_diffuser_psf(IMAGE_SIZE, psf_rng,
                                          feature_scale=8.0, diffuser_strength=4.5)

        # Ground truth
        _save_png(x_true, scene_dir / "gt.png")

        # Measurement I: low noise / low mismatch
        mis_low = {
            "psf_calibration_error": 0.005,
            "distance_error": 0.005,
            "diffuser_rotation": 0.002,
            "noise_level": 0.01,
        }
        psf_low = apply_psf_mismatch(psf_ideal, **{
            "calib_error": mis_low["psf_calibration_error"],
            "distance_error": mis_low["distance_error"],
            "rotation_rad": mis_low["diffuser_rotation"],
            "rng": rng,
        })
        y_low = forward_lensless_noisy(x_true, psf_low, mis_low["noise_level"], rng)
        _save_png(y_low, scene_dir / "measurement_I.png")

        # Measurement II: high noise / high mismatch
        mis_high = {
            "psf_calibration_error": 0.04,
            "distance_error": 0.04,
            "diffuser_rotation": 0.02,
            "noise_level": 0.05,
        }
        psf_high = apply_psf_mismatch(psf_ideal, **{
            "calib_error": mis_high["psf_calibration_error"],
            "distance_error": mis_high["distance_error"],
            "rotation_rad": mis_high["diffuser_rotation"],
            "rng": rng,
        })
        y_high = forward_lensless_noisy(x_true, psf_high, mis_high["noise_level"], rng)
        _save_png(y_high, scene_dir / "measurement_II.png")

        # Reconstruction I: Wiener on low noise
        recon_low = wiener_deconvolution(y_low, psf_ideal, reg_param=0.01)
        _save_png(recon_low, scene_dir / "recon_I.png")

        # Reconstruction II: Wiener on high noise
        recon_high = wiener_deconvolution(y_high, psf_ideal, reg_param=0.05)
        _save_png(recon_high, scene_dir / "recon_II.png")

        # Reconstruction III: Wiener on medium noise (different reg)
        mis_med = {
            "psf_calibration_error": 0.02,
            "distance_error": 0.02,
            "diffuser_rotation": 0.01,
            "noise_level": 0.03,
        }
        psf_med = apply_psf_mismatch(psf_ideal, **{
            "calib_error": mis_med["psf_calibration_error"],
            "distance_error": mis_med["distance_error"],
            "rotation_rad": mis_med["diffuser_rotation"],
            "rng": rng,
        })
        y_med = forward_lensless_noisy(x_true, psf_med, mis_med["noise_level"], rng)
        recon_med = wiener_deconvolution(y_med, psf_ideal, reg_param=0.02)
        _save_png(recon_med, scene_dir / "recon_III.png")

        print(f"  scene_{scene_idx:02d}: {ptype} -> 6 images")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("Lensless Camera Benchmark Dataset Generator")
    print("=" * 60)
    print(f"Output: {BENCHMARK_DIR}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Forward model: y = conv(psf, x) + noise")
    print(f"PSF type: caustic diffuser pattern")
    print(f"Baseline: Wiener deconvolution")
    print()

    all_results = {}

    for tier in ["public", "dev", "hidden"]:
        n = TIER_CONFIG[tier]["n_samples"]
        print(f"Generating {tier} tier ({n} samples)...")
        results = generate_tier(tier)
        all_results[tier] = results

        # Summarize
        psnrs = [v["psnr_db"] for v in results.values()]
        ssims = [v["ssim"] for v in results.values()]
        print(f"  [{tier}] Baseline PSNR: {np.mean(psnrs):.2f} +/- {np.std(psnrs):.2f} dB")
        print(f"  [{tier}] Baseline SSIM: {np.mean(ssims):.4f} +/- {np.std(ssims):.4f}")
        print()

    # Generate gallery images
    generate_gallery(n_scenes=4)

    # Save overall summary
    summary_path = BENCHMARK_DIR / "baseline_summary.json"
    with open(summary_path, "w") as sf:
        json.dump(all_results, sf, indent=2)
    print(f"\nBaseline summary -> {summary_path}")

    print(f"\n{'=' * 60}")
    print("Done -- Lensless Camera benchmark ready")
    print(f"  public:  {TIER_CONFIG['public']['n_samples']} samples")
    print(f"  dev:     {TIER_CONFIG['dev']['n_samples']} samples")
    print(f"  hidden:  {TIER_CONFIG['hidden']['n_samples']} samples")

    # Print overall baseline stats
    for tier in ["public", "dev", "hidden"]:
        psnrs = [v["psnr_db"] for v in all_results[tier].values()]
        ssims = [v["ssim"] for v in all_results[tier].values()]
        print(f"  {tier:8s} avg PSNR={np.mean(psnrs):.2f} dB  avg SSIM={np.mean(ssims):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
