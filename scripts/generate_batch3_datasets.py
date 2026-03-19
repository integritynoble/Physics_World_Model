#!/usr/bin/env python3
"""
Generate benchmark datasets for batch 3 modalities:
  ct_fluorescence, cup, dark_field, desi, dexa, dic,
  digital_breast_tomo, dna_paint, ebsd, eddy_current

Each modality: 3 tiers (public=12, dev=20, hidden=20 samples)
Each tier folder: datasets/benchmark/{modality}/{tier}/
  - {modality}_challenge_{tier}.h5  (sample_NN groups with x_true, y, H_ideal, reconstruction_baseline)
  - spec.json
  - true_spec.json
  - images/  (PNG previews)

Seeds: public = 3000 + i*17, dev = 8500 + i*17, hidden = 9700 + i*17
       where i = modality index (0-9)
"""

import json
import os
import sys

import h5py
import numpy as np
from scipy.ndimage import convolve, gaussian_filter, rotate, uniform_filter
from PIL import Image

# ── Constants ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "datasets", "benchmark"
)

TIER_COUNTS = {"public": 12, "dev": 20, "hidden": 20}

MODALITY_INDEX = {
    "ct_fluorescence":     0,
    "cup":                 1,
    "dark_field":          2,
    "desi":                3,
    "dexa":                4,
    "dic":                 5,
    "digital_breast_tomo": 6,
    "dna_paint":           7,
    "ebsd":                8,
    "eddy_current":        9,
}

TIER_BASE_SEEDS = {"public": 3000, "dev": 8500, "hidden": 9700}


def get_rng(modality: str, tier: str, sample_i: int) -> np.random.Generator:
    idx = MODALITY_INDEX[modality]
    base = TIER_BASE_SEEDS[tier] + idx * 17
    return np.random.default_rng(base + sample_i)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_phantom_base(rng, size: int = 128, n_ellipses: int = 6) -> np.ndarray:
    """Multi-ellipse phantom in [0,1]."""
    img = np.zeros((size, size), dtype=np.float32)
    cx, cy = size // 2, size // 2
    Y, X = np.ogrid[:size, :size]
    for _ in range(n_ellipses):
        ey = int(rng.integers(8, size // 3))
        ex = int(rng.integers(8, size // 3))
        y0 = int(rng.integers(cx - size // 3, cx + size // 3))
        x0 = int(rng.integers(cy - size // 3, cy + size // 3))
        val = float(rng.uniform(0.1, 1.0))
        mask = ((X - x0) / ex) ** 2 + ((Y - y0) / ey) ** 2 <= 1
        img[mask] = np.clip(img[mask] + val, 0, 1)
    return img.astype(np.float32)


def radon_simple(img: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    """Simple Radon via scipy.ndimage.rotate — sum projections."""
    size = img.shape[0]
    sino = np.zeros((len(angles_deg), size), dtype=np.float32)
    for k, ang in enumerate(angles_deg):
        rotated = rotate(img, ang, reshape=False, order=1, mode='constant', cval=0.0)
        sino[k] = rotated.sum(axis=0)
    return sino


def fbp_simple(sino: np.ndarray, angles_deg: np.ndarray, size: int) -> np.ndarray:
    """Simple FBP via back-rotate-accumulate."""
    recon = np.zeros((size, size), dtype=np.float32)
    n_angles = len(angles_deg)
    for k, ang in enumerate(angles_deg):
        # Expand sinogram row to full image column
        row = sino[k]  # (size,)
        proj_img = np.tile(row, (size, 1))  # (size, size)
        # Rotate back
        back = rotate(proj_img, -ang, reshape=False, order=1, mode='constant', cval=0.0)
        recon += back
    recon *= np.pi / (2 * n_angles)
    return recon.astype(np.float32)


def save_preview(arr: np.ndarray, path: str) -> None:
    """Save a 2D array as PNG. Handles 3D by taking first channel."""
    if arr.ndim == 3:
        if arr.shape[2] <= 4:
            a = arr[:, :, 0]
        else:
            a = arr[:, :, 0]
    else:
        a = arr
    a = a.astype(np.float32)
    mn, mx = a.min(), a.max()
    if mx > mn:
        a = (a - mn) / (mx - mn)
    else:
        a = np.zeros_like(a)
    img8 = (a * 255).astype(np.uint8)
    Image.fromarray(img8, mode='L').save(path)


# ── Per-modality generators ───────────────────────────────────────────────────

def gen_ct_fluorescence(rng, tier: str, i: int) -> dict:
    """
    x_true: (128,128) fluorescence density map [0,1]
    y: (90,128) sinogram via radon (90 angles 0-180)
    H_ideal: (90,128) noise-free sinogram
    reconstruction_baseline: FBP of y
    """
    size = 128
    angles = np.linspace(0, 180, 90, endpoint=False)
    x_true = make_phantom_base(rng, size, n_ellipses=int(rng.integers(4, 9)))
    # Scale to realistic fluorescence density
    x_true = x_true * float(rng.uniform(0.5, 2.0))
    x_true = x_true.astype(np.float32)

    sino_ideal = radon_simple(x_true, angles)  # (90, 128)
    # Add Poisson noise for measured sinogram
    scale = float(rng.uniform(50.0, 200.0))
    sino_noisy = rng.poisson(np.maximum(sino_ideal * scale, 0)).astype(np.float32) / scale

    recon = fbp_simple(sino_noisy, angles, size)
    recon = np.clip(recon, 0, None).astype(np.float32)

    H_info = sino_ideal.astype(np.float32)

    return {
        "x_true": x_true,
        "y": sino_noisy.astype(np.float32),
        "H_ideal": H_info,
        "reconstruction_baseline": recon,
        "mismatch_params": {"n_angles": 90, "scale": scale},
    }


def gen_cup(rng, tier: str, i: int) -> dict:
    """
    x_true: (128,128,8) temporal video frames
    y: (128,128) compressed measurement (sum of masked frames)
    H_ideal: (128,128,8) random binary masks
    reconstruction_baseline: y replicated across 8 channels
    """
    size = 128
    n_frames = 8

    # Generate smooth base image + temporal drift
    base = rng.random((size, size)).astype(np.float32)
    base = gaussian_filter(base, sigma=8.0).astype(np.float32)
    base = (base - base.min()) / (base.max() - base.min() + 1e-8)

    cube = np.zeros((size, size, n_frames), dtype=np.float32)
    curr = base.copy()
    for t in range(n_frames):
        drift = rng.normal(0, 0.03, (size, size)).astype(np.float32)
        drift = gaussian_filter(drift, sigma=4.0).astype(np.float32)
        curr = np.clip(curr + drift, 0, 1).astype(np.float32)
        cube[:, :, t] = curr

    # Binary masks
    duty = float(rng.uniform(0.4, 0.6))
    masks = (rng.random((size, size, n_frames)) < duty).astype(np.float32)

    # Compressed measurement
    y = np.sum(masks * cube, axis=2).astype(np.float32)

    # Baseline: replicate y across channels
    recon = np.stack([y] * n_frames, axis=2).astype(np.float32)

    return {
        "x_true": cube,
        "y": y,
        "H_ideal": masks,
        "reconstruction_baseline": recon,
        "mismatch_params": {"duty_cycle": duty, "n_frames": n_frames},
    }


def gen_dark_field(rng, tier: str, i: int) -> dict:
    """
    x_true: (128,128) scattering signal map [0,1]
    y: x_true + multiplicative speckle noise
    H_ideal: x_true (noise-free)
    reconstruction_baseline: Lee filter (local mean)
    """
    size = 128
    x_true = make_phantom_base(rng, size, n_ellipses=int(rng.integers(3, 7)))
    x_true = x_true.astype(np.float32)

    # Multiplicative speckle: y = x * (1 + sigma * noise)
    sigma = float(rng.uniform(0.1, 0.4))
    speckle = rng.normal(0, sigma, (size, size)).astype(np.float32)
    y = (x_true * (1.0 + speckle)).astype(np.float32)
    y = np.clip(y, 0, None)

    # Lee filter: local mean as reconstruction
    local_mean = uniform_filter(y, size=7).astype(np.float32)
    recon = local_mean

    return {
        "x_true": x_true,
        "y": y,
        "H_ideal": x_true.copy(),
        "reconstruction_baseline": recon,
        "mismatch_params": {"speckle_sigma": sigma},
    }


def gen_desi(rng, tier: str, i: int) -> dict:
    """
    x_true: (128,128,10) mass-spec ion image (10 m/z channels) [0,1]
    y: (128,128) total ion current (TIC = sum over channels)
    H_ideal: (128,128,10) channel weights (all ones)
    reconstruction_baseline: y replicated across 10 channels
    """
    size = 128
    n_channels = 10
    cube = np.zeros((size, size, n_channels), dtype=np.float32)

    for ch in range(n_channels):
        base = make_phantom_base(rng, size, n_ellipses=int(rng.integers(2, 6)))
        # Different spatial distributions per channel
        scale = float(rng.uniform(0.1, 1.0))
        cube[:, :, ch] = (base * scale).astype(np.float32)

    # Total ion current
    tic = cube.sum(axis=2).astype(np.float32)
    # Add shot noise
    noise_level = float(rng.uniform(0.01, 0.05))
    tic_noisy = tic + rng.normal(0, noise_level * tic.max(), tic.shape).astype(np.float32)
    y = np.clip(tic_noisy, 0, None).astype(np.float32)

    # H_ideal: uniform weights
    H_ideal = np.ones((size, size, n_channels), dtype=np.float32)

    # Baseline: distribute TIC equally across channels
    recon = np.stack([y / n_channels] * n_channels, axis=2).astype(np.float32)

    return {
        "x_true": cube,
        "y": y,
        "H_ideal": H_ideal,
        "reconstruction_baseline": recon,
        "mismatch_params": {"n_channels": n_channels, "noise_level": noise_level},
    }


def gen_dexa(rng, tier: str, i: int) -> dict:
    """
    x_true: (128,128) bone density map [0,1] (higher=denser bone)
    y: (128,128) — projection data reshaped: 1D projections (column sums + noise)
    H_ideal: (128,128) noise-free projection (column sums broadcast)
    reconstruction_baseline: simple backprojection
    """
    size = 128
    x_true = make_phantom_base(rng, size, n_ellipses=int(rng.integers(3, 7)))
    # Bone density: concentrated regions
    x_true = (x_true * float(rng.uniform(0.3, 1.0))).astype(np.float32)

    # Projection: column sums (simplified DXA line integrals)
    proj_ideal = x_true.sum(axis=0)  # (128,) — projection along rows
    noise_std = float(rng.uniform(0.01, 0.05)) * proj_ideal.max()
    proj_noisy = proj_ideal + rng.normal(0, noise_std, proj_ideal.shape).astype(np.float32)

    # Reshape 1D projection to 2D: broadcast to full image shape for consistency
    y = np.tile(proj_noisy, (size, 1)).astype(np.float32)  # (128,128)
    H_ideal = np.tile(proj_ideal, (size, 1)).astype(np.float32)

    # Backprojection: normalize column sums and broadcast
    norm_proj = proj_noisy / (size + 1e-8)
    recon = np.tile(norm_proj, (size, 1)).astype(np.float32)

    return {
        "x_true": x_true,
        "y": y,
        "H_ideal": H_ideal,
        "reconstruction_baseline": recon,
        "mismatch_params": {"projection": "column_sum", "noise_std": float(noise_std)},
    }


def gen_dic(rng, tier: str, i: int) -> dict:
    """
    x_true: (128,128) DIC image (differential interference contrast, phase gradient)
    y: x_true convolved with directional shear kernel + noise
    H_ideal: (3,3) shear kernel stored as flat (9,) array
    reconstruction_baseline: integration via cumsum along shear direction
    """
    size = 128
    # Phase object (smooth)
    phase = rng.random((size, size)).astype(np.float32)
    phase = gaussian_filter(phase, sigma=10.0).astype(np.float32)
    phase = (phase - phase.min()) / (phase.max() - phase.min() + 1e-8)

    # DIC measures phase gradient in shear direction (horizontal)
    x_true = np.gradient(phase, axis=1).astype(np.float32)
    # Normalize
    mx = np.abs(x_true).max()
    if mx > 0:
        x_true = x_true / mx

    # Directional shear kernel (horizontal shear)
    shear_angle = float(rng.uniform(0, 360))
    ang_rad = np.deg2rad(shear_angle)
    kernel = np.array([
        [0, 0, 0],
        [-np.sin(ang_rad), 0, np.sin(ang_rad)],
        [0, 0, 0],
    ], dtype=np.float32) * 0.5 + np.array([
        [0, -1, 0],
        [0, 0, 0],
        [0, 1, 0],
    ], dtype=np.float32) * 0.5

    y_conv = convolve(x_true, kernel, mode='reflect').astype(np.float32)
    noise_std = float(rng.uniform(0.01, 0.05))
    y = (y_conv + rng.normal(0, noise_std, y_conv.shape)).astype(np.float32)

    # Reconstruction via cumulative sum (integrate gradient)
    recon = np.cumsum(y, axis=1).astype(np.float32)
    # Normalize recon
    rmx = np.abs(recon).max()
    if rmx > 0:
        recon = recon / rmx

    H_ideal = kernel.flatten().astype(np.float32)  # store as (9,)

    return {
        "x_true": x_true,
        "y": y,
        "H_ideal": H_ideal,
        "reconstruction_baseline": recon,
        "mismatch_params": {"shear_angle_deg": shear_angle, "noise_std": noise_std},
    }


def gen_digital_breast_tomo(rng, tier: str, i: int) -> dict:
    """
    x_true: (128,128) breast tissue slice [0,1]
    y: (45,128) projections at angles -22.5 to +22.5 degrees (45 views)
    H_ideal: (45,128) noise-free projections
    reconstruction_baseline: FBP
    """
    size = 128
    n_views = 45
    angles = np.linspace(-22.5, 22.5, n_views, endpoint=True)

    # Breast tissue: smooth background + fibroglandular structures
    background = make_phantom_base(rng, size, n_ellipses=1)
    background = gaussian_filter(background, sigma=15.0).astype(np.float32)
    structures = make_phantom_base(rng, size, n_ellipses=int(rng.integers(4, 9)))
    x_true = np.clip(background * 0.4 + structures * 0.6, 0, 1).astype(np.float32)

    sino_ideal = radon_simple(x_true, angles)  # (45, 128)
    # Poisson noise
    scale = float(rng.uniform(100.0, 500.0))
    sino_noisy = rng.poisson(np.maximum(sino_ideal * scale, 0)).astype(np.float32) / scale

    recon = fbp_simple(sino_noisy, angles, size)
    recon = np.clip(recon, 0, None).astype(np.float32)

    return {
        "x_true": x_true,
        "y": sino_noisy.astype(np.float32),
        "H_ideal": sino_ideal.astype(np.float32),
        "reconstruction_baseline": recon,
        "mismatch_params": {
            "n_views": n_views,
            "angle_range_deg": [-22.5, 22.5],
            "scale": scale,
        },
    }


def gen_dna_paint(rng, tier: str, i: int) -> dict:
    """
    x_true: (128,128) PAINT localizations image (sparse, sub-diffraction)
    y: x_true with Poisson noise + uniform background
    H_ideal: (128,128) PSF kernel (Gaussian, stored as image)
    reconstruction_baseline: background-subtracted image
    """
    size = 128
    # Sparse localization map: random bright spots on dark background
    x_true = np.zeros((size, size), dtype=np.float32)
    n_spots = int(rng.integers(10, 50))
    for _ in range(n_spots):
        cy = int(rng.integers(5, size - 5))
        cx = int(rng.integers(5, size - 5))
        intensity = float(rng.uniform(0.3, 1.0))
        x_true[cy, cx] = intensity

    # Convolve with PSF (Gaussian, ~2px sigma)
    psf_sigma = float(rng.uniform(1.5, 3.0))
    x_blurred = gaussian_filter(x_true, sigma=psf_sigma).astype(np.float32)

    # Add Poisson noise + background
    bg_level = float(rng.uniform(0.02, 0.1))
    scale = float(rng.uniform(200.0, 500.0))
    y_raw = rng.poisson(np.maximum((x_blurred + bg_level) * scale, 0)).astype(np.float32) / scale
    y = y_raw.astype(np.float32)

    # PSF stored as H_ideal
    psf_size = 11
    psf_1d = np.exp(-0.5 * (np.arange(psf_size) - psf_size // 2) ** 2 / psf_sigma ** 2)
    psf_2d = np.outer(psf_1d, psf_1d).astype(np.float32)
    psf_2d /= psf_2d.sum()
    # Pad PSF to (128,128) for consistent shape
    H_ideal = np.zeros((size, size), dtype=np.float32)
    off = (size - psf_size) // 2
    H_ideal[off:off + psf_size, off:off + psf_size] = psf_2d

    # Baseline: subtract background level
    recon = np.clip(y - bg_level, 0, None).astype(np.float32)

    return {
        "x_true": x_true,
        "y": y,
        "H_ideal": H_ideal,
        "reconstruction_baseline": recon,
        "mismatch_params": {"psf_sigma_px": psf_sigma, "bg_level": bg_level, "scale": scale},
    }


def gen_ebsd(rng, tier: str, i: int) -> dict:
    """
    x_true: (128,128) grain orientation map [0,360] degrees
    y: x_true + 5 deg random Gaussian noise (clipped to [0,360))
    H_ideal: (128,128) identity / ones (direct measurement)
    reconstruction_baseline: y directly (direct reconstruction)
    """
    size = 128
    # Generate grain structure: Voronoi-like smooth regions
    n_grains = int(rng.integers(8, 25))
    centers_y = rng.integers(0, size, n_grains)
    centers_x = rng.integers(0, size, n_grains)
    orientations = rng.uniform(0, 360, n_grains).astype(np.float32)

    Y, X = np.mgrid[:size, :size]
    x_true = np.zeros((size, size), dtype=np.float32)
    for py in range(size):
        for px in range(size):
            dists = (centers_y - py) ** 2 + (centers_x - px) ** 2
            nearest = np.argmin(dists)
            x_true[py, px] = orientations[nearest]

    # Smooth grain boundaries slightly
    x_true = gaussian_filter(x_true, sigma=0.5).astype(np.float32)
    x_true = x_true % 360.0

    noise_std = 5.0  # degrees
    noise = rng.normal(0, noise_std, (size, size)).astype(np.float32)
    y = ((x_true + noise) % 360.0).astype(np.float32)

    H_ideal = np.ones((size, size), dtype=np.float32)
    recon = y.copy()

    return {
        "x_true": x_true,
        "y": y,
        "H_ideal": H_ideal,
        "reconstruction_baseline": recon,
        "mismatch_params": {"noise_std_deg": noise_std, "n_grains": n_grains},
    }


def gen_eddy_current(rng, tier: str, i: int) -> dict:
    """
    x_true: (128,128) conductivity variation map [0,1]
    y: x_true convolved with exponential kernel + Gaussian noise
    H_ideal: (128,128) exponential PSF kernel (padded)
    reconstruction_baseline: Wiener-like filter (deconvolution approximation)
    """
    size = 128
    x_true = make_phantom_base(rng, size, n_ellipses=int(rng.integers(3, 8)))
    x_true = x_true.astype(np.float32)

    # Exponential kernel (models eddy current diffusion)
    k_size = 15
    k_half = k_size // 2
    decay = float(rng.uniform(1.5, 4.0))
    r = np.arange(k_size) - k_half
    kernel_1d = np.exp(-np.abs(r) / decay).astype(np.float32)
    kernel_2d = np.outer(kernel_1d, kernel_1d).astype(np.float32)
    kernel_2d /= kernel_2d.sum()

    y_conv = convolve(x_true, kernel_2d, mode='reflect').astype(np.float32)
    noise_std = float(rng.uniform(0.01, 0.05))
    y = (y_conv + rng.normal(0, noise_std, y_conv.shape)).astype(np.float32)

    # Wiener filter (frequency domain)
    from numpy.fft import fft2, ifft2, fftshift
    Y_f = fft2(y)
    # Pad kernel to image size
    H_full = np.zeros((size, size), dtype=np.float32)
    off = (size - k_size) // 2
    H_full[off:off + k_size, off:off + k_size] = kernel_2d
    H_f = fft2(H_full)
    snr_est = float(rng.uniform(10.0, 100.0))
    H_conj = np.conj(H_f)
    denom = np.abs(H_f) ** 2 + 1.0 / snr_est
    recon_f = H_conj * Y_f / (denom + 1e-12)
    recon = np.real(ifft2(recon_f)).astype(np.float32)
    recon = np.clip(recon, 0, None).astype(np.float32)

    # H_ideal: store the exponential kernel padded to image size
    H_ideal = H_full.copy()

    return {
        "x_true": x_true,
        "y": y,
        "H_ideal": H_ideal,
        "reconstruction_baseline": recon,
        "mismatch_params": {
            "kernel_decay": decay,
            "noise_std": noise_std,
            "snr_est": snr_est,
        },
    }


# ── Dispatch table ─────────────────────────────────────────────────────────────
GENERATORS = {
    "ct_fluorescence":     gen_ct_fluorescence,
    "cup":                 gen_cup,
    "dark_field":          gen_dark_field,
    "desi":                gen_desi,
    "dexa":                gen_dexa,
    "dic":                 gen_dic,
    "digital_breast_tomo": gen_digital_breast_tomo,
    "dna_paint":           gen_dna_paint,
    "ebsd":                gen_ebsd,
    "eddy_current":        gen_eddy_current,
}

# ── Spec metadata ──────────────────────────────────────────────────────────────
SPECS = {
    "ct_fluorescence": {
        "modality_id": "ct_fluorescence",
        "display_name": "CT Fluorescence Imaging",
        "category": "Nuclear / Optical Hybrid",
        "carrier": "x-ray + fluorescent_photon",
        "forward_model": "y = Poisson(R_theta(x) * scale)",
        "x_shape": [128, 128],
        "y_shape": [90, 128],
        "metrics": ["PSNR", "SSIM"],
        "primary_metric": "PSNR",
    },
    "cup": {
        "modality_id": "cup",
        "display_name": "Compressed Ultrafast Photography (CUP)",
        "category": "Optical / Computational",
        "carrier": "visible_photon",
        "forward_model": "y[i,j] = sum_t mask[i,j,t] * x[i,j,t]",
        "x_shape": [128, 128, 8],
        "y_shape": [128, 128],
        "metrics": ["PSNR", "SSIM"],
        "primary_metric": "PSNR",
    },
    "dark_field": {
        "modality_id": "dark_field",
        "display_name": "X-ray Dark-Field Imaging",
        "category": "X-ray / Scattering",
        "carrier": "x-ray_photon",
        "forward_model": "y = x * (1 + sigma * epsilon), epsilon ~ N(0,1)",
        "x_shape": [128, 128],
        "y_shape": [128, 128],
        "metrics": ["PSNR", "SSIM"],
        "primary_metric": "SSIM",
    },
    "desi": {
        "modality_id": "desi",
        "display_name": "DESI Mass Spectrometry Imaging",
        "category": "Mass Spectrometry Imaging",
        "carrier": "ion",
        "forward_model": "y[i,j] = sum_k x[i,j,k] + noise",
        "x_shape": [128, 128, 10],
        "y_shape": [128, 128],
        "metrics": ["PSNR", "SAM"],
        "primary_metric": "PSNR",
    },
    "dexa": {
        "modality_id": "dexa",
        "display_name": "Dual-Energy X-ray Absorptiometry (DXA/DEXA)",
        "category": "Medical X-ray",
        "carrier": "x-ray_photon",
        "forward_model": "y[:,j] = sum_i x[i,j] + noise (column projections)",
        "x_shape": [128, 128],
        "y_shape": [128, 128],
        "metrics": ["PSNR", "SSIM"],
        "primary_metric": "PSNR",
    },
    "dic": {
        "modality_id": "dic",
        "display_name": "Differential Interference Contrast (DIC) Microscopy",
        "category": "Optical Microscopy",
        "carrier": "visible_photon",
        "forward_model": "y = (shear_kernel * x) + noise",
        "x_shape": [128, 128],
        "y_shape": [128, 128],
        "metrics": ["PSNR", "SSIM"],
        "primary_metric": "SSIM",
    },
    "digital_breast_tomo": {
        "modality_id": "digital_breast_tomo",
        "display_name": "Digital Breast Tomosynthesis (DBT)",
        "category": "Medical X-ray",
        "carrier": "x-ray_photon",
        "forward_model": "y = Poisson(R_{±22.5deg}(x) * scale)",
        "x_shape": [128, 128],
        "y_shape": [45, 128],
        "metrics": ["PSNR", "SSIM"],
        "primary_metric": "PSNR",
    },
    "dna_paint": {
        "modality_id": "dna_paint",
        "display_name": "DNA-PAINT Super-Resolution Microscopy",
        "category": "Optical Super-Resolution",
        "carrier": "visible_photon",
        "forward_model": "y = Poisson((PSF * x + bg) * scale) / scale",
        "x_shape": [128, 128],
        "y_shape": [128, 128],
        "metrics": ["PSNR", "SSIM"],
        "primary_metric": "SSIM",
    },
    "ebsd": {
        "modality_id": "ebsd",
        "display_name": "Electron Backscatter Diffraction (EBSD)",
        "category": "Electron Microscopy",
        "carrier": "electron",
        "forward_model": "y = x + N(0, 5deg^2) [orientation noise]",
        "x_shape": [128, 128],
        "y_shape": [128, 128],
        "metrics": ["MAE", "SSIM"],
        "primary_metric": "MAE",
    },
    "eddy_current": {
        "modality_id": "eddy_current",
        "display_name": "Eddy Current Non-Destructive Testing",
        "category": "Electromagnetic NDT",
        "carrier": "electromagnetic_field",
        "forward_model": "y = (exp_kernel * x) + noise",
        "x_shape": [128, 128],
        "y_shape": [128, 128],
        "metrics": ["PSNR", "SSIM"],
        "primary_metric": "PSNR",
    },
}

TRUE_SPECS = {
    "ct_fluorescence": {
        "n_angles": 90,
        "angle_range_deg": [0, 180],
        "poisson_scale_range": [50, 200],
    },
    "cup": {
        "n_frames": 8,
        "duty_cycle_range": [0.4, 0.6],
        "mask_type": "random_binary",
    },
    "dark_field": {
        "speckle_sigma_range": [0.1, 0.4],
        "noise_model": "multiplicative_gaussian",
    },
    "desi": {
        "n_channels": 10,
        "noise_level_range": [0.01, 0.05],
        "measurement": "total_ion_current",
    },
    "dexa": {
        "projection": "column_sum",
        "noise_model": "additive_gaussian",
        "noise_std_range": [0.01, 0.05],
    },
    "dic": {
        "shear_angle_deg_range": [0, 360],
        "noise_std_range": [0.01, 0.05],
        "kernel": "directional_shear_3x3",
    },
    "digital_breast_tomo": {
        "n_views": 45,
        "angle_range_deg": [-22.5, 22.5],
        "poisson_scale_range": [100, 500],
    },
    "dna_paint": {
        "psf_sigma_px_range": [1.5, 3.0],
        "bg_level_range": [0.02, 0.1],
        "poisson_scale_range": [200, 500],
    },
    "ebsd": {
        "noise_std_deg": 5.0,
        "orientation_range_deg": [0, 360],
        "n_grains_range": [8, 25],
    },
    "eddy_current": {
        "kernel_decay_range": [1.5, 4.0],
        "noise_std_range": [0.01, 0.05],
        "snr_est_range": [10, 100],
    },
}


# ── H5 writer ─────────────────────────────────────────────────────────────────

def write_h5(modality: str, tier: str) -> None:
    n_samples = TIER_COUNTS[tier]
    gen_fn = GENERATORS[modality]

    out_dir = os.path.join(BASE_DIR, modality, tier)
    os.makedirs(out_dir, exist_ok=True)
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    h5_path = os.path.join(out_dir, f"{modality}_challenge_{tier}.h5")

    with h5py.File(h5_path, "w") as hf:
        hf.attrs["modality"] = modality
        hf.attrs["tier"] = tier
        hf.attrs["version"] = "1.0"
        hf.attrs["n_samples"] = n_samples

        for i in range(n_samples):
            rng = get_rng(modality, tier, i)
            data = gen_fn(rng, tier, i)

            grp = hf.create_group(f"sample_{i:02d}")
            grp.create_dataset("x_true",                  data=data["x_true"],                 compression="gzip")
            grp.create_dataset("y",                       data=data["y"],                      compression="gzip")
            grp.create_dataset("H_ideal",                 data=data["H_ideal"],                compression="gzip")
            grp.create_dataset("reconstruction_baseline", data=data["reconstruction_baseline"], compression="gzip")
            grp.attrs["mismatch_params"] = json.dumps(data["mismatch_params"])

            # PNG preview of x_true
            preview_path = os.path.join(img_dir, f"sample_{i:02d}_x_true.png")
            save_preview(data["x_true"], preview_path)

    # Write spec.json
    spec = dict(SPECS[modality])
    spec["tier"] = tier
    spec["n_samples"] = n_samples
    with open(os.path.join(out_dir, "spec.json"), "w") as f:
        json.dump(spec, f, indent=2)

    # Write true_spec.json
    with open(os.path.join(out_dir, "true_spec.json"), "w") as f:
        json.dump(TRUE_SPECS[modality], f, indent=2)

    size_mb = os.path.getsize(h5_path) / 1e6
    print(f"  [{tier:6s}] {h5_path}  ({size_mb:.1f} MB, {n_samples} samples)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    modalities = list(GENERATORS.keys())
    print(f"Generating batch 3 datasets: {len(modalities)} modalities x 3 tiers")
    print(f"Output base: {BASE_DIR}\n")

    results = {}
    for modality in modalities:
        print(f"[{modality}]")
        ok = True
        try:
            for tier in ("public", "dev", "hidden"):
                write_h5(modality, tier)
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            import traceback; traceback.print_exc()
            ok = False
        results[modality] = "OK" if ok else "FAILED"
        print()

    print("=" * 60)
    print("COMPLETION SUMMARY")
    print("=" * 60)
    for mod, status in results.items():
        tag = "PASS" if status == "OK" else "FAIL"
        print(f"  [{tag}] {mod}")
    n_pass = sum(1 for s in results.values() if s == "OK")
    print(f"\n{n_pass}/{len(modalities)} modalities completed successfully.")


if __name__ == "__main__":
    main()
