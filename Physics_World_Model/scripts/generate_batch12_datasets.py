#!/usr/bin/env python3
"""
Generate benchmark H5 datasets for batch-12 modalities:
  xray_radiography, xrf_imaging, xrf_tomo, ftir_imaging,
  lidar, hyperspectral_remote, nerf, ghost_imaging

Tiers: public (12 samples), dev (20 samples), hidden (20 samples)

Seeds: 1600+i*17 (public), 7600+i*17 (dev), 9650+i*17 (hidden)

Output layout per modality:
  datasets/benchmark/{modality}/{tier}/{modality}_challenge_{tier}.h5
  datasets/benchmark/{modality}/{tier}/spec.json
  datasets/benchmark/{modality}/{tier}/true_spec.json
  datasets/benchmark/{modality}/{tier}/images/sample_NN/{gt,measurement,recon,overview}.png
"""

import json
import os
import sys

import h5py
import numpy as np

try:
    from PIL import Image
except ImportError:
    raise ImportError("pip install Pillow")

try:
    from scipy.ndimage import gaussian_filter
    from scipy.fft import fft2, ifft2, fftshift
except ImportError:
    raise ImportError("pip install scipy")

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
OUT_BASE = os.path.join(ROOT, "datasets", "benchmark")

TIER_SAMPLES = {"public": 12, "dev": 20, "hidden": 20}
SEED_OFFSETS = {"public": 1600, "dev": 7600, "hidden": 9650}

# ── PNG helpers ───────────────────────────────────────────────────────────────

def save_png(arr: np.ndarray, path: str) -> None:
    """Save a 2-D or 3-D float array as 8-bit PNG (auto-scale)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    a = np.asarray(arr, dtype=np.float32)
    lo, hi = a.min(), a.max()
    if hi > lo:
        a = (a - lo) / (hi - lo)
    else:
        a = np.zeros_like(a)
    u8 = (a * 255).clip(0, 255).astype(np.uint8)
    if u8.ndim == 3 and u8.shape[2] == 3:
        Image.fromarray(u8, mode="RGB").save(path)
    elif u8.ndim == 3:
        Image.fromarray(u8[:, :, 0]).save(path)
    else:
        Image.fromarray(u8).save(path)


def side_by_side(arrays, labels=None):
    """Stack up to 4 arrays side by side into one overview RGB image."""
    ars = []
    for a in arrays:
        a = np.asarray(a, dtype=np.float32)
        if a.ndim == 3:
            a = a[:, :, 0]  # take first channel for display
        lo, hi = a.min(), a.max()
        if hi > lo:
            a = (a - lo) / (hi - lo)
        ars.append((a * 255).clip(0, 255).astype(np.uint8))
    h = max(a.shape[0] for a in ars)
    w = max(a.shape[1] for a in ars)
    panels = []
    for a in ars:
        canvas = np.zeros((h, w), dtype=np.uint8)
        canvas[:a.shape[0], :a.shape[1]] = a
        panels.append(canvas)
    combined = np.concatenate(panels, axis=1)
    return Image.fromarray(combined)


# ── smooth phantom helpers ────────────────────────────────────────────────────

def smooth_phantom(shape, rng, n_blobs=6, sigma=15):
    """Create a smooth blob phantom in shape (H, W)."""
    img = np.zeros(shape, dtype=np.float32)
    H, W = shape
    for _ in range(n_blobs):
        cx = rng.integers(W // 4, 3 * W // 4)
        cy = rng.integers(H // 4, 3 * H // 4)
        r = rng.integers(8, min(H, W) // 4)
        amp = rng.uniform(0.3, 1.0)
        yy, xx = np.ogrid[:H, :W]
        img += amp * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * r**2))
    img = gaussian_filter(img, sigma=sigma)
    img = (img - img.min()) / (img.max() - img.min() + 1e-9)
    return img.astype(np.float32)


def fbp_simple(sinogram, angles_deg):
    """Simple filtered back-projection using Ram-Lak filter."""
    from scipy.ndimage import rotate as nd_rotate
    n_angles, n_det = sinogram.shape
    n = n_det

    # Ram-Lak filter in frequency domain
    freq = np.fft.rfftfreq(n)
    filt = np.abs(freq)
    sino_filt = np.zeros_like(sinogram)
    for ia in range(n_angles):
        sino_filt[ia] = np.fft.irfft(np.fft.rfft(sinogram[ia]) * filt, n=n)

    # Back-project
    img = np.zeros((n_det, n_det), dtype=np.float32)
    for ia, angle in enumerate(angles_deg):
        row = sino_filt[ia]
        # Build column image and rotate
        col_img = np.tile(row, (n_det, 1)).astype(np.float32)
        rotated = nd_rotate(col_img, -angle, reshape=False, order=1)
        img += rotated
    img *= np.pi / (2 * n_angles)
    img = np.clip(img, 0, None)
    return img


# ═══════════════════════════════════════════════════════════════════════════════
# MODALITY GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════

# ── 1. xray_radiography ───────────────────────────────────────────────────────

def gen_xray_radiography(i, seed, tier):
    rng = np.random.default_rng(seed)
    H = W = 128
    I0 = 1000.0

    # Ground truth: tissue attenuation coefficient map [0, 1]
    x_true = smooth_phantom((H, W), rng, n_blobs=8, sigma=12).astype(np.float32)
    x_true = x_true * 0.8 + 0.05  # keep in [0.05, 0.85]

    # Forward: Beer-Lambert + Poisson noise
    H_ideal = (np.exp(-x_true) * I0).astype(np.float32)
    lam = H_ideal.astype(np.float64)
    y = rng.poisson(lam).astype(np.float32)
    y = np.clip(y, 1, None)  # avoid log(0)

    # Baseline reconstruction: -log(y / I0)
    recon = -np.log(y / I0).astype(np.float32)

    params = {
        "I0": I0,
        "noise_type": "Poisson",
        "psf_sigma_px": float(rng.uniform(0.5, 1.5)),
        "beam_hardening_coeff": float(rng.uniform(0.0, 0.05)) if tier == "hidden" else 0.0,
    }
    return x_true, y, H_ideal, recon, params


# ── 2. xrf_imaging ────────────────────────────────────────────────────────────

def gen_xrf_imaging(i, seed, tier):
    rng = np.random.default_rng(seed)
    H = W = 128
    N_ELEM = 3

    # Ground truth: 3-channel elemental distribution
    x_true = np.zeros((H, W, N_ELEM), dtype=np.float32)
    for e in range(N_ELEM):
        x_true[:, :, e] = smooth_phantom((H, W), rng, n_blobs=5, sigma=14)
    x_true = (x_true / (x_true.max() + 1e-9)).astype(np.float32)

    # Forward: XRF emission + background + Gaussian noise
    bg_level = float(rng.uniform(0.05, 0.15))
    scale = float(rng.uniform(0.8, 1.2))
    H_ideal = (x_true * scale).astype(np.float32)
    noise_sigma = float(rng.uniform(0.02, 0.06))
    y = (H_ideal + bg_level + rng.normal(0, noise_sigma, H_ideal.shape)).astype(np.float32)
    y = np.clip(y, 0, None)

    # Baseline reconstruction: background subtraction + normalize
    recon = np.clip(y - bg_level, 0, None)
    recon = (recon / (recon.max() + 1e-9)).astype(np.float32)

    params = {
        "n_elements": N_ELEM,
        "bg_level": bg_level,
        "scale": scale,
        "noise_sigma": noise_sigma,
        "detector_efficiency": float(rng.uniform(0.8, 1.0)),
    }
    return x_true, y, H_ideal, recon, params


# ── 3. xrf_tomo ───────────────────────────────────────────────────────────────

def gen_xrf_tomo(i, seed, tier):
    rng = np.random.default_rng(seed)
    SZ = 64
    N_ELEM = 3
    N_ANGLES = 45

    # Ground truth: (64, 64, 3) XRF volume
    x_true = np.zeros((SZ, SZ, N_ELEM), dtype=np.float32)
    for e in range(N_ELEM):
        x_true[:, :, e] = smooth_phantom((SZ, SZ), rng, n_blobs=4, sigma=8)

    angles = np.linspace(0, 180, N_ANGLES, endpoint=False)

    # Forward: Radon projection per element
    H_ideal = np.zeros((N_ANGLES, SZ, N_ELEM), dtype=np.float32)
    for e in range(N_ELEM):
        slice_e = x_true[:, :, e]
        for ia, ang in enumerate(angles):
            from scipy.ndimage import rotate as nd_rotate
            rotated = nd_rotate(slice_e, ang, reshape=False, order=1)
            H_ideal[ia, :, e] = rotated.sum(axis=0)

    noise_sigma = float(rng.uniform(0.01, 0.03))
    y = (H_ideal + rng.normal(0, noise_sigma, H_ideal.shape)).astype(np.float32)

    # Baseline: FBP per element
    recon = np.zeros((SZ, SZ, N_ELEM), dtype=np.float32)
    for e in range(N_ELEM):
        sino_e = y[:, :, e]
        recon[:, :, e] = fbp_simple(sino_e, angles)
    recon = np.clip(recon, 0, None).astype(np.float32)

    # Normalize recon to [0,1]
    mx = recon.max()
    if mx > 0:
        recon /= mx

    params = {
        "n_elements": N_ELEM,
        "n_angles": N_ANGLES,
        "angle_range_deg": [0.0, 180.0],
        "noise_sigma": noise_sigma,
        "self_absorption": float(rng.uniform(0.0, 0.05)) if tier == "hidden" else 0.0,
    }
    return x_true, y, H_ideal, recon, params


# ── 4. ftir_imaging ───────────────────────────────────────────────────────────

def gen_ftir_imaging(i, seed, tier):
    rng = np.random.default_rng(seed)
    H = W = 64
    N_BANDS = 10

    # Ground truth: (64, 64, 10) IR absorption cube
    x_true = np.zeros((H, W, N_BANDS), dtype=np.float32)
    # Each spatial region has a characteristic spectral signature
    base = smooth_phantom((H, W), rng, n_blobs=4, sigma=8)
    for b in range(N_BANDS):
        spectral_weight = rng.uniform(0.1, 1.0)
        spatial_variation = smooth_phantom((H, W), rng, n_blobs=2, sigma=6) * 0.3
        x_true[:, :, b] = np.clip(base * spectral_weight + spatial_variation, 0, 1)

    # Forward: Beer-Lambert + baseline drift + noise
    baseline_drift = np.linspace(0.0, float(rng.uniform(0.02, 0.08)), N_BANDS)
    noise_sigma = float(rng.uniform(0.005, 0.02))
    H_ideal = x_true + baseline_drift[np.newaxis, np.newaxis, :]
    y = (H_ideal + rng.normal(0, noise_sigma, H_ideal.shape)).astype(np.float32)

    # Baseline reconstruction: normalize per-band
    recon = np.zeros_like(y)
    for b in range(N_BANDS):
        band = y[:, :, b] - y[:, :, b].min()
        band_max = band.max()
        recon[:, :, b] = band / (band_max + 1e-9)
    recon = recon.astype(np.float32)

    params = {
        "n_bands": N_BANDS,
        "baseline_drift_max": float(baseline_drift[-1]),
        "noise_sigma": noise_sigma,
        "apodization": "Blackman-Harris" if tier == "hidden" else "boxcar",
        "spectral_resolution_cm-1": float(rng.uniform(4.0, 8.0)),
    }
    return x_true, y, H_ideal, recon, params


# ── 5. lidar ──────────────────────────────────────────────────────────────────

def gen_lidar(i, seed, tier):
    rng = np.random.default_rng(seed)
    H = W = 128

    # Ground truth: depth image (meters, 1–50 m range)
    x_true = smooth_phantom((H, W), rng, n_blobs=6, sigma=16).astype(np.float32)
    x_true = (x_true * 49.0 + 1.0)  # scale to 1–50 m

    # Forward: LiDAR range measurement with range-dependent noise
    range_noise_scale = float(rng.uniform(0.02, 0.05))  # 2-5 cm per meter
    noise = (x_true * range_noise_scale *
             rng.standard_normal((H, W))).astype(np.float32)
    H_ideal = x_true.copy().astype(np.float32)  # noise-free range image
    y = (x_true + noise).astype(np.float32)

    # Baseline: direct (noisy) measurement
    recon = y.copy()

    params = {
        "range_min_m": 1.0,
        "range_max_m": 50.0,
        "range_noise_scale": range_noise_scale,
        "beam_divergence_mrad": float(rng.uniform(0.5, 2.0)),
        "multi_return": tier == "hidden",
    }
    return x_true, y, H_ideal, recon, params


# ── 6. hyperspectral_remote ───────────────────────────────────────────────────

def gen_hyperspectral_remote(i, seed, tier):
    rng = np.random.default_rng(seed)
    H = W = 64
    N_BANDS = 20

    # Ground truth: (64, 64, 20) reflectance cube [0, 1]
    x_true = np.zeros((H, W, N_BANDS), dtype=np.float32)
    # Mix of 3 endmembers (vegetation, soil, water)
    em_maps = [smooth_phantom((H, W), rng, n_blobs=3, sigma=10) for _ in range(3)]
    total = sum(em_maps)
    abundances = [m / (total + 1e-9) for m in em_maps]

    endmember_spectra = np.array([
        np.sin(np.linspace(0, np.pi, N_BANDS)) * 0.6 + 0.1,    # vegetation
        np.linspace(0.3, 0.5, N_BANDS),                           # soil
        np.exp(-np.linspace(0, 3, N_BANDS)) * 0.3,               # water
    ], dtype=np.float32)

    for b in range(N_BANDS):
        x_true[:, :, b] = sum(
            abundances[k] * endmember_spectra[k, b] for k in range(3)
        ).astype(np.float32)

    # Forward: atmospheric path radiance + multiplicative solar irradiance + noise
    path_radiance = float(rng.uniform(0.02, 0.08))
    solar_scale = float(rng.uniform(0.7, 1.3))
    noise_sigma = float(rng.uniform(0.005, 0.015))
    H_ideal = (x_true * solar_scale + path_radiance).astype(np.float32)
    y = (H_ideal + rng.normal(0, noise_sigma, H_ideal.shape)).astype(np.float32)
    y = np.clip(y, 0, None)

    # Baseline: empirical line correction (subtract path, divide by solar)
    recon = ((y - path_radiance) / solar_scale).astype(np.float32)
    recon = np.clip(recon, 0, 1)

    params = {
        "n_bands": N_BANDS,
        "path_radiance": path_radiance,
        "solar_scale": solar_scale,
        "noise_sigma": noise_sigma,
        "adjacency_effect": tier == "hidden",
        "view_zenith_deg": float(rng.uniform(0, 30)),
    }
    return x_true, y, H_ideal, recon, params


# ── 7. nerf ───────────────────────────────────────────────────────────────────

def gen_nerf(i, seed, tier):
    rng = np.random.default_rng(seed)
    H = W = 128
    N_INPUT_VIEWS = 8

    # Novel-view ground truth: RGB (H, W, 3)
    x_true = np.zeros((H, W, 3), dtype=np.float32)
    for c in range(3):
        x_true[:, :, c] = smooth_phantom((H, W), rng, n_blobs=5, sigma=14)

    # y: 8 input views (H, W, 3) each with slight rotation/shift
    y = np.zeros((N_INPUT_VIEWS, H, W, 3), dtype=np.float32)
    for v in range(N_INPUT_VIEWS):
        view_rng = np.random.default_rng(seed + v + 1)
        shift_x = int(view_rng.integers(-8, 9))
        shift_y = int(view_rng.integers(-8, 9))
        brightness = float(view_rng.uniform(0.85, 1.15))
        for c in range(3):
            ch = np.roll(np.roll(x_true[:, :, c], shift_y, axis=0), shift_x, axis=1)
            ch = ch * brightness + view_rng.normal(0, 0.02, ch.shape)
            y[v, :, :, c] = np.clip(ch, 0, 1).astype(np.float32)

    # H_ideal: pose matrix (8, 4, 4) for the 8 input views
    H_ideal = np.zeros((N_INPUT_VIEWS, 4, 4), dtype=np.float32)
    for v in range(N_INPUT_VIEWS):
        angle = 2 * np.pi * v / N_INPUT_VIEWS
        H_ideal[v] = np.eye(4, dtype=np.float32)
        H_ideal[v, 0, 3] = np.cos(angle)
        H_ideal[v, 2, 3] = np.sin(angle)

    # Baseline reconstruction: weighted average of input views
    recon = y.mean(axis=0).astype(np.float32)

    params = {
        "n_input_views": N_INPUT_VIEWS,
        "img_size": H,
        "fov_deg": float(rng.uniform(40, 70)),
        "view_shift_max_px": 8,
        "brightness_jitter": [0.85, 1.15],
        "pose_noise": tier == "hidden",
    }
    return x_true, y, H_ideal, recon, params


# ── 8. ghost_imaging ──────────────────────────────────────────────────────────

def gen_ghost_imaging(i, seed, tier):
    rng = np.random.default_rng(seed)
    SZ = 64
    N_PIXELS = SZ * SZ   # 4096
    N_MEASUREMENTS = 1000

    # Ground truth: binary-ish object (64, 64)
    phantom = smooth_phantom((SZ, SZ), rng, n_blobs=4, sigma=8)
    x_true = (phantom > phantom.mean()).astype(np.float32)

    # Random illumination patterns: H_ideal (1000, 4096)
    H_patterns = rng.standard_normal((N_MEASUREMENTS, N_PIXELS)).astype(np.float32)
    H_patterns = (H_patterns > 0).astype(np.float32)  # binary patterns

    # Bucket detector signal: inner product of pattern with object + noise
    x_flat = x_true.ravel()
    y_clean = H_patterns @ x_flat      # (1000,)
    noise_level = float(rng.uniform(0.02, 0.08)) * float(y_clean.std() + 1e-9)
    y = (y_clean + rng.normal(0, noise_level, y_clean.shape)).astype(np.float32)

    # Baseline: differential ghost imaging
    y_mean = y.mean()
    recon_flat = H_patterns.T @ (y - y_mean)    # (4096,)
    recon = recon_flat.reshape(SZ, SZ).astype(np.float32)
    recon_mn = recon.min()
    recon_mx = recon.max()
    if recon_mx > recon_mn:
        recon = (recon - recon_mn) / (recon_mx - recon_mn)

    params = {
        "n_measurements": N_MEASUREMENTS,
        "n_pixels": N_PIXELS,
        "img_size": SZ,
        "pattern_type": "binary_random",
        "noise_level": float(noise_level),
        "compression_ratio": float(N_MEASUREMENTS / N_PIXELS),
        "thermal_noise": tier == "hidden",
    }
    return x_true, y, H_patterns, recon, params


# ═══════════════════════════════════════════════════════════════════════════════
# MODALITY REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

MODALITIES = {
    "xray_radiography": {
        "gen_fn": gen_xray_radiography,
        "spec": {
            "I0": {"min": 800, "max": 1200, "unit": "photons"},
            "psf_sigma_px": {"min": 0.5, "max": 2.0, "unit": "px"},
            "beam_hardening_coeff": {"min": 0.0, "max": 0.1, "unit": ""},
        },
        "description": "X-ray radiography: Beer-Lambert attenuation with Poisson noise",
        "x_shape": "(128, 128) tissue attenuation map",
        "y_shape": "(128, 128) measured X-ray intensity",
        "H_shape": "(128, 128) noise-free X-ray intensity",
    },
    "xrf_imaging": {
        "gen_fn": gen_xrf_imaging,
        "spec": {
            "bg_level": {"min": 0.05, "max": 0.15, "unit": ""},
            "noise_sigma": {"min": 0.02, "max": 0.06, "unit": ""},
            "scale": {"min": 0.8, "max": 1.2, "unit": ""},
            "detector_efficiency": {"min": 0.8, "max": 1.0, "unit": ""},
        },
        "description": "X-ray fluorescence imaging: 3-element spatial distribution",
        "x_shape": "(128, 128, 3) elemental map (3 elements)",
        "y_shape": "(128, 128, 3) measured XRF spectra with background",
        "H_shape": "(128, 128, 3) noise-free XRF signal",
    },
    "xrf_tomo": {
        "gen_fn": gen_xrf_tomo,
        "spec": {
            "n_angles": {"min": 45, "max": 45, "unit": ""},
            "noise_sigma": {"min": 0.01, "max": 0.03, "unit": ""},
            "self_absorption": {"min": 0.0, "max": 0.05, "unit": ""},
        },
        "description": "XRF tomography: FBP reconstruction of 3-element 2D map",
        "x_shape": "(64, 64, 3) XRF volume (3 elements)",
        "y_shape": "(45, 64, 3) sinogram projections per element",
        "H_shape": "(45, 64, 3) noise-free sinograms",
    },
    "ftir_imaging": {
        "gen_fn": gen_ftir_imaging,
        "spec": {
            "n_bands": {"min": 10, "max": 10, "unit": ""},
            "noise_sigma": {"min": 0.005, "max": 0.02, "unit": ""},
            "baseline_drift_max": {"min": 0.02, "max": 0.08, "unit": "AU"},
            "spectral_resolution_cm-1": {"min": 4.0, "max": 8.0, "unit": "cm-1"},
        },
        "description": "FTIR imaging: 10-band IR absorption with baseline drift",
        "x_shape": "(64, 64, 10) IR absorption cube (10 wavenumbers)",
        "y_shape": "(64, 64, 10) measured absorbance with baseline + noise",
        "H_shape": "(64, 64, 10) noise-free absorbance with baseline",
    },
    "lidar": {
        "gen_fn": gen_lidar,
        "spec": {
            "range_min_m": {"min": 1.0, "max": 1.0, "unit": "m"},
            "range_max_m": {"min": 50.0, "max": 50.0, "unit": "m"},
            "range_noise_scale": {"min": 0.02, "max": 0.05, "unit": "m/m"},
            "beam_divergence_mrad": {"min": 0.5, "max": 2.0, "unit": "mrad"},
        },
        "description": "LiDAR depth imaging: range-dependent noise model",
        "x_shape": "(128, 128) true depth image (m)",
        "y_shape": "(128, 128) noisy LiDAR range image (m)",
        "H_shape": "(128, 128) noise-free range image (m)",
    },
    "hyperspectral_remote": {
        "gen_fn": gen_hyperspectral_remote,
        "spec": {
            "n_bands": {"min": 20, "max": 20, "unit": ""},
            "path_radiance": {"min": 0.02, "max": 0.08, "unit": ""},
            "solar_scale": {"min": 0.7, "max": 1.3, "unit": ""},
            "noise_sigma": {"min": 0.005, "max": 0.015, "unit": ""},
            "view_zenith_deg": {"min": 0.0, "max": 30.0, "unit": "deg"},
        },
        "description": "Hyperspectral remote sensing: 20-band at-sensor radiance",
        "x_shape": "(64, 64, 20) surface reflectance cube (20 bands)",
        "y_shape": "(64, 64, 20) at-sensor radiance with atmospheric path",
        "H_shape": "(64, 64, 20) noise-free at-sensor radiance",
    },
    "nerf": {
        "gen_fn": gen_nerf,
        "spec": {
            "n_input_views": {"min": 8, "max": 8, "unit": ""},
            "fov_deg": {"min": 40.0, "max": 70.0, "unit": "deg"},
            "view_shift_max_px": {"min": 8, "max": 8, "unit": "px"},
            "brightness_jitter_min": {"min": 0.85, "max": 0.85, "unit": ""},
            "brightness_jitter_max": {"min": 1.15, "max": 1.15, "unit": ""},
        },
        "description": "NeRF: novel view synthesis from 8 input views",
        "x_shape": "(128, 128, 3) novel view RGB image",
        "y_shape": "(8, 128, 128, 3) 8 input views",
        "H_shape": "(8, 4, 4) camera pose matrices",
    },
    "ghost_imaging": {
        "gen_fn": gen_ghost_imaging,
        "spec": {
            "n_measurements": {"min": 1000, "max": 1000, "unit": ""},
            "compression_ratio": {"min": 0.24, "max": 0.25, "unit": ""},
            "noise_level": {"min": 0.02, "max": 0.08, "unit": "fraction"},
            "pattern_type": {"values": "binary_random"},
        },
        "description": "Ghost imaging: 1000 bucket measurements, DGI reconstruction",
        "x_shape": "(64, 64) binary object",
        "y_shape": "(1000,) bucket detector measurements",
        "H_shape": "(1000, 4096) random illumination patterns",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET WRITER
# ═══════════════════════════════════════════════════════════════════════════════

def write_tier(modality, gen_fn, out_dir, tier):
    n_samples = TIER_SAMPLES[tier]
    seed_base = SEED_OFFSETS[tier]

    tier_dir = os.path.join(out_dir, tier)
    os.makedirs(tier_dir, exist_ok=True)

    h5_path = os.path.join(tier_dir, f"{modality}_challenge_{tier}.h5")
    img_base = os.path.join(tier_dir, "images")

    true_spec_data = {}

    print(f"  [{tier}] Writing {n_samples} samples -> {h5_path}")

    with h5py.File(h5_path, "w") as hf:
        hf.attrs.update({
            "modality": modality,
            "tier": tier,
            "version": "1.0",
            "n_samples": n_samples,
        })

        for i in range(n_samples):
            seed = seed_base + i * 17
            x_true, y, H_ideal, recon, params = gen_fn(i, seed, tier)

            grp = hf.create_group(f"sample_{i:02d}")
            grp.create_dataset("x_true",                 data=x_true.astype(np.float32), compression="gzip")
            grp.create_dataset("y",                      data=y.astype(np.float32),      compression="gzip")
            grp.create_dataset("H_ideal",                data=H_ideal.astype(np.float32),compression="gzip")
            grp.create_dataset("reconstruction_baseline",data=recon.astype(np.float32),  compression="gzip")
            grp.attrs["sample_params"] = json.dumps(params)

            true_spec_data[f"sample_{i:02d}"] = params

            # ── PNG previews ───────────────────────────────────────────────
            img_dir = os.path.join(img_base, f"sample_{i:02d}")
            os.makedirs(img_dir, exist_ok=True)

            # x_true preview
            if x_true.ndim == 3:
                save_png(x_true[:, :, 0], os.path.join(img_dir, "gt.png"))
            else:
                save_png(x_true, os.path.join(img_dir, "gt.png"))

            # y (measurement) preview
            if y.ndim == 4:
                # nerf: (N_views, H, W, 3) — show first view
                save_png(y[0], os.path.join(img_dir, "measurement.png"))
            elif y.ndim == 3:
                save_png(y[:, :, 0], os.path.join(img_dir, "measurement.png"))
            elif y.ndim == 2:
                save_png(y, os.path.join(img_dir, "measurement.png"))
            else:
                # 1D (ghost imaging): show as 1D signal strip
                row = y.reshape(1, -1)
                row_img = row / (row.max() + 1e-9)
                strip = (np.tile(row_img, (32, 1)) * 255).astype(np.uint8)
                Image.fromarray(strip).save(os.path.join(img_dir, "measurement.png"))

            # recon preview
            if recon.ndim == 3:
                save_png(recon[:, :, 0], os.path.join(img_dir, "recon.png"))
            else:
                save_png(recon, os.path.join(img_dir, "recon.png"))

            # H_ideal preview
            if H_ideal.ndim == 4:
                save_png(H_ideal[0], os.path.join(img_dir, "measurement_noisy.png"))
            elif H_ideal.ndim == 3:
                save_png(H_ideal[:, :, 0], os.path.join(img_dir, "measurement_noisy.png"))
            elif H_ideal.ndim == 2:
                save_png(H_ideal, os.path.join(img_dir, "measurement_noisy.png"))
            else:
                # 2D (ghost_imaging: patterns shape 1000x4096 — skip, save placeholder)
                pmat = H_ideal[:50, :64].reshape(50, 64)
                save_png(pmat, os.path.join(img_dir, "measurement_noisy.png"))

            # overview
            if x_true.ndim >= 2 and recon.ndim >= 2:
                x_disp = x_true if x_true.ndim == 2 else x_true[:, :, 0]
                r_disp = recon if recon.ndim == 2 else recon[:, :, 0]
                if y.ndim == 4:
                    y_disp = y[0, :, :, 0]
                elif y.ndim == 3:
                    y_disp = y[:, :, 0]
                elif y.ndim == 2:
                    y_disp = y
                else:
                    y_disp = np.zeros_like(x_disp)
                ov = side_by_side([x_disp, y_disp, r_disp])
                ov.save(os.path.join(img_dir, "overview.png"))
            else:
                Image.fromarray(np.zeros((64, 64), dtype=np.uint8)).save(
                    os.path.join(img_dir, "overview.png"))

            # per-sample spec.json
            with open(os.path.join(img_dir, "spec.json"), "w") as f:
                json.dump(params, f, indent=2)

    # ── Write tier-level JSONs ─────────────────────────────────────────────
    mod_cfg = MODALITIES[modality]
    spec = {
        "modality": modality,
        "tier": tier,
        "description": mod_cfg["description"],
        "x_shape": mod_cfg["x_shape"],
        "y_shape": mod_cfg["y_shape"],
        "H_shape": mod_cfg["H_shape"],
        "params": mod_cfg["spec"],
    }
    with open(os.path.join(tier_dir, "spec.json"), "w") as f:
        json.dump(spec, f, indent=2)

    with open(os.path.join(tier_dir, "true_spec.json"), "w") as f:
        json.dump(true_spec_data, f, indent=2)

    size_mb = os.path.getsize(h5_path) / 1e6
    print(f"  [{tier}] Done — {size_mb:.1f} MB")
    return size_mb


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"Output base: {OUT_BASE}")
    print(f"Modalities: {list(MODALITIES.keys())}")
    print()

    results = {}
    for modality, cfg in MODALITIES.items():
        print(f"=== {modality.upper()} ===")
        out_dir = os.path.join(OUT_BASE, modality)
        os.makedirs(out_dir, exist_ok=True)
        results[modality] = {}
        for tier in ["public", "dev", "hidden"]:
            try:
                mb = write_tier(modality, cfg["gen_fn"], out_dir, tier)
                results[modality][tier] = {"status": "OK", "size_mb": round(mb, 2)}
            except Exception as e:
                import traceback
                print(f"  [{tier}] ERROR: {e}")
                traceback.print_exc()
                results[modality][tier] = {"status": "ERROR", "error": str(e)}
        print()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_ok = True
    for modality, tiers in results.items():
        for tier, info in tiers.items():
            status = info["status"]
            if status == "OK":
                print(f"  {modality}/{tier}: OK ({info['size_mb']} MB)")
            else:
                print(f"  {modality}/{tier}: ERROR — {info.get('error','?')}")
                all_ok = False

    print()
    if all_ok:
        print("All datasets generated successfully.")
    else:
        print("Some datasets had errors — see above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
