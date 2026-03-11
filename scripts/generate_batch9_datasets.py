#!/usr/bin/env python3
"""
Generate benchmark datasets for batch-9 modalities:
  quantum_illumination, radio_astronomy, radio_interferometry, raman_imaging,
  saxs, seismic_tomo, shearography, shg, sims, solar_imaging

Each modality: 3 tiers (public=12, dev=20, hidden=20)
Seeds: public = 1300 + i*17, dev = 7300 + i*17, hidden = 9400 + i*17
"""

import json
import os

import h5py
import numpy as np
from scipy.ndimage import convolve, gaussian_filter, label
from scipy.fft import fft2, ifft2, fftshift, ifftshift

try:
    from PIL import Image
except ImportError:
    raise ImportError("pip install Pillow")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_BASE = os.path.join(ROOT, "datasets", "benchmark")

TIER_SIZES = {"public": 12, "dev": 20, "hidden": 20}
# base seed offsets
SEED_OFFSETS = {"public": 1300, "dev": 7300, "hidden": 9400}

VERSION = "1.0"

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def save_png(arr: np.ndarray, path: str) -> None:
    """Save a 2D float array as grayscale PNG, normalised to 0-255."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    a = np.asarray(arr, dtype=np.float64)
    a = a - a.min()
    mx = a.max()
    if mx > 0:
        a = a / mx
    img = Image.fromarray((a * 255).astype(np.uint8))
    img.save(path)


def save_png_rgb(arr: np.ndarray, path: str) -> None:
    """Save a (H,W,3) or (H,W,C) array as RGB PNG (use first 3 channels)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    a = arr[..., :3].astype(np.float64)
    a = a - a.min()
    mx = a.max()
    if mx > 0:
        a = a / mx
    img = Image.fromarray((a * 255).astype(np.uint8), mode="RGB")
    img.save(path)


def write_h5_sample(grp: h5py.Group, x_true, y, H_ideal, reconstruction, meta: dict):
    grp.create_dataset("x_true",               data=x_true.astype(np.float32),         compression="gzip")
    grp.create_dataset("y",                     data=y.astype(np.float32),              compression="gzip")
    grp.create_dataset("H_ideal",               data=H_ideal.astype(np.float32),        compression="gzip")
    grp.create_dataset("reconstruction_baseline", data=reconstruction.astype(np.float32), compression="gzip")
    grp.attrs["meta"] = json.dumps(meta)


# ---------------------------------------------------------------------------
# 1. QUANTUM_ILLUMINATION
# ---------------------------------------------------------------------------

def gen_quantum_illumination(rng: np.random.Generator, tier: str):
    """
    x_true : (64,64)  target reflectivity map
    y       : (64,64)  coincidence detection image + thermal background noise
    H_ideal : (64,64)  ideal (noiseless) coincidence image
    recon   : (64,64)  y - background_estimate
    """
    sz = 64
    # Target: sparse reflective patches on low-reflectivity background
    x_true = rng.uniform(0.05, 0.15, (sz, sz)).astype(np.float32)
    n_patches = int(rng.integers(3, 8))
    for _ in range(n_patches):
        cy, cx = rng.integers(8, sz - 8, 2)
        ry, rx = rng.integers(3, 12, 2)
        Y, X = np.ogrid[:sz, :sz]
        mask = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2 <= 1
        x_true[mask] = rng.uniform(0.6, 0.95)
    x_true = np.clip(x_true, 0, 1).astype(np.float32)

    # Thermal background level
    bg_level = float(rng.uniform(0.05, 0.20))
    noise_std = float(rng.uniform(0.02, 0.08))

    # Ideal coincidence ~ reflectivity
    H_ideal = x_true.copy()
    # Measured: coincidence + thermal background + Gaussian noise
    y = H_ideal + bg_level + rng.normal(0, noise_std, (sz, sz)).astype(np.float32)
    y = np.clip(y, 0, None).astype(np.float32)

    # Background estimate: median of corners
    bg_estimate = float(np.median([y[:8, :8], y[:8, -8:], y[-8:, :8], y[-8:, -8:]]))
    reconstruction = np.clip(y - bg_estimate, 0, None).astype(np.float32)

    meta = {"bg_level": bg_level, "noise_std": noise_std,
            "bg_estimate": bg_estimate, "n_patches": n_patches}
    return x_true, y, H_ideal, reconstruction, meta


def build_quantum_illumination_images(sample_dir, x_true, y, H_ideal, recon, meta):
    save_png(x_true, os.path.join(sample_dir, "ground_truth.png"))
    save_png(H_ideal, os.path.join(sample_dir, "ideal_coincidence.png"))
    save_png(y, os.path.join(sample_dir, "measured.png"))
    save_png(recon, os.path.join(sample_dir, "reconstruction.png"))
    with open(os.path.join(sample_dir, "spec.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# 2. RADIO_ASTRONOMY
# ---------------------------------------------------------------------------

def gen_radio_astronomy(rng: np.random.Generator, tier: str):
    """
    x_true : (128,128)  sky brightness distribution
    H_ideal : (128,128)  sparse uv-coverage mask (float 0/1)
    y       : (128,128)  dirty image (IFFT of masked FFT)
    recon   : (128,128)  CLEAN-like: iterative peak subtraction
    """
    sz = 128
    # Sky: point sources + extended emission
    x_true = np.zeros((sz, sz), dtype=np.float32)
    n_sources = int(rng.integers(5, 20))
    for _ in range(n_sources):
        cy, cx = rng.integers(4, sz - 4, 2)
        flux = float(rng.uniform(0.2, 1.0))
        sigma = float(rng.uniform(0.5, 3.0))
        Y, X = np.ogrid[:sz, :sz]
        blob = np.exp(-0.5 * ((X - cx) ** 2 + (Y - cy) ** 2) / sigma ** 2)
        x_true += flux * blob.astype(np.float32)
    x_true = x_true / (x_true.max() + 1e-8)

    # Sparse uv-coverage mask (radial tracks)
    H_ideal = np.zeros((sz, sz), dtype=np.float32)
    n_baselines = int(rng.integers(50, 200))
    half = sz // 2
    for _ in range(n_baselines):
        angle = float(rng.uniform(0, np.pi))
        length = float(rng.uniform(5, half - 2))
        for r in np.linspace(0, length, int(length * 2)):
            u = int(half + r * np.cos(angle)) % sz
            v = int(half + r * np.sin(angle)) % sz
            H_ideal[v, u] = 1.0
            H_ideal[sz - 1 - v, sz - 1 - u] = 1.0  # Hermitian symmetry

    # Dirty image: IFFT of masked Fourier transform
    sky_fft = fftshift(fft2(x_true))
    masked_fft = sky_fft * H_ideal
    dirty = np.real(ifft2(ifftshift(masked_fft))).astype(np.float32)

    # Noise
    noise_std = float(rng.uniform(0.005, 0.02))
    y = dirty + rng.normal(0, noise_std, (sz, sz)).astype(np.float32)

    # CLEAN-like reconstruction: iterative peak subtraction (simplified)
    residual = y.copy()
    clean_model = np.zeros_like(y)
    psf = np.real(ifft2(ifftshift(H_ideal.astype(complex)))).astype(np.float32)
    psf_norm = psf / (np.abs(psf).max() + 1e-8)
    gain = 0.1
    n_iter = 50
    for _ in range(n_iter):
        peak_idx = np.unravel_index(np.argmax(np.abs(residual)), residual.shape)
        peak_val = residual[peak_idx]
        clean_model[peak_idx] += gain * peak_val
        # Subtract scaled PSF centered at peak
        shift_y = peak_idx[0] - sz // 2
        shift_x = peak_idx[1] - sz // 2
        shifted_psf = np.roll(np.roll(psf_norm, shift_y, axis=0), shift_x, axis=1)
        residual -= gain * peak_val * shifted_psf
    reconstruction = (clean_model + residual).astype(np.float32)

    meta = {"n_sources": n_sources, "n_baselines": n_baselines,
            "noise_std": noise_std, "clean_iter": n_iter, "gain": gain}
    return x_true, y, H_ideal, reconstruction, meta


def build_radio_astronomy_images(sample_dir, x_true, y, H_ideal, recon, meta):
    save_png(x_true, os.path.join(sample_dir, "sky_true.png"))
    save_png(H_ideal, os.path.join(sample_dir, "uv_coverage.png"))
    save_png(y, os.path.join(sample_dir, "dirty_image.png"))
    save_png(recon, os.path.join(sample_dir, "reconstruction.png"))
    with open(os.path.join(sample_dir, "spec.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# 3. RADIO_INTERFEROMETRY
# ---------------------------------------------------------------------------

def gen_radio_interferometry(rng: np.random.Generator, tier: str):
    """
    x_true : (128,128)  radio source map
    H_ideal : (2048,2)  visibility u,v coordinates (float32, stored as H_ideal)
    y       : (2048,2)  visibility measurements [real, imag] at sampled (u,v)
    recon   : (128,128) gridded IFFT reconstruction
    """
    sz = 128
    n_vis = 2048

    # Source map
    x_true = np.zeros((sz, sz), dtype=np.float32)
    n_sources = int(rng.integers(3, 12))
    for _ in range(n_sources):
        cy, cx = rng.integers(10, sz - 10, 2)
        flux = float(rng.uniform(0.3, 1.0))
        sigma = float(rng.uniform(1.0, 5.0))
        Y, X = np.ogrid[:sz, :sz]
        blob = np.exp(-0.5 * ((X - cx) ** 2 + (Y - cy) ** 2) / sigma ** 2)
        x_true += flux * blob.astype(np.float32)
    x_true = (x_true / (x_true.max() + 1e-8)).astype(np.float32)

    # Sample random uv points (normalised -0.5 to 0.5)
    uv = rng.uniform(-0.5, 0.5, (n_vis, 2)).astype(np.float32)

    # Compute visibilities via DFT at sampled points
    # V(u,v) = sum_{l,m} I(l,m) * exp(-2pi*i*(u*l + v*m))
    l_coords = (np.arange(sz) - sz // 2) / sz  # normalised
    m_coords = (np.arange(sz) - sz // 2) / sz
    ll, mm = np.meshgrid(l_coords, m_coords, indexing="ij")  # (sz,sz)

    # Efficient: use FFT grid + bilinear lookup
    # Compute full FFT and sample
    src_fft = fftshift(fft2(ifftshift(x_true)))  # (sz,sz) complex

    # Map uv to pixel indices
    u_pix = (uv[:, 0] * sz + sz // 2).astype(int)
    v_pix = (uv[:, 1] * sz + sz // 2).astype(int)
    u_pix = np.clip(u_pix, 0, sz - 1)
    v_pix = np.clip(v_pix, 0, sz - 1)
    vis_complex = src_fft[u_pix, v_pix]

    # Add noise
    noise_std = float(rng.uniform(0.001, 0.01))
    vis_complex = vis_complex + rng.normal(0, noise_std, n_vis) + 1j * rng.normal(0, noise_std, n_vis)

    # y stored as (n_vis, 2) float pairs [real, imag]
    y = np.stack([vis_complex.real, vis_complex.imag], axis=-1).astype(np.float32)

    # H_ideal: store uv coordinates
    H_ideal = uv

    # Reconstruction: grid visibilities back onto FFT grid then IFFT
    grid = np.zeros((sz, sz), dtype=complex)
    cnt = np.zeros((sz, sz), dtype=np.float32)
    for k in range(n_vis):
        grid[u_pix[k], v_pix[k]] += vis_complex[k]
        cnt[u_pix[k], v_pix[k]] += 1
    cnt = np.maximum(cnt, 1)
    grid = grid / cnt
    reconstruction = np.real(fftshift(ifft2(ifftshift(grid)))).astype(np.float32)
    reconstruction = np.clip(reconstruction, 0, None)

    meta = {"n_sources": n_sources, "n_vis": n_vis, "noise_std": noise_std}
    return x_true, y, H_ideal, reconstruction, meta


def build_radio_interferometry_images(sample_dir, x_true, y, H_ideal, recon, meta):
    save_png(x_true, os.path.join(sample_dir, "source_map.png"))
    # uv coverage
    sz = 128
    uv_img = np.zeros((sz, sz), dtype=np.float32)
    u_pix = np.clip((H_ideal[:, 0] * sz + sz // 2).astype(int), 0, sz - 1)
    v_pix = np.clip((H_ideal[:, 1] * sz + sz // 2).astype(int), 0, sz - 1)
    uv_img[u_pix, v_pix] = 1.0
    save_png(uv_img, os.path.join(sample_dir, "uv_coverage.png"))
    # visibility amplitude (1D plot as image)
    vis_amp = np.sqrt(y[:, 0] ** 2 + y[:, 1] ** 2)
    vis_img = vis_amp.reshape(64, 32)
    save_png(vis_img, os.path.join(sample_dir, "visibility_amplitude.png"))
    save_png(recon, os.path.join(sample_dir, "reconstruction.png"))
    with open(os.path.join(sample_dir, "spec.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# 4. RAMAN_IMAGING
# ---------------------------------------------------------------------------

def gen_raman_imaging(rng: np.random.Generator, tier: str):
    """
    x_true : (128,128,5)  Raman spectral image (5 spectral peaks)
    y       : (128,128,5)  measured spectra + fluorescence background
    H_ideal : (128,128,5)  fluorescence background map
    recon   : (128,128,5)  background-subtracted spectra
    """
    sz = 128
    n_peaks = 5

    # Spatial distribution for each Raman peak (different chemical component)
    x_true = np.zeros((sz, sz, n_peaks), dtype=np.float32)
    for p in range(n_peaks):
        # Each peak has a distinctive spatial distribution
        n_regions = int(rng.integers(2, 6))
        for _ in range(n_regions):
            cy, cx = rng.integers(10, sz - 10, 2)
            sigma = float(rng.uniform(5, 20))
            amp = float(rng.uniform(0.4, 1.0))
            Y, X = np.ogrid[:sz, :sz]
            blob = amp * np.exp(-0.5 * ((X - cx) ** 2 + (Y - cy) ** 2) / sigma ** 2)
            x_true[:, :, p] += blob.astype(np.float32)
    x_true = np.clip(x_true, 0, 1).astype(np.float32)

    # Fluorescence background: smooth spatial variation, same for all peaks
    bg_amp = float(rng.uniform(0.1, 0.5))
    bg = bg_amp * gaussian_filter(rng.uniform(0, 1, (sz, sz)).astype(np.float32), sigma=20)
    bg = bg.astype(np.float32)
    H_ideal = np.stack([bg] * n_peaks, axis=-1)  # (sz, sz, n_peaks)

    # Measured: Raman + fluorescence + Poisson noise
    scale = 1000.0
    expected = (x_true + H_ideal) * scale
    y = rng.poisson(np.maximum(expected, 0.1)).astype(np.float32) / scale

    # Background subtraction (polynomial fitting simplified as Gaussian blur)
    recon = np.zeros_like(y)
    for p in range(n_peaks):
        bg_est = gaussian_filter(y[:, :, p], sigma=15)
        recon[:, :, p] = np.clip(y[:, :, p] - bg_est, 0, None)
    reconstruction = recon.astype(np.float32)

    meta = {"bg_amplitude": bg_amp, "n_peaks": n_peaks, "photon_scale": scale}
    return x_true, y, H_ideal, reconstruction, meta


def build_raman_imaging_images(sample_dir, x_true, y, H_ideal, recon, meta):
    # Show first 3 channels as RGB composite
    save_png_rgb(x_true[:, :, :3], os.path.join(sample_dir, "raman_true_rgb.png"))
    save_png_rgb(y[:, :, :3], os.path.join(sample_dir, "raman_measured_rgb.png"))
    save_png_rgb(recon[:, :, :3], os.path.join(sample_dir, "reconstruction_rgb.png"))
    save_png(H_ideal[:, :, 0], os.path.join(sample_dir, "fluorescence_background.png"))
    with open(os.path.join(sample_dir, "spec.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# 5. SAXS
# ---------------------------------------------------------------------------

def gen_saxs(rng: np.random.Generator, tier: str):
    """
    x_true : (128,128)  electron density map (soft matter structure)
    H_ideal : (128,128)  ideal SAXS pattern abs(FFT)^2 (log scale)
    y       : (128,128)  SAXS pattern with noise
    recon   : (128,128)  IFFT of sqrt(y) (phase problem approximation)
    """
    sz = 128

    # Electron density: lamellar / cylindrical soft matter structures
    x_true = np.zeros((sz, sz), dtype=np.float32)
    # Add periodic lamellar structure
    period = float(rng.uniform(8, 30))
    angle = float(rng.uniform(0, np.pi))
    Y, X = np.ogrid[:sz, :sz]
    lamella = np.sin(2 * np.pi * (X * np.cos(angle) + Y * np.sin(angle)) / period)
    x_true += 0.5 * lamella.astype(np.float32)

    # Add some blobs
    n_blobs = int(rng.integers(2, 6))
    for _ in range(n_blobs):
        cy, cx = rng.integers(20, sz - 20, 2)
        sigma = float(rng.uniform(3, 10))
        amp = float(rng.uniform(0.2, 0.6))
        blob = amp * np.exp(-0.5 * ((X - cx) ** 2 + (Y - cy) ** 2) / sigma ** 2)
        x_true += blob.astype(np.float32)

    x_true = (x_true - x_true.min()) / (x_true.max() - x_true.min() + 1e-8)
    x_true = x_true.astype(np.float32)

    # SAXS pattern: |FFT(x_true)|^2
    ft = fftshift(fft2(x_true))
    saxs_ideal = np.abs(ft) ** 2
    H_ideal = np.log1p(saxs_ideal).astype(np.float32)

    # Add Poisson noise scaled by detector counts
    photon_scale = float(rng.uniform(500, 5000))
    y_raw = rng.poisson(np.maximum(saxs_ideal * photon_scale, 0.1))
    y = np.log1p(y_raw.astype(np.float32) / photon_scale)

    # Reconstruction: IFFT of sqrt(measured intensity) (ignores phase)
    amplitude = np.sqrt(np.maximum(y_raw.astype(np.float32) / photon_scale, 0))
    recon_complex = ifft2(ifftshift(amplitude.astype(complex)))
    reconstruction = np.abs(recon_complex).astype(np.float32)
    reconstruction = (reconstruction - reconstruction.min()) / (reconstruction.max() - reconstruction.min() + 1e-8)

    meta = {"period_px": period, "angle_rad": float(angle),
            "photon_scale": photon_scale, "n_blobs": n_blobs}
    return x_true, y, H_ideal, reconstruction, meta


def build_saxs_images(sample_dir, x_true, y, H_ideal, recon, meta):
    save_png(x_true, os.path.join(sample_dir, "electron_density.png"))
    save_png(H_ideal, os.path.join(sample_dir, "saxs_pattern_ideal.png"))
    save_png(y, os.path.join(sample_dir, "saxs_pattern_measured.png"))
    save_png(recon, os.path.join(sample_dir, "reconstruction.png"))
    with open(os.path.join(sample_dir, "spec.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# 6. SEISMIC_TOMO
# ---------------------------------------------------------------------------

def gen_seismic_tomo(rng: np.random.Generator, tier: str):
    """
    x_true  : (64,64)   seismic velocity model
    H_ideal : (32,64)   ideal (noiseless) travel-time picks
    y       : (32,64)   noisy travel-time picks
    recon   : (64,64)   backprojection of travel-time picks
    """
    sz = 64
    n_rays = 32  # ray paths

    # Velocity model: layered with anomalies
    x_true = np.ones((sz, sz), dtype=np.float32) * 3000.0  # m/s background
    # Add layers
    n_layers = int(rng.integers(2, 5))
    for i in range(n_layers):
        y_start = int(rng.integers(8, sz - 8))
        dv = float(rng.uniform(-500, 1000))
        x_true[y_start:, :] += dv
    # Add velocity anomaly (low- or high-velocity zone)
    cy, cx = rng.integers(15, sz - 15, 2)
    ry, rx = rng.integers(5, 15, 2)
    dv_anomaly = float(rng.uniform(-800, 800))
    Y, X = np.ogrid[:sz, :sz]
    mask = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2 <= 1
    x_true[mask] += dv_anomaly
    x_true = np.clip(x_true, 1500, 8000).astype(np.float32)

    # Normalise for output
    x_norm = (x_true - x_true.min()) / (x_true.max() - x_true.min() + 1e-8)

    # Travel-time forward model: rays travel horizontally across model
    # t = sum(dx / v) for each ray (row)
    H_ideal = np.zeros((n_rays, sz), dtype=np.float32)
    ray_ys = np.linspace(0, sz - 1, n_rays).astype(int)
    for i, ry_idx in enumerate(ray_ys):
        # Travel time through each column
        dx = 1.0  # unit pixel size
        H_ideal[i, :] = dx / x_true[ry_idx, :]

    # Add noise
    noise_std = float(rng.uniform(1e-5, 5e-5))
    y = H_ideal + rng.normal(0, noise_std, H_ideal.shape).astype(np.float32)

    # Backprojection: smear travel-time picks back along rays
    reconstruction = np.zeros((sz, sz), dtype=np.float32)
    for i, ry_idx in enumerate(ray_ys):
        # Slowness estimate from ray
        slowness = y[i, :]
        reconstruction[ry_idx, :] += slowness
    # Fill gaps by interpolation (simple nearest neighbour)
    for row in range(sz):
        # find nearest ray
        dists = np.abs(ray_ys - row)
        nearest = np.argmin(dists)
        reconstruction[row, :] = reconstruction[ray_ys[nearest], :]
    # Convert slowness to velocity (approximate)
    reconstruction = 1.0 / (reconstruction + 1e-8)
    reconstruction = (reconstruction - reconstruction.min()) / (reconstruction.max() - reconstruction.min() + 1e-8)
    reconstruction = reconstruction.astype(np.float32)

    meta = {"n_layers": n_layers, "dv_anomaly_ms": float(dv_anomaly),
            "noise_std": noise_std, "n_rays": n_rays}
    return x_norm, y, H_ideal, reconstruction, meta


def build_seismic_tomo_images(sample_dir, x_true, y, H_ideal, recon, meta):
    save_png(x_true, os.path.join(sample_dir, "velocity_model.png"))
    save_png(H_ideal, os.path.join(sample_dir, "traveltimes_ideal.png"))
    save_png(y, os.path.join(sample_dir, "traveltimes_measured.png"))
    save_png(recon, os.path.join(sample_dir, "reconstruction.png"))
    with open(os.path.join(sample_dir, "spec.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# 7. SHEAROGRAPHY
# ---------------------------------------------------------------------------

def gen_shearography(rng: np.random.Generator, tier: str):
    """
    x_true : (128,128)  deformation gradient map
    H_ideal : (128,128)  ideal speckle phase difference
    y       : (128,128)  speckle pattern difference (noisy)
    recon   : (128,128)  phase unwrapping (return y directly as it encodes the phase)
    """
    sz = 128

    # Deformation field: smooth surface under load with defects
    x_true = np.zeros((sz, sz), dtype=np.float32)
    # Background deformation (smooth)
    Y, X = np.ogrid[:sz, :sz]
    bg_deform = gaussian_filter(rng.uniform(0, 1, (sz, sz)).astype(np.float32), sigma=20)
    x_true = bg_deform.astype(np.float32)

    # Defect (delamination) causes local deformation anomaly
    n_defects = int(rng.integers(1, 4))
    for _ in range(n_defects):
        cy, cx = rng.integers(20, sz - 20, 2)
        sigma_d = float(rng.uniform(5, 15))
        amp = float(rng.uniform(0.3, 0.8))
        defect = amp * np.exp(-0.5 * ((X - cx) ** 2 + (Y - cy) ** 2) / sigma_d ** 2)
        x_true += defect.astype(np.float32)
    x_true = (x_true - x_true.min()) / (x_true.max() - x_true.min() + 1e-8)

    # Shear distance (pixels)
    shear_dx = int(rng.integers(3, 8))

    # Shearography: phase = diff of object wave and sheared copy
    # Phase encodes derivative of displacement
    # grad_x = x_true - shifted x_true
    x_shifted = np.roll(x_true, shear_dx, axis=1)
    phase = (x_true - x_shifted) * 2 * np.pi * float(rng.uniform(2, 8))
    H_ideal = np.sin(phase).astype(np.float32)  # wrapped phase

    # Speckle: random carrier + modulated by deformation phase
    speckle_carrier = rng.uniform(-1, 1, (sz, sz)).astype(np.float32)
    noise_level = float(rng.uniform(0.05, 0.2))
    speckle_noise = rng.normal(0, noise_level, (sz, sz)).astype(np.float32)
    y = H_ideal * (1 + speckle_carrier * 0.3) + speckle_noise
    y = y.astype(np.float32)

    # Reconstruction: return y (phase map)
    reconstruction = y.copy()

    meta = {"shear_dx": shear_dx, "n_defects": n_defects, "noise_level": noise_level}
    return x_true, y, H_ideal, reconstruction, meta


def build_shearography_images(sample_dir, x_true, y, H_ideal, recon, meta):
    save_png(x_true, os.path.join(sample_dir, "deformation_map.png"))
    save_png(H_ideal, os.path.join(sample_dir, "phase_ideal.png"))
    save_png(y, os.path.join(sample_dir, "speckle_difference.png"))
    save_png(recon, os.path.join(sample_dir, "reconstruction.png"))
    with open(os.path.join(sample_dir, "spec.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# 8. SHG
# ---------------------------------------------------------------------------

def gen_shg(rng: np.random.Generator, tier: str):
    """
    x_true : (128,128)  SHG signal map (collagen fibres)
    H_ideal : (128,128)  SHG signal^2 (nonlinear forward model, noiseless)
    y       : (128,128)  x_true^2 + noise
    recon   : (128,128)  sqrt(max(y,0))
    """
    sz = 128

    # Collagen fibres: elongated oriented structures
    x_true = np.zeros((sz, sz), dtype=np.float32)
    n_fibres = int(rng.integers(5, 15))
    for _ in range(n_fibres):
        cy, cx = rng.integers(10, sz - 10, 2)
        length = float(rng.uniform(10, 50))
        width = float(rng.uniform(1, 5))
        angle = float(rng.uniform(0, np.pi))
        amp = float(rng.uniform(0.4, 1.0))
        Y, X = np.ogrid[:sz, :sz]
        dy = Y - cy
        dx = X - cx
        # Rotated ellipse
        x_rot = dx * np.cos(angle) + dy * np.sin(angle)
        y_rot = -dx * np.sin(angle) + dy * np.cos(angle)
        fibre = amp * np.exp(-0.5 * ((x_rot / (length / 2)) ** 2 + (y_rot / (width / 2)) ** 2))
        x_true += fibre.astype(np.float32)
    x_true = np.clip(x_true, 0, 1).astype(np.float32)

    # SHG forward model: I_SHG ~ I_excitation^2 (quadratic response)
    H_ideal = x_true ** 2

    # Noise: shot noise (Poisson) + Gaussian readout
    photon_scale = float(rng.uniform(100, 1000))
    shot_y = rng.poisson(np.maximum(H_ideal * photon_scale, 0.1)).astype(np.float32) / photon_scale
    readout_noise = float(rng.uniform(0.005, 0.02))
    y = shot_y + rng.normal(0, readout_noise, (sz, sz)).astype(np.float32)
    y = np.clip(y, 0, None).astype(np.float32)

    # Reconstruction: sqrt(y)
    reconstruction = np.sqrt(np.maximum(y, 0)).astype(np.float32)

    meta = {"n_fibres": n_fibres, "photon_scale": photon_scale,
            "readout_noise": readout_noise}
    return x_true, y, H_ideal, reconstruction, meta


def build_shg_images(sample_dir, x_true, y, H_ideal, recon, meta):
    save_png(x_true, os.path.join(sample_dir, "collagen_true.png"))
    save_png(H_ideal, os.path.join(sample_dir, "shg_ideal.png"))
    save_png(y, os.path.join(sample_dir, "shg_measured.png"))
    save_png(recon, os.path.join(sample_dir, "reconstruction.png"))
    with open(os.path.join(sample_dir, "spec.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# 9. SIMS
# ---------------------------------------------------------------------------

def gen_sims(rng: np.random.Generator, tier: str):
    """
    x_true : (128,128,3)  secondary ion mass spec map (3 elements)
    H_ideal : (128,128,3)  ideal (noiseless) SIMS counts
    y       : (128,128,3)  measured counts + Poisson noise
    recon   : (128,128,3)  y normalised per channel
    """
    sz = 128
    n_elem = 3

    # Each element has different spatial distribution (grain structure)
    x_true = np.zeros((sz, sz, n_elem), dtype=np.float32)
    for e in range(n_elem):
        # Voronoi-like grain structure for each element
        n_grains = int(rng.integers(5, 20))
        grain_centers = rng.integers(0, sz, (n_grains, 2))
        grain_img = np.zeros((sz, sz), dtype=np.float32)
        for gy in range(sz):
            for gx in range(sz):
                dists = np.sqrt(((grain_centers[:, 0] - gy) ** 2 +
                                  (grain_centers[:, 1] - gx) ** 2).astype(np.float32))
                nearest = np.argmin(dists)
                grain_img[gy, gx] = float(nearest % 2)  # binary grain map
        # Smooth edges
        grain_img = gaussian_filter(grain_img, sigma=1.5)
        amp = float(rng.uniform(0.5, 1.0))
        x_true[:, :, e] = amp * grain_img.astype(np.float32)

    x_true = np.clip(x_true, 0, 1).astype(np.float32)

    # SIMS counts: primary ion dose * ionisation yield * element concentration
    ion_dose = float(rng.uniform(1e4, 1e6))
    ionisation_yield = rng.uniform(0.001, 0.01, n_elem).astype(np.float32)
    H_ideal = np.zeros_like(x_true)
    for e in range(n_elem):
        H_ideal[:, :, e] = x_true[:, :, e] * ion_dose * ionisation_yield[e]

    # Poisson noise
    y = rng.poisson(np.maximum(H_ideal, 0.1)).astype(np.float32)

    # Reconstruction: normalise each channel by its max
    reconstruction = np.zeros_like(y)
    for e in range(n_elem):
        ch_max = y[:, :, e].max()
        reconstruction[:, :, e] = y[:, :, e] / (ch_max + 1e-8)

    meta = {"ion_dose": float(ion_dose),
            "ionisation_yield": ionisation_yield.tolist(),
            "n_elements": n_elem}
    return x_true, y, H_ideal, reconstruction, meta


def build_sims_images(sample_dir, x_true, y, H_ideal, recon, meta):
    save_png_rgb(x_true, os.path.join(sample_dir, "element_map_true.png"))
    save_png_rgb(y, os.path.join(sample_dir, "sims_measured.png"))
    save_png_rgb(recon, os.path.join(sample_dir, "reconstruction.png"))
    save_png(H_ideal[:, :, 0], os.path.join(sample_dir, "element0_ideal.png"))
    with open(os.path.join(sample_dir, "spec.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# 10. SOLAR_IMAGING
# ---------------------------------------------------------------------------

def gen_solar_imaging(rng: np.random.Generator, tier: str):
    """
    x_true : (256,256)  solar disk image with active regions
    H_ideal : (256,256)  PSF (Airy disk)
    y       : (256,256)  x_true convolved with telescope PSF + noise
    recon   : (256,256)  Wiener deconvolution
    """
    sz = 256

    # Solar disk
    x_true = np.zeros((sz, sz), dtype=np.float32)
    cy, cx = sz // 2, sz // 2
    radius = int(rng.integers(80, 110))
    Y, X = np.ogrid[:sz, :sz]
    disk_mask = (X - cx) ** 2 + (Y - cy) ** 2 <= radius ** 2

    # Limb darkening
    r = np.sqrt((X - cx).astype(np.float32) ** 2 + (Y - cy).astype(np.float32) ** 2)
    r_norm = r / (radius + 1e-8)
    # Limb darkening law: I(mu) = 1 - u*(1-mu), mu = sqrt(1-r^2)
    u_ld = float(rng.uniform(0.4, 0.7))
    mu = np.sqrt(np.maximum(1 - r_norm ** 2, 0)).astype(np.float32)
    limb = (1 - u_ld * (1 - mu)) * disk_mask

    x_true = limb.astype(np.float32)

    # Active regions (sunspots + faculae)
    n_spots = int(rng.integers(2, 8))
    for _ in range(n_spots):
        # Random position within disk
        for attempt in range(20):
            sy_, sx_ = rng.integers(cy - radius + 10, cy + radius - 10), \
                       rng.integers(cx - radius + 10, cx + radius - 10)
            if (sx_ - cx) ** 2 + (sy_ - cy) ** 2 < (radius - 10) ** 2:
                break
        spot_r = int(rng.integers(3, 15))
        intensity = float(rng.uniform(-0.6, -0.1))  # dark spot
        spot_mask = ((X - sx_) ** 2 + (Y - sy_) ** 2) <= spot_r ** 2
        x_true[spot_mask] = np.clip(x_true[spot_mask] + intensity, 0, 1)
        # Faculae (bright ring around spot)
        fac_r = spot_r + int(rng.integers(2, 5))
        fac_mask = (((X - sx_) ** 2 + (Y - sy_) ** 2) <= fac_r ** 2) & (~spot_mask)
        x_true[fac_mask] = np.clip(x_true[fac_mask] + 0.1, 0, 1)

    x_true = np.clip(x_true, 0, 1).astype(np.float32)

    # PSF: Airy disk pattern
    psf_size = int(rng.integers(9, 25))  # odd size
    if psf_size % 2 == 0:
        psf_size += 1
    half_psf = psf_size // 2
    ky, kx = np.ogrid[-half_psf:half_psf + 1, -half_psf:half_psf + 1]
    k_r = np.sqrt(kx.astype(np.float32) ** 2 + ky.astype(np.float32) ** 2)
    # Airy: (2*J1(pi*k_r/lambda) / (pi*k_r/lambda))^2, approximate with Gaussian
    telescope_lambda = float(rng.uniform(0.8, 2.0))  # "wavelength" in PSF units
    psf_sigma = telescope_lambda
    psf_kernel = np.exp(-0.5 * k_r ** 2 / psf_sigma ** 2).astype(np.float32)
    psf_kernel = psf_kernel / psf_kernel.sum()

    # Pad PSF to full image size for H_ideal
    H_ideal = np.zeros((sz, sz), dtype=np.float32)
    H_ideal[:psf_size, :psf_size] = psf_kernel
    H_ideal = np.roll(np.roll(H_ideal, -half_psf, axis=0), -half_psf, axis=1)

    # Convolution
    blurred = convolve(x_true, psf_kernel, mode='wrap').astype(np.float32)

    # Add noise
    noise_std = float(rng.uniform(0.002, 0.01))
    y = blurred + rng.normal(0, noise_std, (sz, sz)).astype(np.float32)
    y = np.clip(y, 0, None).astype(np.float32)

    # Wiener deconvolution
    snr_est = float(rng.uniform(10, 100))  # estimated SNR
    Y_fft = fft2(y)
    H_fft = fft2(H_ideal)
    H_conj = np.conj(H_fft)
    wiener_filter = H_conj / (np.abs(H_fft) ** 2 + 1.0 / snr_est)
    reconstruction = np.real(ifft2(wiener_filter * Y_fft)).astype(np.float32)
    reconstruction = np.clip(reconstruction, 0, None)
    if reconstruction.max() > 0:
        reconstruction = reconstruction / reconstruction.max()

    meta = {"radius_px": radius, "limb_darkening_u": u_ld,
            "n_spots": n_spots, "psf_sigma": float(psf_sigma),
            "psf_size": psf_size, "noise_std": noise_std, "snr_est": float(snr_est)}
    return x_true, y, H_ideal, reconstruction, meta


def build_solar_imaging_images(sample_dir, x_true, y, H_ideal, recon, meta):
    save_png(x_true, os.path.join(sample_dir, "solar_disk_true.png"))
    save_png(H_ideal, os.path.join(sample_dir, "psf.png"))
    save_png(y, os.path.join(sample_dir, "observed.png"))
    save_png(recon, os.path.join(sample_dir, "reconstruction.png"))
    with open(os.path.join(sample_dir, "spec.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODALITIES = {
    "quantum_illumination": {
        "fn": gen_quantum_illumination,
        "img_fn": build_quantum_illumination_images,
        "x_shape": [64, 64],
        "y_shape": [64, 64],
        "description": "Quantum illumination coincidence imaging with thermal background",
        "physics": "coincidence_detection",
        "reference": "Lloyd, S. (2008). Enhanced sensitivity of photodetection via quantum illumination. Science, 321(5895), 1463–1465.",
    },
    "radio_astronomy": {
        "fn": gen_radio_astronomy,
        "img_fn": build_radio_astronomy_images,
        "x_shape": [128, 128],
        "y_shape": [128, 128],
        "description": "Radio astronomy dirty image synthesis with sparse uv-coverage",
        "physics": "radio_interferometry_fft",
        "reference": "Cornwell, T.J. (2008). Multiscale CLEAN deconvolution of radio synthesis images. IEEE JSTSP, 2(5), 793–801.",
    },
    "radio_interferometry": {
        "fn": gen_radio_interferometry,
        "img_fn": build_radio_interferometry_images,
        "x_shape": [128, 128],
        "y_shape": [2048, 2],
        "description": "Radio interferometry visibility measurements with gridded IFFT reconstruction",
        "physics": "van_cittert_zernike",
        "reference": "Thompson, A.R., Moran, J.M., Swenson, G.W. (2017). Interferometry and Synthesis in Radio Astronomy. Springer.",
    },
    "raman_imaging": {
        "fn": gen_raman_imaging,
        "img_fn": build_raman_imaging_images,
        "x_shape": [128, 128, 5],
        "y_shape": [128, 128, 5],
        "description": "Raman spectral imaging with fluorescence background",
        "physics": "spontaneous_raman_scattering",
        "reference": "Dieing, T., Hollricher, O., Toporski, J. (2011). Confocal Raman Microscopy. Springer.",
    },
    "saxs": {
        "fn": gen_saxs,
        "img_fn": build_saxs_images,
        "x_shape": [128, 128],
        "y_shape": [128, 128],
        "description": "Small-angle X-ray scattering from soft matter with phase problem",
        "physics": "fraunhofer_diffraction",
        "reference": "Glatter, O., Kratky, O. (1982). Small Angle X-ray Scattering. Academic Press.",
    },
    "seismic_tomo": {
        "fn": gen_seismic_tomo,
        "img_fn": build_seismic_tomo_images,
        "x_shape": [64, 64],
        "y_shape": [32, 64],
        "description": "Seismic travel-time tomography for velocity model recovery",
        "physics": "ray_theory_eikonal",
        "reference": "Nolet, G. (2008). A Breviary of Seismic Tomography. Cambridge University Press.",
    },
    "shearography": {
        "fn": gen_shearography,
        "img_fn": build_shearography_images,
        "x_shape": [128, 128],
        "y_shape": [128, 128],
        "description": "Shearography speckle-pattern interference for deformation gradient imaging",
        "physics": "speckle_interferometry_shear",
        "reference": "Hung, Y.Y. (1982). Shearography: a new optical method for strain measurement and nondestructive testing. Optical Engineering, 21(3).",
    },
    "shg": {
        "fn": gen_shg,
        "img_fn": build_shg_images,
        "x_shape": [128, 128],
        "y_shape": [128, 128],
        "description": "Second-harmonic generation imaging of collagen fibres",
        "physics": "chi2_nonlinear_optics",
        "reference": "Zipfel, W.R., Williams, R.M., Webb, W.W. (2003). Nonlinear magic: multiphoton microscopy in the biosciences. Nature Biotechnology, 21(11), 1369–1377.",
    },
    "sims": {
        "fn": gen_sims,
        "img_fn": build_sims_images,
        "x_shape": [128, 128, 3],
        "y_shape": [128, 128, 3],
        "description": "Secondary ion mass spectrometry imaging of 3-element maps",
        "physics": "ion_sputtering_mass_spectrometry",
        "reference": "Benninghoven, A., Rüdenauer, F.G., Werner, H.W. (1987). Secondary Ion Mass Spectrometry. Wiley.",
    },
    "solar_imaging": {
        "fn": gen_solar_imaging,
        "img_fn": build_solar_imaging_images,
        "x_shape": [256, 256],
        "y_shape": [256, 256],
        "description": "Solar disk imaging with telescope PSF and Wiener deconvolution",
        "physics": "telescope_diffraction_psfconv",
        "reference": "Stix, M. (2002). The Sun: An Introduction. Springer.",
    },
}


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate_modality(modality: str, cfg: dict) -> None:
    fn = cfg["fn"]
    img_fn = cfg["img_fn"]
    mod_dir = os.path.join(OUT_BASE, modality)

    all_true_specs = {}

    for tier, n_samples in TIER_SIZES.items():
        tier_dir = os.path.join(mod_dir, tier)
        os.makedirs(tier_dir, exist_ok=True)
        img_base = os.path.join(tier_dir, "images")
        os.makedirs(img_base, exist_ok=True)

        h5_path = os.path.join(tier_dir, f"{modality}_challenge_{tier}.h5")
        print(f"  [{modality}] {tier}: {n_samples} samples -> {h5_path}")

        true_spec = {}

        with h5py.File(h5_path, "w") as hf:
            hf.attrs.update({
                "variant": modality,
                "version": VERSION,
                "tier": tier,
                "description": cfg["description"],
                "physics": cfg["physics"],
                "reference": cfg["reference"],
                "x_shape": json.dumps(cfg["x_shape"]),
                "y_shape": json.dumps(cfg["y_shape"]),
            })

            for i in range(n_samples):
                seed = SEED_OFFSETS[tier] + i * 17
                rng = np.random.default_rng(seed)

                x_true, y, H_ideal, reconstruction, meta = fn(rng, tier)

                grp = hf.create_group(f"sample_{i:02d}")
                write_h5_sample(grp, x_true, y, H_ideal, reconstruction, meta)

                # PNG previews
                sample_img_dir = os.path.join(img_base, f"sample_{i:02d}")
                os.makedirs(sample_img_dir, exist_ok=True)
                img_fn(sample_img_dir, x_true, y, H_ideal, reconstruction, meta)

                true_spec[f"sample_{i:02d}"] = {k: (float(v) if isinstance(v, (np.floating, float)) else
                                                      v.tolist() if isinstance(v, np.ndarray) else v)
                                                 for k, v in meta.items()}

        # Write spec.json (mismatch parameter ranges)
        spec = {
            "modality": modality,
            "tier": tier,
            "n_samples": n_samples,
            "x_shape": cfg["x_shape"],
            "y_shape": cfg["y_shape"],
            "physics": cfg["physics"],
            "description": cfg["description"],
            "seed_base": SEED_OFFSETS[tier],
            "seed_step": 17,
        }
        with open(os.path.join(tier_dir, "spec.json"), "w") as f:
            json.dump(spec, f, indent=2)

        # Write true_spec.json
        with open(os.path.join(tier_dir, "true_spec.json"), "w") as f:
            json.dump(true_spec, f, indent=2)

        sz = os.path.getsize(h5_path) / 1e6
        print(f"    Saved {h5_path} ({sz:.1f} MB), true_spec.json, {n_samples} PNGs")

    print(f"  [{modality}] DONE")


def main():
    print(f"Output base: {OUT_BASE}")
    os.makedirs(OUT_BASE, exist_ok=True)

    modality_list = list(MODALITIES.keys())
    print(f"Generating {len(modality_list)} modalities: {modality_list}\n")

    for modality in modality_list:
        print(f"\n=== {modality.upper()} ===")
        try:
            generate_modality(modality, MODALITIES[modality])
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()

    print("\n\nAll done.")


if __name__ == "__main__":
    main()
