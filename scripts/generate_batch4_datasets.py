"""
Generate benchmark datasets for 10 modalities (Batch 4):
  1. edx_mapping         - EDX elemental mapping
  2. eht_imaging         - Event Horizon Telescope imaging
  3. elastography        - Elastography stiffness mapping
  4. electron_diffraction - Electron diffraction (already exists, re-gen with new seeds)
  5. electron_holography  - Electron holography (already exists, re-gen with new seeds)
  6. electron_tomography  - Electron tomography (already exists, re-gen with new seeds)
  7. entangled_photon     - Entangled photon imaging
  8. event_camera         - Event camera
  9. expansion            - Expansion microscopy
  10. fib_sem             - FIB-SEM

Output layout:
  datasets/benchmark/{modality}/{tier}/
    {modality}_challenge_{tier}.h5
    spec.json
    true_spec.json
    images/sample_NN/{x_true,y,H_ideal,reconstruction_baseline}.png

Tiers: public (12 samples), dev (20), hidden (20)
Seeds: public=4000+i*17, dev=8600+i*17, hidden=9800+i*17
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom
from scipy.fft import fft2, ifft2, fftshift, ifftshift

# ── Root paths ────────────────────────────────────────────────────────────────
ROOT = Path("D:/onedrive/startup/program/physics_world_model/PWM5/Physics_World_Model")
OUT_BASE = ROOT / "datasets" / "benchmark"

TIER_SIZES = {"public": 12, "dev": 20, "hidden": 20}

MODALITIES = [
    "edx_mapping",
    "eht_imaging",
    "elastography",
    "electron_diffraction",
    "electron_holography",
    "electron_tomography",
    "entangled_photon",
    "event_camera",
    "expansion",
    "fib_sem",
]

# Seeds: base_seed + sample_index * 17
TIER_BASE_SEEDS = {"public": 4000, "dev": 8600, "hidden": 9800}
# Per-modality offset (index * 1000 to keep them separated)
MODALITY_OFFSET = {m: i * 1000 for i, m in enumerate(MODALITIES)}


def get_seed(modality: str, tier: str, sample_idx: int) -> int:
    return MODALITY_OFFSET[modality] + TIER_BASE_SEEDS[tier] + sample_idx * 17


# ═══════════════════════════════════════════════════════════════════════════════
# PNG helpers
# ═══════════════════════════════════════════════════════════════════════════════

def save_png(arr: np.ndarray, path: Path) -> None:
    """Save float32 array as 8-bit grayscale or RGB PNG (robust normalization)."""
    a = arr.astype(np.float32)
    if a.ndim == 3 and a.shape[2] == 3:
        # RGB
        lo, hi = a.min(), a.max()
        if hi - lo < 1e-10:
            a = np.zeros_like(a)
        else:
            a = (a - lo) / (hi - lo)
        img_uint8 = (np.clip(a, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(img_uint8, mode="RGB").save(str(path))
    else:
        lo, hi = a.min(), a.max()
        if hi - lo < 1e-10:
            a = np.zeros_like(a)
        else:
            a = (a - lo) / (hi - lo)
        img_uint8 = (np.clip(a, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(img_uint8, mode="L").save(str(path))


# ═══════════════════════════════════════════════════════════════════════════════
# 1. EDX Mapping
#    x_true=(128,128,3) elemental maps, y=(128,128,3) Poisson noisy counts
#    reconstruction=y/y.max()
# ═══════════════════════════════════════════════════════════════════════════════

def make_edx_phantom(rng: np.random.Generator) -> np.ndarray:
    """Three elemental concentration maps (128x128x3) in [0,1]."""
    H, W = 128, 128
    maps = np.zeros((H, W, 3), dtype=np.float32)
    for ch in range(3):
        # Random Gaussian blobs for each element
        n_blobs = rng.integers(3, 8)
        for _ in range(n_blobs):
            cy = rng.integers(10, H - 10)
            cx = rng.integers(10, W - 10)
            sigma = rng.uniform(5, 20)
            amplitude = rng.uniform(0.3, 1.0)
            yy, xx = np.mgrid[0:H, 0:W]
            blob = amplitude * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
            maps[:, :, ch] += blob
        # Normalize channel to [0,1]
        ch_max = maps[:, :, ch].max()
        if ch_max > 0:
            maps[:, :, ch] /= ch_max
    return maps.astype(np.float32)


def generate_edx_sample(rng: np.random.Generator):
    x_true = make_edx_phantom(rng)  # (128,128,3)
    # Forward: Poisson noise (scale by 100 counts)
    scale = 100.0
    counts = rng.poisson(x_true * scale).astype(np.float32)
    y = counts  # (128,128,3)
    # H_ideal: identity-like (scaling factor map)
    H_ideal = np.full((128, 128, 3), scale, dtype=np.float32)
    # Baseline: normalize measured counts
    y_max = y.max()
    reconstruction_baseline = (y / y_max).astype(np.float32) if y_max > 0 else y
    return x_true, y, H_ideal, reconstruction_baseline


# ═══════════════════════════════════════════════════════════════════════════════
# 2. EHT Imaging (Event Horizon Telescope)
#    x_true=(128,128) black hole shadow ring pattern
#    y=interferometric noise in Fourier space → abs
#    reconstruction=IFFT after noise cleaning
# ═══════════════════════════════════════════════════════════════════════════════

def make_eht_phantom(rng: np.random.Generator) -> np.ndarray:
    """Black hole shadow: bright ring with dark center (128x128)."""
    H, W = 128, 128
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = H / 2 + rng.uniform(-5, 5), W / 2 + rng.uniform(-5, 5)
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    # Ring parameters
    r_ring = rng.uniform(18, 28)
    width = rng.uniform(4, 8)
    ring = np.exp(-((r - r_ring) ** 2) / (2 * width ** 2))
    # Asymmetric brightness (Doppler boosting)
    angle = np.arctan2(yy - cy, xx - cx)
    boost = 1.0 + 0.5 * np.cos(angle + rng.uniform(0, 2 * np.pi))
    img = ring * boost
    img = (img / img.max()).astype(np.float32)
    return img


def generate_eht_sample(rng: np.random.Generator):
    x_true = make_eht_phantom(rng)
    # Forward: Fourier-domain interferometric noise
    F = fft2(x_true)
    Fshift = fftshift(F)
    # Add complex Gaussian noise (sparse uv-coverage noise)
    noise_level = rng.uniform(0.05, 0.15) * np.abs(Fshift).max()
    noise = (rng.standard_normal(Fshift.shape) + 1j * rng.standard_normal(Fshift.shape)) * noise_level
    Fshift_noisy = Fshift + noise
    y = np.abs(ifft2(ifftshift(Fshift_noisy))).astype(np.float32)
    y = np.clip(y, 0, None)
    y = (y / y.max()).astype(np.float32)
    # H_ideal: Fourier sampling mask (all ones = full coverage idealization)
    H_ideal = np.ones((128, 128), dtype=np.float32)
    # Reconstruction: CLEAN-like - just IFFT of amplitude-corrected spectrum
    # Threshold small Fourier components (noise cleaning)
    Fclean = fftshift(fft2(y))
    thresh = np.percentile(np.abs(Fclean), 10)
    Fclean[np.abs(Fclean) < thresh] = 0
    reconstruction_baseline = np.abs(ifft2(ifftshift(Fclean))).astype(np.float32)
    reconstruction_baseline = (reconstruction_baseline / reconstruction_baseline.max()).astype(np.float32)
    return x_true, y, H_ideal, reconstruction_baseline


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Elastography
#    x_true=(128,128) stiffness map
#    y=x_true convolved with wave kernel (sin pattern) + noise
#    reconstruction=deconvolution (Wiener)
# ═══════════════════════════════════════════════════════════════════════════════

def make_elastography_phantom(rng: np.random.Generator) -> np.ndarray:
    """Stiffness map: background + inclusions (128x128)."""
    H, W = 128, 128
    img = np.ones((H, W), dtype=np.float32) * rng.uniform(0.2, 0.4)
    # Add 2-4 stiff inclusions
    n_inc = rng.integers(2, 5)
    for _ in range(n_inc):
        cy = rng.integers(20, H - 20)
        cx = rng.integers(20, W - 20)
        ry = rng.integers(10, 30)
        rx = rng.integers(10, 30)
        stiffness = rng.uniform(0.6, 1.0)
        yy, xx = np.mgrid[0:H, 0:W]
        mask = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 < 1
        img[mask] = stiffness
    img = gaussian_filter(img, sigma=1.5)
    img = (img - img.min()) / (img.max() - img.min())
    return img.astype(np.float32)


def make_wave_kernel(rng: np.random.Generator, shape=(128, 128)) -> np.ndarray:
    """Shear wave propagation kernel (sinusoidal)."""
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W]
    freq = rng.uniform(0.05, 0.15)
    angle = rng.uniform(0, np.pi)
    k = 2 * np.pi * freq
    wave = np.sin(k * (xx * np.cos(angle) + yy * np.sin(angle)))
    # Localize the kernel
    cy, cx = H // 2, W // 2
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    kernel = wave * np.exp(-r ** 2 / (2 * 20 ** 2))
    kernel = kernel.astype(np.float32)
    return kernel


def generate_elastography_sample(rng: np.random.Generator):
    x_true = make_elastography_phantom(rng)
    kernel = make_wave_kernel(rng)
    # Forward: convolve stiffness map with wave kernel (Fourier convolution)
    Fx = fft2(x_true)
    Fk = fft2(kernel)
    Fk_norm = Fk / (np.abs(Fk).max() + 1e-8)
    y_clean = np.real(ifft2(Fx * Fk_norm))
    noise = rng.standard_normal(y_clean.shape).astype(np.float32) * 0.05
    y = (y_clean + noise).astype(np.float32)
    # Normalize y to [-1, 1] range
    y_abs_max = np.abs(y).max()
    if y_abs_max > 0:
        y = y / y_abs_max
    # H_ideal: the wave kernel
    H_ideal = kernel
    # Reconstruction: Wiener deconvolution
    snr = 10.0
    Fy = fft2(y)
    Fk_conj = np.conj(Fk_norm)
    Fk_abs2 = np.abs(Fk_norm) ** 2
    Fx_est = Fy * Fk_conj / (Fk_abs2 + 1.0 / snr)
    reconstruction_baseline = np.real(ifft2(Fx_est)).astype(np.float32)
    r_min, r_max = reconstruction_baseline.min(), reconstruction_baseline.max()
    if r_max - r_min > 1e-10:
        reconstruction_baseline = (reconstruction_baseline - r_min) / (r_max - r_min)
    return x_true, y.astype(np.float32), H_ideal, reconstruction_baseline.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Electron Diffraction
#    x_true=(128,128) crystal diffraction pattern (lattice of bright spots)
#    y=x_true + Poisson noise
#    reconstruction=direct (y as baseline)
# ═══════════════════════════════════════════════════════════════════════════════

def make_electron_diffraction_phantom(rng: np.random.Generator) -> np.ndarray:
    """Crystal diffraction pattern: central beam + lattice spots (128x128)."""
    H, W = 128, 128
    img = np.zeros((H, W), dtype=np.float32)
    cy, cx = H // 2, W // 2
    # Central beam
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    img += np.exp(-r ** 2 / (2 * 3 ** 2))
    # Lattice parameters
    a1 = rng.uniform(12, 22)
    a2 = rng.uniform(12, 22)
    theta = rng.uniform(0, np.pi / 3)
    # Basis vectors
    v1 = np.array([a1 * np.cos(theta), a1 * np.sin(theta)])
    v2 = np.array([a2 * np.cos(theta + np.pi / 2), a2 * np.sin(theta + np.pi / 2)])
    # Generate lattice points
    for h in range(-4, 5):
        for k in range(-4, 5):
            if h == 0 and k == 0:
                continue
            pos = h * v1 + k * v2
            spot_y = int(cy + pos[1])
            spot_x = int(cx + pos[0])
            if 0 <= spot_y < H and 0 <= spot_x < W:
                intensity = rng.uniform(0.1, 1.0) / (np.sqrt(h ** 2 + k ** 2) + 0.1)
                spot_r = np.sqrt((yy - spot_y) ** 2 + (xx - spot_x) ** 2)
                img += intensity * np.exp(-spot_r ** 2 / (2 * 1.5 ** 2))
    img = np.clip(img, 0, None)
    img = (img / img.max()).astype(np.float32)
    return img


def generate_electron_diffraction_sample(rng: np.random.Generator):
    x_true = make_electron_diffraction_phantom(rng)
    # Forward: Poisson noise
    scale = 200.0
    counts = rng.poisson(x_true * scale).astype(np.float32)
    y = (counts / counts.max()).astype(np.float32)
    # H_ideal: identity (scaling)
    H_ideal = np.eye(128, dtype=np.float32)
    # Baseline: direct measured pattern
    reconstruction_baseline = y.copy()
    return x_true, y, H_ideal, reconstruction_baseline


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Electron Holography
#    x_true=(128,128) phase map
#    y=cos(x_true + reference_wave) — hologram
#    reconstruction=phase extraction via Fourier filtering
# ═══════════════════════════════════════════════════════════════════════════════

def make_holography_phase_phantom(rng: np.random.Generator) -> np.ndarray:
    """Phase map: smooth potential field (128x128) in [-pi, pi]."""
    H, W = 128, 128
    phase = np.zeros((H, W), dtype=np.float32)
    # Smooth potential from random Gaussians
    yy, xx = np.mgrid[0:H, 0:W]
    n_sources = rng.integers(2, 6)
    for _ in range(n_sources):
        cy = rng.uniform(20, H - 20)
        cx = rng.uniform(20, W - 20)
        sigma = rng.uniform(15, 35)
        amp = rng.uniform(0.5, 1.5) * rng.choice([-1, 1])
        phase += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
    # Scale to [-pi, pi]
    phase = phase / (np.abs(phase).max() + 1e-8) * np.pi
    return phase.astype(np.float32)


def generate_electron_holography_sample(rng: np.random.Generator):
    x_true = make_holography_phase_phantom(rng)  # phase map
    H, W = x_true.shape
    yy, xx = np.mgrid[0:H, 0:W]
    # Reference wave: tilted plane wave
    carrier_freq = rng.uniform(0.1, 0.2)  # cycles/pixel
    reference_wave = 2 * np.pi * carrier_freq * xx
    # Hologram: interference pattern
    y = np.cos(x_true + reference_wave).astype(np.float32)
    # H_ideal: carrier frequency map
    H_ideal = np.cos(reference_wave).astype(np.float32)
    # Reconstruction: Fourier filtering to extract sideband
    Fy = fftshift(fft2(y))
    # Find sideband: the hologram has sidebands at ±carrier_freq
    # Create a bandpass mask around the sideband
    freq_y = np.fft.fftfreq(H)
    freq_x = np.fft.fftfreq(W)
    fy, fx = np.meshgrid(np.fft.fftshift(freq_y), np.fft.fftshift(freq_x), indexing='ij')
    # Sideband at positive carrier frequency
    sideband_x = carrier_freq
    radius = carrier_freq * 0.8
    mask = np.sqrt((fx - sideband_x) ** 2 + fy ** 2) < radius
    Fsideband = Fy * mask
    # Shift sideband to center
    # Shift by -carrier_freq in x
    Fshifted = np.roll(Fsideband, -int(carrier_freq * W), axis=1)
    sideband_complex = ifft2(ifftshift(Fshifted))
    phase_reconstructed = np.angle(sideband_complex).astype(np.float32)
    # Normalize to [0,1] for storage
    reconstruction_baseline = (phase_reconstructed - phase_reconstructed.min()) / (
        phase_reconstructed.max() - phase_reconstructed.min() + 1e-10
    )
    reconstruction_baseline = reconstruction_baseline.astype(np.float32)
    # x_true normalized for storage
    x_true_norm = (x_true - x_true.min()) / (x_true.max() - x_true.min() + 1e-10)
    x_true_norm = x_true_norm.astype(np.float32)
    return x_true_norm, y, H_ideal, reconstruction_baseline


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Electron Tomography
#    x_true=(64,64) cross-section
#    y=(60,64) projections at -60 to 60 deg
#    reconstruction=FBP
# ═══════════════════════════════════════════════════════════════════════════════

def make_electron_tomography_phantom(rng: np.random.Generator) -> np.ndarray:
    """2D cross-section of a nanoparticle (64x64)."""
    H, W = 64, 64
    img = np.zeros((H, W), dtype=np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = H / 2, W / 2
    # Core particle
    r_core = rng.uniform(10, 18)
    core_density = rng.uniform(0.7, 1.0)
    img += core_density * (np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) < r_core).astype(np.float32)
    # Shell
    r_shell = r_core + rng.uniform(2, 5)
    shell_density = rng.uniform(0.3, 0.6)
    shell_mask = (np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) < r_shell) & (
        np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) >= r_core
    )
    img += shell_density * shell_mask.astype(np.float32)
    # Small inclusions
    n_inc = rng.integers(0, 3)
    for _ in range(n_inc):
        iy = rng.integers(10, H - 10)
        ix = rng.integers(10, W - 10)
        r_inc = rng.uniform(2, 5)
        density = rng.uniform(0.8, 1.0)
        inc_mask = np.sqrt((yy - iy) ** 2 + (xx - ix) ** 2) < r_inc
        img += density * inc_mask.astype(np.float32)
    img = gaussian_filter(img, sigma=0.5)
    img = np.clip(img, 0, None)
    if img.max() > 0:
        img /= img.max()
    return img.astype(np.float32)


def radon_project(img: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    """Simple Radon transform via rotation + sum."""
    from scipy.ndimage import rotate
    H, W = img.shape
    n_angles = len(angles_deg)
    sino = np.zeros((n_angles, W), dtype=np.float32)
    for i, ang in enumerate(angles_deg):
        rotated = rotate(img, -ang, reshape=False, order=1, mode='constant', cval=0)
        sino[i] = rotated.sum(axis=0)
    return sino


def fbp_reconstruct(sino: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    """Simple filtered back-projection."""
    from scipy.ndimage import rotate
    n_angles, n_det = sino.shape
    # Ram-Lak filter in Fourier domain
    freqs = np.fft.fftfreq(n_det)
    filt = np.abs(freqs)
    # Apply filter to each projection
    sino_filt = np.zeros_like(sino)
    for i in range(n_angles):
        F = np.fft.fft(sino[i])
        sino_filt[i] = np.real(np.fft.ifft(F * filt))
    # Back-project
    recon = np.zeros((n_det, n_det), dtype=np.float32)
    for i, ang in enumerate(angles_deg):
        proj_2d = np.tile(sino_filt[i], (n_det, 1))
        recon += rotate(proj_2d, ang, reshape=False, order=1, mode='constant', cval=0)
    recon *= np.pi / (2 * n_angles)
    recon = np.clip(recon, 0, None)
    if recon.max() > 0:
        recon /= recon.max()
    return recon.astype(np.float32)


def generate_electron_tomography_sample(rng: np.random.Generator):
    x_true = make_electron_tomography_phantom(rng)  # (64,64)
    angles = np.linspace(-60, 60, 60)
    # Forward: projections
    sino = radon_project(x_true, angles)  # (60,64)
    noise = rng.standard_normal(sino.shape).astype(np.float32) * 0.02
    y = (sino + noise).astype(np.float32)
    y = np.clip(y, 0, None)
    # H_ideal: angles array as 1D (stored as 1D array)
    H_ideal = angles.astype(np.float32)  # (60,) angle array
    # FBP reconstruction
    reconstruction_baseline = fbp_reconstruct(y, angles)
    return x_true, y, H_ideal, reconstruction_baseline


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Entangled Photon Imaging
#    x_true=(64,64) sparse bright spots
#    y=(64,64) coincidence counts with Poisson noise
#    reconstruction=y (direct)
# ═══════════════════════════════════════════════════════════════════════════════

def make_entangled_photon_phantom(rng: np.random.Generator) -> np.ndarray:
    """Sparse bright spots on dark background (64x64)."""
    H, W = 64, 64
    img = np.zeros((H, W), dtype=np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    # 3-8 point-like sources
    n_sources = rng.integers(3, 9)
    for _ in range(n_sources):
        cy = rng.integers(5, H - 5)
        cx = rng.integers(5, W - 5)
        sigma = rng.uniform(1.5, 3.5)
        amplitude = rng.uniform(0.3, 1.0)
        r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        img += amplitude * np.exp(-r ** 2 / (2 * sigma ** 2))
    img = np.clip(img, 0, None)
    if img.max() > 0:
        img /= img.max()
    return img.astype(np.float32)


def generate_entangled_photon_sample(rng: np.random.Generator):
    x_true = make_entangled_photon_phantom(rng)  # (64,64)
    # Forward: Poisson coincidence counts
    scale = 50.0
    counts = rng.poisson(x_true * scale).astype(np.float32)
    y = (counts / (counts.max() + 1e-8)).astype(np.float32)
    # H_ideal: coincidence detection efficiency (uniform)
    H_ideal = np.ones((64, 64), dtype=np.float32) * scale
    # Reconstruction: direct (y as estimate)
    reconstruction_baseline = y.copy()
    return x_true, y, H_ideal, reconstruction_baseline


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Event Camera
#    x_true=(128,128) event-based intensity reconstruction
#    y=(128,128) event polarity map (±1 values)
#    reconstruction=cumsum integration
# ═══════════════════════════════════════════════════════════════════════════════

def make_event_camera_scene(rng: np.random.Generator) -> np.ndarray:
    """Synthetic scene with edges and gradients (128x128)."""
    H, W = 128, 128
    img = np.zeros((H, W), dtype=np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    # Background gradient
    img += rng.uniform(0.1, 0.3) * xx / W
    # Add geometric objects
    n_objects = rng.integers(2, 5)
    for _ in range(n_objects):
        cy = rng.integers(20, H - 20)
        cx = rng.integers(20, W - 20)
        r = rng.integers(10, 30)
        brightness = rng.uniform(0.3, 0.9)
        mask = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) < r
        img[mask] = brightness
    # Add some texture
    img += gaussian_filter(rng.standard_normal((H, W)).astype(np.float32), sigma=3) * 0.05
    img = np.clip(img, 0, 1)
    img = (img - img.min()) / (img.max() - img.min() + 1e-10)
    return img.astype(np.float32)


def generate_event_camera_sample(rng: np.random.Generator):
    x_true = make_event_camera_scene(rng)  # (128,128) intensity
    # Event polarity map: sign of spatial gradient (edge detector)
    # Events fire when log intensity changes exceed threshold
    log_img = np.log(x_true + 0.01)
    # Compute local gradient magnitude and sign
    threshold = 0.15
    # Simulate events as: bright edges -> +1, dark edges -> -1
    from scipy.ndimage import sobel
    gx = sobel(log_img, axis=1)
    gy = sobel(log_img, axis=0)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    # Polarity based on gradient direction (x-dominant)
    polarity = np.sign(gx) * (grad_mag > threshold)
    y = polarity.astype(np.float32)
    # H_ideal: threshold value
    H_ideal = np.full((128, 128), threshold, dtype=np.float32)
    # Reconstruction: cumulative sum integration along x-axis
    # Integrate events to recover intensity
    recon = np.cumsum(y, axis=1)
    recon_min, recon_max = recon.min(), recon.max()
    if recon_max - recon_min > 1e-10:
        recon = (recon - recon_min) / (recon_max - recon_min)
    reconstruction_baseline = recon.astype(np.float32)
    return x_true, y, H_ideal, reconstruction_baseline


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Expansion Microscopy
#    x_true=(128,128) high-res expansion image
#    y=x_true downsampled 2x + noise (64x64 → stored as 64x64)
#    reconstruction=bicubic upsample to 128x128
# ═══════════════════════════════════════════════════════════════════════════════

def make_expansion_phantom(rng: np.random.Generator) -> np.ndarray:
    """High-resolution cell structure (128x128) for expansion microscopy."""
    H, W = 128, 128
    img = np.zeros((H, W), dtype=np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    # Cell membrane network (circles)
    n_cells = rng.integers(3, 7)
    for _ in range(n_cells):
        cy = rng.integers(20, H - 20)
        cx = rng.integers(20, W - 20)
        r = rng.integers(12, 25)
        thickness = rng.uniform(1.5, 3.0)
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        membrane = np.exp(-((dist - r) ** 2) / (2 * thickness ** 2))
        intensity = rng.uniform(0.5, 1.0)
        img += intensity * membrane
    # Organelles (bright spots)
    n_org = rng.integers(5, 15)
    for _ in range(n_org):
        oy = rng.integers(5, H - 5)
        ox = rng.integers(5, W - 5)
        sigma = rng.uniform(1.0, 3.0)
        amp = rng.uniform(0.3, 0.8)
        dist = np.sqrt((yy - oy) ** 2 + (xx - ox) ** 2)
        img += amp * np.exp(-dist ** 2 / (2 * sigma ** 2))
    img = np.clip(img, 0, None)
    if img.max() > 0:
        img /= img.max()
    return img.astype(np.float32)


def generate_expansion_sample(rng: np.random.Generator):
    x_true = make_expansion_phantom(rng)  # (128,128) high-res
    # Forward: downsample 2x + Gaussian noise
    y_lr = zoom(x_true, 0.5, order=3)  # (64,64)
    noise = rng.standard_normal(y_lr.shape).astype(np.float32) * 0.03
    y = np.clip(y_lr + noise, 0, 1).astype(np.float32)
    # H_ideal: downsampling operator representation (averaging kernel)
    kernel_1d = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    H_ideal = np.outer(kernel_1d, kernel_1d).astype(np.float32)  # (3,3) downsampling kernel
    # Reconstruction: bicubic upsample
    reconstruction_baseline = zoom(y, 2.0, order=3).astype(np.float32)  # (128,128)
    reconstruction_baseline = np.clip(reconstruction_baseline, 0, 1).astype(np.float32)
    return x_true, y, H_ideal, reconstruction_baseline


# ═══════════════════════════════════════════════════════════════════════════════
# 10. FIB-SEM
#     x_true=(128,128) material cross-section
#     y=x_true + Gaussian noise (sigma=0.05)
#     reconstruction=Gaussian smoothing
# ═══════════════════════════════════════════════════════════════════════════════

def make_fib_sem_phantom(rng: np.random.Generator) -> np.ndarray:
    """FIB-SEM cross-section: material microstructure (128x128)."""
    H, W = 128, 128
    img = np.zeros((H, W), dtype=np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    # Background matrix material
    img += rng.uniform(0.1, 0.3)
    # Grains / phases via Voronoi-like regions
    n_seeds = rng.integers(8, 20)
    seed_y = rng.integers(0, H, n_seeds)
    seed_x = rng.integers(0, W, n_seeds)
    seed_vals = rng.uniform(0.2, 0.9, n_seeds)
    # Assign each pixel to nearest seed
    for py in range(H):
        for px in range(W):
            dists = (seed_y - py) ** 2 + (seed_x - px) ** 2
            nearest = np.argmin(dists)
            img[py, px] = seed_vals[nearest]
    # Add bright grain boundary lines (edge emphasis)
    boundary = np.abs(np.gradient(img)[0]) + np.abs(np.gradient(img)[1])
    boundary = (boundary / (boundary.max() + 1e-8)) * rng.uniform(0.3, 0.6)
    img = img * 0.7 + boundary * 0.3
    # Add some porosity (dark voids)
    n_pores = rng.integers(2, 8)
    for _ in range(n_pores):
        py = rng.integers(10, H - 10)
        px = rng.integers(10, W - 10)
        r = rng.uniform(3, 8)
        mask = np.sqrt((yy - py) ** 2 + (xx - px) ** 2) < r
        img[mask] = rng.uniform(0.0, 0.05)
    img = np.clip(img, 0, 1)
    img = (img - img.min()) / (img.max() - img.min() + 1e-10)
    return img.astype(np.float32)


def generate_fib_sem_sample(rng: np.random.Generator):
    x_true = make_fib_sem_phantom(rng)  # (128,128)
    # Forward: add Gaussian noise
    noise = rng.standard_normal(x_true.shape).astype(np.float32) * 0.05
    y = np.clip(x_true + noise, 0, 1).astype(np.float32)
    # H_ideal: noise sigma map (uniform)
    H_ideal = np.full((128, 128), 0.05, dtype=np.float32)
    # Reconstruction: Gaussian smoothing
    reconstruction_baseline = gaussian_filter(y, sigma=1.0).astype(np.float32)
    reconstruction_baseline = np.clip(reconstruction_baseline, 0, 1).astype(np.float32)
    return x_true, y, H_ideal, reconstruction_baseline


# ═══════════════════════════════════════════════════════════════════════════════
# Dispatch table
# ═══════════════════════════════════════════════════════════════════════════════

GENERATORS = {
    "edx_mapping":          generate_edx_sample,
    "eht_imaging":          generate_eht_sample,
    "elastography":         generate_elastography_sample,
    "electron_diffraction": generate_electron_diffraction_sample,
    "electron_holography":  generate_electron_holography_sample,
    "electron_tomography":  generate_electron_tomography_sample,
    "entangled_photon":     generate_entangled_photon_sample,
    "event_camera":         generate_event_camera_sample,
    "expansion":            generate_expansion_sample,
    "fib_sem":              generate_fib_sem_sample,
}

# Spec templates per modality
SPECS = {
    "edx_mapping": {
        "modality": "edx_mapping",
        "x_shape": [128, 128, 3],
        "y_shape": [128, 128, 3],
        "H_shape": [128, 128, 3],
        "task": "elemental_map_reconstruction",
        "metric": "NMSE",
        "units": {"x": "concentration [0-1]", "y": "photon_counts"},
    },
    "eht_imaging": {
        "modality": "eht_imaging",
        "x_shape": [128, 128],
        "y_shape": [128, 128],
        "H_shape": [128, 128],
        "task": "black_hole_image_reconstruction",
        "metric": "SSIM",
        "units": {"x": "brightness [0-1]", "y": "visibility_amplitude"},
    },
    "elastography": {
        "modality": "elastography",
        "x_shape": [128, 128],
        "y_shape": [128, 128],
        "H_shape": [128, 128],
        "task": "stiffness_map_reconstruction",
        "metric": "NMSE",
        "units": {"x": "stiffness [0-1]", "y": "wave_displacement"},
    },
    "electron_diffraction": {
        "modality": "electron_diffraction",
        "x_shape": [128, 128],
        "y_shape": [128, 128],
        "H_shape": [128, 128],
        "task": "diffraction_pattern_denoising",
        "metric": "SSIM",
        "units": {"x": "intensity [0-1]", "y": "electron_counts"},
    },
    "electron_holography": {
        "modality": "electron_holography",
        "x_shape": [128, 128],
        "y_shape": [128, 128],
        "H_shape": [128, 128],
        "task": "phase_reconstruction",
        "metric": "NMSE",
        "units": {"x": "phase [0-1 normalized]", "y": "hologram_intensity"},
    },
    "electron_tomography": {
        "modality": "electron_tomography",
        "x_shape": [64, 64],
        "y_shape": [60, 64],
        "H_shape": [60],
        "task": "tomographic_reconstruction",
        "metric": "SSIM",
        "units": {"x": "density [0-1]", "y": "projection_intensity"},
    },
    "entangled_photon": {
        "modality": "entangled_photon",
        "x_shape": [64, 64],
        "y_shape": [64, 64],
        "H_shape": [64, 64],
        "task": "quantum_image_reconstruction",
        "metric": "SSIM",
        "units": {"x": "intensity [0-1]", "y": "coincidence_counts"},
    },
    "event_camera": {
        "modality": "event_camera",
        "x_shape": [128, 128],
        "y_shape": [128, 128],
        "H_shape": [128, 128],
        "task": "intensity_reconstruction_from_events",
        "metric": "SSIM",
        "units": {"x": "intensity [0-1]", "y": "event_polarity [-1,0,+1]"},
    },
    "expansion": {
        "modality": "expansion",
        "x_shape": [128, 128],
        "y_shape": [64, 64],
        "H_shape": [3, 3],
        "task": "super_resolution_reconstruction",
        "metric": "SSIM",
        "units": {"x": "intensity [0-1]", "y": "confocal_intensity [0-1]"},
    },
    "fib_sem": {
        "modality": "fib_sem",
        "x_shape": [128, 128],
        "y_shape": [128, 128],
        "H_shape": [128, 128],
        "task": "image_denoising",
        "metric": "SSIM",
        "units": {"x": "material_contrast [0-1]", "y": "noisy_sem_signal [0-1]"},
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset writer
# ═══════════════════════════════════════════════════════════════════════════════

def write_tier(modality: str, tier: str, n_samples: int):
    out_dir = OUT_BASE / modality / tier
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs_dir = out_dir / "images"
    imgs_dir.mkdir(exist_ok=True)

    h5_path = out_dir / f"{modality}_challenge_{tier}.h5"
    generator_fn = GENERATORS[modality]
    spec = SPECS[modality]

    with h5py.File(h5_path, "w") as hf:
        hf.attrs["modality"] = modality
        hf.attrs["tier"] = tier
        hf.attrs["n_samples"] = n_samples

        for i in range(n_samples):
            seed = get_seed(modality, tier, i)
            rng = np.random.default_rng(seed)
            x_true, y, H_ideal, recon = generator_fn(rng)

            grp = hf.create_group(f"sample_{i:02d}")
            grp.create_dataset("x_true", data=x_true.astype(np.float32), compression="gzip")
            grp.create_dataset("y", data=y.astype(np.float32), compression="gzip")
            grp.create_dataset("H_ideal", data=H_ideal.astype(np.float32), compression="gzip")
            grp.create_dataset("reconstruction_baseline", data=recon.astype(np.float32), compression="gzip")
            grp.attrs["seed"] = seed

            # Save PNG previews
            sample_img_dir = imgs_dir / f"sample_{i:02d}"
            sample_img_dir.mkdir(exist_ok=True)
            save_png(x_true, sample_img_dir / "x_true.png")
            save_png(y, sample_img_dir / "y.png")
            save_png(H_ideal, sample_img_dir / "H_ideal.png")
            save_png(recon, sample_img_dir / "reconstruction_baseline.png")

    # Write spec.json (public-facing, no ground truth shapes of hidden data)
    spec_public = {**spec, "tier": tier, "n_samples": n_samples, "version": "1.0"}
    with open(out_dir / "spec.json", "w") as f:
        json.dump(spec_public, f, indent=2)

    # Write true_spec.json (full details)
    true_spec = {
        **spec,
        "tier": tier,
        "n_samples": n_samples,
        "version": "1.0",
        "seed_formula": f"MODALITY_OFFSET[{modality}] + TIER_BASE[{tier}] + sample_idx * 17",
        "h5_path": str(h5_path),
    }
    with open(out_dir / "true_spec.json", "w") as f:
        json.dump(true_spec, f, indent=2)

    print(f"  [{tier:6s}] {n_samples} samples -> {h5_path.relative_to(ROOT)}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"Batch 4 dataset generation -- {len(MODALITIES)} modalities")
    print(f"Output base: {OUT_BASE}\n")

    for modality in MODALITIES:
        print(f"[{modality}]")
        for tier, n_samples in TIER_SIZES.items():
            try:
                write_tier(modality, tier, n_samples)
            except Exception as e:
                import traceback
                print(f"  ERROR in {modality}/{tier}: {e}")
                traceback.print_exc()
        print()

    print("Done.")


if __name__ == "__main__":
    main()
