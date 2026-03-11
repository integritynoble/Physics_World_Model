"""
generate_batch11_datasets.py
Generate benchmark datasets for batch 11 modalities:
  tirf, tof_camera, ultrasonic_phased_array, us_mri, waxs,
  weather_radar, widefield_lowdose, xfel_sfx, xray_crystallography, xray_ndt

Tiers: public (12 samples), dev (20 samples), hidden (20 samples)
Seeds: public=1500+i*17, dev=7500+i*17, hidden=9550+i*17
"""

import os
import json
import numpy as np
import h5py
from PIL import Image
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import fftconvolve

ROOT = "D:/onedrive/startup/program/physics_world_model/PWM5/Physics_World_Model"
DATASETS_BASE = os.path.join(ROOT, "datasets", "benchmark")

TIER_CONFIGS = {
    "public": {"n_samples": 12, "seed_base": 1500},
    "dev":    {"n_samples": 20, "seed_base": 7500},
    "hidden": {"n_samples": 20, "seed_base": 9550},
}

SEED_STEP = 17

# ---------------------------------------------------------------------------
# Physics simulation functions
# ---------------------------------------------------------------------------

def sim_tirf(rng, idx):
    """TIRF: near-surface fluorescence with evanescent-field PSF."""
    H = 128
    # x_true: sparse fluorescent emitters near surface
    x_true = np.zeros((H, H), dtype=np.float32)
    n_emitters = rng.integers(30, 80)
    xs = rng.integers(0, H, size=n_emitters)
    ys = rng.integers(0, H, size=n_emitters)
    intensities = rng.uniform(0.3, 1.0, size=n_emitters).astype(np.float32)
    for xi, yi, iv in zip(xs, ys, intensities):
        x_true[xi, yi] += iv

    # PSF: narrow Gaussian (sigma=1.5) simulating evanescent-field confinement
    sigma_psf = 1.5
    # H_ideal: PSF kernel (stored as 128x128 centred kernel)
    H_ideal = np.zeros((H, H), dtype=np.float32)
    cy, cx = H // 2, H // 2
    for r in range(H):
        for c in range(H):
            H_ideal[r, c] = np.exp(-((r - cy)**2 + (c - cx)**2) / (2 * sigma_psf**2))
    H_ideal /= H_ideal.sum()

    # y: convolved signal + background fluorescence
    background = rng.uniform(0.02, 0.08)
    y = gaussian_filter(x_true, sigma=sigma_psf) + background
    y = y.astype(np.float32)

    # reconstruction: background subtraction + 5-iter RL deconv
    y_bg_sub = np.clip(y - background, 1e-6, None)
    psf_1d = np.array([[np.exp(-((r - cy)**2 + (c - cx)**2) / (2 * sigma_psf**2))
                        for c in range(H)] for r in range(H)], dtype=np.float32)
    psf_1d /= psf_1d.sum()
    recon = y_bg_sub.copy()
    for _ in range(5):
        conv = gaussian_filter(recon, sigma=sigma_psf) + 1e-8
        recon = recon * gaussian_filter(y_bg_sub / conv, sigma=sigma_psf)
        recon = np.clip(recon, 0, None)
    recon = recon.astype(np.float32)

    return x_true, y, H_ideal, recon


def sim_tof_camera(rng, idx):
    """ToF camera: depth map reconstruction from raw phase image."""
    H = 128
    # x_true: smooth depth map [0,1]
    depth = rng.uniform(0.1, 0.9)
    x_true = np.zeros((H, H), dtype=np.float32)
    # Create regions at different depths
    for _ in range(rng.integers(2, 5)):
        cx, cy = rng.integers(20, H - 20, size=2)
        r = rng.integers(10, 35)
        d = rng.uniform(0.1, 0.9)
        yy, xx = np.ogrid[:H, :H]
        mask = (xx - cx)**2 + (yy - cy)**2 < r**2
        x_true[mask] = d
    x_true = gaussian_filter(x_true + rng.uniform(0.05, 0.2), sigma=3).astype(np.float32)
    x_true = (x_true - x_true.min()) / (x_true.max() - x_true.min() + 1e-8)

    # H_ideal: identity-like (direct phase→depth mapping)
    H_ideal = np.eye(H, dtype=np.float32)

    # y: raw phase image with multipath interference noise
    phase = x_true * 2 * np.pi
    multipath_noise = rng.normal(0, 0.05, (H, H)).astype(np.float32)
    multipath_noise += 0.03 * np.sin(phase * 3 + rng.uniform(0, np.pi))
    y = (phase + multipath_noise).astype(np.float32)

    # reconstruction: phase unwrapping (use y directly, normalize)
    recon = (y / (2 * np.pi)).astype(np.float32)
    recon = np.clip(recon, 0, 1)

    return x_true, y, H_ideal, recon


def sim_ultrasonic_phased_array(rng, idx):
    """Ultrasonic phased array: defect map from delay-and-sum traces."""
    H = 128
    # x_true: defect map with sparse point/line defects
    x_true = np.zeros((H, H), dtype=np.float32)
    n_defects = rng.integers(3, 8)
    for _ in range(n_defects):
        cx, cy = rng.integers(10, H - 10, size=2)
        kind = rng.choice(['point', 'line', 'blob'])
        if kind == 'point':
            x_true[cx, cy] = rng.uniform(0.5, 1.0)
        elif kind == 'line':
            length = rng.integers(5, 20)
            x_true[cx:cx+length, cy] = rng.uniform(0.4, 0.9)
        else:
            r = rng.integers(3, 10)
            yy, xx = np.ogrid[:H, :H]
            mask = (xx - cy)**2 + (yy - cx)**2 < r**2
            x_true[mask] = rng.uniform(0.3, 0.8)
    x_true = gaussian_filter(x_true, sigma=0.8).astype(np.float32)
    x_true /= (x_true.max() + 1e-8)

    # H_ideal: (64,128) — DAS projection operator (stored as 2D)
    H_ideal = np.zeros((64, H), dtype=np.float32)
    for i in range(64):
        H_ideal[i, i * 2] = 1.0

    # y: (64,128) delay-and-sum raw traces
    y = np.zeros((64, H), dtype=np.float32)
    for i in range(64):
        # Simulate DAS: project along diagonal + noise
        y[i, :] = x_true[i * 2, :] + rng.normal(0, 0.05, H).astype(np.float32)

    # reconstruction: FBP (simple backprojection)
    recon = np.zeros((H, H), dtype=np.float32)
    for i in range(64):
        recon[i * 2, :] += y[i, :]
        if i * 2 + 1 < H:
            recon[i * 2 + 1, :] += y[i, :]
    recon = gaussian_filter(recon, sigma=1.0).astype(np.float32)
    recon /= (recon.max() + 1e-8)

    return x_true, y, H_ideal, recon


def sim_us_mri(rng, idx):
    """US-MRI hybrid: combined ultrasound-MRI image reconstruction."""
    H = 128
    # x_true: hybrid anatomical image (soft tissue contrast)
    x_true = np.zeros((H, H), dtype=np.float32)
    # Background tissue
    x_true += rng.uniform(0.1, 0.3)
    # Anatomical structures
    for _ in range(rng.integers(3, 7)):
        cx, cy = rng.integers(20, H - 20, size=2)
        r = rng.integers(8, 25)
        yy, xx = np.ogrid[:H, :H]
        mask = (xx - cy)**2 + (yy - cx)**2 < r**2
        x_true[mask] = rng.uniform(0.4, 1.0)
    x_true = gaussian_filter(x_true, sigma=2).astype(np.float32)
    x_true = (x_true - x_true.min()) / (x_true.max() - x_true.min() + 1e-8)

    # H_ideal: blur kernel (Gaussian, sigma=2)
    H_ideal = np.zeros((H, H), dtype=np.float32)
    cy, cx = H // 2, H // 2
    sigma_h = 2.0
    for r in range(H):
        for c in range(H):
            H_ideal[r, c] = np.exp(-((r - cy)**2 + (c - cx)**2) / (2 * sigma_h**2))
    H_ideal /= H_ideal.sum()

    # y: degraded measurement (blurred + combined US/MRI noise)
    blurred = gaussian_filter(x_true, sigma=2.0)
    us_noise = rng.rayleigh(0.03, (H, H)).astype(np.float32)
    mri_noise = rng.normal(0, 0.02, (H, H)).astype(np.float32)
    y = (blurred + us_noise + mri_noise).astype(np.float32)
    y = np.clip(y, 0, None)

    # reconstruction: Wiener deconvolution
    from numpy.fft import fft2, ifft2, fftshift
    Y = fft2(y)
    H_freq = fft2(H_ideal)
    noise_var = 0.02**2 + 0.03**2
    signal_var = np.var(x_true)
    wiener_k = noise_var / (signal_var + 1e-8)
    H_conj = np.conj(H_freq)
    recon_freq = H_conj / (np.abs(H_freq)**2 + wiener_k) * Y
    recon = np.real(ifft2(recon_freq)).astype(np.float32)
    recon = np.clip(recon, 0, 1)

    return x_true, y, H_ideal, recon


def sim_waxs(rng, idx):
    """WAXS: wide-angle X-ray scattering pattern from crystalline structure."""
    H = 128
    # x_true: crystalline structure (lattice with disorder)
    x_true = np.zeros((H, H), dtype=np.float32)
    # Create lattice pattern
    spacing = rng.integers(8, 16)
    for i in range(0, H, spacing):
        for j in range(0, H, spacing):
            noise_i = rng.integers(-2, 3)
            noise_j = rng.integers(-2, 3)
            ni, nj = i + noise_i, j + noise_j
            if 0 <= ni < H and 0 <= nj < H:
                x_true[ni, nj] = rng.uniform(0.5, 1.0)
    x_true = gaussian_filter(x_true, sigma=0.5).astype(np.float32)

    # H_ideal: FFT magnitude operator (stored as ones — conceptual)
    H_ideal = np.ones((H, H), dtype=np.float32)

    # y: |FFT2(x_true)|^2 + Poisson noise (WAXS diffraction pattern)
    ft = np.fft.fftshift(np.fft.fft2(x_true))
    intensity = np.abs(ft)**2
    scale = 100.0
    y_poisson = rng.poisson(intensity * scale).astype(np.float32) / scale
    y = y_poisson.astype(np.float32)

    # reconstruction: IFFT of sqrt(y)
    sqrt_y = np.sqrt(np.abs(y))
    recon_ft = np.fft.ifftshift(sqrt_y)
    recon = np.abs(np.fft.ifft2(recon_ft)).astype(np.float32)
    recon = (recon - recon.min()) / (recon.max() - recon.min() + 1e-8)

    return x_true, y, H_ideal, recon


def sim_weather_radar(rng, idx):
    """Weather radar: reflectivity/precipitation map reconstruction."""
    H = 128
    # x_true: precipitation map with storm cells
    x_true = np.zeros((H, H), dtype=np.float32)
    n_cells = rng.integers(2, 6)
    for _ in range(n_cells):
        cx, cy = rng.integers(15, H - 15, size=2)
        r = rng.integers(10, 30)
        intensity = rng.uniform(0.4, 1.0)
        yy, xx = np.ogrid[:H, :H]
        dist = np.sqrt((xx - cy)**2 + (yy - cx)**2)
        cell = intensity * np.exp(-dist**2 / (2 * (r/2)**2))
        x_true += cell
    x_true = np.clip(x_true, 0, 1).astype(np.float32)

    # H_ideal: identity (direct measurement model)
    H_ideal = np.eye(H, dtype=np.float32)

    # y: x_true + range-dependent noise + ground clutter
    # Range-dependent noise: increases with distance from center
    yy, xx = np.ogrid[:H, :H]
    range_dep = np.sqrt((xx - H//2)**2 + (yy - H//2)**2) / (H * np.sqrt(2) / 2)
    range_noise = rng.normal(0, 0.05, (H, H)) * (1 + range_dep)
    # Ground clutter: near-center speckle
    clutter = np.zeros((H, H), dtype=np.float32)
    clutter_mask = range_dep < 0.15
    clutter[clutter_mask] = rng.uniform(0, 0.3, clutter_mask.sum()).astype(np.float32)
    y = (x_true + range_noise + clutter).astype(np.float32)
    y = np.clip(y, 0, None)

    # reconstruction: clutter filter (median filter, 3x3)
    recon = median_filter(y, size=3).astype(np.float32)
    recon = np.clip(recon, 0, 1)

    return x_true, y, H_ideal, recon


def sim_widefield_lowdose(rng, idx):
    """Widefield low-dose: few-photon fluorescence microscopy."""
    H = 128
    # x_true: fluorescence image (cells/organelles)
    x_true = np.zeros((H, H), dtype=np.float32)
    n_cells = rng.integers(3, 8)
    for _ in range(n_cells):
        cx, cy = rng.integers(15, H - 15, size=2)
        r = rng.integers(8, 20)
        brightness = rng.uniform(0.4, 1.0)
        yy, xx = np.ogrid[:H, :H]
        mask = (xx - cy)**2 + (yy - cx)**2 < r**2
        x_true[mask] = brightness
    x_true = gaussian_filter(x_true, sigma=1.5).astype(np.float32)
    x_true = (x_true - x_true.min()) / (x_true.max() - x_true.min() + 1e-8)

    # H_ideal: PSF kernel (Gaussian, sigma=2)
    H_ideal = np.zeros((H, H), dtype=np.float32)
    cy, cx = H // 2, H // 2
    sigma_psf = 2.0
    for r in range(H):
        for c in range(H):
            H_ideal[r, c] = np.exp(-((r - cy)**2 + (c - cx)**2) / (2 * sigma_psf**2))
    H_ideal /= H_ideal.sum()

    # y: low-dose measurement (I0=50 photons, Poisson noise)
    low_dose_factor = 50.0  # photons
    photon_image = x_true * low_dose_factor
    y_counts = rng.poisson(photon_image).astype(np.float32)
    y = (y_counts / low_dose_factor).astype(np.float32)

    # reconstruction: 10-iter RL deconvolution
    recon = y.copy() + 1e-6
    for _ in range(10):
        conv = gaussian_filter(recon, sigma=sigma_psf) + 1e-8
        recon = recon * gaussian_filter(y / conv, sigma=sigma_psf)
        recon = np.clip(recon, 0, None)
    recon = recon.astype(np.float32)
    recon /= (recon.max() + 1e-8)

    return x_true, y, H_ideal, recon


def sim_xfel_sfx(rng, idx):
    """XFEL SFX: serial femtosecond crystallography diffraction pattern."""
    H = 128
    # x_true: protein diffraction pattern (concentric rings + speckles)
    yy, xx = np.ogrid[:H, :H]
    cy, cx = H // 2, H // 2
    r_map = np.sqrt((xx - cx)**2 + (yy - cy)**2)

    x_true = np.zeros((H, H), dtype=np.float32)
    # Bragg rings at specific q-values
    n_rings = rng.integers(3, 7)
    ring_radii = rng.uniform(10, 55, n_rings)
    ring_widths = rng.uniform(1.5, 4.0, n_rings)
    ring_amps = rng.uniform(0.3, 1.0, n_rings)
    for rad, wid, amp in zip(ring_radii, ring_widths, ring_amps):
        x_true += amp * np.exp(-(r_map - rad)**2 / (2 * wid**2))
    # Speckle pattern overlay
    speckle = rng.exponential(0.1, (H, H)).astype(np.float32)
    speckle = gaussian_filter(speckle, sigma=1.5)
    x_true = (x_true + speckle).astype(np.float32)
    x_true /= (x_true.max() + 1e-8)

    # H_ideal: detector sensitivity (flat-field)
    H_ideal = np.ones((H, H), dtype=np.float32)

    # y: shot noise + detector gaps
    shot_scale = 1000.0
    y_counts = rng.poisson(x_true * shot_scale).astype(np.float32) / shot_scale
    # Detector gaps (dead pixels in cross pattern)
    gap_y = H // 2
    gap_x = H // 2
    gap_width = rng.integers(2, 5)
    y_counts[gap_y - gap_width:gap_y + gap_width, :] = -1.0  # gap marker
    y_counts[:, gap_x - gap_width:gap_x + gap_width] = -1.0
    y = y_counts.astype(np.float32)

    # reconstruction: detector gap interpolation + normalize
    gap_mask = y < 0
    y_interp = y.copy()
    from scipy.ndimage import label
    # Replace gaps with local median
    y_valid = np.where(gap_mask, np.nan, y)
    # Simple interpolation: replace with gaussian-smoothed version
    y_filled = y.copy()
    y_filled[gap_mask] = 0
    y_smooth = gaussian_filter(y_filled, sigma=3)
    y_interp[gap_mask] = y_smooth[gap_mask]
    recon = y_interp.astype(np.float32)
    recon = np.clip(recon, 0, None)
    recon /= (recon.max() + 1e-8)

    return x_true, y, H_ideal, recon


def sim_xray_crystallography(rng, idx):
    """X-ray crystallography: electron density map from diffraction intensities."""
    H = 128
    # x_true: electron density map (smooth blobs)
    x_true = np.zeros((H, H), dtype=np.float32)
    n_atoms = rng.integers(5, 15)
    for _ in range(n_atoms):
        cx, cy = rng.integers(10, H - 10, size=2)
        sigma_atom = rng.uniform(2, 6)
        amp = rng.uniform(0.3, 1.0)
        yy, xx = np.ogrid[:H, :H]
        x_true += amp * np.exp(-((xx - cy)**2 + (yy - cx)**2) / (2 * sigma_atom**2))
    x_true = x_true.astype(np.float32)
    x_true /= (x_true.max() + 1e-8)

    # H_ideal: FFT magnitude operator
    H_ideal = np.ones((H, H), dtype=np.float32)

    # y: |FFT2(x_true)|^2 with Friedel symmetry (real pattern)
    ft = np.fft.fft2(x_true)
    intensities = np.abs(ft)**2
    # Apply Friedel symmetry (already satisfied by |FFT|^2)
    # Add noise
    noise_level = 0.01
    y = (intensities + rng.normal(0, noise_level * intensities.max(), (H, H))).astype(np.float32)
    y = np.abs(y)  # intensities must be non-negative

    # reconstruction: direct methods (sqrt amplitude, then IFFT)
    amplitudes = np.sqrt(np.abs(y))
    recon = np.abs(np.fft.ifft2(amplitudes)).astype(np.float32)
    recon = (recon - recon.min()) / (recon.max() - recon.min() + 1e-8)

    return x_true, y, H_ideal, recon


def sim_xray_ndt(rng, idx):
    """X-ray NDT: industrial part with internal defects from radiographic projections."""
    H = 128
    n_angles = 90
    # x_true: industrial part (solid block with internal defects)
    x_true = np.zeros((H, H), dtype=np.float32)
    # Solid background material
    yy, xx = np.ogrid[:H, :H]
    # Rectangular part body
    body_mask = (xx > 20) & (xx < H - 20) & (yy > 20) & (yy < H - 20)
    x_true[body_mask] = rng.uniform(0.6, 0.9)
    # Internal defects (voids/cracks)
    n_defects = rng.integers(2, 6)
    for _ in range(n_defects):
        cx, cy = rng.integers(30, H - 30, size=2)
        defect_type = rng.choice(['void', 'crack', 'inclusion'])
        if defect_type == 'void':
            r = rng.integers(3, 10)
            mask = (xx - cy)**2 + (yy - cx)**2 < r**2
            x_true[mask] = 0.0
        elif defect_type == 'crack':
            length = rng.integers(5, 20)
            width = rng.integers(1, 3)
            x_true[cx:cx+length, cy:cy+width] = 0.0
        else:  # inclusion
            r = rng.integers(2, 7)
            mask = (xx - cy)**2 + (yy - cx)**2 < r**2
            x_true[mask] = 1.0
    x_true = gaussian_filter(x_true, sigma=0.5).astype(np.float32)
    x_true = np.clip(x_true, 0, 1)

    # H_ideal: (n_angles, H) projection matrix (simplified)
    H_ideal = np.zeros((n_angles, H), dtype=np.float32)
    for i in range(n_angles):
        H_ideal[i, i % H] = 1.0

    # y: (n_angles, H) radiographic projections
    from scipy.ndimage import rotate
    angles = np.linspace(0, 180, n_angles, endpoint=False)
    y = np.zeros((n_angles, H), dtype=np.float32)
    for i, angle in enumerate(angles):
        rotated = rotate(x_true, angle, reshape=False, order=1)
        projection = rotated.sum(axis=0)
        # Add noise
        noise = rng.normal(0, 0.02 * projection.max(), H).astype(np.float32)
        y[i, :] = projection + noise

    # reconstruction: FBP (filtered backprojection via ramp filter + backproject)
    from numpy.fft import fft, ifft, fftfreq
    recon = np.zeros((H, H), dtype=np.float32)
    freq = fftfreq(H)
    ramp = np.abs(freq)
    filtered_y = np.zeros_like(y)
    for i in range(n_angles):
        proj_fft = fft(y[i])
        filtered_proj = np.real(ifft(proj_fft * ramp))
        filtered_y[i] = filtered_proj.astype(np.float32)
    # Backproject
    for i, angle in enumerate(angles):
        proj = filtered_y[i]
        # Create backprojection slice
        bp = np.tile(proj, (H, 1))
        rotated_back = rotate(bp, -angle, reshape=False, order=1)
        recon += rotated_back
    recon = np.clip(recon, 0, None).astype(np.float32)
    recon /= (recon.max() + 1e-8)

    return x_true, y, H_ideal, recon


# Map modality name to simulation function and shapes
MODALITIES = {
    "tirf":                   (sim_tirf,                  (128, 128), (128, 128)),
    "tof_camera":             (sim_tof_camera,             (128, 128), (128, 128)),
    "ultrasonic_phased_array":(sim_ultrasonic_phased_array,(128, 128), (64, 128)),
    "us_mri":                 (sim_us_mri,                 (128, 128), (128, 128)),
    "waxs":                   (sim_waxs,                   (128, 128), (128, 128)),
    "weather_radar":          (sim_weather_radar,           (128, 128), (128, 128)),
    "widefield_lowdose":      (sim_widefield_lowdose,       (128, 128), (128, 128)),
    "xfel_sfx":               (sim_xfel_sfx,               (128, 128), (128, 128)),
    "xray_crystallography":   (sim_xray_crystallography,   (128, 128), (128, 128)),
    "xray_ndt":               (sim_xray_ndt,               (128, 128), (90, 128)),
}


def save_png(arr2d, path):
    """Save a 2D float32 array as a grayscale PNG."""
    a = arr2d.copy()
    a = (a - a.min()) / (a.max() - a.min() + 1e-8)
    img = Image.fromarray((a * 255).astype(np.uint8), mode="L")
    img.save(path)


def generate_tier(modality, sim_fn, x_shape, y_shape, tier, n_samples, seed_base):
    tier_dir = os.path.join(DATASETS_BASE, modality, tier)
    img_dir = os.path.join(tier_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    h5_name = f"{modality}_challenge_{tier}.h5"
    h5_path = os.path.join(tier_dir, h5_name)

    print(f"  [{modality}] {tier}: {n_samples} samples -> {h5_path}")

    with h5py.File(h5_path, "w") as hf:
        for i in range(n_samples):
            seed = seed_base + i * SEED_STEP
            rng = np.random.default_rng(seed)
            x_true, y, H_ideal, recon = sim_fn(rng, i)

            grp = hf.create_group(f"sample_{i:02d}")
            grp.create_dataset("x_true", data=x_true.astype(np.float32), compression="gzip")
            grp.create_dataset("y", data=y.astype(np.float32), compression="gzip")
            grp.create_dataset("H_ideal", data=H_ideal.astype(np.float32), compression="gzip")
            grp.create_dataset("reconstruction_baseline", data=recon.astype(np.float32), compression="gzip")

            # Save preview PNGs
            save_png(x_true, os.path.join(img_dir, f"sample_{i:02d}_x_true.png"))
            # y may be non-square; save as-is
            y_vis = y if y.ndim == 2 else y[0]
            save_png(y_vis, os.path.join(img_dir, f"sample_{i:02d}_y.png"))
            save_png(recon if recon.ndim == 2 else recon[0],
                     os.path.join(img_dir, f"sample_{i:02d}_recon.png"))

    # Write spec.json
    spec = {
        "modality": modality,
        "tier": tier,
        "n_samples": n_samples,
        "x_shape": list(x_shape),
        "y_shape": list(y_shape),
        "seed_base": seed_base,
        "seed_step": SEED_STEP,
        "h5_file": h5_name,
        "dtype": "float32",
    }
    with open(os.path.join(tier_dir, "spec.json"), "w") as f:
        json.dump(spec, f, indent=2)

    # Write true_spec.json (metadata about ground truth)
    true_spec = {
        "modality": modality,
        "tier": tier,
        "n_samples": n_samples,
        "x_shape": list(x_shape),
        "description": f"Ground truth x_true arrays for {modality} {tier} tier",
        "units": "normalized [0,1]",
    }
    with open(os.path.join(tier_dir, "true_spec.json"), "w") as f:
        json.dump(true_spec, f, indent=2)

    return n_samples


def main():
    print("=" * 60)
    print("Batch 11 Dataset Generation")
    print("=" * 60)

    results = {}
    for modality, (sim_fn, x_shape, y_shape) in MODALITIES.items():
        print(f"\n[{modality}]")
        mod_results = {}
        for tier, cfg in TIER_CONFIGS.items():
            n = generate_tier(
                modality, sim_fn, x_shape, y_shape,
                tier, cfg["n_samples"], cfg["seed_base"]
            )
            mod_results[tier] = n
        results[modality] = mod_results

    print("\n" + "=" * 60)
    print("COMPLETION SUMMARY")
    print("=" * 60)
    total_files = 0
    for modality, tiers in results.items():
        tier_str = ", ".join(f"{t}={n}" for t, n in tiers.items())
        total = sum(tiers.values())
        total_files += total
        print(f"  {modality}: {tier_str} => {total} samples")
    print(f"\nTotal samples generated: {total_files}")
    print("All datasets written successfully.")


if __name__ == "__main__":
    main()
