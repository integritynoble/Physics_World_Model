"""
Generate benchmark datasets for 7 electron microscopy variant modalities:
  1. EELS   - Electron Energy Loss Spectroscopy
  2. EBSD   - Electron Backscatter Diffraction
  3. electron_tomography - Electron Tomography (STEM-HAADF style)
  4. cryo_et             - Cryo-Electron Tomography
  5. stem                - Scanning TEM HAADF
  6. electron_diffraction
  7. electron_holography

Output layout:
  datasets/benchmark/{modality}/{tier}/
    {modality}_challenge_{tier}.h5
    spec.json
    true_spec.json
    images/sample_NN/{x_true,y,H_ideal,reconstruction_baseline}.png

Tiers: public (12 samples), dev (20), hidden (20)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, label, distance_transform_edt
from scipy.spatial import Voronoi

# ── Root paths ────────────────────────────────────────────────────────────────
ROOT = Path("D:/onedrive/startup/program/physics_world_model/PWM5/Physics_World_Model")
OUT_BASE = ROOT / "datasets" / "benchmark"

SHAPE = (256, 256)
TIER_SIZES = {"public": 12, "dev": 20, "hidden": 20}

# Deterministic seeds per modality/tier
SEEDS = {
    "eels":                {"public": 7000, "dev": 7100, "hidden": 7200},
    "ebsd":                {"public": 7300, "dev": 7400, "hidden": 7500},
    "electron_tomography": {"public": 7600, "dev": 7700, "hidden": 7800},
    "cryo_et":             {"public": 7900, "dev": 8000, "hidden": 8100},
    "stem":                {"public": 8200, "dev": 8300, "hidden": 8400},
    "electron_diffraction":{"public": 8500, "dev": 8600, "hidden": 8700},
    "electron_holography": {"public": 8800, "dev": 8900, "hidden": 9000},
}


# ═══════════════════════════════════════════════════════════════════════════════
# PNG helpers
# ═══════════════════════════════════════════════════════════════════════════════

def save_png(arr: np.ndarray, path: Path) -> None:
    """Save float32 [0,1] array as 8-bit grayscale PNG (robust normalization)."""
    a = arr.astype(np.float32)
    lo, hi = a.min(), a.max()
    if hi - lo < 1e-10:
        a = np.zeros_like(a)
    else:
        a = (a - lo) / (hi - lo)
    img_uint8 = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img_uint8, mode="L").save(str(path))


def save_sino_png(sino: np.ndarray, path: Path) -> None:
    """Save sinogram with robust normalization as grayscale PNG."""
    save_png(sino, path)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. EELS — Electron Energy Loss Spectroscopy
# ═══════════════════════════════════════════════════════════════════════════════

def make_eels_phantom(rng: np.random.Generator) -> np.ndarray:
    """
    2D elemental concentration map (256x256 float32 in [0,1]).
    Models a thin section with 2-4 distinct elemental phases.
    """
    H, W = SHAPE
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cy, cx = H / 2.0, W / 2.0
    img = np.full((H, W), rng.uniform(0.05, 0.15), dtype=np.float32)

    # 3-6 elliptical elemental domains
    n_domains = rng.integers(3, 7)
    for _ in range(n_domains):
        dy = rng.uniform(-0.6, 0.6) * (H / 2)
        dx = rng.uniform(-0.6, 0.6) * (W / 2)
        ry = rng.uniform(15, 70)
        rx = rng.uniform(15, 70)
        angle = rng.uniform(0, np.pi)
        concentration = rng.uniform(0.3, 1.0)

        ddy = yy - (cy + dy)
        ddx = xx - (cx + dx)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        dy_rot = cos_a * ddy + sin_a * ddx
        dx_rot = -sin_a * ddy + cos_a * ddx
        inside = (dy_rot / ry) ** 2 + (dx_rot / rx) ** 2 <= 1.0
        img[inside] = concentration

    img = np.clip(img, 0, 1).astype(np.float32)
    return img


def make_gaussian_psf(size: int, sigma: float) -> np.ndarray:
    """Create (size x size) Gaussian PSF normalized to sum=1."""
    ax = np.arange(size) - size // 2
    yy, xx = np.meshgrid(ax, ax, indexing="ij")
    psf = np.exp(-(yy ** 2 + xx ** 2) / (2 * sigma ** 2))
    psf /= psf.sum() + 1e-12
    return psf.astype(np.float32)


def eels_forward(x_true: np.ndarray, sigma: float, rng: np.random.Generator,
                 poisson_scale: float = 500.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply EELS forward model:
      y = Poisson(Gaussian_blur(x_true, sigma) * poisson_scale) / poisson_scale
    Returns (y, H_ideal) where H_ideal is (11,11) Gaussian PSF.
    """
    # H_ideal: (11,11) PSF kernel
    H_ideal = make_gaussian_psf(11, sigma)

    # Convolve x_true with PSF via gaussian_filter
    blurred = gaussian_filter(x_true.astype(np.float64), sigma=sigma).astype(np.float32)
    blurred = np.clip(blurred, 0, None)

    # Poisson noise
    counts = blurred * poisson_scale
    noisy = rng.poisson(np.maximum(counts, 0)).astype(np.float32) / poisson_scale
    y = np.clip(noisy, 0, 1).astype(np.float32)
    return y, H_ideal


def richardson_lucy_deconv(y: np.ndarray, psf: np.ndarray, n_iter: int = 10) -> np.ndarray:
    """
    Richardson-Lucy deconvolution (spatial domain, 2D).
    psf should be (11,11) normalized to sum=1.
    """
    from scipy.signal import fftconvolve
    # Pad PSF to image size via fftconvolve approach
    u = y.astype(np.float64).copy()
    u = np.clip(u, 1e-8, None)
    psf_flipped = psf[::-1, ::-1].astype(np.float64)

    for _ in range(n_iter):
        conv_u = fftconvolve(u, psf.astype(np.float64), mode="same")
        conv_u = np.clip(conv_u, 1e-8, None)
        ratio = y.astype(np.float64) / conv_u
        ratio = np.clip(ratio, 0, None)
        correction = fftconvolve(ratio, psf_flipped, mode="same")
        u *= correction
        u = np.clip(u, 1e-8, None)

    u -= u.min()
    u /= u.max() + 1e-8
    return np.clip(u, 0, 1).astype(np.float32)


def generate_eels_sample(sample_idx: int, tier: str, rng: np.random.Generator,
                         img_dir: Path) -> tuple:
    # Vary sigma per tier
    if tier == "public":
        sigma = float(rng.uniform(1.5, 2.5))
        poisson_scale = float(rng.uniform(400, 700))
    elif tier == "dev":
        sigma = float(rng.uniform(1.0, 3.5))
        poisson_scale = float(rng.uniform(200, 900))
    else:  # hidden
        sigma = float(rng.uniform(0.8, 4.0))
        poisson_scale = float(rng.uniform(100, 1200))

    energy_loss_eV = float(rng.uniform(100, 1000))
    convergence_angle_mrad = float(rng.uniform(1, 30))
    dwell_time_ms = float(rng.uniform(0.1, 10))

    x_true = make_eels_phantom(rng)
    y, H_ideal = eels_forward(x_true, sigma, rng, poisson_scale)
    recon = richardson_lucy_deconv(y, H_ideal, n_iter=10)

    # Save PNGs
    sdir = img_dir / f"sample_{sample_idx:02d}"
    sdir.mkdir(parents=True, exist_ok=True)
    save_png(x_true, sdir / "x_true.png")
    save_png(y, sdir / "y.png")
    save_png(H_ideal, sdir / "H_ideal.png")
    save_png(recon, sdir / "reconstruction_baseline.png")

    params = {
        "psf_sigma_px": sigma,
        "poisson_scale": poisson_scale,
        "energy_loss_eV": energy_loss_eV,
        "convergence_angle_mrad": convergence_angle_mrad,
        "dwell_time_ms": dwell_time_ms,
    }
    return x_true, y, H_ideal, recon, params


EELS_SPEC = {
    "modality": "eels",
    "description": "EELS mismatch parameter ranges",
    "params": {
        "psf_sigma_px": {
            "unit": "pixels", "nominal": 2.0, "range": [1.5, 2.5],
            "mismatch_range": [0.8, 4.0],
            "description": "Gaussian PSF sigma (probe broadening)"
        },
        "energy_loss_eV": {
            "unit": "eV", "nominal": 500.0, "range": [100, 1000],
            "description": "Energy loss edge position"
        },
        "convergence_angle_mrad": {
            "unit": "mrad", "nominal": 10.0, "range": [1, 30],
            "description": "EELS collection convergence angle"
        },
        "dwell_time_ms": {
            "unit": "ms", "nominal": 1.0, "range": [0.1, 10],
            "description": "Pixel dwell time"
        },
    }
}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. EBSD — Electron Backscatter Diffraction
# ═══════════════════════════════════════════════════════════════════════════════

def make_ebsd_phantom(rng: np.random.Generator) -> np.ndarray:
    """
    Voronoi grain structure with random orientation values per grain.
    Returns (256,256) float32 in [0,1] — grain orientation map.
    """
    H, W = SHAPE
    n_grains = rng.integers(15, 50)

    # Random seed points for Voronoi
    points = rng.uniform(0, 1, size=(n_grains, 2))
    points[:, 0] *= H
    points[:, 1] *= W

    # Assign random orientation value to each grain
    grain_values = rng.uniform(0.0, 1.0, size=n_grains)

    # Build grain map using nearest-seed assignment
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    pix_coords = np.stack([yy.ravel(), xx.ravel()], axis=1)  # (N_pix, 2)

    # Distance from each pixel to each seed
    diffs = pix_coords[:, None, :] - points[None, :, :]  # (N_pix, n_grains, 2)
    dists = np.sum(diffs ** 2, axis=-1)  # (N_pix, n_grains)
    nearest = np.argmin(dists, axis=-1)   # (N_pix,)

    grain_map = grain_values[nearest].reshape(H, W).astype(np.float32)
    return grain_map


def ebsd_forward(x_true: np.ndarray, noise_sigma: float, boundary_strength: float,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    EBSD forward model: pattern quality map with noise + grain boundaries.
    y = x_true + Gaussian_noise + boundary_enhancement
    H_ideal: (256,256) float32 ones (identity operator).
    """
    H, W = x_true.shape

    # Detect grain boundaries via gradient
    from scipy.ndimage import sobel
    sx = sobel(x_true.astype(np.float64), axis=0)
    sy = sobel(x_true.astype(np.float64), axis=1)
    boundary = np.sqrt(sx ** 2 + sy ** 2).astype(np.float32)
    boundary /= boundary.max() + 1e-8

    # Add noise
    noise = rng.normal(0, noise_sigma, size=(H, W)).astype(np.float32)
    y = x_true + noise + boundary_strength * boundary
    y = np.clip(y, 0, 1).astype(np.float32)

    # H_ideal: identity (ones matrix)
    H_ideal = np.ones((H, W), dtype=np.float32)
    return y, H_ideal


def ebsd_baseline_reconstruction(y: np.ndarray) -> np.ndarray:
    """Gaussian smoothing baseline (sigma=1)."""
    recon = gaussian_filter(y.astype(np.float64), sigma=1.0).astype(np.float32)
    recon = np.clip(recon, 0, 1)
    return recon


def generate_ebsd_sample(sample_idx: int, tier: str, rng: np.random.Generator,
                         img_dir: Path) -> tuple:
    if tier == "public":
        noise_sigma = float(rng.uniform(0.01, 0.05))
        boundary_strength = float(rng.uniform(0.05, 0.15))
    elif tier == "dev":
        noise_sigma = float(rng.uniform(0.005, 0.08))
        boundary_strength = float(rng.uniform(0.02, 0.20))
    else:
        noise_sigma = float(rng.uniform(0.003, 0.12))
        boundary_strength = float(rng.uniform(0.01, 0.30))

    step_size_nm = float(rng.uniform(10, 100))
    beam_voltage_kV = float(rng.uniform(10, 30))
    tilt_deg = float(rng.uniform(60, 70))

    x_true = make_ebsd_phantom(rng)
    y, H_ideal = ebsd_forward(x_true, noise_sigma, boundary_strength, rng)
    recon = ebsd_baseline_reconstruction(y)

    sdir = img_dir / f"sample_{sample_idx:02d}"
    sdir.mkdir(parents=True, exist_ok=True)
    save_png(x_true, sdir / "x_true.png")
    save_png(y, sdir / "y.png")
    save_png(H_ideal, sdir / "H_ideal.png")
    save_png(recon, sdir / "reconstruction_baseline.png")

    params = {
        "noise_sigma": noise_sigma,
        "boundary_strength": boundary_strength,
        "step_size_nm": step_size_nm,
        "beam_voltage_kV": beam_voltage_kV,
        "tilt_deg": tilt_deg,
    }
    return x_true, y, H_ideal, recon, params


EBSD_SPEC = {
    "modality": "ebsd",
    "description": "EBSD mismatch parameter ranges",
    "params": {
        "noise_sigma": {
            "unit": "normalized intensity", "nominal": 0.03,
            "range": [0.01, 0.05], "mismatch_range": [0.003, 0.12],
            "description": "Gaussian noise standard deviation on pattern quality map"
        },
        "boundary_strength": {
            "unit": "dimensionless", "nominal": 0.10,
            "range": [0.05, 0.15], "mismatch_range": [0.01, 0.30],
            "description": "Grain boundary enhancement amplitude"
        },
        "step_size_nm": {
            "unit": "nm", "nominal": 50.0, "range": [10, 100],
            "description": "EBSD scan step size"
        },
        "beam_voltage_kV": {
            "unit": "kV", "nominal": 20.0, "range": [10, 30],
            "description": "Electron beam accelerating voltage"
        },
        "tilt_deg": {
            "unit": "deg", "nominal": 70.0, "range": [60, 70],
            "description": "Sample tilt angle"
        },
    }
}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Electron Tomography (Radon-based)
# ═══════════════════════════════════════════════════════════════════════════════

def make_generic_phantom(rng: np.random.Generator) -> np.ndarray:
    """
    Generic 2D slice phantom for tomography — multi-ellipse Shepp-Logan style.
    Returns (256,256) float32 in [0,1].
    """
    H, W = SHAPE
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cy, cx = H / 2.0, W / 2.0

    img = np.zeros((H, W), dtype=np.float32)
    # Large outer ellipse
    big_ry, big_rx = rng.uniform(80, 110), rng.uniform(80, 110)
    outer = ((yy - cy) / big_ry) ** 2 + ((xx - cx) / big_rx) ** 2 <= 1.0
    img[outer] = rng.uniform(0.1, 0.3)

    # Inner structures
    n_inner = rng.integers(3, 8)
    for _ in range(n_inner):
        dy = rng.uniform(-60, 60)
        dx = rng.uniform(-60, 60)
        ry = rng.uniform(8, 40)
        rx = rng.uniform(8, 40)
        angle = rng.uniform(0, np.pi)
        val = rng.uniform(0.2, 1.0)

        ddy = yy - (cy + dy)
        ddx = xx - (cx + dx)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        dy_rot = cos_a * ddy + sin_a * ddx
        dx_rot = -sin_a * ddy + cos_a * ddx
        inside = (dy_rot / ry) ** 2 + (dx_rot / rx) ** 2 <= 1.0
        img[inside] = val

    img = np.clip(img, 0, 1).astype(np.float32)
    return img


def radon_transform(x: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    """
    Simple Radon transform via scipy. Returns (n_angles, n_det) sinogram.
    """
    try:
        from skimage.transform import radon
        # skimage radon returns (n_det, n_angles), transpose to (n_angles, n_det)
        sino = radon(x.astype(np.float64), theta=angles_deg, circle=False)
        return sino.T.astype(np.float32)
    except ImportError:
        # Fallback: numpy-only DFT-based Radon (slower)
        H, W = x.shape
        cx, cy = W / 2, H / 2
        n_angles = len(angles_deg)
        n_det = int(np.ceil(np.sqrt(H ** 2 + W ** 2)))
        sino = np.zeros((n_angles, n_det), dtype=np.float32)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        for ai, angle in enumerate(angles_deg):
            theta = np.deg2rad(angle)
            # Projection coordinate along detector
            t = (xx - cx) * np.cos(theta) + (yy - cy) * np.sin(theta)
            t_idx = (t + n_det / 2).astype(int)
            valid = (t_idx >= 0) & (t_idx < n_det)
            np.add.at(sino[ai], t_idx[valid], x[valid])
        return sino


def fbp_reconstruction(sino: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    """
    Filtered back-projection using skimage if available, else simple BP.
    """
    try:
        from skimage.transform import iradon
        # iradon expects (n_det, n_angles)
        recon = iradon(sino.T.astype(np.float64), theta=angles_deg,
                       filter_name="ramp", circle=False)
        recon -= recon.min()
        recon /= recon.max() + 1e-8
        return np.clip(recon, 0, 1).astype(np.float32)
    except ImportError:
        # Crude back-projection
        H = sino.shape[1]
        recon = np.zeros((H, H), dtype=np.float64)
        yy, xx = np.mgrid[0:H, 0:H].astype(np.float64) - H / 2
        for ai, angle in enumerate(angles_deg):
            theta = np.deg2rad(angle)
            t = xx * np.cos(theta) + yy * np.sin(theta) + H / 2
            t_idx = np.clip(t.astype(int), 0, H - 1)
            recon += sino[ai][t_idx]
        recon -= recon.min()
        recon /= recon.max() + 1e-8
        return np.clip(recon, 0, 1).astype(np.float32)


def electron_tomo_forward(x_true: np.ndarray, angles_deg: np.ndarray,
                          noise_level: float, rng: np.random.Generator
                          ) -> tuple[np.ndarray, np.ndarray]:
    """
    Electron tomography forward model:
      sino_clean = Radon(x_true, angles)
      y = sino_clean + Poisson noise
      H_ideal = angles array (60,) float32
    Returns (y sinogram (n_angles, 256), H_ideal (n_angles,))
    """
    sino = radon_transform(x_true, angles_deg)  # (n_angles, n_det)

    # Normalize sinogram to [0,1]
    sino_norm = sino - sino.min()
    sino_norm = sino_norm / (sino_norm.max() + 1e-8)

    # Poisson noise
    scale = 1.0 / (noise_level + 1e-8)
    counts = sino_norm * scale
    noisy = rng.poisson(np.maximum(counts, 0)).astype(np.float32) / scale
    noisy = np.clip(noisy, 0, None)

    # Crop/pad detector dimension to exactly 256
    n_angles, n_det = noisy.shape
    target_det = 256
    if n_det > target_det:
        trim = (n_det - target_det) // 2
        noisy = noisy[:, trim:trim + target_det]
        sino_norm = sino_norm[:, trim:trim + target_det]
    elif n_det < target_det:
        pad = target_det - n_det
        noisy = np.pad(noisy, ((0, 0), (pad // 2, pad - pad // 2)))
        sino_norm = np.pad(sino_norm, ((0, 0), (pad // 2, pad - pad // 2)))

    H_ideal = angles_deg.astype(np.float32)  # (n_angles,)
    return noisy.astype(np.float32), H_ideal


def generate_electron_tomo_sample(sample_idx: int, tier: str, rng: np.random.Generator,
                                   img_dir: Path, n_tilts: int = 60,
                                   tilt_min: float = -70.0, tilt_max: float = 70.0
                                   ) -> tuple:
    if tier == "public":
        noise_level = float(rng.uniform(0.02, 0.08))
        dose_e_per_A2 = float(rng.uniform(50, 500))
    elif tier == "dev":
        noise_level = float(rng.uniform(0.01, 0.15))
        dose_e_per_A2 = float(rng.uniform(10, 800))
    else:
        noise_level = float(rng.uniform(0.005, 0.25))
        dose_e_per_A2 = float(rng.uniform(1, 1000))

    tilt_range_deg = tilt_max - tilt_min  # 140 deg
    tilt_increment_deg = float(tilt_range_deg / (n_tilts - 1))

    angles_deg = np.linspace(tilt_min, tilt_max, n_tilts)
    x_true = make_generic_phantom(rng)
    y, H_ideal = electron_tomo_forward(x_true, angles_deg, noise_level, rng)

    # FBP reconstruction
    recon = fbp_reconstruction(y, angles_deg)

    # Pad/crop recon to 256x256
    H_r, W_r = recon.shape
    if H_r != 256 or W_r != 256:
        from PIL import Image as PILImage
        recon_img = PILImage.fromarray((np.clip(recon, 0, 1) * 255).astype(np.uint8), mode="L")
        recon_img = recon_img.resize((256, 256), PILImage.BILINEAR)
        recon = np.array(recon_img).astype(np.float32) / 255.0

    sdir = img_dir / f"sample_{sample_idx:02d}"
    sdir.mkdir(parents=True, exist_ok=True)
    save_png(x_true, sdir / "x_true.png")
    save_sino_png(y, sdir / "y.png")
    # H_ideal is 1D angles — save as 1D array image (1 row)
    H_disp = np.tile(H_ideal[None, :], (20, 1)).astype(np.float32)
    save_png(H_disp, sdir / "H_ideal.png")
    save_png(recon, sdir / "reconstruction_baseline.png")

    params = {
        "noise_level": noise_level,
        "dose_e_per_A2": dose_e_per_A2,
        "tilt_range_deg_min": float(tilt_min),
        "tilt_range_deg_max": float(tilt_max),
        "tilt_increment_deg": tilt_increment_deg,
        "n_tilts": n_tilts,
    }
    return x_true, y, H_ideal, recon, params


ELECTRON_TOMO_SPEC = {
    "modality": "electron_tomography",
    "description": "Electron tomography mismatch parameter ranges",
    "params": {
        "noise_level": {
            "unit": "dimensionless", "nominal": 0.05,
            "range": [0.02, 0.08], "mismatch_range": [0.005, 0.25],
            "description": "Relative Poisson noise level in sinogram"
        },
        "tilt_range_deg": {
            "unit": "deg", "nominal": 140.0, "range": [-70, 70],
            "description": "Full tilt range from -70 to +70 degrees"
        },
        "tilt_increment_deg": {
            "unit": "deg", "nominal": 2.37, "range": [1, 5],
            "description": "Angular increment between tilt series images"
        },
        "dose_e_per_A2": {
            "unit": "e/A^2", "nominal": 100.0, "range": [1, 1000],
            "description": "Total electron dose per tilt"
        },
    }
}


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Cryo-ET — Cryo-Electron Tomography
# ═══════════════════════════════════════════════════════════════════════════════

def make_cryo_et_phantom(rng: np.random.Generator) -> np.ndarray:
    """
    Cryo-ET phantom: sparse blobs representing macromolecular complexes in ice.
    Returns (256,256) float32 in [0,1].
    """
    H, W = SHAPE
    img = np.zeros((H, W), dtype=np.float32)

    # Background: low-frequency ice noise
    noise = rng.standard_normal((H, W)).astype(np.float32)
    ice_bg = gaussian_filter(noise, sigma=rng.uniform(3, 8)) * 0.08
    img += ice_bg

    # Sparse protein complexes (blobs of varying size and density)
    n_complexes = rng.integers(5, 20)
    for _ in range(n_complexes):
        cy = rng.integers(20, H - 20)
        cx = rng.integers(20, W - 20)
        sigma = rng.uniform(3, 12)
        amplitude = rng.uniform(0.3, 1.0)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        blob = amplitude * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
        img += blob

    img = np.clip(img, 0, None)
    img /= img.max() + 1e-8
    return img.astype(np.float32)


def ctf_phase_modulation(sino: np.ndarray, defocus_um: float) -> np.ndarray:
    """
    Apply simplified CTF phase modulation to each projection row.
    CTF(k) = cos(pi * lambda * defocus * k^2) at 300 kV (lambda ~ 1.97 pm)
    """
    lambda_pm = 1.97  # 300 kV
    lambda_um = lambda_pm * 1e-6  # convert pm to um
    n_angles, n_det = sino.shape
    freq = np.fft.fftfreq(n_det).astype(np.float64)  # cycles/pixel
    # CTF: cos(pi * lambda_um * defocus_um * (freq/pixel_size_um)^2)
    # Simplified: use pixel_size as 1 um unit
    chi = np.pi * lambda_um * defocus_um * freq ** 2
    ctf = np.cos(chi).astype(np.float32)

    sino_ctf = np.zeros_like(sino)
    for i in range(n_angles):
        row_f = np.fft.fft(sino[i].astype(np.float64))
        sino_ctf[i] = np.real(np.fft.ifft(row_f * ctf)).astype(np.float32)
    return sino_ctf


def cryo_et_forward(x_true: np.ndarray, angles_deg: np.ndarray,
                    defocus_um: float, dose_per_tilt: float,
                    rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Cryo-ET forward: limited-angle Radon + CTF + Poisson noise.
    Returns (y: (n_angles, 256), H_ideal: (n_angles,) tilt angles).
    """
    sino = radon_transform(x_true, angles_deg)  # (n_angles, n_det)

    # Normalize
    sino_norm = sino - sino.min()
    sino_norm = sino_norm / (sino_norm.max() + 1e-8)

    # Apply CTF
    sino_ctf = ctf_phase_modulation(sino_norm, defocus_um)
    sino_ctf = np.clip(sino_ctf, 0, None)

    # Poisson noise (low dose cryo)
    counts = sino_ctf * dose_per_tilt
    noisy = rng.poisson(np.maximum(counts, 0)).astype(np.float32) / (dose_per_tilt + 1e-8)
    noisy = np.clip(noisy, 0, None)

    # Normalize
    noisy /= noisy.max() + 1e-8

    # Crop/pad detector to 256
    n_angles, n_det = noisy.shape
    target_det = 256
    if n_det > target_det:
        trim = (n_det - target_det) // 2
        noisy = noisy[:, trim:trim + target_det]
    elif n_det < target_det:
        pad = target_det - n_det
        noisy = np.pad(noisy, ((0, 0), (pad // 2, pad - pad // 2)))

    H_ideal = angles_deg.astype(np.float32)  # (n_angles,)
    return noisy.astype(np.float32), H_ideal


def generate_cryo_et_sample(sample_idx: int, tier: str, rng: np.random.Generator,
                             img_dir: Path, n_tilts: int = 41,
                             tilt_min: float = -60.0, tilt_max: float = 60.0
                             ) -> tuple:
    if tier == "public":
        defocus_um = float(rng.uniform(2, 4))
        dose_per_tilt = float(rng.uniform(5, 8))
    elif tier == "dev":
        defocus_um = float(rng.uniform(1, 5))
        dose_per_tilt = float(rng.uniform(2, 9))
    else:
        defocus_um = float(rng.uniform(1, 6))
        dose_per_tilt = float(rng.uniform(1, 10))

    angles_deg = np.linspace(tilt_min, tilt_max, n_tilts)
    x_true = make_cryo_et_phantom(rng)
    y, H_ideal = cryo_et_forward(x_true, angles_deg, defocus_um, dose_per_tilt, rng)

    # FBP reconstruction
    recon = fbp_reconstruction(y, angles_deg)
    H_r, W_r = recon.shape
    if H_r != 256 or W_r != 256:
        from PIL import Image as PILImage
        recon_img = PILImage.fromarray((np.clip(recon, 0, 1) * 255).astype(np.uint8), mode="L")
        recon_img = recon_img.resize((256, 256), PILImage.BILINEAR)
        recon = np.array(recon_img).astype(np.float32) / 255.0

    sdir = img_dir / f"sample_{sample_idx:02d}"
    sdir.mkdir(parents=True, exist_ok=True)
    save_png(x_true, sdir / "x_true.png")
    save_sino_png(y, sdir / "y.png")
    H_disp = np.tile(H_ideal[None, :], (10, 1)).astype(np.float32)
    save_png(H_disp, sdir / "H_ideal.png")
    save_png(recon, sdir / "reconstruction_baseline.png")

    params = {
        "defocus_um": defocus_um,
        "dose_per_tilt_e_per_A2": dose_per_tilt,
        "tilt_range_deg_min": float(tilt_min),
        "tilt_range_deg_max": float(tilt_max),
        "n_tilts": n_tilts,
    }
    return x_true, y, H_ideal, recon, params


CRYO_ET_SPEC = {
    "modality": "cryo_et",
    "description": "Cryo-ET mismatch parameter ranges",
    "params": {
        "defocus_um": {
            "unit": "um", "nominal": 3.0, "range": [1, 6],
            "description": "CTF defocus (underfocus positive)"
        },
        "dose_per_tilt_e_per_A2": {
            "unit": "e/A^2", "nominal": 5.0, "range": [1, 10],
            "description": "Electron dose per tilt image"
        },
        "tilt_range_deg": {
            "unit": "deg", "nominal": 120.0, "range": [-60, 60],
            "description": "Limited-angle tilt range"
        },
    }
}


# ═══════════════════════════════════════════════════════════════════════════════
# 5. STEM — Scanning TEM HAADF
# ═══════════════════════════════════════════════════════════════════════════════

def make_stem_phantom(rng: np.random.Generator) -> np.ndarray:
    """
    Atomic column density map with crystalline structure.
    Random Gaussian blobs at quasi-periodic positions.
    Returns (256,256) float32 in [0,1].
    """
    H, W = SHAPE
    img = np.zeros((H, W), dtype=np.float32)

    # Background amorphous contribution
    bg_noise = rng.standard_normal((H, W)).astype(np.float32)
    bg = gaussian_filter(bg_noise, sigma=rng.uniform(2, 5)) * 0.05
    bg -= bg.min()
    img += bg

    # Crystalline domains with lattice
    n_crystals = rng.integers(1, 4)
    for _ in range(n_crystals):
        # Crystal center and extent
        ccy = rng.uniform(0.2, 0.8) * H
        ccx = rng.uniform(0.2, 0.8) * W
        half_size = rng.uniform(30, 80)
        spacing = rng.uniform(8, 18)  # atomic column spacing in pixels
        amplitude = rng.uniform(0.5, 1.0)

        # Lattice vectors (slightly rotated)
        angle = rng.uniform(0, np.pi / 4)
        a1 = spacing * np.array([np.cos(angle), np.sin(angle)])
        a2 = spacing * np.array([-np.sin(angle), np.cos(angle)])

        # Generate lattice points within crystal domain
        n_cells = int(half_size / spacing) + 2
        for i in range(-n_cells, n_cells + 1):
            for j in range(-n_cells, n_cells + 1):
                col_y = ccy + i * a1[0] + j * a2[0]
                col_x = ccx + i * a1[1] + j * a2[1]
                # Check if within crystal domain
                if ((col_y - ccy) ** 2 + (col_x - ccx) ** 2) > half_size ** 2:
                    continue
                # Add Gaussian atom
                cy_int = int(np.round(col_y))
                cx_int = int(np.round(col_x))
                if 0 <= cy_int < H and 0 <= cx_int < W:
                    atom_sigma = rng.uniform(1.0, 2.5)
                    atom_amp = amplitude * rng.uniform(0.7, 1.0)
                    yy = np.arange(max(0, cy_int - 6), min(H, cy_int + 7))
                    xx = np.arange(max(0, cx_int - 6), min(W, cx_int + 7))
                    YY, XX = np.meshgrid(yy, xx, indexing="ij")
                    blob = atom_amp * np.exp(
                        -((YY - col_y) ** 2 + (XX - col_x) ** 2) / (2 * atom_sigma ** 2)
                    )
                    img[YY, XX] += blob

    img = np.clip(img, 0, None)
    img /= img.max() + 1e-8
    return img.astype(np.float32)


def stem_forward(x_true: np.ndarray, probe_sigma: float, dose_electrons: float,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    STEM HAADF forward: convolve with probe PSF + Poisson noise.
    H_ideal: (11,11) probe PSF.
    """
    H_ideal = make_gaussian_psf(11, probe_sigma)
    blurred = gaussian_filter(x_true.astype(np.float64), sigma=probe_sigma).astype(np.float32)
    blurred = np.clip(blurred, 0, None)

    # Poisson noise (electron counting)
    counts = blurred * dose_electrons
    noisy = rng.poisson(np.maximum(counts, 0)).astype(np.float32) / (dose_electrons + 1e-8)
    y = np.clip(noisy, 0, 1).astype(np.float32)
    return y, H_ideal


def generate_stem_sample(sample_idx: int, tier: str, rng: np.random.Generator,
                          img_dir: Path) -> tuple:
    if tier == "public":
        probe_sigma = float(rng.uniform(0.8, 1.5))
        dose_electrons = float(rng.uniform(500, 2000))
    elif tier == "dev":
        probe_sigma = float(rng.uniform(0.5, 2.0))
        dose_electrons = float(rng.uniform(200, 3000))
    else:
        probe_sigma = float(rng.uniform(0.3, 3.0))
        dose_electrons = float(rng.uniform(50, 5000))

    convergence_angle_mrad = float(rng.uniform(15, 30))
    probe_size_pm = float(rng.uniform(60, 200))
    dwell_time_us = float(rng.uniform(1, 50))

    x_true = make_stem_phantom(rng)
    y, H_ideal = stem_forward(x_true, probe_sigma, dose_electrons, rng)
    recon = richardson_lucy_deconv(y, H_ideal, n_iter=10)

    sdir = img_dir / f"sample_{sample_idx:02d}"
    sdir.mkdir(parents=True, exist_ok=True)
    save_png(x_true, sdir / "x_true.png")
    save_png(y, sdir / "y.png")
    save_png(H_ideal, sdir / "H_ideal.png")
    save_png(recon, sdir / "reconstruction_baseline.png")

    params = {
        "probe_sigma_px": probe_sigma,
        "dose_electrons": dose_electrons,
        "convergence_angle_mrad": convergence_angle_mrad,
        "probe_size_pm": probe_size_pm,
        "dwell_time_us": dwell_time_us,
    }
    return x_true, y, H_ideal, recon, params


STEM_SPEC = {
    "modality": "stem",
    "description": "STEM HAADF mismatch parameter ranges",
    "params": {
        "probe_sigma_px": {
            "unit": "pixels", "nominal": 1.0,
            "range": [0.8, 1.5], "mismatch_range": [0.3, 3.0],
            "description": "Gaussian probe PSF sigma"
        },
        "dose_electrons": {
            "unit": "e/pixel", "nominal": 1000.0,
            "range": [500, 2000], "mismatch_range": [50, 5000],
            "description": "Electron dose per pixel"
        },
        "convergence_angle_mrad": {
            "unit": "mrad", "nominal": 22.0, "range": [15, 30],
            "description": "Probe-forming convergence semi-angle"
        },
        "probe_size_pm": {
            "unit": "pm", "nominal": 100.0, "range": [60, 200],
            "description": "Probe FWHM size"
        },
        "dwell_time_us": {
            "unit": "us", "nominal": 10.0, "range": [1, 50],
            "description": "Pixel dwell time"
        },
    }
}


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Electron Diffraction
# ═══════════════════════════════════════════════════════════════════════════════

def make_diffraction_phantom(rng: np.random.Generator) -> np.ndarray:
    """
    Crystal structure factor map: diffraction spots on background.
    Returns (256,256) float32 in [0,1].
    """
    H, W = SHAPE
    img = np.zeros((H, W), dtype=np.float32)
    cy, cx = H // 2, W // 2

    # Central beam (bright direct beam)
    img[cy, cx] = 1.0
    sigma_beam = rng.uniform(2, 5)
    img = gaussian_filter(img.astype(np.float64), sigma=sigma_beam).astype(np.float32)

    # Diffraction spots in systematic rows/columns
    lattice_a = rng.uniform(20, 50)  # pixels
    lattice_b = rng.uniform(20, 50)
    angle = rng.uniform(0, np.pi / 4)
    a1 = lattice_a * np.array([np.cos(angle), np.sin(angle)])
    a2 = lattice_b * np.array([-np.sin(angle), np.cos(angle)])

    spot_sigma = rng.uniform(1.5, 4.0)
    n_orders = rng.integers(2, 5)

    for h in range(-n_orders, n_orders + 1):
        for k in range(-n_orders, n_orders + 1):
            if h == 0 and k == 0:
                continue
            # Spot position
            sy = cy + h * a1[0] + k * a2[0]
            sx = cx + h * a1[1] + k * a2[1]
            if not (0 < sy < H and 0 < sx < W):
                continue

            # Structure factor amplitude (decreasing with order)
            amplitude = rng.uniform(0.2, 1.0) / (abs(h) + abs(k) + 1)

            # Add Gaussian spot
            yy = np.arange(max(0, int(sy) - 12), min(H, int(sy) + 13))
            xx = np.arange(max(0, int(sx) - 12), min(W, int(sx) + 13))
            YY, XX = np.meshgrid(yy, xx, indexing="ij")
            spot = amplitude * np.exp(
                -((YY - sy) ** 2 + (XX - sx) ** 2) / (2 * spot_sigma ** 2)
            )
            img[YY, XX] += spot

    img = np.clip(img, 0, None)
    img /= img.max() + 1e-8
    return img.astype(np.float32)


def diffraction_forward(x_true: np.ndarray, background_level: float,
                        dose_electrons: float, rng: np.random.Generator
                        ) -> tuple[np.ndarray, np.ndarray]:
    """
    Electron diffraction forward model:
      background = smooth low-freq noise
      y = Poisson(|x_true + background|^2 * dose) / dose
      H_ideal: (256,256) background correction pattern
    """
    H, W = x_true.shape

    # Generate background (low-freq noise)
    bg_noise = rng.standard_normal((H, W)).astype(np.float32)
    background = gaussian_filter(bg_noise, sigma=rng.uniform(10, 30))
    background -= background.min()
    background = background / (background.max() + 1e-8) * background_level

    # Diffraction intensity = |object + background|^2 (simplified scalar)
    intensity = (x_true + background) ** 2
    intensity = np.clip(intensity, 0, None)
    intensity /= intensity.max() + 1e-8

    # Poisson noise
    counts = intensity * dose_electrons
    noisy = rng.poisson(np.maximum(counts, 0)).astype(np.float32) / (dose_electrons + 1e-8)
    y = np.clip(noisy, 0, None)
    y /= y.max() + 1e-8

    # H_ideal: background correction map
    H_ideal = background.astype(np.float32)
    H_ideal /= H_ideal.max() + 1e-8

    return y.astype(np.float32), H_ideal


def diffraction_baseline_reconstruction(y: np.ndarray, H_ideal: np.ndarray
                                        ) -> np.ndarray:
    """
    Baseline: sqrt(y - background_estimate) -> phase estimate.
    H_ideal is the background pattern.
    """
    y_corr = np.clip(y.astype(np.float64) - H_ideal.astype(np.float64), 0, None)
    recon = np.sqrt(y_corr)
    recon -= recon.min()
    recon /= recon.max() + 1e-8
    return np.clip(recon, 0, 1).astype(np.float32)


def generate_diffraction_sample(sample_idx: int, tier: str, rng: np.random.Generator,
                                  img_dir: Path) -> tuple:
    if tier == "public":
        background_level = float(rng.uniform(0.05, 0.15))
        dose_electrons = float(rng.uniform(1000, 5000))
    elif tier == "dev":
        background_level = float(rng.uniform(0.02, 0.25))
        dose_electrons = float(rng.uniform(500, 8000))
    else:
        background_level = float(rng.uniform(0.01, 0.35))
        dose_electrons = float(rng.uniform(100, 10000))

    wavelength_pm = float(rng.uniform(1.97, 3.0))  # 100-300 kV range
    camera_length_mm = float(rng.uniform(100, 1000))
    tilt_angle_deg = float(rng.uniform(0, 30))

    x_true = make_diffraction_phantom(rng)
    y, H_ideal = diffraction_forward(x_true, background_level, dose_electrons, rng)
    recon = diffraction_baseline_reconstruction(y, H_ideal)

    sdir = img_dir / f"sample_{sample_idx:02d}"
    sdir.mkdir(parents=True, exist_ok=True)
    save_png(x_true, sdir / "x_true.png")
    save_png(y, sdir / "y.png")
    save_png(H_ideal, sdir / "H_ideal.png")
    save_png(recon, sdir / "reconstruction_baseline.png")

    params = {
        "background_level": background_level,
        "dose_electrons": dose_electrons,
        "wavelength_pm": wavelength_pm,
        "camera_length_mm": camera_length_mm,
        "tilt_angle_deg": tilt_angle_deg,
    }
    return x_true, y, H_ideal, recon, params


DIFFRACTION_SPEC = {
    "modality": "electron_diffraction",
    "description": "Electron diffraction mismatch parameter ranges",
    "params": {
        "background_level": {
            "unit": "normalized intensity", "nominal": 0.10,
            "range": [0.05, 0.15], "mismatch_range": [0.01, 0.35],
            "description": "Background diffuse scattering amplitude"
        },
        "dose_electrons": {
            "unit": "e/pixel", "nominal": 2000.0,
            "range": [1000, 5000], "mismatch_range": [100, 10000],
            "description": "Total electron dose"
        },
        "wavelength_pm": {
            "unit": "pm", "nominal": 2.51, "range": [1.97, 3.0],
            "description": "Electron wavelength (100-300 kV)"
        },
        "camera_length_mm": {
            "unit": "mm", "nominal": 500.0, "range": [100, 1000],
            "description": "Camera length (diffraction scale)"
        },
        "tilt_angle_deg": {
            "unit": "deg", "nominal": 0.0, "range": [0, 30],
            "description": "Crystal tilt angle"
        },
    }
}


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Electron Holography
# ═══════════════════════════════════════════════════════════════════════════════

def make_holography_phantom(rng: np.random.Generator) -> np.ndarray:
    """
    Electrostatic potential map (projected).
    Returns (256,256) float32 in [0,1].
    """
    H, W = SHAPE
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    img = np.zeros((H, W), dtype=np.float32)

    # Smooth potential background
    bg_noise = rng.standard_normal((H, W)).astype(np.float32)
    bg = gaussian_filter(bg_noise, sigma=rng.uniform(5, 15)) * 0.1
    img += bg

    # Electrostatic domains (e.g., p-n junctions, charged interfaces)
    n_domains = rng.integers(2, 6)
    for _ in range(n_domains):
        cy = rng.uniform(30, H - 30)
        cx = rng.uniform(30, W - 30)
        ry = rng.uniform(20, 80)
        rx = rng.uniform(20, 80)
        angle = rng.uniform(0, np.pi)
        potential = rng.uniform(-1.0, 1.0)

        ddy = yy - cy
        ddx = xx - cx
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        dy_rot = cos_a * ddy + sin_a * ddx
        dx_rot = -sin_a * ddy + cos_a * ddx
        inside = (dy_rot / ry) ** 2 + (dx_rot / rx) ** 2 <= 1.0
        img[inside] += potential

    # Normalize to [0, 1]
    img -= img.min()
    img /= img.max() + 1e-8
    return img.astype(np.float32)


def holography_forward(x_true: np.ndarray, carrier_freq: float,
                       noise_level: float, rng: np.random.Generator
                       ) -> tuple[np.ndarray, np.ndarray]:
    """
    Off-axis electron hologram formation:
      object_wave = exp(i * pi * x_true)
      ref_wave = exp(i * 2*pi * carrier_freq * x_grid)
      hologram = |object_wave + ref_wave|^2 + noise

    carrier_freq: in cycles/pixel (0.03 to 0.1)
    Returns (y: (256,256) real hologram, H_ideal: (256,256) carrier fringe pattern)
    """
    H, W = x_true.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)

    # Object wave (pure phase object)
    phase = np.pi * x_true.astype(np.float64)
    obj_wave = np.exp(1j * phase)

    # Reference wave (plane wave with carrier fringe)
    ref_phase = 2 * np.pi * carrier_freq * xx
    ref_wave = np.exp(1j * ref_phase)

    # Hologram intensity
    total = obj_wave + ref_wave
    hologram = np.abs(total) ** 2  # (256,256) real

    # Normalize hologram
    hologram -= hologram.min()
    hologram /= hologram.max() + 1e-8

    # Poisson noise
    scale = 1.0 / (noise_level + 1e-8)
    counts = hologram * scale
    noisy = rng.poisson(np.maximum(counts, 0)).astype(np.float32) / scale
    y = noisy.astype(np.float32)
    y -= y.min()
    y /= y.max() + 1e-8

    # H_ideal: carrier fringe pattern
    fringe = (1 + np.cos(2 * np.pi * carrier_freq * xx)) / 2.0
    H_ideal = fringe.astype(np.float32)

    return y, H_ideal


def holography_baseline_reconstruction(y: np.ndarray, H_ideal: np.ndarray,
                                        carrier_freq: float) -> np.ndarray:
    """
    Fourier filter sideband reconstruction:
    1. FFT of hologram
    2. Select sideband near carrier frequency
    3. IFFT of sideband
    4. phase = angle(sideband_ifft) / pi
    """
    H, W = y.shape

    # FFT of hologram
    Y_f = np.fft.fft2(y.astype(np.float64))
    Y_f_shift = np.fft.fftshift(Y_f)

    # Sideband center: carrier_freq * W pixels from center in x-direction
    cx_shift = int(np.round(W / 2 + carrier_freq * W))
    cy_shift = H // 2

    # Sideband filter: Gaussian window around carrier
    radius = max(10, int(carrier_freq * W * 0.5))
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    mask = np.exp(-((yy - cy_shift) ** 2 + (xx - cx_shift) ** 2) / (2 * radius ** 2))

    # Filtered sideband
    sideband_f = Y_f_shift * mask
    sideband_f_unshift = np.fft.ifftshift(sideband_f)
    sideband_ifft = np.fft.ifft2(sideband_f_unshift)

    # Phase reconstruction
    phase_recon = np.angle(sideband_ifft) / np.pi
    phase_recon -= phase_recon.min()
    phase_recon /= phase_recon.max() + 1e-8
    return np.clip(phase_recon, 0, 1).astype(np.float32)


def generate_holography_sample(sample_idx: int, tier: str, rng: np.random.Generator,
                                img_dir: Path) -> tuple:
    if tier == "public":
        carrier_freq = float(rng.uniform(0.04, 0.08))
        noise_level = float(rng.uniform(0.01, 0.05))
    elif tier == "dev":
        carrier_freq = float(rng.uniform(0.03, 0.09))
        noise_level = float(rng.uniform(0.005, 0.10))
    else:
        carrier_freq = float(rng.uniform(0.03, 0.10))
        noise_level = float(rng.uniform(0.002, 0.15))

    beam_energy_keV = float(rng.uniform(100, 300))
    biprism_voltage_V = float(rng.uniform(1, 200))

    x_true = make_holography_phantom(rng)
    y, H_ideal = holography_forward(x_true, carrier_freq, noise_level, rng)
    recon = holography_baseline_reconstruction(y, H_ideal, carrier_freq)

    sdir = img_dir / f"sample_{sample_idx:02d}"
    sdir.mkdir(parents=True, exist_ok=True)
    save_png(x_true, sdir / "x_true.png")
    save_png(y, sdir / "y.png")
    save_png(H_ideal, sdir / "H_ideal.png")
    save_png(recon, sdir / "reconstruction_baseline.png")

    params = {
        "carrier_freq_1_per_px": carrier_freq,
        "noise_level": noise_level,
        "beam_energy_keV": beam_energy_keV,
        "biprism_voltage_V": biprism_voltage_V,
    }
    return x_true, y, H_ideal, recon, params


HOLOGRAPHY_SPEC = {
    "modality": "electron_holography",
    "description": "Electron holography mismatch parameter ranges",
    "params": {
        "carrier_freq_1_per_px": {
            "unit": "cycles/pixel", "nominal": 0.05,
            "range": [0.03, 0.1], "mismatch_range": [0.03, 0.1],
            "description": "Off-axis carrier fringe frequency"
        },
        "noise_level": {
            "unit": "normalized intensity", "nominal": 0.03,
            "range": [0.01, 0.05], "mismatch_range": [0.002, 0.15],
            "description": "Poisson noise amplitude (relative)"
        },
        "beam_energy_keV": {
            "unit": "keV", "nominal": 200.0, "range": [100, 300],
            "description": "Electron beam energy"
        },
        "biprism_voltage_V": {
            "unit": "V", "nominal": 50.0, "range": [1, 200],
            "description": "Electrostatic biprism voltage"
        },
    }
}


# ═══════════════════════════════════════════════════════════════════════════════
# Generic tier generator
# ═══════════════════════════════════════════════════════════════════════════════

# Map modality -> (generate_fn, spec_dict, array_info)
# array_info: dict with dataset names and expected shapes
MODALITY_CONFIG = {
    "eels": {
        "fn": generate_eels_sample,
        "spec": EELS_SPEC,
        "datasets": {
            "x_true": (256, 256),
            "y": (256, 256),
            "H_ideal": (11, 11),
            "reconstruction_baseline": (256, 256),
        },
        "extra_kwargs": {},
    },
    "ebsd": {
        "fn": generate_ebsd_sample,
        "spec": EBSD_SPEC,
        "datasets": {
            "x_true": (256, 256),
            "y": (256, 256),
            "H_ideal": (256, 256),
            "reconstruction_baseline": (256, 256),
        },
        "extra_kwargs": {},
    },
    "electron_tomography": {
        "fn": generate_electron_tomo_sample,
        "spec": ELECTRON_TOMO_SPEC,
        "datasets": {
            "x_true": (256, 256),
            "y": (60, 256),
            "H_ideal": (60,),
            "reconstruction_baseline": (256, 256),
        },
        "extra_kwargs": {"n_tilts": 60, "tilt_min": -70.0, "tilt_max": 70.0},
    },
    "cryo_et": {
        "fn": generate_cryo_et_sample,
        "spec": CRYO_ET_SPEC,
        "datasets": {
            "x_true": (256, 256),
            "y": (41, 256),
            "H_ideal": (41,),
            "reconstruction_baseline": (256, 256),
        },
        "extra_kwargs": {"n_tilts": 41, "tilt_min": -60.0, "tilt_max": 60.0},
    },
    "stem": {
        "fn": generate_stem_sample,
        "spec": STEM_SPEC,
        "datasets": {
            "x_true": (256, 256),
            "y": (256, 256),
            "H_ideal": (11, 11),
            "reconstruction_baseline": (256, 256),
        },
        "extra_kwargs": {},
    },
    "electron_diffraction": {
        "fn": generate_diffraction_sample,
        "spec": DIFFRACTION_SPEC,
        "datasets": {
            "x_true": (256, 256),
            "y": (256, 256),
            "H_ideal": (256, 256),
            "reconstruction_baseline": (256, 256),
        },
        "extra_kwargs": {},
    },
    "electron_holography": {
        "fn": generate_holography_sample,
        "spec": HOLOGRAPHY_SPEC,
        "datasets": {
            "x_true": (256, 256),
            "y": (256, 256),
            "H_ideal": (256, 256),
            "reconstruction_baseline": (256, 256),
        },
        "extra_kwargs": {},
    },
}


def generate_tier(modality: str, tier: str) -> dict:
    """Generate one (modality, tier) combination. Returns verification dict."""
    cfg = MODALITY_CONFIG[modality]
    n_samples = TIER_SIZES[tier]
    base_seed = SEEDS[modality][tier]

    tier_dir = OUT_BASE / modality / tier
    h5_path = tier_dir / f"{modality}_challenge_{tier}.h5"
    img_dir = tier_dir / "images"
    tier_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {modality.upper()} | tier={tier} | n={n_samples}")
    print(f"  Output: {h5_path}")
    print(f"{'='*60}")

    true_spec_records = {}

    with h5py.File(h5_path, "w") as hf:
        hf.attrs["modality"] = modality
        hf.attrs["tier"] = tier
        hf.attrs["n_samples"] = n_samples
        hf.attrs["generator"] = "generate_electron_microscopy_h5.py"
        hf.attrs["date"] = "2026-03-10"
        hf.attrs["version"] = "1.0"

        for i in range(n_samples):
            seed = base_seed + i
            rng = np.random.default_rng(seed)

            # Call modality-specific generator
            x_true, y, H_ideal, recon, params = cfg["fn"](
                i, tier, rng, img_dir, **cfg["extra_kwargs"]
            )

            grp_name = f"sample_{i:02d}"
            grp = hf.create_group(grp_name)
            grp.create_dataset("x_true", data=x_true, dtype="float32", compression="gzip")
            grp.create_dataset("y", data=y, dtype="float32", compression="gzip")
            grp.create_dataset("H_ideal", data=H_ideal, dtype="float32", compression="gzip")
            grp.create_dataset("reconstruction_baseline", data=recon, dtype="float32",
                               compression="gzip")

            for k, v in params.items():
                grp.attrs[k] = float(v) if isinstance(v, (int, float)) else v
            grp.attrs["seed"] = seed

            true_spec_records[grp_name] = {k: float(v) if isinstance(v, (int, float)) else v
                                            for k, v in params.items()}
            true_spec_records[grp_name]["seed"] = int(seed)

            print(
                f"  [{i:02d}] x={x_true.shape}[{x_true.min():.3f},{x_true.max():.3f}]"
                f"  y={y.shape}[{y.min():.3f},{y.max():.3f}]"
                f"  H={H_ideal.shape}[{H_ideal.min():.3f},{H_ideal.max():.3f}]"
                f"  recon=[{recon.min():.3f},{recon.max():.3f}]"
            )

    print(f"  Written H5: {h5_path} ({os.path.getsize(h5_path)/1e6:.2f} MB)")

    # spec.json
    spec_path = tier_dir / "spec.json"
    with open(spec_path, "w") as f:
        json.dump(cfg["spec"], f, indent=2)
    print(f"  Written spec: {spec_path}")

    # true_spec.json
    true_spec_path = tier_dir / "true_spec.json"
    with open(true_spec_path, "w") as f:
        json.dump(true_spec_records, f, indent=2)
    print(f"  Written true_spec: {true_spec_path}")

    return true_spec_records


# ═══════════════════════════════════════════════════════════════════════════════
# Verification pass
# ═══════════════════════════════════════════════════════════════════════════════

def verify_all():
    print("\n" + "=" * 60)
    print("  VERIFICATION PASS")
    print("=" * 60)

    all_ok = True
    for modality, cfg in MODALITY_CONFIG.items():
        for tier in ["public", "dev", "hidden"]:
            h5_path = OUT_BASE / modality / tier / f"{modality}_challenge_{tier}.h5"
            expected_n = TIER_SIZES[tier]
            expected_shapes = cfg["datasets"]

            try:
                with h5py.File(h5_path, "r") as hf:
                    keys = sorted(hf.keys())
                    n = len(keys)
                    ok = n == expected_n
                    print(f"\n  {modality}/{tier}: {n}/{expected_n} samples {'OK' if ok else 'MISMATCH'}")
                    if not ok:
                        all_ok = False

                    # Check first sample
                    grp = hf[keys[0]]
                    for ds_name, exp_shape in expected_shapes.items():
                        d = grp[ds_name][:]
                        shape_ok = d.shape == exp_shape
                        dtype_ok = d.dtype == np.float32
                        finite_ok = np.all(np.isfinite(d))
                        range_ok = d.min() >= 0 and d.max() <= 1.0 if ds_name != "H_ideal" or "tilt" not in modality else True
                        status = "OK" if (shape_ok and dtype_ok and finite_ok) else "FAIL"
                        if not (shape_ok and dtype_ok and finite_ok):
                            all_ok = False
                        print(f"    {ds_name}: shape={d.shape} exp={exp_shape} "
                              f"dtype={d.dtype} range=[{d.min():.4f},{d.max():.4f}] {status}")

            except Exception as e:
                print(f"  ERROR opening {h5_path}: {e}")
                all_ok = False

    print("\n" + "=" * 60)
    print(f"  OVERALL: {'ALL OK' if all_ok else 'SOME FAILURES'}")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    modalities = list(MODALITY_CONFIG.keys())
    tiers = ["public", "dev", "hidden"]

    print("Generating electron microscopy benchmark datasets")
    print(f"Modalities: {modalities}")
    print(f"Output base: {OUT_BASE}")

    for modality in modalities:
        for tier in tiers:
            generate_tier(modality, tier)

    verify_all()
    print("\nDone.")
