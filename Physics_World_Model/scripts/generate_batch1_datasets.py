#!/usr/bin/env python3
"""
Generate benchmark datasets for batch 1 modalities:
  acoustic_emission, acoustic_microscopy, active_thermography, adaptive_optics,
  afm, angiography, atom_probe, bioluminescence_tomo, brachytherapy_img, brillouin

Tiers: public (12 samples), dev (20 samples), hidden (20 samples)
Seeds: public  1000+i*17
       dev     7000+i*17
       hidden  9000+i*17
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "datasets" / "benchmark"

TIERS = [
    ("public",  12, 1000),
    ("dev",     20, 7000),
    ("hidden",  20, 9000),
]
SEED_STEP = 17


# ── Utility helpers ───────────────────────────────────────────────────────────

def _norm01(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return ((a - lo) / (hi - lo + 1e-12)).astype(np.float32)


def _save_png(arr: np.ndarray, path: Path) -> None:
    if not HAS_PIL:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        img = np.clip(_norm01(arr) * 255, 0, 255).astype(np.uint8)
        if img.ndim == 3:
            # take middle slice for 3D
            img = img[img.shape[0] // 2]
        Image.fromarray(img, "L").save(str(path))
    except Exception:
        pass


def compute_psnr(gt: np.ndarray, recon: np.ndarray) -> float:
    mse = float(np.mean((gt.astype(np.float64) - recon.astype(np.float64)) ** 2))
    if mse < 1e-12:
        return 100.0
    dr = float(gt.max() - gt.min())
    if dr < 1e-12:
        return 0.0
    return float(10.0 * np.log10(dr ** 2 / mse))


def make_blob_phantom(rng: np.random.Generator, size: int = 128, n_blobs: int = 6) -> np.ndarray:
    """Generic blob phantom: random Gaussians on a canvas."""
    canvas = np.zeros((size, size), dtype=np.float64)
    yy, xx = np.ogrid[:size, :size]
    for _ in range(n_blobs):
        cy = rng.integers(10, size - 10)
        cx = rng.integers(10, size - 10)
        sigma = rng.uniform(4, 20)
        amp = rng.uniform(0.3, 1.0)
        canvas += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
    return _norm01(canvas)


def make_structured_phantom(rng: np.random.Generator, size: int = 128) -> np.ndarray:
    """Phantom with rectangular regions (material boundaries)."""
    canvas = np.zeros((size, size), dtype=np.float64)
    n_rect = rng.integers(3, 8)
    for _ in range(n_rect):
        r0 = rng.integers(0, size - 10)
        c0 = rng.integers(0, size - 10)
        rh = rng.integers(5, size // 3)
        cw = rng.integers(5, size // 3)
        val = rng.uniform(0.2, 1.0)
        r1 = min(r0 + rh, size)
        c1 = min(c0 + cw, size)
        canvas[r0:r1, c0:c1] += val
    bg = gaussian_filter(rng.standard_normal((size, size)).astype(np.float64), sigma=5)
    canvas += 0.05 * bg
    return _norm01(canvas)


def wiener_deconv_2d(y: np.ndarray, psf: np.ndarray, reg: float = 0.01) -> np.ndarray:
    """Simple Wiener deconvolution in frequency domain."""
    from numpy.fft import fft2, ifft2, ifftshift
    Y = fft2(y.astype(np.float64))
    H = fft2(np.fft.ifftshift(psf.astype(np.float64)), s=y.shape)
    H2 = np.abs(H) ** 2
    recon = np.real(ifft2(np.conj(H) * Y / (H2 + reg)))
    return _norm01(recon)


# ── 1. ACOUSTIC EMISSION ─────────────────────────────────────────────────────
# x_true=(128,128) source map, y=(64,128) time-series, baseline=delay-and-sum

def gen_acoustic_emission(idx: int, seed: int):
    rng = np.random.default_rng(seed)
    size = 128
    # Source map: sparse point sources
    x_true = np.zeros((size, size), dtype=np.float64)
    n_src = rng.integers(2, 7)
    for _ in range(n_src):
        r = rng.integers(10, size - 10)
        c = rng.integers(10, size - 10)
        amp = rng.uniform(0.4, 1.0)
        x_true[r, c] = amp
    x_true = gaussian_filter(x_true, sigma=2.0)
    x_true = _norm01(x_true)

    # Sensor positions: 64 sensors on the boundary
    n_sensors = 64
    thetas = np.linspace(0, 2 * np.pi, n_sensors, endpoint=False)
    radius = size / 2 * 0.9
    centre = size / 2
    sx = centre + radius * np.cos(thetas)  # col
    sy = centre + radius * np.sin(thetas)  # row
    H_ideal = np.column_stack([sy, sx]).astype(np.float32)  # (64,2)

    n_time = 128
    y = np.zeros((n_sensors, n_time), dtype=np.float64)
    # Delay-and-sum forward: each sensor records delayed signal from each source
    sources_rc = np.argwhere(x_true > 0.01)
    for si in range(n_sensors):
        for src in sources_rc:
            dr = float(src[0]) - sy[si]
            dc = float(src[1]) - sx[si]
            dist = np.sqrt(dr ** 2 + dc ** 2)
            delay = int(np.round(dist / (size / n_time)))
            amp = float(x_true[src[0], src[1]])
            if delay < n_time:
                y[si, delay] += amp
    y += rng.normal(0, 0.02, y.shape)
    y = y.astype(np.float32)

    # Baseline: simple delay-and-sum back-projection
    recon = np.zeros((size, size), dtype=np.float64)
    rr, cc = np.mgrid[:size, :size]
    for si in range(n_sensors):
        for ti in range(n_time):
            if abs(y[si, ti]) < 1e-6:
                continue
            # Back-project along circle of radius proportional to ti
            r_px = ti * (size / n_time)
            dist = np.sqrt((rr - sy[si]) ** 2 + (cc - sx[si]) ** 2)
            mask = np.abs(dist - r_px) < 1.5
            recon[mask] += float(y[si, ti])
    recon = _norm01(recon)

    spec = {
        "n_sources": int(n_src),
        "noise_sigma": 0.02,
        "sensor_radius_px": float(radius),
    }
    meta = {
        "x_true_shape": list(x_true.shape),
        "y_shape": list(y.shape),
        "H_ideal_shape": list(H_ideal.shape),
        "description": "acoustic emission source map; y=(n_sensors,n_time) time-series",
    }
    return x_true, y, H_ideal, recon.astype(np.float32), spec, meta


# ── 2. ACOUSTIC MICROSCOPY ────────────────────────────────────────────────────
# x_true=(128,128), y=PSF convolved + noise, baseline=Wiener deconv

def gen_acoustic_microscopy(idx: int, seed: int):
    rng = np.random.default_rng(seed)
    size = 128
    x_true = make_structured_phantom(rng, size)

    psf_sigma = rng.uniform(2.0, 5.0)
    psf_size = 15
    cy_p, cx_p = psf_size // 2, psf_size // 2
    yy, xx = np.ogrid[:psf_size, :psf_size]
    psf = np.exp(-((yy - cy_p) ** 2 + (xx - cx_p) ** 2) / (2 * psf_sigma ** 2))
    psf = (psf / psf.sum()).astype(np.float32)

    y_clean = gaussian_filter(x_true.astype(np.float64), sigma=psf_sigma)
    noise_level = rng.uniform(0.01, 0.05)
    y = (y_clean + rng.normal(0, noise_level, y_clean.shape)).astype(np.float32)
    y = np.clip(y, 0, None)

    # Wiener deconvolution baseline
    psf_full = np.zeros((size, size), dtype=np.float64)
    psf_full[:psf_size, :psf_size] = psf
    recon = wiener_deconv_2d(y.astype(np.float64), psf_full, reg=0.05)

    H_ideal = psf  # (15,15) PSF kernel

    spec = {
        "psf_sigma": float(psf_sigma),
        "noise_level": float(noise_level),
        "psf_size": psf_size,
    }
    meta = {
        "x_true_shape": list(x_true.shape),
        "y_shape": list(y.shape),
        "H_ideal_shape": list(H_ideal.shape),
        "description": "acoustic microscopy; y=x*PSF+noise; H_ideal=Gaussian PSF",
    }
    return x_true, y, H_ideal, recon.astype(np.float32), spec, meta


# ── 3. ACTIVE THERMOGRAPHY ────────────────────────────────────────────────────
# x_true=(128,128) thermal diffusivity map, y=(64,128) surface temp sequence, baseline=avg

def gen_active_thermography(idx: int, seed: int):
    rng = np.random.default_rng(seed)
    size = 128
    n_frames = 64

    # Thermal diffusivity map: background + defects (lower diffusivity)
    x_true = np.ones((size, size), dtype=np.float64) * 0.5
    n_defects = rng.integers(2, 6)
    for _ in range(n_defects):
        r0 = rng.integers(10, size - 20)
        c0 = rng.integers(10, size - 20)
        rh = rng.integers(5, 25)
        cw = rng.integers(5, 25)
        val = rng.uniform(0.1, 0.4)
        x_true[r0:r0+rh, c0:c0+cw] = val
    x_true = _norm01(x_true)

    # Surface temperature sequence: heat diffuses, defects appear as hot spots
    y = np.zeros((n_frames, size), dtype=np.float64)
    for t in range(n_frames):
        sigma_t = 1.0 + t * 0.5
        # Integrate along columns (depth), weighted by diffusivity
        col_mean = x_true.mean(axis=0)  # (128,)
        smoothed = gaussian_filter(col_mean, sigma=sigma_t)
        noise = rng.normal(0, 0.02, smoothed.shape)
        y[t] = smoothed + noise

    y = _norm01(y.astype(np.float32))

    # Baseline: average over time frames (each frame is a 1D row)
    # Expand back to 2D by tiling
    y_avg = y.mean(axis=0)  # (128,)
    recon = np.tile(y_avg, (size, 1))  # (128,128)
    recon = _norm01(recon)

    H_ideal = np.array([n_frames, size, 0.5], dtype=np.float32)  # metadata vector

    spec = {
        "n_defects": int(n_defects),
        "n_frames": n_frames,
        "background_diffusivity": 0.5,
    }
    meta = {
        "x_true_shape": list(x_true.shape),
        "y_shape": list(y.shape),
        "H_ideal_shape": list(H_ideal.shape),
        "description": "active thermography; y=(n_frames,128) surface temp; H_ideal=[n_frames,size,bg_alpha]",
    }
    return x_true, y.astype(np.float32), H_ideal, recon.astype(np.float32), spec, meta


# ── 4. ADAPTIVE OPTICS ────────────────────────────────────────────────────────
# x_true=(128,128), y=aberrated (multiply by random phase), baseline=direct correction

def gen_adaptive_optics(idx: int, seed: int):
    rng = np.random.default_rng(seed)
    size = 128

    # Clean star/object image
    x_true = make_blob_phantom(rng, size, n_blobs=rng.integers(3, 8))

    # Wavefront aberration: smooth random phase
    phase_rms = rng.uniform(0.5, 2.5)  # radians
    raw_phase = rng.standard_normal((size, size))
    phase = gaussian_filter(raw_phase, sigma=10) * phase_rms
    phase = phase.astype(np.float64)

    # Aberrated image: multiply complex amplitude by phase, take magnitude
    amplitude = np.sqrt(np.maximum(x_true.astype(np.float64), 0))
    complex_field = amplitude * np.exp(1j * phase)
    y = np.abs(complex_field) ** 2  # intensity
    noise = rng.normal(0, 0.02, y.shape)
    y = _norm01((y + noise).astype(np.float32))

    # Baseline: divide by estimated phase correction (simple direct correction)
    phase_est = gaussian_filter(phase, sigma=5)  # smoothed phase estimate
    corrected = np.abs(complex_field * np.exp(-1j * phase_est)) ** 2
    recon = _norm01(corrected.astype(np.float32))

    H_ideal = phase.astype(np.float32)  # (128,128) true wavefront

    spec = {
        "phase_rms_rad": float(phase_rms),
        "aberration_smoothing_sigma": 10.0,
        "noise_sigma": 0.02,
    }
    meta = {
        "x_true_shape": list(x_true.shape),
        "y_shape": list(y.shape),
        "H_ideal_shape": list(H_ideal.shape),
        "description": "adaptive optics; y=|sqrt(x)*exp(i*phase)|^2; H_ideal=true wavefront",
    }
    return x_true, y, H_ideal, recon, spec, meta


# ── 5. AFM ────────────────────────────────────────────────────────────────────
# x_true=(128,128) surface height map, y=x + tip convolution, baseline=deconv

def gen_afm(idx: int, seed: int):
    rng = np.random.default_rng(seed)
    size = 128

    # Surface height map: smooth terrain with features
    base = gaussian_filter(rng.standard_normal((size, size)).astype(np.float64), sigma=15)
    # Add sharp features
    n_features = rng.integers(3, 9)
    yy, xx = np.ogrid[:size, :size]
    for _ in range(n_features):
        cy = rng.integers(10, size - 10)
        cx = rng.integers(10, size - 10)
        h = rng.uniform(0.3, 1.0)
        r = rng.uniform(2, 8)
        base += h * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r ** 2))
    x_true = _norm01(base.astype(np.float32))

    # Tip convolution: small Gaussian tip
    tip_sigma = rng.uniform(1.5, 3.5)
    y_clean = gaussian_filter(x_true.astype(np.float64), sigma=tip_sigma)
    noise_level = rng.uniform(0.005, 0.02)
    y = (y_clean + rng.normal(0, noise_level, y_clean.shape)).astype(np.float32)
    y = np.clip(y, 0, None)

    # Baseline: Wiener deconvolution
    psf_full = np.zeros((size, size), dtype=np.float64)
    cy_p, cx_p = size // 2, size // 2
    yg, xg = np.ogrid[:size, :size]
    psf_full = np.exp(-((yg - cy_p) ** 2 + (xg - cx_p) ** 2) / (2 * tip_sigma ** 2))
    psf_full /= psf_full.sum()
    recon = wiener_deconv_2d(y.astype(np.float64), psf_full, reg=0.02)

    H_ideal = np.array([[tip_sigma, noise_level]], dtype=np.float32)  # (1,2) tip params

    spec = {
        "tip_sigma_px": float(tip_sigma),
        "noise_level": float(noise_level),
    }
    meta = {
        "x_true_shape": list(x_true.shape),
        "y_shape": list(y.shape),
        "H_ideal_shape": list(H_ideal.shape),
        "description": "AFM; y=x*tip_PSF+noise; H_ideal=[[tip_sigma, noise_level]]",
    }
    return x_true, y, H_ideal, recon.astype(np.float32), spec, meta


# ── 6. ANGIOGRAPHY ────────────────────────────────────────────────────────────
# x_true=(256,256) vessel map, y=projection + Poisson noise, baseline=FBP-like

def gen_angiography(idx: int, seed: int):
    rng = np.random.default_rng(seed)
    size = 256
    n_angles = 64

    # Vessel map: branching structure
    canvas = np.zeros((size, size), dtype=np.float64)
    centre = size / 2
    # Random branching vessels
    for _ in range(rng.integers(3, 7)):
        angle = rng.uniform(0, 2 * np.pi)
        cy, cx = float(centre), float(centre)
        thick = rng.uniform(2.0, 6.0)
        for _ in range(int(size * 0.4)):
            angle += rng.normal(0, 0.06)
            cy += np.sin(angle)
            cx += np.cos(angle)
            if not (1 <= cy < size - 1 and 1 <= cx < size - 1):
                break
            r = max(1, int(np.ceil(thick)))
            yy = np.arange(max(0, int(cy) - r), min(size, int(cy) + r + 1))
            xx = np.arange(max(0, int(cx) - r), min(size, int(cx) + r + 1))
            for ry in yy:
                for rx in xx:
                    d2 = (ry - cy) ** 2 + (rx - cx) ** 2
                    canvas[ry, rx] = max(canvas[ry, rx],
                                         np.exp(-d2 / (2 * thick ** 2)))
    x_true = _norm01(canvas.astype(np.float32))

    # Projection at random angles (simple line integral)
    angles = np.linspace(0, np.pi, n_angles, endpoint=False)
    projections = np.zeros((n_angles, size), dtype=np.float64)
    for ai, ang in enumerate(angles):
        cos_a, sin_a = np.cos(ang), np.sin(ang)
        for col in range(size):
            t = col - size / 2
            # Line integral along the projection direction
            rr = np.arange(size) - size / 2
            pts_row = (rr * cos_a - t * sin_a + size / 2).astype(int)
            pts_col = (rr * sin_a + t * cos_a + size / 2).astype(int)
            valid = (pts_row >= 0) & (pts_row < size) & (pts_col >= 0) & (pts_col < size)
            projections[ai, col] = x_true[pts_row[valid], pts_col[valid]].sum()

    # Poisson noise
    scale = 100.0
    projections_noisy = rng.poisson(np.maximum(projections * scale, 0)).astype(np.float64)
    projections_noisy = projections_noisy / scale
    y = projections_noisy.astype(np.float32)

    # Baseline: simple back-projection (FBP-like)
    recon = np.zeros((size, size), dtype=np.float64)
    for ai, ang in enumerate(angles):
        cos_a, sin_a = np.cos(ang), np.sin(ang)
        rr_grid, cc_grid = np.mgrid[:size, :size]
        t_coords = ((cc_grid - size / 2) * cos_a + (rr_grid - size / 2) * sin_a
                    + size / 2).astype(int)
        t_coords = np.clip(t_coords, 0, size - 1)
        recon += projections_noisy[ai][t_coords]
    recon = _norm01(recon.astype(np.float32))

    H_ideal = angles.astype(np.float32)  # (64,) projection angles in radians

    spec = {
        "n_angles": n_angles,
        "poisson_scale": scale,
    }
    meta = {
        "x_true_shape": list(x_true.shape),
        "y_shape": list(y.shape),
        "H_ideal_shape": list(H_ideal.shape),
        "description": "angiography; y=(n_angles,256) projections+Poisson; H_ideal=angles(rad)",
    }
    return x_true, y, H_ideal, recon, spec, meta


# ── 7. ATOM PROBE ─────────────────────────────────────────────────────────────
# x_true=(64,64,64) 3D atomic map, y=(64,4096) detector hits, baseline=KDE reshape

def gen_atom_probe(idx: int, seed: int):
    rng = np.random.default_rng(seed)
    vol_size = 64
    n_det_rows = 64
    n_det_cols = 4096

    # 3D atomic position density map
    x_true = np.zeros((vol_size, vol_size, vol_size), dtype=np.float64)
    n_phases = rng.integers(2, 5)
    for _ in range(n_phases):
        cz = rng.integers(5, vol_size - 5)
        cy = rng.integers(5, vol_size - 5)
        cx = rng.integers(5, vol_size - 5)
        sigma = rng.uniform(5, 15)
        amp = rng.uniform(0.3, 1.0)
        zg, yg, xg = np.ogrid[:vol_size, :vol_size, :vol_size]
        x_true += amp * np.exp(-((zg - cz)**2 + (yg - cy)**2 + (xg - cx)**2) / (2 * sigma**2))
    x_true = _norm01(x_true.astype(np.float32))

    # Detector: project atoms onto 2D detector with mass-to-charge (flight time)
    # Simplified: flatten z (depth) into time-of-flight axis, x/y onto detector
    x_flat = x_true.reshape(vol_size, -1)  # (64, 64*64) = (64, 4096)
    noise = rng.normal(0, 0.01, x_flat.shape)
    y = (x_flat + noise).astype(np.float32)

    # H_ideal: voxel size and detector geometry
    H_ideal = np.array([vol_size, vol_size, vol_size, n_det_rows, n_det_cols],
                       dtype=np.float32)  # (5,)

    # Baseline: reshape y back and apply KDE (Gaussian smoothing)
    recon_flat = y.reshape(vol_size, vol_size, vol_size)
    recon = gaussian_filter(recon_flat.astype(np.float64), sigma=1.5)
    recon = _norm01(recon.astype(np.float32))

    spec = {
        "n_phases": int(n_phases),
        "vol_size": vol_size,
        "noise_sigma": 0.01,
    }
    meta = {
        "x_true_shape": list(x_true.shape),
        "y_shape": list(y.shape),
        "H_ideal_shape": list(H_ideal.shape),
        "description": "atom probe; x=(64,64,64) 3D; y=(64,4096) detector hits; H_ideal=[vz,vy,vx,dr,dc]",
    }
    return x_true, y, H_ideal, recon, spec, meta


# ── 8. BIOLUMINESCENCE TOMOGRAPHY ─────────────────────────────────────────────
# x_true=(128,128) source, y=(32,128) projections, baseline=FBP approximation

def gen_bioluminescence_tomo(idx: int, seed: int):
    rng = np.random.default_rng(seed)
    size = 128
    n_proj = 32

    # Bioluminescent source map: sparse glowing regions
    x_true = make_blob_phantom(rng, size, n_blobs=rng.integers(2, 5))
    x_true = x_true * (x_true > 0.1)  # threshold for sparsity
    x_true = _norm01(x_true)

    # Projections at n_proj angles (simplified Radon-like)
    angles = np.linspace(0, np.pi, n_proj, endpoint=False)
    y = np.zeros((n_proj, size), dtype=np.float64)
    for ai, ang in enumerate(angles):
        cos_a, sin_a = np.cos(ang), np.sin(ang)
        for col in range(size):
            t = col - size / 2
            rr = np.arange(size) - size / 2
            pts_row = (rr * cos_a - t * sin_a + size / 2).astype(int)
            pts_col = (rr * sin_a + t * cos_a + size / 2).astype(int)
            valid = ((pts_row >= 0) & (pts_row < size) &
                     (pts_col >= 0) & (pts_col < size))
            y[ai, col] = x_true[pts_row[valid], pts_col[valid]].sum()

    noise_level = rng.uniform(0.01, 0.05)
    y += rng.normal(0, noise_level, y.shape)
    y = np.maximum(y, 0).astype(np.float32)

    # Baseline: simple back-projection
    recon = np.zeros((size, size), dtype=np.float64)
    for ai, ang in enumerate(angles):
        cos_a, sin_a = np.cos(ang), np.sin(ang)
        rr_g, cc_g = np.mgrid[:size, :size]
        t_coords = ((cc_g - size / 2) * cos_a + (rr_g - size / 2) * sin_a
                    + size / 2).astype(int)
        t_coords = np.clip(t_coords, 0, size - 1)
        recon += y[ai][t_coords]
    recon = _norm01(recon.astype(np.float32))

    H_ideal = angles.astype(np.float32)  # (32,) angles in radians

    spec = {
        "n_projections": n_proj,
        "noise_level": float(noise_level),
    }
    meta = {
        "x_true_shape": list(x_true.shape),
        "y_shape": list(y.shape),
        "H_ideal_shape": list(H_ideal.shape),
        "description": "bioluminescence tomo; y=(32,128) projections; H_ideal=angles(rad)",
    }
    return x_true, y, H_ideal, recon, spec, meta


# ── 9. BRACHYTHERAPY IMAGING ──────────────────────────────────────────────────
# x_true=(128,128) dose map, y=x + Gaussian noise, baseline=Wiener filter

def gen_brachytherapy_img(idx: int, seed: int):
    rng = np.random.default_rng(seed)
    size = 128

    # Dose distribution: superposition of seed dose kernels (1/r^2 falloff)
    x_true = np.zeros((size, size), dtype=np.float64)
    n_seeds = rng.integers(3, 8)
    yy, xx = np.ogrid[:size, :size]
    seed_positions = []
    for _ in range(n_seeds):
        sy = rng.integers(20, size - 20)
        sx = rng.integers(20, size - 20)
        strength = rng.uniform(0.5, 1.0)
        dist2 = (yy - sy) ** 2 + (xx - sx) ** 2 + 4.0  # avoid 1/0
        x_true += strength / dist2
        seed_positions.append([int(sy), int(sx), float(strength)])
    x_true = _norm01(x_true.astype(np.float32))

    # Measurement: Gaussian blur (detector PSF) + noise
    blur_sigma = rng.uniform(1.5, 4.0)
    noise_level = rng.uniform(0.02, 0.08)
    y_blurred = gaussian_filter(x_true.astype(np.float64), sigma=blur_sigma)
    y = (y_blurred + rng.normal(0, noise_level, y_blurred.shape)).astype(np.float32)
    y = np.clip(y, 0, None)

    # Baseline: Wiener filter deconvolution
    psf_full = np.zeros((size, size), dtype=np.float64)
    yg, xg = np.ogrid[:size, :size]
    cy_p, cx_p = size // 2, size // 2
    psf_full = np.exp(-((yg - cy_p) ** 2 + (xg - cx_p) ** 2) / (2 * blur_sigma ** 2))
    psf_full /= psf_full.sum()
    recon = wiener_deconv_2d(y.astype(np.float64), psf_full, reg=0.05)

    H_ideal = np.array(seed_positions, dtype=np.float32)  # (n_seeds, 3)

    spec = {
        "n_seeds": int(n_seeds),
        "blur_sigma": float(blur_sigma),
        "noise_level": float(noise_level),
    }
    meta = {
        "x_true_shape": list(x_true.shape),
        "y_shape": list(y.shape),
        "H_ideal_shape": list(H_ideal.shape),
        "description": "brachytherapy; y=dose_blurred+noise; H_ideal=(n_seeds,3) [row,col,strength]",
    }
    return x_true, y, H_ideal, recon.astype(np.float32), spec, meta


# ── 10. BRILLOUIN IMAGING ─────────────────────────────────────────────────────
# x_true=(128,128) Brillouin shift map, y=noisy spectral data, baseline=median filter

def gen_brillouin(idx: int, seed: int):
    rng = np.random.default_rng(seed)
    size = 128

    # Brillouin frequency shift map: smooth spatial variation with step features
    base = gaussian_filter(rng.standard_normal((size, size)).astype(np.float64), sigma=20)
    # Add step-like features for different material regions
    n_regions = rng.integers(2, 5)
    yy, xx = np.ogrid[:size, :size]
    for _ in range(n_regions):
        cy = rng.integers(10, size - 10)
        cx = rng.integers(10, size - 10)
        sigma = rng.uniform(10, 40)
        val = rng.uniform(0.3, 0.8)
        base += val * np.exp(-((yy - cy)**2 + (xx - cx)**2) / (2 * sigma**2))
    x_true = _norm01(base.astype(np.float32))

    # Simulated spectral measurement: each pixel has a Lorentzian peak at the
    # Brillouin shift frequency; we just add Poisson + Gaussian noise
    shot_noise_scale = rng.uniform(30, 100)
    read_noise = rng.uniform(0.01, 0.05)
    y_clean = x_true.astype(np.float64)
    y_noisy = (rng.poisson(np.maximum(y_clean * shot_noise_scale, 0)).astype(np.float64)
               / shot_noise_scale)
    y_noisy += rng.normal(0, read_noise, y_noisy.shape)
    y = np.clip(y_noisy, 0, None).astype(np.float32)

    # Baseline: median filter to suppress shot noise
    recon = median_filter(y.astype(np.float64), size=5)
    recon = _norm01(recon.astype(np.float32))

    H_ideal = np.array([shot_noise_scale, read_noise], dtype=np.float32)  # (2,)

    spec = {
        "shot_noise_scale": float(shot_noise_scale),
        "read_noise": float(read_noise),
        "n_regions": int(n_regions),
    }
    meta = {
        "x_true_shape": list(x_true.shape),
        "y_shape": list(y.shape),
        "H_ideal_shape": list(H_ideal.shape),
        "description": "Brillouin; y=Poisson+Gaussian noise on shift map; baseline=median filter",
    }
    return x_true, y, H_ideal, recon, spec, meta


# ── Registry ──────────────────────────────────────────────────────────────────

MODALITIES = [
    ("acoustic_emission",    gen_acoustic_emission),
    ("acoustic_microscopy",  gen_acoustic_microscopy),
    ("active_thermography",  gen_active_thermography),
    ("adaptive_optics",      gen_adaptive_optics),
    ("afm",                  gen_afm),
    ("angiography",          gen_angiography),
    ("atom_probe",           gen_atom_probe),
    ("bioluminescence_tomo", gen_bioluminescence_tomo),
    ("brachytherapy_img",    gen_brachytherapy_img),
    ("brillouin",            gen_brillouin),
]


# ── Tier generator ────────────────────────────────────────────────────────────

def generate_tier(modality: str, gen_fn, tier: str, n_samples: int, base_seed: int) -> dict:
    tier_dir = BENCH / modality / tier
    images_dir = tier_dir / "images"
    tier_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    h5_path = tier_dir / f"{modality}_challenge_{tier}.h5"
    true_specs: dict = {}
    psnrs = []

    print(f"  [{modality}][{tier}] {n_samples} samples -> {h5_path.name}", flush=True)

    with h5py.File(h5_path, "w") as f:
        f.attrs["modality"] = modality
        f.attrs["tier"] = tier
        f.attrs["n_samples"] = n_samples

        for idx in range(n_samples):
            seed = base_seed + idx * SEED_STEP
            key = f"sample_{idx:02d}"

            x_true, y, H_ideal, recon, spec, meta = gen_fn(idx, seed)

            psnr = compute_psnr(x_true, recon)
            psnrs.append(psnr)
            true_specs[key] = spec

            grp = f.create_group(key)
            grp.create_dataset("x_true", data=x_true.astype(np.float32), compression="gzip")
            grp.create_dataset("y", data=y.astype(np.float32), compression="gzip")
            grp.create_dataset("H_ideal", data=H_ideal.astype(np.float32), compression="gzip")
            grp.create_dataset("reconstruction_baseline",
                               data=recon.astype(np.float32), compression="gzip")
            grp.attrs["true_spec"] = json.dumps(spec)
            grp.attrs["metadata"] = json.dumps({
                **meta,
                "psnr_baseline_db": round(psnr, 2),
                "seed": int(seed),
            })

            # Save PNG previews
            sample_img_dir = images_dir / f"sample_{idx:02d}"
            sample_img_dir.mkdir(parents=True, exist_ok=True)
            _save_png(x_true, sample_img_dir / "x_true.png")
            _save_png(y, sample_img_dir / "y_measurement.png")
            _save_png(recon, sample_img_dir / "reconstruction_baseline.png")

    # spec.json
    n_dim = x_true.ndim
    spec_doc = {
        "modality": modality,
        "tier": tier,
        "n_samples": n_samples,
        "measurement_key": "y",
        "groundtruth_key": "x_true",
        "forward_operator_key": "H_ideal",
        "baseline_key": "reconstruction_baseline",
        "x_true_shape": list(x_true.shape),
        "y_shape": list(y.shape),
        "dtype": "float32",
        "mean_psnr_baseline_db": round(float(np.mean(psnrs)), 2),
    }
    with open(tier_dir / "spec.json", "w") as sf:
        json.dump(spec_doc, sf, indent=2)

    with open(tier_dir / "true_spec.json", "w") as tf:
        json.dump(true_specs, tf, indent=2)

    size_mb = os.path.getsize(h5_path) / 1e6
    print(f"    -> PSNR mean={np.mean(psnrs):.1f}dB  size={size_mb:.1f}MB", flush=True)
    return {"tier": tier, "n_samples": n_samples, "mean_psnr": float(np.mean(psnrs)),
            "h5_size_mb": size_mb}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("Batch 1 Dataset Generator: 10 modalities x 3 tiers")
    print(f"Output root: {BENCH}")
    print("=" * 70)

    summary = {}
    for mod_name, gen_fn in MODALITIES:
        print(f"\n[{mod_name}]")
        mod_results = {}
        for tier, n_samples, base_seed in TIERS:
            try:
                result = generate_tier(mod_name, gen_fn, tier, n_samples, base_seed)
                mod_results[tier] = result
            except Exception as e:
                print(f"  ERROR [{mod_name}][{tier}]: {e}")
                import traceback
                traceback.print_exc()
                mod_results[tier] = {"error": str(e)}
        summary[mod_name] = mod_results

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for mod, tiers in summary.items():
        ok = all("error" not in v for v in tiers.values())
        status = "OK" if ok else "FAIL"
        print(f"  {status:4s}  {mod}")
        for t, v in tiers.items():
            if "error" in v:
                print(f"         [{t}] ERROR: {v['error']}")
            else:
                print(f"         [{t}] {v['n_samples']} samples, "
                      f"PSNR={v['mean_psnr']:.1f}dB, {v['h5_size_mb']:.1f}MB")

    print("\nDone.")


if __name__ == "__main__":
    main()
