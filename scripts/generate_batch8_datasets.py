#!/usr/bin/env python3
"""
Generate benchmark datasets for batch 8 modalities:
  particle_calorimetry, passive_microwave, phase_contrast, photometric_stereo,
  polarization, polsar, portal_imaging, proton_radiography, proton_therapy_img,
  pump_probe

Tiers: public (12 samples), dev (20 samples), hidden (20 samples)
Seeds:  public  = 1200 + i*17
        dev     = 7200 + i*17
        hidden  = 9300 + i*17
Image size: 128x128
"""

import json
import os
import struct
import zlib
import numpy as np
import h5py
from pathlib import Path
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

# ── Root directory ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "datasets" / "benchmark"

# ── Tier sample counts ────────────────────────────────────────────────────────
TIER_COUNTS = {"public": 12, "dev": 20, "hidden": 20}

# ── Seed formulas ─────────────────────────────────────────────────────────────
SEED_BASE = {"public": 1200, "dev": 7200, "hidden": 9300}
SEED_STEP = 17


# ── PNG writer (pure numpy, no pillow required) ────────────────────────────────
def _write_png(path: Path, arr: np.ndarray) -> None:
    """Write a float32 2D array as an 8-bit greyscale PNG (pure stdlib)."""
    arr = np.asarray(arr, dtype=np.float32)
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    else:
        arr = np.zeros_like(arr)
    img = (arr * 255).clip(0, 255).astype(np.uint8)

    h, w = img.shape
    raw_rows = b"".join(b"\x00" + img[r].tobytes() for r in range(h))
    def make_chunk(tag, data):
        c = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", c)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)
    ihdr = make_chunk(b"IHDR", ihdr_data)
    idat = make_chunk(b"IDAT", zlib.compress(raw_rows, 9))
    iend = make_chunk(b"IEND", b"")

    path.write_bytes(sig + ihdr + idat + iend)


def save_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.ndim == 3:
        arr = arr[..., 0] if arr.shape[2] <= 4 else arr[0]
    elif arr.ndim > 3:
        arr = arr[0]
    _write_png(path, arr.astype(np.float32))


# ── Gaussian PSF kernel ───────────────────────────────────────────────────────
def gauss_kernel(size: int, sigma: float) -> np.ndarray:
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return (k / k.sum()).astype(np.float32)


def wiener_deconv(y: np.ndarray, k: np.ndarray, snr: float = 20.0) -> np.ndarray:
    """Frequency-domain Wiener deconvolution."""
    Y = np.fft.fft2(y)
    K = np.fft.fft2(k, s=y.shape)
    nsr = 1.0 / snr
    X_hat = Y * np.conj(K) / (np.abs(K)**2 + nsr)
    x = np.real(np.fft.ifft2(X_hat))
    return x.astype(np.float32)


# ── Helper: write spec.json + true_spec.json ─────────────────────────────────
def write_specs(tier_dir: Path, modality: str, tier: str, n: int, extra: dict = None):
    spec = {
        "modality": modality,
        "tier": tier,
        "n_samples": n,
        "version": "1.0",
        "created": "2026-03-10",
    }
    if extra:
        spec.update(extra)
    true_spec = dict(spec)
    true_spec["contains_ground_truth"] = True

    (tier_dir / "spec.json").write_text(json.dumps(spec, indent=2))
    (tier_dir / "true_spec.json").write_text(json.dumps(true_spec, indent=2))


# ════════════════════════════════════════════════════════════════════════════
# 1. PARTICLE CALORIMETRY
# ════════════════════════════════════════════════════════════════════════════
def gen_particle_calorimetry(tier: str, n: int, seed_base: int):
    mod = "particle_calorimetry"
    tier_dir = BENCH / mod / tier
    tier_dir.mkdir(parents=True, exist_ok=True)
    img_dir = tier_dir / "images"
    img_dir.mkdir(exist_ok=True)

    fname = tier_dir / f"{mod}_challenge_{tier}.h5"
    sigma_psf = 3.0
    k = gauss_kernel(15, sigma_psf)

    with h5py.File(fname, "w") as hf:
        hf.attrs.update({"tier": tier, "variant": mod, "version": "1.0",
                          "sigma_psf": sigma_psf})
        for i in range(n):
            rng = np.random.default_rng(seed_base + i * SEED_STEP)
            # Sparse shower: Poisson-distributed depositions
            x = np.zeros((128, 128), dtype=np.float32)
            n_hits = rng.integers(30, 120)
            cx, cy = rng.integers(20, 108, size=2)
            # Shower core cluster
            angles = rng.uniform(0, 2 * np.pi, n_hits)
            radii = rng.exponential(scale=15, size=n_hits)
            xs = (cx + radii * np.cos(angles)).astype(int).clip(0, 127)
            ys = (cy + radii * np.sin(angles)).astype(int).clip(0, 127)
            for xp, yp in zip(xs, ys):
                x[yp, xp] += rng.exponential(1.0)

            H_ideal = gaussian_filter(x, sigma=sigma_psf)
            noise = rng.normal(0, 0.02 * H_ideal.max() + 1e-6, (128, 128)).astype(np.float32)
            y = (H_ideal + noise).astype(np.float32)
            recon = wiener_deconv(y, k, snr=20.0)

            grp = hf.create_group(f"sample_{i:02d}")
            grp.create_dataset("x_true", data=x)
            grp.create_dataset("y", data=y)
            grp.create_dataset("H_ideal", data=H_ideal.astype(np.float32))
            grp.create_dataset("reconstruction_baseline", data=recon)

            save_png(img_dir / f"sample_{i:02d}_x.png", x)
            save_png(img_dir / f"sample_{i:02d}_y.png", y)

    write_specs(tier_dir, mod, tier, n, {"sigma_psf": sigma_psf})
    print(f"  [OK] {mod}/{tier}: {n} samples -> {fname.name}")


# ════════════════════════════════════════════════════════════════════════════
# 2. PASSIVE MICROWAVE
# ════════════════════════════════════════════════════════════════════════════
def gen_passive_microwave(tier: str, n: int, seed_base: int):
    mod = "passive_microwave"
    tier_dir = BENCH / mod / tier
    tier_dir.mkdir(parents=True, exist_ok=True)
    img_dir = tier_dir / "images"
    img_dir.mkdir(exist_ok=True)

    fname = tier_dir / f"{mod}_challenge_{tier}.h5"
    sigma_beam = 5.0
    k = gauss_kernel(21, sigma_beam)

    with h5py.File(fname, "w") as hf:
        hf.attrs.update({"tier": tier, "variant": mod, "version": "1.0",
                          "sigma_beam_pixels": sigma_beam})
        for i in range(n):
            rng = np.random.default_rng(seed_base + i * SEED_STEP)
            # Brightness temperature map: smooth patches (land/sea/cloud)
            x = np.zeros((128, 128), dtype=np.float32)
            # Background ocean ~180K
            x[:] = 180.0
            # Land patches ~260K
            n_land = rng.integers(2, 6)
            for _ in range(n_land):
                cx, cy = rng.integers(10, 118, 2)
                r = rng.integers(10, 35)
                yy, xx = np.ogrid[:128, :128]
                mask = (xx - cx)**2 + (yy - cy)**2 < r**2
                x[mask] = rng.uniform(240, 280)
            # Cloud cold patches ~220K
            n_cloud = rng.integers(1, 4)
            for _ in range(n_cloud):
                cx, cy = rng.integers(10, 118, 2)
                r = rng.integers(5, 20)
                yy, xx = np.ogrid[:128, :128]
                mask = (xx - cx)**2 + (yy - cy)**2 < r**2
                x[mask] = rng.uniform(210, 230)

            H_ideal = gaussian_filter(x, sigma=sigma_beam)
            noise_std = 2.0  # 2K NEDT
            noise = rng.normal(0, noise_std, (128, 128)).astype(np.float32)
            y = (H_ideal + noise).astype(np.float32)
            recon = wiener_deconv(y, k, snr=30.0)

            grp = hf.create_group(f"sample_{i:02d}")
            grp.create_dataset("x_true", data=x)
            grp.create_dataset("y", data=y)
            grp.create_dataset("H_ideal", data=H_ideal.astype(np.float32))
            grp.create_dataset("reconstruction_baseline", data=recon)

            save_png(img_dir / f"sample_{i:02d}_x.png", x)
            save_png(img_dir / f"sample_{i:02d}_y.png", y)

    write_specs(tier_dir, mod, tier, n, {"sigma_beam_pixels": sigma_beam, "nedt_K": 2.0})
    print(f"  [OK] {mod}/{tier}: {n} samples -> {fname.name}")


# ════════════════════════════════════════════════════════════════════════════
# 3. PHASE CONTRAST
# ════════════════════════════════════════════════════════════════════════════
def gen_phase_contrast(tier: str, n: int, seed_base: int):
    mod = "phase_contrast"
    tier_dir = BENCH / mod / tier
    tier_dir.mkdir(parents=True, exist_ok=True)
    img_dir = tier_dir / "images"
    img_dir.mkdir(exist_ok=True)

    fname = tier_dir / f"{mod}_challenge_{tier}.h5"
    prop_phase = 0.3  # propagation phase factor (radians)

    with h5py.File(fname, "w") as hf:
        hf.attrs.update({"tier": tier, "variant": mod, "version": "1.0",
                          "propagation_phase": prop_phase})
        for i in range(n):
            rng = np.random.default_rng(seed_base + i * SEED_STEP)
            # Phase object: smooth random phase distribution (transparent specimen)
            x_raw = rng.standard_normal((128, 128)).astype(np.float32)
            x = gaussian_filter(x_raw, sigma=5.0).astype(np.float32)
            # Normalize to [-pi/4, pi/4]
            x = x / (np.abs(x).max() + 1e-8) * (np.pi / 4)

            # TIE forward model: I = 1 + 2*phi*sin(prop_phase)
            # Simplified: y = 1 + 2*x*sin(prop_phase)
            H_ideal = (1.0 + 2.0 * x * np.sin(prop_phase)).astype(np.float32)
            noise = rng.normal(0, 0.01, (128, 128)).astype(np.float32)
            y = (H_ideal + noise).clip(0).astype(np.float32)

            # TIE reconstruction: differentiate intensity pattern
            # phi ~ (I-1) / (2*sin(prop_phase)) smoothed
            recon = gaussian_filter(
                (y - 1.0) / (2.0 * np.sin(prop_phase) + 1e-8), sigma=1.0
            ).astype(np.float32)

            # H_ideal stored as the noise-free intensity
            grp = hf.create_group(f"sample_{i:02d}")
            grp.create_dataset("x_true", data=x)
            grp.create_dataset("y", data=y)
            grp.create_dataset("H_ideal", data=H_ideal)
            grp.create_dataset("reconstruction_baseline", data=recon)

            save_png(img_dir / f"sample_{i:02d}_x.png", x)
            save_png(img_dir / f"sample_{i:02d}_y.png", y)

    write_specs(tier_dir, mod, tier, n, {"propagation_phase_rad": prop_phase})
    print(f"  [OK] {mod}/{tier}: {n} samples -> {fname.name}")


# ════════════════════════════════════════════════════════════════════════════
# 4. PHOTOMETRIC STEREO
# ════════════════════════════════════════════════════════════════════════════
def gen_photometric_stereo(tier: str, n: int, seed_base: int):
    mod = "photometric_stereo"
    tier_dir = BENCH / mod / tier
    tier_dir.mkdir(parents=True, exist_ok=True)
    img_dir = tier_dir / "images"
    img_dir.mkdir(exist_ok=True)

    fname = tier_dir / f"{mod}_challenge_{tier}.h5"

    # 4 light directions (unit vectors) arranged at 45-deg azimuth intervals, 45-deg elevation
    el = np.deg2rad(45.0)
    light_dirs = np.array([
        [np.cos(el)*np.cos(0),     np.cos(el)*np.sin(0),     np.sin(el)],
        [np.cos(el)*np.cos(np.pi/2), np.cos(el)*np.sin(np.pi/2), np.sin(el)],
        [np.cos(el)*np.cos(np.pi),  np.cos(el)*np.sin(np.pi),  np.sin(el)],
        [np.cos(el)*np.cos(3*np.pi/2), np.cos(el)*np.sin(3*np.pi/2), np.sin(el)],
    ], dtype=np.float32)  # (4,3)

    # Pseudo-inverse for reconstruction
    L_pinv = np.linalg.pinv(light_dirs)  # (3,4)

    with h5py.File(fname, "w") as hf:
        hf.attrs.update({"tier": tier, "variant": mod, "version": "1.0",
                          "n_lights": 4})
        for i in range(n):
            rng = np.random.default_rng(seed_base + i * SEED_STEP)
            # Surface normal map: smooth random normals, normalized
            nx_raw = gaussian_filter(rng.standard_normal((128, 128)), sigma=8.0)
            ny_raw = gaussian_filter(rng.standard_normal((128, 128)), sigma=8.0)
            # Mostly upward normals
            nz_raw = np.abs(gaussian_filter(rng.standard_normal((128, 128)), sigma=8.0)) + 0.5
            norm = np.sqrt(nx_raw**2 + ny_raw**2 + nz_raw**2) + 1e-8
            normals = np.stack([nx_raw/norm, ny_raw/norm, nz_raw/norm], axis=-1).astype(np.float32)
            # x_true: take just the XY normal components (2-channel surface normal map)
            # Store as (128,128) — use nx as representative
            x_true = normals[..., 0].astype(np.float32)

            # Forward model: y_k = max(0, dot(n, l_k)) + noise
            y = np.zeros((4, 128, 128), dtype=np.float32)
            H_ideal = np.zeros((4, 128, 128), dtype=np.float32)
            for k in range(4):
                L = light_dirs[k]
                irr = (normals @ L).clip(0).astype(np.float32)
                H_ideal[k] = irr
                noise = rng.normal(0, 0.02, (128, 128)).astype(np.float32)
                y[k] = (irr + noise).clip(0)

            # Reconstruction: pseudo-inverse of lighting matrix
            # N_hat = L_pinv @ y  => shape (3, 128*128)
            y_flat = y.reshape(4, -1)
            n_hat_flat = L_pinv @ y_flat  # (3, 128*128)
            n_hat = n_hat_flat.reshape(3, 128, 128)
            recon = n_hat[0].astype(np.float32)  # recover nx

            grp = hf.create_group(f"sample_{i:02d}")
            grp.create_dataset("x_true", data=x_true)
            grp.create_dataset("y", data=y)
            grp.create_dataset("H_ideal", data=H_ideal)
            grp.create_dataset("reconstruction_baseline", data=recon)

            save_png(img_dir / f"sample_{i:02d}_x.png", x_true)
            save_png(img_dir / f"sample_{i:02d}_y.png", y[0])

    write_specs(tier_dir, mod, tier, n, {"n_lights": 4, "elevation_deg": 45.0})
    print(f"  [OK] {mod}/{tier}: {n} samples -> {fname.name}")


# ════════════════════════════════════════════════════════════════════════════
# 5. POLARIZATION
# ════════════════════════════════════════════════════════════════════════════
def gen_polarization(tier: str, n: int, seed_base: int):
    mod = "polarization"
    tier_dir = BENCH / mod / tier
    tier_dir.mkdir(parents=True, exist_ok=True)
    img_dir = tier_dir / "images"
    img_dir.mkdir(exist_ok=True)

    fname = tier_dir / f"{mod}_challenge_{tier}.h5"

    with h5py.File(fname, "w") as hf:
        hf.attrs.update({"tier": tier, "variant": mod, "version": "1.0",
                          "n_stokes": 4})
        for i in range(n):
            rng = np.random.default_rng(seed_base + i * SEED_STEP)
            # Degree of linear polarization (DoLP) map in [0,1]
            dolp_raw = gaussian_filter(rng.uniform(0, 1, (128, 128)), sigma=6.0)
            dolp_raw = dolp_raw / (dolp_raw.max() + 1e-8)
            x_true = dolp_raw.astype(np.float32)

            # Total intensity S0 ~ smooth positive
            S0_raw = gaussian_filter(rng.uniform(0.5, 1.5, (128, 128)), sigma=4.0)
            S0 = S0_raw.astype(np.float32)

            # Polarization angle map (AOLP) in [0, pi)
            aolp = gaussian_filter(rng.uniform(0, np.pi, (128, 128)), sigma=5.0).astype(np.float32)

            # Stokes parameters
            S1 = S0 * x_true * np.cos(2 * aolp)
            S2 = S0 * x_true * np.sin(2 * aolp)
            S3_raw = rng.normal(0, 0.05, (128, 128)).astype(np.float32)

            H_ideal = np.stack([S0, S1, S2, S3_raw], axis=0).astype(np.float32)  # (4,128,128)
            noise_level = 0.02 * S0.max()
            noise = rng.normal(0, noise_level, (4, 128, 128)).astype(np.float32)
            y = (H_ideal + noise).astype(np.float32)

            # Reconstruction: DoLP = sqrt(S1^2 + S2^2) / S0
            eps = 1e-8
            recon = (np.sqrt(y[1]**2 + y[2]**2) / (y[0] + eps)).clip(0, 1).astype(np.float32)

            grp = hf.create_group(f"sample_{i:02d}")
            grp.create_dataset("x_true", data=x_true)
            grp.create_dataset("y", data=y)
            grp.create_dataset("H_ideal", data=H_ideal)
            grp.create_dataset("reconstruction_baseline", data=recon)

            save_png(img_dir / f"sample_{i:02d}_x.png", x_true)
            save_png(img_dir / f"sample_{i:02d}_y.png", y[0])

    write_specs(tier_dir, mod, tier, n, {"n_stokes": 4})
    print(f"  [OK] {mod}/{tier}: {n} samples -> {fname.name}")


# ════════════════════════════════════════════════════════════════════════════
# 6. POLSAR
# ════════════════════════════════════════════════════════════════════════════
def gen_polsar(tier: str, n: int, seed_base: int):
    mod = "polsar"
    tier_dir = BENCH / mod / tier
    tier_dir.mkdir(parents=True, exist_ok=True)
    img_dir = tier_dir / "images"
    img_dir.mkdir(exist_ok=True)

    fname = tier_dir / f"{mod}_challenge_{tier}.h5"

    with h5py.File(fname, "w") as hf:
        hf.attrs.update({"tier": tier, "variant": mod, "version": "1.0"})
        for i in range(n):
            rng = np.random.default_rng(seed_base + i * SEED_STEP)
            # SAR coherence map: smooth [0,1]
            x_raw = gaussian_filter(rng.uniform(0.1, 1.0, (128, 128)), sigma=5.0)
            x_true = (x_raw / x_raw.max()).astype(np.float32)

            # Polarimetric channels: HH, HV, VH, VV as complex (use magnitude for simplicity)
            # HH and VV are primary backscatter channels
            HH = gaussian_filter(rng.uniform(0.3, 1.0, (128, 128)), sigma=4.0).astype(np.float32)
            HV = gaussian_filter(rng.uniform(0.0, 0.4, (128, 128)), sigma=4.0).astype(np.float32)
            VH = HV + rng.normal(0, 0.02, (128, 128)).astype(np.float32)
            VV = gaussian_filter(rng.uniform(0.2, 0.9, (128, 128)), sigma=4.0).astype(np.float32)

            H_ideal = (HH + VV).astype(np.float32)  # total power (HH+VV)
            noise = rng.normal(0, 0.05 * H_ideal.max(), (128, 128)).astype(np.float32)
            y = (H_ideal + noise).clip(0).astype(np.float32)

            # Pauli decomposition reconstruction:
            # Odd bounce (surface): |HH+VV|^2 -> ~ y
            # Even bounce (dihedral): |HH-VV|^2
            # Volume: 2|HV|^2
            # Represent reconstruction as the Pauli odd-bounce component normalized
            pauli_odd = y  # HH+VV already
            recon = (pauli_odd / (pauli_odd.max() + 1e-8)).astype(np.float32)

            grp = hf.create_group(f"sample_{i:02d}")
            grp.create_dataset("x_true", data=x_true)
            grp.create_dataset("y", data=y)
            grp.create_dataset("H_ideal", data=H_ideal)
            grp.create_dataset("reconstruction_baseline", data=recon)

            save_png(img_dir / f"sample_{i:02d}_x.png", x_true)
            save_png(img_dir / f"sample_{i:02d}_y.png", y)

    write_specs(tier_dir, mod, tier, n)
    print(f"  [OK] {mod}/{tier}: {n} samples -> {fname.name}")


# ════════════════════════════════════════════════════════════════════════════
# 7. PORTAL IMAGING
# ════════════════════════════════════════════════════════════════════════════
def gen_portal_imaging(tier: str, n: int, seed_base: int):
    mod = "portal_imaging"
    tier_dir = BENCH / mod / tier
    tier_dir.mkdir(parents=True, exist_ok=True)
    img_dir = tier_dir / "images"
    img_dir.mkdir(exist_ok=True)

    fname = tier_dir / f"{mod}_challenge_{tier}.h5"
    sigma_psf = 4.0  # MV X-ray has broad PSF
    k = gauss_kernel(17, sigma_psf)

    with h5py.File(fname, "w") as hf:
        hf.attrs.update({"tier": tier, "variant": mod, "version": "1.0",
                          "sigma_psf": sigma_psf, "energy_MV": 6.0})
        for i in range(n):
            rng = np.random.default_rng(seed_base + i * SEED_STEP)
            # Portal image: anatomy-like (bone + soft tissue structures)
            x = np.zeros((128, 128), dtype=np.float32)
            # Background soft tissue
            x[:] = rng.uniform(0.3, 0.5)
            # Bone regions (higher attenuation)
            n_bones = rng.integers(3, 8)
            for _ in range(n_bones):
                cx, cy = rng.integers(15, 113, 2)
                w, h_b = rng.integers(5, 20, 2)
                x1, x2 = max(0, cx-w), min(128, cx+w)
                y1, y2 = max(0, cy-h_b), min(128, cy+h_b)
                x[y1:y2, x1:x2] = rng.uniform(0.7, 1.0)

            H_ideal = gaussian_filter(x, sigma=sigma_psf)
            noise_std = 0.03
            noise = rng.normal(0, noise_std, (128, 128)).astype(np.float32)
            y = (H_ideal + noise).clip(0).astype(np.float32)
            recon = wiener_deconv(y, k, snr=15.0)

            grp = hf.create_group(f"sample_{i:02d}")
            grp.create_dataset("x_true", data=x)
            grp.create_dataset("y", data=y)
            grp.create_dataset("H_ideal", data=H_ideal.astype(np.float32))
            grp.create_dataset("reconstruction_baseline", data=recon)

            save_png(img_dir / f"sample_{i:02d}_x.png", x)
            save_png(img_dir / f"sample_{i:02d}_y.png", y)

    write_specs(tier_dir, mod, tier, n, {"sigma_psf": sigma_psf, "energy_MV": 6.0})
    print(f"  [OK] {mod}/{tier}: {n} samples -> {fname.name}")


# ════════════════════════════════════════════════════════════════════════════
# 8. PROTON RADIOGRAPHY
# ════════════════════════════════════════════════════════════════════════════
def gen_proton_radiography(tier: str, n: int, seed_base: int):
    mod = "proton_radiography"
    tier_dir = BENCH / mod / tier
    tier_dir.mkdir(parents=True, exist_ok=True)
    img_dir = tier_dir / "images"
    img_dir.mkdir(exist_ok=True)

    fname = tier_dir / f"{mod}_challenge_{tier}.h5"
    n_angles = 90

    with h5py.File(fname, "w") as hf:
        hf.attrs.update({"tier": tier, "variant": mod, "version": "1.0",
                          "n_projections": n_angles})
        for i in range(n):
            rng = np.random.default_rng(seed_base + i * SEED_STEP)
            # Stopping power map: smooth phantom with structures
            x = np.ones((128, 128), dtype=np.float32) * 0.3  # water-like background
            # Add structures (bone, tissue, air)
            n_structs = rng.integers(3, 8)
            for _ in range(n_structs):
                cx, cy = rng.integers(15, 113, 2)
                r = rng.integers(8, 25)
                val = rng.uniform(0.1, 1.0)
                yy, xx = np.ogrid[:128, :128]
                mask = (xx - cx)**2 + (yy - cy)**2 < r**2
                x[mask] = val
            x = gaussian_filter(x, sigma=1.5).astype(np.float32)

            # Forward: parallel-beam projections (simplified Radon)
            angles = np.linspace(0, np.pi, n_angles, endpoint=False)
            sinogram = np.zeros((n_angles, 128), dtype=np.float32)
            for j, theta in enumerate(angles):
                cos_t, sin_t = np.cos(theta), np.sin(theta)
                for det in range(128):
                    # Integrate along ray parallel to (cos_t, sin_t)
                    t = det - 63.5
                    # Ray: points (t*cos_t - s*sin_t, t*sin_t + s*cos_t) for s in [-64,64]
                    s_vals = np.linspace(-63.5, 63.5, 128)
                    px = (t * cos_t - s_vals * sin_t + 63.5).clip(0, 127)
                    py = (t * sin_t + s_vals * cos_t + 63.5).clip(0, 127)
                    # Bilinear interp (simplified: nearest)
                    sinogram[j, det] = x[py.astype(int), px.astype(int)].mean()

            H_ideal = sinogram.copy()
            noise = rng.normal(0, 0.01 * sinogram.max() + 1e-6, sinogram.shape).astype(np.float32)
            y = (sinogram + noise).clip(0).astype(np.float32)

            # FBP reconstruction (ramp filter + backproject)
            freq = np.fft.rfftfreq(128)
            ramp = np.abs(freq)
            filtered = np.zeros_like(y)
            for j in range(n_angles):
                F = np.fft.rfft(y[j])
                filtered[j] = np.fft.irfft(F * ramp, n=128)

            recon = np.zeros((128, 128), dtype=np.float32)
            for j, theta in enumerate(angles):
                cos_t, sin_t = np.cos(theta), np.sin(theta)
                for r_idx in range(128):
                    for s_idx in range(128):
                        t_coord = (r_idx - 63.5) * cos_t + (s_idx - 63.5) * sin_t + 63.5
                        det_idx = int(np.clip(t_coord, 0, 127))
                        recon[s_idx, r_idx] += filtered[j, det_idx]
            recon *= np.pi / (2 * n_angles)
            recon = recon.astype(np.float32)

            grp = hf.create_group(f"sample_{i:02d}")
            grp.create_dataset("x_true", data=x)
            grp.create_dataset("y", data=y)
            grp.create_dataset("H_ideal", data=H_ideal)
            grp.create_dataset("reconstruction_baseline", data=recon)

            save_png(img_dir / f"sample_{i:02d}_x.png", x)
            save_png(img_dir / f"sample_{i:02d}_y.png", y)

    write_specs(tier_dir, mod, tier, n, {"n_projections": n_angles})
    print(f"  [OK] {mod}/{tier}: {n} samples -> {fname.name}")


# ════════════════════════════════════════════════════════════════════════════
# 9. PROTON THERAPY IMAGING
# ════════════════════════════════════════════════════════════════════════════
def gen_proton_therapy_img(tier: str, n: int, seed_base: int):
    mod = "proton_therapy_img"
    tier_dir = BENCH / mod / tier
    tier_dir.mkdir(parents=True, exist_ok=True)
    img_dir = tier_dir / "images"
    img_dir.mkdir(exist_ok=True)

    fname = tier_dir / f"{mod}_challenge_{tier}.h5"
    sigma_psf = 2.5

    with h5py.File(fname, "w") as hf:
        hf.attrs.update({"tier": tier, "variant": mod, "version": "1.0",
                          "sigma_psf": sigma_psf})
        for i in range(n):
            rng = np.random.default_rng(seed_base + i * SEED_STEP)
            # Dose distribution: Bragg-peak-like structure
            x = np.zeros((128, 128), dtype=np.float32)
            # Multiple Bragg peaks
            n_beams = rng.integers(1, 4)
            for _ in range(n_beams):
                cx = rng.integers(30, 100)
                cy = rng.integers(20, 108)
                # Bragg peak: sharp distal fall-off along x
                for r in range(128):
                    depth = r - cx
                    if depth < -30:
                        dose = 0.1 * rng.uniform(0.8, 1.2)
                    elif depth < 0:
                        dose = (0.1 + 0.9 * (1.0 - np.abs(depth)/30)) * rng.uniform(0.9, 1.1)
                    elif depth < 5:
                        dose = 1.0 * rng.uniform(0.95, 1.05)  # peak
                    else:
                        dose = 0.05 * np.exp(-depth / 5.0)
                    dy = abs(np.arange(128) - cy)
                    lateral = np.exp(-dy**2 / (2 * 8**2))
                    x[np.arange(128), r] += dose * lateral

            x = (x / (x.max() + 1e-8)).astype(np.float32)
            x_smooth = gaussian_filter(x, sigma=sigma_psf)
            H_ideal = x_smooth.astype(np.float32)
            noise_std = 0.03
            noise = rng.normal(0, noise_std, (128, 128)).astype(np.float32)
            y = (H_ideal + noise).clip(0).astype(np.float32)

            # Reconstruction: Wiener filter
            k = gauss_kernel(11, sigma_psf)
            recon = wiener_deconv(y, k, snr=20.0)

            grp = hf.create_group(f"sample_{i:02d}")
            grp.create_dataset("x_true", data=x)
            grp.create_dataset("y", data=y)
            grp.create_dataset("H_ideal", data=H_ideal)
            grp.create_dataset("reconstruction_baseline", data=recon)

            save_png(img_dir / f"sample_{i:02d}_x.png", x)
            save_png(img_dir / f"sample_{i:02d}_y.png", y)

    write_specs(tier_dir, mod, tier, n, {"sigma_psf": sigma_psf})
    print(f"  [OK] {mod}/{tier}: {n} samples -> {fname.name}")


# ════════════════════════════════════════════════════════════════════════════
# 10. PUMP PROBE
# ════════════════════════════════════════════════════════════════════════════
def gen_pump_probe(tier: str, n: int, seed_base: int):
    mod = "pump_probe"
    tier_dir = BENCH / mod / tier
    tier_dir.mkdir(parents=True, exist_ok=True)
    img_dir = tier_dir / "images"
    img_dir.mkdir(exist_ok=True)

    fname = tier_dir / f"{mod}_challenge_{tier}.h5"
    n_times = 8

    with h5py.File(fname, "w") as hf:
        hf.attrs.update({"tier": tier, "variant": mod, "version": "1.0",
                          "n_time_points": n_times})
        for i in range(n):
            rng = np.random.default_rng(seed_base + i * SEED_STEP)
            # Time-resolved frames: spatial structure that evolves after pump
            # Base image: smooth random
            base = gaussian_filter(rng.standard_normal((128, 128)), sigma=8.0).astype(np.float32)
            base = (base - base.min()) / (base.max() - base.min() + 1e-8)

            # Excited region: a patch that changes over time
            cx, cy = rng.integers(20, 108, 2)
            r_exc = rng.integers(10, 30)
            yy, xx = np.ogrid[:128, :128]
            exc_mask = ((xx - cx)**2 + (yy - cy)**2 < r_exc**2).astype(np.float32)

            # Time decay constant
            tau = rng.uniform(2.0, 5.0)
            time_pts = np.arange(n_times, dtype=np.float32)

            x_true = np.zeros((128, 128, n_times), dtype=np.float32)
            for t in range(n_times):
                # Signal: base + excited region decaying exponentially
                decay = np.exp(-time_pts[t] / tau)
                frame = base + exc_mask * decay * rng.uniform(0.3, 0.8)
                x_true[:, :, t] = frame.clip(0, 1).astype(np.float32)

            # Measurement: shot noise (Poisson-like) scaled by photon count
            photon_scale = rng.uniform(50, 200)
            H_ideal = x_true * photon_scale
            y_counts = rng.poisson(np.maximum(H_ideal.astype(np.float64), 0)).astype(np.float32)
            y = y_counts

            # Reconstruction: normalize by photon scale estimate (median of bright pixels)
            scale_est = np.median(y[y > np.percentile(y, 90)]) if np.any(y > 0) else 1.0
            recon = (y / (scale_est + 1e-8)).clip(0, 1).astype(np.float32)

            grp = hf.create_group(f"sample_{i:02d}")
            grp.create_dataset("x_true", data=x_true)
            grp.create_dataset("y", data=y)
            grp.create_dataset("H_ideal", data=H_ideal.astype(np.float32))
            grp.create_dataset("reconstruction_baseline", data=recon)

            # Save frame 0 as PNG preview
            save_png(img_dir / f"sample_{i:02d}_x.png", x_true[:, :, 0])
            save_png(img_dir / f"sample_{i:02d}_y.png", y[:, :, 0])

    write_specs(tier_dir, mod, tier, n, {"n_time_points": n_times})
    print(f"  [OK] {mod}/{tier}: {n} samples -> {fname.name}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
GENERATORS = {
    "particle_calorimetry": gen_particle_calorimetry,
    "passive_microwave":    gen_passive_microwave,
    "phase_contrast":       gen_phase_contrast,
    "photometric_stereo":   gen_photometric_stereo,
    "polarization":         gen_polarization,
    "polsar":               gen_polsar,
    "portal_imaging":       gen_portal_imaging,
    "proton_radiography":   gen_proton_radiography,
    "proton_therapy_img":   gen_proton_therapy_img,
    "pump_probe":           gen_pump_probe,
}

if __name__ == "__main__":
    print("=" * 64)
    print("Batch 8 Dataset Generator")
    print("=" * 64)
    errors = []
    for mod_name, gen_fn in GENERATORS.items():
        print(f"\n[{mod_name}]")
        for tier, count in TIER_COUNTS.items():
            seed_b = SEED_BASE[tier]
            try:
                gen_fn(tier, count, seed_b)
            except Exception as e:
                msg = f"  [FAIL] {mod_name}/{tier}: {e}"
                print(msg)
                errors.append(msg)

    print("\n" + "=" * 64)
    if errors:
        print(f"COMPLETED WITH {len(errors)} ERRORS:")
        for e in errors:
            print(e)
    else:
        print("ALL MODALITIES COMPLETED SUCCESSFULLY")
    print("=" * 64)
