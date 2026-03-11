#!/usr/bin/env python3
"""Generate the SEM (Scanning Electron Microscopy) benchmark dataset.

Forward model (SEM imaging):
    y = Poisson(eta * (BSE_yield(Z, theta) * x + SE_yield * edge_enhancement)) *
        detector_response + noise

where:
    x             : surface/material property map (256x256)
    Z             : atomic number of material
    theta         : tilt angle
    BSE_yield     : backscattered electron yield (Z-dependent, angle-dependent)
    SE_yield      : secondary electron yield (enhanced at edges/topography)
    edge_enhancement : gradient-based topographic contrast
    detector_response : Everhart-Thornley detector geometry (cosine falloff)
    eta           : beam current / dose scaling
    noise         : Poisson counting noise + electronic readout noise

Mismatch parameters:
    beam_voltage_kV     : accelerating voltage (controls interaction volume / PSF)
    working_distance_mm : sample-to-lens distance (affects aberrations)
    detector_bias       : ET detector collection efficiency bias
    charging_artifact   : sample charging distortion (insulating samples)

Phantoms (256x256, synthetic):
    - Semiconductor features (lines, contacts, vias)
    - Fracture surfaces (rough topography)
    - Nanoparticles on substrate
    - Biological tissue cross-sections

Tiers:
    Public  : 12 samples (seed=0)
    Dev     : 20 samples (seed=10000)
    Hidden  : 20 samples (seed=20000)

CPU Baseline: Non-local means denoising + edge preservation (~22-28 dB)

Usage:
    cd datasets/benchmark/sem
    python3 generate_dataset.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy.ndimage import (
    gaussian_filter,
    sobel,
    uniform_filter,
    rotate as nd_rotate,
    zoom as nd_zoom,
)

BENCHMARK_DIR = Path(__file__).resolve().parent

# ── Geometry ──────────────────────────────────────────────────────────────────

IMAGE_SIZE = 256

# ── SEM Physics Constants ─────────────────────────────────────────────────────

# Default beam parameters
DEFAULT_BEAM_VOLTAGE_KV = 10.0      # accelerating voltage
DEFAULT_WORKING_DISTANCE_MM = 5.0   # working distance
DEFAULT_DETECTOR_BIAS = 1.0         # detector collection bias (1.0 = ideal)
DEFAULT_CHARGING_ARTIFACT = 0.0     # no charging for conductive samples
DEFAULT_BEAM_CURRENT_PA = 50.0      # beam current in pA

# Noise parameters
READOUT_SIGMA = 3.0                 # electronic readout noise std

# ── Mismatch ranges per tier ─────────────────────────────────────────────────

SPEC = {
    "public": {
        "beam_voltage_kV":      {"min": 5.0,   "max": 15.0,  "unit": "kV"},
        "working_distance_mm":  {"min": 3.0,   "max": 8.0,   "unit": "mm"},
        "detector_bias":        {"min": 0.8,   "max": 1.2,   "unit": ""},
        "charging_artifact":    {"min": 0.0,   "max": 0.05,  "unit": ""},
    },
    "dev": {
        "beam_voltage_kV":      {"min": 3.0,   "max": 20.0,  "unit": "kV"},
        "working_distance_mm":  {"min": 2.0,   "max": 12.0,  "unit": "mm"},
        "detector_bias":        {"min": 0.6,   "max": 1.4,   "unit": ""},
        "charging_artifact":    {"min": 0.0,   "max": 0.15,  "unit": ""},
    },
    "hidden": {
        "beam_voltage_kV":      {"min": 1.0,   "max": 30.0,  "unit": "kV"},
        "working_distance_mm":  {"min": 1.0,   "max": 15.0,  "unit": "mm"},
        "detector_bias":        {"min": 0.4,   "max": 1.6,   "unit": ""},
        "charging_artifact":    {"min": 0.0,   "max": 0.30,  "unit": ""},
    },
}


# ── BSE Yield Physics ────────────────────────────────────────────────────────

def bse_yield(Z: float, theta_deg: float = 0.0) -> float:
    """Backscattered electron yield as function of atomic number and tilt.

    Empirical fit from Heinrich (1966) / Reuter (1972):
        eta_BSE ~ -0.0254 + 0.016*Z - 1.86e-4*Z^2 + 8.3e-7*Z^3

    Tilt correction (Arnal et al.):
        eta(theta) = eta(0) / cos(theta)^q,  q ~ 0.9 for most materials

    Args:
        Z: atomic number (14=Si, 29=Cu, 79=Au, etc.)
        theta_deg: sample tilt angle in degrees

    Returns:
        BSE yield coefficient in [0, 1]
    """
    # Heinrich empirical polynomial (valid for Z ~ 6-92)
    eta0 = -0.0254 + 0.016 * Z - 1.86e-4 * Z**2 + 8.3e-7 * Z**3
    eta0 = np.clip(eta0, 0.01, 0.60)

    # Tilt correction
    theta_rad = np.deg2rad(theta_deg)
    cos_theta = max(np.cos(theta_rad), 0.1)
    q = 0.9
    eta = eta0 / cos_theta**q

    return float(np.clip(eta, 0.01, 0.80))


def se_yield(beam_voltage_kV: float, Z: float = 14.0) -> float:
    """Secondary electron yield.

    SE yield peaks at low beam energies (~1 kV) and decreases at higher
    voltages due to deeper interaction volume. Typical values 0.1-2.0.

    Simplified Sternglass (1957) model:
        delta_SE ~ 1.28 * E_max / E_0 * (1 - exp(-1.5 * E_0 / E_max))

    where E_max depends on Z (roughly ~ 0.4 + 0.005*Z kV).
    """
    E_max = 0.4 + 0.005 * Z  # energy at peak SE yield (kV)
    ratio = beam_voltage_kV / E_max if E_max > 0 else 100.0
    delta = 1.28 * (1.0 / ratio) * (1.0 - np.exp(-1.5 * ratio))
    return float(np.clip(delta, 0.05, 2.5))


# ── Interaction Volume PSF ───────────────────────────────────────────────────

def interaction_volume_sigma(beam_voltage_kV: float, working_distance_mm: float,
                              Z: float = 14.0) -> float:
    """Effective PSF sigma (probe size + aberrations) in pixels.

    SEM resolution is determined by the electron probe diameter, not the
    full interaction volume (Kanaya-Okayama range gives penetration depth,
    not lateral resolution). The probe size depends on:

    1. Electron source brightness (thermionic, Schottky, cold FEG)
    2. Beam voltage (higher V -> smaller wavelength -> better diffraction)
    3. Working distance (larger WD -> more aberrations -> worse resolution)
    4. Aperture optimization (spherical + chromatic aberration trade-off)

    Typical SEM probe sizes:
        FEG at 1 kV:  ~2 nm
        FEG at 5 kV:  ~1 nm
        FEG at 15 kV: ~0.8 nm
        FEG at 30 kV: ~0.5 nm
        W-filament:   ~5-50 nm (much worse)

    At 5 nm/pixel, probe sizes are 0.1-1.0 pixels for FEG sources.
    We model a Schottky FEG source (intermediate performance).

    Returns sigma in pixels.
    """
    pixel_size_nm = 5.0

    # Probe diameter (nm): decreases with voltage (smaller diffraction disk)
    # d_probe ~ 2 + 15/E_0 (nm) for Schottky FEG (simplified)
    d_probe_nm = 1.5 + 10.0 / max(beam_voltage_kV, 0.5)
    sigma_probe = d_probe_nm / (2.35 * pixel_size_nm)  # FWHM -> sigma

    # SE escape depth contribution (increases with voltage but saturates)
    # SE come from top ~5 nm; BSE from deeper -> larger effective PSF at higher V
    se_depth_nm = 2.0 + 0.3 * beam_voltage_kV  # nm, simplified
    sigma_depth = se_depth_nm / (3.0 * pixel_size_nm)

    # Aberration from working distance
    # Spherical aberration ~ Cs * alpha^3, chromatic ~ Cc * alpha * dE/E
    # At WD=5mm, aberration adds ~0.5 nm; scales roughly as WD^1.5
    aberration_nm = 0.5 * (working_distance_mm / 5.0)**1.5
    sigma_aberration = aberration_nm / (2.35 * pixel_size_nm)

    sigma_total = np.sqrt(sigma_probe**2 + sigma_depth**2 + sigma_aberration**2)
    return float(np.clip(sigma_total, 0.3, 4.0))


# ── Phantom Generators ───────────────────────────────────────────────────────

def make_semiconductor_phantom(H: int, W: int, seed: int,
                                variant: int = 0) -> tuple[np.ndarray, dict]:
    """Semiconductor surface: lines, contacts, vias, trenches.

    Returns:
        x_true: (H, W) float64 in [0, 1] - surface height/material map
        info: dict with scene metadata
    """
    rng = np.random.default_rng(seed)
    x = np.zeros((H, W), dtype=np.float64)

    # Substrate background (silicon, moderate Z)
    x[:] = 0.3 + rng.uniform(-0.02, 0.02)

    # Metal lines (horizontal and vertical)
    n_lines = rng.integers(5, 12)
    line_width = rng.integers(3, 10, size=n_lines)
    for i in range(n_lines):
        if rng.random() < 0.5:
            # Horizontal line
            y0 = rng.integers(10, H - 10)
            lw = line_width[i]
            x[max(0, y0 - lw // 2):min(H, y0 + lw // 2 + 1), :] = (
                0.7 + rng.uniform(-0.05, 0.05)
            )
        else:
            # Vertical line
            x0 = rng.integers(10, W - 10)
            lw = line_width[i]
            x[:, max(0, x0 - lw // 2):min(W, x0 + lw // 2 + 1)] = (
                0.7 + rng.uniform(-0.05, 0.05)
            )

    # Contact pads (rectangular regions with higher Z material)
    n_pads = rng.integers(3, 8)
    for _ in range(n_pads):
        pad_h = rng.integers(10, 30)
        pad_w = rng.integers(10, 30)
        y0 = rng.integers(5, H - pad_h - 5)
        x0 = rng.integers(5, W - pad_w - 5)
        x[y0:y0 + pad_h, x0:x0 + pad_w] = 0.85 + rng.uniform(-0.05, 0.05)

    # Via holes (circular dark spots)
    n_vias = rng.integers(5, 20)
    for _ in range(n_vias):
        cy = rng.integers(15, H - 15)
        cx = rng.integers(15, W - 15)
        r = rng.integers(2, 6)
        yy, xx = np.ogrid[:H, :W]
        mask = ((yy - cy)**2 + (xx - cx)**2) <= r**2
        x[mask] = 0.1 + rng.uniform(-0.02, 0.02)

    # Trench structures
    n_trenches = rng.integers(1, 4)
    for _ in range(n_trenches):
        y0 = rng.integers(20, H - 40)
        x0 = rng.integers(20, W - 40)
        tw = rng.integers(2, 5)
        tl = rng.integers(30, 80)
        if rng.random() < 0.5:
            x[y0:y0 + tw, x0:x0 + tl] = 0.05 + rng.uniform(0, 0.05)
        else:
            x[y0:y0 + tl, x0:x0 + tw] = 0.05 + rng.uniform(0, 0.05)

    # Add sub-pixel texture noise (grain boundaries, roughness)
    texture = rng.standard_normal((H, W)) * 0.02
    texture = gaussian_filter(texture, sigma=1.5)
    x += texture

    # Apply variant-specific rotation for diversity
    if variant > 0:
        angle = variant * 17 % 360
        x = nd_rotate(x, angle, reshape=False, mode='constant', cval=0.3)

    x = np.clip(x, 0.0, 1.0)
    return x, {"type": "semiconductor", "Z_primary": 14, "Z_metal": 29,
               "name": f"semiconductor_{variant:02d}"}


def make_fracture_phantom(H: int, W: int, seed: int,
                           variant: int = 0) -> tuple[np.ndarray, dict]:
    """Fracture surface: rough topography with multi-scale features.

    Returns:
        x_true: (H, W) float64 in [0, 1] - topographic height map
        info: dict with scene metadata
    """
    rng = np.random.default_rng(seed)

    # Multi-scale fracture surface via fractional Brownian motion approximation
    x = np.zeros((H, W), dtype=np.float64)

    # Large-scale undulation
    for scale in [64, 32, 16, 8, 4]:
        small = rng.standard_normal((H // scale + 2, W // scale + 2))
        upsampled = nd_zoom(small, (H / (H // scale + 2), W / (W // scale + 2)),
                            order=1)[:H, :W]
        weight = 1.0 / (scale**0.5)  # Rougher fracture has larger weights
        x += upsampled * weight * 0.15

    # Sharp crack features
    n_cracks = rng.integers(2, 6)
    for _ in range(n_cracks):
        # Random walk crack path
        cy = rng.integers(20, H - 20)
        cx = rng.integers(5, W // 3)
        crack_len = rng.integers(W // 3, W)
        crack_width = rng.integers(1, 4)
        for step in range(crack_len):
            cy += int(rng.integers(-2, 3))
            cy = np.clip(cy, 1, H - 2)
            cx_end = min(cx + 1, W)
            if cx_end > 0 and cx_end <= W:
                x[max(0, cy - crack_width):min(H, cy + crack_width + 1),
                  cx] = -0.3
            cx += 1
            if cx >= W:
                break

    # Cleavage planes (flat facets at different angles)
    n_facets = rng.integers(3, 7)
    for _ in range(n_facets):
        y0 = rng.integers(0, H - 30)
        x0 = rng.integers(0, W - 30)
        fh = rng.integers(20, 60)
        fw = rng.integers(20, 60)
        slope_y = rng.uniform(-0.005, 0.005)
        slope_x = rng.uniform(-0.005, 0.005)
        offset = rng.uniform(-0.1, 0.1)
        yy = np.arange(fh)[:, None] * slope_y
        xx = np.arange(fw)[None, :] * slope_x
        facet = offset + yy + xx
        y1 = min(y0 + fh, H)
        x1 = min(x0 + fw, W)
        x[y0:y1, x0:x1] = facet[:y1 - y0, :x1 - x0]

    # Normalize to [0, 1]
    x -= x.min()
    if x.max() > 0:
        x /= x.max()

    # Add fine grain texture
    grain = rng.standard_normal((H, W)) * 0.03
    grain = gaussian_filter(grain, sigma=0.8)
    x += grain
    x = np.clip(x, 0.0, 1.0)

    if variant > 0:
        angle = variant * 23 % 360
        x = nd_rotate(x, angle, reshape=False, mode='reflect')
        x = np.clip(x, 0.0, 1.0)

    return x, {"type": "fracture", "Z_primary": 26, "name": f"fracture_{variant:02d}"}


def make_nanoparticle_phantom(H: int, W: int, seed: int,
                               variant: int = 0) -> tuple[np.ndarray, dict]:
    """Nanoparticles on flat substrate: high-Z particles on low-Z background.

    Returns:
        x_true: (H, W) float64 in [0, 1]
        info: dict
    """
    rng = np.random.default_rng(seed)

    # Flat substrate (e.g., silicon or carbon)
    substrate_level = 0.15 + rng.uniform(-0.02, 0.02)
    x = np.full((H, W), substrate_level, dtype=np.float64)

    # Add subtle substrate texture
    texture = rng.standard_normal((H, W)) * 0.01
    texture = gaussian_filter(texture, sigma=2.0)
    x += texture

    # Nanoparticles of various sizes and compositions
    n_particles = rng.integers(20, 80)
    particle_Z = rng.choice([47, 79, 78, 29, 82], size=n_particles)  # Ag, Au, Pt, Cu, Pb

    for i in range(n_particles):
        cy = rng.integers(5, H - 5)
        cx = rng.integers(5, W - 5)
        r = rng.uniform(1.5, 8.0)

        # Particle brightness depends on Z (higher Z = brighter in BSE)
        Z = particle_Z[i]
        brightness = 0.4 + 0.6 * bse_yield(Z) / bse_yield(79)

        # Gaussian profile (spherical particle projection)
        yy, xx = np.ogrid[:H, :W]
        dist2 = (yy - cy)**2.0 + (xx - cx)**2.0
        particle = brightness * np.exp(-dist2 / (2.0 * r**2))
        x = np.maximum(x, particle)

    # Occasional particle clusters
    n_clusters = rng.integers(1, 4)
    for _ in range(n_clusters):
        cluster_cy = rng.integers(30, H - 30)
        cluster_cx = rng.integers(30, W - 30)
        cluster_n = rng.integers(5, 15)
        for _ in range(cluster_n):
            cy = cluster_cy + int(rng.normal(0, 8))
            cx = cluster_cx + int(rng.normal(0, 8))
            cy = np.clip(cy, 2, H - 2)
            cx = np.clip(cx, 2, W - 2)
            r = rng.uniform(1.0, 4.0)
            Z = rng.choice([47, 79])
            brightness = 0.4 + 0.6 * bse_yield(Z) / bse_yield(79)
            yy, xx = np.ogrid[:H, :W]
            dist2 = (yy - cy)**2.0 + (xx - cx)**2.0
            particle = brightness * np.exp(-dist2 / (2.0 * r**2))
            x = np.maximum(x, particle)

    x = np.clip(x, 0.0, 1.0)

    if variant > 0:
        if rng.random() < 0.5:
            x = np.fliplr(x)
        if rng.random() < 0.5:
            x = np.flipud(x)

    return x, {"type": "nanoparticles", "Z_primary": 14, "Z_particles": "mixed",
               "name": f"nanoparticles_{variant:02d}"}


def make_biological_phantom(H: int, W: int, seed: int,
                              variant: int = 0) -> tuple[np.ndarray, dict]:
    """Biological tissue cross-section: cell membranes, organelles, fibers.

    Returns:
        x_true: (H, W) float64 in [0, 1]
        info: dict
    """
    rng = np.random.default_rng(seed)

    # Background tissue matrix (low contrast, moderate density)
    x = np.full((H, W), 0.35, dtype=np.float64)

    # Large-scale tissue structure (fibrous matrix)
    for scale in [48, 24, 12]:
        small = rng.standard_normal((H // scale + 2, W // scale + 2))
        upsampled = nd_zoom(small, (H / (H // scale + 2), W / (W // scale + 2)),
                            order=1)[:H, :W]
        x += upsampled * 0.04

    # Cell bodies (elliptical regions)
    n_cells = rng.integers(8, 25)
    for _ in range(n_cells):
        cy = rng.integers(15, H - 15)
        cx = rng.integers(15, W - 15)
        ry = rng.uniform(8, 25)
        rx = rng.uniform(8, 25)
        angle = rng.uniform(0, 360)

        yy, xx = np.ogrid[:H, :W]
        ca, sa = np.cos(np.radians(angle)), np.sin(np.radians(angle))
        yr = (yy - cy) * ca - (xx - cx) * sa
        xr = (yy - cy) * sa + (xx - cx) * ca
        cell_mask = (xr / rx)**2 + (yr / ry)**2 <= 1.0

        # Cell cytoplasm
        cell_val = rng.uniform(0.25, 0.45)
        x[cell_mask] = cell_val

        # Nucleus (darker, denser)
        nr = min(ry, rx) * rng.uniform(0.3, 0.5)
        nuc_mask = (xr / nr)**2 + (yr / nr)**2 <= 1.0
        x[nuc_mask] = cell_val + rng.uniform(0.1, 0.2)

        # Cell membrane (bright edge from heavy metal staining)
        membrane_outer = (xr / (rx + 1))**2 + (yr / (ry + 1))**2 <= 1.0
        membrane = membrane_outer & ~cell_mask
        x[membrane] = rng.uniform(0.6, 0.8)

    # Extracellular matrix fibers
    n_fibers = rng.integers(10, 30)
    for _ in range(n_fibers):
        y0 = rng.integers(0, H)
        x0 = rng.integers(0, W)
        angle = rng.uniform(0, 180)
        length = rng.integers(20, 80)
        ca, sa = np.cos(np.radians(angle)), np.sin(np.radians(angle))
        for t in range(length):
            py = int(y0 + t * sa)
            px = int(x0 + t * ca)
            if 0 <= py < H and 0 <= px < W:
                x[py, px] = rng.uniform(0.5, 0.65)

    # Organelles (mitochondria-like elongated bright structures)
    n_organelles = rng.integers(10, 40)
    for _ in range(n_organelles):
        cy = rng.integers(10, H - 10)
        cx = rng.integers(10, W - 10)
        ry = rng.uniform(1, 4)
        rx = rng.uniform(3, 10)
        angle = rng.uniform(0, 360)
        yy, xx = np.ogrid[:H, :W]
        ca, sa = np.cos(np.radians(angle)), np.sin(np.radians(angle))
        yr = (yy - cy) * ca - (xx - cx) * sa
        xr = (yy - cy) * sa + (xx - cx) * ca
        org_mask = (xr / rx)**2 + (yr / ry)**2 <= 1.0
        x[org_mask] = rng.uniform(0.55, 0.75)

    # Fine texture (sub-cellular detail)
    fine = rng.standard_normal((H, W)) * 0.02
    fine = gaussian_filter(fine, sigma=1.0)
    x += fine

    x = np.clip(x, 0.0, 1.0)

    if variant > 0:
        angle = variant * 31 % 360
        x = nd_rotate(x, angle, reshape=False, mode='reflect')
        x = np.clip(x, 0.0, 1.0)

    return x, {"type": "biological", "Z_primary": 8,
               "name": f"biological_{variant:02d}"}


# ── Phantom pool ─────────────────────────────────────────────────────────────

PHANTOM_GENERATORS = [
    make_semiconductor_phantom,
    make_fracture_phantom,
    make_nanoparticle_phantom,
    make_biological_phantom,
]


def generate_phantoms_public(n: int = 12) -> list[tuple[np.ndarray, dict]]:
    """Generate 12 public-tier phantoms: 3 semiconductor + 3 fracture +
    3 nanoparticle + 3 biological."""
    phantoms = []
    for i in range(3):
        phantoms.append(make_semiconductor_phantom(IMAGE_SIZE, IMAGE_SIZE,
                                                    seed=100 + i, variant=i))
    for i in range(3):
        phantoms.append(make_fracture_phantom(IMAGE_SIZE, IMAGE_SIZE,
                                               seed=200 + i, variant=i))
    for i in range(3):
        phantoms.append(make_nanoparticle_phantom(IMAGE_SIZE, IMAGE_SIZE,
                                                    seed=300 + i, variant=i))
    for i in range(3):
        phantoms.append(make_biological_phantom(IMAGE_SIZE, IMAGE_SIZE,
                                                 seed=400 + i, variant=i))
    return phantoms[:n]


def generate_phantoms_dev(n: int = 20) -> list[tuple[np.ndarray, dict]]:
    """Generate 20 dev-tier phantoms with augmented diversity."""
    phantoms = []
    rng = np.random.default_rng(5000)
    for i in range(n):
        gen_fn = PHANTOM_GENERATORS[i % 4]
        x, info = gen_fn(IMAGE_SIZE, IMAGE_SIZE, seed=10500 + i, variant=i + 5)
        # Augment: rotation + flip + mild zoom
        angle = float(rng.uniform(15, 345))
        x = nd_rotate(x, angle, reshape=False, mode='constant', cval=0.2)
        if rng.random() < 0.5:
            x = np.fliplr(x)
        if rng.random() < 0.5:
            x = np.flipud(x)
        zoom_f = float(rng.uniform(0.85, 1.15))
        if abs(zoom_f - 1.0) > 0.01:
            x = _zoom_crop(x, zoom_f, IMAGE_SIZE)
        x = np.clip(x, 0.0, 1.0)
        info["name"] = f"dev_{info['name']}"
        phantoms.append((x, info))
    return phantoms


def generate_phantoms_hidden(n: int = 20) -> list[tuple[np.ndarray, dict]]:
    """Generate 20 hidden-tier phantoms with adversarial modifications."""
    phantoms = []
    rng = np.random.default_rng(8000)
    for i in range(n):
        gen_fn = PHANTOM_GENERATORS[i % 4]
        x, info = gen_fn(IMAGE_SIZE, IMAGE_SIZE, seed=20800 + i, variant=i + 12)

        # Aggressive augmentation
        angle = float(rng.uniform(20, 340))
        x = nd_rotate(x, angle, reshape=False, mode='constant', cval=0.2)
        if rng.random() < 0.7:
            x = np.fliplr(x)
        if rng.random() < 0.5:
            x = np.flipud(x)
        zoom_f = float(rng.uniform(0.70, 1.30))
        x = _zoom_crop(x, zoom_f, IMAGE_SIZE)

        # Adversarial: add subtle contamination spots
        n_contam = rng.integers(3, 10)
        for _ in range(n_contam):
            cy = rng.integers(5, IMAGE_SIZE - 5)
            cx = rng.integers(5, IMAGE_SIZE - 5)
            r = rng.uniform(1, 5)
            yy, xx = np.ogrid[:IMAGE_SIZE, :IMAGE_SIZE]
            dist2 = (yy - cy)**2.0 + (xx - cx)**2.0
            contam = rng.uniform(0.6, 0.95) * np.exp(-dist2 / (2.0 * r**2))
            x = np.maximum(x, contam)

        # Add scanning artifact stripes
        if rng.random() < 0.3:
            n_stripes = rng.integers(2, 6)
            for _ in range(n_stripes):
                row = rng.integers(0, IMAGE_SIZE)
                x[row, :] += rng.uniform(0.02, 0.08)

        x = np.clip(x, 0.0, 1.0)
        info["name"] = f"hidden_{info['name']}"
        phantoms.append((x, info))
    return phantoms


def _zoom_crop(arr: np.ndarray, zoom_f: float, size: int) -> np.ndarray:
    """Zoom and crop/pad to target size."""
    zoomed = nd_zoom(arr, zoom_f, order=1)
    H, W = zoomed.shape
    if H >= size and W >= size:
        y0 = (H - size) // 2
        x0 = (W - size) // 2
        return zoomed[y0:y0 + size, x0:x0 + size]
    else:
        out = np.full((size, size), 0.2, dtype=arr.dtype)
        y0 = max(0, (size - H) // 2)
        x0 = max(0, (size - W) // 2)
        out[y0:y0 + min(H, size), x0:x0 + min(W, size)] = zoomed[:min(H, size), :min(W, size)]
        return out


# ── SEM Forward Model ────────────────────────────────────────────────────────

def compute_edge_enhancement(x: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Compute edge enhancement map simulating SE topographic contrast.

    Secondary electrons are preferentially emitted from edges and
    steep topography, producing bright edge contrast in SEM images.

    Args:
        x: (H, W) surface property map
        strength: edge enhancement multiplier

    Returns:
        edge_map: (H, W) float64, edge enhancement contribution
    """
    # Sobel gradient magnitude
    sx = sobel(x, axis=1, mode='reflect')
    sy = sobel(x, axis=0, mode='reflect')
    gradient_mag = np.sqrt(sx**2 + sy**2)

    # Normalize and scale
    if gradient_mag.max() > 0:
        gradient_mag /= gradient_mag.max()

    return gradient_mag * strength


def everhart_thornley_response(H: int, W: int,
                                detector_pos: str = "right",
                                bias: float = 1.0) -> np.ndarray:
    """Simulate Everhart-Thornley detector geometry response.

    The ET detector has a positional bias: regions facing the detector
    appear brighter due to geometric collection efficiency (cosine law).

    Args:
        H, W: image dimensions
        detector_pos: detector position ("right", "left", "top", "bottom")
        bias: collection efficiency bias factor

    Returns:
        response: (H, W) float64, detector response map [0.5, 1.5]
    """
    yy = np.linspace(-1, 1, H)[:, None]
    xx = np.linspace(-1, 1, W)[None, :]

    if detector_pos == "right":
        angle_factor = xx  # right side brighter
    elif detector_pos == "left":
        angle_factor = -xx
    elif detector_pos == "top":
        angle_factor = -yy
    else:
        angle_factor = yy

    # Cosine-law modulation: subtle brightness gradient across image
    # Bias controls magnitude: 1.0 = standard, >1 = stronger shadowing
    # Reduced to 0.05 from 0.15 so detector shading is a small perturbation
    response = 1.0 + 0.05 * bias * angle_factor
    return response


def apply_charging_artifact(image: np.ndarray, strength: float,
                             rng: np.random.Generator) -> np.ndarray:
    """Simulate sample charging artifact for insulating samples.

    Charging causes:
    1. Local brightness shifts (accumulated charge deflects SE)
    2. Image drift/distortion
    3. Bright/dark banding

    Args:
        image: (H, W) input image
        strength: charging strength [0, 1]
        rng: random generator

    Returns:
        distorted: (H, W) with charging artifacts
    """
    if strength < 1e-4:
        return image.copy()

    H, W = image.shape
    distorted = image.copy()

    # 1. Low-frequency brightness shift (charge accumulation pattern)
    charge_pattern = np.zeros((H, W), dtype=np.float64)
    n_centers = rng.integers(2, 6)
    for _ in range(n_centers):
        cy = rng.integers(0, H)
        cx = rng.integers(0, W)
        sigma = rng.uniform(30, 80)
        yy, xx = np.ogrid[:H, :W]
        charge_pattern += np.exp(-((yy - cy)**2 + (xx - cx)**2) / (2 * sigma**2))

    if charge_pattern.max() > 0:
        charge_pattern /= charge_pattern.max()
    distorted += strength * 0.3 * charge_pattern

    # 2. Horizontal banding (scan-line charging)
    n_bands = rng.integers(3, 10)
    for _ in range(n_bands):
        row = rng.integers(0, H)
        band_width = rng.integers(2, 8)
        band_shift = rng.uniform(-0.1, 0.1) * strength
        y0 = max(0, row - band_width // 2)
        y1 = min(H, row + band_width // 2 + 1)
        distorted[y0:y1, :] += band_shift

    # 3. Subtle pixel displacement (image drift due to beam deflection)
    if strength > 0.05:
        shift_y = int(round(strength * rng.uniform(-3, 3)))
        shift_x = int(round(strength * rng.uniform(-2, 2)))
        distorted = np.roll(np.roll(distorted, shift_y, axis=0), shift_x, axis=1)

    return distorted


def sem_forward_model(
    x_true: np.ndarray,
    beam_voltage_kV: float,
    working_distance_mm: float,
    detector_bias: float,
    charging_artifact: float,
    rng: np.random.Generator,
    beam_current_pA: float = DEFAULT_BEAM_CURRENT_PA,
) -> dict:
    """Apply the full SEM forward model.

    y = Poisson(eta * (BSE_yield(Z,theta) * PSF(x) + SE_yield * edge_enh)) *
        detector_response + readout_noise

    The measurement y is designed to be a degraded (blurred + noisy + edge-
    enhanced) version of x_true, maintaining structural similarity so that
    denoising/deblurring can achieve ~22-28 dB PSNR.

    Args:
        x_true: (H, W) ground truth surface/material map [0, 1]
        beam_voltage_kV: accelerating voltage
        working_distance_mm: working distance
        detector_bias: ET detector collection bias
        charging_artifact: charging strength
        rng: random generator
        beam_current_pA: probe current

    Returns:
        dict with y (measured), H_ideal (ideal system response), etc.
    """
    H, W = x_true.shape

    # Material atomic number (estimate from x_true intensity)
    Z_avg = 14.0 + 20.0 * x_true.mean()

    # 1. PSF blur (interaction volume + aberrations)
    sigma = interaction_volume_sigma(beam_voltage_kV, working_distance_mm, Z_avg)
    x_blurred = gaussian_filter(x_true, sigma=sigma)

    # 2. BSE signal (Z-contrast, proportional to x_true)
    eta_bse = bse_yield(Z_avg, theta_deg=0.0)

    # 3. SE signal (topographic contrast from edges)
    delta_se = se_yield(beam_voltage_kV, Z_avg)
    edge_map = compute_edge_enhancement(x_true, strength=1.0)
    edge_map_blurred = gaussian_filter(edge_map, sigma=max(sigma * 0.5, 0.5))

    # 4. Combined signal: primarily BSE-proportional to x + small SE edge term
    # Keep BSE as dominant so y correlates strongly with x_true
    se_weight = 0.05 * delta_se  # SE contributes ~5% edge enhancement
    combined = x_blurred + se_weight * edge_map_blurred

    # 5. Detector response (Everhart-Thornley geometry)
    # Reduced shading amplitude so it doesn't dominate the reconstruction error
    det_pos = rng.choice(["right", "left", "top", "bottom"])
    det_response = everhart_thornley_response(H, W, det_pos, detector_bias)
    signal_detected = combined * det_response

    # Normalize signal to [0, 1] range for Poisson scaling
    sig_max = signal_detected.max()
    if sig_max > 0:
        signal_norm = signal_detected / sig_max
    else:
        signal_norm = signal_detected

    # 6. Poisson noise with controlled SNR
    # Higher counts = cleaner image = higher baseline PSNR
    dose_scale = beam_current_pA / 10.0
    # Target: mean count ~2000-5000 for PSNR ~ 22-30 dB regime
    mean_count = 500.0 * dose_scale  # ~2500 counts at 50 pA

    signal_scaled = np.maximum(signal_norm * mean_count, 0.01)

    # 7. Poisson noise (shot noise)
    y_poisson = rng.poisson(signal_scaled).astype(np.float64)

    # 8. Electronic readout noise (small relative to signal)
    readout_level = READOUT_SIGMA
    y_noisy = y_poisson + rng.normal(0.0, readout_level, (H, W))

    # 9. Charging artifacts
    if charging_artifact > 0:
        # Scale charging effect relative to signal level
        y_noisy = apply_charging_artifact(y_noisy, charging_artifact, rng)

    # Clamp to non-negative
    y = np.maximum(y_noisy, 0.0)

    # Normalize y to [0, 1]
    y_max = y.max()
    if y_max > 0:
        y_norm = (y / y_max).astype(np.float32)
    else:
        y_norm = y.astype(np.float32)

    # H_ideal: the ideal (noiseless) forward model output, normalized
    H_ideal = signal_norm.astype(np.float32)

    return {
        "y": y_norm,
        "y_raw": y.astype(np.float32),
        "H_ideal": H_ideal,
        "psf_sigma": sigma,
        "eta_bse": eta_bse,
        "delta_se": delta_se,
        "dose_scale": dose_scale,
        "mean_count": mean_count,
        "y_max": float(y_max),
        "detector_pos": det_pos,
    }


# ── CPU Baseline: Non-local Means + Edge Preservation ────────────────────────

def nlm_denoise(image: np.ndarray, h: float = 0.08, patch_size: int = 5,
                search_size: int = 11) -> np.ndarray:
    """Non-local means denoising (simplified, CPU-only).

    Simplified NLM: for each pixel, compute weighted average over a
    search window, where weights depend on patch similarity.

    Args:
        image: (H, W) float input
        h: filtering parameter (larger = more smoothing)
        patch_size: patch radius
        search_size: search window radius

    Returns:
        denoised: (H, W) float
    """
    H, W = image.shape
    pad = search_size // 2 + patch_size // 2
    img_pad = np.pad(image, pad, mode='reflect')
    result = np.zeros_like(image)
    h2 = h**2
    ps = patch_size // 2

    # Downsample for speed: process every 2nd pixel, interpolate
    step = 2
    for iy in range(0, H, step):
        for ix in range(0, W, step):
            # Center patch
            cy, cx = iy + pad, ix + pad
            center_patch = img_pad[cy - ps:cy + ps + 1, cx - ps:cx + ps + 1]

            # Search window
            sy0 = max(cy - search_size // 2, ps)
            sy1 = min(cy + search_size // 2 + 1, img_pad.shape[0] - ps)
            sx0 = max(cx - search_size // 2, ps)
            sx1 = min(cx + search_size // 2 + 1, img_pad.shape[1] - ps)

            total_weight = 0.0
            weighted_sum = 0.0

            for jy in range(sy0, sy1, step):
                for jx in range(sx0, sx1, step):
                    neighbor_patch = img_pad[jy - ps:jy + ps + 1,
                                              jx - ps:jx + ps + 1]
                    dist = np.sum((center_patch - neighbor_patch)**2)
                    dist /= (patch_size**2)
                    w = np.exp(-dist / h2)
                    total_weight += w
                    weighted_sum += w * img_pad[jy, jx]

            if total_weight > 0:
                val = weighted_sum / total_weight
            else:
                val = image[iy, ix]

            # Fill step x step block
            y1 = min(iy + step, H)
            x1 = min(ix + step, W)
            result[iy:y1, ix:x1] = val

    return result


def wiener_deconvolution(image: np.ndarray, sigma: float,
                          noise_power: float = 0.005) -> np.ndarray:
    """Wiener deconvolution assuming Gaussian PSF.

    Args:
        image: (H, W) blurred noisy image
        sigma: PSF sigma in pixels
        noise_power: estimated noise-to-signal power ratio

    Returns:
        deconvolved: (H, W)
    """
    H, W = image.shape
    # Build Gaussian PSF in frequency domain
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.fftfreq(W)[None, :]
    # OTF of Gaussian PSF: exp(-2*pi^2*sigma^2*(fx^2+fy^2))
    otf = np.exp(-2.0 * np.pi**2 * sigma**2 * (fx**2 + fy**2))

    # Wiener filter: H* / (|H|^2 + NSR)
    img_fft = np.fft.fft2(image)
    wiener_filter = np.conj(otf) / (np.abs(otf)**2 + noise_power)
    result = np.real(np.fft.ifft2(img_fft * wiener_filter))

    return result


def baseline_reconstruct(y: np.ndarray, H_ideal: np.ndarray,
                          psf_sigma: float = 1.5) -> np.ndarray:
    """CPU baseline: Gaussian denoising + mild Wiener deconvolution +
    edge-preserving refinement.

    The measurement y is a noisy, blurred, edge-enhanced version of x_true.
    This baseline applies:
    1. Gaussian denoising (primary noise reduction)
    2. Mild Wiener deconvolution to partially undo PSF blur
    3. Edge-adaptive smoothing for noise vs edge trade-off

    Expected PSNR: ~22-28 dB.

    Args:
        y: (H, W) measured SEM image (normalized to [0, 1])
        H_ideal: (H, W) ideal system response (normalized)
        psf_sigma: PSF sigma in pixels (from forward model)

    Returns:
        recon: (H, W) reconstructed surface map in [0, 1]
    """
    img = y.astype(np.float64)

    # Step 1: Gaussian denoising (primary)
    # Sigma chosen to balance noise removal vs detail preservation
    denoise_sigma = min(max(psf_sigma * 0.3, 0.5), 2.0)
    denoised = gaussian_filter(img, sigma=denoise_sigma)

    # Step 2: Mild Wiener deconvolution
    # Use conservative parameters to avoid noise amplification
    deconv_sigma = max(psf_sigma * 0.5, 0.3)
    nsr = 0.02 + 0.01 * psf_sigma  # generous regularization
    deconvolved = wiener_deconvolution(denoised, deconv_sigma, nsr)
    deconvolved = np.clip(deconvolved, 0.0, None)
    if deconvolved.max() > 0:
        deconvolved = deconvolved / deconvolved.max()

    # Step 3: Edge-preserving adaptive smoothing
    # Detect edges from local variance
    local_mean = uniform_filter(deconvolved, size=5)
    local_sq_mean = uniform_filter(deconvolved**2, size=5)
    local_var = np.maximum(local_sq_mean - local_mean**2, 0.0)
    edge_strength = np.sqrt(local_var)
    if edge_strength.max() > 0:
        edge_strength = edge_strength / edge_strength.max()

    # Blend: keep sharp details at edges, smooth flat regions
    smooth_light = gaussian_filter(deconvolved, sigma=0.6)
    smooth_heavy = gaussian_filter(deconvolved, sigma=1.5)
    w_edge = np.clip(edge_strength * 2.0, 0.0, 1.0)
    refined = w_edge * smooth_light + (1.0 - w_edge) * smooth_heavy

    # Final clip to [0, 1]
    refined = np.clip(refined, 0.0, 1.0).astype(np.float32)

    return refined


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_psnr(gt: np.ndarray, recon: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio."""
    gt64 = gt.astype(np.float64)
    recon64 = recon.astype(np.float64)
    mse = np.mean((gt64 - recon64)**2)
    if mse < 1e-12:
        return 100.0
    data_range = float(gt64.max() - gt64.min())
    if data_range < 1e-12:
        return 0.0
    return float(10.0 * np.log10(data_range**2 / mse))


def compute_ssim(gt: np.ndarray, recon: np.ndarray) -> float:
    """Structural Similarity Index."""
    gt64 = gt.astype(np.float64)
    recon64 = recon.astype(np.float64)
    data_range = gt64.max() - gt64.min()
    if data_range < 1e-12:
        return 0.0
    c1 = (0.01 * data_range)**2
    c2 = (0.03 * data_range)**2
    mu_x = gt64.mean()
    mu_y = recon64.mean()
    var_x = gt64.var()
    var_y = recon64.var()
    cov_xy = np.mean((gt64 - mu_x) * (recon64 - mu_y))
    num = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
    den = (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    return float(num / den)


# ── Image helpers ────────────────────────────────────────────────────────────

def _norm(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo + 1e-8)


def _save_png(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(
        np.clip(_norm(arr) * 255, 0, 255).astype(np.uint8), "L"
    ).save(str(path))


# ── Tier generation ──────────────────────────────────────────────────────────

def sample_mismatch(rng: np.random.Generator, spec: dict) -> dict:
    return {k: float(rng.uniform(v["min"], v["max"])) for k, v in spec.items()}


def generate_tier(
    tier: str,
    phantoms: list[tuple[np.ndarray, dict]],
    base_seed: int,
) -> None:
    """Generate one tier of the SEM benchmark."""
    spec_ranges = SPEC[tier]
    tier_dir = BENCHMARK_DIR / tier
    images_dir = tier_dir / "images"
    if tier_dir.exists():
        shutil.rmtree(tier_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    h5_path = tier_dir / f"sem_challenge_{tier}.h5"
    rng = np.random.default_rng(base_seed)
    true_specs = {}
    all_psnrs = []
    all_ssims = []

    with h5py.File(h5_path, "w") as f:
        f.attrs["description"] = (
            f"PWM SEM benchmark -- {tier} tier "
            f"(BSE+SE imaging with Poisson noise, charging artifacts, "
            f"Everhart-Thornley detector)"
        )
        f.attrs["spec_ranges"] = json.dumps(spec_ranges)
        f.attrs["geometry"] = json.dumps({
            "image_size": IMAGE_SIZE,
            "pixel_size_nm": 5.0,
            "fov_nm": IMAGE_SIZE * 5.0,
        })
        f.attrs["forward_model"] = (
            "y = Poisson(eta * (BSE_yield(Z,theta) * PSF(x) + "
            "SE_yield * edge_enhancement)) * detector_response + noise"
        )

        for idx, (x_true, info) in enumerate(phantoms):
            key = f"sample_{idx:02d}"
            scene_name = info["name"]
            print(f"  [{tier}] Generating {key} ({scene_name})...", end="",
                  flush=True)

            mis = sample_mismatch(rng, spec_ranges)
            true_specs[key] = mis

            # Apply SEM forward model
            result = sem_forward_model(
                x_true,
                beam_voltage_kV=mis["beam_voltage_kV"],
                working_distance_mm=mis["working_distance_mm"],
                detector_bias=mis["detector_bias"],
                charging_artifact=mis["charging_artifact"],
                rng=rng,
            )

            y = result["y"]
            H_ideal = result["H_ideal"]

            # CPU baseline reconstruction
            recon = baseline_reconstruct(y, H_ideal,
                                          psf_sigma=result["psf_sigma"])

            psnr = compute_psnr(x_true, recon)
            ssim = compute_ssim(x_true, recon)
            all_psnrs.append(psnr)
            all_ssims.append(ssim)

            # Save to HDF5
            grp = f.create_group(key)
            grp.create_dataset("x_true", data=x_true.astype(np.float32),
                               compression="gzip")
            grp.create_dataset("y", data=y, compression="gzip")
            grp.create_dataset("H_ideal", data=H_ideal, compression="gzip")
            grp.create_dataset("reconstruction_baseline", data=recon,
                               compression="gzip")
            grp.attrs["metadata"] = json.dumps({
                "scene": scene_name,
                "shape": list(x_true.shape),
                "psf_sigma_px": result["psf_sigma"],
                "eta_bse": result["eta_bse"],
                "delta_se": result["delta_se"],
                "dose_scale": result["dose_scale"],
                "detector_pos": result["detector_pos"],
                "psnr_baseline": float(psnr),
                "ssim_baseline": float(ssim),
            })
            grp.attrs["true_spec"] = json.dumps(mis)
            grp.attrs["spec_ranges"] = json.dumps(spec_ranges)

            # Save preview images
            sample_dir = images_dir / f"sample_{idx:02d}_{scene_name}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            _save_png(x_true, sample_dir / "ground_truth.png")
            _save_png(y, sample_dir / "measurement.png")
            _save_png(H_ideal, sample_dir / "ideal_response.png")
            _save_png(recon, sample_dir / "reconstruction_baseline.png")
            with open(sample_dir / "spec.json", "w") as sf:
                json.dump({
                    "scene": scene_name,
                    "spec_ranges": spec_ranges,
                    "true_spec": mis,
                    "psnr_baseline": psnr,
                    "ssim_baseline": ssim,
                }, sf, indent=2)

            print(f"  PSNR={psnr:.2f} dB, SSIM={ssim:.3f}  "
                  f"V={mis['beam_voltage_kV']:.1f}kV  "
                  f"WD={mis['working_distance_mm']:.1f}mm  "
                  f"bias={mis['detector_bias']:.2f}  "
                  f"charge={mis['charging_artifact']:.3f}")

    # Save spec files
    with open(tier_dir / "spec.json", "w") as sf:
        json.dump(spec_ranges, sf, indent=2)
    with open(tier_dir / "true_spec.json", "w") as tf:
        json.dump(true_specs, tf, indent=2)

    mean_psnr = np.mean(all_psnrs)
    mean_ssim = np.mean(all_ssims)
    print(f"  [{tier}] {len(phantoms)} samples | "
          f"Mean PSNR={mean_psnr:.2f} dB | Mean SSIM={mean_ssim:.3f}")
    print(f"  [{tier}] HDF5 -> {h5_path.name}")


# ── Gallery image generation ────────────────────────────────────────────────

def generate_gallery_images() -> None:
    """Generate gallery images for the platform benchmark page.

    Creates scene_00 through scene_03 with:
        gt.png, measurement_I.png, measurement_II.png,
        recon_I.png, recon_II.png
    """
    gallery_base = (BENCHMARK_DIR.parent.parent.parent /
                    "platform" / "pwm_platform" / "static" / "img" /
                    "benchmark_gallery" / "sem")

    h5_path = BENCHMARK_DIR / "public" / "sem_challenge_public.h5"
    if not h5_path.exists():
        print("  [gallery] Public HDF5 not found, skipping gallery generation.")
        return

    # Pick 4 diverse samples: semiconductor (0), fracture (3),
    # nanoparticle (6), biological (9)
    gallery_sample_indices = [0, 3, 6, 9]

    with h5py.File(h5_path, "r") as f:
        for scene_idx, sample_idx in enumerate(gallery_sample_indices):
            scene_dir = gallery_base / f"scene_{scene_idx:02d}"
            scene_dir.mkdir(parents=True, exist_ok=True)

            key = f"sample_{sample_idx:02d}"
            if key not in f:
                print(f"  [gallery] {key} not found in HDF5, skipping.")
                continue

            grp = f[key]
            x_true = grp["x_true"][:]
            y = grp["y"][:]
            H_ideal = grp["H_ideal"][:]
            recon = grp["reconstruction_baseline"][:]

            # gt.png -- ground truth
            _save_png(x_true, scene_dir / "gt.png")

            # measurement_I.png -- measured SEM image (noisy)
            _save_png(y, scene_dir / "measurement_I.png")

            # measurement_II.png -- ideal system response
            _save_png(H_ideal, scene_dir / "measurement_II.png")

            # recon_I.png -- baseline reconstruction
            _save_png(recon, scene_dir / "recon_I.png")

            # recon_II.png -- difference |GT - recon|
            diff = np.abs(x_true - recon)
            _save_png(diff, scene_dir / "recon_II.png")

            print(f"  [gallery] scene_{scene_idx:02d} -> {scene_dir}")


# ── README ───────────────────────────────────────────────────────────────────

def _write_top_readme() -> None:
    txt = """# SEM -- Scanning Electron Microscopy Benchmark

## Overview

Scanning Electron Microscopy (SEM) benchmark with physics-based forward model:
focused electron beam interaction with sample surface, BSE/SE emission,
Everhart-Thornley detector geometry, Poisson counting noise, and charging artifacts.

## Forward Model

```
y = Poisson(eta * (BSE_yield(Z, theta) * PSF(x) + SE_yield * edge_enhancement))
    * detector_response + readout_noise

where:
    x                : surface/material property map (256x256)
    Z                : atomic number of material
    BSE_yield(Z,theta) : backscattered electron yield (Heinrich/Reuter model)
    SE_yield         : secondary electron yield (Sternglass model)
    edge_enhancement : gradient-based topographic contrast (Sobel operator)
    PSF              : Gaussian blur from interaction volume + aberrations
    detector_response: Everhart-Thornley cosine-law collection geometry
    eta              : dose scaling from beam current
    readout_noise    : Gaussian electronic noise (sigma=3.0)
```

## Geometry

| Parameter | Value |
|-----------|-------|
| IMAGE_SIZE | 256 x 256 |
| pixel_size | 5 nm/px |
| FOV | 1.28 um x 1.28 um |

## Mismatch Parameters

| Parameter | Description | Public | Dev | Hidden |
|-----------|-------------|--------|-----|--------|
| beam_voltage_kV | Accelerating voltage | 5-15 kV | 3-20 kV | 1-30 kV |
| working_distance_mm | Sample-lens distance | 3-8 mm | 2-12 mm | 1-15 mm |
| detector_bias | ET collection bias | 0.8-1.2 | 0.6-1.4 | 0.4-1.6 |
| charging_artifact | Sample charging | 0-0.05 | 0-0.15 | 0-0.30 |

## Phantoms

| Type | Description |
|------|-------------|
| Semiconductor | Lines, contacts, vias, trenches (IC surface) |
| Fracture | Multi-scale rough topography, crack features |
| Nanoparticles | High-Z particles on flat substrate |
| Biological | Cell membranes, organelles, fibers (tissue cross-section) |

## Dataset Structure

```
sem/
+-- README.md
+-- generate_dataset.py
+-- public/    12 samples (GT + ideal response + true spec visible)
+-- dev/       20 samples (blind eval, augmented variants)
+-- hidden/    20 samples (adversarial: contamination, scan artifacts)
```

## HDF5 Structure (per sample)

```
sample_XX/
+-- x_true (256, 256) float32            # Ground truth surface/material map
+-- y (256, 256) float32                  # Measured SEM image (normalized)
+-- H_ideal (256, 256) float32            # Ideal system response (normalized)
+-- reconstruction_baseline (256, 256) float32  # NLM+edge baseline
```

## CPU Baseline

Non-local means denoising + edge-preserving filter. Expected PSNR ~22-28 dB.

## Scoring

```
Score = 0.4 * PSNR_norm + 0.4 * SSIM + 0.2 * Consistency
```

## References

1. Heinrich, K.F.J. (1966) "Electron beam x-ray microanalysis."
2. Reuter, W. (1972) "Electron backscattering coefficients."
3. Sternglass, E.J. (1957) "Theory of secondary electron emission."
4. Kanaya, K. & Okayama, S. (1972) "Penetration and energy-loss theory
   of electrons in solid targets," J. Phys. D.
"""
    with open(BENCHMARK_DIR / "README.md", "w") as f:
        f.write(txt)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("SEM Benchmark Dataset Generator")
    print("=" * 68)
    print(f"Output: {BENCHMARK_DIR}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}, pixel size: 5 nm\n")

    # ── Public tier (12 samples) ────────────────────────────────────────────
    print("Generating public tier (12 samples)...")
    public_phantoms = generate_phantoms_public(12)
    generate_tier("public", public_phantoms, base_seed=0)

    # ── Dev tier (20 samples) ──────────────────────────────────────────────
    print("\nGenerating dev tier (20 samples)...")
    dev_phantoms = generate_phantoms_dev(20)
    generate_tier("dev", dev_phantoms, base_seed=10000)

    # ── Hidden tier (20 samples) ──────────────────────────────────────────
    print("\nGenerating hidden tier (20 samples)...")
    hidden_phantoms = generate_phantoms_hidden(20)
    generate_tier("hidden", hidden_phantoms, base_seed=20000)

    # ── README ──────────────────────────────────────────────────────────────
    _write_top_readme()

    # ── Gallery images ──────────────────────────────────────────────────────
    print("\nGenerating gallery images...")
    generate_gallery_images()

    print(f"\n{'=' * 68}")
    print("SEM benchmark dataset generation complete!")
    print(f"Output: {BENCHMARK_DIR}")
    print("=" * 68)


if __name__ == "__main__":
    main()
