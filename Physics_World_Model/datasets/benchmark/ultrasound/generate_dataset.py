#!/usr/bin/env python3
"""Generate high-quality B-mode ultrasound benchmark dataset.

PICMUS/CIRS-style tissue-mimicking phantoms with physically accurate forward model.

Forward model (depth-dependent PSF convolution + multiplicative speckle):
    scatterer_field(r) = x_true(r) * rayleigh_scatterers(r)
    For each depth row d:
        ideal(d, :)  = |conv1D_lateral(scatterer_field(d, :), PSF_lateral(d))|
    ideal(r)     = gaussian_axial(ideal(r))
    atten(r)     = ideal(r) * 10^(-2*alpha*f*depth(r)/20)
    y_linear(r)  = atten(r) + electronic_noise
    y(r)         = 20 * log10(y_linear / max) clipped to [-DR, 0] -> [0, 1]

PSF model (depth-dependent, physically motivated):
    sigma_lateral(d) = (f_number * lambda / 2) * (1 + depth_defocus_rate * d)
    sigma_axial      = n_cycles * lambda / 2   (constant, determined by pulse)
    where lambda = c / f, and depth_defocus_rate models geometric beam spread

Ground truth phantoms (256x256, PICMUS/CIRS-style):
    1. Anechoic cysts at multiple depths (standard resolution/contrast test)
    2. Hyperechoic lesions (bright inclusions)
    3. Mixed cyst+lesion phantoms
    4. Layered tissue interfaces (fascia, organ boundaries)
    5. Point targets (PSF/resolution measurement grid)
    6. Complex anatomy (masses, calcifications, shadows)

Mismatch parameters:
    speed_of_sound_error_pct : SoS error affecting focus (0-8%)
    attenuation_dB_cm_MHz    : tissue attenuation coefficient (0.3-1.2)
    speckle_density          : sub-resolution scatterers per cell (5-50)
    snr_db                   : electronic SNR (20-40 dB)

Tiers:
    public : 12 samples (PICMUS-standard phantoms)
    dev    : 20 samples (augmented, medium mismatch, different seeds)
    hidden : 20 samples (adversarial, wide mismatch, different seeds)

CPU reconstruction: Wiener deconvolution

Usage:
    cd datasets/benchmark/ultrasound
    python3 generate_dataset.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import fftconvolve

BENCHMARK_DIR = Path(__file__).resolve().parent
IMAGE_SIZE = 256

# -- Physics constants --------------------------------------------------------

SPEED_OF_SOUND = 1540.0       # m/s in soft tissue
FREQUENCY_HZ   = 5.0e6        # 5 MHz centre frequency
F_NUMBER       = 1.5           # aperture f-number
N_CYCLES       = 3.5           # pulse length in cycles
PIXEL_SIZE_MM  = 0.15          # mm per pixel (0.15mm -> 38.4mm FOV)
DYNAMIC_RANGE  = 60.0          # dB dynamic range for log-compression
DEPTH_DEFOCUS_RATE = 0.003     # lateral PSF broadening per pixel depth

# -- Mismatch spec ranges per tier -------------------------------------------

SPEC = {
    "public": {
        "speed_of_sound_error_pct": {"min": 0.0,  "max": 3.0,  "unit": "%"},
        "attenuation_dB_cm_MHz":    {"min": 0.3,  "max": 0.7,  "unit": "dB/cm/MHz"},
        "speckle_density":          {"min": 10,   "max": 25,   "unit": "scatterers/res_cell"},
        "snr_db":                   {"min": 30.0, "max": 40.0, "unit": "dB"},
    },
    "dev": {
        "speed_of_sound_error_pct": {"min": 0.0,  "max": 5.0,  "unit": "%"},
        "attenuation_dB_cm_MHz":    {"min": 0.3,  "max": 0.9,  "unit": "dB/cm/MHz"},
        "speckle_density":          {"min": 8,    "max": 35,   "unit": "scatterers/res_cell"},
        "snr_db":                   {"min": 25.0, "max": 38.0, "unit": "dB"},
    },
    "hidden": {
        "speed_of_sound_error_pct": {"min": 0.0,  "max": 8.0,  "unit": "%"},
        "attenuation_dB_cm_MHz":    {"min": 0.3,  "max": 1.2,  "unit": "dB/cm/MHz"},
        "speckle_density":          {"min": 5,    "max": 50,   "unit": "scatterers/res_cell"},
        "snr_db":                   {"min": 20.0, "max": 35.0, "unit": "dB"},
    },
}


# =============================================================================
# PSF generation (depth-dependent)
# =============================================================================

def _compute_wavelength_mm(
    c: float = SPEED_OF_SOUND,
    f: float = FREQUENCY_HZ,
    sos_error_pct: float = 0.0,
) -> float:
    """Acoustic wavelength in mm, including SoS mismatch."""
    c_eff = c * (1.0 + sos_error_pct / 100.0)
    return (c_eff / f) * 1000.0


def make_psf(
    f_number: float = F_NUMBER,
    frequency_hz: float = FREQUENCY_HZ,
    c: float = SPEED_OF_SOUND,
    n_cycles: float = N_CYCLES,
    pixel_size_mm: float = PIXEL_SIZE_MM,
    sos_error_pct: float = 0.0,
    depth_px: float = 0.0,
    depth_defocus_rate: float = DEPTH_DEFOCUS_RATE,
) -> np.ndarray:
    """Generate depth-dependent Gaussian PSF for B-mode ultrasound.

    Lateral resolution degrades with depth due to geometric beam spread.
    Axial resolution is constant (determined by pulse bandwidth).

    Parameters
    ----------
    depth_px : float
        Depth in pixels (row index). Lateral sigma grows with depth.
    depth_defocus_rate : float
        Rate at which lateral sigma grows per pixel depth.

    Returns
    -------
    psf : ndarray (K, K), sum = 1
    """
    wavelength_mm = _compute_wavelength_mm(c, frequency_hz, sos_error_pct)

    # Base lateral and axial resolution (in pixels)
    sigma_lateral_mm = f_number * wavelength_mm / 2.0
    sigma_axial_mm = n_cycles * wavelength_mm / 2.0

    sigma_lat_px = sigma_lateral_mm / pixel_size_mm
    sigma_ax_px = sigma_axial_mm / pixel_size_mm

    # Depth-dependent lateral broadening
    sigma_lat_px *= (1.0 + depth_defocus_rate * depth_px)

    # PSF kernel size
    k_lat = max(3, int(6 * sigma_lat_px) | 1)
    k_ax = max(3, int(6 * sigma_ax_px) | 1)
    k = max(k_lat, k_ax)
    if k % 2 == 0:
        k += 1
    k = min(k, 63)

    half = k // 2
    y = np.arange(-half, half + 1, dtype=np.float64)
    x = np.arange(-half, half + 1, dtype=np.float64)
    yy, xx = np.meshgrid(y, x, indexing="ij")

    psf = np.exp(-0.5 * ((yy / max(sigma_ax_px, 0.5)) ** 2 +
                          (xx / max(sigma_lat_px, 0.5)) ** 2))
    psf /= psf.sum() + 1e-12
    return psf.astype(np.float32)


def make_nominal_psf(sos_error_pct: float = 0.0) -> np.ndarray:
    """Generate a single nominal PSF at mid-depth (for Wiener recon)."""
    return make_psf(sos_error_pct=sos_error_pct,
                    depth_px=IMAGE_SIZE / 2.0)


# =============================================================================
# Phantom generators: PICMUS/CIRS-style tissue-mimicking phantoms
# =============================================================================

def _rayleigh_scatterer_field(
    rng: np.random.Generator,
    size: int = IMAGE_SIZE,
    scale: float = 0.3,
) -> np.ndarray:
    """Generate Rayleigh-distributed scatterer field.

    This is the fundamental source of ultrasound speckle texture.
    Real tissue contains millions of sub-resolution scatterers; the
    returned field represents the local scatterer amplitude distribution.
    Rayleigh distribution models the envelope of random-phase scatterer sums.
    """
    return rng.rayleigh(scale, (size, size)).astype(np.float32)


def _add_speckle_texture(
    phantom: np.ndarray,
    rng: np.random.Generator,
    scale: float = 0.25,
) -> np.ndarray:
    """Add multi-scale sub-resolution scatterer texture to a phantom.

    Creates the microstructure that produces speckle when convolved with PSF:
    - Fine texture (sigma~1.5 px): individual scatterer clumps
    - Medium texture (sigma~6 px): tissue heterogeneity / grain
    - Rayleigh base: fundamental scatterer distribution
    """
    size = phantom.shape[0]

    # Fine-scale scatterer distribution (Gaussian smoothed noise)
    fine = rng.standard_normal((size, size)).astype(np.float64)
    fine = gaussian_filter(fine, sigma=1.5) * scale * 0.5

    # Medium-scale tissue heterogeneity
    medium = gaussian_filter(
        rng.standard_normal((size, size)), sigma=6.0
    ) * scale * 0.25

    # Rayleigh scatterer base (multiplicative)
    rayleigh_base = rng.rayleigh(1.0, (size, size)).astype(np.float64)
    rayleigh_base = gaussian_filter(rayleigh_base, sigma=0.8)
    # Scale: rayleigh has mean ~1.25, normalise so mean ~1.0
    rayleigh_base /= rayleigh_base.mean() + 1e-10

    # Combine: phantom * rayleigh_modulation + additive texture
    result = phantom * rayleigh_base * (1.0 + fine + medium)
    return np.clip(result, 0.0, 5.0).astype(np.float32)


def _make_circle_mask(
    size: int, cy: int, cx: int, ry: float, rx: float,
) -> np.ndarray:
    """Elliptical mask (float, 0-1) with smooth boundary."""
    yy, xx = np.ogrid[:size, :size]
    dist = ((yy - cy) / max(ry, 1.0)) ** 2 + ((xx - cx) / max(rx, 1.0)) ** 2
    # Smooth edge (anti-aliased, ~2 px transition)
    mask = np.clip(1.0 - (dist - 0.9) / 0.2, 0.0, 1.0)
    return mask.astype(np.float64)


def _phantom_anechoic_cysts(
    rng: np.random.Generator,
    n_cysts: int = 6,
    size: int = IMAGE_SIZE,
) -> np.ndarray:
    """PICMUS-style anechoic cyst phantom.

    Standard test target: dark (fluid-filled) circular regions at different
    depths in a uniform speckle background. This is the gold standard for
    evaluating contrast resolution and cyst detectability.

    Cysts are placed at different depths (rows) to test depth-dependent
    performance. Cyst sizes vary from 4mm to 10mm equivalent diameter.
    """
    # Uniform tissue background (reflectivity ~1.0)
    phantom = np.ones((size, size), dtype=np.float64)
    # Slow spatial variation (tissue inhomogeneity)
    phantom += gaussian_filter(rng.standard_normal((size, size)),
                               sigma=35.0) * 0.10

    # Place cysts at different depths (PICMUS standard: 6 cysts)
    # Distribute across depth range, with random lateral positions
    depth_positions = np.linspace(size * 0.15, size * 0.85, n_cysts)
    for i in range(n_cysts):
        cy = int(depth_positions[i]) + rng.integers(-8, 9)
        cx = rng.integers(size // 5, 4 * size // 5)
        # Radius: 12-28 px (equivalent to ~1.8-4.2 mm at 0.15 mm/px)
        r = rng.integers(12, 29)
        # Slight ellipticity
        ry = r + rng.integers(-3, 4)
        rx = r + rng.integers(-3, 4)

        mask = _make_circle_mask(size, cy, cx, ry, rx)

        # Anechoic interior (very low reflectivity, ~0.01-0.03)
        cyst_val = rng.uniform(0.01, 0.03)
        # Bright wall (capsule/interface) ~2.0-3.0
        wall_mask = _make_circle_mask(size, cy, cx, ry + 2, rx + 2)
        wall_only = np.clip(wall_mask - mask, 0, 1)

        phantom = phantom * (1.0 - mask) + cyst_val * mask
        wall_brightness = rng.uniform(1.8, 2.8)
        phantom = phantom * (1.0 - wall_only) + wall_brightness * wall_only

    # Add 1-2 thin vessels (narrow anechoic lines)
    n_vessels = rng.integers(1, 3)
    for _ in range(n_vessels):
        vy = rng.integers(size // 6, 5 * size // 6)
        vx_start = rng.integers(0, size // 4)
        vx_end = rng.integers(3 * size // 4, size)
        thickness = rng.integers(2, 4)
        angle = rng.uniform(-0.12, 0.12)
        for vx in range(vx_start, vx_end):
            y_pos = int(vy + (vx - vx_start) * angle)
            y_lo = max(0, y_pos - thickness)
            y_hi = min(size, y_pos + thickness)
            if y_lo < y_hi:
                phantom[y_lo:y_hi, vx] = 0.01

    return _add_speckle_texture(
        np.clip(phantom, 0.0, 3.5).astype(np.float32), rng, scale=0.22)


def _phantom_hyperechoic_lesions(
    rng: np.random.Generator,
    n_lesions: int = 4,
    size: int = IMAGE_SIZE,
) -> np.ndarray:
    """Hyperechoic lesion phantom.

    Bright circular inclusions (simulating solid masses, calcifications,
    or fibrous tissue) at various depths. Contrast ratio typically 6-15 dB
    above background.
    """
    # Tissue background
    phantom = np.ones((size, size), dtype=np.float64) * 0.9
    phantom += gaussian_filter(rng.standard_normal((size, size)),
                               sigma=30.0) * 0.12

    # Place hyperechoic lesions
    depth_positions = np.linspace(size * 0.18, size * 0.82, n_lesions)
    for i in range(n_lesions):
        cy = int(depth_positions[i]) + rng.integers(-10, 11)
        cx = rng.integers(size // 5, 4 * size // 5)
        r = rng.integers(14, 32)
        ry = r + rng.integers(-4, 5)
        rx = r + rng.integers(-4, 5)

        mask = _make_circle_mask(size, cy, cx, ry, rx)

        # Hyperechoic interior: 2x-4x background brightness
        lesion_val = rng.uniform(2.0, 3.5)
        # Internal heterogeneity (realistic irregular echogenicity)
        internal_texture = gaussian_filter(
            rng.standard_normal((size, size)), sigma=4.0
        ) * 0.3
        lesion_field = lesion_val + internal_texture

        phantom = phantom * (1.0 - mask) + lesion_field * mask

    return _add_speckle_texture(
        np.clip(phantom, 0.0, 4.0).astype(np.float32), rng, scale=0.20)


def _phantom_mixed_cyst_lesion(
    rng: np.random.Generator,
    n_anechoic: int = 3,
    n_hyperechoic: int = 2,
    size: int = IMAGE_SIZE,
) -> np.ndarray:
    """Mixed phantom with both anechoic cysts and hyperechoic lesions.

    Realistic clinical scenario: tissue containing both fluid-filled
    cysts and solid masses, often seen in breast/thyroid imaging.
    """
    phantom = np.ones((size, size), dtype=np.float64) * 1.0
    phantom += gaussian_filter(rng.standard_normal((size, size)),
                               sigma=32.0) * 0.12

    # Anechoic cysts
    for _ in range(n_anechoic):
        cy = rng.integers(size // 6, 5 * size // 6)
        cx = rng.integers(size // 6, 5 * size // 6)
        r = rng.integers(10, 25)
        mask = _make_circle_mask(size, cy, cx, r, r + rng.integers(-3, 4))
        wall = _make_circle_mask(size, cy, cx, r + 2, r + 2)
        wall_only = np.clip(wall - mask, 0, 1)
        phantom = phantom * (1.0 - mask) + rng.uniform(0.01, 0.03) * mask
        phantom = phantom * (1.0 - wall_only) + rng.uniform(2.0, 2.8) * wall_only

    # Hyperechoic lesions
    for _ in range(n_hyperechoic):
        cy = rng.integers(size // 6, 5 * size // 6)
        cx = rng.integers(size // 6, 5 * size // 6)
        r = rng.integers(12, 28)
        mask = _make_circle_mask(size, cy, cx, r, r + rng.integers(-3, 4))
        phantom = phantom * (1.0 - mask) + rng.uniform(2.2, 3.2) * mask

    return _add_speckle_texture(
        np.clip(phantom, 0.0, 4.0).astype(np.float32), rng, scale=0.22)


def _phantom_layered_tissue(
    rng: np.random.Generator,
    size: int = IMAGE_SIZE,
) -> np.ndarray:
    """Layered tissue phantom (skin/fat/muscle/organ boundaries).

    Simulates a typical abdominal scan cross-section with distinct
    tissue layers separated by bright specular interfaces. Each layer
    has a characteristic echogenicity and fine texture.
    """
    phantom = np.zeros((size, size), dtype=np.float64)
    x_coords = np.arange(size, dtype=np.float64)

    n_layers = rng.integers(4, 7)
    boundaries = sorted(rng.integers(size // 8, 7 * size // 8, size=n_layers))

    # Layer reflectivities: each models a different tissue type
    tissue_types = [
        ("subcutaneous", rng.uniform(0.4, 0.6)),
        ("fat", rng.uniform(0.7, 1.0)),
        ("muscle", rng.uniform(1.2, 1.6)),
        ("fascia", rng.uniform(0.5, 0.8)),
        ("organ_parenchyma", rng.uniform(0.8, 1.3)),
        ("deeper_tissue", rng.uniform(0.5, 0.7)),
        ("background", rng.uniform(0.9, 1.1)),
    ]

    # Pre-compute undulations for boundaries (organ surfaces are not flat)
    undulations = []
    for _ in range(len(boundaries)):
        freq = rng.uniform(1.5, 4.0)
        amp = rng.uniform(3.0, 10.0)
        phase = rng.uniform(0, 2 * np.pi)
        und = (np.sin(2 * np.pi * x_coords / size * freq + phase) * amp).astype(int)
        undulations.append(und)

    prev_row = 0
    for i, boundary in enumerate(boundaries):
        und = undulations[i]
        _, val = tissue_types[i % len(tissue_types)]
        for col in range(size):
            row_end = max(prev_row, min(boundary + und[col], size - 1))
            phantom[prev_row:row_end, col] = val
        prev_row = boundary
    phantom[prev_row:, :] = tissue_types[-1][1]

    # Bright specular interface reflections (1-2 px wide bright lines)
    for i, boundary in enumerate(boundaries):
        und = undulations[i]
        bright_val = rng.uniform(2.2, 3.2)
        for col in range(size):
            row = boundary + und[col]
            if 1 <= row < size - 1:
                phantom[row - 1:row + 2, col] = bright_val

    return _add_speckle_texture(
        np.clip(phantom, 0.0, 3.5).astype(np.float32), rng, scale=0.22)


def _phantom_point_targets(
    rng: np.random.Generator,
    size: int = IMAGE_SIZE,
) -> np.ndarray:
    """Point target / resolution phantom (PICMUS-style).

    Low-reflectivity background with a regular grid of bright point
    reflectors and thin wire targets. This is the standard test for
    measuring axial/lateral resolution and sidelobe levels.

    The grid spacing and target brightness are chosen to match PICMUS
    resolution phantom specifications.
    """
    # Low background (makes point targets stand out)
    phantom = np.ones((size, size), dtype=np.float64) * 0.15

    # Regular grid of point targets
    n_rows = rng.integers(5, 8)
    n_cols = rng.integers(5, 8)
    row_sp = size // (n_rows + 1)
    col_sp = size // (n_cols + 1)

    for r in range(1, n_rows + 1):
        for c in range(1, n_cols + 1):
            py = r * row_sp + rng.integers(-2, 3)
            px = c * col_sp + rng.integers(-2, 3)
            if 1 <= py < size - 1 and 1 <= px < size - 1:
                # Single bright pixel (point scatterer)
                phantom[py, px] = 4.0
                # Tiny 3x3 halo (diffraction-limited point response)
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        phantom[py + dy, px + dx] = max(
                            phantom[py + dy, px + dx], 1.5)

    # Horizontal wire targets (1 px thick, for lateral resolution)
    n_wires = rng.integers(2, 5)
    for _ in range(n_wires):
        wy = rng.integers(size // 8, 7 * size // 8)
        wx_s = rng.integers(size // 8, size // 3)
        wx_e = rng.integers(2 * size // 3, 7 * size // 8)
        phantom[wy, wx_s:wx_e] = 3.5

    # Vertical wire targets (1 px thick, for axial resolution)
    n_vwires = rng.integers(1, 3)
    for _ in range(n_vwires):
        wx = rng.integers(size // 8, 7 * size // 8)
        wy_s = rng.integers(size // 8, size // 3)
        wy_e = rng.integers(2 * size // 3, 7 * size // 8)
        phantom[wy_s:wy_e, wx] = 3.5

    return _add_speckle_texture(
        np.clip(phantom, 0.0, 5.0).astype(np.float32), rng, scale=0.06)


def _phantom_complex_anatomy(
    rng: np.random.Generator,
    size: int = IMAGE_SIZE,
) -> np.ndarray:
    """Complex anatomy phantom for adversarial evaluation.

    Combines layered tissue with micro-calcifications, irregular
    hypoechoic masses, acoustic shadow effects, and heterogeneous
    tissue. Designed to challenge reconstruction algorithms.
    """
    phantom = _phantom_layered_tissue(rng, size)

    # Micro-calcifications (tiny bright spots, 1-3 px)
    n_calc = rng.integers(10, 25)
    for _ in range(n_calc):
        cy = rng.integers(15, size - 15)
        cx = rng.integers(15, size - 15)
        r = rng.integers(1, 3)
        yy, xx = np.ogrid[:size, :size]
        dist = (yy - cy) ** 2 + (xx - cx) ** 2
        phantom[dist <= r ** 2] = rng.uniform(3.0, 4.0)

    # Irregular hypoechoic mass with lobulated boundary
    mass_cy = rng.integers(size // 4, 3 * size // 4)
    mass_cx = rng.integers(size // 4, 3 * size // 4)
    mass_r = rng.integers(15, 35)

    yy, xx = np.ogrid[:size, :size]
    angles = np.arctan2(yy - mass_cy, xx - mass_cx)
    n_lobes = rng.integers(3, 8)
    radial_mod = 1.0 + 0.35 * np.sin(
        n_lobes * angles + rng.uniform(0, 2 * np.pi))
    dist = np.sqrt(
        (yy - mass_cy) ** 2 + (xx - mass_cx) ** 2
    ) / (mass_r * radial_mod)
    mass_mask = dist <= 1.0
    phantom[mass_mask] = rng.uniform(0.05, 0.25)

    # Acoustic shadow below mass (strong attenuator blocks sound)
    shadow_top = mass_cy + mass_r
    if shadow_top < size:
        sw = int(mass_r * 1.3)
        x_lo = max(0, mass_cx - sw)
        x_hi = min(size, mass_cx + sw)
        for row in range(shadow_top, size):
            decay = 0.5 * np.exp(-(row - shadow_top) / 35.0)
            phantom[row, x_lo:x_hi] *= (1.0 - decay)

    # Small bright lesion (calcified nodule)
    lx = rng.integers(size // 5, 4 * size // 5)
    ly = rng.integers(size // 5, 4 * size // 5)
    lr = rng.integers(5, 14)
    dist2 = (yy - ly) ** 2 + (xx - lx) ** 2
    phantom[dist2 <= lr ** 2] = rng.uniform(2.5, 3.8)

    return np.clip(phantom, 0.0, 5.0).astype(np.float32)


# -- Scene name tables --------------------------------------------------------

PUBLIC_SCENE_NAMES = [
    "anechoic_cyst_01", "anechoic_cyst_02",
    "hyperechoic_lesion_01", "hyperechoic_lesion_02",
    "mixed_cyst_lesion_01", "mixed_cyst_lesion_02",
    "layered_tissue_01", "layered_tissue_02",
    "point_target_01", "point_target_02",
    "point_target_03", "point_target_04",
]

DEV_SCENE_NAMES = [f"dev_tissue_{i:02d}" for i in range(20)]
HIDDEN_SCENE_NAMES = [f"hidden_adversarial_{i:02d}" for i in range(20)]


# -- Phantom generation per tier ----------------------------------------------

def generate_public_phantoms(
    rng: np.random.Generator,
) -> list[tuple[str, np.ndarray]]:
    """Generate 12 public phantoms: PICMUS/CIRS-standard test configurations.

    2 anechoic cyst + 2 hyperechoic lesion + 2 mixed + 2 layered + 4 point target
    """
    phantoms = []
    # 2 anechoic cyst phantoms (varying number and depth)
    for i in range(2):
        n_c = rng.integers(4, 7)
        x = _phantom_anechoic_cysts(rng, n_cysts=n_c)
        phantoms.append((PUBLIC_SCENE_NAMES[i], x))
    # 2 hyperechoic lesion phantoms
    for i in range(2):
        n_l = rng.integers(3, 6)
        x = _phantom_hyperechoic_lesions(rng, n_lesions=n_l)
        phantoms.append((PUBLIC_SCENE_NAMES[2 + i], x))
    # 2 mixed cyst+lesion phantoms
    for i in range(2):
        x = _phantom_mixed_cyst_lesion(
            rng,
            n_anechoic=rng.integers(2, 4),
            n_hyperechoic=rng.integers(1, 3),
        )
        phantoms.append((PUBLIC_SCENE_NAMES[4 + i], x))
    # 2 layered tissue phantoms
    for i in range(2):
        x = _phantom_layered_tissue(rng)
        phantoms.append((PUBLIC_SCENE_NAMES[6 + i], x))
    # 4 point target phantoms
    for i in range(4):
        x = _phantom_point_targets(rng)
        phantoms.append((PUBLIC_SCENE_NAMES[8 + i], x))
    return phantoms


def generate_dev_phantoms(
    rng: np.random.Generator,
) -> list[tuple[str, np.ndarray]]:
    """Generate 20 dev phantoms: mixed types with augmentation."""
    phantoms = []
    generators = [
        lambda r: _phantom_anechoic_cysts(r, n_cysts=r.integers(3, 7)),
        lambda r: _phantom_hyperechoic_lesions(r, n_lesions=r.integers(2, 6)),
        lambda r: _phantom_mixed_cyst_lesion(r),
        lambda r: _phantom_layered_tissue(r),
        lambda r: _phantom_point_targets(r),
    ]
    for i in range(20):
        gen = generators[i % len(generators)]
        x = gen(rng)
        # Random augmentation (flip/rotate)
        k = rng.integers(0, 4)
        if k > 0:
            x = np.rot90(x, k).copy()
        if rng.random() < 0.5:
            x = np.fliplr(x).copy()
        if rng.random() < 0.3:
            x = np.flipud(x).copy()
        phantoms.append((DEV_SCENE_NAMES[i], x))
    return phantoms


def generate_hidden_phantoms(
    rng: np.random.Generator,
) -> list[tuple[str, np.ndarray]]:
    """Generate 20 hidden phantoms: adversarial complex anatomy."""
    phantoms = []
    for i in range(20):
        if i < 10:
            x = _phantom_complex_anatomy(rng)
        elif i < 14:
            x = _phantom_anechoic_cysts(rng, n_cysts=rng.integers(5, 9))
        elif i < 17:
            x = _phantom_mixed_cyst_lesion(
                rng, n_anechoic=rng.integers(3, 6),
                n_hyperechoic=rng.integers(2, 5))
        else:
            x = _phantom_layered_tissue(rng)
        # Random augmentation
        k = rng.integers(0, 4)
        if k > 0:
            x = np.rot90(x, k).copy()
        if rng.random() < 0.7:
            x = np.fliplr(x).copy()
        if rng.random() < 0.5:
            x = np.flipud(x).copy()
        phantoms.append((HIDDEN_SCENE_NAMES[i], x))
    return phantoms


# =============================================================================
# Forward model: depth-dependent PSF + multiplicative speckle
# =============================================================================

def apply_depth_dependent_psf(
    x_field: np.ndarray,
    sos_error_pct: float = 0.0,
    n_depth_bands: int = 8,
) -> np.ndarray:
    """Apply depth-dependent PSF convolution.

    Divides the image into depth bands, each convolved with a PSF
    whose lateral width increases with depth (modeling beam divergence
    and focusing degradation). Adjacent bands are blended smoothly.

    Parameters
    ----------
    x_field : ndarray (H, W)
        Input scatterer field (tissue reflectivity * scatterer amplitude).
    sos_error_pct : float
        Speed-of-sound error percentage (affects PSF width).
    n_depth_bands : int
        Number of depth bands for piecewise PSF application.

    Returns
    -------
    envelope : ndarray (H, W), float32
        PSF-convolved envelope (always non-negative).
    """
    H, W = x_field.shape
    result = np.zeros((H, W), dtype=np.float64)
    x64 = x_field.astype(np.float64)

    band_height = H / n_depth_bands

    for band_idx in range(n_depth_bands):
        # Depth at the centre of this band
        depth_px = (band_idx + 0.5) * band_height
        psf = make_psf(sos_error_pct=sos_error_pct, depth_px=depth_px)

        # Convolve full image with this PSF
        convolved = fftconvolve(x64, psf.astype(np.float64), mode="same")

        # Soft weighting: this band's contribution (triangle window)
        row_start = int(band_idx * band_height)
        row_end = min(H, int((band_idx + 1) * band_height))

        # Create weight ramp for smooth blending between bands
        weights = np.zeros(H, dtype=np.float64)
        band_center = (band_idx + 0.5) * band_height
        for r in range(H):
            d = abs(r - band_center) / band_height
            if d < 1.0:
                weights[r] = 1.0 - d
        weights_2d = weights[:, None]
        result += convolved * weights_2d

    # Normalise (weights sum to ~1 at each row for sufficiently many bands)
    weight_sum = np.zeros(H, dtype=np.float64)
    for band_idx in range(n_depth_bands):
        band_center = (band_idx + 0.5) * band_height
        for r in range(H):
            d = abs(r - band_center) / band_height
            if d < 1.0:
                weight_sum[r] += 1.0 - d
    weight_sum = np.maximum(weight_sum, 1e-10)
    result /= weight_sum[:, None]

    return np.abs(result).astype(np.float32)


def apply_attenuation(
    bmode: np.ndarray,
    attenuation_dB_cm_MHz: float,
    frequency_MHz: float = 5.0,
    pixel_size_mm: float = PIXEL_SIZE_MM,
) -> np.ndarray:
    """Apply depth-dependent tissue attenuation (round-trip).

    Standard tissue attenuation model: signal loss = 2*alpha*f*depth
    where factor 2 accounts for round-trip propagation.
    """
    H = bmode.shape[0]
    depth_cm = np.arange(H, dtype=np.float64) * pixel_size_mm / 10.0
    atten_dB = 2.0 * attenuation_dB_cm_MHz * frequency_MHz * depth_cm
    gain = 10.0 ** (-atten_dB / 20.0)
    return (bmode * gain[:, None]).astype(np.float32)


def apply_speckle_noise(
    bmode: np.ndarray,
    speckle_density: int,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply multiplicative speckle noise and additive electronic noise.

    Speckle model: coherent sum of N random-phase scatterers per
    resolution cell. The envelope follows Rayleigh statistics when N
    is large. Implemented via I/Q summation.

    Electronic noise: additive Gaussian at specified SNR.
    """
    N = max(speckle_density, 1)
    H, W = bmode.shape

    # Efficient speckle: sum N random phasors in I/Q
    # For large N, use batch processing to avoid slow Python loop
    if N <= 30:
        I_sum = np.zeros((H, W), dtype=np.float64)
        Q_sum = np.zeros((H, W), dtype=np.float64)
        for _ in range(N):
            phase = rng.uniform(0, 2 * np.pi, (H, W))
            amp = rng.rayleigh(1.0, (H, W))
            I_sum += amp * np.cos(phase)
            Q_sum += amp * np.sin(phase)
    else:
        # Batch: generate all at once for speed
        phases = rng.uniform(0, 2 * np.pi, (N, H, W))
        amps = rng.rayleigh(1.0, (N, H, W))
        I_sum = np.sum(amps * np.cos(phases), axis=0)
        Q_sum = np.sum(amps * np.sin(phases), axis=0)

    speckle_envelope = np.sqrt(I_sum ** 2 + Q_sum ** 2) / N

    # Multiplicative speckle
    bmode_speckled = bmode * speckle_envelope.astype(np.float32)

    # Additive electronic noise
    sig_power = np.mean(bmode_speckled ** 2) + 1e-12
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise_sigma = np.sqrt(noise_power)
    elec_noise = rng.normal(0.0, noise_sigma, (H, W)).astype(np.float32)

    return np.maximum(bmode_speckled + elec_noise, 1e-10).astype(np.float32)


def log_compress(
    bmode: np.ndarray,
    dynamic_range_dB: float = DYNAMIC_RANGE,
) -> np.ndarray:
    """Log-compress B-mode: y = 20*log10(B/B_max), clipped to [-DR, 0] dB.

    Output normalised to [0, 1] where 0 = bottom of dynamic range, 1 = peak.
    """
    bmode_pos = np.maximum(bmode, 1e-10)
    b_max = bmode_pos.max()
    y_dB = 20.0 * np.log10(bmode_pos / b_max)
    y_clipped = np.clip(y_dB, -dynamic_range_dB, 0.0)
    y_norm = (y_clipped + dynamic_range_dB) / dynamic_range_dB
    return y_norm.astype(np.float32)


def forward_model(
    x_true: np.ndarray,
    sos_error_pct: float,
    attenuation_dB_cm_MHz: float,
    speckle_density: int,
    snr_db: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Complete B-mode ultrasound forward model.

    1. Depth-dependent PSF convolution (lateral resolution degrades with depth)
    2. Depth-dependent tissue attenuation
    3. Multiplicative speckle noise
    4. Additive electronic noise
    5. Log-compression to standard B-mode display

    Returns
    -------
    bmode_ideal : ndarray (H, W) float32
        Clean B-mode (PSF + attenuation only), log-compressed [0, 1]
    bmode_measured : ndarray (H, W) float32
        Noisy B-mode (full degradation), log-compressed [0, 1]
    psf_nominal : ndarray (K, K) float32
        Nominal PSF at mid-depth (for reconstruction)
    """
    # Depth-dependent PSF convolution
    envelope = apply_depth_dependent_psf(x_true, sos_error_pct=sos_error_pct)

    # Depth-dependent attenuation
    envelope_atten = apply_attenuation(envelope, attenuation_dB_cm_MHz)

    # Ideal B-mode (no speckle, no noise) for reference
    bmode_ideal = log_compress(envelope_atten)

    # Full noisy measurement
    bmode_noisy = apply_speckle_noise(
        envelope_atten, speckle_density, snr_db, rng)
    bmode_measured = log_compress(bmode_noisy)

    # Nominal PSF at mid-depth (provided to reconstruction algorithms)
    psf_nominal = make_nominal_psf(sos_error_pct=sos_error_pct)

    return bmode_ideal, bmode_measured, psf_nominal


# =============================================================================
# CPU Reconstruction: Wiener deconvolution
# =============================================================================

def reconstruct_wiener(
    bmode_measured: np.ndarray,
    psf: np.ndarray,
    noise_variance: float = 0.005,
) -> np.ndarray:
    """Wiener deconvolution reconstruction from log-compressed noisy B-mode.

    Inverts log-compression, applies Wiener filter in frequency domain,
    then normalises to [0, 1].
    """
    # Invert log-compression to linear domain
    bmode_linear = 10.0 ** (
        (bmode_measured * DYNAMIC_RANGE - DYNAMIC_RANGE) / 20.0)

    H, W = bmode_linear.shape
    ph, pw = psf.shape

    # Zero-pad PSF to image size, centred at origin
    psf_padded = np.zeros((H, W), dtype=np.float64)
    py = (H - ph) // 2
    px = (W - pw) // 2
    psf_padded[py:py + ph, px:px + pw] = psf
    psf_shifted = np.roll(
        np.roll(psf_padded, -ph // 2, axis=0), -pw // 2, axis=1)

    # Frequency-domain Wiener filter: H* / (|H|^2 + K)
    F_sig = np.fft.fft2(bmode_linear.astype(np.float64))
    F_psf = np.fft.fft2(psf_shifted)
    F_recon = F_sig * np.conj(F_psf) / (np.abs(F_psf) ** 2 + noise_variance)
    recon = np.real(np.fft.ifft2(F_recon))
    recon = np.clip(recon, 0.0, None)

    r_max = recon.max()
    if r_max > 1e-8:
        recon /= r_max
    return recon.astype(np.float32)


# =============================================================================
# Metrics
# =============================================================================

def compute_psnr(gt: np.ndarray, recon: np.ndarray) -> float:
    mse = float(np.mean(
        (gt.astype(np.float64) - recon.astype(np.float64)) ** 2))
    if mse < 1e-12:
        return 100.0
    dr = float(gt.max() - gt.min())
    if dr < 1e-12:
        return 0.0
    return float(10.0 * np.log10(dr ** 2 / mse))


def compute_ssim(gt: np.ndarray, recon: np.ndarray) -> float:
    """Simple global SSIM (no skimage dependency)."""
    g = gt.astype(np.float64)
    r = recon.astype(np.float64)
    dr = float(g.max() - g.min())
    if dr < 1e-12:
        return 0.0
    c1 = (0.01 * dr) ** 2
    c2 = (0.03 * dr) ** 2
    mu_g, mu_r = g.mean(), r.mean()
    var_g, var_r = g.var(), r.var()
    cov = float(np.mean((g - mu_g) * (r - mu_r)))
    return float(((2 * mu_g * mu_r + c1) * (2 * cov + c2)) /
                 ((mu_g ** 2 + mu_r ** 2 + c1) * (var_g + var_r + c2)))


# =============================================================================
# Image helpers
# =============================================================================

def _norm(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-8:
        return np.zeros_like(a, dtype=np.float32)
    return ((a - lo) / (hi - lo)).astype(np.float32)


def _save_png(arr: np.ndarray, path: Path) -> None:
    """Save a 2D array as 8-bit grayscale PNG (256x256)."""
    normed = np.clip(_norm(arr) * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(normed, "L")
    if img.size != (IMAGE_SIZE, IMAGE_SIZE):
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    img.save(str(path))


def _save_overview(
    x_true: np.ndarray,
    bmode_ideal: np.ndarray,
    bmode_measured: np.ndarray,
    recon: np.ndarray,
    path: Path,
) -> None:
    """Save 2x2 overview panel: GT | Ideal | Measured | Recon (256x256)."""
    th, tw = 128, 128

    def _r(a: np.ndarray) -> np.ndarray:
        pil = Image.fromarray(
            np.clip(_norm(a) * 255, 0, 255).astype(np.uint8), "L"
        )
        return np.array(pil.resize((tw, th), Image.LANCZOS))

    top = np.hstack([_r(x_true), _r(bmode_ideal)])
    bot = np.hstack([_r(bmode_measured), _r(recon)])
    ov = np.vstack([top, bot])
    Image.fromarray(ov, "L").save(str(path))


# =============================================================================
# Mismatch sampling
# =============================================================================

def sample_mismatch(rng: np.random.Generator, spec: dict) -> dict:
    result = {}
    for k, v in spec.items():
        lo, hi = v["min"], v["max"]
        if isinstance(lo, int) and isinstance(hi, int):
            result[k] = int(rng.integers(lo, hi + 1))
        else:
            result[k] = float(rng.uniform(lo, hi))
    return result


# =============================================================================
# Tier generator
# =============================================================================

def generate_tier(
    tier: str,
    phantoms: list[tuple[str, np.ndarray]],
    base_seed: int,
) -> None:
    """Generate one tier of the ultrasound benchmark."""
    spec_ranges = SPEC[tier]
    tier_dir = BENCHMARK_DIR / tier
    images_dir = tier_dir / "images"
    if tier_dir.exists():
        shutil.rmtree(tier_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    h5_path = tier_dir / f"ultrasound_challenge_{tier}.h5"
    rng = np.random.default_rng(base_seed)
    rows, true_specs = [], {}
    psnr_list, ssim_list = [], []

    with h5py.File(h5_path, "w") as f:
        f.attrs["description"] = (
            f"PWM Ultrasound B-mode benchmark -- {tier} tier "
            f"(depth-dependent PSF + multiplicative speckle + "
            f"log-compression)")
        f.attrs["spec_ranges"] = json.dumps(spec_ranges)
        f.attrs["geometry"] = json.dumps({
            "image_size": IMAGE_SIZE,
            "pixel_size_mm": PIXEL_SIZE_MM,
            "fov_mm": IMAGE_SIZE * PIXEL_SIZE_MM,
            "frequency_MHz": FREQUENCY_HZ / 1e6,
            "f_number": F_NUMBER,
            "n_cycles": N_CYCLES,
            "speed_of_sound_m_s": SPEED_OF_SOUND,
            "dynamic_range_dB": DYNAMIC_RANGE,
            "depth_defocus_rate": DEPTH_DEFOCUS_RATE,
        })

        for idx, (scene_name, x_true) in enumerate(phantoms):
            key = f"sample_{idx:02d}"
            mis = sample_mismatch(rng, spec_ranges)
            true_specs[key] = {**mis, "scene": scene_name}

            bmode_ideal, bmode_measured, psf_nominal = forward_model(
                x_true,
                sos_error_pct=mis["speed_of_sound_error_pct"],
                attenuation_dB_cm_MHz=mis["attenuation_dB_cm_MHz"],
                speckle_density=mis["speckle_density"],
                snr_db=mis["snr_db"],
                rng=rng,
            )

            recon = reconstruct_wiener(
                bmode_measured, psf_nominal, noise_variance=0.005)
            gt_norm = _norm(x_true)
            recon_norm = _norm(recon)
            psnr = compute_psnr(gt_norm, recon_norm)
            ssim = compute_ssim(gt_norm, recon_norm)
            psnr_list.append(psnr)
            ssim_list.append(ssim)

            grp = f.create_group(key)
            grp.create_dataset("x_true", data=x_true, compression="gzip")
            grp.create_dataset("bmode_ideal", data=bmode_ideal,
                               compression="gzip")
            grp.create_dataset("bmode_measured", data=bmode_measured,
                               compression="gzip")
            grp.create_dataset("psf", data=psf_nominal, compression="gzip")

            grp.attrs["metadata"] = json.dumps({
                "scene": scene_name,
                "shape": list(x_true.shape),
                "psf_shape": list(psf_nominal.shape),
                "wiener_psnr_dB": round(psnr, 2),
                "wiener_ssim": round(ssim, 4),
            })
            grp.attrs["spec_ranges"] = json.dumps(spec_ranges)
            grp.attrs["true_spec"] = json.dumps(
                {**mis, "scene": scene_name})

            sample_dir = images_dir / f"sample_{idx:02d}_{scene_name}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            _save_png(x_true, sample_dir / "ground_truth.png")
            _save_png(bmode_ideal, sample_dir / "bmode_ideal.png")
            _save_png(bmode_measured, sample_dir / "bmode_measured.png")
            _save_png(recon, sample_dir / "reconstruction.png")
            _save_overview(x_true, bmode_ideal, bmode_measured, recon,
                           sample_dir / "overview.png")

            with open(sample_dir / "spec.json", "w") as sf:
                json.dump({
                    "scene": scene_name,
                    "spec_ranges": spec_ranges,
                    "true_spec": mis,
                    "wiener_psnr_dB": round(psnr, 2),
                    "wiener_ssim": round(ssim, 4),
                }, sf, indent=2)

            rows.append((key, scene_name, mis, psnr, ssim))
            print(f"  [{tier}] {key} {scene_name}  "
                  f"SoS_err={mis['speed_of_sound_error_pct']:.1f}%  "
                  f"atten={mis['attenuation_dB_cm_MHz']:.2f}  "
                  f"speckle={mis['speckle_density']}  "
                  f"SNR={mis['snr_db']:.1f}dB  "
                  f"PSNR={psnr:.2f}dB  SSIM={ssim:.4f}")

    with open(tier_dir / "spec.json", "w") as sf:
        json.dump(spec_ranges, sf, indent=2)
    with open(tier_dir / "true_spec.json", "w") as tf:
        json.dump(true_specs, tf, indent=2)

    _write_tier_readme(tier, tier_dir, rows)
    print(f"  [{tier}] HDF5 -> {h5_path.name}  "
          f"avg PSNR={np.mean(psnr_list):.2f} dB  "
          f"avg SSIM={np.mean(ssim_list):.4f}")


# =============================================================================
# README writers
# =============================================================================

def _write_tier_readme(tier: str, tier_dir: Path, rows: list) -> None:
    spec = SPEC[tier]
    if tier == "public":
        access = "Full (GT + true spec + ideal B-mode)"
        source = ("PICMUS/CIRS-style tissue-mimicking phantoms: anechoic cysts, "
                  "hyperechoic lesions, mixed, layered tissue, point targets "
                  "(12 samples)")
    elif tier == "dev":
        access = "Blind (measured B-mode + spec ranges only)"
        source = "Augmented phantoms -- mixed types (20 samples)"
    else:
        access = "Server-only"
        source = ("Adversarial phantoms -- complex anatomy, "
                  "micro-calcifications, irregular masses (20 samples)")

    param_desc = {
        "speed_of_sound_error_pct": "SoS error affecting PSF focus",
        "attenuation_dB_cm_MHz": "Tissue attenuation coefficient",
        "speckle_density": "Sub-resolution scatterers per cell",
        "snr_db": "Electronic signal-to-noise ratio",
    }

    lines = [
        f"# Ultrasound B-Mode {tier.capitalize()} Tier\n\n",
        f"**Source:** {source}\n\n",
        f"**Access:** {access}\n\n",
        "## Mismatch Parameters\n\n",
        "| Parameter | Description | Range |\n",
        "|-----------|-------------|-------|\n",
    ]
    for k, v in spec.items():
        lo, hi, u = v["min"], v["max"], v.get("unit", "")
        lines.append(
            f"| `{k}` | {param_desc.get(k, k)} | [{lo}, {hi}] {u} |\n")

    lines += [
        "\n## Samples\n\n",
        "| Sample | Scene | SoS Err (%) | Atten | Speckle | SNR (dB) | "
        "PSNR (dB) | SSIM |\n",
        "|--------|-------|-------------|-------|---------|----------|"
        "-----------|------|\n",
    ]
    for key, scene, mis, psnr, ssim in rows:
        lines.append(
            f"| {key} | {scene}"
            f" | {mis['speed_of_sound_error_pct']:.1f}"
            f" | {mis['attenuation_dB_cm_MHz']:.2f}"
            f" | {mis['speckle_density']}"
            f" | {mis['snr_db']:.1f}"
            f" | {psnr:.2f}"
            f" | {ssim:.4f} |\n")

    lines += [
        "\n## HDF5 Datasets per Sample\n\n",
        "| Key | Shape | Dtype | Description |\n",
        "|-----|-------|-------|-------------|\n",
        "| `x_true` | (256, 256) | float32 | "
        "Ground-truth tissue reflectivity |\n",
        "| `bmode_ideal` | (256, 256) | float32 | "
        "Clean B-mode (PSF only, log-compressed [0,1]) |\n",
        "| `bmode_measured` | (256, 256) | float32 | "
        "Noisy B-mode (speckle + noise, log-compressed [0,1]) |\n",
        "| `psf` | (K, K) | float32 | "
        "Nominal PSF at mid-depth |\n",
    ]

    with open(tier_dir / "README.md", "w") as f:
        f.writelines(lines)


def _write_top_readme() -> None:
    txt = """# Ultrasound B-Mode Benchmark Dataset

## Overview

High-quality B-mode ultrasound imaging benchmark with PICMUS/CIRS-style
tissue-mimicking phantoms, depth-dependent PSF convolution, multiplicative
speckle noise, and log-compression forward model.

## Forward Model

```
scatterer_field = x_true * rayleigh_scatterers
envelope        = depth_dependent_PSF_conv(scatterer_field)
attenuated      = envelope * 10^(-2*alpha*f*depth/20)
y_linear        = attenuated * speckle_envelope + electronic_noise
y               = 20*log10(y_linear / max) -> clipped to [-DR, 0] -> [0,1]
```

## Depth-Dependent PSF Model

The PSF varies with depth to model beam divergence:
  - sigma_lateral(d) = (f_number * lambda / 2) * (1 + defocus_rate * d)
  - sigma_axial = n_cycles * lambda / 2 (constant)
  - Image divided into 8 depth bands, PSFs blended smoothly

| Parameter | Value |
|-----------|-------|
| Frequency | 5 MHz |
| Speed of sound | 1540 m/s |
| F-number | 1.5 |
| Pulse cycles | 3.5 |
| Pixel size | 0.15 mm |
| FOV | 38.4 mm |
| Dynamic range | 60 dB |
| Depth defocus rate | 0.003 per pixel |

## Mismatch Parameters (ThetaSpace)

| Knob | Symbol | Description | Public | Dev | Hidden |
|------|--------|-------------|--------|-----|--------|
| `speed_of_sound_error_pct` | SoS err | Focus error from SoS mismatch | 0-3% | 0-5% | 0-8% |
| `attenuation_dB_cm_MHz` | alpha | Tissue attenuation | 0.3-0.7 | 0.3-0.9 | 0.3-1.2 |
| `speckle_density` | N_s | Scatterers per resolution cell | 10-25 | 8-35 | 5-50 |
| `snr_db` | SNR | Electronic SNR | 30-40 dB | 25-38 dB | 20-35 dB |

## Phantom Types (PICMUS/CIRS-Style)

| Type | Description | Tier |
|------|-------------|------|
| Anechoic cysts | Dark fluid-filled cysts at varying depths (PICMUS standard) | Public, Dev |
| Hyperechoic lesions | Bright solid inclusions with internal texture | Public, Dev |
| Mixed cyst+lesion | Both anechoic and hyperechoic targets (clinical scenario) | Public, Dev |
| Layered tissue | Skin/fat/muscle/organ with specular interfaces | Public, Dev |
| Point targets | Grid of point reflectors + wire targets (resolution test) | Public, Dev |
| Complex anatomy | Micro-calcifications, irregular masses, acoustic shadows | Hidden |

## Dataset Structure

```
ultrasound/
+-- README.md
+-- generate_dataset.py
+-- public/    12 samples (PICMUS-standard test configurations)
|   +-- ultrasound_challenge_public.h5
|   +-- spec.json / true_spec.json
|   +-- images/sample_XX_*/
+-- dev/       20 samples (augmented, medium mismatch)
|   +-- ultrasound_challenge_dev.h5
|   +-- spec.json / true_spec.json
|   +-- images/sample_XX_*/
+-- hidden/    20 samples (adversarial, wide mismatch)
    +-- ultrasound_challenge_hidden.h5
    +-- spec.json / true_spec.json
    +-- images/sample_XX_*/
```

## HDF5 Structure (per sample)

```
sample_XX/
+-- x_true (256, 256) float32         -- Ground-truth tissue reflectivity
+-- bmode_ideal (256, 256) float32    -- Clean B-mode (log-compressed [0,1])
+-- bmode_measured (256, 256) float32 -- Noisy B-mode (log-compressed [0,1])
+-- psf (K, K) float32               -- Nominal PSF at mid-depth
```

## CPU Reconstruction

Wiener deconvolution in the frequency domain:
  F_recon = F_signal * conj(F_psf) / (|F_psf|^2 + K)
where K is the noise regularization parameter.

## Scoring

```
Score = 0.4 * PSNR_norm + 0.4 * SSIM + 0.2 * Consistency
```

## References

- Liebgott et al. (2016) "Plane-wave imaging challenge in medical ultrasound" IEEE IUS.
- Perrot et al. (2021) "So you think you can DAS?" IEEE TUFFC 68(2):355-381.
- Matrone et al. (2015) "DMAS" IEEE TUFFC 62(3):537-545.
- Jensen (1996) Field II ultrasound simulation program. MPC 4:351-353.
- CIRS Inc. Multi-Purpose Multi-Tissue Ultrasound Phantom (Model 040GSE).
"""
    with open(BENCHMARK_DIR / "README.md", "w") as f:
        f.write(txt)


# =============================================================================
# Gallery image generation
# =============================================================================

def generate_gallery_images(
    phantoms: list[tuple[str, np.ndarray]],
    gallery_dir: Path,
    n_scenes: int = 4,
    seed: int = 5555,
) -> None:
    """Generate gallery images for the platform benchmark page.

    Each scene: gt.png, measurement_I.png, measurement_II.png,
                recon_I.png, recon_II.png, recon_III.png
    All 256x256 grayscale PNGs.
    """
    rng = np.random.default_rng(seed)

    # Use diverse phantom types for the 4 gallery scenes:
    #   scene_00: anechoic cyst (index 0)
    #   scene_01: hyperechoic lesion (index 2)
    #   scene_02: mixed cyst+lesion (index 4)
    #   scene_03: point target (index 8)
    gallery_indices = [0, 2, 4, 8]

    for si, pidx in enumerate(gallery_indices[:n_scenes]):
        if pidx >= len(phantoms):
            pidx = si % len(phantoms)
        scene_name, x_true = phantoms[pidx]
        scene_dir = gallery_dir / f"scene_{si:02d}"
        scene_dir.mkdir(parents=True, exist_ok=True)

        _save_png(x_true, scene_dir / "gt.png")

        # Measurement I: mild conditions
        _, bm1, psf1 = forward_model(x_true, 1.0, 0.5, 15, 35.0, rng)
        _save_png(bm1, scene_dir / "measurement_I.png")

        # Measurement II: heavier degradation
        _, bm2, psf2 = forward_model(x_true, 4.0, 0.8, 8, 25.0, rng)
        _save_png(bm2, scene_dir / "measurement_II.png")

        # Recon I: Wiener from mild
        r1 = reconstruct_wiener(bm1, psf1, 0.005)
        _save_png(r1, scene_dir / "recon_I.png")

        # Recon II: Wiener from heavy
        r2 = reconstruct_wiener(bm2, psf2, 0.01)
        _save_png(r2, scene_dir / "recon_II.png")

        # Recon III: Wiener with mismatched PSF
        psf3 = make_nominal_psf(sos_error_pct=6.0)
        r3 = reconstruct_wiener(bm1, psf3, 0.005)
        _save_png(r3, scene_dir / "recon_III.png")

        print(f"  Gallery scene_{si:02d} ({scene_name}): 6 images saved")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("Ultrasound B-Mode Benchmark Dataset Generator")
    print("  PICMUS/CIRS-style tissue-mimicking phantoms")
    print("  Depth-dependent PSF + multiplicative speckle")
    print("=" * 60)
    print(f"Output: {BENCHMARK_DIR}\n")

    # -- Public tier (12 samples) -----------------------------------------
    print("Generating public tier (12 samples)...")
    rng_pub = np.random.default_rng(1000)
    public_phantoms = generate_public_phantoms(rng_pub)
    generate_tier("public", public_phantoms, base_seed=1000)

    # -- Dev tier (20 samples) --------------------------------------------
    print("\nGenerating dev tier (20 samples, augmented)...")
    rng_dev = np.random.default_rng(2000 + 10000)
    dev_phantoms = generate_dev_phantoms(rng_dev)
    generate_tier("dev", dev_phantoms, base_seed=2000 + 10000)

    # -- Hidden tier (20 samples) -----------------------------------------
    print("\nGenerating hidden tier (20 samples, adversarial)...")
    rng_hid = np.random.default_rng(3000 + 20000)
    hidden_phantoms = generate_hidden_phantoms(rng_hid)
    generate_tier("hidden", hidden_phantoms, base_seed=3000 + 20000)

    # -- Gallery images ---------------------------------------------------
    gallery_dir = (Path(__file__).resolve().parent.parent.parent.parent /
                   "platform" / "pwm_platform" / "static" / "img" /
                   "benchmark_gallery" / "ultrasound")
    print(f"\nGenerating gallery images at {gallery_dir}...")
    generate_gallery_images(public_phantoms, gallery_dir, n_scenes=4)

    # -- Top-level README -------------------------------------------------
    _write_top_readme()

    print(f"\n{'=' * 60}")
    print(f"Done -- Ultrasound B-mode benchmark ready at {BENCHMARK_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
