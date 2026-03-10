#!/usr/bin/env python3
"""Generate B-mode ultrasound benchmark dataset.

Forward model (PSF convolution + speckle + log-compression):
    ideal(r)     = |s(r) * PSF(r)|            -- PSF-convolved reflectivity
    B(r)         = ideal(r) * speckle + noise  -- speckle + electronic noise
    y(r)         = 20 * log10(B(r) / B_max)   -- log-compressed B-mode

PSF model:
    Gaussian separable PSF with:
      sigma_lateral = f_number * lambda / 2
      sigma_axial   = n_cycles * lambda / 2
    where lambda = c / f (acoustic wavelength)

Ground truth phantoms (256x256):
    Tissue-mimicking phantoms with cysts, vessels, layered tissues,
    point targets for resolution assessment.

Mismatch parameters:
    speed_of_sound_error_pct : SoS error affecting focus (0-8%)
    attenuation_dB_cm_MHz    : tissue attenuation (0.3-1.2)
    speckle_density          : sub-resolution scatterers (5-50)
    snr_db                   : electronic SNR (20-40 dB)

Tiers:
    public : 12 samples (4 cyst, 4 tissue, 4 point/resolution)
    dev    : 20 samples (augmented, medium mismatch)
    hidden : 20 samples (adversarial, wide mismatch)

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
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

BENCHMARK_DIR = Path(__file__).resolve().parent
IMAGE_SIZE = 256

# ── Physics constants ────────────────────────────────────────────────────────

SPEED_OF_SOUND = 1540.0       # m/s in soft tissue
FREQUENCY_HZ   = 5.0e6        # 5 MHz centre frequency
F_NUMBER       = 1.5           # aperture f-number
N_CYCLES       = 3.5           # pulse length in cycles
PIXEL_SIZE_MM  = 0.15          # mm per pixel (0.15mm -> 38.4mm FOV)
DYNAMIC_RANGE  = 60.0          # dB dynamic range for log-compression

# ── Mismatch spec ranges per tier ────────────────────────────────────────────

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


# ── PSF generation ───────────────────────────────────────────────────────────

def make_psf(
    f_number: float = F_NUMBER,
    frequency_hz: float = FREQUENCY_HZ,
    c: float = SPEED_OF_SOUND,
    n_cycles: float = N_CYCLES,
    pixel_size_mm: float = PIXEL_SIZE_MM,
    sos_error_pct: float = 0.0,
) -> np.ndarray:
    """Generate a Gaussian PSF for B-mode ultrasound.

    The speed-of-sound error changes the effective wavelength, causing
    defocusing (wider PSF) and axial misregistration.

    Returns:
        psf: 2D array (K, K), normalized to sum=1
    """
    c_eff = c * (1.0 + sos_error_pct / 100.0)
    wavelength_mm = (c_eff / frequency_hz) * 1000.0

    sigma_lateral_mm = f_number * wavelength_mm / 2.0
    sigma_axial_mm = n_cycles * wavelength_mm / 2.0

    sigma_lat_px = sigma_lateral_mm / pixel_size_mm
    sigma_ax_px = sigma_axial_mm / pixel_size_mm

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


# ── Phantom generators ───────────────────────────────────────────────────────
# These create tissue reflectivity maps where:
#   background tissue ~ 1.0 (with sub-resolution scatterer texture)
#   bright structures (walls, calcifications) ~ 2.0-3.0
#   dark regions (cysts, vessels) ~ 0.0-0.1
#   The fine-scale texture simulates sub-resolution scatterer distribution.

def _add_scatterer_texture(
    phantom: np.ndarray,
    rng: np.random.Generator,
    scale: float = 0.25,
) -> np.ndarray:
    """Add fine-grained sub-resolution scatterer texture.

    This creates the underlying microstructure that produces speckle
    when convolved with the PSF. Uses a mix of:
    - Fine texture (sigma=1-2 px): individual scatterer clumps
    - Medium texture (sigma=5-8 px): tissue heterogeneity
    """
    size = phantom.shape[0]
    # Fine-scale scatterer distribution
    fine = rng.standard_normal((size, size)).astype(np.float64)
    fine = gaussian_filter(fine, sigma=1.5) * scale * 0.7
    # Medium-scale tissue heterogeneity
    medium = gaussian_filter(rng.standard_normal((size, size)), sigma=6.0) * scale * 0.3
    phantom = phantom + fine + medium
    return np.clip(phantom, 0.0, 4.0).astype(np.float32)


def _phantom_cyst(
    rng: np.random.Generator,
    n_cysts: int = 3,
    bright: bool = True,
    size: int = IMAGE_SIZE,
) -> np.ndarray:
    """Generate tissue phantom with cysts (bright or dark).

    Background: tissue ~1.0 with fine scatterer texture
    Bright cysts: hyperechoic (walls ~2.5, interior ~1.8)
    Dark cysts: anechoic (fluid-filled ~0.02, thin bright wall ~2.0)
    """
    phantom = np.ones((size, size), dtype=np.float64) * 1.0
    # Slow spatial variation in background tissue
    phantom += gaussian_filter(rng.standard_normal((size, size)),
                               sigma=30.0) * 0.15

    for _ in range(n_cysts):
        cy = rng.integers(size // 5, 4 * size // 5)
        cx = rng.integers(size // 5, 4 * size // 5)
        ry = rng.integers(15, 40)
        rx = rng.integers(15, 40)

        yy, xx = np.ogrid[:size, :size]
        dist = ((yy - cy) / max(ry, 1)) ** 2 + ((xx - cx) / max(rx, 1)) ** 2

        if bright:
            # Hyperechoic: bright wall + moderately bright interior
            wall_mask = (dist <= 1.15) & (dist > 0.85)
            interior_mask = dist <= 0.85
            phantom[wall_mask] = 2.5 + rng.uniform(-0.3, 0.3)
            phantom[interior_mask] = 1.8 + rng.uniform(-0.2, 0.2)
        else:
            # Anechoic: thin bright wall + very dark (fluid) interior
            wall_mask = (dist <= 1.12) & (dist > 0.92)
            interior_mask = dist <= 0.92
            phantom[wall_mask] = 2.0 + rng.uniform(-0.2, 0.2)
            phantom[interior_mask] = 0.02 + rng.uniform(0.0, 0.03)

    # Add 1-2 small vessels (thin dark lines ~3-4 px wide)
    n_vessels = rng.integers(1, 3)
    for _ in range(n_vessels):
        vy = rng.integers(size // 5, 4 * size // 5)
        vx_start = rng.integers(0, size // 4)
        vx_end = rng.integers(3 * size // 4, size)
        thickness = rng.integers(2, 4)
        angle = rng.uniform(-0.15, 0.15)
        for vx in range(vx_start, vx_end):
            y_pos = int(vy + (vx - vx_start) * angle)
            y_lo = max(0, y_pos - thickness)
            y_hi = min(size, y_pos + thickness)
            if y_lo < y_hi:
                phantom[y_lo:y_hi, vx] = 0.01

    return _add_scatterer_texture(
        np.clip(phantom, 0.0, 3.5).astype(np.float32), rng, scale=0.20)


def _phantom_layered_tissue(
    rng: np.random.Generator,
    size: int = IMAGE_SIZE,
) -> np.ndarray:
    """Generate layered tissue phantom (skin/fat/muscle/organ).

    Realistic tissue layer structure with varying echogenicity
    and bright interface reflections at layer boundaries.
    """
    phantom = np.zeros((size, size), dtype=np.float64)
    x_coords = np.arange(size, dtype=np.float64)

    n_layers = rng.integers(4, 7)
    boundaries = sorted(rng.integers(size // 8, 7 * size // 8, size=n_layers))

    # Layer reflectivities (tissue types)
    layer_vals = [
        rng.uniform(0.4, 0.6),   # skin/subcutaneous
        rng.uniform(0.8, 1.2),   # fat
        rng.uniform(1.2, 1.6),   # muscle
        rng.uniform(0.5, 0.8),   # fascia
        rng.uniform(0.9, 1.4),   # organ parenchyma
        rng.uniform(0.4, 0.7),   # deeper tissue
        rng.uniform(1.0, 1.2),   # background
    ]

    # Pre-compute undulations (one per boundary)
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
        val = layer_vals[i % len(layer_vals)]
        for col in range(size):
            row_end = max(prev_row, min(boundary + und[col], size - 1))
            phantom[prev_row:row_end, col] = val
        prev_row = boundary
    phantom[prev_row:, :] = layer_vals[-1]

    # Bright interface reflections (1-2 px wide bright lines)
    for i, boundary in enumerate(boundaries):
        und = undulations[i]
        bright_val = rng.uniform(2.0, 3.0)
        for col in range(size):
            row = boundary + und[col]
            if 1 <= row < size - 1:
                phantom[row - 1:row + 2, col] = bright_val

    return _add_scatterer_texture(
        np.clip(phantom, 0.0, 3.5).astype(np.float32), rng, scale=0.22)


def _phantom_point_targets(
    rng: np.random.Generator,
    size: int = IMAGE_SIZE,
) -> np.ndarray:
    """Generate point target / resolution phantom.

    Low-reflectivity background with grid of bright point reflectors
    and wire targets for resolution assessment.
    """
    phantom = np.ones((size, size), dtype=np.float64) * 0.25

    # Grid of point targets
    n_rows = rng.integers(4, 7)
    n_cols = rng.integers(5, 8)
    row_sp = size // (n_rows + 1)
    col_sp = size // (n_cols + 1)

    for r in range(1, n_rows + 1):
        for c in range(1, n_cols + 1):
            py = r * row_sp + rng.integers(-2, 3)
            px = c * col_sp + rng.integers(-2, 3)
            if 1 <= py < size - 1 and 1 <= px < size - 1:
                phantom[py, px] = 3.5
                # Small 3x3 halo
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        phantom[py + dy, px + dx] = max(
                            phantom[py + dy, px + dx], 1.8)

    # Wire targets (horizontal lines, 1 px thick)
    n_wires = rng.integers(2, 5)
    for _ in range(n_wires):
        wy = rng.integers(size // 6, 5 * size // 6)
        wx_s = rng.integers(size // 8, size // 3)
        wx_e = rng.integers(2 * size // 3, 7 * size // 8)
        phantom[wy, wx_s:wx_e] = 3.0

    return _add_scatterer_texture(
        np.clip(phantom, 0.0, 4.0).astype(np.float32), rng, scale=0.08)


def _phantom_complex_anatomy(
    rng: np.random.Generator,
    size: int = IMAGE_SIZE,
) -> np.ndarray:
    """Generate complex anatomy phantom for adversarial tier.

    Includes micro-calcifications, irregular masses, acoustic shadow effects,
    and heterogeneous tissue. Designed to be challenging for reconstruction.
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
        phantom[dist <= r ** 2] = rng.uniform(3.0, 3.5)

    # Irregular hypoechoic mass
    mass_cy = rng.integers(size // 4, 3 * size // 4)
    mass_cx = rng.integers(size // 4, 3 * size // 4)
    mass_r = rng.integers(12, 30)

    yy, xx = np.ogrid[:size, :size]
    angles = np.arctan2(yy - mass_cy, xx - mass_cx)
    n_lobes = rng.integers(3, 8)
    radial_mod = 1.0 + 0.35 * np.sin(n_lobes * angles + rng.uniform(0, 2 * np.pi))
    dist = np.sqrt((yy - mass_cy) ** 2 + (xx - mass_cx) ** 2) / (mass_r * radial_mod)
    mass_mask = dist <= 1.0
    phantom[mass_mask] = rng.uniform(0.05, 0.3)

    # Acoustic shadow below mass (attenuation effect)
    shadow_top = mass_cy + mass_r
    if shadow_top < size:
        sw = int(mass_r * 1.3)
        x_lo = max(0, mass_cx - sw)
        x_hi = min(size, mass_cx + sw)
        for row in range(shadow_top, size):
            decay = 0.4 * np.exp(-(row - shadow_top) / 40.0)
            phantom[row, x_lo:x_hi] *= (1.0 - decay)

    # Small bright lesion (calcified)
    lx = rng.integers(size // 5, 4 * size // 5)
    ly = rng.integers(size // 5, 4 * size // 5)
    lr = rng.integers(5, 12)
    dist2 = (yy - ly) ** 2 + (xx - lx) ** 2
    phantom[dist2 <= lr ** 2] = rng.uniform(2.5, 3.5)

    return np.clip(phantom, 0.0, 4.0).astype(np.float32)


# ── Scene name tables ────────────────────────────────────────────────────────

PUBLIC_SCENE_NAMES = [
    "bright_cyst_01", "bright_cyst_02", "dark_cyst_01", "dark_cyst_02",
    "layered_tissue_01", "layered_tissue_02", "layered_tissue_03",
    "layered_tissue_04",
    "point_target_01", "point_target_02", "point_target_03", "point_target_04",
]

DEV_SCENE_NAMES = [f"dev_tissue_{i:02d}" for i in range(20)]
HIDDEN_SCENE_NAMES = [f"hidden_adversarial_{i:02d}" for i in range(20)]


# ── Phantom generation per tier ──────────────────────────────────────────────

def generate_public_phantoms(
    rng: np.random.Generator,
) -> list[tuple[str, np.ndarray]]:
    """Generate 12 public phantoms: 4 cyst + 4 layered + 4 point targets."""
    phantoms = []
    for i in range(2):
        x = _phantom_cyst(rng, n_cysts=rng.integers(2, 5), bright=True)
        phantoms.append((PUBLIC_SCENE_NAMES[i], x))
    for i in range(2):
        x = _phantom_cyst(rng, n_cysts=rng.integers(2, 4), bright=False)
        phantoms.append((PUBLIC_SCENE_NAMES[2 + i], x))
    for i in range(4):
        x = _phantom_layered_tissue(rng)
        phantoms.append((PUBLIC_SCENE_NAMES[4 + i], x))
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
        lambda r: _phantom_cyst(r, n_cysts=r.integers(2, 5),
                                bright=r.random() > 0.5),
        lambda r: _phantom_layered_tissue(r),
        lambda r: _phantom_point_targets(r),
    ]
    for i in range(20):
        gen = generators[i % len(generators)]
        x = gen(rng)
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
        if i < 12:
            x = _phantom_complex_anatomy(rng)
        elif i < 16:
            x = _phantom_cyst(rng, n_cysts=rng.integers(4, 8),
                              bright=rng.random() > 0.5)
        else:
            x = _phantom_layered_tissue(rng)
        k = rng.integers(0, 4)
        if k > 0:
            x = np.rot90(x, k).copy()
        if rng.random() < 0.7:
            x = np.fliplr(x).copy()
        if rng.random() < 0.5:
            x = np.flipud(x).copy()
        phantoms.append((HIDDEN_SCENE_NAMES[i], x))
    return phantoms


# ── Forward model ────────────────────────────────────────────────────────────

def apply_psf(x_true: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """Convolve tissue reflectivity with PSF to get ideal B-mode envelope."""
    convolved = fftconvolve(x_true.astype(np.float64),
                            psf.astype(np.float64), mode="same")
    return np.abs(convolved).astype(np.float32)


def apply_attenuation(
    bmode: np.ndarray,
    attenuation_dB_cm_MHz: float,
    frequency_MHz: float = 5.0,
    pixel_size_mm: float = PIXEL_SIZE_MM,
) -> np.ndarray:
    """Apply depth-dependent tissue attenuation.

    Round-trip attenuation: signal travels down and back, so 2x.
    """
    H = bmode.shape[0]
    depth_cm = np.arange(H, dtype=np.float64) * pixel_size_mm / 10.0
    # Round-trip attenuation
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

    Speckle: Rayleigh-distributed multiplicative noise modelling
    coherent interference from sub-resolution scatterers.
    speckle_density = N scatterers per resolution cell.
    Speckle contrast = 1/sqrt(N).

    Electronic noise: additive Gaussian at given SNR.
    """
    N = max(speckle_density, 1)
    # Sum of N random phasors: I/Q components per scatterer
    H, W = bmode.shape
    # Accumulate I/Q for N scatterers
    I_sum = np.zeros((H, W), dtype=np.float64)
    Q_sum = np.zeros((H, W), dtype=np.float64)
    for _ in range(N):
        phase = rng.uniform(0, 2 * np.pi, (H, W))
        amp = rng.rayleigh(1.0, (H, W))
        I_sum += amp * np.cos(phase)
        Q_sum += amp * np.sin(phase)
    # Envelope (Rayleigh distributed when N is large)
    speckle_envelope = np.sqrt(I_sum ** 2 + Q_sum ** 2) / N

    # Apply multiplicative speckle
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

    Output in [0, 1] where 0 = bottom of dynamic range, 1 = peak.
    """
    bmode_pos = np.maximum(bmode, 1e-10)
    b_max = bmode_pos.max()
    y_dB = 20.0 * np.log10(bmode_pos / b_max)
    y_clipped = np.clip(y_dB, -dynamic_range_dB, 0.0)
    y_norm = (y_clipped + dynamic_range_dB) / dynamic_range_dB
    return y_norm.astype(np.float32)


def forward_model(
    x_true: np.ndarray,
    psf: np.ndarray,
    attenuation_dB_cm_MHz: float,
    speckle_density: int,
    snr_db: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Complete B-mode ultrasound forward model.

    Returns:
        bmode_ideal:    clean B-mode (PSF + attenuation), log-compressed [0,1]
        bmode_measured: noisy B-mode (+ speckle + noise), log-compressed [0,1]
    """
    # PSF convolution
    envelope = apply_psf(x_true, psf)
    # Depth-dependent attenuation
    envelope_atten = apply_attenuation(envelope, attenuation_dB_cm_MHz)
    # Ideal (no speckle, no noise)
    bmode_ideal = log_compress(envelope_atten)
    # Speckle + electronic noise
    bmode_noisy = apply_speckle_noise(envelope_atten, speckle_density,
                                       snr_db, rng)
    bmode_measured = log_compress(bmode_noisy)
    return bmode_ideal, bmode_measured


# ── CPU Reconstruction: Wiener deconvolution ─────────────────────────────────

def reconstruct_wiener(
    bmode_measured: np.ndarray,
    psf: np.ndarray,
    noise_variance: float = 0.005,
) -> np.ndarray:
    """Wiener deconvolution reconstruction from log-compressed noisy B-mode.

    Inverts log-compression, applies Wiener filter in frequency domain,
    then normalises to [0,1].
    """
    # Invert log-compression to linear domain
    bmode_linear = 10.0 ** ((bmode_measured * DYNAMIC_RANGE - DYNAMIC_RANGE) / 20.0)

    H, W = bmode_linear.shape
    ph, pw = psf.shape

    # Zero-pad PSF to image size, centred at origin
    psf_padded = np.zeros((H, W), dtype=np.float64)
    py = (H - ph) // 2
    px = (W - pw) // 2
    psf_padded[py:py + ph, px:px + pw] = psf
    psf_shifted = np.roll(np.roll(psf_padded, -ph // 2, axis=0),
                          -pw // 2, axis=1)

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


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_psnr(gt: np.ndarray, recon: np.ndarray) -> float:
    mse = float(np.mean((gt.astype(np.float64) - recon.astype(np.float64)) ** 2))
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


# ── Image helpers ────────────────────────────────────────────────────────────

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


# ── Mismatch sampling ───────────────────────────────────────────────────────

def sample_mismatch(rng: np.random.Generator, spec: dict) -> dict:
    result = {}
    for k, v in spec.items():
        lo, hi = v["min"], v["max"]
        if isinstance(lo, int) and isinstance(hi, int):
            result[k] = int(rng.integers(lo, hi + 1))
        else:
            result[k] = float(rng.uniform(lo, hi))
    return result


# ── Tier generator ───────────────────────────────────────────────────────────

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
            f"(PSF convolution + speckle + log-compression)")
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
        })

        for idx, (scene_name, x_true) in enumerate(phantoms):
            key = f"sample_{idx:02d}"
            mis = sample_mismatch(rng, spec_ranges)
            true_specs[key] = {**mis, "scene": scene_name}

            psf = make_psf(sos_error_pct=mis["speed_of_sound_error_pct"])
            bmode_ideal, bmode_measured = forward_model(
                x_true, psf,
                attenuation_dB_cm_MHz=mis["attenuation_dB_cm_MHz"],
                speckle_density=mis["speckle_density"],
                snr_db=mis["snr_db"],
                rng=rng,
            )

            recon = reconstruct_wiener(bmode_measured, psf, noise_variance=0.005)
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
            grp.create_dataset("psf", data=psf, compression="gzip")

            grp.attrs["metadata"] = json.dumps({
                "scene": scene_name,
                "shape": list(x_true.shape),
                "psf_shape": list(psf.shape),
                "wiener_psnr_dB": round(psnr, 2),
                "wiener_ssim": round(ssim, 4),
            })
            grp.attrs["spec_ranges"] = json.dumps(spec_ranges)
            grp.attrs["true_spec"] = json.dumps({**mis, "scene": scene_name})

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


# ── README writers ───────────────────────────────────────────────────────────

def _write_tier_readme(tier: str, tier_dir: Path, rows: list) -> None:
    spec = SPEC[tier]
    if tier == "public":
        access = "Full (GT + true spec + ideal B-mode)"
        source = ("Synthetic tissue-mimicking phantoms: cysts, layered tissue, "
                  "point targets (12 samples)")
    elif tier == "dev":
        access = "Blind (measured B-mode + spec ranges only)"
        source = "Augmented synthetic phantoms -- mixed types (20 samples)"
    else:
        access = "Server-only"
        source = ("Adversarial synthetic phantoms -- complex anatomy, "
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
        "| `psf` | (K, K) | float32 | Point spread function used |\n",
    ]

    with open(tier_dir / "README.md", "w") as f:
        f.writelines(lines)


def _write_top_readme() -> None:
    txt = """# Ultrasound B-Mode Benchmark Dataset

## Overview

B-mode ultrasound imaging benchmark with PSF convolution, speckle noise,
and log-compression forward model. Uses synthetic tissue-mimicking phantoms
with realistic tissue reflectivity values and sub-resolution scatterer texture.

## Forward Model

```
ideal(r)     = |s(r) * PSF(r)|            -- PSF-convolved reflectivity
B(r)         = ideal(r) * speckle + noise  -- speckle + electronic noise
y(r)         = 20 * log10(B(r) / B_max)   -- log-compressed B-mode

where:
  s(r)     -- tissue reflectivity map (ground truth)
  PSF(r)   -- Gaussian point spread function
  speckle  -- Rayleigh-distributed multiplicative noise (N scatterers/cell)
  noise    -- additive Gaussian electronic noise
```

## PSF Model

Gaussian separable PSF:
  - sigma_lateral = f_number * lambda / 2
  - sigma_axial = n_cycles * lambda / 2
  - lambda = c / f (acoustic wavelength)

| Parameter | Value |
|-----------|-------|
| Frequency | 5 MHz |
| Speed of sound | 1540 m/s |
| F-number | 1.5 |
| Pulse cycles | 3.5 |
| Pixel size | 0.15 mm |
| Dynamic range | 60 dB |

## Mismatch Parameters (ThetaSpace)

| Knob | Symbol | Description | Public | Dev | Hidden |
|------|--------|-------------|--------|-----|--------|
| `speed_of_sound_error_pct` | SoS err | Focus error from SoS mismatch | 0-3% | 0-5% | 0-8% |
| `attenuation_dB_cm_MHz` | alpha | Tissue attenuation | 0.3-0.7 | 0.3-0.9 | 0.3-1.2 |
| `speckle_density` | N_s | Scatterers per resolution cell | 10-25 | 8-35 | 5-50 |
| `snr_db` | SNR | Electronic SNR | 30-40 dB | 25-38 dB | 20-35 dB |

## Phantom Types

| Type | Description | Tier |
|------|-------------|------|
| Bright cysts | Hyperechoic cysts with bright walls, tissue background | Public |
| Dark cysts | Anechoic fluid-filled cysts, vessel structures | Public |
| Layered tissue | Skin/fat/muscle/organ layers with interface reflections | Public, Dev |
| Point targets | Grid of point reflectors, wire phantoms | Public |
| Complex anatomy | Micro-calcifications, irregular masses, acoustic shadows | Hidden |

## Dataset Structure

```
ultrasound/
+-- README.md
+-- generate_dataset.py
+-- public/    12 samples (4 cyst + 4 layered + 4 point target)
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
+-- psf (K, K) float32               -- Point spread function used
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

- Perrot et al. (2021) "So you think you can DAS?" IEEE TUFFC 68(2):355-381.
- Matrone et al. (2015) "DMAS" IEEE TUFFC 62(3):537-545.
- Gasse et al. (2017) "IQ-Net" IEEE TUFFC 64(10):1535-1543.
- Jensen (1996) Field II ultrasound simulation program. MPC 4:351-353.
"""
    with open(BENCHMARK_DIR / "README.md", "w") as f:
        f.write(txt)


# ── Gallery image generation ─────────────────────────────────────────────────

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
    #   scene_00: bright cyst (index 0)
    #   scene_01: dark cyst (index 2)
    #   scene_02: layered tissue (index 4)
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
        psf1 = make_psf(sos_error_pct=1.0)
        _, bm1 = forward_model(x_true, psf1, 0.5, 15, 35.0, rng)
        _save_png(bm1, scene_dir / "measurement_I.png")

        # Measurement II: heavier degradation
        psf2 = make_psf(sos_error_pct=4.0)
        _, bm2 = forward_model(x_true, psf2, 0.8, 8, 25.0, rng)
        _save_png(bm2, scene_dir / "measurement_II.png")

        # Recon I: Wiener from mild
        r1 = reconstruct_wiener(bm1, psf1, 0.005)
        _save_png(r1, scene_dir / "recon_I.png")

        # Recon II: Wiener from heavy
        r2 = reconstruct_wiener(bm2, psf2, 0.01)
        _save_png(r2, scene_dir / "recon_II.png")

        # Recon III: Wiener with mismatched PSF
        psf3 = make_psf(sos_error_pct=6.0)
        r3 = reconstruct_wiener(bm1, psf3, 0.005)
        _save_png(r3, scene_dir / "recon_III.png")

        print(f"  Gallery scene_{si:02d} ({scene_name}): 6 images saved")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Ultrasound B-Mode Benchmark Dataset Generator")
    print("=" * 60)
    print(f"Output: {BENCHMARK_DIR}\n")

    # ── Public tier (12 samples) ─────────────────────────────────────────
    print("Generating public tier (12 samples)...")
    rng_pub = np.random.default_rng(1000)
    public_phantoms = generate_public_phantoms(rng_pub)
    generate_tier("public", public_phantoms, base_seed=1000)

    # ── Dev tier (20 samples) ────────────────────────────────────────────
    print("\nGenerating dev tier (20 samples, augmented)...")
    rng_dev = np.random.default_rng(2000)
    dev_phantoms = generate_dev_phantoms(rng_dev)
    generate_tier("dev", dev_phantoms, base_seed=2000)

    # ── Hidden tier (20 samples) ─────────────────────────────────────────
    print("\nGenerating hidden tier (20 samples, adversarial)...")
    rng_hid = np.random.default_rng(3000)
    hidden_phantoms = generate_hidden_phantoms(rng_hid)
    generate_tier("hidden", hidden_phantoms, base_seed=3000)

    # ── Gallery images ───────────────────────────────────────────────────
    gallery_dir = (Path(__file__).resolve().parent.parent.parent.parent /
                   "platform" / "pwm_platform" / "static" / "img" /
                   "benchmark_gallery" / "ultrasound")
    print(f"\nGenerating gallery images at {gallery_dir}...")
    generate_gallery_images(public_phantoms, gallery_dir, n_scenes=4)

    # ── Top-level README ─────────────────────────────────────────────────
    _write_top_readme()

    print(f"\n{'=' * 60}")
    print(f"Done -- Ultrasound B-mode benchmark ready at {BENCHMARK_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
