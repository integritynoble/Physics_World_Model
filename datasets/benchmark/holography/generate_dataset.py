#!/usr/bin/env python3
"""Generate Digital Holography benchmark dataset.

Forward model (off-axis digital holographic microscopy):
    O(x,y) = F^{-1}{ H_prop(kx,ky) * F{ x_true(x,y) } }
    y(x,y) = |R(x,y) + O(x,y)|^2
           = |R|^2 + |O|^2 + R*.O + R.O*

where:
    x_true      -- complex-valued ground truth (amplitude * exp(i*phase))
    O           -- object wave after free-space propagation
    R           -- plane-wave reference beam (tilted for off-axis)
    H_prop      -- angular spectrum propagation kernel
    y           -- recorded hologram intensity (measurement)

Angular spectrum propagation kernel:
    H_prop(kx,ky) = exp(i * 2*pi*d/lambda * sqrt(1 - (lambda*kx)^2 - (lambda*ky)^2))

Mismatch parameters (ThetaSpace):
    propagation_distance_error : delta_d relative error on propagation distance
    wavelength_error           : delta_lambda relative error on wavelength
    reference_tilt_error       : delta_tilt error on reference beam tilt angle (rad)
    noise_level                : sigma for additive Gaussian noise on hologram

Ground truth phantoms (256x256, complex-valued):
    Phase objects   : cells (smooth phase profiles), transparent specimens
    Amplitude objects: resolution targets, patterned surfaces
    Mixed           : complex objects with both amplitude and phase structure

Tiers:
    public : 12 samples (seeds from 0)
    dev    : 20 samples (seeds from 10000)
    hidden : 20 samples (seeds from 20000)

CPU baseline: Angular spectrum back-propagation. Expected ~22-28 dB.

Usage:
    cd datasets/benchmark/holography
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

BENCHMARK_DIR = Path(__file__).resolve().parent
IMAGE_SIZE = 256

# -- Physics constants --------------------------------------------------------

WAVELENGTH_M = 532e-9       # green laser wavelength (532 nm)
PIXEL_PITCH_M = 3.45e-6     # typical camera pixel pitch (3.45 um)
PROP_DISTANCE_M = 0.5e-3    # propagation distance (0.5 mm) -- short for DHM
REF_TILT_RAD = 0.03         # off-axis reference tilt angle (radians)

# -- Mismatch spec ranges per tier -------------------------------------------

SPEC = {
    "public": {
        "propagation_distance_error": {"min": -0.02, "max": 0.02, "unit": "relative"},
        "wavelength_error":           {"min": -0.01, "max": 0.01, "unit": "relative"},
        "reference_tilt_error":       {"min": -0.005, "max": 0.005, "unit": "radians"},
        "noise_level":                {"min": 0.01,  "max": 0.05, "unit": "fraction"},
    },
    "dev": {
        "propagation_distance_error": {"min": -0.05, "max": 0.05, "unit": "relative"},
        "wavelength_error":           {"min": -0.02, "max": 0.02, "unit": "relative"},
        "reference_tilt_error":       {"min": -0.01, "max": 0.01, "unit": "radians"},
        "noise_level":                {"min": 0.02,  "max": 0.08, "unit": "fraction"},
    },
    "hidden": {
        "propagation_distance_error": {"min": -0.10, "max": 0.10, "unit": "relative"},
        "wavelength_error":           {"min": -0.05, "max": 0.05, "unit": "relative"},
        "reference_tilt_error":       {"min": -0.02, "max": 0.02, "unit": "radians"},
        "noise_level":                {"min": 0.05,  "max": 0.15, "unit": "fraction"},
    },
}


# =============================================================================
# Angular Spectrum Propagation
# =============================================================================

def angular_spectrum_kernel(
    N: int,
    wavelength: float,
    pixel_pitch: float,
    distance: float,
) -> np.ndarray:
    """Compute the angular spectrum propagation transfer function H_prop.

    Parameters
    ----------
    N : int
        Grid size (N x N).
    wavelength : float
        Wavelength in metres.
    pixel_pitch : float
        Pixel pitch in metres.
    distance : float
        Propagation distance in metres.

    Returns
    -------
    H_prop : np.ndarray, complex128, shape (N, N)
        The angular spectrum propagation kernel in frequency domain.
    """
    fx = np.fft.fftfreq(N, d=pixel_pitch)
    FX, FY = np.meshgrid(fx, fx)
    # Spatial frequency squared
    fsq = FX**2 + FY**2
    # Evanescent wave cutoff
    k = 1.0 / wavelength
    propagating = fsq < k**2
    # Transfer function
    kz_sq = np.where(propagating, k**2 - fsq, 0.0)
    kz = np.sqrt(np.maximum(kz_sq, 0.0))
    H = np.where(propagating, np.exp(1j * 2 * np.pi * distance * kz), 0.0 + 0j)
    return H.astype(np.complex128)


def propagate_field(
    field: np.ndarray,
    wavelength: float,
    pixel_pitch: float,
    distance: float,
) -> np.ndarray:
    """Propagate a complex field using the angular spectrum method.

    Parameters
    ----------
    field : np.ndarray, complex, shape (N, N)
        Input complex field.
    wavelength, pixel_pitch, distance : float
        Optical parameters.

    Returns
    -------
    propagated : np.ndarray, complex128, shape (N, N)
    """
    N = field.shape[0]
    H = angular_spectrum_kernel(N, wavelength, pixel_pitch, distance)
    F_field = np.fft.fft2(field)
    F_prop = F_field * H
    return np.fft.ifft2(F_prop)


def reference_wave(
    N: int,
    pixel_pitch: float,
    wavelength: float,
    tilt_rad: float,
) -> np.ndarray:
    """Generate a tilted plane wave reference beam.

    The tilt is applied along the x-axis to create off-axis fringes.

    Parameters
    ----------
    N : int
        Grid size.
    pixel_pitch : float
        Pixel pitch in metres.
    wavelength : float
        Wavelength in metres.
    tilt_rad : float
        Tilt angle in radians.

    Returns
    -------
    R : np.ndarray, complex128, shape (N, N)
    """
    x = np.arange(N) * pixel_pitch
    # Tilt introduces a linear phase ramp
    kx_tilt = 2 * np.pi * np.sin(tilt_rad) / wavelength
    phase_ramp = kx_tilt * x[np.newaxis, :]
    return np.exp(1j * phase_ramp).astype(np.complex128)


# =============================================================================
# Forward Model: Hologram Formation
# =============================================================================

def forward_hologram(
    x_true: np.ndarray,
    wavelength: float = WAVELENGTH_M,
    pixel_pitch: float = PIXEL_PITCH_M,
    distance: float = PROP_DISTANCE_M,
    tilt_rad: float = REF_TILT_RAD,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate hologram formation.

    Parameters
    ----------
    x_true : np.ndarray, complex128, shape (N, N)
        Complex-valued ground truth object.
    wavelength, pixel_pitch, distance, tilt_rad : float
        Optical configuration.

    Returns
    -------
    hologram : np.ndarray, float64, shape (N, N)
        Recorded intensity pattern |R + O|^2.
    H_ideal : np.ndarray, complex128, shape (N, N)
        The ideal propagation kernel used (for reconstruction reference).
    """
    N = x_true.shape[0]
    # Object wave after propagation
    O = propagate_field(x_true, wavelength, pixel_pitch, distance)
    # Reference beam
    R = reference_wave(N, pixel_pitch, wavelength, tilt_rad)
    # Hologram = |R + O|^2
    total = R + O
    hologram = np.abs(total) ** 2
    # Ideal propagation kernel (for back-propagation)
    H_ideal = angular_spectrum_kernel(N, wavelength, pixel_pitch, distance)
    return hologram.astype(np.float64), H_ideal


def forward_hologram_mismatched(
    x_true: np.ndarray,
    mismatch: dict,
    rng: np.random.Generator,
    wavelength: float = WAVELENGTH_M,
    pixel_pitch: float = PIXEL_PITCH_M,
    distance: float = PROP_DISTANCE_M,
    tilt_rad: float = REF_TILT_RAD,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate hologram with mismatch and noise.

    The hologram is formed with the TRUE physical parameters, but the
    H_ideal kernel returned uses the NOMINAL (mismatched) parameters.
    This creates the calibration mismatch that algorithms must overcome.

    Parameters
    ----------
    x_true : complex array (N, N)
        Ground truth complex object.
    mismatch : dict
        Keys: propagation_distance_error, wavelength_error,
              reference_tilt_error, noise_level.
    rng : np.random.Generator
        For reproducibility.

    Returns
    -------
    y : float64 array (N, N)
        Noisy hologram with mismatched forward model.
    H_ideal : complex128 array (N, N)
        Propagation kernel using NOMINAL parameters (contains mismatch).
    """
    N = x_true.shape[0]

    # TRUE physical parameters (what nature uses)
    d_true = distance * (1.0 + mismatch["propagation_distance_error"])
    lam_true = wavelength * (1.0 + mismatch["wavelength_error"])
    tilt_true = tilt_rad + mismatch["reference_tilt_error"]

    # Object wave with TRUE parameters
    O = propagate_field(x_true, lam_true, pixel_pitch, d_true)
    R = reference_wave(N, pixel_pitch, lam_true, tilt_true)

    # Hologram intensity
    total = R + O
    hologram = np.abs(total) ** 2

    # Add noise
    noise_level = mismatch["noise_level"]
    holo_max = hologram.max() + 1e-12
    noise = rng.normal(0.0, noise_level * holo_max, hologram.shape)
    y = np.maximum(hologram + noise, 0.0)

    # NOMINAL propagation kernel (what the algorithm thinks the physics is)
    # Uses the nominal (uncorrected) parameters -- the mismatch
    H_ideal = angular_spectrum_kernel(N, wavelength, pixel_pitch, distance)

    return y.astype(np.float64), H_ideal


# =============================================================================
# CPU Baseline: Angular Spectrum Back-Propagation
# =============================================================================

def baseline_reconstruct(
    y: np.ndarray,
    H_ideal: np.ndarray,
    wavelength: float = WAVELENGTH_M,
    pixel_pitch: float = PIXEL_PITCH_M,
    tilt_rad: float = REF_TILT_RAD,
) -> np.ndarray:
    """CPU baseline reconstruction using angular spectrum back-propagation.

    Standard off-axis holography reconstruction:
    1. Multiply hologram by conjugate reference R* to demodulate
       y * R* = R* + R*|O|^2 + O + O*R*^2
       After demodulation, the +1 order (containing O) sits at DC,
       while other terms are shifted to +/-fx_tilt and +/-2*fx_tilt.
    2. Low-pass filter in Fourier domain to isolate O at DC
    3. Back-propagate using conjugate of H_prop to recover x_true

    Parameters
    ----------
    y : float64 array (N, N)
        Hologram measurement.
    H_ideal : complex128 array (N, N)
        Nominal propagation kernel.
    wavelength, pixel_pitch, tilt_rad : float
        Nominal optical parameters.

    Returns
    -------
    recon : complex128 array (N, N)
        Reconstructed complex field.
    """
    N = y.shape[0]
    # Step 1: Demodulate by multiplying with conjugate reference
    # This shifts the +1 order (R*.O) to baseband (DC)
    R_conj = np.conj(reference_wave(N, pixel_pitch, wavelength, tilt_rad))
    demod = y * R_conj

    # Step 2: Low-pass filter in Fourier domain to isolate O at DC
    # After demodulation, the DC/autocorrelation terms are at fx_tilt,
    # and the twin image is at 2*fx_tilt. LP filter keeps only O.
    F_demod = np.fft.fft2(demod)
    fx = np.fft.fftfreq(N)
    FX, FY = np.meshgrid(fx, fx)
    # Hard circular LP with smooth (Butterworth-like) rolloff
    kx_tilt = np.sin(tilt_rad) / wavelength
    fx_tilt = kx_tilt * pixel_pitch  # normalized tilt frequency
    # Cutoff at 55% of tilt frequency to cleanly separate the +1 order
    # from the DC/autocorrelation term (which is at fx_tilt after demod)
    cutoff = 0.55 * fx_tilt
    r_freq = np.sqrt(FX ** 2 + FY ** 2)
    # Butterworth order-6 LP filter for smooth rolloff
    bp_filter = 1.0 / (1.0 + (r_freq / cutoff) ** 12)
    F_filtered = F_demod * bp_filter

    # Step 3: Back-propagate using conjugate kernel
    H_back = np.conj(H_ideal)
    F_recon = F_filtered * H_back
    recon = np.fft.ifft2(F_recon)
    return recon


# =============================================================================
# Phantom Generators
# =============================================================================

def _smooth_random_field(N: int, rng: np.random.Generator, sigma: float = 15.0) -> np.ndarray:
    """Generate a smooth random field in [0, 1] via Gaussian-filtered noise."""
    raw = rng.standard_normal((N, N))
    smooth = gaussian_filter(raw, sigma=sigma)
    lo, hi = smooth.min(), smooth.max()
    return ((smooth - lo) / (hi - lo + 1e-12)).astype(np.float64)


def _make_circle(N: int, cy: float, cx: float, r: float) -> np.ndarray:
    """Binary circle mask."""
    yy, xx = np.mgrid[0:N, 0:N]
    return ((yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2).astype(np.float64)


def _make_ellipse(N: int, cy: float, cx: float, ry: float, rx: float,
                  angle: float = 0.0) -> np.ndarray:
    """Binary ellipse mask with optional rotation."""
    yy, xx = np.mgrid[0:N, 0:N]
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    dy = yy - cy
    dx = xx - cx
    yr = dy * cos_a + dx * sin_a
    xr = -dy * sin_a + dx * cos_a
    return ((yr / ry) ** 2 + (xr / rx) ** 2 <= 1.0).astype(np.float64)


def generate_cell_phantom(
    N: int, rng: np.random.Generator, complexity: str = "simple"
) -> np.ndarray:
    """Phase object: biological cell (transparent, phase-only or weak amplitude).

    Returns complex-valued ground truth with:
    - Amplitude close to 1 (transparent)
    - Phase in [0, 2*pi] encoding optical path length differences
    """
    phase = np.zeros((N, N), dtype=np.float64)
    amplitude = np.ones((N, N), dtype=np.float64)

    if complexity == "simple":
        # Single cell: smooth phase bump
        n_cells = int(rng.integers(1, 4))
    elif complexity == "medium":
        n_cells = int(rng.integers(3, 8))
    else:
        n_cells = int(rng.integers(6, 15))

    for _ in range(n_cells):
        cy = float(rng.uniform(N * 0.15, N * 0.85))
        cx = float(rng.uniform(N * 0.15, N * 0.85))
        ry = float(rng.uniform(N * 0.05, N * 0.18))
        rx = float(rng.uniform(N * 0.05, N * 0.18))
        angle = float(rng.uniform(0, np.pi))
        mask = _make_ellipse(N, cy, cx, ry, rx, angle)

        # Smooth internal phase structure (organelles, nucleus)
        # Realistic DHM phase shifts: 0.1-0.8 rad (thin transparent specimens)
        internal = _smooth_random_field(N, rng, sigma=float(rng.uniform(8, 25)))
        max_phase = float(rng.uniform(0.1, 0.8))  # radians
        cell_phase = mask * internal * max_phase

        # Sub-structures (nucleus with higher phase)
        if rng.random() > 0.3:
            nucleus_r = float(rng.uniform(ry * 0.2, ry * 0.5))
            nucleus_mask = _make_circle(N, cy, cx, nucleus_r)
            nucleus_phase = float(rng.uniform(0.1, 0.5))
            cell_phase += nucleus_mask * nucleus_phase

        phase += cell_phase
        # Slight amplitude modulation (absorption)
        amplitude -= mask * float(rng.uniform(0.0, 0.08))

    # Background phase texture (slight wavefront aberrations)
    bg_phase = _smooth_random_field(N, rng, sigma=40.0) * float(rng.uniform(0.02, 0.1))
    phase += bg_phase

    amplitude = np.clip(amplitude, 0.3, 1.0)
    return (amplitude * np.exp(1j * phase)).astype(np.complex128)


def generate_resolution_target(N: int, rng: np.random.Generator) -> np.ndarray:
    """Amplitude object: USAF-like resolution target.

    Returns complex-valued ground truth with:
    - Binary amplitude pattern (0 or 1)
    - Zero or weak phase
    """
    amplitude = np.ones((N, N), dtype=np.float64) * 0.4  # moderate background

    # Create bar groups at different scales
    n_groups = int(rng.integers(3, 7))
    for g in range(n_groups):
        # Bar width decreases with group number
        bar_width = max(2, int(N / (4 + g * 3)))
        n_bars = int(rng.integers(3, 6))
        # Random position
        y0 = int(rng.integers(10, N - 50))
        x0 = int(rng.integers(10, N - 50))

        # Horizontal bars
        for b in range(n_bars):
            y_start = y0 + b * 2 * bar_width
            y_end = min(y_start + bar_width, N)
            x_end = min(x0 + bar_width * n_bars, N)
            if y_end <= N and x_end <= N:
                amplitude[y_start:y_end, x0:x_end] = 0.9

        # Vertical bars (offset)
        vx0 = x0 + bar_width * n_bars + bar_width
        for b in range(n_bars):
            x_start = vx0 + b * 2 * bar_width
            x_end = min(x_start + bar_width, N)
            y_end = min(y0 + bar_width * n_bars, N)
            if x_end <= N and y_end <= N:
                amplitude[y0:y_end, x_start:x_end] = 0.9

    # Add concentric rings (Siemens star-like)
    cy, cx = N // 2, N // 2
    yy, xx = np.mgrid[0:N, 0:N]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    theta = np.arctan2(yy - cy, xx - cx)
    n_spokes = int(rng.integers(8, 24))
    ring_pattern = (np.sin(n_spokes * theta) > 0).astype(np.float64)
    ring_mask = (r > N * 0.05) & (r < N * 0.2)
    amplitude[ring_mask] = ring_pattern[ring_mask] * 0.5 + 0.4

    # Weak phase (surface roughness)
    phase = _smooth_random_field(N, rng, sigma=20.0) * float(rng.uniform(0.02, 0.1))
    return (amplitude * np.exp(1j * phase)).astype(np.complex128)


def generate_patterned_surface(N: int, rng: np.random.Generator) -> np.ndarray:
    """Mixed amplitude+phase object: patterned micro-surface.

    Returns complex-valued ground truth with structured amplitude and phase.
    Simulates etched microstructures, MEMS surfaces, or diffractive elements.
    """
    # Phase: etched step pattern (multiple height levels)
    phase = np.zeros((N, N), dtype=np.float64)
    amplitude = np.ones((N, N), dtype=np.float64) * 0.7

    # Grid pattern (like MEMS or photonic crystal)
    period_y = int(rng.integers(8, 30))
    period_x = int(rng.integers(8, 30))
    yy, xx = np.mgrid[0:N, 0:N]
    grid = ((yy % period_y < period_y // 2) ^ (xx % period_x < period_x // 2)).astype(np.float64)
    step_height = float(rng.uniform(0.2, 0.8))  # radians (realistic etch depth)
    phase += grid * step_height

    # Random etch defects
    n_defects = int(rng.integers(2, 8))
    for _ in range(n_defects):
        cy = float(rng.uniform(20, N - 20))
        cx = float(rng.uniform(20, N - 20))
        r = float(rng.uniform(3, 15))
        defect = _make_circle(N, cy, cx, r)
        phase += defect * float(rng.uniform(-0.4, 0.4))
        amplitude -= defect * float(rng.uniform(0.0, 0.15))

    # Smooth wavefront curvature (lens-like)
    curvature = float(rng.uniform(0.0, 0.2))
    r_norm = np.sqrt(((yy - N / 2) / N) ** 2 + ((xx - N / 2) / N) ** 2)
    phase += curvature * r_norm ** 2 * np.pi

    amplitude = np.clip(amplitude, 0.3, 1.0)
    return (amplitude * np.exp(1j * phase)).astype(np.complex128)


def generate_transparent_specimen(N: int, rng: np.random.Generator) -> np.ndarray:
    """Pure phase object: transparent specimen (e.g., unstained tissue section).

    Constant amplitude = 1, all information is in the phase channel.
    Simulates optical path length variations from refractive index differences.
    """
    phase = np.zeros((N, N), dtype=np.float64)

    # Layered tissue-like structure
    n_layers = int(rng.integers(3, 8))
    layer_heights = np.sort(rng.uniform(0.1, 0.9, n_layers)) * N
    for i in range(n_layers - 1):
        y_lo = int(layer_heights[i])
        y_hi = int(layer_heights[i + 1])
        if y_hi <= y_lo:
            continue
        # Each layer has a different refractive index (phase offset)
        # Realistic: 0.05-0.4 rad per layer for thin tissue
        layer_phase = float(rng.uniform(0.05, 0.4))
        # Wavy boundary
        boundary_wave = np.sin(np.linspace(0, float(rng.uniform(2, 8)) * np.pi, N))
        boundary_amp = float(rng.uniform(2, 8))
        for y in range(y_lo, min(y_hi, N)):
            boundary_offset = int(boundary_amp * boundary_wave[min(y, N - 1)])
            row_phase = np.roll(np.ones(N) * layer_phase, boundary_offset)
            phase[y, :] += row_phase

    # Inclusions (vacuoles, organelles)
    n_inclusions = int(rng.integers(3, 12))
    for _ in range(n_inclusions):
        cy = float(rng.uniform(20, N - 20))
        cx = float(rng.uniform(20, N - 20))
        r = float(rng.uniform(3, 20))
        inclusion = _make_circle(N, cy, cx, r)
        phase += inclusion * float(rng.uniform(-0.3, 0.3))

    # Smooth background aberration
    phase += _smooth_random_field(N, rng, sigma=50.0) * float(rng.uniform(0.02, 0.08))

    amplitude = np.ones((N, N), dtype=np.float64)
    return (amplitude * np.exp(1j * phase)).astype(np.complex128)


def generate_microbead_phantom(N: int, rng: np.random.Generator) -> np.ndarray:
    """Mixed object: polystyrene microbeads (calibration standard).

    Known spherical phase objects with well-defined refractive index.
    Useful for quantitative phase validation.
    """
    phase = np.zeros((N, N), dtype=np.float64)
    amplitude = np.ones((N, N), dtype=np.float64)

    n_beads = int(rng.integers(5, 25))
    for _ in range(n_beads):
        cy = float(rng.uniform(20, N - 20))
        cx = float(rng.uniform(20, N - 20))
        r = float(rng.uniform(5, 25))

        yy, xx = np.mgrid[0:N, 0:N]
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        mask = dist < r

        # Spherical phase profile: phase = (2*pi/lambda) * delta_n * thickness
        # thickness = 2 * sqrt(r^2 - rho^2) for a sphere
        rho = dist[mask]
        thickness = 2.0 * np.sqrt(np.maximum(r ** 2 - rho ** 2, 0.0))
        # delta_n ~ 0.05 for polystyrene in water (realistic: 0.1-0.5 rad peak)
        bead_phase = float(rng.uniform(0.1, 0.5)) * thickness / r
        phase[mask] += bead_phase

        # Slight amplitude reduction (scattering)
        amplitude[mask] -= float(rng.uniform(0.01, 0.05))

    amplitude = np.clip(amplitude, 0.3, 1.0)
    return (amplitude * np.exp(1j * phase)).astype(np.complex128)


# =============================================================================
# Phantom Dispatcher
# =============================================================================

PHANTOM_TYPES = {
    "cell_simple":      lambda N, rng: generate_cell_phantom(N, rng, "simple"),
    "cell_medium":      lambda N, rng: generate_cell_phantom(N, rng, "medium"),
    "cell_complex":     lambda N, rng: generate_cell_phantom(N, rng, "complex"),
    "resolution_target": generate_resolution_target,
    "patterned_surface": generate_patterned_surface,
    "transparent_specimen": generate_transparent_specimen,
    "microbead":         generate_microbead_phantom,
}

# Per-tier phantom assignment
TIER_PHANTOMS = {
    "public": [
        "cell_simple", "cell_medium", "cell_complex",
        "resolution_target", "patterned_surface", "transparent_specimen",
        "microbead", "cell_simple", "cell_medium",
        "resolution_target", "patterned_surface", "transparent_specimen",
    ],
    "dev": [
        "cell_simple", "cell_medium", "cell_complex",
        "resolution_target", "patterned_surface", "transparent_specimen",
        "microbead", "cell_simple", "cell_medium", "cell_complex",
        "resolution_target", "patterned_surface", "transparent_specimen",
        "microbead", "cell_medium", "cell_complex",
        "patterned_surface", "transparent_specimen", "microbead",
        "cell_complex",
    ],
    "hidden": [
        "cell_complex", "cell_complex", "cell_complex",
        "resolution_target", "patterned_surface", "transparent_specimen",
        "microbead", "cell_complex", "cell_medium", "cell_complex",
        "resolution_target", "patterned_surface", "transparent_specimen",
        "microbead", "cell_complex", "cell_complex",
        "patterned_surface", "transparent_specimen", "microbead",
        "cell_complex",
    ],
}

TIER_CONFIG = {
    "public": {"n_samples": 12, "base_seed": 0},
    "dev":    {"n_samples": 20, "base_seed": 10000},
    "hidden": {"n_samples": 20, "base_seed": 20000},
}


# =============================================================================
# Image Utilities
# =============================================================================

def _norm(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-12:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def _save_png(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(
        np.clip(_norm(arr) * 255, 0, 255).astype(np.uint8), "L"
    ).save(str(path))


def _save_complex_png(z: np.ndarray, path_amp: Path, path_phase: Path) -> None:
    """Save amplitude and phase of a complex array as separate PNGs."""
    _save_png(np.abs(z), path_amp)
    _save_png(np.angle(z), path_phase)


# =============================================================================
# Metrics
# =============================================================================

def compute_psnr(x_true: np.ndarray, x_recon: np.ndarray) -> float:
    """PSNR on amplitude images with optimal linear scaling.

    Uses least-squares optimal scaling: alpha = <a_true, a_recon> / <a_recon, a_recon>
    This is standard practice for holographic / phase-retrieval benchmarks where
    the reconstruction has an arbitrary global amplitude factor.
    """
    amp_true = np.abs(x_true).astype(np.float64)
    amp_recon = np.abs(x_recon).astype(np.float64)
    # Optimal linear scaling
    alpha = np.sum(amp_true * amp_recon) / (np.sum(amp_recon ** 2) + 1e-15)
    amp_recon_s = alpha * amp_recon
    # PSNR with data range = max of true amplitude
    data_range = amp_true.max() + 1e-12
    mse = np.mean((amp_true - amp_recon_s) ** 2)
    if mse < 1e-15:
        return 60.0
    return float(10 * np.log10(data_range ** 2 / mse))


def compute_ssim_simple(x: np.ndarray, y: np.ndarray) -> float:
    """Simplified SSIM computation for amplitude images with optimal scaling."""
    ax = np.abs(x).astype(np.float64)
    ay = np.abs(y).astype(np.float64)
    # Optimal linear scaling for the reconstruction
    alpha = np.sum(ax * ay) / (np.sum(ay ** 2) + 1e-15)
    ay = alpha * ay
    # Normalize both to [0, 1] by true image range
    scale = ax.max() + 1e-12
    ax = ax / scale
    ay = ay / scale
    mu_x = ax.mean()
    mu_y = ay.mean()
    sig_x = ax.std()
    sig_y = ay.std()
    sig_xy = np.mean((ax - mu_x) * (ay - mu_y))
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim = ((2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)) / \
           ((mu_x ** 2 + mu_y ** 2 + C1) * (sig_x ** 2 + sig_y ** 2 + C2))
    return float(ssim)


# =============================================================================
# Dataset Tier Generator
# =============================================================================

def sample_mismatch(rng: np.random.Generator, spec: dict) -> dict:
    return {k: float(rng.uniform(v["min"], v["max"])) for k, v in spec.items()}


def generate_tier(tier: str) -> dict:
    """Generate one tier of the holography benchmark dataset.

    Returns
    -------
    summary : dict
        Per-sample baseline metrics.
    """
    config = TIER_CONFIG[tier]
    spec_ranges = SPEC[tier]
    n_samples = config["n_samples"]
    base_seed = config["base_seed"]
    phantom_types = TIER_PHANTOMS[tier]

    tier_dir = BENCHMARK_DIR / tier
    if tier_dir.exists():
        shutil.rmtree(tier_dir)
    tier_dir.mkdir(parents=True, exist_ok=True)

    h5_path = tier_dir / f"holography_challenge_{tier}.h5"
    rng = np.random.default_rng(base_seed)

    results = {}

    with h5py.File(h5_path, "w") as f:
        f.attrs["description"] = (
            f"PWM Digital Holography benchmark -- {tier} tier "
            f"(angular spectrum propagation, off-axis configuration)"
        )
        f.attrs["spec_ranges"] = json.dumps(spec_ranges)
        f.attrs["physics"] = json.dumps({
            "wavelength_m": WAVELENGTH_M,
            "pixel_pitch_m": PIXEL_PITCH_M,
            "propagation_distance_m": PROP_DISTANCE_M,
            "reference_tilt_rad": REF_TILT_RAD,
            "forward_model": "y = |R + O|^2, O = IFFT{H_prop * FFT{x_true}}",
            "measurement_type": "intensity_hologram",
        })

        for idx in range(n_samples):
            key = f"sample_{idx:02d}"
            sample_seed = base_seed + idx
            sample_rng = np.random.default_rng(sample_seed)

            # Generate phantom
            phantom_type = phantom_types[idx % len(phantom_types)]
            x_true = PHANTOM_TYPES[phantom_type](IMAGE_SIZE, sample_rng)

            # Sample mismatch
            mis = sample_mismatch(rng, spec_ranges)

            # Generate hologram with mismatch
            y, H_ideal = forward_hologram_mismatched(
                x_true, mis, rng,
                wavelength=WAVELENGTH_M,
                pixel_pitch=PIXEL_PITCH_M,
                distance=PROP_DISTANCE_M,
                tilt_rad=REF_TILT_RAD,
            )

            # CPU baseline reconstruction
            recon = baseline_reconstruct(y, H_ideal)

            # Metrics
            psnr_val = compute_psnr(x_true, recon)
            ssim_val = compute_ssim_simple(x_true, recon)

            # Store in HDF5
            grp = f.create_group(key)
            # Store complex x_true as two real arrays (amplitude, phase)
            grp.create_dataset("x_true_amplitude", data=np.abs(x_true).astype(np.float32),
                               compression="gzip")
            grp.create_dataset("x_true_phase", data=np.angle(x_true).astype(np.float32),
                               compression="gzip")
            grp.create_dataset("y", data=y.astype(np.float32), compression="gzip")
            grp.create_dataset("H_ideal_real", data=H_ideal.real.astype(np.float32),
                               compression="gzip")
            grp.create_dataset("H_ideal_imag", data=H_ideal.imag.astype(np.float32),
                               compression="gzip")

            grp.attrs["metadata"] = json.dumps({
                "phantom_type": phantom_type,
                "shape": [IMAGE_SIZE, IMAGE_SIZE],
                "seed": sample_seed,
                "baseline_psnr_db": round(psnr_val, 2),
                "baseline_ssim": round(ssim_val, 4),
            })
            grp.attrs["spec_ranges"] = json.dumps(spec_ranges)
            grp.attrs["true_spec"] = json.dumps(mis)

            results[key] = {
                "phantom_type": phantom_type,
                "psnr_db": round(psnr_val, 2),
                "ssim": round(ssim_val, 4),
                "mismatch": mis,
            }

            print(f"  [{tier}] {key} {phantom_type:25s} "
                  f"PSNR={psnr_val:.2f} dB  SSIM={ssim_val:.4f}  "
                  f"d_err={mis['propagation_distance_error']:.4f} "
                  f"lam_err={mis['wavelength_error']:.4f} "
                  f"tilt_err={mis['reference_tilt_error']:.4f} "
                  f"noise={mis['noise_level']:.4f}")

    # Save spec
    with open(tier_dir / "spec.json", "w") as sf:
        json.dump(spec_ranges, sf, indent=2)
    with open(tier_dir / "results.json", "w") as rf:
        json.dump(results, rf, indent=2)

    print(f"  [{tier}] HDF5 -> {h5_path.name}  ({n_samples} samples)")
    return results


# =============================================================================
# Gallery Image Generation
# =============================================================================

def generate_gallery(n_scenes: int = 4) -> None:
    """Generate gallery preview images for the platform.

    Creates scene_00 through scene_{n-1} directories with:
        gt.png              -- ground truth amplitude
        measurement_I.png   -- hologram (low noise)
        measurement_II.png  -- hologram (high noise)
        recon_I.png         -- baseline reconstruction (low noise)
        recon_II.png        -- baseline reconstruction (high noise)
        recon_III.png       -- baseline reconstruction (medium noise, phase view)
    """
    gallery_root = (
        Path(__file__).resolve().parents[3]
        / "platform" / "pwm_platform" / "static" / "img"
        / "benchmark_gallery" / "holography"
    )

    print(f"\nGenerating gallery images -> {gallery_root}")

    for scene_idx in range(n_scenes):
        scene_dir = gallery_root / f"scene_{scene_idx:02d}"
        scene_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(42000 + scene_idx)

        # Choose phantom type per scene
        phantom_types_gallery = [
            "cell_medium", "resolution_target", "patterned_surface", "microbead"
        ]
        ptype = phantom_types_gallery[scene_idx % len(phantom_types_gallery)]
        x_true = PHANTOM_TYPES[ptype](IMAGE_SIZE, rng)

        # Ground truth: amplitude image
        _save_png(np.abs(x_true), scene_dir / "gt.png")

        # Measurement I: low noise hologram
        mis_low = {
            "propagation_distance_error": 0.01,
            "wavelength_error": 0.005,
            "reference_tilt_error": 0.002,
            "noise_level": 0.02,
        }
        y_low, H_low = forward_hologram_mismatched(x_true, mis_low, rng)
        _save_png(y_low, scene_dir / "measurement_I.png")

        # Measurement II: high noise hologram
        mis_high = {
            "propagation_distance_error": 0.05,
            "wavelength_error": 0.02,
            "reference_tilt_error": 0.01,
            "noise_level": 0.10,
        }
        y_high, H_high = forward_hologram_mismatched(x_true, mis_high, rng)
        _save_png(y_high, scene_dir / "measurement_II.png")

        # Reconstruction I: baseline on low noise
        recon_low = baseline_reconstruct(y_low, H_low)
        _save_png(np.abs(recon_low), scene_dir / "recon_I.png")

        # Reconstruction II: baseline on high noise
        recon_high = baseline_reconstruct(y_high, H_high)
        _save_png(np.abs(recon_high), scene_dir / "recon_II.png")

        # Reconstruction III: phase image from medium noise
        mis_med = {
            "propagation_distance_error": 0.03,
            "wavelength_error": 0.01,
            "reference_tilt_error": 0.005,
            "noise_level": 0.05,
        }
        y_med, H_med = forward_hologram_mismatched(x_true, mis_med, rng)
        recon_med = baseline_reconstruct(y_med, H_med)
        _save_png(np.angle(recon_med), scene_dir / "recon_III.png")

        print(f"  scene_{scene_idx:02d}: {ptype} -> 6 images")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("Digital Holography Benchmark Dataset Generator")
    print("=" * 60)
    print(f"Output: {BENCHMARK_DIR}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Wavelength: {WAVELENGTH_M * 1e9:.0f} nm")
    print(f"Pixel pitch: {PIXEL_PITCH_M * 1e6:.2f} um")
    print(f"Propagation distance: {PROP_DISTANCE_M * 1e6:.0f} um")
    print(f"Reference tilt: {REF_TILT_RAD:.3f} rad")
    print()

    all_results = {}

    for tier in ["public", "dev", "hidden"]:
        n = TIER_CONFIG[tier]["n_samples"]
        print(f"Generating {tier} tier ({n} samples)...")
        results = generate_tier(tier)
        all_results[tier] = results

        # Summarize
        psnrs = [v["psnr_db"] for v in results.values()]
        ssims = [v["ssim"] for v in results.values()]
        print(f"  [{tier}] Baseline PSNR: {np.mean(psnrs):.2f} +/- {np.std(psnrs):.2f} dB")
        print(f"  [{tier}] Baseline SSIM: {np.mean(ssims):.4f} +/- {np.std(ssims):.4f}")
        print()

    # Generate gallery images
    generate_gallery(n_scenes=4)

    # Save overall summary
    summary_path = BENCHMARK_DIR / "baseline_summary.json"
    with open(summary_path, "w") as sf:
        json.dump(all_results, sf, indent=2)
    print(f"\nBaseline summary -> {summary_path}")

    print(f"\n{'=' * 60}")
    print("Done -- Digital Holography benchmark ready")
    print(f"  public:  {TIER_CONFIG['public']['n_samples']} samples")
    print(f"  dev:     {TIER_CONFIG['dev']['n_samples']} samples")
    print(f"  hidden:  {TIER_CONFIG['hidden']['n_samples']} samples")

    # Print overall baseline stats
    for tier in ["public", "dev", "hidden"]:
        psnrs = [v["psnr_db"] for v in all_results[tier].values()]
        ssims = [v["ssim"] for v in all_results[tier].values()]
        print(f"  {tier:8s} avg PSNR={np.mean(psnrs):.2f} dB  avg SSIM={np.mean(ssims):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
