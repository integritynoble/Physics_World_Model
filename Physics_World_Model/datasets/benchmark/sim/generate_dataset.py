#!/usr/bin/env python3
"""Generate Structured Illumination Microscopy (SIM) benchmark dataset.

Forward model (per orientation/phase):
    y_k = Poisson(PSF * (I_k * x) + background) + readout_noise

where:
    I_k = 1 + m * cos(2*pi*f*r_hat + phi_k)
        is a sinusoidal illumination pattern
    m       : modulation depth (0 < m <= 1)
    f       : illumination spatial frequency (cycles/pixel)
    r_hat   : projection of position onto pattern orientation
    phi_k   : phase shift (3 phases per orientation, 3 orientations = 9 frames)
    PSF     : Gaussian point spread function of the microscope
    x       : ground truth fluorophore distribution (256x256)

The measurement y is the average of the 9 raw SIM frames (sum / 9).
The ideal measurement H_ideal stores the PSF-convolved noiseless widefield image.

Mismatch parameters:
    pattern_frequency_error : fractional error in assumed pattern frequency
    modulation_depth        : actual modulation depth m (< 1 means reduced contrast)
    phase_error_deg         : systematic error in phase steps (degrees)
    noise_level             : Poisson noise scaling (mean photon count)

Phantoms:
    Biological structures: actin filaments, mitochondrial networks, microtubules.
    Thin curved lines and branching structures characteristic of fluorescence
    microscopy samples.

CPU Baseline reconstruction:
    Wiener-filtered SIM reconstruction in the frequency domain.
    Expected: ~24-30 dB PSNR.

Tiers:
    Public  : 12 samples (seed offset 0)
    Dev     : 20 samples (seed offset 10000)
    Hidden  : 20 samples (seed offset 20000)

Usage:
    cd datasets/benchmark/sim
    python3 generate_dataset.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

BENCHMARK_DIR = Path(__file__).resolve().parent

# -- Image dimensions --------------------------------------------------------

IMAGE_SIZE = 256           # ground truth and measurement size
PIXEL_SIZE_NM = 50.0       # 50 nm/pixel (super-resolution scale)

# -- SIM optical parameters --------------------------------------------------

PSF_SIGMA_PX = 2.5         # PSF sigma in pixels (~125 nm FWHM at 50 nm/px)
N_ORIENTATIONS = 3          # 3 illumination orientations (0, 60, 120 degrees)
N_PHASES = 3                # 3 phase steps per orientation (0, 2pi/3, 4pi/3)
PATTERN_FREQ_NOMINAL = 0.15  # illumination pattern frequency (cycles/pixel)
BACKGROUND_LEVEL = 5.0      # background fluorescence (photons/pixel)
READOUT_NOISE_STD = 2.0     # sCMOS readout noise (electrons)

# -- Mismatch ranges per tier ------------------------------------------------

SPEC = {
    "public": {
        "pattern_frequency_error": {"min": -0.05, "max": 0.05, "unit": "fraction"},
        "modulation_depth":        {"min": 0.7,   "max": 1.0,  "unit": ""},
        "phase_error_deg":         {"min": -3.0,  "max": 3.0,  "unit": "degrees"},
        "noise_level":             {"min": 500,   "max": 2000, "unit": "photons"},
    },
    "dev": {
        "pattern_frequency_error": {"min": -0.08, "max": 0.08, "unit": "fraction"},
        "modulation_depth":        {"min": 0.5,   "max": 1.0,  "unit": ""},
        "phase_error_deg":         {"min": -5.0,  "max": 5.0,  "unit": "degrees"},
        "noise_level":             {"min": 300,   "max": 2000, "unit": "photons"},
    },
    "hidden": {
        "pattern_frequency_error": {"min": -0.12, "max": 0.12, "unit": "fraction"},
        "modulation_depth":        {"min": 0.3,   "max": 0.9,  "unit": ""},
        "phase_error_deg":         {"min": -8.0,  "max": 8.0,  "unit": "degrees"},
        "noise_level":             {"min": 200,   "max": 1500, "unit": "photons"},
    },
}


# -- Phantom generators (biological structures) -----------------------------

def _smooth_curve_points(
    control_pts: np.ndarray, n_interp: int = 500
) -> np.ndarray:
    """Generate smooth curve via cubic interpolation of control points.

    Args:
        control_pts: (N, 2) array of (y, x) control points
        n_interp: number of interpolated points

    Returns:
        (n_interp, 2) array of (y, x) interpolated points
    """
    n = len(control_pts)
    if n < 2:
        return control_pts
    t = np.linspace(0, 1, n)
    t_new = np.linspace(0, 1, n_interp)
    cs_y = CubicSpline(t, control_pts[:, 0], bc_type="natural")
    cs_x = CubicSpline(t, control_pts[:, 1], bc_type="natural")
    return np.column_stack([cs_y(t_new), cs_x(t_new)])


def _draw_thick_curve(
    img: np.ndarray, curve: np.ndarray, thickness: float, intensity: float = 1.0,
) -> None:
    """Draw a smooth thick curve onto an image using Gaussian cross-section."""
    H, W = img.shape
    sigma = thickness / 2.0
    r = int(np.ceil(3 * sigma))
    for pt in curve:
        py, px = pt
        iy, ix = int(round(py)), int(round(px))
        y0, y1 = max(0, iy - r), min(H, iy + r + 1)
        x0, x1 = max(0, ix - r), min(W, ix + r + 1)
        if y1 <= y0 or x1 <= x0:
            continue
        yy = np.arange(y0, y1)[:, None]
        xx = np.arange(x0, x1)[None, :]
        g = intensity * np.exp(-((yy - py) ** 2 + (xx - px) ** 2) / (2 * sigma ** 2))
        img[y0:y1, x0:x1] += g


def make_actin_filament_phantom(
    H: int, W: int, rng: np.random.Generator,
) -> np.ndarray:
    """Actin filaments: long thin curved lines forming a cytoskeletal mesh.

    Actin networks consist of thin (~7 nm) filaments arranged in branching
    networks. We simulate them as smooth curved lines with occasional branching.

    Returns:
        x_true: (H, W) float64 ground truth image
    """
    x_true = np.zeros((H, W), dtype=np.float64)
    n_filaments = rng.integers(12, 25)

    for _ in range(n_filaments):
        # Start position with margin
        cy = rng.uniform(H * 0.05, H * 0.95)
        cx = rng.uniform(W * 0.05, W * 0.95)
        angle = rng.uniform(0, 2 * np.pi)
        length = rng.uniform(H * 0.15, H * 0.55)

        # Curved path with 4-8 control points
        n_ctrl = rng.integers(4, 9)
        ctrl = np.zeros((n_ctrl, 2))
        for j in range(n_ctrl):
            t = j / max(n_ctrl - 1, 1)
            ctrl[j, 0] = cy + length * t * np.sin(angle) + rng.uniform(-20, 20)
            ctrl[j, 1] = cx + length * t * np.cos(angle) + rng.uniform(-20, 20)

        curve = _smooth_curve_points(ctrl, n_interp=600)
        # Filter out-of-bounds
        valid = (
            (curve[:, 0] >= 1) & (curve[:, 0] < H - 1)
            & (curve[:, 1] >= 1) & (curve[:, 1] < W - 1)
        )
        curve = curve[valid]
        if len(curve) < 10:
            continue

        # Thin line with slight intensity variation
        thickness = rng.uniform(0.8, 1.8)
        intensity = rng.uniform(0.5, 1.0)
        # Subsample curve to avoid excessive overdraw
        step = max(1, len(curve) // 200)
        _draw_thick_curve(x_true, curve[::step], thickness, intensity)

        # Occasional branching point
        if rng.random() < 0.4 and len(curve) > 20:
            branch_idx = rng.integers(len(curve) // 4, 3 * len(curve) // 4)
            branch_angle = angle + rng.uniform(-np.pi / 3, np.pi / 3)
            branch_len = rng.uniform(H * 0.05, H * 0.15)
            n_bctrl = rng.integers(3, 5)
            bctrl = np.zeros((n_bctrl, 2))
            for j in range(n_bctrl):
                t = j / max(n_bctrl - 1, 1)
                bctrl[j, 0] = curve[branch_idx, 0] + branch_len * t * np.sin(branch_angle) + rng.normal(0, 5)
                bctrl[j, 1] = curve[branch_idx, 1] + branch_len * t * np.cos(branch_angle) + rng.normal(0, 5)
            bcurve = _smooth_curve_points(bctrl, n_interp=200)
            bvalid = (
                (bcurve[:, 0] >= 1) & (bcurve[:, 0] < H - 1)
                & (bcurve[:, 1] >= 1) & (bcurve[:, 1] < W - 1)
            )
            bcurve = bcurve[bvalid]
            if len(bcurve) > 5:
                bstep = max(1, len(bcurve) // 80)
                _draw_thick_curve(x_true, bcurve[::bstep], thickness * 0.8, intensity * 0.8)

    return x_true


def make_mitochondria_network_phantom(
    H: int, W: int, rng: np.random.Generator,
) -> np.ndarray:
    """Mitochondrial network: elongated tubular organelles with branching.

    Mitochondria form dynamic networks of thin tubules (~200-500 nm diameter)
    with frequent branching and merging.

    Returns:
        x_true: (H, W) float64 ground truth image
    """
    x_true = np.zeros((H, W), dtype=np.float64)
    n_tubules = rng.integers(8, 18)

    for _ in range(n_tubules):
        cy = rng.uniform(H * 0.1, H * 0.9)
        cx = rng.uniform(W * 0.1, W * 0.9)
        angle = rng.uniform(0, 2 * np.pi)
        length = rng.uniform(H * 0.1, H * 0.4)

        # Winding path
        n_ctrl = rng.integers(5, 10)
        ctrl = np.zeros((n_ctrl, 2))
        for j in range(n_ctrl):
            t = j / max(n_ctrl - 1, 1)
            # Mitochondria tend to be more winding than actin
            ctrl[j, 0] = cy + length * t * np.sin(angle) + rng.uniform(-25, 25)
            ctrl[j, 1] = cx + length * t * np.cos(angle) + rng.uniform(-25, 25)

        curve = _smooth_curve_points(ctrl, n_interp=500)
        valid = (
            (curve[:, 0] >= 1) & (curve[:, 0] < H - 1)
            & (curve[:, 1] >= 1) & (curve[:, 1] < W - 1)
        )
        curve = curve[valid]
        if len(curve) < 10:
            continue

        # Slightly thicker than actin (tubular membranes)
        thickness = rng.uniform(1.5, 3.0)
        intensity = rng.uniform(0.6, 1.0)
        step = max(1, len(curve) // 200)
        _draw_thick_curve(x_true, curve[::step], thickness, intensity)

        # Branching is very common in mitochondrial networks
        n_branches = rng.integers(1, 4)
        for _ in range(n_branches):
            if len(curve) < 20:
                break
            branch_idx = rng.integers(len(curve) // 5, 4 * len(curve) // 5)
            branch_angle = angle + rng.uniform(-np.pi / 2, np.pi / 2)
            branch_len = rng.uniform(H * 0.05, H * 0.2)
            n_bctrl = rng.integers(3, 6)
            bctrl = np.zeros((n_bctrl, 2))
            for j in range(n_bctrl):
                t = j / max(n_bctrl - 1, 1)
                bctrl[j, 0] = curve[branch_idx, 0] + branch_len * t * np.sin(branch_angle) + rng.normal(0, 8)
                bctrl[j, 1] = curve[branch_idx, 1] + branch_len * t * np.cos(branch_angle) + rng.normal(0, 8)
            bcurve = _smooth_curve_points(bctrl, n_interp=200)
            bvalid = (
                (bcurve[:, 0] >= 1) & (bcurve[:, 0] < H - 1)
                & (bcurve[:, 1] >= 1) & (bcurve[:, 1] < W - 1)
            )
            bcurve = bcurve[bvalid]
            if len(bcurve) > 5:
                bstep = max(1, len(bcurve) // 80)
                _draw_thick_curve(x_true, bcurve[::bstep], thickness * 0.7, intensity * 0.7)

    return x_true


def make_microtubule_phantom(
    H: int, W: int, rng: np.random.Generator,
) -> np.ndarray:
    """Microtubules: long straight-ish filaments radiating from a centrosome.

    Microtubules are stiff hollow tubes (~25 nm diameter) that radiate outward
    from centrosomal organizing centers. They are straighter than actin.

    Returns:
        x_true: (H, W) float64 ground truth image
    """
    x_true = np.zeros((H, W), dtype=np.float64)

    # Centrosome position (organizing center)
    center_y = H / 2 + rng.uniform(-H * 0.15, H * 0.15)
    center_x = W / 2 + rng.uniform(-W * 0.15, W * 0.15)
    n_filaments = rng.integers(15, 30)

    for _ in range(n_filaments):
        angle = rng.uniform(0, 2 * np.pi)
        length = rng.uniform(H * 0.2, H * 0.5)

        # Microtubules are relatively straight (less curvature control pts)
        n_ctrl = rng.integers(3, 5)
        ctrl = np.zeros((n_ctrl, 2))
        for j in range(n_ctrl):
            t = j / max(n_ctrl - 1, 1)
            # Small lateral deviations (stiff polymer)
            ctrl[j, 0] = center_y + length * t * np.sin(angle) + rng.uniform(-8, 8)
            ctrl[j, 1] = center_x + length * t * np.cos(angle) + rng.uniform(-8, 8)

        curve = _smooth_curve_points(ctrl, n_interp=400)
        valid = (
            (curve[:, 0] >= 1) & (curve[:, 0] < H - 1)
            & (curve[:, 1] >= 1) & (curve[:, 1] < W - 1)
        )
        curve = curve[valid]
        if len(curve) < 10:
            continue

        thickness = rng.uniform(0.7, 1.5)
        intensity = rng.uniform(0.5, 1.0)
        step = max(1, len(curve) // 200)
        _draw_thick_curve(x_true, curve[::step], thickness, intensity)

    return x_true


# Phantom pool
PHANTOM_FNS = [
    make_actin_filament_phantom,
    make_mitochondria_network_phantom,
    make_microtubule_phantom,
]
PHANTOM_NAMES = ["actin_filament", "mitochondria_network", "microtubule"]


# -- PSF generation ---------------------------------------------------------

def make_gaussian_psf(sigma: float, size: int = 0) -> np.ndarray:
    """Create a 2D Gaussian PSF kernel.

    Args:
        sigma: standard deviation in pixels
        size: kernel size (if 0, auto = 6*sigma+1)

    Returns:
        psf: (size, size) float64 normalized PSF kernel
    """
    if size == 0:
        size = int(6 * sigma) + 1
    if size % 2 == 0:
        size += 1
    half = size // 2
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    psf = np.exp(-(yy ** 2 + xx ** 2) / (2 * sigma ** 2))
    psf /= psf.sum()
    return psf.astype(np.float64)


# -- SIM illumination pattern -----------------------------------------------

def make_illumination_pattern(
    H: int,
    W: int,
    orientation_deg: float,
    phase_rad: float,
    frequency: float,
    modulation_depth: float,
) -> np.ndarray:
    """Generate a 2D sinusoidal illumination pattern.

    I(r) = 1 + m * cos(2*pi*f*r_hat + phi)

    where r_hat is the coordinate projected onto the pattern direction.

    Args:
        H, W: image size
        orientation_deg: pattern orientation in degrees
        phase_rad: phase offset in radians
        frequency: spatial frequency in cycles/pixel
        modulation_depth: modulation contrast m (0 to 1)

    Returns:
        pattern: (H, W) float64 illumination intensity
    """
    yy, xx = np.mgrid[:H, :W].astype(np.float64)
    theta = np.deg2rad(orientation_deg)
    # Project position onto pattern direction
    r_proj = yy * np.cos(theta) + xx * np.sin(theta)
    pattern = 1.0 + modulation_depth * np.cos(2 * np.pi * frequency * r_proj + phase_rad)
    return pattern


# -- SIM forward model -------------------------------------------------------

def sim_forward(
    x_true: np.ndarray,
    psf: np.ndarray,
    pattern_frequency: float,
    modulation_depth: float,
    phase_error_deg: float,
    noise_level: float,
    rng: np.random.Generator,
    background: float = BACKGROUND_LEVEL,
    readout_std: float = READOUT_NOISE_STD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SIM forward model: generate 9 raw frames and the averaged measurement.

    For each orientation (0, 60, 120 deg) and each phase (0, 2pi/3, 4pi/3):
        raw_k = Poisson(PSF * (I_k * x_true) * photon_scale + background) + readout

    Args:
        x_true: (H, W) ground truth fluorophore distribution (0 to 1)
        psf: PSF kernel
        pattern_frequency: illumination pattern frequency (cycles/pixel)
        modulation_depth: illumination modulation depth m
        phase_error_deg: systematic phase step error in degrees
        noise_level: mean photon count scaling (higher = less noise)
        rng: random generator
        background: background fluorescence level
        readout_std: readout noise standard deviation

    Returns:
        y: (H, W) float32 averaged measurement (sum of 9 frames / 9)
        raw_frames: (9, H, W) float32 all 9 raw SIM frames
        H_ideal: (H, W) float32 noiseless widefield (PSF * x_true)
    """
    H, W = x_true.shape
    x64 = x_true.astype(np.float64)

    # Noiseless widefield: PSF-convolved x_true
    widefield_ideal = fftconvolve(x64, psf, mode="same")
    widefield_ideal = np.maximum(widefield_ideal, 0.0)

    # Orientation angles
    orientations_deg = [0.0, 60.0, 120.0]
    # Nominal phase steps: 0, 2pi/3, 4pi/3
    phase_error_rad = np.deg2rad(phase_error_deg)

    raw_frames = np.zeros((N_ORIENTATIONS * N_PHASES, H, W), dtype=np.float64)
    frame_idx = 0

    for orient_deg in orientations_deg:
        for k in range(N_PHASES):
            # Nominal phase for this step
            nominal_phase = 2 * np.pi * k / N_PHASES
            # Add systematic phase error (same direction each time)
            actual_phase = nominal_phase + phase_error_rad * (k + 1) / N_PHASES

            # Generate illumination pattern
            pattern = make_illumination_pattern(
                H, W, orient_deg, actual_phase, pattern_frequency, modulation_depth,
            )

            # Modulated sample: I_k * x_true
            modulated = pattern * x64

            # Apply PSF (convolution)
            blurred = fftconvolve(modulated, psf, mode="same")
            blurred = np.maximum(blurred, 0.0)

            # Scale by photon level
            signal = blurred * noise_level + background

            # Poisson shot noise
            noisy = rng.poisson(np.maximum(signal, 0.01)).astype(np.float64)

            # Readout noise
            noisy += rng.normal(0, readout_std, (H, W))
            noisy = np.maximum(noisy, 0.0)

            raw_frames[frame_idx] = noisy
            frame_idx += 1

    # Measurement: average of all 9 raw frames
    y = raw_frames.mean(axis=0)

    return (
        y.astype(np.float32),
        raw_frames.astype(np.float32),
        widefield_ideal.astype(np.float32),
    )


# -- CPU Baseline: Wiener-filtered SIM reconstruction -----------------------

def wiener_sim_reconstruct(
    raw_frames: np.ndarray,
    psf: np.ndarray,
    pattern_freq_est: float,
    mod_depth_est: float = 0.9,
    wiener_param: float = 0.01,
) -> np.ndarray:
    """Wiener-filtered SIM reconstruction.

    This implements a simplified frequency-domain SIM reconstruction:
    1. Separate the 3 frequency components per orientation using
       phase-stepping algebra.
    2. Shift separated components to their correct positions in
       frequency space.
    3. Apply Wiener deconvolution for OTF compensation.
    4. Sum all frequency components and inverse-FFT.

    This is a simplified version of the Gustafsson (2000) algorithm.

    Args:
        raw_frames: (9, H, W) raw SIM frames
        psf: PSF kernel (for OTF estimation)
        pattern_freq_est: estimated pattern frequency (cycles/pixel)
        mod_depth_est: estimated modulation depth
        wiener_param: Wiener regularization parameter

    Returns:
        recon: (H, W) float32 reconstructed super-resolved image
    """
    n_frames, H, W = raw_frames.shape
    assert n_frames == 9, f"Expected 9 frames, got {n_frames}"

    # Estimate the OTF from the PSF
    psf_padded = np.zeros((H, W), dtype=np.float64)
    ph, pw = psf.shape
    py = (H - ph) // 2
    px = (W - pw) // 2
    psf_padded[py:py + ph, px:px + pw] = psf
    OTF = np.fft.fft2(np.fft.ifftshift(psf_padded))
    OTF_conj = np.conj(OTF)
    OTF_sq = np.abs(OTF) ** 2

    orientations_deg = [0.0, 60.0, 120.0]
    reconstructed_spectrum = np.zeros((H, W), dtype=np.complex128)
    weight_map = np.zeros((H, W), dtype=np.float64)

    for o_idx, orient_deg in enumerate(orientations_deg):
        # Extract 3 phase-stepped images for this orientation
        I0 = raw_frames[o_idx * N_PHASES + 0].astype(np.float64)
        I1 = raw_frames[o_idx * N_PHASES + 1].astype(np.float64)
        I2 = raw_frames[o_idx * N_PHASES + 2].astype(np.float64)

        # Phase-stepping separation (3-phase algorithm):
        # D0 = (I0 + I1 + I2) / 3  -- DC component (widefield equivalent)
        # D+ = (I0 + I1*exp(-i*2pi/3) + I2*exp(-i*4pi/3)) / 3  -- +1 order
        # D- = (I0 + I1*exp(+i*2pi/3) + I2*exp(+i*4pi/3)) / 3  -- -1 order
        w = np.exp(1j * 2 * np.pi / 3)
        D0 = (I0 + I1 + I2) / 3.0
        Dp = (I0 + I1 * np.conj(w) + I2 * np.conj(w ** 2)) / 3.0
        Dm = (I0 + I1 * w + I2 * w ** 2) / 3.0

        # FFT of separated components
        F_D0 = np.fft.fft2(D0)
        F_Dp = np.fft.fft2(Dp)
        F_Dm = np.fft.fft2(Dm)

        # Frequency shift for +/- orders
        theta = np.deg2rad(orient_deg)
        ky_shift = pattern_freq_est * np.cos(theta) * H
        kx_shift = pattern_freq_est * np.sin(theta) * W

        # Create frequency shift grids
        fy = np.fft.fftfreq(H) * H
        fx = np.fft.fftfreq(W) * W
        FY, FX = np.meshgrid(fy, fx, indexing="ij")

        # Shift +1 order: move by -k_ill to align in frequency domain
        shift_p = np.exp(-1j * 2 * np.pi * (FY * ky_shift / H + FX * kx_shift / W))
        shift_m = np.exp(+1j * 2 * np.pi * (FY * ky_shift / H + FX * kx_shift / W))

        # Wiener deconvolution for each component
        wiener_filter = OTF_conj / (OTF_sq + wiener_param)

        # DC component: standard Wiener deconvolution
        reconstructed_spectrum += wiener_filter * F_D0
        weight_map += np.abs(wiener_filter) ** 2

        # Shifted components (contribute to extended frequency support)
        # Scale by modulation depth estimate (m/2 factor in SIM theory)
        scale = 2.0 / max(mod_depth_est, 0.1)
        F_Dp_shifted = F_Dp * shift_p * scale
        F_Dm_shifted = F_Dm * shift_m * scale

        reconstructed_spectrum += wiener_filter * F_Dp_shifted
        weight_map += np.abs(wiener_filter * scale) ** 2
        reconstructed_spectrum += wiener_filter * F_Dm_shifted
        weight_map += np.abs(wiener_filter * scale) ** 2

    # Normalize by weight map
    weight_map = np.maximum(weight_map, 1e-10)
    reconstructed_spectrum /= np.sqrt(weight_map)

    # Apply a gentle apodization to suppress ringing
    fy = np.fft.fftfreq(H)
    fx = np.fft.fftfreq(W)
    FY, FX = np.meshgrid(fy, fx, indexing="ij")
    freq_radius = np.sqrt(FY ** 2 + FX ** 2)
    # Butterworth low-pass at extended cutoff
    cutoff = 0.45  # near Nyquist
    order = 4
    apodization = 1.0 / (1.0 + (freq_radius / cutoff) ** (2 * order))
    reconstructed_spectrum *= apodization

    # Inverse FFT
    recon = np.real(np.fft.ifft2(reconstructed_spectrum))

    # Post-processing: clip negatives, normalize
    recon = np.maximum(recon, 0.0)
    if recon.max() > 0:
        recon /= recon.max()

    return recon.astype(np.float32)


# -- Metrics -----------------------------------------------------------------

def compute_psnr(gt: np.ndarray, recon: np.ndarray) -> float:
    gt64 = gt.astype(np.float64)
    recon64 = recon.astype(np.float64)
    mse = np.mean((gt64 - recon64) ** 2)
    if mse < 1e-12:
        return 100.0
    data_range = float(gt64.max() - gt64.min())
    if data_range < 1e-12:
        return 0.0
    return float(10 * np.log10(data_range ** 2 / mse))


def compute_ssim(gt: np.ndarray, recon: np.ndarray) -> float:
    gt64 = gt.astype(np.float64)
    recon64 = recon.astype(np.float64)
    data_range = gt64.max() - gt64.min()
    if data_range < 1e-12:
        return 0.0
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    # Compute over 8x8 blocks for a more robust SSIM
    from scipy.ndimage import uniform_filter
    win_size = 7
    mu_x = uniform_filter(gt64, size=win_size)
    mu_y = uniform_filter(recon64, size=win_size)
    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y
    sigma_x_sq = uniform_filter(gt64 ** 2, size=win_size) - mu_x_sq
    sigma_y_sq = uniform_filter(recon64 ** 2, size=win_size) - mu_y_sq
    sigma_xy = uniform_filter(gt64 * recon64, size=win_size) - mu_xy

    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / \
               ((mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2))
    return float(ssim_map.mean())


# -- Image helpers -----------------------------------------------------------

def _norm(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo + 1e-8)


def _save_png(arr: np.ndarray, path: Path, percentile_clip: bool = False) -> None:
    if percentile_clip and arr.max() > 0:
        nonzero = arr[arr > 0]
        if len(nonzero) > 0:
            lo, hi = np.percentile(nonzero, [1, 99])
            arr = np.clip(arr, lo, hi)
    Image.fromarray(
        np.clip(_norm(arr) * 255, 0, 255).astype(np.uint8), "L"
    ).save(str(path))


# -- Phantom generation for each tier ----------------------------------------

def generate_phantoms(
    n: int, seed_offset: int,
) -> list[tuple[np.ndarray, str]]:
    """Generate phantom set for a tier.

    Returns list of (x_true, scene_name).
    """
    phantoms = []

    for i in range(n):
        fn_idx = i % len(PHANTOM_FNS)
        fn = PHANTOM_FNS[fn_idx]
        name = PHANTOM_NAMES[fn_idx]

        phantom_rng = np.random.default_rng(seed_offset + i)
        x_true = fn(IMAGE_SIZE, IMAGE_SIZE, phantom_rng)

        # Normalize to [0, 1]
        if x_true.max() > 0:
            x_true /= x_true.max()

        scene_name = f"{name}_{i:02d}"
        phantoms.append((x_true.astype(np.float32), scene_name))

    return phantoms


# -- Tier generation ---------------------------------------------------------

def sample_mismatch(rng: np.random.Generator, spec: dict) -> dict:
    result = {}
    for k, v in spec.items():
        val = float(rng.uniform(v["min"], v["max"]))
        if k == "noise_level":
            val = round(val)
        result[k] = val
    return result


def generate_tier(
    tier: str,
    phantoms: list[tuple[np.ndarray, str]],
    base_seed: int,
) -> None:
    """Generate one tier of the SIM benchmark."""
    spec_ranges = SPEC[tier]
    tier_dir = BENCHMARK_DIR / tier
    images_dir = tier_dir / "images"
    if tier_dir.exists():
        shutil.rmtree(tier_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    h5_path = tier_dir / f"sim_challenge_{tier}.h5"
    rng = np.random.default_rng(base_seed)
    true_specs = {}
    all_psnrs = []
    all_ssims = []

    # Build PSF once
    psf = make_gaussian_psf(PSF_SIGMA_PX)

    with h5py.File(h5_path, "w") as f:
        f.attrs["description"] = (
            f"PWM SIM benchmark -- {tier} tier "
            f"(Structured Illumination Microscopy)"
        )
        f.attrs["spec_ranges"] = json.dumps(spec_ranges)
        f.attrs["forward_model"] = (
            "y_k = Poisson(PSF * (I_k * x_true) * noise_level + bg) + readout; "
            "I_k = 1 + m * cos(2*pi*f*r_hat + phi_k); "
            "y = mean(y_0, ..., y_8)"
        )
        f.attrs["geometry"] = json.dumps({
            "image_size": IMAGE_SIZE,
            "pixel_size_nm": PIXEL_SIZE_NM,
            "psf_sigma_px": PSF_SIGMA_PX,
            "n_orientations": N_ORIENTATIONS,
            "n_phases": N_PHASES,
            "nominal_pattern_freq": PATTERN_FREQ_NOMINAL,
        })

        for idx, (x_true, scene_name) in enumerate(phantoms):
            key = f"sample_{idx:02d}"
            print(f"  [{tier}] Generating {key} ({scene_name})...",
                  end="", flush=True)

            mis = sample_mismatch(rng, spec_ranges)
            true_specs[key] = mis

            # Actual pattern frequency (with error)
            actual_freq = PATTERN_FREQ_NOMINAL * (1.0 + mis["pattern_frequency_error"])

            # Forward model
            y, raw_frames, H_ideal = sim_forward(
                x_true, psf,
                pattern_frequency=actual_freq,
                modulation_depth=mis["modulation_depth"],
                phase_error_deg=mis["phase_error_deg"],
                noise_level=mis["noise_level"],
                rng=rng,
            )

            # CPU baseline: Wiener SIM reconstruction
            # Use nominal frequency (participant doesn't know the error)
            recon = wiener_sim_reconstruct(
                raw_frames, psf,
                pattern_freq_est=PATTERN_FREQ_NOMINAL,
                mod_depth_est=0.9,
                wiener_param=0.005,
            )

            psnr = compute_psnr(x_true, recon)
            ssim = compute_ssim(x_true, recon)
            all_psnrs.append(psnr)
            all_ssims.append(ssim)

            # Save to HDF5
            grp = f.create_group(key)
            grp.create_dataset("x_true", data=x_true, compression="gzip")
            grp.create_dataset("y", data=y, compression="gzip")
            grp.create_dataset("raw_frames", data=raw_frames, compression="gzip")
            grp.create_dataset("H_ideal", data=H_ideal, compression="gzip")
            grp.create_dataset("reconstruction_baseline", data=recon,
                               compression="gzip")
            grp.attrs["metadata"] = json.dumps({
                "scene": scene_name,
                "shape": list(x_true.shape),
                "n_raw_frames": int(raw_frames.shape[0]),
                "psnr_baseline": float(psnr),
                "ssim_baseline": float(ssim),
            })
            grp.attrs["true_spec"] = json.dumps(mis)
            grp.attrs["spec_ranges"] = json.dumps(spec_ranges)

            # Save per-sample images
            sample_dir = images_dir / f"sample_{idx:02d}_{scene_name}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            _save_png(x_true, sample_dir / "gt.png")
            _save_png(y, sample_dir / "measurement.png", percentile_clip=True)
            _save_png(recon, sample_dir / "recon.png")
            _save_png(H_ideal, sample_dir / "H_ideal.png", percentile_clip=True)
            # Save one raw frame as example
            _save_png(raw_frames[0], sample_dir / "raw_frame_00.png", percentile_clip=True)
            with open(sample_dir / "spec.json", "w") as sf:
                json.dump({
                    "scene": scene_name,
                    "spec_ranges": spec_ranges,
                    "true_spec": mis,
                    "psnr_baseline": psnr,
                    "ssim_baseline": ssim,
                }, sf, indent=2)

            print(f"  PSNR={psnr:.2f} dB, SSIM={ssim:.3f}  "
                  f"freq_err={mis['pattern_frequency_error']:.3f}  "
                  f"mod={mis['modulation_depth']:.2f}  "
                  f"phase_err={mis['phase_error_deg']:.1f}deg  "
                  f"photons={mis['noise_level']}")

    # Save spec files
    with open(tier_dir / "spec.json", "w") as sf:
        json.dump(spec_ranges, sf, indent=2)
    if tier == "public":
        with open(tier_dir / "true_spec.json", "w") as tf:
            json.dump(true_specs, tf, indent=2)

    mean_psnr = np.mean(all_psnrs)
    mean_ssim = np.mean(all_ssims)
    print(f"  [{tier}] {len(phantoms)} samples | "
          f"Mean PSNR={mean_psnr:.2f} dB | Mean SSIM={mean_ssim:.3f}")
    print(f"  [{tier}] HDF5 -> {h5_path.name}")


# -- Gallery image generation ------------------------------------------------

def generate_gallery_images() -> None:
    """Generate gallery images for the platform benchmark page.

    Creates scene_00 through scene_03 with:
        gt.png, measurement_I.png, measurement_II.png,
        recon_I.png, recon_II.png
    """
    gallery_base = (
        BENCHMARK_DIR.parent.parent.parent
        / "platform" / "pwm_platform" / "static" / "img"
        / "benchmark_gallery" / "sim"
    )

    h5_path = BENCHMARK_DIR / "public" / "sim_challenge_public.h5"
    if not h5_path.exists():
        print("  [gallery] Public HDF5 not found, skipping gallery generation.")
        return

    # Pick 4 diverse samples: actin, mitochondria, microtubule, actin
    gallery_indices = [0, 1, 2, 3]

    with h5py.File(h5_path, "r") as f:
        for scene_idx, sample_idx in enumerate(gallery_indices):
            scene_dir = gallery_base / f"scene_{scene_idx:02d}"
            scene_dir.mkdir(parents=True, exist_ok=True)

            key = f"sample_{sample_idx:02d}"
            if key not in f:
                print(f"  [gallery] {key} not found, skipping.")
                continue

            grp = f[key]
            x_true = grp["x_true"][:]
            y = grp["y"][:]
            H_ideal = grp["H_ideal"][:]
            recon = grp["reconstruction_baseline"][:]
            raw_frames = grp["raw_frames"][:]

            # gt.png -- ground truth (fluorophore distribution)
            _save_png(x_true, scene_dir / "gt.png")

            # measurement_I.png -- averaged SIM measurement
            _save_png(y, scene_dir / "measurement_I.png", percentile_clip=True)

            # measurement_II.png -- one raw SIM frame (structured illumination visible)
            _save_png(raw_frames[0], scene_dir / "measurement_II.png", percentile_clip=True)

            # recon_I.png -- Wiener SIM baseline reconstruction
            _save_png(recon, scene_dir / "recon_I.png")

            # recon_II.png -- difference |GT - recon|
            diff = np.abs(x_true - recon)
            _save_png(diff, scene_dir / "recon_II.png")

            print(f"  [gallery] scene_{scene_idx:02d} -> {scene_dir}")


# -- Main --------------------------------------------------------------------

def main() -> None:
    print("SIM Benchmark Dataset Generator")
    print("=" * 68)
    print(f"Output: {BENCHMARK_DIR}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}, pixel size: {PIXEL_SIZE_NM} nm")
    print(f"PSF sigma: {PSF_SIGMA_PX} px, Pattern freq: {PATTERN_FREQ_NOMINAL} cyc/px")
    print(f"Orientations: {N_ORIENTATIONS}, Phases: {N_PHASES} "
          f"(9 raw frames total)\n")

    # Public tier (12 samples)
    print("Generating public tier (12 samples)...")
    public_phantoms = generate_phantoms(12, seed_offset=0)
    generate_tier("public", public_phantoms, base_seed=1000)

    # Dev tier (20 samples)
    print("\nGenerating dev tier (20 samples)...")
    dev_phantoms = generate_phantoms(20, seed_offset=10000)
    generate_tier("dev", dev_phantoms, base_seed=11000)

    # Hidden tier (20 samples)
    print("\nGenerating hidden tier (20 samples)...")
    hidden_phantoms = generate_phantoms(20, seed_offset=20000)
    generate_tier("hidden", hidden_phantoms, base_seed=21000)

    # Gallery images
    print("\nGenerating gallery images...")
    generate_gallery_images()

    print(f"\n{'=' * 68}")
    print("SIM benchmark dataset generation complete!")
    print(f"Output: {BENCHMARK_DIR}")
    print("=" * 68)


if __name__ == "__main__":
    main()
