#!/usr/bin/env python3
"""Generate Two-Photon Excitation Microscopy (2PEM) benchmark dataset.

Forward model:
    y = Poisson(PSF_2p * x * depth_attenuation + bg) + readout_noise

where:
    x               : fluorophore density (neuronal structures, 256x256)
    PSF_2p          : squared excitation PSF (inherently confocal-like, sigma ~1.5-3 px)
    depth_attenuation : exponential signal decay with imaging depth (scattering + absorption)
    bg              : autofluorescence / dark-count background
    readout_noise   : Gaussian readout noise from PMT/GaAsP detector

The quadratic intensity dependence of two-photon excitation means the
effective PSF is the *square* of the single-photon excitation PSF, yielding
inherent optical sectioning and a narrower effective PSF (by sqrt(2)).

Mismatch parameters:
    excitation_power   : laser excitation power factor (affects signal-to-noise)
    scattering_length  : tissue scattering mean free path (affects depth attenuation)
    pulse_dispersion   : GDD-induced pulse broadening (widens PSF, reduces peak intensity)
    noise_level        : combined shot + electronic noise scaling

Phantoms:
    Brain tissue sections: neuronal cell bodies (soma), dendritic trees,
    blood vessels, GCaMP calcium-indicator signals (active neurons with
    bright transient signals).

CPU Baseline reconstruction:
    Depth-corrected Richardson-Lucy deconvolution.
    Expected: ~22-28 dB.

Tiers:
    Public  : 12 samples (seed offset 0)
    Dev     : 20 samples (seed offset 10000)
    Hidden  : 20 samples (seed offset 20000)

Usage:
    cd datasets/benchmark/two_photon
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
from scipy.interpolate import CubicSpline

BENCHMARK_DIR = Path(__file__).resolve().parent

# -- Image dimensions --------------------------------------------------------

IMAGE_SIZE = 256          # ground truth 256x256
PIXEL_SIZE_UM = 0.5       # 0.5 um/pixel (typical 2P imaging)

# -- Mismatch ranges per tier ------------------------------------------------

SPEC = {
    "public": {
        "excitation_power":    {"min": 0.6, "max": 1.5,  "unit": "relative"},
        "scattering_length":   {"min": 150, "max": 300,  "unit": "um"},
        "pulse_dispersion":    {"min": 0.0, "max": 0.3,  "unit": "relative_GDD"},
        "noise_level":         {"min": 0.5, "max": 2.0,  "unit": "relative"},
    },
    "dev": {
        "excitation_power":    {"min": 0.4, "max": 2.0,  "unit": "relative"},
        "scattering_length":   {"min": 100, "max": 350,  "unit": "um"},
        "pulse_dispersion":    {"min": 0.0, "max": 0.5,  "unit": "relative_GDD"},
        "noise_level":         {"min": 0.5, "max": 3.0,  "unit": "relative"},
    },
    "hidden": {
        "excitation_power":    {"min": 0.3, "max": 2.5,  "unit": "relative"},
        "scattering_length":   {"min": 80,  "max": 400,  "unit": "um"},
        "pulse_dispersion":    {"min": 0.0, "max": 0.7,  "unit": "relative_GDD"},
        "noise_level":         {"min": 0.8, "max": 4.0,  "unit": "relative"},
    },
}

# -- Physics constants -------------------------------------------------------

BASE_PSF_SIGMA = 2.0       # base two-photon PSF sigma in pixels (~1 um FWHM)
BASE_DEPTH_UM = 200.0      # reference imaging depth (um)
BACKGROUND_LEVEL = 3.0     # mean background photons/pixel (autofluorescence)
READOUT_NOISE_STD = 1.5    # readout noise std (electrons)
BASE_PHOTON_SCALE = 2000.0 # mean signal photons at reference depth


# -- Smooth curve utilities --------------------------------------------------

def _smooth_curve_points(
    control_pts: np.ndarray, n_interp: int = 300
) -> np.ndarray:
    """Generate smooth curve via cubic interpolation of control points."""
    n = len(control_pts)
    if n < 3:
        return control_pts.copy()
    t = np.linspace(0, 1, n)
    t_new = np.linspace(0, 1, n_interp)
    cs_y = CubicSpline(t, control_pts[:, 0], bc_type="natural")
    cs_x = CubicSpline(t, control_pts[:, 1], bc_type="natural")
    return np.column_stack([cs_y(t_new), cs_x(t_new)])


def _clip_to_bounds(pts: np.ndarray, H: int, W: int,
                    margin: int = 2) -> np.ndarray:
    """Filter points to be within image bounds."""
    valid = (
        (pts[:, 0] >= margin) & (pts[:, 0] < H - margin)
        & (pts[:, 1] >= margin) & (pts[:, 1] < W - margin)
    )
    return pts[valid]


# -- Phantom generators (brain tissue structures) ---------------------------

def make_cell_body_phantom(
    H: int, W: int, rng: np.random.Generator,
) -> np.ndarray:
    """Neuronal cell bodies (soma): circular blobs of varying size.

    Simulates a cortical layer with scattered neuronal somata,
    some brighter (GCaMP active) than others.
    """
    x = np.zeros((H, W), dtype=np.float64)

    n_cells = rng.integers(15, 40)
    for _ in range(n_cells):
        cy = rng.uniform(10, H - 10)
        cx = rng.uniform(10, W - 10)
        radius = rng.uniform(3, 10)  # soma radius 1.5-5 um
        brightness = rng.uniform(0.3, 1.0)

        # Some cells are "active" (GCaMP calcium transient) -- brighter
        if rng.random() < 0.25:
            brightness = rng.uniform(0.7, 1.0)

        yy, xx = np.ogrid[
            max(0, int(cy - radius * 2)):min(H, int(cy + radius * 2 + 1)),
            max(0, int(cx - radius * 2)):min(W, int(cx + radius * 2 + 1)),
        ]
        dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
        cell = brightness * np.exp(-dist2 / (2 * (radius * 0.6) ** 2))
        # Hollow out center slightly for soma ring appearance
        inner_mask = dist2 < (radius * 0.3) ** 2
        cell[inner_mask] *= 0.6

        y0 = max(0, int(cy - radius * 2))
        y1 = min(H, int(cy + radius * 2 + 1))
        x0 = max(0, int(cx - radius * 2))
        x1 = min(W, int(cx + radius * 2 + 1))
        x[y0:y1, x0:x1] = np.maximum(x[y0:y1, x0:x1], cell)

    return x


def make_dendrite_phantom(
    H: int, W: int, rng: np.random.Generator,
) -> np.ndarray:
    """Dendritic trees: branching filamentous structures.

    Simulates apical dendrites extending from layer V pyramidal cells,
    with secondary and tertiary branching.
    """
    x = np.zeros((H, W), dtype=np.float64)

    n_trees = rng.integers(3, 8)
    for _ in range(n_trees):
        _draw_branching_tree(x, H, W, rng, depth=0, max_depth=3,
                             start_y=rng.uniform(H * 0.7, H * 0.9),
                             start_x=rng.uniform(W * 0.15, W * 0.85),
                             angle=rng.uniform(-np.pi / 2 - 0.5, -np.pi / 2 + 0.5),
                             length=rng.uniform(H * 0.15, H * 0.4),
                             thickness=rng.uniform(1.5, 3.0),
                             brightness=rng.uniform(0.5, 1.0))
    return x


def _draw_branching_tree(
    canvas: np.ndarray, H: int, W: int, rng: np.random.Generator,
    depth: int, max_depth: int,
    start_y: float, start_x: float,
    angle: float, length: float,
    thickness: float, brightness: float,
) -> None:
    """Recursively draw a branching dendritic tree."""
    if depth > max_depth or length < 5 or thickness < 0.5:
        return

    # Generate curved segment with control points
    n_ctrl = rng.integers(3, 6)
    ctrl = np.zeros((n_ctrl, 2))
    for j in range(n_ctrl):
        t = j / (n_ctrl - 1)
        ctrl[j, 0] = start_y + length * t * np.sin(angle) + rng.uniform(-8, 8)
        ctrl[j, 1] = start_x + length * t * np.cos(angle) + rng.uniform(-8, 8)

    curve = _smooth_curve_points(ctrl, n_interp=max(50, int(length * 3)))
    curve = _clip_to_bounds(curve, H, W, margin=1)

    # Draw along curve with Gaussian cross-section
    for py, px in curve:
        r = int(np.ceil(thickness * 2.5))
        y0 = max(0, int(py) - r)
        y1 = min(H, int(py) + r + 1)
        x0 = max(0, int(px) - r)
        x1 = min(W, int(px) + r + 1)
        yy = np.arange(y0, y1)[:, None]
        xx = np.arange(x0, x1)[None, :]
        g = brightness * np.exp(-((yy - py) ** 2 + (xx - px) ** 2) / (2 * thickness ** 2))
        canvas[y0:y1, x0:x1] = np.maximum(canvas[y0:y1, x0:x1], g)

    # Branch points
    if len(curve) > 10:
        n_branches = rng.integers(1, 3 + 1)
        for _ in range(n_branches):
            branch_idx = rng.integers(len(curve) // 3, len(curve))
            branch_start = curve[min(branch_idx, len(curve) - 1)]
            branch_angle = angle + rng.uniform(-1.0, 1.0)
            branch_length = length * rng.uniform(0.3, 0.6)
            branch_thick = thickness * rng.uniform(0.5, 0.8)
            _draw_branching_tree(
                canvas, H, W, rng,
                depth=depth + 1, max_depth=max_depth,
                start_y=branch_start[0], start_x=branch_start[1],
                angle=branch_angle, length=branch_length,
                thickness=branch_thick,
                brightness=brightness * rng.uniform(0.6, 0.9),
            )


def make_blood_vessel_phantom(
    H: int, W: int, rng: np.random.Generator,
) -> np.ndarray:
    """Blood vessels: thick branching tubes (negative contrast or labeled).

    Simulates cortical vasculature: arterioles and venules with
    Texas Red dextran or other intravascular labels.
    """
    x = np.zeros((H, W), dtype=np.float64)

    n_vessels = rng.integers(4, 10)
    for _ in range(n_vessels):
        # Main vessel
        start_edge = rng.integers(0, 4)  # which edge to start from
        if start_edge == 0:  # top
            sy, sx = 0, rng.uniform(0, W)
        elif start_edge == 1:  # bottom
            sy, sx = H - 1, rng.uniform(0, W)
        elif start_edge == 2:  # left
            sy, sx = rng.uniform(0, H), 0
        else:  # right
            sy, sx = rng.uniform(0, H), W - 1

        ey = rng.uniform(H * 0.2, H * 0.8)
        ex = rng.uniform(W * 0.2, W * 0.8)

        n_ctrl = rng.integers(4, 8)
        ctrl = np.zeros((n_ctrl, 2))
        for j in range(n_ctrl):
            t = j / (n_ctrl - 1)
            ctrl[j, 0] = sy + (ey - sy) * t + rng.uniform(-20, 20)
            ctrl[j, 1] = sx + (ex - sx) * t + rng.uniform(-20, 20)

        curve = _smooth_curve_points(ctrl, n_interp=400)
        curve = _clip_to_bounds(curve, H, W, margin=0)

        thickness = rng.uniform(2.0, 6.0)
        brightness = rng.uniform(0.4, 0.9)

        for py, px in curve:
            r = int(np.ceil(thickness * 2.5))
            y0 = max(0, int(py) - r)
            y1 = min(H, int(py) + r + 1)
            x0 = max(0, int(px) - r)
            x1 = min(W, int(px) + r + 1)
            yy = np.arange(y0, y1)[:, None]
            xx = np.arange(x0, x1)[None, :]
            g = brightness * np.exp(-((yy - py) ** 2 + (xx - px) ** 2) / (2 * thickness ** 2))
            x[y0:y1, x0:x1] = np.maximum(x[y0:y1, x0:x1], g)

    return x


def make_calcium_signal_phantom(
    H: int, W: int, rng: np.random.Generator,
) -> np.ndarray:
    """GCaMP calcium signals: mixture of soma, neuropil, and transient activity.

    Simulates a GCaMP6 two-photon calcium imaging field of view with:
    - Labeled neuronal somata at varying activity levels
    - Diffuse neuropil fluorescence background
    - A few cells showing bright calcium transients
    """
    x = np.zeros((H, W), dtype=np.float64)

    # Diffuse neuropil background (low-level fluorescence from dendrites/axons)
    neuropil = np.zeros((H, W), dtype=np.float64)
    n_neuropil_blobs = rng.integers(20, 50)
    for _ in range(n_neuropil_blobs):
        cy = rng.uniform(0, H)
        cx = rng.uniform(0, W)
        sig = rng.uniform(10, 30)
        amp = rng.uniform(0.05, 0.15)
        yy, xx = np.ogrid[0:H, 0:W]
        neuropil += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sig ** 2))
    x += np.clip(neuropil, 0, 0.2)

    # Neuronal somata
    n_cells = rng.integers(20, 50)
    for i in range(n_cells):
        cy = rng.uniform(12, H - 12)
        cx = rng.uniform(12, W - 12)
        radius = rng.uniform(4, 9)

        # Activity level: baseline vs transient
        is_active = rng.random() < 0.3  # 30% of cells active
        if is_active:
            brightness = rng.uniform(0.6, 1.0)
        else:
            brightness = rng.uniform(0.15, 0.35)

        yy, xx = np.ogrid[
            max(0, int(cy - radius * 3)):min(H, int(cy + radius * 3 + 1)),
            max(0, int(cx - radius * 3)):min(W, int(cx + radius * 3 + 1)),
        ]
        dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
        # Filled soma (GCaMP is cytoplasmic)
        cell = brightness * np.exp(-dist2 / (2 * (radius * 0.7) ** 2))
        # Nucleus exclusion (slightly dimmer center)
        nuc_mask = dist2 < (radius * 0.25) ** 2
        cell[nuc_mask] *= 0.5

        y0 = max(0, int(cy - radius * 3))
        y1 = min(H, int(cy + radius * 3 + 1))
        x0 = max(0, int(cx - radius * 3))
        x1 = min(W, int(cx + radius * 3 + 1))
        x[y0:y1, x0:x1] = np.maximum(x[y0:y1, x0:x1], cell)

    return x


# -- Phantom pool for each tier -----------------------------------------------

PHANTOM_FNS = [
    make_cell_body_phantom,
    make_dendrite_phantom,
    make_blood_vessel_phantom,
    make_calcium_signal_phantom,
]

PHANTOM_NAMES = ["cell_body", "dendrite", "blood_vessel", "calcium_signal"]


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
        x_raw = fn(IMAGE_SIZE, IMAGE_SIZE, phantom_rng)

        # Normalize to [0, 1]
        if x_raw.max() > 0:
            x_raw /= x_raw.max()
        x_true = x_raw.astype(np.float32)

        scene_name = f"{name}_{i:02d}"
        phantoms.append((x_true, scene_name))

    return phantoms


# -- Two-Photon forward model ------------------------------------------------

def make_two_photon_psf(
    H: int, W: int,
    psf_sigma: float,
) -> np.ndarray:
    """Create the effective two-photon PSF (squared Gaussian).

    The two-photon PSF is the square of the single-photon excitation PSF.
    For a Gaussian PSF with sigma, the squared PSF has effective sigma
    = sigma / sqrt(2), but we keep the specified sigma as the effective value.
    """
    cy, cx = H // 2, W // 2
    yy = np.arange(H)[:, None] - cy
    xx = np.arange(W)[None, :] - cx
    # Gaussian PSF
    psf_single = np.exp(-(yy ** 2 + xx ** 2) / (2 * psf_sigma ** 2))
    # Two-photon: square of excitation PSF
    psf_2p = psf_single ** 2
    # Normalize to sum to 1
    psf_2p /= psf_2p.sum()
    return psf_2p.astype(np.float64)


def compute_depth_attenuation(
    H: int, W: int,
    depth_um: float,
    scattering_length_um: float,
) -> np.ndarray:
    """Compute exponential depth attenuation map.

    In two-photon microscopy, signal decays exponentially with depth
    due to excitation beam scattering and absorption. The two-photon
    signal decays as exp(-2 * depth / scattering_length) because
    both excitation photons must reach the focal plane.
    """
    # Two-photon signal decays as exp(-2z/ls) for excitation beam attenuation.
    # But near-IR has much longer scattering length than visible, so the
    # effective attenuation is gentler. Use exp(-z/ls) as a practical model.
    attenuation = np.exp(-depth_um / max(scattering_length_um, 1.0))
    # Clamp minimum attenuation to avoid complete signal loss
    attenuation = max(attenuation, 0.05)
    # Add spatial variation (tissue surface not perfectly flat)
    yy = np.linspace(-0.05, 0.05, H)[:, None]
    xx = np.linspace(-0.05, 0.05, W)[None, :]
    spatial_variation = 1.0 + 0.1 * (yy + xx)
    depth_map = attenuation * np.clip(spatial_variation, 0.85, 1.15)
    return depth_map.astype(np.float64)


def two_photon_forward(
    x_true: np.ndarray,
    excitation_power: float,
    scattering_length: float,
    pulse_dispersion: float,
    noise_level: float,
    rng: np.random.Generator,
    depth_um: float = 200.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-photon forward model.

    y = Poisson(PSF_2p * x * depth_attenuation + bg) + readout_noise

    Args:
        x_true: (H, W) ground truth fluorophore density [0, 1]
        excitation_power: relative laser power (affects signal intensity)
        scattering_length: tissue scattering mean free path (um)
        pulse_dispersion: GDD-induced pulse broadening (widens PSF)
        noise_level: noise scaling factor
        rng: random generator
        depth_um: imaging depth in tissue (um)

    Returns:
        y: (H, W) float32 measured image (noisy)
        H_ideal: (H, W) float32 ideal (noiseless) convolved image
    """
    H, W = x_true.shape
    x64 = x_true.astype(np.float64)

    # 1. PSF: base sigma + pulse dispersion broadening
    # Pulse dispersion widens the temporal pulse, reducing peak intensity
    # and broadening the effective PSF
    effective_sigma = BASE_PSF_SIGMA * (1.0 + pulse_dispersion)

    # 2. Depth attenuation
    depth_atten = compute_depth_attenuation(H, W, depth_um, scattering_length)

    # 3. Two-photon signal: power^2 dependence (quadratic)
    # Two-photon fluorescence scales as I^2, where I is excitation intensity
    signal_scale = BASE_PHOTON_SCALE * (excitation_power ** 2)

    # 4. Apply PSF convolution via FFT
    # Convolve x_true with the effective 2P PSF
    psf = make_two_photon_psf(H, W, effective_sigma)
    from numpy.fft import fft2, ifft2, fftshift
    x_padded = x64
    psf_shifted = fftshift(psf)
    H_ideal = np.real(ifft2(fft2(x_padded) * fft2(psf_shifted)))
    H_ideal = np.maximum(H_ideal, 0.0)

    # 5. Apply depth attenuation and signal scaling
    H_ideal_scaled = H_ideal * depth_atten * signal_scale

    # 6. Add background (autofluorescence + dark counts)
    bg = BACKGROUND_LEVEL * noise_level
    signal_plus_bg = H_ideal_scaled + bg

    # 7. Poisson shot noise
    y = rng.poisson(np.maximum(signal_plus_bg, 0.01)).astype(np.float64)

    # 8. Readout noise (PMT/GaAsP electronic noise)
    readout_std = READOUT_NOISE_STD * noise_level
    y += rng.normal(0, readout_std, (H, W))
    y = np.maximum(y, 0)

    return y.astype(np.float32), H_ideal.astype(np.float32)


# -- CPU Baseline: Depth-corrected Richardson-Lucy deconvolution -------------

def depth_corrected_richardson_lucy(
    y: np.ndarray,
    psf_sigma_est: float = 2.5,
    n_iterations: int = 50,
    depth_correction: bool = True,
) -> np.ndarray:
    """Depth-corrected Richardson-Lucy deconvolution for two-photon data.

    Pipeline:
        1. Estimate and correct for depth-dependent attenuation
        2. Background estimation and subtraction
        3. Standard RL deconvolution with estimated 2P PSF
        4. TV-like regularization via smoothing + early stopping

    Args:
        y: (H, W) noisy measurement
        psf_sigma_est: estimated effective PSF sigma
        n_iterations: number of RL iterations
        depth_correction: whether to apply depth attenuation correction

    Returns:
        recon: (H, W) float32 reconstruction
    """
    from numpy.fft import fft2, ifft2, fftshift

    H, W = y.shape
    y64 = y.astype(np.float64)

    # Step 1: Depth correction — estimate attenuation from image statistics
    if depth_correction:
        # Estimate depth-varying attenuation from row-wise and col-wise means
        row_means = gaussian_filter(y64.mean(axis=1), sigma=30)
        col_means = gaussian_filter(y64.mean(axis=0), sigma=30)
        if row_means.max() > 0 and col_means.max() > 0:
            row_corr = row_means.max() / np.maximum(row_means, row_means.max() * 0.05)
            col_corr = col_means.max() / np.maximum(col_means, col_means.max() * 0.05)
            row_corr = np.clip(row_corr, 0.5, 3.0)
            col_corr = np.clip(col_corr, 0.5, 3.0)
            depth_corr_map = np.sqrt(row_corr[:, None] * col_corr[None, :])
            y_corrected = y64 * depth_corr_map
        else:
            y_corrected = y64.copy()
    else:
        y_corrected = y64.copy()

    # Step 2: Background subtraction (robust percentile-based)
    bg_est = 0.0
    if np.any(y_corrected > 0):
        bg_est = float(np.percentile(y_corrected, 3))
    y_corrected = np.maximum(y_corrected - bg_est, 0)

    # Step 3: Pre-denoise to reduce noise before RL (mild Gaussian)
    y_denoised = gaussian_filter(y_corrected, sigma=0.7)
    y_denoised = np.maximum(y_denoised, 0)

    # Step 4: Build PSF for RL
    psf = make_two_photon_psf(H, W, psf_sigma_est)
    psf_ft = fft2(fftshift(psf))
    psf_ft_conj = np.conj(psf_ft)

    # Step 5: Richardson-Lucy iterations with damping
    # RL update: x_{k+1} = x_k * (PSF^T * (y / (PSF * x_k + eps)))
    # Use pre-denoised y as the target (reduces noise amplification)
    estimate = np.maximum(y_denoised, 1e-6)
    eps = 1e-8

    for iteration in range(n_iterations):
        # Forward: PSF * estimate
        est_ft = fft2(estimate)
        blurred = np.real(ifft2(est_ft * psf_ft))
        blurred = np.maximum(blurred, eps)

        # Ratio
        ratio = y_denoised / blurred

        # Back-project: PSF^T * ratio
        ratio_ft = fft2(ratio)
        correction = np.real(ifft2(ratio_ft * psf_ft_conj))
        correction = np.maximum(correction, 0)

        # Damped update (prevents oscillation)
        damping = 0.9
        estimate = estimate * (1.0 - damping + damping * correction)
        estimate = np.maximum(estimate, 0)

        # Regularization: mild smoothing every 15 iterations
        if (iteration + 1) % 15 == 0:
            estimate = gaussian_filter(estimate, sigma=0.3)

    # Final mild smoothing for noise suppression
    estimate = gaussian_filter(estimate, sigma=0.4)

    # Normalize to [0, 1]
    if estimate.max() > 0:
        estimate /= estimate.max()

    return estimate.astype(np.float32)


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
    # Windowed SSIM (11x11 Gaussian window)
    from scipy.ndimage import uniform_filter
    win_size = 11
    mu_x = uniform_filter(gt64, size=win_size)
    mu_y = uniform_filter(recon64, size=win_size)
    mu_x_sq = uniform_filter(gt64 ** 2, size=win_size)
    mu_y_sq = uniform_filter(recon64 ** 2, size=win_size)
    mu_xy = uniform_filter(gt64 * recon64, size=win_size)
    var_x = mu_x_sq - mu_x ** 2
    var_y = mu_y_sq - mu_y ** 2
    cov_xy = mu_xy - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / \
               ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2))
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


# -- Tier generation ---------------------------------------------------------

def sample_mismatch(rng: np.random.Generator, spec: dict) -> dict:
    result = {}
    for k, v in spec.items():
        val = float(rng.uniform(v["min"], v["max"]))
        result[k] = val
    return result


def generate_tier(
    tier: str,
    phantoms: list[tuple[np.ndarray, str]],
    base_seed: int,
) -> tuple[float, float]:
    """Generate one tier of the two-photon benchmark.

    Returns:
        (mean_psnr, mean_ssim) for the tier.
    """
    spec_ranges = SPEC[tier]
    tier_dir = BENCHMARK_DIR / tier
    images_dir = tier_dir / "images"
    if tier_dir.exists():
        shutil.rmtree(tier_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    h5_path = tier_dir / f"two_photon_challenge_{tier}.h5"
    rng = np.random.default_rng(base_seed)
    true_specs = {}
    all_psnrs = []
    all_ssims = []

    with h5py.File(h5_path, "w") as f:
        f.attrs["description"] = (
            f"PWM Two-Photon Microscopy benchmark -- {tier} tier "
            f"(two-photon excitation fluorescence)"
        )
        f.attrs["spec_ranges"] = json.dumps(spec_ranges)
        f.attrs["forward_model"] = (
            "y = Poisson(PSF_2p * x * depth_attenuation + bg) + readout_noise"
        )
        f.attrs["geometry"] = json.dumps({
            "image_size": IMAGE_SIZE,
            "pixel_size_um": PIXEL_SIZE_UM,
            "base_psf_sigma_px": BASE_PSF_SIGMA,
            "base_depth_um": BASE_DEPTH_UM,
            "background_level": BACKGROUND_LEVEL,
        })

        for idx, (x_true, scene_name) in enumerate(phantoms):
            key = f"sample_{idx:02d}"
            print(f"  [{tier}] Generating {key} ({scene_name})...", end="", flush=True)

            mis = sample_mismatch(rng, spec_ranges)
            # Imaging depth varies per sample (50-300 um is typical for 2P)
            depth_um = float(rng.uniform(50, 300))
            true_specs[key] = {**mis, "depth_um": depth_um}

            # Forward model
            y, H_ideal = two_photon_forward(
                x_true,
                excitation_power=mis["excitation_power"],
                scattering_length=mis["scattering_length"],
                pulse_dispersion=mis["pulse_dispersion"],
                noise_level=mis["noise_level"],
                rng=rng,
                depth_um=depth_um,
            )

            # CPU baseline reconstruction
            eff_sigma_est = BASE_PSF_SIGMA * (1.0 + mis["pulse_dispersion"] * 0.5)
            recon = depth_corrected_richardson_lucy(
                y,
                psf_sigma_est=eff_sigma_est,
                n_iterations=30,
                depth_correction=True,
            )

            psnr = compute_psnr(x_true, recon)
            ssim = compute_ssim(x_true, recon)
            all_psnrs.append(psnr)
            all_ssims.append(ssim)

            # Save to HDF5
            grp = f.create_group(key)
            grp.create_dataset("x_true", data=x_true, compression="gzip")
            grp.create_dataset("y", data=y, compression="gzip")
            grp.create_dataset("H_ideal", data=H_ideal, compression="gzip")
            grp.create_dataset("reconstruction_baseline", data=recon,
                               compression="gzip")
            grp.attrs["metadata"] = json.dumps({
                "scene": scene_name,
                "shape": list(x_true.shape),
                "psnr_baseline": float(psnr),
                "ssim_baseline": float(ssim),
            })
            grp.attrs["true_spec"] = json.dumps({**mis, "depth_um": depth_um})
            grp.attrs["spec_ranges"] = json.dumps(spec_ranges)

            # Save per-sample images
            sample_dir = images_dir / f"sample_{idx:02d}_{scene_name}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            _save_png(x_true, sample_dir / "gt.png")
            _save_png(y, sample_dir / "measurement.png", percentile_clip=True)
            _save_png(recon, sample_dir / "recon.png")
            _save_png(H_ideal, sample_dir / "H_ideal.png", percentile_clip=True)
            with open(sample_dir / "spec.json", "w") as sf:
                json.dump({
                    "scene": scene_name,
                    "spec_ranges": spec_ranges,
                    "true_spec": {**mis, "depth_um": depth_um},
                    "psnr_baseline": psnr,
                    "ssim_baseline": ssim,
                }, sf, indent=2)

            print(f"  PSNR={psnr:.2f} dB, SSIM={ssim:.3f}  "
                  f"power={mis['excitation_power']:.2f}  "
                  f"scat_len={mis['scattering_length']:.0f} um  "
                  f"dispersion={mis['pulse_dispersion']:.2f}  "
                  f"noise={mis['noise_level']:.2f}  "
                  f"depth={depth_um:.0f} um")

    # Save spec files
    with open(tier_dir / "spec.json", "w") as sf:
        json.dump(spec_ranges, sf, indent=2)
    if tier == "public":
        with open(tier_dir / "true_spec.json", "w") as tf:
            json.dump(true_specs, tf, indent=2)

    mean_psnr = float(np.mean(all_psnrs))
    mean_ssim = float(np.mean(all_ssims))
    print(f"  [{tier}] {len(phantoms)} samples | "
          f"Mean PSNR={mean_psnr:.2f} dB | Mean SSIM={mean_ssim:.3f}")
    print(f"  [{tier}] HDF5 -> {h5_path.name}")
    return mean_psnr, mean_ssim


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
        / "benchmark_gallery" / "two_photon"
    )

    h5_path = BENCHMARK_DIR / "public" / "two_photon_challenge_public.h5"
    if not h5_path.exists():
        print("  [gallery] Public HDF5 not found, skipping gallery generation.")
        return

    # Pick 4 diverse samples: cell_body, dendrite, blood_vessel, calcium_signal
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

            # gt.png -- ground truth fluorophore density
            _save_png(x_true, scene_dir / "gt.png")

            # measurement_I.png -- noisy two-photon measurement
            _save_png(y, scene_dir / "measurement_I.png", percentile_clip=True)

            # measurement_II.png -- ideal PSF-convolved image (no noise)
            _save_png(H_ideal, scene_dir / "measurement_II.png", percentile_clip=True)

            # recon_I.png -- baseline RL reconstruction
            _save_png(recon, scene_dir / "recon_I.png")

            # recon_II.png -- residual |GT - recon|
            diff = np.abs(x_true - recon)
            _save_png(diff, scene_dir / "recon_II.png")

            print(f"  [gallery] scene_{scene_idx:02d} -> {scene_dir}")


# -- Main --------------------------------------------------------------------

def main() -> None:
    print("Two-Photon Microscopy Benchmark Dataset Generator")
    print("=" * 68)
    print(f"Output: {BENCHMARK_DIR}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}, pixel size: {PIXEL_SIZE_UM} um\n")

    # Public tier (12 samples)
    print("Generating public tier (12 samples)...")
    public_phantoms = generate_phantoms(12, seed_offset=0)
    pub_psnr, pub_ssim = generate_tier("public", public_phantoms, base_seed=1000)

    # Dev tier (20 samples)
    print("\nGenerating dev tier (20 samples)...")
    dev_phantoms = generate_phantoms(20, seed_offset=10000)
    dev_psnr, dev_ssim = generate_tier("dev", dev_phantoms, base_seed=11000)

    # Hidden tier (20 samples)
    print("\nGenerating hidden tier (20 samples)...")
    hidden_phantoms = generate_phantoms(20, seed_offset=20000)
    hid_psnr, hid_ssim = generate_tier("hidden", hidden_phantoms, base_seed=21000)

    # Gallery images
    print("\nGenerating gallery images...")
    generate_gallery_images()

    print(f"\n{'=' * 68}")
    print("Two-Photon Microscopy benchmark dataset generation complete!")
    print(f"Output: {BENCHMARK_DIR}")
    print(f"\nBaseline summary:")
    print(f"  Public:  PSNR={pub_psnr:.2f} dB, SSIM={pub_ssim:.3f}")
    print(f"  Dev:     PSNR={dev_psnr:.2f} dB, SSIM={dev_ssim:.3f}")
    print(f"  Hidden:  PSNR={hid_psnr:.2f} dB, SSIM={hid_ssim:.3f}")
    print("=" * 68)


if __name__ == "__main__":
    main()
