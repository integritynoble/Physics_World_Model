#!/usr/bin/env python3
"""Generate Widefield Fluorescence Microscopy benchmark dataset.

Forward model:
    y = Poisson(PSF * x + out_of_focus_haze + autofluorescence) + readout_noise

where:
    x               : ground truth fluorescence distribution (256x256)
    PSF             : widefield PSF (Gaussian with sigma ~3-4 pixels, broader
                      than confocal because the entire z-volume is excited and
                      no pinhole rejects out-of-focus light)
    out_of_focus_haze : blurred contribution from adjacent z-planes, convolved
                      with a much broader (depth-dependent) defocused PSF.
                      This is the dominant artefact in widefield microscopy.
    autofluorescence: spatially smooth, sample-dependent background from
                      intrinsic tissue fluorescence (NAD(P)H, flavins, lipofuscin)
    readout_noise   : additive Gaussian from sCMOS camera electronics

Mismatch parameters (vary per sample):
    NA_error         : deviation in effective numerical aperture (changes PSF width)
    defocus_amount   : axial defocus distance from nominal focal plane (um)
    background_level : autofluorescence + ambient background (photons/pixel)
    photobleaching   : fraction of signal lost due to photobleaching during
                       exposure (spatially non-uniform, centre-of-field dominant)

Phantoms:
    Fluorescently labelled cells with three staining patterns:
    - DAPI nuclei (large bright elliptical regions)
    - Actin filaments (thin curved networks, phalloidin-ATTO488)
    - Mitochondrial networks (tubular meshworks, MitoTracker)

CPU Baseline:
    Richardson-Lucy deconvolution (50 iterations) with background subtraction.
    Expected ~22-28 dB PSNR.

Tiers:
    Public  : 12 samples (seed offset 0)
    Dev     : 20 samples (seed offset 10000)
    Hidden  : 20 samples (seed offset 20000)

Usage:
    cd datasets/benchmark/widefield
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

# -- Image dimensions --------------------------------------------------------

IMAGE_SIZE = 256            # 256x256 single z-slice
PIXEL_SIZE_UM = 0.100       # 100 nm/pixel (typical widefield at 60x-100x)

# -- Optical parameters (defaults) ------------------------------------------

WAVELENGTH_EX_UM = 0.488    # excitation wavelength (488 nm, GFP/FITC)
WAVELENGTH_EM_UM = 0.525    # emission wavelength (525 nm)
NA_NOMINAL = 1.25           # numerical aperture (oil immersion, widefield)
N_IMMERSION = 1.515         # refractive index (oil)

# Widefield PSF is broader than confocal because there is no pinhole to
# reject out-of-focus light. Lateral FWHM ~ 0.51 * lambda / NA for
# Rayleigh, and we model the PSF as a Gaussian with sigma ~3-4 pixels.
WIDEFIELD_PSF_SIGMA_PX = 3.5   # default lateral PSF sigma (pixels)

# -- Mismatch ranges per tier -----------------------------------------------

SPEC = {
    "public": {
        "NA_error":           {"min": -0.05, "max": 0.05, "unit": "delta-NA"},
        "defocus_amount":     {"min": 0.0,   "max": 0.5,  "unit": "um"},
        "background_level":   {"min": 10,    "max": 40,   "unit": "photons/pixel"},
        "photobleaching":     {"min": 0.0,   "max": 0.10, "unit": "fraction"},
        "noise_level":        {"min": 500,   "max": 2000, "unit": "peak photons"},
    },
    "dev": {
        "NA_error":           {"min": -0.10, "max": 0.10, "unit": "delta-NA"},
        "defocus_amount":     {"min": 0.0,   "max": 1.0,  "unit": "um"},
        "background_level":   {"min": 15,    "max": 60,   "unit": "photons/pixel"},
        "photobleaching":     {"min": 0.0,   "max": 0.20, "unit": "fraction"},
        "noise_level":        {"min": 300,   "max": 2000, "unit": "peak photons"},
    },
    "hidden": {
        "NA_error":           {"min": -0.15, "max": 0.15, "unit": "delta-NA"},
        "defocus_amount":     {"min": 0.0,   "max": 1.5,  "unit": "um"},
        "background_level":   {"min": 20,    "max": 80,   "unit": "photons/pixel"},
        "photobleaching":     {"min": 0.05,  "max": 0.30, "unit": "fraction"},
        "noise_level":        {"min": 200,   "max": 1500, "unit": "peak photons"},
    },
}


# ============================================================================
# PSF generation
# ============================================================================

def make_widefield_psf(
    size: int,
    na_error: float = 0.0,
    defocus_um: float = 0.0,
) -> np.ndarray:
    """Generate a 2D widefield fluorescence PSF (Gaussian model).

    The widefield PSF is broader than confocal because no pinhole rejects
    out-of-focus light. We model the in-focus lateral PSF as a Gaussian
    whose width depends on effective NA. Defocus broadens the PSF further
    (quadratic with defocus distance).

    Args:
        size: PSF patch size (pixels, should be odd)
        na_error: deviation from nominal NA (changes PSF width)
        defocus_um: axial defocus distance (um)

    Returns:
        psf: (size, size) float64, normalised to sum=1
    """
    if size % 2 == 0:
        size += 1

    na_eff = max(NA_NOMINAL + na_error, 0.3)  # clamp to physical minimum

    # Lateral PSF sigma from diffraction limit:
    # sigma ~ 0.21 * lambda_em / NA / pixel_size
    sigma_lateral = 0.21 * WAVELENGTH_EM_UM / na_eff / PIXEL_SIZE_UM

    # Defocus broadening: sigma_defocus ~ defocus * NA / (n * pixel_size)
    # This is a simplified Born & Wolf defocus model
    sigma_defocus = 0.0
    if defocus_um > 0:
        sigma_defocus = defocus_um * na_eff / (N_IMMERSION * PIXEL_SIZE_UM) * 0.5

    sigma_total = np.sqrt(sigma_lateral**2 + sigma_defocus**2)

    # Build 2D Gaussian PSF
    half = size // 2
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float64)
    r2 = xx**2 + yy**2
    psf = np.exp(-r2 / (2 * sigma_total**2))

    # Normalize
    psf_sum = psf.sum()
    if psf_sum > 0:
        psf /= psf_sum

    return psf


def make_out_of_focus_haze_psf(
    size: int,
    defocus_um: float = 0.0,
    scale_factor: float = 5.0,
) -> np.ndarray:
    """Generate a wider PSF for out-of-focus haze from other z-planes.

    In widefield microscopy, the entire z-volume is illuminated, so all
    planes contribute a blurred haze to the image. This is modelled as
    a very broad Gaussian (the averaged defocused PSF from multiple
    z-planes above and below focus).

    Args:
        size: PSF patch size
        defocus_um: current defocus amount (increases haze spread)
        scale_factor: baseline scale relative to in-focus PSF

    Returns:
        psf: (size, size) float64, normalised to sum=1
    """
    if size % 2 == 0:
        size += 1

    # Out-of-focus haze PSF sigma (much broader than in-focus)
    sigma_base = WIDEFIELD_PSF_SIGMA_PX * scale_factor
    # Additional spread from defocus
    sigma_defocus = defocus_um * 3.0 / PIXEL_SIZE_UM if defocus_um > 0 else 0.0
    sigma_oof = np.sqrt(sigma_base**2 + sigma_defocus**2)

    half = size // 2
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float64)
    r2 = xx**2 + yy**2
    psf = np.exp(-r2 / (2 * sigma_oof**2))

    psf_sum = psf.sum()
    if psf_sum > 0:
        psf /= psf_sum

    return psf


# ============================================================================
# Phantom generators (fluorescently labelled cells)
# ============================================================================

def _smooth_curve_points(
    control_pts: np.ndarray, n_interp: int = 500,
) -> np.ndarray:
    """Smooth curve via cubic interpolation of control points."""
    from scipy.interpolate import CubicSpline

    n = len(control_pts)
    if n < 3:
        return control_pts
    t = np.linspace(0, 1, n)
    t_new = np.linspace(0, 1, n_interp)
    cs_y = CubicSpline(t, control_pts[:, 0], bc_type="natural")
    cs_x = CubicSpline(t, control_pts[:, 1], bc_type="natural")
    return np.column_stack([cs_y(t_new), cs_x(t_new)])


def _draw_thick_curve(
    canvas: np.ndarray,
    curve: np.ndarray,
    thickness: float,
    intensity: float,
) -> None:
    """Draw a thick curve on a canvas using Gaussian cross-section."""
    H, W = canvas.shape
    for py, px in curve:
        iy, ix = int(round(py)), int(round(px))
        r = int(np.ceil(thickness * 3))
        y0, y1 = max(0, iy - r), min(H, iy + r + 1)
        x0, x1 = max(0, ix - r), min(W, ix + r + 1)
        if y1 <= y0 or x1 <= x0:
            continue
        yy = np.arange(y0, y1)[:, None]
        xx = np.arange(x0, x1)[None, :]
        d2 = (yy - py)**2 + (xx - px)**2
        canvas[y0:y1, x0:x1] += intensity * np.exp(-d2 / (2 * thickness**2))


def make_dapi_nuclei_phantom(
    H: int, W: int, rng: np.random.Generator,
) -> np.ndarray:
    """DAPI-stained nuclei: bright elliptical regions with chromatin texture.

    DAPI (4',6-diamidino-2-phenylindole) binds to A-T rich regions of dsDNA
    in the nucleus. In widefield fluorescence, DAPI-stained nuclei appear as
    large (~10-20 um) bright regions with internal chromatin texture. The
    nuclear envelope is slightly brighter than the interior.

    Returns:
        x_true: (H, W) float64 ground truth fluorescence [0, 1]
    """
    x_true = np.zeros((H, W), dtype=np.float64)
    n_nuclei = rng.integers(6, 18)
    yy, xx = np.mgrid[:H, :W]

    for _ in range(n_nuclei):
        cy = rng.uniform(20, H - 20)
        cx = rng.uniform(20, W - 20)
        # Semi-axes (nuclei are 10-20 um, so 100-200 pixels at 100 nm/px,
        # but scaled for 256x256 FOV)
        a = rng.uniform(10, 28)
        b = rng.uniform(0.6, 1.0) * a
        angle = rng.uniform(0, np.pi)
        base_intensity = rng.uniform(0.4, 1.0)

        # Rotated ellipse distance
        dy = yy - cy
        dx = xx - cx
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        r_rot = ((dy * cos_a + dx * sin_a) / a)**2 + \
                ((-dy * sin_a + dx * cos_a) / b)**2

        # Nucleus mask with smooth boundary
        in_nucleus = r_rot < 1.0
        falloff = np.exp(-np.clip(r_rot - 0.75, 0, None) * 8.0)

        # Internal chromatin texture (heterochromatin = bright patches)
        texture = rng.standard_normal((H, W))
        texture = gaussian_filter(texture, sigma=rng.uniform(2.5, 6.0))
        texture = (texture - texture.min()) / (texture.max() - texture.min() + 1e-8)

        # Nuclear envelope: slight bright ring at boundary
        envelope = np.exp(-((r_rot - 0.90)**2) / (2 * 0.04**2)) * 0.3

        contribution = base_intensity * falloff * (0.5 + 0.5 * texture) + envelope
        x_true += contribution * in_nucleus

    # Add some dim cytoplasmic autofluorescence around nuclei
    cyto_bg = gaussian_filter(x_true, sigma=12) * 0.08
    x_true += cyto_bg

    return x_true


def make_actin_filament_phantom(
    H: int, W: int, rng: np.random.Generator,
) -> np.ndarray:
    """Actin filaments stained with phalloidin: thin curved networks.

    Phalloidin conjugates (e.g., Alexa Fluor 488-Phalloidin) bind F-actin
    filaments. In widefield microscopy, stress fibres appear as bright linear
    bundles spanning the cell, while cortical actin forms a mesh at cell edges.
    Lamellipodia show bright leading edges with a meshwork interior.

    Returns:
        x_true: (H, W) float64 ground truth fluorescence [0, 1]
    """
    x_true = np.zeros((H, W), dtype=np.float64)

    # Generate 2-4 cells, each with stress fibres and cortical actin
    n_cells = rng.integers(2, 5)

    for _ in range(n_cells):
        # Cell body centre
        cell_cy = rng.uniform(H * 0.15, H * 0.85)
        cell_cx = rng.uniform(W * 0.15, W * 0.85)
        cell_radius = rng.uniform(30, 60)

        # Cell boundary (cortical actin ring)
        n_boundary = 800
        boundary_width = rng.uniform(1.5, 3.0)
        for i in range(n_boundary):
            theta = 2 * np.pi * i / n_boundary
            # Slightly irregular boundary
            r_vary = cell_radius * (1.0 + 0.1 * np.sin(5 * theta + rng.uniform(0, 2 * np.pi)))
            py = cell_cy + r_vary * np.sin(theta)
            px = cell_cx + r_vary * np.cos(theta)
            iy, ix = int(round(py)), int(round(px))
            r = int(np.ceil(boundary_width * 2.5))
            y0, y1 = max(0, iy - r), min(H, iy + r + 1)
            x0, x1 = max(0, ix - r), min(W, ix + r + 1)
            if y1 <= y0 or x1 <= x0:
                continue
            yyg = np.arange(y0, y1)[:, None]
            xxg = np.arange(x0, x1)[None, :]
            d2 = (yyg - py)**2 + (xxg - px)**2
            x_true[y0:y1, x0:x1] += 0.6 * np.exp(-d2 / (2 * boundary_width**2))

        # Stress fibres: thick bundles running roughly parallel across the cell
        n_fibres = rng.integers(5, 12)
        fibre_angle_base = rng.uniform(0, np.pi)

        for _ in range(n_fibres):
            angle = fibre_angle_base + rng.normal(0, 0.3)
            length = cell_radius * rng.uniform(1.2, 2.0)

            # Starting point offset from cell centre
            offset = rng.uniform(-cell_radius * 0.6, cell_radius * 0.6)
            start_y = cell_cy + offset * np.cos(angle)
            start_x = cell_cx - offset * np.sin(angle)

            n_ctrl = rng.integers(4, 7)
            ctrl = np.zeros((n_ctrl, 2))
            for j in range(n_ctrl):
                t = j / max(n_ctrl - 1, 1) - 0.5
                ctrl[j, 0] = start_y + length * t * np.sin(angle) + rng.normal(0, 3)
                ctrl[j, 1] = start_x + length * t * np.cos(angle) + rng.normal(0, 3)

            curve = _smooth_curve_points(ctrl, n_interp=400)
            valid = (
                (curve[:, 0] >= 0) & (curve[:, 0] < H)
                & (curve[:, 1] >= 0) & (curve[:, 1] < W)
            )
            curve = curve[valid]
            if len(curve) < 5:
                continue

            thickness = rng.uniform(1.2, 2.5)
            intensity = rng.uniform(0.5, 0.9)
            _draw_thick_curve(x_true, curve, thickness, intensity)

    # Faint cytoplasmic staining (G-actin background)
    cyto_bg = gaussian_filter(rng.uniform(0, 0.05, (H, W)), sigma=8)
    x_true += cyto_bg

    return x_true


def make_mitochondria_phantom(
    H: int, W: int, rng: np.random.Generator,
) -> np.ndarray:
    """Mitochondrial networks stained with MitoTracker: tubular meshworks.

    MitoTracker dyes (Green, Red, Deep Red) accumulate in active
    mitochondria. In widefield microscopy, mitochondria appear as bright
    tubular and punctate structures forming interconnected networks within
    the cytoplasm. The network radiates from the perinuclear region and
    extends toward the cell periphery.

    Returns:
        x_true: (H, W) float64 ground truth fluorescence [0, 1]
    """
    x_true = np.zeros((H, W), dtype=np.float64)

    # Generate 2-3 cells with mitochondrial networks
    n_cells = rng.integers(2, 4)

    for _ in range(n_cells):
        cell_cy = rng.uniform(H * 0.20, H * 0.80)
        cell_cx = rng.uniform(W * 0.20, W * 0.80)
        cell_radius = rng.uniform(35, 65)

        # Perinuclear concentration: nucleus at cell centre
        nuc_radius = cell_radius * rng.uniform(0.25, 0.40)

        # Mitochondrial tubules: short curved segments radiating from
        # perinuclear region
        n_tubules = rng.integers(30, 70)

        for _ in range(n_tubules):
            # Start near nucleus boundary, extend outward
            start_angle = rng.uniform(0, 2 * np.pi)
            start_r = nuc_radius * rng.uniform(0.8, 1.3)
            start_y = cell_cy + start_r * np.sin(start_angle)
            start_x = cell_cx + start_r * np.cos(start_angle)

            # Tubule direction: roughly radial with some randomness
            tubule_angle = start_angle + rng.normal(0, 0.6)
            tubule_length = rng.uniform(8, 35)

            n_ctrl = rng.integers(3, 6)
            ctrl = np.zeros((n_ctrl, 2))
            for j in range(n_ctrl):
                t = j / max(n_ctrl - 1, 1)
                ctrl[j, 0] = start_y + tubule_length * t * np.sin(tubule_angle) + rng.normal(0, 2)
                ctrl[j, 1] = start_x + tubule_length * t * np.cos(tubule_angle) + rng.normal(0, 2)

            if n_ctrl >= 3:
                curve = _smooth_curve_points(ctrl, n_interp=150)
            else:
                curve = ctrl

            # Filter valid points within cell boundary
            dist_from_centre = np.sqrt(
                (curve[:, 0] - cell_cy)**2 + (curve[:, 1] - cell_cx)**2
            )
            valid = (
                (curve[:, 0] >= 0) & (curve[:, 0] < H)
                & (curve[:, 1] >= 0) & (curve[:, 1] < W)
                & (dist_from_centre < cell_radius * 0.95)
            )
            curve = curve[valid]
            if len(curve) < 3:
                continue

            thickness = rng.uniform(0.8, 1.8)
            intensity = rng.uniform(0.4, 0.9)
            _draw_thick_curve(x_true, curve, thickness, intensity)

        # Add some punctate mitochondria (fragmented / fission)
        n_puncta = rng.integers(10, 25)
        for _ in range(n_puncta):
            r = rng.uniform(nuc_radius * 0.5, cell_radius * 0.9)
            theta = rng.uniform(0, 2 * np.pi)
            py = cell_cy + r * np.sin(theta)
            px = cell_cx + r * np.cos(theta)
            iy, ix = int(round(py)), int(round(px))
            if 0 <= iy < H and 0 <= ix < W:
                sigma = rng.uniform(0.8, 1.5)
                intensity = rng.uniform(0.5, 0.9)
                rr = int(np.ceil(sigma * 3))
                y0, y1 = max(0, iy - rr), min(H, iy + rr + 1)
                x0, x1 = max(0, ix - rr), min(W, ix + rr + 1)
                if y1 > y0 and x1 > x0:
                    yyg = np.arange(y0, y1)[:, None]
                    xxg = np.arange(x0, x1)[None, :]
                    d2 = (yyg - py)**2 + (xxg - px)**2
                    x_true[y0:y1, x0:x1] += intensity * np.exp(-d2 / (2 * sigma**2))

    return x_true


PHANTOM_FNS = [make_dapi_nuclei_phantom, make_actin_filament_phantom, make_mitochondria_phantom]
PHANTOM_NAMES = ["dapi_nuclei", "actin_filaments", "mitochondria"]


# ============================================================================
# Widefield forward model
# ============================================================================

def widefield_forward(
    x_true: np.ndarray,
    na_error: float,
    defocus_um: float,
    background_level: float,
    photobleaching: float,
    noise_level: float,
    rng: np.random.Generator,
    readout_noise_std: float = 3.5,
    oof_fraction: float = 0.30,
) -> tuple[np.ndarray, np.ndarray]:
    """Widefield fluorescence forward model.

    y = Poisson(PSF * x + out_of_focus_haze + autofluorescence) + readout_noise

    In widefield microscopy, the entire specimen is illuminated uniformly.
    Every z-plane emits fluorescence that is detected by the camera.
    Only the in-focus plane is sharply imaged; all other planes contribute
    a blurred haze that is the dominant image degradation.

    Args:
        x_true: (H, W) ground truth fluorescence [0, 1]
        na_error: deviation from nominal NA
        defocus_um: axial defocus distance (um)
        background_level: autofluorescence + ambient background (photons/pixel)
        photobleaching: fraction of signal lost
        noise_level: peak photon count (scales signal amplitude)
        rng: random number generator
        readout_noise_std: sCMOS readout noise std (electrons)
        oof_fraction: fraction of total signal from out-of-focus planes

    Returns:
        y: (H, W) float32 noisy measurement
        H_ideal: (H, W) float32 noiseless blurred image
    """
    H, W = x_true.shape
    psf_size = 51  # larger kernel for widefield (broader PSF)

    # Widefield PSF (in-focus)
    psf_wf = make_widefield_psf(psf_size, na_error, defocus_um)

    # Out-of-focus haze PSF (very broad)
    oof_scale = 4.0 + abs(defocus_um) * 2.0 + abs(na_error) * 5.0
    psf_oof = make_out_of_focus_haze_psf(psf_size, defocus_um, scale_factor=oof_scale)

    # Apply photobleaching: spatially non-uniform (centre-of-field stronger)
    if photobleaching > 0:
        # Illumination is typically Koehler (uniform), but photobleaching
        # depends on cumulative exposure which is proportional to illumination
        # intensity -> centre can bleach more with arc-lamp
        yg, xg = np.mgrid[:H, :W].astype(np.float64)
        dist_centre = np.sqrt((yg - H / 2)**2 + (xg - W / 2)**2)
        # Gaussian bleaching profile (more at centre)
        bleach_profile = np.exp(-dist_centre**2 / (2 * (H * 0.4)**2))
        bleach_map = 1.0 - photobleaching * (0.5 + 0.5 * bleach_profile)
        bleach_map = np.clip(bleach_map, 0.0, 1.0)
        # Add random per-pixel noise to bleaching
        bleach_map *= (1.0 + rng.normal(0, 0.02, (H, W)))
        bleach_map = np.clip(bleach_map, 0.0, 1.0)
        x_bleached = x_true * bleach_map
    else:
        x_bleached = x_true.copy()

    # Scale to photon counts
    x_photons = x_bleached.astype(np.float64) * noise_level

    # In-focus signal: convolve with widefield PSF
    in_focus = fftconvolve(x_photons, psf_wf, mode="same")

    # Out-of-focus haze: blurred version from other z-planes
    # Simulate as a broadly-blurred version of the same sample (which in a
    # thick specimen would be from adjacent z-planes with correlated structure)
    oof_signal = fftconvolve(x_photons, psf_oof, mode="same") * oof_fraction

    # Autofluorescence: smooth spatially-varying background
    # (from intrinsic tissue fluorescence, lipofuscin, NAD(P)H, etc.)
    auto_fluor = np.abs(gaussian_filter(rng.standard_normal((H, W)), sigma=30))
    auto_fluor = auto_fluor / (auto_fluor.max() + 1e-8) * background_level * 0.3
    auto_fluor += background_level * 0.7  # uniform component

    # Total ideal signal (noiseless)
    ideal = np.maximum(in_focus + oof_signal + auto_fluor, 0.01)
    H_ideal = ideal.astype(np.float32)

    # Poisson noise (shot noise from photon counting)
    y = rng.poisson(np.maximum(ideal, 0.01)).astype(np.float64)

    # Readout noise (Gaussian, sCMOS camera)
    y += rng.normal(0, readout_noise_std, (H, W))
    y = np.maximum(y, 0)

    return y.astype(np.float32), H_ideal


# ============================================================================
# CPU Baseline: Richardson-Lucy deconvolution + background subtraction
# ============================================================================

def richardson_lucy_deconv(
    y: np.ndarray,
    psf: np.ndarray,
    n_iter: int = 50,
    clip_negative: bool = True,
) -> np.ndarray:
    """Richardson-Lucy deconvolution (ML for Poisson noise).

    Standard iterative deconvolution algorithm:
        x_{k+1} = x_k * (PSF^T * (y / (PSF * x_k + eps)))

    This is the most common CPU baseline for widefield fluorescence
    deconvolution, used in ImageJ/Fiji DeconvolutionLab2, Huygens, etc.

    Args:
        y: (H, W) noisy measurement
        psf: (K, K) PSF kernel (normalised to sum=1)
        n_iter: number of RL iterations
        clip_negative: clip result to be non-negative

    Returns:
        recon: (H, W) float32 reconstruction
    """
    y64 = np.maximum(y.astype(np.float64), 0)
    psf64 = psf.astype(np.float64)
    psf_flip = psf64[::-1, ::-1].copy()

    # Initialize with the measurement (common for RL)
    x_est = np.maximum(y64.copy(), 1e-6)

    eps = 1e-10
    for _ in range(n_iter):
        blurred = fftconvolve(x_est, psf64, mode="same")
        blurred = np.maximum(blurred, eps)
        ratio = y64 / blurred
        correction = fftconvolve(ratio, psf_flip, mode="same")
        x_est *= correction
        if clip_negative:
            x_est = np.maximum(x_est, 0)

    return x_est.astype(np.float32)


def baseline_reconstruct(y: np.ndarray) -> np.ndarray:
    """CPU baseline: Richardson-Lucy with background subtraction.

    Uses an estimated widefield PSF (assuming nominal parameters = mild
    mismatch) and performs background subtraction before RL deconvolution.

    Args:
        y: (H, W) noisy measurement

    Returns:
        recon: (H, W) float32, normalised to [0, 1]
    """
    # Background estimation: rolling-ball style (large Gaussian filter)
    bg_estimate = gaussian_filter(y.astype(np.float64), sigma=25)
    bg_estimate = np.minimum(bg_estimate, y.astype(np.float64))

    # Subtract background
    y_sub = np.maximum(y.astype(np.float64) - bg_estimate * 0.8, 0)

    # Estimated PSF (assume nominal parameters -- slight mismatch)
    psf_est = make_widefield_psf(
        size=51,
        na_error=0.0,      # assume nominal NA
        defocus_um=0.0,     # assume in focus
    )

    recon = richardson_lucy_deconv(y_sub.astype(np.float32), psf_est, n_iter=50)

    # Normalise to [0, 1]
    rmin, rmax = float(recon.min()), float(recon.max())
    if rmax - rmin > 1e-8:
        recon = (recon - rmin) / (rmax - rmin)
    else:
        recon = np.zeros_like(recon)

    return recon.astype(np.float32)


# ============================================================================
# Metrics
# ============================================================================

def compute_psnr(gt: np.ndarray, recon: np.ndarray) -> float:
    gt64 = gt.astype(np.float64)
    recon64 = recon.astype(np.float64)
    mse = np.mean((gt64 - recon64)**2)
    if mse < 1e-12:
        return 100.0
    data_range = float(gt64.max() - gt64.min())
    if data_range < 1e-12:
        return 0.0
    return float(10 * np.log10(data_range**2 / mse))


def compute_ssim(gt: np.ndarray, recon: np.ndarray) -> float:
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
    ssim_val = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / \
               ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))
    return float(ssim_val)


# ============================================================================
# Image helpers
# ============================================================================

def _norm(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo + 1e-8)


def _save_png(arr: np.ndarray, path: Path, percentile_clip: bool = False) -> None:
    a = arr.astype(np.float64)
    if percentile_clip and a.max() > 0:
        nonzero = a[a > 0]
        if len(nonzero) > 0:
            lo, hi = np.percentile(nonzero, [1, 99])
            a = np.clip(a, lo, hi)
    Image.fromarray(
        np.clip(_norm(a) * 255, 0, 255).astype(np.uint8), "L"
    ).save(str(path))


# ============================================================================
# Phantom pool per tier
# ============================================================================

def generate_phantoms(
    n: int, seed_offset: int,
) -> list[tuple[np.ndarray, str]]:
    """Generate phantom ground truths with different seeds per tier.

    Returns list of (x_true, scene_name).
    """
    phantoms = []
    for i in range(n):
        fn_idx = i % len(PHANTOM_FNS)
        fn = PHANTOM_FNS[fn_idx]
        name = PHANTOM_NAMES[fn_idx]

        phantom_rng = np.random.default_rng(seed_offset + i)
        x_raw = fn(IMAGE_SIZE, IMAGE_SIZE, phantom_rng)

        # Normalise to [0, 1]
        xmin, xmax = float(x_raw.min()), float(x_raw.max())
        if xmax - xmin > 1e-8:
            x_true = ((x_raw - xmin) / (xmax - xmin)).astype(np.float32)
        else:
            x_true = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

        scene_name = f"{name}_{i:02d}"
        phantoms.append((x_true, scene_name))

    return phantoms


# ============================================================================
# Tier generation
# ============================================================================

def sample_mismatch(rng: np.random.Generator, spec: dict) -> dict:
    result = {}
    for k, v in spec.items():
        val = float(rng.uniform(v["min"], v["max"]))
        if k in ("noise_level", "background_level"):
            val = round(val)
        result[k] = val
    return result


def generate_tier(
    tier: str,
    phantoms: list[tuple[np.ndarray, str]],
    base_seed: int,
) -> None:
    """Generate one tier of the Widefield Fluorescence benchmark."""
    spec_ranges = SPEC[tier]
    tier_dir = BENCHMARK_DIR / tier
    images_dir = tier_dir / "images"
    if tier_dir.exists():
        shutil.rmtree(tier_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    h5_path = tier_dir / f"widefield_challenge_{tier}.h5"
    rng = np.random.default_rng(base_seed)
    true_specs = {}
    all_psnrs = []
    all_ssims = []

    with h5py.File(h5_path, "w") as f:
        f.attrs["description"] = (
            f"PWM Widefield Fluorescence Microscopy benchmark -- {tier} tier "
            f"(epifluorescence with out-of-focus haze)"
        )
        f.attrs["spec_ranges"] = json.dumps(spec_ranges)
        f.attrs["forward_model"] = (
            "y = Poisson(PSF * x + out_of_focus_haze + autofluorescence) + readout_noise"
        )
        f.attrs["geometry"] = json.dumps({
            "image_size": IMAGE_SIZE,
            "pixel_size_um": PIXEL_SIZE_UM,
            "wavelength_ex_um": WAVELENGTH_EX_UM,
            "wavelength_em_um": WAVELENGTH_EM_UM,
            "NA_nominal": NA_NOMINAL,
            "n_immersion": N_IMMERSION,
            "widefield_psf_sigma_px": WIDEFIELD_PSF_SIGMA_PX,
        })

        for idx, (x_true, scene_name) in enumerate(phantoms):
            key = f"sample_{idx:02d}"
            print(f"  [{tier}] Generating {key} ({scene_name})...",
                  end="", flush=True)

            mis = sample_mismatch(rng, spec_ranges)
            true_specs[key] = {**mis, "scene": scene_name}

            # Forward model
            y, H_ideal = widefield_forward(
                x_true,
                na_error=mis["NA_error"],
                defocus_um=mis["defocus_amount"],
                background_level=mis["background_level"],
                photobleaching=mis["photobleaching"],
                noise_level=mis["noise_level"],
                rng=rng,
            )

            # CPU baseline: RL + background subtraction
            recon = baseline_reconstruct(y)

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
            grp.attrs["true_spec"] = json.dumps(mis)
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
                    "true_spec": mis,
                    "psnr_baseline": psnr,
                    "ssim_baseline": ssim,
                }, sf, indent=2)

            print(f"  PSNR={psnr:.2f} dB, SSIM={ssim:.3f}  "
                  f"NA_err={mis['NA_error']:.3f}  "
                  f"defocus={mis['defocus_amount']:.3f} um  "
                  f"bg={mis['background_level']}  "
                  f"bleach={mis['photobleaching']:.3f}  "
                  f"photons={mis['noise_level']}")

        # Final HDF5 summary
        f.attrs["baseline_method"] = (
            "Richardson-Lucy deconvolution (50 iter) + background subtraction"
        )
        f.attrs["mean_psnr_baseline"] = float(np.mean(all_psnrs))
        f.attrs["mean_ssim_baseline"] = float(np.mean(all_ssims))

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


# ============================================================================
# Gallery image generation
# ============================================================================

def generate_gallery_images() -> None:
    """Generate gallery images for the platform benchmark page.

    Creates scene_00 through scene_03 with:
        gt.png, measurement_I.png, measurement_II.png,
        recon_I.png, recon_II.png
    """
    gallery_base = (
        BENCHMARK_DIR.parent.parent.parent
        / "platform" / "pwm_platform" / "static" / "img"
        / "benchmark_gallery" / "widefield"
    )

    h5_path = BENCHMARK_DIR / "public" / "widefield_challenge_public.h5"
    if not h5_path.exists():
        print("  [gallery] Public HDF5 not found, skipping gallery generation.")
        return

    # Pick 4 samples: indices 0, 1, 2, 3
    # (dapi_nuclei, actin_filaments, mitochondria, dapi_nuclei again)
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
            y_meas = grp["y"][:]
            H_ideal = grp["H_ideal"][:]
            recon = grp["reconstruction_baseline"][:]

            # gt.png -- ground truth fluorescence
            _save_png(x_true, scene_dir / "gt.png")

            # measurement_I.png -- noisy widefield acquisition
            _save_png(y_meas, scene_dir / "measurement_I.png",
                      percentile_clip=True)

            # measurement_II.png -- noiseless ideal (blurred + haze + bg)
            _save_png(H_ideal, scene_dir / "measurement_II.png",
                      percentile_clip=True)

            # recon_I.png -- Richardson-Lucy baseline
            _save_png(recon, scene_dir / "recon_I.png")

            # recon_II.png -- |GT - recon| difference map
            diff = np.abs(x_true.astype(np.float64) - recon.astype(np.float64))
            _save_png(diff, scene_dir / "recon_II.png")

            print(f"  [gallery] scene_{scene_idx:02d} -> {scene_dir}")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("Widefield Fluorescence Microscopy Benchmark Dataset Generator")
    print("=" * 68)
    print(f"Output: {BENCHMARK_DIR}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Pixel size: {PIXEL_SIZE_UM} um")
    print(f"Optics: NA={NA_NOMINAL}, lambda_ex={WAVELENGTH_EX_UM * 1000:.0f} nm, "
          f"lambda_em={WAVELENGTH_EM_UM * 1000:.0f} nm")
    print(f"Widefield PSF sigma: {WIDEFIELD_PSF_SIGMA_PX} px\n")

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
    print("Widefield Fluorescence Microscopy benchmark dataset generation complete!")
    print(f"Output: {BENCHMARK_DIR}")
    print("=" * 68)


if __name__ == "__main__":
    main()
