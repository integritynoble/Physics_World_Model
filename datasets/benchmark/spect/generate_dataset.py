#!/usr/bin/env python3
"""Generate SPECT (Single-Photon Emission Computed Tomography) benchmark dataset.

Forward model (2D SPECT with depth-dependent collimator blur):
    y_theta = P_theta * A_theta * x + s + n

where:
    x           : radionuclide activity distribution (ground truth, 256x256)
    A_theta     : diagonal attenuation matrix for angle theta
                  (A_theta)_ii = exp(-integral mu(r) dr along ray i)
    P_theta     : projector with depth-dependent collimator-detector response (CDR)
                  modelled as Gaussian blur with FWHM increasing with source depth
    s           : scatter contribution (smooth, spatially varying)
    n           ~ Poisson(P_theta * A_theta * x + s)
    y_theta     : measured counts in detector bins at angle theta

Key SPECT-specific physics (vs PET):
    1. Single gamma-ray (no coincidence) -> direction from collimator geometry
    2. Depth-dependent resolution: FWHM(d) = FWHM_0 * (1 + d / rotation_radius)
    3. Single-photon attenuation: exp(-int mu dr) along ONE ray (not two-sided)
    4. Higher scatter fraction than PET (0.20-0.50 vs 0.10-0.40)
    5. Parallel-hole collimator geometry -> Radon transform with blur kernel

Geometry:
    256 angles over [0, 360) degrees (full rotation, SPECT standard)
    367 detector bins (default for 256x256 Radon)
    Parallel-hole collimator

Mismatch parameters (from check.md):
    mu_map_scale        : attenuation coeff scaling; nominal 1.0, perturbed 0.85-1.15
    cdr_fwhm_mm         : collimator-detector response FWHM at reference depth; 7.5-12 mm
    scatter_fraction    : scatter-to-primary ratio; 0.05-0.25
    rotation_radius_mm  : camera orbit radius; 180-230 mm

Phantoms:
    Cardiac perfusion (Tc-99m sestamibi) -- primary SPECT application
    Brain perfusion (Tc-99m HMPAO / I-123 IMP)
    Bone scan (Tc-99m MDP)

Tiers:
    Public  : 12 samples (4 cardiac + 4 brain + 4 bone)  seed offset 0
    Dev     : 20 samples (augmented variants)             seed offset 10000
    Hidden  : 20 samples (adversarial: subtle defects)    seed offset 20000

Usage:
    python3 generate_dataset.py
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy.ndimage import rotate as nd_rotate, gaussian_filter, zoom as nd_zoom

BENCHMARK_DIR = Path(__file__).resolve().parent

# -- Geometry -----------------------------------------------------------------

IMAGE_SIZE = 256
N_ANGLES = 256
N_DET = 367  # default for 256x256 Radon transform (ceil(sqrt(2)*256))

# SPECT uses FULL 360-degree rotation (vs PET 180-degree)
ANGLE_RANGE_DEG = 360.0

# Physical scale: 400 mm FOV (body SPECT typical)
FOV_MM = 400.0
PIXEL_SIZE_MM = FOV_MM / IMAGE_SIZE  # ~1.56 mm/px

# -- Seed offsets per tier (from project convention) --------------------------
TIER_SEED_OFFSETS = {"public": 0, "dev": 10000, "hidden": 20000}

# -- Mismatch ranges per tier (from check.md) ---------------------------------

SPEC = {
    "public": {
        "mu_map_scale":       {"min": 0.95, "max": 1.05, "unit": ""},
        "cdr_fwhm_mm":        {"min": 8.5,  "max": 10.5, "unit": "mm"},
        "scatter_fraction":   {"min": 0.08, "max": 0.15, "unit": ""},
        "rotation_radius_mm": {"min": 190,  "max": 210,  "unit": "mm"},
    },
    "dev": {
        "mu_map_scale":       {"min": 0.90, "max": 1.10, "unit": ""},
        "cdr_fwhm_mm":        {"min": 7.5,  "max": 11.5, "unit": "mm"},
        "scatter_fraction":   {"min": 0.05, "max": 0.20, "unit": ""},
        "rotation_radius_mm": {"min": 185,  "max": 220,  "unit": "mm"},
    },
    "hidden": {
        "mu_map_scale":       {"min": 0.85, "max": 1.15, "unit": ""},
        "cdr_fwhm_mm":        {"min": 7.5,  "max": 12.0, "unit": "mm"},
        "scatter_fraction":   {"min": 0.05, "max": 0.25, "unit": ""},
        "rotation_radius_mm": {"min": 180,  "max": 230,  "unit": "mm"},
    },
}


# -- Radon Transform (optimized) ---------------------------------------------

def _pad_image(image: np.ndarray) -> tuple[np.ndarray, int]:
    """Pad image to diagonal size for rotation without clipping."""
    H, W = image.shape
    diag = int(np.ceil(np.sqrt(H**2 + W**2)))
    if diag % 2 != 0:
        diag += 1
    pad_h = (diag - H) // 2
    pad_w = (diag - W) // 2
    padded = np.zeros((diag, diag), dtype=np.float64)
    padded[pad_h:pad_h + H, pad_w:pad_w + W] = image
    return padded, diag


def radon_transform(image: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Parallel-beam Radon transform using scipy.ndimage.rotate.

    Args:
        image: (H, W) float64 input image
        theta: array of projection angles in degrees

    Returns:
        sinogram: (len(theta), N_det) float64
    """
    padded, diag = _pad_image(image.astype(np.float64))
    sinogram = np.zeros((len(theta), diag), dtype=np.float64)
    for i, angle in enumerate(theta):
        rotated = nd_rotate(padded, angle, reshape=False, order=1,
                            mode='constant', cval=0.0)
        sinogram[i] = rotated.sum(axis=0)
    return sinogram


def radon_with_cdr(
    image: np.ndarray,
    theta: np.ndarray,
    cdr_fwhm_mm: float,
    rotation_radius_mm: float,
) -> np.ndarray:
    """Radon transform with depth-dependent collimator-detector response (CDR).

    The CDR FWHM increases with source-to-detector distance:
        FWHM(d) = cdr_fwhm_mm * (1 + d / rotation_radius_mm)

    Optimized: groups rows into sigma bins to reduce per-row overhead.

    Args:
        image: (H, W) float64 input image
        theta: projection angles in degrees
        cdr_fwhm_mm: FWHM at detector face in mm
        rotation_radius_mm: camera orbit radius in mm

    Returns:
        sinogram: (len(theta), N_det) float64 with depth-dependent blur
    """
    padded, diag = _pad_image(image.astype(np.float64))
    sinogram = np.zeros((len(theta), diag), dtype=np.float64)
    center = diag // 2

    # Precompute sigma for each row
    row_indices = np.arange(diag)
    depths_mm = np.abs(row_indices - center).astype(np.float64) * PIXEL_SIZE_MM
    fwhms = cdr_fwhm_mm * (1.0 + depths_mm / rotation_radius_mm)
    sigmas = fwhms / PIXEL_SIZE_MM / 2.355  # FWHM to sigma in pixels

    # Group rows into bins by sigma for batch processing
    n_bins = 12
    sigma_min, sigma_max = sigmas.min(), sigmas.max()
    bin_edges = np.linspace(sigma_min - 0.01, sigma_max + 0.01, n_bins + 1)
    row_bins = np.digitize(sigmas, bin_edges) - 1  # 0-based bin index
    # Precompute bin average sigmas and row masks
    bin_sigmas = []
    bin_row_masks = []
    for b in range(n_bins):
        mask = row_bins == b
        if mask.any():
            bin_sigmas.append(float(sigmas[mask].mean()))
            bin_row_masks.append(mask)

    for i, angle in enumerate(theta):
        rotated = nd_rotate(padded, angle, reshape=False, order=1,
                            mode='constant', cval=0.0)
        blurred = rotated.copy()
        for avg_sigma, row_mask in zip(bin_sigmas, bin_row_masks):
            if avg_sigma > 0.3:
                rows = rotated[row_mask]  # shape: (n_rows_in_bin, diag)
                for local_idx, global_idx in enumerate(
                        np.where(row_mask)[0]):
                    blurred[global_idx] = gaussian_filter(
                        rotated[global_idx], sigma=avg_sigma)
        sinogram[i] = blurred.sum(axis=0)

    return sinogram


def compute_attenuation_sinogram(
    mu_map: np.ndarray,
    theta: np.ndarray,
) -> np.ndarray:
    """Compute single-photon attenuation factors for SPECT.

    Returns:
        attn_factors: (N_angles, N_det) float64 in [0, 1]
    """
    sino_mu = radon_transform(mu_map, theta)
    pixel_size_cm = PIXEL_SIZE_MM / 10.0
    sino_mu_physical = sino_mu * pixel_size_cm
    attn_factors = np.exp(-sino_mu_physical)
    return attn_factors


# -- FBP Reconstruction -------------------------------------------------------

def _ramp_filter(n: int) -> np.ndarray:
    """Ram-Lak (ramp) filter with Hamming window for FBP."""
    freq = np.fft.fftfreq(n)
    filt = np.abs(freq) * 2.0
    hamming = 0.54 + 0.46 * np.cos(
        np.pi * freq / (np.abs(freq).max() + 1e-10))
    filt *= hamming
    return filt


def fbp_reconstruct(sinogram: np.ndarray, theta: np.ndarray,
                     output_size: int = IMAGE_SIZE) -> np.ndarray:
    """Filtered Back-Projection (FBP) reconstruction.

    For SPECT, FBP + Chang correction is the standard analytical method.

    Args:
        sinogram: (N_angles, N_det) float64
        theta: projection angles in degrees
        output_size: output image size

    Returns:
        recon: (output_size, output_size) float64
    """
    n_angles, n_det = sinogram.shape
    sinogram = sinogram.astype(np.float64)

    # Apply ramp filter
    n_fft = max(64, int(2 ** np.ceil(np.log2(2 * n_det))))
    ramp = _ramp_filter(n_fft)

    filtered = np.zeros((n_angles, n_det), dtype=np.float64)
    for i in range(n_angles):
        proj = np.zeros(n_fft, dtype=np.float64)
        proj[:n_det] = sinogram[i]
        proj_fft = np.fft.fft(proj)
        proj_fft *= ramp
        proj_filtered = np.real(np.fft.ifft(proj_fft))[:n_det]
        filtered[i] = proj_filtered

    # Back-projection
    diag = n_det
    recon = np.zeros((diag, diag), dtype=np.float64)
    center = diag // 2
    y_grid, x_grid = np.mgrid[:diag, :diag] - center
    det_center = n_det // 2

    for i, angle in enumerate(theta):
        angle_rad = np.deg2rad(angle)
        t = x_grid * np.cos(angle_rad) + y_grid * np.sin(angle_rad) + det_center
        t0 = np.floor(t).astype(int)
        t1 = t0 + 1
        w = t - t0
        valid = (t0 >= 0) & (t1 < n_det)
        proj_line = filtered[i]
        vals = np.where(valid,
                        (1 - w) * proj_line[np.clip(t0, 0, n_det - 1)] +
                        w * proj_line[np.clip(t1, 0, n_det - 1)], 0.0)
        recon += vals

    recon *= np.pi / (2 * n_angles)

    # Crop to output size
    crop_start = (diag - output_size) // 2
    recon = recon[crop_start:crop_start + output_size,
                  crop_start:crop_start + output_size]

    return np.maximum(recon, 0.0)


def chang_attenuation_correction(
    recon: np.ndarray,
    mu_map: np.ndarray,
    theta: np.ndarray,
) -> np.ndarray:
    """Apply Chang first-order attenuation correction to FBP reconstruction.

    Vectorized implementation: for each pixel, compute the average attenuation
    factor over a subset of projection angles using vectorized ray tracing.

    Chang, L.T. (1978) IEEE TNS 25(1):638-643.

    Args:
        recon: (H, W) FBP reconstruction
        mu_map: (H, W) attenuation map [cm^-1]
        theta: projection angles in degrees

    Returns:
        corrected: (H, W) attenuation-corrected reconstruction
    """
    H, W = recon.shape
    pixel_size_cm = PIXEL_SIZE_MM / 10.0

    # Subsample angles for speed (every 8th angle)
    theta_sub = theta[::8]

    center_y, center_x = H / 2.0, W / 2.0
    correction = np.ones((H, W), dtype=np.float64)

    # For each subsample angle, compute cumulative mu along rays
    for angle in theta_sub:
        angle_rad = np.deg2rad(angle)
        dx = np.cos(angle_rad)
        dy = np.sin(angle_rad)

        # For each pixel, integrate mu from pixel to image edge
        # Use a simplified approach: compute line integrals via
        # the Radon of mu_map and map back
        pass  # The sinogram-based approach below is faster

    # Alternative fast approach: use sinogram of mu_map
    # For each pixel (i,j), the attenuation at angle theta is
    # exp(-integral from (i,j) to edge along theta direction)
    # Average over all angles gives Chang correction factor
    sino_mu = radon_transform(mu_map, theta_sub)
    pixel_size_cm_val = PIXEL_SIZE_MM / 10.0

    # For each pixel, project to detector coordinate and look up
    # cumulative attenuation. This is an approximation using the
    # sinogram values.
    n_det = sino_mu.shape[1]
    diag = n_det
    pad = (diag - H) // 2
    det_center = n_det // 2

    total_attn = np.zeros((H, W), dtype=np.float64)
    n_valid = 0

    for idx, angle in enumerate(theta_sub):
        angle_rad = np.deg2rad(angle)
        # Map each pixel to its detector bin
        yy = np.arange(H).reshape(-1, 1) + pad - diag // 2
        xx = np.arange(W).reshape(1, -1) + pad - diag // 2
        t = (xx * np.cos(angle_rad) +
             yy * np.sin(angle_rad) + det_center).astype(np.float64)

        # Bilinear interpolation of mu sinogram
        t0 = np.floor(t).astype(int)
        t1 = t0 + 1
        w = t - t0
        valid = (t0 >= 0) & (t1 < n_det)

        mu_line = sino_mu[idx] * pixel_size_cm_val
        atten_at_pixel = np.where(
            valid,
            (1 - w) * mu_line[np.clip(t0, 0, n_det - 1)] +
            w * mu_line[np.clip(t1, 0, n_det - 1)],
            0.0)
        # Approximate: the pixel sees roughly half the total
        # line integral through it
        total_attn += np.exp(-atten_at_pixel * 0.5)
        n_valid += 1

    if n_valid > 0:
        avg_attn = total_attn / n_valid
        avg_attn = np.maximum(avg_attn, 0.01)
        correction = avg_attn

    corrected = recon / correction
    return np.maximum(corrected, 0.0)


# -- Phantom Generators (SPECT-specific) --------------------------------------

def _ellipse_mask(H: int, W: int, cx: float, cy: float,
                  a: float, b: float, angle_deg: float) -> np.ndarray:
    """Generate a binary ellipse mask (coordinates in normalized [-1, 1]).
    Returns boolean array."""
    yy = np.linspace(-1.0, 1.0, H)[:, None]
    xx = np.linspace(-1.0, 1.0, W)[None, :]
    ca, sa = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
    yr = (yy - cy) * ca - (xx - cx) * sa
    xr = (yy - cy) * sa + (xx - cx) * ca
    return (xr / a)**2 + (yr / b)**2 <= 1.0


def make_cardiac_perfusion_phantom(
    H: int, W: int, seed: int, variant: int = 0
) -> tuple[np.ndarray, np.ndarray, str]:
    """Generate a cardiac perfusion SPECT phantom (Tc-99m sestamibi).

    Cardiac SPECT is the most common clinical SPECT application.
    Models myocardial perfusion with possible ischemic defects.

    Returns:
        activity: (H, W) float64 [0, ~1]
        mu_map:   (H, W) float64 attenuation coefficients [cm^-1]
        name:     scene name
    """
    rng = np.random.default_rng(seed)

    activity = np.zeros((H, W), dtype=np.float64)
    mu_map = np.zeros((H, W), dtype=np.float64)

    # Chest outline (torso cross-section)
    chest = _ellipse_mask(H, W, 0.0, 0.0, 0.44, 0.36, 0)
    mu_map[chest] = 0.151  # soft tissue at 140 keV (Tc-99m)
    activity[chest] = 0.08  # low background uptake

    # Ribs/sternum (high attenuation bone)
    for rib_y in [-0.15, -0.05, 0.05, 0.15]:
        rib_l = _ellipse_mask(H, W, -0.38, rib_y, 0.04, 0.015, -5) & chest
        rib_r = _ellipse_mask(H, W, 0.38, rib_y, 0.04, 0.015, 5) & chest
        mu_map[rib_l] = 0.28  # bone at 140 keV
        mu_map[rib_r] = 0.28
        activity[rib_l] = 0.02
        activity[rib_r] = 0.02

    # Spine (posterior, dense bone)
    spine = _ellipse_mask(
        H, W, 0.0, 0.25 + variant * 0.005, 0.06, 0.05, 0) & chest
    mu_map[spine] = 0.28
    activity[spine] = 0.04

    # Lungs (very low attenuation, very low uptake)
    lung_l = _ellipse_mask(
        H, W, -0.20, -0.02 + variant * 0.01, 0.14, 0.22, -4) & chest
    lung_r = _ellipse_mask(
        H, W, 0.20, -0.02 + variant * 0.01, 0.14, 0.22, 4) & chest
    for lung in [lung_l, lung_r]:
        mu_map[lung] = 0.045  # inflated lung at 140 keV
        activity[lung] = 0.02

    # Myocardium (the main feature in cardiac SPECT)
    heart_cx = -0.06 + variant * 0.01
    heart_cy = -0.03 + variant * 0.005
    heart_angle = 12 + variant * 3

    # Left ventricle myocardium (thick ring)
    lv_outer = _ellipse_mask(
        H, W, heart_cx, heart_cy,
        0.14 + variant * 0.003, 0.13, heart_angle)
    lv_inner = _ellipse_mask(
        H, W, heart_cx, heart_cy,
        0.08 + variant * 0.002, 0.07, heart_angle)
    myocardium = (lv_outer & ~lv_inner) & chest
    lv_cavity = lv_inner & chest

    # Normal myocardial uptake (Tc-99m sestamibi)
    base_myo_uptake = 0.95 + rng.uniform(-0.05, 0.05)
    activity[myocardium] = base_myo_uptake
    activity[lv_cavity] = 0.15  # blood pool (low in delayed imaging)

    # Right ventricle (thinner wall, lower uptake)
    rv_outer = _ellipse_mask(
        H, W, heart_cx + 0.12, heart_cy + 0.02,
        0.06, 0.10, heart_angle + 10) & chest
    rv_inner = _ellipse_mask(
        H, W, heart_cx + 0.12, heart_cy + 0.02,
        0.04, 0.07, heart_angle + 10) & chest
    rv_wall = rv_outer & ~rv_inner
    rv_wall = rv_wall & ~lv_outer
    activity[rv_wall] = 0.35 + rng.uniform(-0.05, 0.05)

    # Perfusion defects (0-3 segments with reduced uptake)
    n_defects = rng.integers(0, 4)
    for _ in range(n_defects):
        angle_start = rng.uniform(0, 360)
        angle_span = rng.uniform(25, 80)
        y_coords, x_coords = np.where(myocardium)
        if len(y_coords) == 0:
            continue
        ctr_y = int((heart_cy + 1.0) * H / 2)
        ctr_x = int((heart_cx + 1.0) * W / 2)
        angles = np.degrees(
            np.arctan2(y_coords - ctr_y, x_coords - ctr_x)) % 360
        in_sector = ((angles >= angle_start) &
                     (angles < angle_start + angle_span))
        if in_sector.sum() > 0:
            defect_severity = rng.uniform(0.15, 0.65)
            activity[y_coords[in_sector], x_coords[in_sector]] = (
                defect_severity)

    # Liver (high uptake in sestamibi, appears inferior to heart)
    liver = _ellipse_mask(H, W, 0.12, 0.12, 0.18, 0.08, -8) & chest
    liver = liver & ~lv_outer & ~rv_outer
    activity[liver] = 0.55 + rng.uniform(-0.10, 0.10)
    mu_map[liver] = 0.158  # liver slightly denser

    # Stomach/bowel (variable uptake)
    stomach = _ellipse_mask(H, W, -0.10, 0.15, 0.06, 0.04, 10) & chest
    activity[stomach] = rng.uniform(0.20, 0.60)

    # Smooth slightly
    activity = gaussian_filter(activity, sigma=0.8)
    activity = np.clip(activity, 0.0, None)
    if activity.max() > 0:
        activity /= activity.max()

    return activity, mu_map, f"cardiac_{variant:02d}"


def make_brain_perfusion_phantom(
    H: int, W: int, seed: int, variant: int = 0
) -> tuple[np.ndarray, np.ndarray, str]:
    """Generate a brain perfusion SPECT phantom (Tc-99m HMPAO / I-123 IMP).

    Brain SPECT measures regional cerebral blood flow.
    Gray matter has 3-4x higher perfusion than white matter.

    Returns:
        activity: (H, W) float64 [0, ~1]
        mu_map:   (H, W) float64 attenuation coefficients [cm^-1]
        name:     scene name
    """
    rng = np.random.default_rng(seed)

    activity = np.zeros((H, W), dtype=np.float64)
    mu_map = np.zeros((H, W), dtype=np.float64)

    # Skull (bone: high attenuation)
    skull_outer = _ellipse_mask(
        H, W, 0.0, 0.0,
        0.40 + variant * 0.008, 0.46 + variant * 0.005, 0)
    skull_inner = _ellipse_mask(
        H, W, 0.0, 0.0,
        0.36 + variant * 0.008, 0.42 + variant * 0.005, 0)
    bone = skull_outer & ~skull_inner
    brain = skull_inner.copy()

    mu_map[bone] = 0.28   # bone at 140 keV
    mu_map[brain] = 0.151  # soft tissue at 140 keV

    # White matter: moderate perfusion
    activity[brain] = 0.30 + rng.uniform(-0.03, 0.03)

    # Gray matter cortex: high perfusion (3-4x white matter)
    cortex_outer = _ellipse_mask(H, W, 0.0, 0.0, 0.35, 0.41, 0)
    cortex_inner = _ellipse_mask(H, W, 0.0, 0.0, 0.29, 0.35, 0)
    cortex = cortex_outer & ~cortex_inner & brain
    activity[cortex] = 0.92 + rng.uniform(-0.05, 0.05)

    # Deep gray matter nuclei (high perfusion)
    caudate_l = _ellipse_mask(
        H, W, -0.08, 0.04, 0.03, 0.07, -8 + variant * 2) & brain
    caudate_r = _ellipse_mask(
        H, W, 0.08, 0.04, 0.03, 0.07, 8 - variant * 2) & brain
    putamen_l = _ellipse_mask(
        H, W, -0.14, 0.0, 0.04, 0.025, 0) & brain
    putamen_r = _ellipse_mask(
        H, W, 0.14, 0.0, 0.04, 0.025, 0) & brain
    thalamus_l = _ellipse_mask(
        H, W, -0.05, -0.04, 0.035, 0.025, 0) & brain
    thalamus_r = _ellipse_mask(
        H, W, 0.05, -0.04, 0.035, 0.025, 0) & brain

    for structure in [caudate_l, caudate_r, putamen_l, putamen_r]:
        uptake = 0.88 + rng.uniform(-0.05, 0.05)
        activity[structure] = uptake
    for structure in [thalamus_l, thalamus_r]:
        uptake = 0.82 + rng.uniform(-0.05, 0.05)
        activity[structure] = uptake

    # Cerebellum (high perfusion)
    cerebellum = _ellipse_mask(
        H, W, 0.0, -0.30, 0.18, 0.08, 0) & brain
    activity[cerebellum] = 0.95 + rng.uniform(-0.05, 0.05)

    # Ventricles (CSF: no perfusion)
    vent_l = _ellipse_mask(
        H, W, -0.025, 0.05, 0.015, 0.055, -5) & brain
    vent_r = _ellipse_mask(
        H, W, 0.025, 0.05, 0.015, 0.055, 5) & brain
    activity[vent_l] = 0.05
    activity[vent_r] = 0.05
    mu_map[vent_l] = 0.151
    mu_map[vent_r] = 0.151

    # Perfusion defects (stroke, dementia patterns)
    n_defects = rng.integers(0, 3)
    for _ in range(n_defects):
        dx = rng.uniform(-0.25, 0.25)
        dy = rng.uniform(-0.30, 0.20)
        dr = rng.uniform(0.03, 0.08)
        defect = _ellipse_mask(
            H, W, dx, dy, dr,
            dr * rng.uniform(0.5, 1.5),
            rng.uniform(-45, 45)) & brain
        if defect.sum() > 20:
            activity[defect] = rng.uniform(0.15, 0.45)

    activity = gaussian_filter(activity, sigma=1.0)
    activity = np.clip(activity, 0.0, None)
    if activity.max() > 0:
        activity /= activity.max()

    return activity, mu_map, f"brain_{variant:02d}"


def make_bone_scan_phantom(
    H: int, W: int, seed: int, variant: int = 0
) -> tuple[np.ndarray, np.ndarray, str]:
    """Generate a bone scan SPECT phantom (Tc-99m MDP).

    Bone SPECT shows skeletal metabolic activity. Hot spots indicate
    metastases, fractures, or infection.

    Returns:
        activity: (H, W) float64 [0, ~1]
        mu_map:   (H, W) float64 attenuation coefficients [cm^-1]
        name:     scene name
    """
    rng = np.random.default_rng(seed)

    activity = np.zeros((H, W), dtype=np.float64)
    mu_map = np.zeros((H, W), dtype=np.float64)

    # Body outline (torso cross-section)
    body = _ellipse_mask(H, W, 0.0, 0.0, 0.42, 0.34, 0)
    mu_map[body] = 0.151
    activity[body] = 0.10  # soft tissue background

    # Spine (vertebral body -- high bone uptake)
    spine = _ellipse_mask(
        H, W, 0.0, 0.22 + variant * 0.005, 0.06, 0.05, 0) & body
    mu_map[spine] = 0.28
    activity[spine] = 0.70 + rng.uniform(-0.05, 0.05)

    # Spinous process (posterior)
    spinous = _ellipse_mask(
        H, W, 0.0, 0.30 + variant * 0.005, 0.02, 0.03, 0) & body
    mu_map[spinous] = 0.28
    activity[spinous] = 0.55

    # Ribs (bilateral, moderate uptake)
    for side in [-1, 1]:
        for rib_idx in range(5):
            rib_y = -0.18 + rib_idx * 0.08 + rng.uniform(-0.01, 0.01)
            rib_x = side * (0.30 + rng.uniform(-0.03, 0.03))
            rib_angle = side * (10 + rib_idx * 2)
            rib = _ellipse_mask(
                H, W, rib_x, rib_y, 0.10, 0.012, rib_angle) & body
            mu_map[rib] = 0.28
            activity[rib] = 0.50 + rng.uniform(-0.05, 0.10)

    # Sternum (anterior)
    sternum = _ellipse_mask(H, W, 0.0, -0.28, 0.03, 0.08, 0) & body
    mu_map[sternum] = 0.28
    activity[sternum] = 0.55 + rng.uniform(-0.05, 0.05)

    # Scapulae (lateral/posterior)
    scap_l = _ellipse_mask(H, W, -0.32, 0.08, 0.04, 0.12, -15) & body
    scap_r = _ellipse_mask(H, W, 0.32, 0.08, 0.04, 0.12, 15) & body
    for scap in [scap_l, scap_r]:
        mu_map[scap] = 0.28
        activity[scap] = 0.45 + rng.uniform(-0.05, 0.05)

    # Lungs (low attenuation, very low uptake)
    lung_l = _ellipse_mask(
        H, W, -0.18, -0.02, 0.12, 0.18, -3) & body
    lung_r = _ellipse_mask(
        H, W, 0.18, -0.02, 0.12, 0.18, 3) & body
    for lung in [lung_l, lung_r]:
        mu_map[lung] = 0.045
        activity[lung] = 0.03

    # Kidneys (high MDP clearance)
    kidney_l = _ellipse_mask(
        H, W, -0.14, 0.12, 0.04, 0.055, -8) & body
    kidney_r = _ellipse_mask(
        H, W, 0.14, 0.12, 0.04, 0.055, 8) & body
    activity[kidney_l] = 0.60 + rng.uniform(-0.05, 0.10)
    activity[kidney_r] = 0.60 + rng.uniform(-0.05, 0.10)

    # Bone metastases / hot spots (1-5 focal lesions)
    n_mets = rng.integers(1, 6)
    for _ in range(n_mets):
        mx = rng.uniform(-0.35, 0.35)
        my = rng.uniform(-0.25, 0.25)
        mr = rng.uniform(0.01, 0.03)
        met = _ellipse_mask(
            H, W, mx, my, mr,
            mr * rng.uniform(0.7, 1.3),
            rng.uniform(-30, 30)) & body
        if met.sum() > 5:
            activity[met] = rng.uniform(1.2, 2.5)

    # Cold lesion (lytic metastasis)
    if rng.random() < 0.3:
        cx = rng.uniform(-0.05, 0.05)
        cy = rng.uniform(0.15, 0.28)
        cr = rng.uniform(0.015, 0.03)
        cold = _ellipse_mask(H, W, cx, cy, cr, cr * 0.8, 0) & spine
        activity[cold] = 0.10

    activity = gaussian_filter(activity, sigma=0.7)
    activity = np.clip(activity, 0.0, None)
    if activity.max() > 0:
        activity /= activity.max()

    return activity, mu_map, f"bone_{variant:02d}"


# -- Phantom diversity pool ---------------------------------------------------

PHANTOM_GENERATORS = [
    make_cardiac_perfusion_phantom,
    make_brain_perfusion_phantom,
    make_bone_scan_phantom,
]


def generate_phantoms_public(
    n: int = 12,
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Generate diverse public-tier phantoms: 4 cardiac + 4 brain + 4 bone."""
    phantoms = []
    for i in range(4):
        phantoms.append(make_cardiac_perfusion_phantom(
            IMAGE_SIZE, IMAGE_SIZE, seed=100 + i, variant=i))
    for i in range(4):
        phantoms.append(make_brain_perfusion_phantom(
            IMAGE_SIZE, IMAGE_SIZE, seed=200 + i, variant=i))
    for i in range(4):
        phantoms.append(make_bone_scan_phantom(
            IMAGE_SIZE, IMAGE_SIZE, seed=300 + i, variant=i))
    return phantoms[:n]


def generate_phantoms_dev(
    n: int = 20,
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Generate dev-tier phantoms with augmented diversity."""
    phantoms = []
    rng = np.random.default_rng(5000)
    for i in range(n):
        gen_fn = PHANTOM_GENERATORS[i % 3]
        activity, mu_map, name = gen_fn(
            IMAGE_SIZE, IMAGE_SIZE, seed=10500 + i, variant=i)
        # Augment: rotation + flip + mild zoom
        angle = float(rng.uniform(15, 345))
        activity = nd_rotate(activity, angle, reshape=False,
                             mode='constant', cval=0.0)
        mu_map = nd_rotate(mu_map, angle, reshape=False,
                           mode='constant', cval=0.0)
        if rng.random() < 0.5:
            activity = np.fliplr(activity)
            mu_map = np.fliplr(mu_map)
        zoom_f = float(rng.uniform(0.85, 1.15))
        if zoom_f != 1.0:
            activity = _zoom_crop(activity, zoom_f, IMAGE_SIZE)
            mu_map = _zoom_crop(mu_map, zoom_f, IMAGE_SIZE)
        activity = np.clip(activity, 0.0, None)
        if activity.max() > 0:
            activity /= activity.max()
        phantoms.append((activity, mu_map, f"dev_{name}"))
    return phantoms


def generate_phantoms_hidden(
    n: int = 20,
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Generate hidden-tier phantoms with adversarial modifications."""
    phantoms = []
    rng = np.random.default_rng(8000)
    for i in range(n):
        gen_fn = PHANTOM_GENERATORS[i % 3]
        activity, mu_map, name = gen_fn(
            IMAGE_SIZE, IMAGE_SIZE, seed=20800 + i, variant=i + 10)

        # Adversarial augmentation
        angle = float(rng.uniform(20, 340))
        activity = nd_rotate(activity, angle, reshape=False,
                             mode='constant', cval=0.0)
        mu_map = nd_rotate(mu_map, angle, reshape=False,
                           mode='constant', cval=0.0)
        if rng.random() < 0.7:
            activity = np.fliplr(activity)
            mu_map = np.fliplr(mu_map)
        if rng.random() < 0.5:
            activity = np.flipud(activity)
            mu_map = np.flipud(mu_map)

        # Aggressive zoom
        zoom_f = float(rng.uniform(0.70, 1.30))
        activity = _zoom_crop(activity, zoom_f, IMAGE_SIZE)
        mu_map = _zoom_crop(mu_map, zoom_f, IMAGE_SIZE)

        # Add subtle micro-lesions (hard to detect)
        n_micro = rng.integers(2, 6)
        for _ in range(n_micro):
            cy = rng.integers(40, IMAGE_SIZE - 40)
            cx = rng.integers(40, IMAGE_SIZE - 40)
            r = rng.integers(2, 5)
            yy, xx = np.ogrid[-r:r+1, -r:r+1]
            circle = (yy**2 + xx**2 <= r**2).astype(np.float64)
            y0, y1 = max(0, cy - r), min(IMAGE_SIZE, cy + r + 1)
            x0, x1 = max(0, cx - r), min(IMAGE_SIZE, cx + r + 1)
            c_y0, c_y1 = r - (cy - y0), r + (y1 - cy)
            c_x0, c_x1 = r - (cx - x0), r + (x1 - cx)
            if activity[y0:y1, x0:x1].mean() > 0.05:
                intensity = rng.uniform(1.2, 3.0)
                activity[y0:y1, x0:x1] = np.maximum(
                    activity[y0:y1, x0:x1],
                    circle[c_y0:c_y1, c_x0:c_x1] * intensity)

        activity = np.clip(activity, 0.0, None)
        if activity.max() > 0:
            activity /= activity.max()
        phantoms.append((activity, mu_map, f"hidden_{name}"))
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
        out = np.zeros((size, size), dtype=arr.dtype)
        y0 = (size - H) // 2
        x0 = (size - W) // 2
        out[y0:y0 + H, x0:x0 + W] = zoomed
        return out


# -- SPECT Forward Model ------------------------------------------------------

def spect_forward_model(
    activity: np.ndarray,
    mu_map: np.ndarray,
    theta_deg: np.ndarray,
    mu_map_scale: float,
    cdr_fwhm_mm: float,
    scatter_fraction: float,
    rotation_radius_mm: float,
    rng: np.random.Generator,
) -> dict:
    """Apply full SPECT forward model with depth-dependent CDR.

    y_theta = Poisson(P_theta * A_theta * x + scatter)

    Args:
        activity:          (H, W) ground-truth activity map [0, 1]
        mu_map:            (H, W) attenuation map [cm^-1]
        theta_deg:         projection angles in degrees
        mu_map_scale:      scaling factor for attenuation (mismatch)
        cdr_fwhm_mm:       collimator FWHM at detector face
        scatter_fraction:  scatter / (primary + scatter)
        rotation_radius_mm: camera orbit radius
        rng:               random generator

    Returns:
        dict with sinogram_ideal, sinogram_measured, attenuation_factors, etc.
    """
    # 1. Ideal sinogram with CDR blur
    sino_ideal_cdr = radon_with_cdr(
        activity, theta_deg, cdr_fwhm_mm, rotation_radius_mm)
    sino_ideal_cdr = np.maximum(sino_ideal_cdr, 0.0)

    # Also compute clean Radon (no CDR) for reference / H_ideal
    sino_ideal_clean = radon_transform(activity, theta_deg)
    sino_ideal_clean = np.maximum(sino_ideal_clean, 0.0)
    n_det = sino_ideal_cdr.shape[1]

    # 2. Attenuation: single-photon (from source to detector along each ray)
    attn_true = compute_attenuation_sinogram(mu_map, theta_deg)

    # Mismatched attenuation (what recon "thinks")
    mu_map_mismatch = mu_map * mu_map_scale
    attn_used = compute_attenuation_sinogram(mu_map_mismatch, theta_deg)

    # 3. Scale to physical count level
    # Clinical SPECT: target ~50-100 mean counts per detector bin AFTER
    # attenuation, giving moderate Poisson noise (SNR ~ 7-10).
    # We set scale so that mean(sino_cdr * scale * attn) ~ target_mean.
    n_bins = sino_ideal_cdr.shape[0] * sino_ideal_cdr.shape[1]
    target_mean_counts = 80.0  # mean detected counts per bin
    sensitivity_factor = (200.0 / rotation_radius_mm) ** 2

    # Estimate post-attenuation mean to calibrate scale
    sino_pre_attn = sino_ideal_cdr * attn_true
    mean_pre_attn = sino_pre_attn.mean() if sino_pre_attn.mean() > 1e-12 else 1e-12
    scale = target_mean_counts * sensitivity_factor / mean_pre_attn
    sino_primary = sino_ideal_cdr * scale * attn_true

    # 4. Scatter
    if scatter_fraction > 0:
        sf_ratio = scatter_fraction / max(1.0 - scatter_fraction, 0.1)
        scatter_base = sf_ratio * np.maximum(sino_primary.mean(), 0.01)
        scatter = np.ones_like(sino_primary) * scatter_base
        scatter_noise = (rng.standard_normal(sino_primary.shape) *
                         scatter_base * 0.15)
        scatter += gaussian_filter(scatter_noise, sigma=[8.0, 15.0])
        scatter = np.maximum(scatter, 0.0)
    else:
        scatter = np.zeros_like(sino_primary)

    # 5. Expected counts (no randoms in SPECT)
    expected = sino_primary + scatter
    expected = np.maximum(expected, 0.01)

    # 6. Poisson sampling
    sino_measured = rng.poisson(expected).astype(np.float64)

    return {
        "sinogram_ideal": sino_ideal_clean.astype(np.float32),
        "sinogram_ideal_cdr": sino_ideal_cdr.astype(np.float32),
        "sinogram_measured": sino_measured.astype(np.float32),
        "attenuation_factors": attn_true.astype(np.float32),
        "attenuation_factors_used": attn_used.astype(np.float32),
        "scatter": scatter.astype(np.float32),
        "expected_counts": expected.astype(np.float32),
        "scale_factor": float(scale),
    }


# -- Metrics ------------------------------------------------------------------

def compute_psnr(gt: np.ndarray, recon: np.ndarray) -> float:
    mse = np.mean((gt.astype(np.float64) - recon.astype(np.float64)) ** 2)
    if mse < 1e-12:
        return 100.0
    data_range = float(gt.max() - gt.min())
    if data_range < 1e-12:
        return 0.0
    return float(10 * np.log10(data_range ** 2 / mse))


def compute_ssim(gt: np.ndarray, recon: np.ndarray) -> float:
    gt = gt.astype(np.float64)
    recon = recon.astype(np.float64)
    data_range = gt.max() - gt.min()
    if data_range < 1e-12:
        return 0.0
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_x = gt.mean()
    mu_y = recon.mean()
    var_x = gt.var()
    var_y = recon.var()
    cov_xy = np.mean((gt - mu_x) * (recon - mu_y))
    ssim_val = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / \
               ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))
    return float(ssim_val)


# -- Image helpers ------------------------------------------------------------

def _norm(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo + 1e-8)


def _save_png(arr: np.ndarray, path: Path,
              percentile_clip: bool = False) -> None:
    if percentile_clip and arr.max() > 0:
        lo, hi = np.percentile(arr[arr > 0], [1, 99])
        arr = np.clip(arr, lo, hi)
    Image.fromarray(
        np.clip(_norm(arr) * 255, 0, 255).astype(np.uint8), "L"
    ).save(str(path))


def _save_overview(x_true, sino_ideal, sino_meas, recon_fbp,
                   path: Path) -> None:
    """4-panel overview: GT | ideal sino | measured sino | FBP recon."""
    th, tw = 128, 128

    def _r(a):
        pil = Image.fromarray(
            np.clip(_norm(a) * 255, 0, 255).astype(np.uint8), "L")
        return np.array(pil.resize((tw, th), Image.LANCZOS)) / 255.0

    ov = np.zeros((th, 4 * tw), dtype=np.float32)
    ov[:, 0:tw] = _r(x_true)
    ov[:, tw:2*tw] = _r(sino_ideal)
    ov[:, 2*tw:3*tw] = _r(sino_meas)
    ov[:, 3*tw:4*tw] = _r(recon_fbp)
    _save_png(ov, path)


# -- Tier generation ----------------------------------------------------------

def sample_mismatch(rng: np.random.Generator, spec: dict) -> dict:
    return {
        k: float(rng.uniform(v["min"], v["max"]))
        for k, v in spec.items()
    }


def generate_tier(
    tier: str,
    phantoms: list[tuple[np.ndarray, np.ndarray, str]],
    base_seed: int,
) -> dict:
    """Generate one tier of the SPECT benchmark.

    Returns:
        dict with tier summary stats.
    """
    spec_ranges = SPEC[tier]
    tier_dir = BENCHMARK_DIR / tier
    images_dir = tier_dir / "images"
    if tier_dir.exists():
        shutil.rmtree(tier_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    # SPECT uses full 360-degree rotation
    theta_deg = np.linspace(0, ANGLE_RANGE_DEG, N_ANGLES,
                            endpoint=False).astype(np.float64)

    h5_path = tier_dir / f"spect_challenge_{tier}.h5"
    rng = np.random.default_rng(base_seed)
    true_specs = {}
    all_psnrs = []
    all_ssims = []

    with h5py.File(h5_path, "w") as f:
        f.attrs["description"] = (
            f"PWM SPECT benchmark -- {tier} tier "
            f"(parallel-hole collimator + depth-dependent CDR + "
            f"single-photon attenuation + scatter + Poisson noise)")
        f.attrs["spec_ranges"] = json.dumps(spec_ranges)
        f.attrs["geometry"] = json.dumps({
            "image_size": IMAGE_SIZE,
            "n_angles": N_ANGLES,
            "n_det": N_DET,
            "angle_range_deg": [0, ANGLE_RANGE_DEG],
            "fov_mm": FOV_MM,
            "pixel_size_mm": PIXEL_SIZE_MM,
            "collimator": "parallel-hole",
            "isotope": "Tc-99m (140 keV)",
        })
        f.attrs["forward_model"] = (
            "y_theta ~ Poisson(P_theta * A_theta * x + scatter), "
            "P_theta includes depth-dependent CDR, "
            "A_theta = exp(-integral mu dr) single-photon attenuation")

        for idx, (activity, mu_map, scene_name) in enumerate(phantoms):
            key = f"sample_{idx:02d}"
            t0 = time.time()
            print(f"  [{tier}] Generating {key} ({scene_name})...",
                  end="", flush=True)

            mis = sample_mismatch(rng, spec_ranges)
            true_specs[key] = mis

            # Apply SPECT forward model
            result = spect_forward_model(
                activity, mu_map, theta_deg,
                mu_map_scale=mis["mu_map_scale"],
                cdr_fwhm_mm=mis["cdr_fwhm_mm"],
                scatter_fraction=mis["scatter_fraction"],
                rotation_radius_mm=mis["rotation_radius_mm"],
                rng=rng,
            )

            sino_ideal = result["sinogram_ideal"]
            sino_measured = result["sinogram_measured"]
            n_det = sino_ideal.shape[1]

            # FBP reconstruction with scatter subtraction and
            # attenuation correction (sinogram domain)
            attn_used = result["attenuation_factors_used"]
            attn_safe = np.where(attn_used > 0.01, attn_used, 0.01)

            # Corrected sinogram: (measured - scatter_est) / attenuation
            sino_corrected = sino_measured - result["scatter"]
            sino_corrected = np.maximum(sino_corrected, 0.0) / attn_safe

            # Rescale back to activity units
            if result["scale_factor"] > 0:
                sino_corrected /= result["scale_factor"]

            # Mild smoothing to suppress noise amplification from
            # attenuation correction (sigma=1 along detector, 0.5 angles)
            sino_corrected = gaussian_filter(
                sino_corrected, sigma=[0.5, 1.0])

            recon_fbp = fbp_reconstruct(
                sino_corrected, theta_deg, IMAGE_SIZE)
            recon_fbp = np.maximum(recon_fbp, 0.0).astype(np.float32)

            # Amplitude normalization: match to [0, 1] range of GT
            # using least-squares optimal scaling
            gt_flat = activity.flatten()
            r_flat = recon_fbp.flatten().astype(np.float64)
            opt_scale = (np.dot(gt_flat, r_flat) /
                         (np.dot(r_flat, r_flat) + 1e-12))
            # Clamp scale to reasonable range to avoid pathological cases
            opt_scale = float(np.clip(opt_scale, 0.5, 2.0))
            recon_final = np.maximum(
                recon_fbp * opt_scale, 0.0).astype(np.float32)

            # Compute metrics
            psnr = compute_psnr(activity, recon_final)
            ssim_val = compute_ssim(activity, recon_final)
            all_psnrs.append(psnr)
            all_ssims.append(ssim_val)

            elapsed = time.time() - t0

            # Save to HDF5
            grp = f.create_group(key)
            grp.create_dataset(
                "x_true", data=activity.astype(np.float32),
                compression="gzip")
            grp.create_dataset(
                "y", data=sino_measured,
                compression="gzip")
            grp.create_dataset(
                "H_ideal", data=sino_ideal,
                compression="gzip")
            grp.create_dataset(
                "sinogram_measured", data=sino_measured,
                compression="gzip")
            grp.create_dataset(
                "sinogram_ideal", data=sino_ideal,
                compression="gzip")
            grp.create_dataset(
                "attenuation_map",
                data=mu_map.astype(np.float32),
                compression="gzip")
            grp.create_dataset(
                "angles_deg",
                data=theta_deg.astype(np.float32))
            grp.create_dataset(
                "reconstruction_fbp", data=recon_final,
                compression="gzip")
            grp.attrs["metadata"] = json.dumps({
                "scene": scene_name,
                "shape": list(activity.shape),
                "n_angles": N_ANGLES,
                "n_det": int(n_det),
                "psnr_fbp": float(psnr),
                "ssim_fbp": float(ssim_val),
            })
            grp.attrs["true_spec"] = json.dumps(mis)
            grp.attrs["spec_ranges"] = json.dumps(spec_ranges)

            # Save images
            sample_dir = images_dir / f"sample_{idx:02d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            _save_png(activity, sample_dir / "gt.png")
            _save_png(sino_ideal, sample_dir / "measurement.png")
            _save_png(sino_measured,
                      sample_dir / "measurement_noisy.png")
            _save_png(recon_final, sample_dir / "recon.png")
            _save_overview(
                activity, sino_ideal, sino_measured, recon_final,
                sample_dir / "overview.png")
            with open(sample_dir / "spec.json", "w") as sf:
                json.dump({
                    "scene": scene_name,
                    "spec_ranges": spec_ranges,
                    "true_spec": mis,
                    "psnr_fbp": psnr,
                    "ssim_fbp": ssim_val,
                }, sf, indent=2)

            print(f"  PSNR={psnr:.2f} dB, SSIM={ssim_val:.3f}  "
                  f"({elapsed:.1f}s)  "
                  f"mu_scale={mis['mu_map_scale']:.3f}  "
                  f"cdr={mis['cdr_fwhm_mm']:.1f}mm  "
                  f"scatter={mis['scatter_fraction']:.3f}  "
                  f"radius={mis['rotation_radius_mm']:.0f}mm")

    # Save spec files
    with open(tier_dir / "spec.json", "w") as sf:
        json.dump(spec_ranges, sf, indent=2)
    with open(tier_dir / "true_spec.json", "w") as tf:
        json.dump(true_specs, tf, indent=2)

    mean_psnr = float(np.mean(all_psnrs))
    mean_ssim = float(np.mean(all_ssims))
    print(f"  [{tier}] {len(phantoms)} samples | "
          f"Mean PSNR={mean_psnr:.2f} dB | Mean SSIM={mean_ssim:.3f}")
    print(f"  [{tier}] HDF5 -> {h5_path.name}")

    return {
        "tier": tier,
        "n_samples": len(phantoms),
        "mean_psnr": mean_psnr,
        "mean_ssim": mean_ssim,
        "h5_path": str(h5_path),
    }


# -- Gallery images -----------------------------------------------------------

def generate_gallery_images() -> None:
    """Generate gallery images for the platform benchmark page.

    Creates scene_00 through scene_03 with:
      gt.png, measurement_I.png, measurement_II.png,
      recon_I.png, recon_II.png
    """
    gallery_base = (BENCHMARK_DIR.parent.parent.parent /
                    "platform" / "pwm_platform" / "static" / "img" /
                    "benchmark_gallery" / "spect")

    h5_path = BENCHMARK_DIR / "public" / "spect_challenge_public.h5"
    if not h5_path.exists():
        print("  [gallery] Public HDF5 not found, skipping.")
        return

    # Pick diverse samples: cardiac(0), brain(4), bone(8), cardiac(3)
    gallery_sample_indices = [0, 4, 8, 3]

    with h5py.File(h5_path, "r") as f:
        for scene_idx, sample_idx in enumerate(gallery_sample_indices):
            scene_dir = gallery_base / f"scene_{scene_idx:02d}"
            scene_dir.mkdir(parents=True, exist_ok=True)

            key = f"sample_{sample_idx:02d}"
            if key not in f:
                print(f"  [gallery] {key} not found, skipping.")
                continue

            grp = f[key]
            x_true = grp["x_true"][:]
            sino_ideal = grp["sinogram_ideal"][:]
            sino_meas = grp["sinogram_measured"][:]
            recon_fbp = grp["reconstruction_fbp"][:]
            attn_map = grp["attenuation_map"][:]

            # gt.png -- ground truth activity
            _save_png(x_true, scene_dir / "gt.png")

            # measurement_I.png -- measured sinogram
            _save_png(sino_meas, scene_dir / "measurement_I.png")

            # measurement_II.png -- ideal sinogram
            _save_png(sino_ideal, scene_dir / "measurement_II.png")

            # recon_I.png -- FBP reconstruction
            _save_png(recon_fbp, scene_dir / "recon_I.png")

            # recon_II.png -- attenuation map
            _save_png(attn_map, scene_dir / "recon_II.png")

            print(f"  [gallery] scene_{scene_idx:02d} -> {scene_dir}")


# -- Main ---------------------------------------------------------------------

def main() -> None:
    t_start = time.time()
    print("SPECT Benchmark Dataset Generator")
    print("=" * 68)
    print(f"Output: {BENCHMARK_DIR}")
    print(f"Geometry: {N_ANGLES} angles over [0, {ANGLE_RANGE_DEG}) deg, "
          f"{IMAGE_SIZE}x{IMAGE_SIZE} images")
    print(f"Collimator: parallel-hole, isotope: Tc-99m (140 keV)\n")

    tier_stats = []

    # -- Public tier (12 samples) -----------------------------------------
    print("Generating public tier (12 samples)...")
    public_phantoms = generate_phantoms_public(12)
    tier_stats.append(generate_tier(
        "public", public_phantoms,
        base_seed=TIER_SEED_OFFSETS["public"]))

    # -- Dev tier (20 samples) --------------------------------------------
    print("\nGenerating dev tier (20 samples)...")
    dev_phantoms = generate_phantoms_dev(20)
    tier_stats.append(generate_tier(
        "dev", dev_phantoms,
        base_seed=TIER_SEED_OFFSETS["dev"]))

    # -- Hidden tier (20 samples) -----------------------------------------
    print("\nGenerating hidden tier (20 samples)...")
    hidden_phantoms = generate_phantoms_hidden(20)
    tier_stats.append(generate_tier(
        "hidden", hidden_phantoms,
        base_seed=TIER_SEED_OFFSETS["hidden"]))

    # -- Gallery images ---------------------------------------------------
    print("\nGenerating gallery images...")
    generate_gallery_images()

    elapsed = time.time() - t_start
    print(f"\n{'=' * 68}")
    print("SPECT benchmark dataset generation complete!")
    print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    for s in tier_stats:
        print(f"  {s['tier']:8s}: {s['n_samples']:2d} samples | "
              f"PSNR={s['mean_psnr']:.2f} dB | SSIM={s['mean_ssim']:.3f}")
    print(f"Output: {BENCHMARK_DIR}")
    print("=" * 68)


if __name__ == "__main__":
    main()
