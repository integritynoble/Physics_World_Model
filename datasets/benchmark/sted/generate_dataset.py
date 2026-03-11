#!/usr/bin/env python3
"""Generate STED (Stimulated Emission Depletion) microscopy benchmark dataset.

Forward model:
    y = Poisson(PSF_sted * x + background) + readout_noise

where:
    x            : true fluorophore density map (256x256)
    PSF_sted     : effective STED PSF (sub-diffraction, sigma ~1-2 pixels)
                   h_eff(r) = h_exc(r) * exp(-ln2 * I_dep(r) / I_sat)
                   resulting in a narrower Gaussian-like profile
    background   : autofluorescence / detector dark counts
    readout_noise: Gaussian camera/APD noise

Mismatch parameters:
    depletion_power : ratio I_STED/I_sat (affects PSF width)
    background_level: background photons per pixel
    photon_budget   : mean photons per fluorophore
    photobleaching_fraction: fraction of fluorophores lost during scan

Phantoms:
    Subcellular structures relevant to STED imaging:
    - Cytoskeleton filaments (actin/microtubule networks)
    - Synaptic vesicles (clusters of small puncta)
    - Nuclear pore complexes (ring patterns on nuclear envelope)

CPU Baseline:
    Richardson-Lucy deconvolution (50-100 iterations)
    Expected: ~22-28 dB PSNR

Tiers:
    Public  : 12 samples (seed offset 0)
    Dev     : 20 samples (seed offset 10000)
    Hidden  : 20 samples (seed offset 20000)

Usage:
    cd datasets/benchmark/sted
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

IMAGE_SIZE = 256           # ground truth and measurement share the same grid
PIXEL_SIZE_NM = 25.0       # 25 nm/pixel (Nyquist for ~50 nm STED resolution)

# -- STED PSF parameters ----------------------------------------------------
# Confocal (diffraction-limited) PSF sigma ~ 100-120 nm => 4-5 pixels
# STED effective PSF sigma ~ 25-80 nm => 1-3 pixels depending on depletion power

CONFOCAL_SIGMA_PX = 4.5    # diffraction-limited confocal PSF sigma (pixels)

# -- Mismatch ranges per tier -----------------------------------------------

SPEC = {
    "public": {
        "depletion_power":       {"min": 8.0,  "max": 15.0, "unit": "I_STED/I_sat"},
        "background_level":      {"min": 2.0,  "max": 10.0, "unit": "photons/pixel"},
        "photon_budget":         {"min": 300,  "max": 1000, "unit": "photons"},
        "photobleaching_fraction": {"min": 0.0, "max": 0.15, "unit": "fraction"},
    },
    "dev": {
        "depletion_power":       {"min": 6.0,  "max": 18.0, "unit": "I_STED/I_sat"},
        "background_level":      {"min": 3.0,  "max": 20.0, "unit": "photons/pixel"},
        "photon_budget":         {"min": 200,  "max": 1000, "unit": "photons"},
        "photobleaching_fraction": {"min": 0.0, "max": 0.25, "unit": "fraction"},
    },
    "hidden": {
        "depletion_power":       {"min": 5.0,  "max": 20.0, "unit": "I_STED/I_sat"},
        "background_level":      {"min": 5.0,  "max": 30.0, "unit": "photons/pixel"},
        "photon_budget":         {"min": 150,  "max": 800,  "unit": "photons"},
        "photobleaching_fraction": {"min": 0.05, "max": 0.35, "unit": "fraction"},
    },
}


# ============================================================================
# Phantom generators -- subcellular structures for STED
# ============================================================================

def _smooth_curve_points(
    control_pts: np.ndarray, n_interp: int = 500
) -> np.ndarray:
    """Generate smooth curve via cubic interpolation of control points."""
    from scipy.interpolate import CubicSpline

    n = len(control_pts)
    if n < 3:
        return control_pts
    t = np.linspace(0, 1, n)
    t_new = np.linspace(0, 1, n_interp)
    cs_y = CubicSpline(t, control_pts[:, 0], bc_type="natural")
    cs_x = CubicSpline(t, control_pts[:, 1], bc_type="natural")
    return np.column_stack([cs_y(t_new), cs_x(t_new)])


def make_cytoskeleton_phantom(
    H: int, W: int, rng: np.random.Generator,
) -> np.ndarray:
    """Cytoskeleton filaments: thin curved lines (actin + microtubule network).

    STED frequently images actin filaments labeled with phalloidin-ATTO647N
    or microtubules labeled with anti-tubulin antibodies. These appear as
    thin (~25-50 nm width) curved filaments crossing the field of view.

    Returns:
        x_true: (H, W) float64 ground truth fluorophore density [0, 1]
    """
    x_true = np.zeros((H, W), dtype=np.float64)
    n_filaments = rng.integers(8, 20)

    for _ in range(n_filaments):
        # Start point: anywhere in the image
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

        curve = _smooth_curve_points(ctrl, n_interp=800)

        # Filter valid points
        valid = (
            (curve[:, 0] >= 0) & (curve[:, 0] < H)
            & (curve[:, 1] >= 0) & (curve[:, 1] < W)
        )
        curve = curve[valid]
        if len(curve) < 5:
            continue

        # Draw thin line (1 pixel wide -> sub-pixel rendering)
        # Filament intensity varies slightly along its length
        intensity = rng.uniform(0.5, 1.0)
        for pt in curve:
            iy, ix = int(round(pt[0])), int(round(pt[1]))
            if 0 <= iy < H and 0 <= ix < W:
                x_true[iy, ix] = max(x_true[iy, ix], intensity)

    # Add some branching points / junctions
    n_branches = rng.integers(2, 6)
    for _ in range(n_branches):
        by = rng.uniform(H * 0.1, H * 0.9)
        bx = rng.uniform(W * 0.1, W * 0.9)
        n_arms = rng.integers(2, 5)
        for _ in range(n_arms):
            angle = rng.uniform(0, 2 * np.pi)
            arm_len = rng.uniform(10, 40)
            n_pts = int(arm_len * 2)
            for k in range(n_pts):
                t = k / max(n_pts - 1, 1)
                py = by + arm_len * t * np.sin(angle) + rng.normal(0, 0.5)
                px = bx + arm_len * t * np.cos(angle) + rng.normal(0, 0.5)
                iy, ix = int(round(py)), int(round(px))
                if 0 <= iy < H and 0 <= ix < W:
                    x_true[iy, ix] = max(x_true[iy, ix], 0.7)

    # Slight Gaussian blur to give realistic width (~1 pixel = 25 nm)
    x_true = gaussian_filter(x_true, sigma=0.6)

    if x_true.max() > 0:
        x_true /= x_true.max()

    return x_true.astype(np.float64)


def make_synaptic_vesicle_phantom(
    H: int, W: int, rng: np.random.Generator,
) -> np.ndarray:
    """Synaptic vesicles: clusters of small bright puncta.

    STED is widely used to image synaptic vesicle pools (e.g., synaptophysin
    or VAMP2 labeling). Vesicles appear as ~40-60 nm bright dots clustered
    at synaptic boutons.

    Returns:
        x_true: (H, W) float64 ground truth [0, 1]
    """
    x_true = np.zeros((H, W), dtype=np.float64)

    # Generate 5-15 synaptic boutons
    n_boutons = rng.integers(5, 16)

    for _ in range(n_boutons):
        # Bouton center
        by = rng.uniform(H * 0.08, H * 0.92)
        bx = rng.uniform(W * 0.08, W * 0.92)
        bouton_radius = rng.uniform(8, 25)  # bouton size in pixels

        # Number of vesicles in this bouton
        n_vesicles = rng.integers(8, 40)

        for _ in range(n_vesicles):
            # Vesicle position within bouton (concentrated at center)
            r = rng.exponential(bouton_radius * 0.4)
            theta = rng.uniform(0, 2 * np.pi)
            vy = by + r * np.sin(theta)
            vx = bx + r * np.cos(theta)
            iy, ix = int(round(vy)), int(round(vx))
            if 0 <= iy < H and 0 <= ix < W:
                # Each vesicle is a tiny bright spot
                intensity = rng.uniform(0.5, 1.0)
                # Render as small Gaussian (~1 pixel radius)
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        ny, nx = iy + dy, ix + dx
                        if 0 <= ny < H and 0 <= nx < W:
                            g = intensity * np.exp(
                                -(dy ** 2 + dx ** 2) / (2 * 0.8 ** 2)
                            )
                            x_true[ny, nx] = max(x_true[ny, nx], g)

    # Add some isolated vesicles (scattered)
    n_isolated = rng.integers(10, 30)
    for _ in range(n_isolated):
        vy = rng.uniform(0, H - 1)
        vx = rng.uniform(0, W - 1)
        iy, ix = int(round(vy)), int(round(vx))
        if 0 <= iy < H and 0 <= ix < W:
            intensity = rng.uniform(0.3, 0.8)
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    ny, nx = iy + dy, ix + dx
                    if 0 <= ny < H and 0 <= nx < W:
                        g = intensity * np.exp(
                            -(dy ** 2 + dx ** 2) / (2 * 0.7 ** 2)
                        )
                        x_true[ny, nx] = max(x_true[ny, nx], g)

    if x_true.max() > 0:
        x_true /= x_true.max()

    return x_true.astype(np.float64)


def make_nuclear_pore_phantom(
    H: int, W: int, rng: np.random.Generator,
) -> np.ndarray:
    """Nuclear pore complexes: ring patterns on nuclear envelope.

    STED microscopy pioneered imaging of nuclear pore complexes (NPCs) with
    antibodies against Nup153, gp210, etc. Each NPC appears as an 8-fold
    symmetric ring (~120 nm diameter = ~5 pixels).

    Returns:
        x_true: (H, W) float64 ground truth [0, 1]
    """
    x_true = np.zeros((H, W), dtype=np.float64)

    # Nuclear envelope: roughly elliptical
    nuc_cy = H / 2 + rng.uniform(-15, 15)
    nuc_cx = W / 2 + rng.uniform(-15, 15)
    nuc_a = rng.uniform(H * 0.28, H * 0.40)  # semi-major
    nuc_b = rng.uniform(H * 0.22, H * 0.35)  # semi-minor
    nuc_angle = rng.uniform(0, np.pi)

    # Draw nuclear envelope as a thin line
    n_env_pts = 1500
    for i in range(n_env_pts):
        theta = 2 * np.pi * i / n_env_pts
        r_y = nuc_a * np.sin(theta)
        r_x = nuc_b * np.cos(theta)
        py = nuc_cy + r_y * np.cos(nuc_angle) - r_x * np.sin(nuc_angle)
        px = nuc_cx + r_y * np.sin(nuc_angle) + r_x * np.cos(nuc_angle)
        iy, ix = int(round(py)), int(round(px))
        if 0 <= iy < H and 0 <= ix < W:
            x_true[iy, ix] = max(x_true[iy, ix], 0.15)

    # Place NPCs along envelope
    n_pores = rng.integers(15, 40)
    pore_angles = np.sort(rng.uniform(0, 2 * np.pi, n_pores))

    for pa in pore_angles:
        r_y = nuc_a * np.sin(pa)
        r_x = nuc_b * np.cos(pa)
        pore_cy = nuc_cy + r_y * np.cos(nuc_angle) - r_x * np.sin(nuc_angle)
        pore_cx = nuc_cx + r_y * np.sin(nuc_angle) + r_x * np.cos(nuc_angle)

        pore_cy += rng.normal(0, 1.5)
        pore_cx += rng.normal(0, 1.5)

        # Each NPC: 8-fold symmetric ring, diameter ~ 5 pixels (120 nm)
        pore_r = rng.uniform(2.0, 3.5)  # radius in pixels
        n_subunits = 8
        subunit_phase = rng.uniform(0, 2 * np.pi / n_subunits)

        for k in range(n_subunits):
            theta = 2 * np.pi * k / n_subunits + subunit_phase
            sy = pore_cy + pore_r * np.sin(theta) + rng.normal(0, 0.2)
            sx = pore_cx + pore_r * np.cos(theta) + rng.normal(0, 0.2)
            iy, ix = int(round(sy)), int(round(sx))

            # Render each subunit as a tiny Gaussian
            intensity = rng.uniform(0.7, 1.0)
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    ny, nx = iy + dy, ix + dx
                    if 0 <= ny < H and 0 <= nx < W:
                        g = intensity * np.exp(
                            -(dy ** 2 + dx ** 2) / (2 * 0.6 ** 2)
                        )
                        x_true[ny, nx] = max(x_true[ny, nx], g)

    if x_true.max() > 0:
        x_true /= x_true.max()

    return x_true.astype(np.float64)


def make_mixed_phantom(
    H: int, W: int, rng: np.random.Generator,
) -> np.ndarray:
    """Mixed subcellular scene: filaments + vesicles + membrane structures.

    Combines cytoskeleton filaments with some vesicular puncta and short
    membrane segments, representing a typical STED field of view.

    Returns:
        x_true: (H, W) float64 ground truth [0, 1]
    """
    x_true = np.zeros((H, W), dtype=np.float64)

    # Filament component (reduced density)
    n_filaments = rng.integers(4, 10)
    for _ in range(n_filaments):
        cy = rng.uniform(H * 0.05, H * 0.95)
        cx = rng.uniform(W * 0.05, W * 0.95)
        angle = rng.uniform(0, 2 * np.pi)
        length = rng.uniform(H * 0.1, H * 0.4)

        n_ctrl = rng.integers(3, 7)
        ctrl = np.zeros((n_ctrl, 2))
        for j in range(n_ctrl):
            t = j / max(n_ctrl - 1, 1)
            ctrl[j, 0] = cy + length * t * np.sin(angle) + rng.uniform(-15, 15)
            ctrl[j, 1] = cx + length * t * np.cos(angle) + rng.uniform(-15, 15)

        curve = _smooth_curve_points(ctrl, n_interp=600)
        valid = (
            (curve[:, 0] >= 0) & (curve[:, 0] < H)
            & (curve[:, 1] >= 0) & (curve[:, 1] < W)
        )
        curve = curve[valid]
        if len(curve) < 5:
            continue

        intensity = rng.uniform(0.5, 1.0)
        for pt in curve:
            iy, ix = int(round(pt[0])), int(round(pt[1]))
            if 0 <= iy < H and 0 <= ix < W:
                x_true[iy, ix] = max(x_true[iy, ix], intensity)

    # Vesicle component
    n_boutons = rng.integers(3, 8)
    for _ in range(n_boutons):
        by = rng.uniform(H * 0.1, H * 0.9)
        bx = rng.uniform(W * 0.1, W * 0.9)
        n_vesicles = rng.integers(5, 20)
        for _ in range(n_vesicles):
            r = rng.exponential(8)
            theta = rng.uniform(0, 2 * np.pi)
            vy = by + r * np.sin(theta)
            vx = bx + r * np.cos(theta)
            iy, ix = int(round(vy)), int(round(vx))
            if 0 <= iy < H and 0 <= ix < W:
                intensity = rng.uniform(0.4, 0.9)
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        ny, nx = iy + dy, ix + dx
                        if 0 <= ny < H and 0 <= nx < W:
                            g = intensity * np.exp(
                                -(dy ** 2 + dx ** 2) / (2 * 0.8 ** 2)
                            )
                            x_true[ny, nx] = max(x_true[ny, nx], g)

    # Membrane segments
    n_membranes = rng.integers(2, 5)
    for _ in range(n_membranes):
        my = rng.uniform(H * 0.1, H * 0.9)
        mx = rng.uniform(W * 0.1, W * 0.9)
        length = rng.uniform(30, 80)
        angle = rng.uniform(0, 2 * np.pi)
        curvature = rng.uniform(-0.05, 0.05)
        n_pts = int(length * 3)
        for k in range(n_pts):
            t = k / max(n_pts - 1, 1)
            a = angle + curvature * t * length
            py = my + length * t * np.sin(a)
            px = mx + length * t * np.cos(a)
            iy, ix = int(round(py)), int(round(px))
            if 0 <= iy < H and 0 <= ix < W:
                x_true[iy, ix] = max(x_true[iy, ix], 0.6)

    # Small blur for realistic width
    x_true = gaussian_filter(x_true, sigma=0.5)

    if x_true.max() > 0:
        x_true /= x_true.max()

    return x_true.astype(np.float64)


PHANTOM_FNS = [
    make_cytoskeleton_phantom,
    make_synaptic_vesicle_phantom,
    make_nuclear_pore_phantom,
    make_mixed_phantom,
]

PHANTOM_NAMES = ["cytoskeleton", "synaptic_vesicle", "nuclear_pore", "mixed"]


# ============================================================================
# STED Forward Model
# ============================================================================

def make_sted_psf(
    size: int,
    depletion_power: float,
    confocal_sigma: float = CONFOCAL_SIGMA_PX,
) -> np.ndarray:
    """Create effective STED PSF.

    The STED PSF is modeled as:
        h_eff(r) = h_exc(r) * exp(-ln2 * I_dep(r) / I_sat)

    where h_exc is the confocal Gaussian PSF and I_dep is the doughnut
    depletion beam. The effective resolution scales as:
        sigma_eff ~ sigma_confocal / sqrt(1 + depletion_power)

    Args:
        size: PSF kernel size (should be odd)
        depletion_power: I_STED / I_sat ratio
        confocal_sigma: diffraction-limited confocal PSF sigma in pixels

    Returns:
        psf: (size, size) normalized PSF kernel
    """
    if size % 2 == 0:
        size += 1

    half = size // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float64)
    r2 = x ** 2 + y ** 2

    # Excitation PSF (Gaussian)
    h_exc = np.exp(-r2 / (2 * confocal_sigma ** 2))

    # Depletion beam: doughnut profile (Laguerre-Gaussian LG01)
    # I_dep(r) ~ r^2 * exp(-r^2 / (2 * sigma_dep^2))
    # sigma_dep is similar to confocal_sigma
    sigma_dep = confocal_sigma * 1.0
    I_dep = (r2 / (sigma_dep ** 2)) * np.exp(-r2 / (2 * sigma_dep ** 2))
    # Normalize depletion beam peak to 1
    I_dep_max = I_dep.max()
    if I_dep_max > 0:
        I_dep /= I_dep_max

    # Effective PSF: excitation * depletion suppression
    # exp(-ln2 * depletion_power * I_dep) kills fluorescence at periphery
    h_eff = h_exc * np.exp(-np.log(2) * depletion_power * I_dep)

    # Normalize
    h_sum = h_eff.sum()
    if h_sum > 0:
        h_eff /= h_sum

    return h_eff


def compute_effective_sigma(depletion_power: float) -> float:
    """Compute approximate effective PSF sigma in pixels.

    sigma_eff ~ sigma_confocal / sqrt(1 + depletion_power)
    """
    return CONFOCAL_SIGMA_PX / np.sqrt(1 + depletion_power)


def sted_forward(
    x_true: np.ndarray,
    depletion_power: float,
    photon_budget: float,
    background_level: float,
    photobleaching_fraction: float,
    rng: np.random.Generator,
    readout_noise_std: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """STED forward model.

    Args:
        x_true: (H, W) ground truth fluorophore density [0, 1]
        depletion_power: I_STED/I_sat ratio
        photon_budget: mean photons per fluorophore at peak
        background_level: background photons per pixel
        photobleaching_fraction: fraction of signal lost due to bleaching
        rng: random number generator
        readout_noise_std: Gaussian readout noise std

    Returns:
        y: (H, W) float32 noisy measurement
        H_ideal: (H, W) float32 noiseless blurred signal (PSF * x)
    """
    H, W = x_true.shape

    # Compute STED PSF
    psf_size = int(6 * CONFOCAL_SIGMA_PX) * 2 + 1  # big enough for confocal PSF
    psf = make_sted_psf(psf_size, depletion_power)

    # Apply photobleaching: reduce signal in a spatially varying pattern
    # Bleaching is stronger where depletion beam is stronger
    if photobleaching_fraction > 0:
        # Spatially uniform bleaching (simplified model)
        bleach_mask = 1.0 - photobleaching_fraction * rng.uniform(0.5, 1.5, (H, W))
        bleach_mask = np.clip(bleach_mask, 0.0, 1.0)
        x_bleached = x_true * bleach_mask
    else:
        x_bleached = x_true.copy()

    # Convolve with STED PSF
    signal = fftconvolve(x_bleached * photon_budget, psf, mode="same")
    signal = np.maximum(signal, 0)

    # Ideal image (noiseless)
    H_ideal = signal.copy()

    # Add background
    signal_plus_bg = signal + background_level

    # Poisson noise (shot noise)
    y = rng.poisson(np.maximum(signal_plus_bg, 0.01)).astype(np.float64)

    # Readout noise (Gaussian)
    y += rng.normal(0, readout_noise_std, (H, W))
    y = np.maximum(y, 0)

    return y.astype(np.float32), H_ideal.astype(np.float32)


# ============================================================================
# CPU Baseline: Richardson-Lucy Deconvolution
# ============================================================================

def richardson_lucy_deconv(
    y: np.ndarray,
    psf: np.ndarray,
    n_iter: int = 80,
    clip_negative: bool = True,
) -> np.ndarray:
    """Richardson-Lucy deconvolution (maximum-likelihood for Poisson noise).

    Algorithm:
        x_{k+1} = x_k * (PSF^T * (y / (PSF * x_k + eps)))

    This is the standard baseline for fluorescence microscopy deconvolution,
    widely used in STED post-processing.

    Args:
        y: (H, W) noisy measurement
        psf: (K, K) PSF kernel (normalized to sum=1)
        n_iter: number of RL iterations
        clip_negative: clip result to be non-negative

    Returns:
        recon: (H, W) float32 reconstruction
    """
    y64 = y.astype(np.float64)
    y64 = np.maximum(y64, 0)

    # Initial estimate: the measurement itself
    x_est = y64.copy()
    x_est = np.maximum(x_est, 1e-6)

    # Flipped PSF for correlation step
    psf_flipped = psf[::-1, ::-1].copy()

    eps = 1e-10

    for _ in range(n_iter):
        # Forward: PSF * x_est
        blurred = fftconvolve(x_est, psf, mode="same")
        blurred = np.maximum(blurred, eps)

        # Ratio
        ratio = y64 / blurred

        # Backward: PSF^T * ratio
        correction = fftconvolve(ratio, psf_flipped, mode="same")

        # Update
        x_est *= correction

        if clip_negative:
            x_est = np.maximum(x_est, 0)

    # Normalize to [0, 1]
    if x_est.max() > 0:
        x_est /= x_est.max()

    return x_est.astype(np.float32)


# ============================================================================
# Metrics
# ============================================================================

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
    mu_x = gt64.mean()
    mu_y = recon64.mean()
    var_x = gt64.var()
    var_y = recon64.var()
    cov_xy = np.mean((gt64 - mu_x) * (recon64 - mu_y))
    ssim_val = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / \
               ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2))
    return float(ssim_val)


# ============================================================================
# Image helpers
# ============================================================================

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


# ============================================================================
# Phantom generation
# ============================================================================

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
        if k in ("photon_budget", "background_level"):
            val = round(val)
        result[k] = val
    return result


def generate_tier(
    tier: str,
    phantoms: list[tuple[np.ndarray, str]],
    base_seed: int,
) -> None:
    """Generate one tier of the STED benchmark."""
    spec_ranges = SPEC[tier]
    tier_dir = BENCHMARK_DIR / tier
    images_dir = tier_dir / "images"
    if tier_dir.exists():
        shutil.rmtree(tier_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    h5_path = tier_dir / f"sted_challenge_{tier}.h5"
    rng = np.random.default_rng(base_seed)
    true_specs = {}
    all_psnrs = []
    all_ssims = []

    with h5py.File(h5_path, "w") as f:
        f.attrs["description"] = (
            f"PWM STED benchmark -- {tier} tier "
            f"(Stimulated Emission Depletion microscopy)"
        )
        f.attrs["spec_ranges"] = json.dumps(spec_ranges)
        f.attrs["forward_model"] = (
            "y = Poisson(PSF_sted * x + background) + readout_noise"
        )
        f.attrs["geometry"] = json.dumps({
            "image_size": IMAGE_SIZE,
            "pixel_size_nm": PIXEL_SIZE_NM,
            "confocal_sigma_px": CONFOCAL_SIGMA_PX,
        })

        for idx, (x_true, scene_name) in enumerate(phantoms):
            key = f"sample_{idx:02d}"
            print(f"  [{tier}] Generating {key} ({scene_name})...",
                  end="", flush=True)

            mis = sample_mismatch(rng, spec_ranges)
            true_specs[key] = {**mis, "scene": scene_name}

            # Effective PSF sigma for this sample
            eff_sigma = compute_effective_sigma(mis["depletion_power"])

            # Forward model
            y, H_ideal = sted_forward(
                x_true,
                depletion_power=mis["depletion_power"],
                photon_budget=mis["photon_budget"],
                background_level=mis["background_level"],
                photobleaching_fraction=mis["photobleaching_fraction"],
                rng=rng,
            )

            # CPU baseline: Richardson-Lucy with estimated STED PSF
            # Use the ideal PSF (slight mismatch may help robustness)
            psf_size = int(6 * CONFOCAL_SIGMA_PX) * 2 + 1
            psf_for_rl = make_sted_psf(
                psf_size, mis["depletion_power"]
            )
            recon = richardson_lucy_deconv(y, psf_for_rl, n_iter=80)

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
                "effective_psf_sigma_px": float(eff_sigma),
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
                    "effective_psf_sigma_px": float(eff_sigma),
                    "psnr_baseline": psnr,
                    "ssim_baseline": ssim,
                }, sf, indent=2)

            print(f"  PSNR={psnr:.2f} dB, SSIM={ssim:.3f}  "
                  f"depletion={mis['depletion_power']:.1f}  "
                  f"photons={mis['photon_budget']}  "
                  f"bg={mis['background_level']}  "
                  f"bleach={mis['photobleaching_fraction']:.2f}  "
                  f"sigma_eff={eff_sigma:.2f} px")

        # Final HDF5 summary
        f.attrs["baseline_method"] = "Richardson-Lucy deconvolution (80 iter)"
        f.attrs["mean_psnr_baseline"] = float(np.mean(all_psnrs))
        f.attrs["mean_ssim_baseline"] = float(np.mean(all_ssims))

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
        / "benchmark_gallery" / "sted"
    )

    h5_path = BENCHMARK_DIR / "public" / "sted_challenge_public.h5"
    if not h5_path.exists():
        print("  [gallery] Public HDF5 not found, skipping gallery generation.")
        return

    # Pick 4 diverse samples: one of each phantom type
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

            # measurement_I.png -- noisy STED measurement
            _save_png(y, scene_dir / "measurement_I.png", percentile_clip=True)

            # measurement_II.png -- noiseless blurred signal (H_ideal)
            _save_png(H_ideal, scene_dir / "measurement_II.png",
                      percentile_clip=True)

            # recon_I.png -- Richardson-Lucy reconstruction
            _save_png(recon, scene_dir / "recon_I.png")

            # recon_II.png -- difference |GT - recon|
            diff = np.abs(x_true - recon)
            _save_png(diff, scene_dir / "recon_II.png")

            print(f"  [gallery] scene_{scene_idx:02d} -> {scene_dir}")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("STED Benchmark Dataset Generator")
    print("=" * 68)
    print(f"Output: {BENCHMARK_DIR}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}, "
          f"pixel size: {PIXEL_SIZE_NM} nm\n")

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
    print("STED benchmark dataset generation complete!")
    print(f"Output: {BENCHMARK_DIR}")
    print("=" * 68)


if __name__ == "__main__":
    main()
