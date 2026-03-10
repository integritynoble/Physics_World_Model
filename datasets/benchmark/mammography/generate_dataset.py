#!/usr/bin/env python3
"""Generate 2D mammography benchmark dataset.

Forward model (Beer-Lambert X-ray attenuation through compressed breast):

    y_i = I_0 * exp(-sum(mu(x,E) * dl_i)) + n_i

where:
    x_true      : 2D attenuation map (256x256) of breast tissue
    I_0         : incident X-ray fluence (controlled by dose_mGy)
    mu(x,E)     : linear attenuation coefficient at each pixel
    dl_i        : ray path through compressed breast (breast_thickness_cm)
    n_i         : Poisson noise (quantum-limited at clinical doses of 0.3-3.0 mGy)

Additional degradation:
    scatter     : additive scatter fraction (low-frequency background)
    detector_blur: Gaussian PSF of the flat-panel detector

Ground truth phantoms (256x256):
    Realistic breast tissue attenuation maps with:
    - Adipose tissue (low attenuation ~0.1-0.2)
    - Fibroglandular tissue (irregular dense regions, ~0.3-0.5)
    - Masses (round/spiculated lesions, slightly higher than glandular)
    - Microcalcifications (tiny bright dots, high attenuation ~0.8-1.0)
    - Cooper's ligaments (thin curved lines)
    - Perlin-noise-like texture for realistic tissue heterogeneity

Phantoms per tier:
    Public  : 12 samples (4 fatty + 4 dense + 4 with calcifications/masses)
    Dev     : 20 samples (augmented variants)
    Hidden  : 20 samples (adversarial: extreme dose, scatter, blur)

Mismatch parameters:
    dose_mGy           : radiation dose, controls Poisson noise (1.0-3.0 public, 0.3-3.0 hidden)
    scatter_fraction   : scatter contribution (0.1-0.25 public, 0.1-0.4 hidden)
    detector_blur_sigma: detector PSF sigma in pixels (0.5-1.5 public, 0.5-3.0 hidden)
    breast_thickness_cm: compressed breast thickness (3-6 cm, affects attenuation)

CPU reconstruction: Wiener filter + TV denoising (speckle/scatter/blur reduction)

Usage:
    cd datasets/benchmark/mammography
    python3 generate_dataset.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, uniform_filter
from scipy.signal import fftconvolve

BENCHMARK_DIR = Path(__file__).resolve().parent
IMAGE_SIZE = 256

# ── Physics constants ────────────────────────────────────────────────────────

# Attenuation coefficients at ~20 keV (mammographic energy range)
MU_ADIPOSE = 0.15          # cm^-1 (adipose/fat)
MU_GLANDULAR = 0.40        # cm^-1 (fibroglandular tissue)
MU_MASS = 0.50             # cm^-1 (mass/tumour, slightly above glandular)
MU_CALCIFICATION = 1.20    # cm^-1 (microcalcifications, high attenuation)
MU_LIGAMENT = 0.30         # cm^-1 (Cooper's ligament, connective tissue)
MU_SKIN = 0.35             # cm^-1 (skin layer)

PIXEL_SIZE_MM = 0.3        # mm per pixel (256 px -> 76.8 mm FOV ~ breast width)
PIXEL_SIZE_CM = PIXEL_SIZE_MM / 10.0

# Nominal I0: number of photons per pixel at 1 mGy dose
# Clinical mammography: ~200-400 photons/pixel at ~1 mGy for amorphous selenium detector
I0_PER_MGY = 300.0

# ── Mismatch spec ranges per tier ────────────────────────────────────────────

SPEC = {
    "public": {
        "dose_mGy":            {"min": 1.0,  "max": 3.0,  "unit": "mGy"},
        "scatter_fraction":    {"min": 0.10, "max": 0.25, "unit": ""},
        "detector_blur_sigma": {"min": 0.5,  "max": 1.5,  "unit": "pixels"},
        "breast_thickness_cm": {"min": 3.0,  "max": 6.0,  "unit": "cm"},
    },
    "dev": {
        "dose_mGy":            {"min": 0.5,  "max": 3.0,  "unit": "mGy"},
        "scatter_fraction":    {"min": 0.10, "max": 0.30, "unit": ""},
        "detector_blur_sigma": {"min": 0.5,  "max": 2.0,  "unit": "pixels"},
        "breast_thickness_cm": {"min": 3.0,  "max": 6.0,  "unit": "cm"},
    },
    "hidden": {
        "dose_mGy":            {"min": 0.3,  "max": 3.0,  "unit": "mGy"},
        "scatter_fraction":    {"min": 0.10, "max": 0.40, "unit": ""},
        "detector_blur_sigma": {"min": 0.5,  "max": 3.0,  "unit": "pixels"},
        "breast_thickness_cm": {"min": 3.0,  "max": 6.0,  "unit": "cm"},
    },
}

# ── Perlin-like noise for tissue texture ─────────────────────────────────────


def _octave_noise(H: int, W: int, rng: np.random.Generator,
                  octaves: int = 5, persistence: float = 0.5) -> np.ndarray:
    """Generate multi-octave smooth noise (Perlin-like) using Gaussian filtering.

    Produces spatially correlated noise with features at multiple scales,
    mimicking the heterogeneous texture of breast tissue.

    Returns array in [0, 1].
    """
    result = np.zeros((H, W), dtype=np.float64)
    amplitude = 1.0
    total_amp = 0.0
    for octave in range(octaves):
        freq = 2 ** octave
        sigma = max(H, W) / (freq * 4.0)
        noise = rng.standard_normal((H, W))
        smooth = gaussian_filter(noise, sigma=max(sigma, 1.0))
        result += amplitude * smooth
        total_amp += amplitude
        amplitude *= persistence
    result /= total_amp
    # Normalize to [0, 1]
    lo, hi = result.min(), result.max()
    if hi - lo > 1e-10:
        result = (result - lo) / (hi - lo)
    return result


# ── Breast shape mask ────────────────────────────────────────────────────────


def _breast_mask(H: int, W: int, rng: np.random.Generator,
                 shape_type: str = "standard") -> np.ndarray:
    """Generate a realistic breast contour mask.

    The breast is positioned with the chest wall on the left edge,
    filling roughly 60-80% of the image width.

    Returns boolean mask.
    """
    yy = np.linspace(-1.0, 1.0, H)[:, None]
    xx = np.linspace(-1.0, 1.0, W)[None, :]

    if shape_type == "wide":
        # Dense/large breast
        a_x = rng.uniform(0.75, 0.85)
        a_y = rng.uniform(0.55, 0.65)
    elif shape_type == "narrow":
        # Smaller breast
        a_x = rng.uniform(0.55, 0.65)
        a_y = rng.uniform(0.40, 0.50)
    else:
        a_x = rng.uniform(0.60, 0.75)
        a_y = rng.uniform(0.45, 0.55)

    # Chest wall at left edge (x = -1), breast extends rightward
    cx = -0.35 + rng.uniform(-0.05, 0.05)
    cy = rng.uniform(-0.05, 0.05)

    # Semi-elliptical shape (only right half visible, chest wall crops left)
    dist = ((xx - cx) / a_x) ** 2 + ((yy - cy) / a_y) ** 2
    mask = dist <= 1.0

    # Crop the left edge (chest wall) — keep only x > -0.9
    mask &= (xx > -0.9)

    # Add slight boundary irregularity using noise
    boundary_noise = gaussian_filter(rng.standard_normal((H, W)), sigma=8.0) * 0.06
    dist_noisy = dist + boundary_noise
    mask = (dist_noisy <= 1.0) & (xx > -0.9)

    return mask


# ── Skin layer ───────────────────────────────────────────────────────────────


def _skin_layer(mask: np.ndarray, thickness_px: int = 3) -> np.ndarray:
    """Generate a thin skin layer at the breast boundary.

    Returns boolean mask of the skin pixels (ring around the breast).
    """
    from scipy.ndimage import binary_erosion
    inner = binary_erosion(mask, iterations=thickness_px)
    return mask & ~inner


# ── Cooper's ligaments ───────────────────────────────────────────────────────


def _coopers_ligaments(H: int, W: int, mask: np.ndarray,
                       rng: np.random.Generator,
                       n_ligaments: int = 8) -> np.ndarray:
    """Generate thin curved Cooper's ligament structures.

    These are connective tissue strands that run perpendicular to the
    chest wall, appearing as thin bright curves on mammograms.

    Returns float64 array with ligament intensity [0, 1].
    """
    ligaments = np.zeros((H, W), dtype=np.float64)
    for _ in range(n_ligaments):
        # Start from near the chest wall (left side)
        y_start = rng.uniform(0.15 * H, 0.85 * H)
        x_start = rng.uniform(0.05 * W, 0.20 * W)

        # Create a curved path from chest wall outward
        n_points = 60
        t = np.linspace(0, 1, n_points)
        # Curve outward with some random curvature
        curve_amp = rng.uniform(-0.3, 0.3) * H
        x_path = x_start + t * rng.uniform(0.3, 0.7) * W
        y_path = y_start + curve_amp * np.sin(np.pi * t) + \
            rng.uniform(-0.05, 0.05) * H * np.sin(2 * np.pi * t)

        # Draw the path with anti-aliased thickness
        for j in range(len(t) - 1):
            y0, x0 = int(round(y_path[j])), int(round(x_path[j]))
            y1, x1 = int(round(y_path[j + 1])), int(round(x_path[j + 1]))
            # Bresenham-like thick line
            thickness = rng.uniform(0.8, 1.5)
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    yy = y0 + dy
                    xx = x0 + dx
                    if 0 <= yy < H and 0 <= xx < W:
                        dist = np.sqrt(dy ** 2 + dx ** 2)
                        if dist <= thickness:
                            val = 0.4 * max(0, 1.0 - dist / thickness)
                            ligaments[yy, xx] = max(ligaments[yy, xx], val)

    # Slight blur to make them look more natural
    ligaments = gaussian_filter(ligaments, sigma=0.5)
    ligaments *= mask
    return ligaments


# ── Fibroglandular tissue regions ────────────────────────────────────────────


def _glandular_tissue(H: int, W: int, mask: np.ndarray,
                      rng: np.random.Generator,
                      density_class: str = "heterogeneous") -> np.ndarray:
    """Generate fibroglandular tissue distribution.

    density_class:
        "fatty"          : mostly fat, ~10-25% glandular (BI-RADS A)
        "scattered"      : 25-50% glandular (BI-RADS B)
        "heterogeneous"  : 50-75% glandular (BI-RADS C)
        "extremely_dense": >75% glandular (BI-RADS D)

    Returns float64 array with glandular fraction [0, 1].
    """
    density_fractions = {
        "fatty": (0.10, 0.25),
        "scattered": (0.25, 0.50),
        "heterogeneous": (0.50, 0.75),
        "extremely_dense": (0.75, 0.95),
    }
    lo_frac, hi_frac = density_fractions.get(density_class, (0.30, 0.60))
    target_frac = rng.uniform(lo_frac, hi_frac)

    # Generate multi-scale texture
    texture = _octave_noise(H, W, rng, octaves=5, persistence=0.55)

    # Create clustered glandular regions (central tendency)
    yy = np.linspace(-1.0, 1.0, H)[:, None]
    xx = np.linspace(-1.0, 1.0, W)[None, :]
    # Glandular tissue concentrates centrally (retromammary fat layer at boundary)
    central_weight = np.exp(-2.0 * ((xx + 0.1) ** 2 + yy ** 2))
    central_weight = central_weight / central_weight.max()

    # Combine texture with central preference
    glandular = texture * 0.6 + central_weight * 0.4
    glandular = glandular * mask

    # Threshold to achieve target density fraction
    valid_pixels = mask.sum()
    if valid_pixels > 0:
        sorted_vals = np.sort(glandular[mask])[::-1]
        n_glandular = int(target_frac * valid_pixels)
        if n_glandular > 0 and n_glandular < len(sorted_vals):
            threshold = sorted_vals[n_glandular]
            glandular = np.where(glandular >= threshold, glandular, 0.0)
        # Smooth boundaries
        glandular = gaussian_filter(glandular, sigma=2.0)
        glandular *= mask
        if glandular.max() > 0:
            glandular /= glandular.max()

    return glandular


# ── Masses (tumors/lesions) ──────────────────────────────────────────────────


def _masses(H: int, W: int, mask: np.ndarray,
            rng: np.random.Generator,
            n_masses: int = 1,
            spiculated: bool = False) -> np.ndarray:
    """Generate mass lesions (round or spiculated).

    Returns float64 array with mass intensity [0, 1].
    """
    masses = np.zeros((H, W), dtype=np.float64)
    yy = np.linspace(-1.0, 1.0, H)[:, None]
    xx = np.linspace(-1.0, 1.0, W)[None, :]

    for _ in range(n_masses):
        # Place mass within the breast
        for _attempt in range(20):
            cy = rng.uniform(-0.3, 0.3)
            cx = rng.uniform(-0.3, 0.4)
            iy, ix = int((cy + 1) * H / 2), int((cx + 1) * W / 2)
            if 0 <= iy < H and 0 <= ix < W and mask[iy, ix]:
                break

        radius = rng.uniform(0.03, 0.08)
        # Elliptical mass with random eccentricity
        a = radius * rng.uniform(0.8, 1.2)
        b = radius * rng.uniform(0.8, 1.2)
        angle = rng.uniform(0, 360)
        ca, sa = np.cos(np.radians(angle)), np.sin(np.radians(angle))
        yr = (yy - cy) * ca - (xx - cx) * sa
        xr = (yy - cy) * sa + (xx - cx) * ca
        dist = (xr / a) ** 2 + (yr / b) ** 2

        if spiculated:
            # Add radial spiculations
            angles_grid = np.arctan2(yy - cy, xx - cx)
            n_spic = rng.integers(5, 12)
            spic_pattern = np.zeros_like(dist)
            for _ in range(n_spic):
                spic_angle = rng.uniform(0, 2 * np.pi)
                spic_width = rng.uniform(0.02, 0.05)
                angular_dist = np.abs(
                    np.arctan2(np.sin(angles_grid - spic_angle),
                               np.cos(angles_grid - spic_angle))
                )
                spic_pattern += np.exp(-angular_dist ** 2 / (2 * spic_width ** 2))
            spic_pattern /= max(spic_pattern.max(), 1e-10)
            # Spiculations extend further from core
            core = np.exp(-dist * 3.0)
            spicule_ext = np.exp(-dist * 0.8) * spic_pattern * 0.5
            mass_val = np.clip(core + spicule_ext, 0, 1)
        else:
            # Smooth round/oval mass
            mass_val = np.exp(-dist * 4.0)

        mass_val *= mask
        masses = np.maximum(masses, mass_val)

    # Slight smoothing
    masses = gaussian_filter(masses, sigma=0.8)
    return masses


# ── Microcalcifications ──────────────────────────────────────────────────────


def _microcalcifications(H: int, W: int, mask: np.ndarray,
                         rng: np.random.Generator,
                         n_clusters: int = 2,
                         points_per_cluster: int = 8) -> np.ndarray:
    """Generate microcalcification clusters.

    Clusters of tiny high-attenuation dots (50-500 um),
    appearing as bright spots on mammograms.

    Returns float64 array with calcification intensity [0, 1].
    """
    calcs = np.zeros((H, W), dtype=np.float64)

    for _ in range(n_clusters):
        # Cluster center within the breast
        for _attempt in range(30):
            cy = rng.integers(int(0.15 * H), int(0.85 * H))
            cx = rng.integers(int(0.15 * W), int(0.85 * W))
            if mask[cy, cx]:
                break

        # Cluster spread (in pixels)
        spread = rng.uniform(5, 20)
        n_points = rng.integers(max(3, points_per_cluster - 3),
                                points_per_cluster + 5)

        for _ in range(n_points):
            py = int(cy + rng.normal(0, spread))
            px = int(cx + rng.normal(0, spread))
            if 0 <= py < H and 0 <= px < W and mask[py, px]:
                # Size: 1-3 pixels (50-150 um at 0.3mm/pixel)
                size = rng.integers(1, 4)
                intensity = rng.uniform(0.7, 1.0)
                for dy in range(-size, size + 1):
                    for dx in range(-size, size + 1):
                        yy, xx = py + dy, px + dx
                        if (0 <= yy < H and 0 <= xx < W and
                                dy ** 2 + dx ** 2 <= size ** 2):
                            calcs[yy, xx] = max(calcs[yy, xx], intensity)

    return calcs


# ── Phantom generators ───────────────────────────────────────────────────────


def make_fatty_phantom(H: int, W: int, seed: int,
                       variant: int = 0) -> tuple[np.ndarray, str]:
    """Mostly fatty breast (BI-RADS A/B).

    Returns: (attenuation_map float64 [0,1], scene_name)
    """
    rng = np.random.default_rng(seed)
    atten = np.zeros((H, W), dtype=np.float64)

    density_cls = "fatty" if variant < 2 else "scattered"
    mask = _breast_mask(H, W, rng, "standard")
    skin = _skin_layer(mask, thickness_px=2)
    gland = _glandular_tissue(H, W, mask, rng, density_cls)
    ligaments = _coopers_ligaments(H, W, mask, rng, n_ligaments=6 + variant)

    # Background texture
    bg_texture = _octave_noise(H, W, rng, octaves=4, persistence=0.4)
    bg_texture = bg_texture * 0.05  # subtle texture variation

    # Build attenuation map
    atten[mask] = MU_ADIPOSE + bg_texture[mask] * 0.05
    atten += gland * (MU_GLANDULAR - MU_ADIPOSE)
    atten += ligaments * MU_LIGAMENT
    atten[skin] = MU_SKIN

    # Normalize to [0, 1]
    atten *= mask
    if atten.max() > 0:
        atten /= max(atten.max(), MU_CALCIFICATION)

    return atten.astype(np.float64), f"fatty_{variant:02d}"


def make_dense_phantom(H: int, W: int, seed: int,
                       variant: int = 0) -> tuple[np.ndarray, str]:
    """Dense breast (BI-RADS C/D) with prominent glandular tissue.

    Returns: (attenuation_map float64 [0,1], scene_name)
    """
    rng = np.random.default_rng(seed)
    atten = np.zeros((H, W), dtype=np.float64)

    density_cls = "heterogeneous" if variant < 2 else "extremely_dense"
    mask = _breast_mask(H, W, rng, "wide")
    skin = _skin_layer(mask, thickness_px=3)
    gland = _glandular_tissue(H, W, mask, rng, density_cls)
    ligaments = _coopers_ligaments(H, W, mask, rng, n_ligaments=10 + variant)

    bg_texture = _octave_noise(H, W, rng, octaves=5, persistence=0.5)
    bg_texture = bg_texture * 0.08

    atten[mask] = MU_ADIPOSE + bg_texture[mask] * 0.05
    atten += gland * (MU_GLANDULAR - MU_ADIPOSE) * 1.2  # stronger glandular
    atten += ligaments * MU_LIGAMENT
    atten[skin] = MU_SKIN

    atten *= mask
    if atten.max() > 0:
        atten /= max(atten.max(), MU_CALCIFICATION)

    return atten.astype(np.float64), f"dense_{variant:02d}"


def make_lesion_phantom(H: int, W: int, seed: int,
                        variant: int = 0) -> tuple[np.ndarray, str]:
    """Breast with calcifications and/or masses.

    Returns: (attenuation_map float64 [0,1], scene_name)
    """
    rng = np.random.default_rng(seed)
    atten = np.zeros((H, W), dtype=np.float64)

    density_cls = "scattered" if variant < 2 else "heterogeneous"
    shape = "standard" if variant < 2 else "wide"
    mask = _breast_mask(H, W, rng, shape)
    skin = _skin_layer(mask, thickness_px=2)
    gland = _glandular_tissue(H, W, mask, rng, density_cls)
    ligaments = _coopers_ligaments(H, W, mask, rng, n_ligaments=7 + variant)

    bg_texture = _octave_noise(H, W, rng, octaves=4, persistence=0.45)
    bg_texture = bg_texture * 0.06

    atten[mask] = MU_ADIPOSE + bg_texture[mask] * 0.05
    atten += gland * (MU_GLANDULAR - MU_ADIPOSE)
    atten += ligaments * MU_LIGAMENT

    # Add masses
    spiculated = variant % 2 == 1
    mass_map = _masses(H, W, mask, rng, n_masses=1 + variant // 2,
                       spiculated=spiculated)
    atten += mass_map * MU_MASS

    # Add microcalcification clusters
    calc_map = _microcalcifications(H, W, mask, rng,
                                    n_clusters=1 + variant,
                                    points_per_cluster=8 + variant * 2)
    atten += calc_map * MU_CALCIFICATION

    atten[skin] = MU_SKIN

    atten *= mask
    if atten.max() > 0:
        atten /= max(atten.max(), MU_CALCIFICATION)

    return atten.astype(np.float64), f"lesion_{variant:02d}"


# ── Forward Model ────────────────────────────────────────────────────────────


def mammography_forward_model(
    x_true: np.ndarray,
    dose_mGy: float,
    scatter_fraction: float,
    detector_blur_sigma: float,
    breast_thickness_cm: float,
    rng: np.random.Generator,
) -> dict:
    """Apply mammography forward model (Beer-Lambert projection + degradation).

    Since mammography is a single 2D projection (not tomographic), the
    forward model is:
        1. Compute line integral of attenuation through the breast thickness
        2. Apply Beer-Lambert law: I = I0 * exp(-mu * thickness)
        3. Add scatter (low-frequency background)
        4. Apply detector blur (Gaussian PSF)
        5. Add Poisson noise (quantum-limited)

    Args:
        x_true:              (H, W) attenuation map [0, 1]
        dose_mGy:            radiation dose in mGy (controls photon count)
        scatter_fraction:    fraction of signal due to scatter
        detector_blur_sigma: detector PSF sigma in pixels
        breast_thickness_cm: compressed breast thickness in cm
        rng:                 random generator

    Returns:
        dict with projection_ideal, projection_measured
    """
    H, W = x_true.shape

    # 1. Scale attenuation map to physical units
    # x_true is in [0, 1], scale to physical attenuation (cm^-1)
    mu_max = MU_CALCIFICATION  # maximum attenuation in our phantoms
    mu_map = x_true * mu_max  # physical attenuation coefficients

    # 2. Line integral through compressed breast (thickness * attenuation)
    line_integral = mu_map * breast_thickness_cm  # unitless (cm^-1 * cm)

    # 3. Beer-Lambert: ideal intensity (no noise, no scatter)
    I0 = I0_PER_MGY * dose_mGy
    I_ideal = I0 * np.exp(-line_integral)

    # The ideal projection is the clean, noise-free, scatter-free image
    projection_ideal = I_ideal.astype(np.float32)

    # 4. Add scatter (low-frequency background proportional to breast tissue)
    # Scatter is a smooth, slowly varying background
    scatter_base = gaussian_filter(I_ideal, sigma=30.0)
    scatter = scatter_fraction * scatter_base
    # Add slight spatial variation to scatter
    scatter_noise = gaussian_filter(rng.standard_normal((H, W)), sigma=20.0)
    scatter += scatter_noise * scatter.mean() * 0.05
    scatter = np.maximum(scatter, 0.0)

    I_with_scatter = I_ideal + scatter

    # 5. Apply detector blur
    if detector_blur_sigma > 0.1:
        I_blurred = gaussian_filter(I_with_scatter, sigma=detector_blur_sigma)
    else:
        I_blurred = I_with_scatter.copy()

    # 6. Poisson noise (quantum-limited)
    I_blurred = np.maximum(I_blurred, 0.01)
    I_noisy = rng.poisson(I_blurred).astype(np.float64)

    # 7. Add small readout noise
    readout_sigma = 2.0  # electrons
    I_noisy += rng.normal(0, readout_sigma, I_noisy.shape)
    I_noisy = np.maximum(I_noisy, 1.0)

    projection_measured = I_noisy.astype(np.float32)

    return {
        "projection_ideal": projection_ideal,
        "projection_measured": projection_measured,
    }


# ── Reconstruction: Wiener + TV denoising ────────────────────────────────────


def _tv_denoise(image: np.ndarray, weight: float = 0.1,
                n_iter: int = 50) -> np.ndarray:
    """Simple Total Variation denoising (Chambolle 2004 dual projection).

    Minimizes: ||u - image||^2 / 2 + weight * TV(u)

    Args:
        image: input image (2D float64)
        weight: TV regularization weight
        n_iter: number of iterations

    Returns:
        denoised image
    """
    H, W = image.shape
    px = np.zeros((H, W), dtype=np.float64)
    py = np.zeros((H, W), dtype=np.float64)
    tau = 0.25  # step size

    for _ in range(n_iter):
        # Compute divergence of (px, py)
        div = np.zeros((H, W), dtype=np.float64)
        div[1:, :] += px[1:, :] - px[:-1, :]
        div[:, 1:] += py[:, 1:] - py[:, :-1]

        # Update primal
        u = image + weight * div

        # Gradient of u
        gx = np.zeros_like(u)
        gy = np.zeros_like(u)
        gx[:-1, :] = u[1:, :] - u[:-1, :]
        gy[:, :-1] = u[:, 1:] - u[:, :-1]

        # Gradient magnitude
        norm = np.sqrt(gx ** 2 + gy ** 2 + 1e-10)

        # Update dual (projection onto unit ball)
        px = (px + tau * gx / weight) / (1.0 + tau * norm / weight)
        py = (py + tau * gy / weight) / (1.0 + tau * norm / weight)

    # Final reconstruction
    div = np.zeros((H, W), dtype=np.float64)
    div[1:, :] += px[1:, :] - px[:-1, :]
    div[:, 1:] += py[:, 1:] - py[:, :-1]
    return image + weight * div


def reconstruct_mammogram(projection_measured: np.ndarray,
                          projection_ideal: np.ndarray,
                          dose_mGy: float,
                          detector_blur_sigma: float,
                          breast_thickness_cm: float) -> np.ndarray:
    """CPU baseline reconstruction for mammography.

    Since mammography is a single 2D projection, "reconstruction" is
    essentially denoising + deblurring to recover the attenuation map.

    Pipeline:
        1. Convert measured projection back to attenuation domain
           (inverse Beer-Lambert)
        2. Wiener deconvolution for detector blur
        3. TV denoising for noise reduction

    Returns:
        recon: (H, W) float64 — estimated attenuation map [0, 1]
    """
    # 1. Inverse Beer-Lambert: mu * t = -log(I_meas / I0)
    I0 = I0_PER_MGY * dose_mGy
    I_meas = np.maximum(projection_measured.astype(np.float64), 1.0)

    # Estimate the clean projection by simple noise reduction first
    # (helps stabilize the log transform at low counts)
    I_smoothed = gaussian_filter(I_meas, sigma=0.5)
    I_smoothed = np.maximum(I_smoothed, 1.0)

    # Convert to attenuation*thickness
    mu_t = -np.log(I_smoothed / I0)
    mu_t = np.maximum(mu_t, 0.0)

    # Convert to attenuation (divide by thickness)
    mu_est = mu_t / breast_thickness_cm

    # Normalize to [0, 1] using known max attenuation
    atten_est = mu_est / MU_CALCIFICATION
    atten_est = np.clip(atten_est, 0.0, 1.0)

    # 2. Wiener deconvolution for detector blur
    if detector_blur_sigma > 0.3:
        # Build PSF kernel
        k_size = int(np.ceil(detector_blur_sigma * 6)) | 1
        k_half = k_size // 2
        y_k, x_k = np.mgrid[-k_half:k_half + 1, -k_half:k_half + 1]
        psf = np.exp(-(x_k ** 2 + y_k ** 2) / (2 * detector_blur_sigma ** 2))
        psf /= psf.sum()

        # Wiener deconvolution in Fourier domain
        pad_shape = (atten_est.shape[0] + psf.shape[0],
                     atten_est.shape[1] + psf.shape[1])
        # Pad to avoid circular artifacts
        psf_pad = np.zeros(pad_shape, dtype=np.float64)
        psf_pad[:psf.shape[0], :psf.shape[1]] = psf
        # Center PSF
        psf_pad = np.roll(psf_pad, -k_half, axis=0)
        psf_pad = np.roll(psf_pad, -k_half, axis=1)

        img_pad = np.zeros(pad_shape, dtype=np.float64)
        img_pad[:atten_est.shape[0], :atten_est.shape[1]] = atten_est

        PSF_F = np.fft.fft2(psf_pad)
        IMG_F = np.fft.fft2(img_pad)

        # Estimate noise power (from high-frequency content)
        noise_power = 1e-3 / max(dose_mGy, 0.1)
        wiener = np.conj(PSF_F) / (np.abs(PSF_F) ** 2 + noise_power)
        result_f = IMG_F * wiener
        result = np.real(np.fft.ifft2(result_f))
        atten_est = result[:atten_est.shape[0], :atten_est.shape[1]]
        atten_est = np.clip(atten_est, 0.0, 1.0)

    # 3. TV denoising
    tv_weight = 0.08 / max(dose_mGy, 0.1)  # stronger TV at lower dose
    tv_weight = min(tv_weight, 0.3)
    atten_est = _tv_denoise(atten_est, weight=tv_weight, n_iter=60)
    atten_est = np.clip(atten_est, 0.0, 1.0)

    return atten_est.astype(np.float64)


# ── Metrics ──────────────────────────────────────────────────────────────────


def compute_psnr(gt: np.ndarray, recon: np.ndarray) -> float:
    mse = np.mean((gt.astype(np.float64) - recon.astype(np.float64)) ** 2)
    if mse < 1e-12:
        return 100.0
    data_range = float(gt.max() - gt.min())
    if data_range < 1e-12:
        return 0.0
    return float(10 * np.log10(data_range ** 2 / mse))


def compute_ssim(gt: np.ndarray, recon: np.ndarray) -> float:
    """Windowed SSIM (11x11 blocks)."""
    gt = gt.astype(np.float64)
    recon = recon.astype(np.float64)
    data_range = gt.max() - gt.min()
    if data_range < 1e-12:
        return 0.0
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    win_size = 11

    mu_x = uniform_filter(gt, size=win_size, mode='reflect')
    mu_y = uniform_filter(recon, size=win_size, mode='reflect')
    mu_x2 = uniform_filter(gt ** 2, size=win_size, mode='reflect')
    mu_y2 = uniform_filter(recon ** 2, size=win_size, mode='reflect')
    mu_xy = uniform_filter(gt * recon, size=win_size, mode='reflect')

    var_x = mu_x2 - mu_x ** 2
    var_y = mu_y2 - mu_y ** 2
    cov_xy = mu_xy - mu_x * mu_y

    num = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
    ssim_map = num / den
    return float(ssim_map.mean())


# ── Image helpers ────────────────────────────────────────────────────────────


def _norm(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo + 1e-8)


def _save_png(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(
        np.clip(_norm(arr) * 255, 0, 255).astype(np.uint8), "L"
    ).save(str(path))


def _save_overview(x_true, proj_ideal, proj_meas, recon, path: Path) -> None:
    """4-panel overview: GT | ideal projection | measured projection | recon."""
    th, tw = 128, 128

    def _r(a):
        pil = Image.fromarray(
            np.clip(_norm(a) * 255, 0, 255).astype(np.uint8), "L")
        return np.array(pil.resize((tw, th), Image.LANCZOS)) / 255.0

    ov = np.zeros((th, 4 * tw), dtype=np.float32)
    ov[:, 0:tw] = _r(x_true)
    ov[:, tw:2 * tw] = _r(proj_ideal)
    ov[:, 2 * tw:3 * tw] = _r(proj_meas)
    ov[:, 3 * tw:4 * tw] = _r(recon)
    _save_png(ov, path)


# ── Phantom pools per tier ───────────────────────────────────────────────────


def generate_phantoms_public(n: int = 12
                             ) -> list[tuple[np.ndarray, str]]:
    """12 public phantoms: 4 fatty + 4 dense + 4 with lesions."""
    phantoms = []
    for i in range(4):
        phantoms.append(make_fatty_phantom(IMAGE_SIZE, IMAGE_SIZE,
                                           seed=100 + i, variant=i))
    for i in range(4):
        phantoms.append(make_dense_phantom(IMAGE_SIZE, IMAGE_SIZE,
                                           seed=200 + i, variant=i))
    for i in range(4):
        phantoms.append(make_lesion_phantom(IMAGE_SIZE, IMAGE_SIZE,
                                            seed=300 + i, variant=i))
    return phantoms[:n]


def generate_phantoms_dev(n: int = 20
                          ) -> list[tuple[np.ndarray, str]]:
    """20 dev phantoms: augmented with rotation/flip/zoom."""
    from scipy.ndimage import rotate as nd_rotate, zoom as nd_zoom

    generators = [make_fatty_phantom, make_dense_phantom, make_lesion_phantom]
    phantoms = []
    rng = np.random.default_rng(5000)

    for i in range(n):
        gen_fn = generators[i % 3]
        atten, name = gen_fn(IMAGE_SIZE, IMAGE_SIZE, seed=500 + i, variant=i)

        # Augment: rotation
        angle = float(rng.uniform(15, 345))
        atten = nd_rotate(atten, angle, reshape=False, mode='constant',
                          cval=0.0)

        # Augment: flip
        if rng.random() < 0.5:
            atten = np.fliplr(atten)
        if rng.random() < 0.3:
            atten = np.flipud(atten)

        # Augment: zoom
        zoom_f = float(rng.uniform(0.85, 1.15))
        if abs(zoom_f - 1.0) > 0.02:
            atten = _zoom_crop(atten, zoom_f, IMAGE_SIZE)

        atten = np.clip(atten, 0.0, 1.0)
        phantoms.append((atten, f"dev_{name}"))

    return phantoms


def generate_phantoms_hidden(n: int = 20
                             ) -> list[tuple[np.ndarray, str]]:
    """20 hidden phantoms: adversarial modifications."""
    from scipy.ndimage import rotate as nd_rotate

    generators = [make_fatty_phantom, make_dense_phantom, make_lesion_phantom]
    phantoms = []
    rng = np.random.default_rng(8000)

    for i in range(n):
        gen_fn = generators[i % 3]
        atten, name = gen_fn(IMAGE_SIZE, IMAGE_SIZE, seed=800 + i,
                             variant=i + 10)

        # Augment: aggressive rotation + flip
        angle = float(rng.uniform(20, 340))
        atten = nd_rotate(atten, angle, reshape=False, mode='constant',
                          cval=0.0)
        if rng.random() < 0.7:
            atten = np.fliplr(atten)
        if rng.random() < 0.5:
            atten = np.flipud(atten)

        # Aggressive zoom
        zoom_f = float(rng.uniform(0.70, 1.30))
        atten = _zoom_crop(atten, zoom_f, IMAGE_SIZE)

        # Add extra adversarial micro-calcifications (very small, hard to detect)
        n_micro = rng.integers(3, 8)
        for _ in range(n_micro):
            py = rng.integers(20, IMAGE_SIZE - 20)
            px = rng.integers(20, IMAGE_SIZE - 20)
            if atten[py, px] > 0.05:  # only in tissue regions
                size = rng.integers(1, 3)
                for dy in range(-size, size + 1):
                    for dx in range(-size, size + 1):
                        yy, xx = py + dy, px + dx
                        if (0 <= yy < IMAGE_SIZE and 0 <= xx < IMAGE_SIZE and
                                dy ** 2 + dx ** 2 <= size ** 2):
                            atten[yy, xx] = min(1.0,
                                                atten[yy, xx] +
                                                rng.uniform(0.3, 0.6))

        # Add subtle mass in some samples
        if rng.random() < 0.4:
            mask_any = atten > 0.02
            mass_map = _masses(IMAGE_SIZE, IMAGE_SIZE, mask_any, rng,
                               n_masses=1, spiculated=rng.random() < 0.5)
            atten += mass_map * 0.3
            atten = np.clip(atten, 0.0, 1.0)

        phantoms.append((atten, f"hidden_{name}"))

    return phantoms


def _zoom_crop(arr: np.ndarray, zoom_f: float, size: int) -> np.ndarray:
    """Zoom and crop/pad to target size."""
    from scipy.ndimage import zoom as nd_zoom
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


# ── Tier generation ──────────────────────────────────────────────────────────


def sample_mismatch(rng: np.random.Generator, spec: dict) -> dict:
    return {k: float(rng.uniform(v["min"], v["max"])) for k, v in spec.items()}


def generate_tier(
    tier: str,
    phantoms: list[tuple[np.ndarray, str]],
    base_seed: int,
) -> None:
    """Generate one tier of the mammography benchmark."""
    spec_ranges = SPEC[tier]
    tier_dir = BENCHMARK_DIR / tier
    images_dir = tier_dir / "images"
    if tier_dir.exists():
        shutil.rmtree(tier_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    h5_path = tier_dir / f"mammography_challenge_{tier}.h5"
    rng = np.random.default_rng(base_seed)
    true_specs = {}
    all_psnrs = []
    all_ssims = []

    with h5py.File(h5_path, "w") as f:
        f.attrs["description"] = (
            f"PWM mammography benchmark -- {tier} tier "
            f"(Beer-Lambert projection + Poisson noise + scatter + detector blur)"
        )
        f.attrs["spec_ranges"] = json.dumps(spec_ranges)
        f.attrs["geometry"] = json.dumps({
            "image_size": IMAGE_SIZE,
            "pixel_size_mm": PIXEL_SIZE_MM,
            "fov_mm": IMAGE_SIZE * PIXEL_SIZE_MM,
            "I0_per_mGy": I0_PER_MGY,
        })
        f.attrs["forward_model"] = (
            "y_i = I_0 * exp(-mu(x,E) * breast_thickness) + scatter + noise"
        )

        for idx, (x_true, scene_name) in enumerate(phantoms):
            key = f"sample_{idx:02d}"
            print(f"  [{tier}] Generating {key} ({scene_name})...",
                  end="", flush=True)

            mis = sample_mismatch(rng, spec_ranges)
            true_specs[key] = mis

            # Apply forward model
            result = mammography_forward_model(
                x_true,
                dose_mGy=mis["dose_mGy"],
                scatter_fraction=mis["scatter_fraction"],
                detector_blur_sigma=mis["detector_blur_sigma"],
                breast_thickness_cm=mis["breast_thickness_cm"],
                rng=rng,
            )
            proj_ideal = result["projection_ideal"]
            proj_meas = result["projection_measured"]

            # Baseline reconstruction
            recon = reconstruct_mammogram(
                proj_meas, proj_ideal,
                dose_mGy=mis["dose_mGy"],
                detector_blur_sigma=mis["detector_blur_sigma"],
                breast_thickness_cm=mis["breast_thickness_cm"],
            )
            recon = recon.astype(np.float32)

            # Metrics
            psnr = compute_psnr(x_true, recon)
            ssim = compute_ssim(x_true, recon)
            all_psnrs.append(psnr)
            all_ssims.append(ssim)

            # Save to HDF5
            grp = f.create_group(key)
            grp.create_dataset("x_true", data=x_true.astype(np.float32),
                               compression="gzip")
            grp.create_dataset("projection_ideal", data=proj_ideal,
                               compression="gzip")
            grp.create_dataset("projection_measured", data=proj_meas,
                               compression="gzip")
            grp.create_dataset("reconstruction", data=recon,
                               compression="gzip")
            grp.attrs["metadata"] = json.dumps({
                "scene": scene_name,
                "shape": list(x_true.shape),
                "psnr_baseline": float(psnr),
                "ssim_baseline": float(ssim),
            })
            grp.attrs["true_spec"] = json.dumps(mis)
            grp.attrs["spec_ranges"] = json.dumps(spec_ranges)

            # Save images
            sample_dir = images_dir / f"sample_{idx:02d}_{scene_name}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            _save_png(x_true, sample_dir / "ground_truth.png")
            _save_png(proj_ideal, sample_dir / "projection_ideal.png")
            _save_png(proj_meas, sample_dir / "projection_measured.png")
            _save_png(recon, sample_dir / "reconstruction.png")
            _save_overview(x_true, proj_ideal, proj_meas, recon,
                           sample_dir / "overview.png")
            with open(sample_dir / "spec.json", "w") as sf:
                json.dump({
                    "scene": scene_name,
                    "spec_ranges": spec_ranges,
                    "true_spec": mis,
                    "psnr_baseline": psnr,
                    "ssim_baseline": ssim,
                }, sf, indent=2)

            print(f"  PSNR={psnr:.2f} dB, SSIM={ssim:.3f}  "
                  f"dose={mis['dose_mGy']:.1f} mGy  "
                  f"scatter={mis['scatter_fraction']:.2f}  "
                  f"blur={mis['detector_blur_sigma']:.2f} px  "
                  f"thick={mis['breast_thickness_cm']:.1f} cm")

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


# ── README ───────────────────────────────────────────────────────────────────


def _write_top_readme() -> None:
    txt = """# Mammography -- 2-D Beer-Lambert X-ray Projection

## Overview

Full-field digital mammography (FFDM) benchmark with realistic breast tissue
phantoms and clinical-grade physics: Beer-Lambert attenuation, Poisson quantum
noise, X-ray scatter, and detector point-spread function.

## Forward Model

```
y_i = I_0 * exp(-mu(x, E) * breast_thickness) + scatter + noise

where:
    x_true           : 2D attenuation map (256x256) of breast tissue
    I_0              : incident X-ray fluence (Mo/Rh target, 25-35 kVp)
    mu(x, E)         : linear attenuation coefficient (cm^-1)
    breast_thickness : compressed breast thickness (3-6 cm)
    scatter          : low-frequency scatter background
    noise            : Poisson (quantum) + readout noise
```

## Geometry

| Parameter | Value |
|-----------|-------|
| IMAGE_SIZE | 256 x 256 |
| pixel_size | 0.3 mm/px |
| FOV | 76.8 mm |
| I0_per_mGy | 300 photons/pixel/mGy |
| readout_noise | 2.0 electrons sigma |

## Tissue Attenuation Coefficients (20 keV)

| Tissue | mu (cm^-1) |
|--------|-----------|
| Adipose | 0.15 |
| Fibroglandular | 0.40 |
| Mass/Tumour | 0.50 |
| Calcification | 1.20 |
| Cooper's ligament | 0.30 |
| Skin | 0.35 |

## Mismatch Parameters

| Parameter | Description | Public | Dev | Hidden |
|-----------|-------------|--------|-----|--------|
| dose_mGy | Radiation dose | 1.0-3.0 mGy | 0.5-3.0 mGy | 0.3-3.0 mGy |
| scatter_fraction | Scatter / total | 0.10-0.25 | 0.10-0.30 | 0.10-0.40 |
| detector_blur_sigma | Detector PSF | 0.5-1.5 px | 0.5-2.0 px | 0.5-3.0 px |
| breast_thickness_cm | Compressed thickness | 3-6 cm | 3-6 cm | 3-6 cm |

## Phantoms

| Type | Samples | Description |
|------|---------|-------------|
| Fatty (BI-RADS A/B) | 4/tier | Mostly adipose with scattered glandular |
| Dense (BI-RADS C/D) | 4/tier | Prominent fibroglandular tissue |
| Lesion | 4/tier | Masses + microcalcification clusters |

## Dataset Structure

```
mammography/
+-- README.md
+-- generate_dataset.py
+-- public/    12 samples (GT + ideal projection + true spec visible)
+-- dev/       20 samples (blind eval, augmented variants)
+-- hidden/    20 samples (adversarial: micro-calcifications, extreme params)
```

## HDF5 Structure (per sample)

```
sample_XX/
+-- x_true (256, 256) float32          # Ground truth attenuation map [0, 1]
+-- projection_ideal (256, 256) float32 # Clean projection (no noise/scatter)
+-- projection_measured (256, 256) float32 # Measured (noisy) projection
+-- reconstruction (256, 256) float32   # Baseline Wiener+TV reconstruction
```

## Scoring

```
Score = 0.4 * PSNR_norm + 0.4 * SSIM + 0.2 * Consistency
```

## References

1. Dance, D.R. et al. (2000) "Additional factors for the estimation of mean
   glandular dose using the UK mammography dosimetry protocol,"
   Phys. Med. Biol. 45, 3225-3240.
2. Siddon, R.L. (1985) "Fast calculation of the exact radiological path
   for a three-dimensional CT array," Med. Phys. 12, 252-255.
3. Vedantham, S. et al. (2015) "Digital Breast Tomosynthesis: State of the
   Art," Radiology 277, 663-684.
4. PWM Benchmark: https://pwm.platformai.org/benchmark/mammography
"""
    with open(BENCHMARK_DIR / "README.md", "w") as f:
        f.write(txt)


# ── Gallery image generation ─────────────────────────────────────────────────


def generate_gallery_images() -> None:
    """Generate gallery images for the platform benchmark page.

    Creates scene_00 through scene_03 with:
      gt.png, measurement_I.png, measurement_II.png,
      recon_I.png, recon_II.png, recon_III.png
    """
    gallery_base = (BENCHMARK_DIR.parent.parent.parent /
                    "platform" / "pwm_platform" / "static" / "img" /
                    "benchmark_gallery" / "mammography")

    h5_path = BENCHMARK_DIR / "public" / "mammography_challenge_public.h5"
    if not h5_path.exists():
        print("  [gallery] Public HDF5 not found, skipping gallery generation.")
        return

    # Pick diverse samples: fatty(0), dense(4), lesion(8), fatty_variant(1)
    gallery_sample_indices = [0, 4, 8, 1]

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
            proj_ideal = grp["projection_ideal"][:]
            proj_meas = grp["projection_measured"][:]
            recon = grp["reconstruction"][:]

            # gt.png — ground truth attenuation
            _save_png(x_true, scene_dir / "gt.png")

            # measurement_I.png — measured projection (noisy)
            _save_png(proj_meas, scene_dir / "measurement_I.png")

            # measurement_II.png — ideal projection (clean)
            _save_png(proj_ideal, scene_dir / "measurement_II.png")

            # recon_I.png — Wiener+TV reconstruction
            _save_png(recon, scene_dir / "recon_I.png")

            # recon_II.png — difference |GT - recon|
            diff = np.abs(x_true - recon)
            _save_png(diff, scene_dir / "recon_II.png")

            # recon_III.png — log-compressed measured projection
            proj_log = np.log1p(proj_meas)
            _save_png(proj_log, scene_dir / "recon_III.png")

            print(f"  [gallery] scene_{scene_idx:02d} saved to {scene_dir}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    print("Mammography Benchmark Dataset Generator")
    print("=" * 68)
    print(f"Output: {BENCHMARK_DIR}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}, "
          f"pixel size: {PIXEL_SIZE_MM} mm\n")

    # ── Public tier (12 samples) ────────────────────────────────────────────
    print("Generating public tier (12 samples: 4 fatty + 4 dense + 4 lesion)...")
    public_phantoms = generate_phantoms_public(12)
    generate_tier("public", public_phantoms, base_seed=1000)

    # ── Dev tier (20 samples) ──────────────────────────────────────────────
    print("\nGenerating dev tier (20 samples, augmented)...")
    dev_phantoms = generate_phantoms_dev(20)
    generate_tier("dev", dev_phantoms, base_seed=5000)

    # ── Hidden tier (20 samples) ──────────────────────────────────────────
    print("\nGenerating hidden tier (20 samples, adversarial)...")
    hidden_phantoms = generate_phantoms_hidden(20)
    generate_tier("hidden", hidden_phantoms, base_seed=8000)

    # ── README ──────────────────────────────────────────────────────────────
    _write_top_readme()

    # ── Gallery images ──────────────────────────────────────────────────────
    print("\nGenerating gallery images...")
    generate_gallery_images()

    print(f"\n{'=' * 68}")
    print("Mammography benchmark dataset generation complete!")
    print(f"Output: {BENCHMARK_DIR}")
    print("=" * 68)


if __name__ == "__main__":
    main()
