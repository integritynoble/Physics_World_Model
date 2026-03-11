#!/usr/bin/env python3
"""Generate PET (Positron Emission Tomography) benchmark dataset.

Forward model (2D PET emission tomography):
    y_i ~ Poisson( eta_i * (a_i * [A * x]_i + s_i + r_i) )

where:
    x       : activity map (ground truth, 256x256), normalized to [0, 1]
    A       : system matrix (parallel-beam Radon transform, 180 angles)
    a_i     : attenuation correction factors  a = exp(-Radon(mu_map))
    s_i     : scatter contribution (smooth spatially-varying background)
    r_i     : random coincidence background (approximately uniform)
    eta_i   : detector efficiency variation (per-detector normalization)
    y_i     : measured sinogram (counts)

Geometry:
    180 angles over [0, 180) degrees
    367 detector bins (ceil(sqrt(2)*256))
    Parallel-beam geometry, 220 mm FOV

Mismatch parameters (ThetaSpace):
    scatter_fraction            : scattered photons fraction (0.20-0.55)
    attenuation_error           : relative error in mu-map (0-10%)
    randoms_fraction            : random coincidence fraction (0.05-0.50)
    detector_efficiency_variation : non-uniform detector response sigma (0-10%)

Phantoms (anatomically-realistic Zubal-like):
    Brain FDG   : gray matter (high uptake 4:1), white matter, deep structures
    Torso/Body  : liver, lungs, heart, kidneys, spine, hot/cold lesions
    Cardiac     : myocardial perfusion ring with sector defects

Tiers:
    Public  : 12 samples (4 brain + 4 body + 4 cardiac)
    Dev     : 20 samples (different orientations/lesion patterns, higher mismatch)
    Hidden  : 20 samples (completely different anatomies, extreme mismatch)

Usage:
    python3 generate_dataset.py
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy.ndimage import rotate as nd_rotate, gaussian_filter, zoom as nd_zoom

BENCHMARK_DIR = Path(__file__).resolve().parent

# -- Geometry -----------------------------------------------------------------

IMAGE_SIZE = 256
N_ANGLES = 180
N_DET = 367  # ceil(sqrt(2)*256) = 363, round up to odd for symmetry
FOV_MM = 220.0
PIXEL_SIZE_MM = FOV_MM / IMAGE_SIZE  # ~0.86 mm/px
PIXEL_SIZE_CM = PIXEL_SIZE_MM / 10.0  # ~0.086 cm/px

# -- Seed offsets per tier (project convention) --------------------------------

TIER_SEED_OFFSETS = {"public": 0, "dev": 10000, "hidden": 20000}

# -- Mismatch ranges per tier -------------------------------------------------

SPEC = {
    "public": {
        "scatter_fraction":              {"min": 0.20, "max": 0.35, "unit": ""},
        "attenuation_error":             {"min": 0.00, "max": 0.03, "unit": "relative"},
        "randoms_fraction":              {"min": 0.05, "max": 0.20, "unit": ""},
        "detector_efficiency_variation":  {"min": 0.00, "max": 0.03, "unit": "sigma"},
    },
    "dev": {
        "scatter_fraction":              {"min": 0.25, "max": 0.45, "unit": ""},
        "attenuation_error":             {"min": 0.00, "max": 0.06, "unit": "relative"},
        "randoms_fraction":              {"min": 0.10, "max": 0.35, "unit": ""},
        "detector_efficiency_variation":  {"min": 0.00, "max": 0.06, "unit": "sigma"},
    },
    "hidden": {
        "scatter_fraction":              {"min": 0.30, "max": 0.55, "unit": ""},
        "attenuation_error":             {"min": 0.00, "max": 0.10, "unit": "relative"},
        "randoms_fraction":              {"min": 0.10, "max": 0.50, "unit": ""},
        "detector_efficiency_variation":  {"min": 0.00, "max": 0.10, "unit": "sigma"},
    },
}

# Count rate ranges per tier (Mcps -- mega counts per second)
COUNT_RATE = {
    "public": (2.0, 5.0),
    "dev":    (1.0, 5.0),
    "hidden": (0.5, 5.0),
}


# =============================================================================
# Radon Transform (numpy + scipy, no external tomography library)
# =============================================================================

def radon_transform(image: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Parallel-beam Radon transform via scipy.ndimage.rotate.

    Args:
        image: (H, W) float64 input image.
        theta: 1-D array of projection angles in degrees.

    Returns:
        sinogram: (len(theta), N_det) float64, where N_det = diag of padded.
    """
    image = image.astype(np.float64)
    H, W = image.shape
    diag = int(np.ceil(np.sqrt(H**2 + W**2)))
    if diag % 2 != 0:
        diag += 1
    pad_h = (diag - H) // 2
    pad_w = (diag - W) // 2
    padded = np.zeros((diag, diag), dtype=np.float64)
    padded[pad_h : pad_h + H, pad_w : pad_w + W] = image

    sinogram = np.zeros((len(theta), diag), dtype=np.float64)
    for i, angle in enumerate(theta):
        rotated = nd_rotate(padded, angle, reshape=False, order=1,
                            mode="constant", cval=0.0)
        sinogram[i] = rotated.sum(axis=0)
    return sinogram


def _ramp_filter(n: int) -> np.ndarray:
    """Ram-Lak (ramp) filter with Hamming window for FBP."""
    freq = np.fft.fftfreq(n)
    filt = np.abs(freq) * 2.0
    fmax = np.abs(freq).max() + 1e-10
    hamming = 0.54 + 0.46 * np.cos(np.pi * freq / fmax)
    return filt * hamming


def fbp_reconstruct(sinogram: np.ndarray, theta: np.ndarray,
                     output_size: int = IMAGE_SIZE) -> np.ndarray:
    """Filtered Back-Projection (FBP) reconstruction.

    Args:
        sinogram: (N_angles, N_det) float64
        theta: projection angles in degrees
        output_size: spatial extent of output image

    Returns:
        recon: (output_size, output_size) float64
    """
    n_angles, n_det = sinogram.shape
    sinogram = sinogram.astype(np.float64)

    n_fft = max(64, int(2 ** np.ceil(np.log2(2 * n_det))))
    ramp = _ramp_filter(n_fft)

    filtered = np.zeros((n_angles, n_det), dtype=np.float64)
    for i in range(n_angles):
        proj = np.zeros(n_fft, dtype=np.float64)
        proj[:n_det] = sinogram[i]
        proj_fft = np.fft.fft(proj) * ramp
        filtered[i] = np.real(np.fft.ifft(proj_fft))[:n_det]

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
        proj = filtered[i]
        vals = np.where(valid,
                        (1 - w) * proj[np.clip(t0, 0, n_det - 1)]
                        + w * proj[np.clip(t1, 0, n_det - 1)],
                        0.0)
        recon += vals

    recon *= np.pi / (2 * n_angles)
    crop_start = (diag - output_size) // 2
    recon = recon[crop_start : crop_start + output_size,
                  crop_start : crop_start + output_size]
    return np.maximum(recon, 0.0)


# =============================================================================
# Ellipse primitive (normalized [-1,1] coordinate system)
# =============================================================================

def _ellipse_mask(H: int, W: int, cx: float, cy: float,
                  a: float, b: float, angle_deg: float = 0.0) -> np.ndarray:
    """Boolean mask for an axis-aligned or rotated ellipse.

    Coordinates in [-1,1] x [-1,1] covering the (H, W) grid.
    """
    yy = np.linspace(-1.0, 1.0, H)[:, None]
    xx = np.linspace(-1.0, 1.0, W)[None, :]
    ca, sa = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
    yr = (yy - cy) * ca - (xx - cx) * sa
    xr = (yy - cy) * sa + (xx - cx) * ca
    return (xr / a)**2 + (yr / b)**2 <= 1.0


def _rect_mask(H: int, W: int, cx: float, cy: float,
               hw: float, hh: float, angle_deg: float = 0.0) -> np.ndarray:
    """Boolean mask for a rotated rectangle in normalized coords."""
    yy = np.linspace(-1.0, 1.0, H)[:, None]
    xx = np.linspace(-1.0, 1.0, W)[None, :]
    ca, sa = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
    yr = (yy - cy) * ca - (xx - cx) * sa
    xr = (yy - cy) * sa + (xx - cx) * ca
    return (np.abs(xr) <= hw) & (np.abs(yr) <= hh)


# =============================================================================
# Anatomically-realistic PET phantom generators
# =============================================================================

def make_brain_fdg_phantom(
    H: int, W: int, seed: int, variant: int = 0
) -> tuple[np.ndarray, np.ndarray, str]:
    """Generate brain FDG-PET activity phantom + attenuation map.

    Anatomy:
        - Elliptical skull (bone, mu ~ 0.15 cm^-1)
        - Gray matter cortical shell (FDG uptake ~ 4:1 vs white)
        - White matter core (baseline activity)
        - Lateral ventricles (CSF, no activity)
        - Deep gray nuclei: caudate, putamen, thalamus (high uptake)
        - Variable hot lesions (tumors)

    Returns:
        activity: (H, W) float64 in [0, 1]
        mu_map:   (H, W) float64 (attenuation coefficients, cm^-1)
        name:     descriptive scene label
    """
    rng = np.random.default_rng(seed)
    activity = np.zeros((H, W), dtype=np.float64)
    mu_map = np.zeros((H, W), dtype=np.float64)

    # -- Skull outline (slight per-variant shape variation) ---
    a_skull = 0.42 + variant * 0.008 + rng.uniform(-0.005, 0.005)
    b_skull = 0.48 + variant * 0.004 + rng.uniform(-0.005, 0.005)
    skull_angle = rng.uniform(-3, 3)
    skull = _ellipse_mask(H, W, 0.0, 0.0, a_skull, b_skull, skull_angle)
    skull_inner = _ellipse_mask(H, W, 0.0, 0.0,
                                a_skull - 0.04, b_skull - 0.04, skull_angle)
    bone = skull & ~skull_inner
    brain = skull_inner.copy()

    mu_map[bone] = 0.15    # bone attenuation
    mu_map[brain] = 0.096  # soft tissue

    # -- White matter: baseline activity ~ 0.25 ---
    wm_base = 0.25 + rng.uniform(-0.03, 0.03)
    activity[brain] = wm_base

    # -- Gray matter cortex (high FDG uptake, ~4:1 ratio) ---
    gm_outer = _ellipse_mask(H, W, 0.0, 0.0,
                             a_skull - 0.05, b_skull - 0.05, skull_angle)
    gm_inner = _ellipse_mask(H, W, 0.0, 0.0,
                             a_skull - 0.10, b_skull - 0.10, skull_angle)
    cortex = gm_outer & ~gm_inner & brain
    gm_uptake = wm_base * (3.5 + rng.uniform(0, 1.0))  # 3.5-4.5:1 ratio
    activity[cortex] = gm_uptake

    # -- Cerebellum (posterior, slightly lower than cortex) ---
    cb_cy = 0.30 + rng.uniform(-0.02, 0.02)
    cb = _ellipse_mask(H, W, 0.0, cb_cy, 0.20, 0.08, 0) & brain
    activity[cb] = gm_uptake * 0.85

    # -- Deep gray nuclei ---
    nuclei = [
        # (cx, cy, a, b, angle, relative_uptake_to_gm)
        (-0.10, 0.05, 0.035, 0.075, -10, 1.0),   # left caudate
        (0.10, 0.05, 0.035, 0.075, 10, 1.0),      # right caudate
        (-0.16, -0.02, 0.045, 0.025, 0, 0.95),    # left putamen
        (0.16, -0.02, 0.045, 0.025, 0, 0.95),     # right putamen
        (-0.06, -0.06, 0.035, 0.025, 0, 0.85),    # left thalamus
        (0.06, -0.06, 0.035, 0.025, 0, 0.85),     # right thalamus
    ]
    for cx, cy, a, b, ang, rel in nuclei:
        cx += rng.uniform(-0.01, 0.01)
        cy += rng.uniform(-0.01, 0.01)
        mask = _ellipse_mask(H, W, cx, cy, a, b,
                             ang + variant * 2 + rng.uniform(-3, 3)) & brain
        activity[mask] = gm_uptake * rel + rng.uniform(-0.03, 0.03)

    # -- Lateral ventricles (CSF -- near-zero activity) ---
    vent_cy = 0.04 + variant * 0.008 + rng.uniform(-0.01, 0.01)
    vent_l = _ellipse_mask(H, W, -0.03, vent_cy, 0.018, 0.055,
                           -5 + rng.uniform(-3, 3)) & brain
    vent_r = _ellipse_mask(H, W, 0.03, vent_cy, 0.018, 0.055,
                           5 + rng.uniform(-3, 3)) & brain
    activity[vent_l] = 0.02
    activity[vent_r] = 0.02
    mu_map[vent_l] = 0.096   # CSF ~ water
    mu_map[vent_r] = 0.096

    # -- Third ventricle ---
    v3 = _ellipse_mask(H, W, 0.0, 0.04, 0.006, 0.025, 0) & brain
    activity[v3] = 0.02

    # -- Hot lesions (tumors) -- 1 to 3 per phantom ---
    n_tumors = rng.integers(1, 4)
    for _ in range(n_tumors):
        tx = rng.uniform(-0.25, 0.25)
        ty = rng.uniform(-0.30, 0.30)
        tr = rng.uniform(0.012, 0.04)
        aspect = rng.uniform(0.7, 1.3)
        tangle = rng.uniform(-45, 45)
        tumor = _ellipse_mask(H, W, tx, ty, tr, tr * aspect, tangle) & brain
        if tumor.sum() > 10:
            # Tumor uptake: 1.5-3x above gray matter
            activity[tumor] = gm_uptake * rng.uniform(1.5, 3.0)

    # -- Smooth and normalize ---
    activity = gaussian_filter(activity, sigma=1.0)
    activity = np.clip(activity, 0.0, None)
    if activity.max() > 0:
        activity /= activity.max()

    return activity, mu_map, f"brain_fdg_{variant:02d}"


def make_body_phantom(
    H: int, W: int, seed: int, variant: int = 0
) -> tuple[np.ndarray, np.ndarray, str]:
    """Generate torso PET phantom with anatomically-realistic organs and lesions.

    Anatomy:
        - Elliptical body contour (soft tissue)
        - Lungs (low attenuation, low activity)
        - Heart/myocardium (moderate-high FDG uptake)
        - Liver (moderate uptake, common metastasis site)
        - Kidneys (moderate-high uptake, FDG excretion)
        - Spine (bone, high attenuation, low activity)
        - Hot lesions in liver/lung + optional cold lesion

    Returns:
        activity, mu_map, name
    """
    rng = np.random.default_rng(seed)
    activity = np.zeros((H, W), dtype=np.float64)
    mu_map = np.zeros((H, W), dtype=np.float64)

    # -- Body contour ---
    body_a = 0.40 + rng.uniform(-0.02, 0.02)
    body_b = 0.32 + rng.uniform(-0.02, 0.02)
    body_angle = rng.uniform(-3, 3)
    body = _ellipse_mask(H, W, 0.0, 0.0, body_a, body_b, body_angle)
    mu_map[body] = 0.096  # soft tissue
    activity[body] = 0.15 + rng.uniform(-0.02, 0.02)  # background FDG

    # -- Spine (posterior) ---
    spine_cy = 0.22 + rng.uniform(-0.02, 0.02)
    spine = _ellipse_mask(H, W, 0.0, spine_cy, 0.04, 0.03, 0) & body
    mu_map[spine] = 0.15   # bone
    activity[spine] = 0.05  # low FDG in bone

    # -- Ribs (simplified as small ellipses) ---
    for rib_x in [-0.30, -0.25, 0.25, 0.30]:
        for rib_y in [-0.10, 0.05, 0.15]:
            rib = _ellipse_mask(H, W, rib_x, rib_y, 0.015, 0.008,
                                rng.uniform(-10, 10)) & body
            mu_map[rib] = 0.14
            activity[rib] = 0.04

    # -- Lungs ---
    lung_dy = rng.uniform(-0.02, 0.02)
    lung_l = _ellipse_mask(H, W, -0.18, -0.02 + lung_dy,
                           0.12 + variant * 0.005, 0.18, -5 + rng.uniform(-3, 3)) & body
    lung_r = _ellipse_mask(H, W, 0.18, -0.02 + lung_dy,
                           0.12 + variant * 0.005, 0.18, 5 + rng.uniform(-3, 3)) & body
    for lung in [lung_l, lung_r]:
        mu_map[lung] = 0.022  # inflated lung
        activity[lung] = 0.03  # very low FDG

    # -- Heart (myocardium ring) ---
    hcx = -0.05 + rng.uniform(-0.02, 0.02)
    hcy = -0.03 + rng.uniform(-0.02, 0.02)
    hang = 15 + variant * 5 + rng.uniform(-5, 5)
    heart_out = _ellipse_mask(H, W, hcx, hcy, 0.08, 0.07, hang) & body
    heart_in = _ellipse_mask(H, W, hcx, hcy, 0.05, 0.04, hang) & body
    myocardium = heart_out & ~heart_in
    activity[myocardium] = 0.70 + rng.uniform(-0.08, 0.08)
    activity[heart_in & body] = 0.20  # blood pool

    # -- Liver (right side, common metastasis site) ---
    liver_cx = 0.14 + rng.uniform(-0.02, 0.02)
    liver_cy = 0.08 + rng.uniform(-0.02, 0.02)
    liver = _ellipse_mask(H, W, liver_cx, liver_cy, 0.15, 0.10,
                          -10 + rng.uniform(-5, 5)) & body
    liver = liver & ~lung_r  # ensure no overlap with lung
    activity[liver] = 0.45 + rng.uniform(-0.05, 0.05)
    mu_map[liver] = 0.098  # liver slightly denser

    # -- Kidneys ---
    kid_l = _ellipse_mask(H, W, -0.15, 0.12 + rng.uniform(-0.02, 0.02),
                          0.04, 0.06, -10 + rng.uniform(-5, 5)) & body
    kid_r = _ellipse_mask(H, W, 0.15, 0.12 + rng.uniform(-0.02, 0.02),
                          0.04, 0.06, 10 + rng.uniform(-5, 5)) & body
    activity[kid_l] = 0.55 + rng.uniform(-0.05, 0.05)
    activity[kid_r] = 0.55 + rng.uniform(-0.05, 0.05)

    # -- Spleen (left upper quadrant) ---
    spleen = _ellipse_mask(H, W, -0.22, 0.05, 0.05, 0.04,
                           rng.uniform(-10, 10)) & body
    spleen = spleen & ~lung_l
    activity[spleen] = 0.40 + rng.uniform(-0.05, 0.05)

    # -- Hot lesions (tumors) -- 1-4 in liver/lung ---
    n_tumors = rng.integers(1, 5)
    for t_idx in range(n_tumors):
        # Alternate between lung and liver lesions
        if t_idx % 2 == 0:
            # Lung lesion
            tx = rng.choice([-1, 1]) * rng.uniform(0.10, 0.25)
            ty = rng.uniform(-0.15, 0.10)
        else:
            # Liver lesion
            tx = rng.uniform(0.05, 0.25)
            ty = rng.uniform(0.00, 0.15)
        tr = rng.uniform(0.010, 0.035)
        tumor = _ellipse_mask(H, W, tx, ty, tr, tr * rng.uniform(0.7, 1.3),
                              rng.uniform(-45, 45)) & body
        if tumor.sum() > 5:
            activity[tumor] = rng.uniform(1.5, 3.5)  # hot relative to background

    # -- Cold lesion (necrotic core / cyst, 50% chance) ---
    if rng.random() < 0.5:
        cx = rng.uniform(-0.20, 0.20)
        cy = rng.uniform(-0.15, 0.15)
        cr = rng.uniform(0.015, 0.035)
        cold = _ellipse_mask(H, W, cx, cy, cr, cr * 0.9, 0) & body
        activity[cold] = 0.02

    activity = gaussian_filter(activity, sigma=0.8)
    activity = np.clip(activity, 0.0, None)
    if activity.max() > 0:
        activity /= activity.max()

    return activity, mu_map, f"body_{variant:02d}"


def make_cardiac_phantom(
    H: int, W: int, seed: int, variant: int = 0
) -> tuple[np.ndarray, np.ndarray, str]:
    """Generate cardiac PET phantom (myocardial perfusion / viability).

    Anatomy:
        - Chest contour
        - Lungs (bilateral)
        - Prominent myocardial ring (high uptake)
        - Blood pool (moderate)
        - Perfusion defects in 0-2 angular sectors
        - Liver appearing at inferior portion of FOV
        - Spine

    Returns:
        activity, mu_map, name
    """
    rng = np.random.default_rng(seed)
    activity = np.zeros((H, W), dtype=np.float64)
    mu_map = np.zeros((H, W), dtype=np.float64)

    # -- Chest outline ---
    chest_a = 0.42 + rng.uniform(-0.02, 0.02)
    chest_b = 0.35 + rng.uniform(-0.02, 0.02)
    chest = _ellipse_mask(H, W, 0.0, 0.0, chest_a, chest_b, 0)
    mu_map[chest] = 0.096
    activity[chest] = 0.10 + rng.uniform(-0.02, 0.02)

    # -- Lungs ---
    for side, sx in [("L", -0.20), ("R", 0.20)]:
        lung = _ellipse_mask(H, W, sx, -0.02 + rng.uniform(-0.02, 0.02),
                             0.13, 0.20, rng.uniform(-5, 5)) & chest
        mu_map[lung] = 0.022
        activity[lung] = 0.03

    # -- Myocardium (prominent ring) ---
    heart_cx = -0.05 + variant * 0.01 + rng.uniform(-0.01, 0.01)
    heart_cy = -0.02 + rng.uniform(-0.01, 0.01)
    h_angle = 10 + rng.uniform(-5, 5)
    h_out_a = 0.12 + variant * 0.004 + rng.uniform(-0.005, 0.005)
    h_out_b = 0.11 + rng.uniform(-0.005, 0.005)
    h_in_a = 0.07 + variant * 0.002 + rng.uniform(-0.003, 0.003)
    h_in_b = 0.06 + rng.uniform(-0.003, 0.003)

    heart_outer = _ellipse_mask(H, W, heart_cx, heart_cy, h_out_a, h_out_b, h_angle)
    heart_inner = _ellipse_mask(H, W, heart_cx, heart_cy, h_in_a, h_in_b, h_angle)
    myocardium = (heart_outer & ~heart_inner) & chest
    blood_pool = heart_inner & chest

    base_myo = 0.90 + rng.uniform(-0.05, 0.05)
    activity[myocardium] = base_myo
    activity[blood_pool] = 0.25

    # -- Perfusion defects (0-2 sectors with reduced uptake) ---
    n_defects = rng.integers(0, 3)
    if n_defects > 0 and myocardium.sum() > 0:
        y_coords, x_coords = np.where(myocardium)
        ctr_y = int((heart_cy + 1.0) * H / 2)
        ctr_x = int((heart_cx + 1.0) * W / 2)
        angles = np.degrees(np.arctan2(y_coords - ctr_y, x_coords - ctr_x)) % 360
        for _ in range(n_defects):
            angle_start = rng.uniform(0, 360)
            angle_span = rng.uniform(30, 90)
            angle_end = (angle_start + angle_span) % 360
            if angle_end > angle_start:
                in_sector = (angles >= angle_start) & (angles < angle_end)
            else:
                in_sector = (angles >= angle_start) | (angles < angle_end)
            if in_sector.sum() > 0:
                defect_uptake = rng.uniform(0.25, 0.60)
                activity[y_coords[in_sector], x_coords[in_sector]] = defect_uptake

    # -- Liver (inferior) ---
    liver = _ellipse_mask(H, W, 0.10, 0.20 + rng.uniform(-0.02, 0.02),
                          0.20, 0.08, -5 + rng.uniform(-5, 5)) & chest
    activity[liver] = 0.50 + rng.uniform(-0.05, 0.05)

    # -- Spine (posterior) ---
    spine = _ellipse_mask(H, W, 0.0, 0.28, 0.04, 0.03, 0) & chest
    mu_map[spine] = 0.15
    activity[spine] = 0.05

    # -- Sternum (anterior) ---
    stern = _ellipse_mask(H, W, 0.0, -0.28, 0.02, 0.015, 0) & chest
    mu_map[stern] = 0.14
    activity[stern] = 0.04

    activity = gaussian_filter(activity, sigma=0.8)
    activity = np.clip(activity, 0.0, None)
    if activity.max() > 0:
        activity /= activity.max()

    return activity, mu_map, f"cardiac_{variant:02d}"


# =============================================================================
# Phantom pool generators per tier
# =============================================================================

PHANTOM_GENERATORS = [make_brain_fdg_phantom, make_body_phantom, make_cardiac_phantom]


def generate_phantoms_public(n: int = 12) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """12 public phantoms: 4 brain + 4 body + 4 cardiac (diverse seeds)."""
    phantoms: list[tuple[np.ndarray, np.ndarray, str]] = []
    for i in range(4):
        phantoms.append(make_brain_fdg_phantom(IMAGE_SIZE, IMAGE_SIZE,
                                               seed=100 + i, variant=i))
    for i in range(4):
        phantoms.append(make_body_phantom(IMAGE_SIZE, IMAGE_SIZE,
                                          seed=200 + i, variant=i))
    for i in range(4):
        phantoms.append(make_cardiac_phantom(IMAGE_SIZE, IMAGE_SIZE,
                                             seed=300 + i, variant=i))
    return phantoms[:n]


def generate_phantoms_dev(n: int = 20) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """20 dev phantoms: rotated/flipped/zoomed variants with different seeds."""
    phantoms: list[tuple[np.ndarray, np.ndarray, str]] = []
    rng = np.random.default_rng(5000)
    for i in range(n):
        gen_fn = PHANTOM_GENERATORS[i % 3]
        activity, mu_map, name = gen_fn(IMAGE_SIZE, IMAGE_SIZE,
                                        seed=10500 + i, variant=i)
        # Augment: rotation + flip + mild zoom
        angle = float(rng.uniform(15, 345))
        activity = nd_rotate(activity, angle, reshape=False,
                             mode="constant", cval=0.0)
        mu_map = nd_rotate(mu_map, angle, reshape=False,
                           mode="constant", cval=0.0)
        if rng.random() < 0.5:
            activity = np.fliplr(activity)
            mu_map = np.fliplr(mu_map)
        zoom_f = float(rng.uniform(0.85, 1.15))
        if abs(zoom_f - 1.0) > 0.01:
            activity = _zoom_crop(activity, zoom_f, IMAGE_SIZE)
            mu_map = _zoom_crop(mu_map, zoom_f, IMAGE_SIZE)
        activity = np.clip(activity, 0.0, None)
        if activity.max() > 0:
            activity /= activity.max()
        phantoms.append((activity, mu_map, f"dev_{name}"))
    return phantoms


def generate_phantoms_hidden(n: int = 20) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """20 hidden phantoms: adversarial augmentations + micro-lesions."""
    phantoms: list[tuple[np.ndarray, np.ndarray, str]] = []
    rng = np.random.default_rng(8000)
    for i in range(n):
        gen_fn = PHANTOM_GENERATORS[i % 3]
        activity, mu_map, name = gen_fn(IMAGE_SIZE, IMAGE_SIZE,
                                        seed=20800 + i, variant=i + 10)
        # Aggressive augmentation
        angle = float(rng.uniform(20, 340))
        activity = nd_rotate(activity, angle, reshape=False,
                             mode="constant", cval=0.0)
        mu_map = nd_rotate(mu_map, angle, reshape=False,
                           mode="constant", cval=0.0)
        if rng.random() < 0.7:
            activity = np.fliplr(activity)
            mu_map = np.fliplr(mu_map)
        if rng.random() < 0.5:
            activity = np.flipud(activity)
            mu_map = np.flipud(mu_map)

        # Aggressive zoom
        zoom_f = float(rng.uniform(0.75, 1.25))
        activity = _zoom_crop(activity, zoom_f, IMAGE_SIZE)
        mu_map = _zoom_crop(mu_map, zoom_f, IMAGE_SIZE)

        # Subtle micro-lesions (hard to detect)
        n_micro = rng.integers(2, 6)
        for _ in range(n_micro):
            cy = rng.integers(40, IMAGE_SIZE - 40)
            cx = rng.integers(40, IMAGE_SIZE - 40)
            r = rng.integers(2, 6)
            yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
            circle = (yy**2 + xx**2 <= r**2).astype(np.float64)
            y0 = max(0, cy - r)
            y1 = min(IMAGE_SIZE, cy + r + 1)
            x0 = max(0, cx - r)
            x1 = min(IMAGE_SIZE, cx + r + 1)
            c_y0 = r - (cy - y0)
            c_y1 = r + (y1 - cy)
            c_x0 = r - (cx - x0)
            c_x1 = r + (x1 - cx)
            patch = activity[y0:y1, x0:x1]
            if patch.size > 0 and patch.mean() > 0.1:
                intensity = rng.uniform(1.5, 4.0)
                activity[y0:y1, x0:x1] = np.maximum(
                    patch, circle[c_y0:c_y1, c_x0:c_x1] * intensity
                )

        activity = np.clip(activity, 0.0, None)
        if activity.max() > 0:
            activity /= activity.max()
        phantoms.append((activity, mu_map, f"hidden_{name}"))
    return phantoms


def _zoom_crop(arr: np.ndarray, zoom_f: float, size: int) -> np.ndarray:
    """Zoom an image and crop/pad to target size."""
    zoomed = nd_zoom(arr, zoom_f, order=1)
    H, W = zoomed.shape
    if H >= size and W >= size:
        y0 = (H - size) // 2
        x0 = (W - size) // 2
        return zoomed[y0 : y0 + size, x0 : x0 + size]
    out = np.zeros((size, size), dtype=arr.dtype)
    y0 = (size - H) // 2
    x0 = (size - W) // 2
    out[y0 : y0 + H, x0 : x0 + W] = zoomed
    return out


# =============================================================================
# PET Forward Model
# =============================================================================

def pet_forward_model(
    activity: np.ndarray,
    mu_map: np.ndarray,
    theta_deg: np.ndarray,
    count_rate_mcps: float,
    scatter_fraction: float,
    randoms_fraction: float,
    attenuation_error: float,
    detector_efficiency_variation: float,
    rng: np.random.Generator,
) -> dict:
    """Full PET forward model with Poisson statistics.

    y_i ~ Poisson( eta_i * (a_i * [A * x]_i + s_i + r_i) )

    Args:
        activity:                    (H, W) ground-truth activity [0, 1]
        mu_map:                      (H, W) attenuation map [cm^-1]
        theta_deg:                   projection angles (degrees)
        count_rate_mcps:             total count rate (Mega counts per second)
        scatter_fraction:            s / total
        randoms_fraction:            r / total
        attenuation_error:           relative error in mu-map
        detector_efficiency_variation: sigma of per-detector efficiency noise
        rng:                         random generator

    Returns:
        dict with sinogram_ideal, sinogram_measured, attenuation_factors, etc.
    """
    # 1. Ideal sinogram (Radon of activity)
    sino_ideal = radon_transform(activity, theta_deg)
    sino_ideal = np.maximum(sino_ideal, 0.0)
    n_det = sino_ideal.shape[1]

    # 2. Attenuation factors: a = exp(-Radon(mu) * pixel_size_cm)
    sino_mu = radon_transform(mu_map, theta_deg)
    sino_mu_physical = sino_mu * PIXEL_SIZE_CM
    attn_factors_true = np.exp(-sino_mu_physical)

    # Attenuation map with error (for imperfect correction)
    if attenuation_error > 0:
        mu_err_map = mu_map * (1.0 + rng.uniform(
            -attenuation_error, attenuation_error, mu_map.shape))
    else:
        mu_err_map = mu_map.copy()
    sino_mu_err = radon_transform(mu_err_map, theta_deg) * PIXEL_SIZE_CM
    attn_factors_used = np.exp(-sino_mu_err)

    # 3. Scale to physical count level
    total_expected_trues = count_rate_mcps * 1e6
    sino_sum = sino_ideal.sum()
    scale = total_expected_trues / sino_sum if sino_sum > 0 else 1.0
    sino_trues = sino_ideal * scale

    # 4. Attenuated trues
    sino_atten = sino_trues * attn_factors_true

    # 5. Scatter: smooth spatially-varying background
    mean_signal = max(sino_atten.mean(), 1.0)
    denom = max(1.0 - scatter_fraction - randoms_fraction, 0.1)
    scatter_level = scatter_fraction / denom * mean_signal
    scatter = np.full_like(sino_atten, scatter_level)
    scatter_noise = rng.standard_normal(sino_atten.shape) * scatter_level * 0.10
    scatter += gaussian_filter(scatter_noise, sigma=[5.0, 10.0])
    scatter = np.maximum(scatter, 0.0)

    # 6. Randoms: approximately uniform background
    randoms_level = randoms_fraction / denom * mean_signal
    randoms = np.full_like(sino_atten, randoms_level)
    randoms += rng.standard_normal(sino_atten.shape) * randoms_level * 0.05
    randoms = np.maximum(randoms, 0.0)

    # 7. Detector efficiency variation (per-detector normalization)
    # eta_i = 1 + N(0, sigma) per detector bin, constant across angles
    if detector_efficiency_variation > 0:
        eta = 1.0 + rng.normal(0, detector_efficiency_variation, size=n_det)
        eta = np.clip(eta, 0.5, 1.5)  # physical limit
        eta_2d = eta[np.newaxis, :]  # broadcast over angles
    else:
        eta = np.ones(n_det)
        eta_2d = 1.0

    # 8. Expected counts before Poisson sampling
    expected = eta_2d * (sino_atten + scatter + randoms)
    expected = np.maximum(expected, 0.01)

    # 9. Poisson sampling
    sino_measured = rng.poisson(expected).astype(np.float64)

    return {
        "sinogram_ideal": sino_ideal.astype(np.float32),
        "sinogram_measured": sino_measured.astype(np.float32),
        "attenuation_factors": attn_factors_true.astype(np.float32),
        "attenuation_factors_used": attn_factors_used.astype(np.float32),
        "scatter": scatter.astype(np.float32),
        "randoms": randoms.astype(np.float32),
        "detector_efficiency": eta.astype(np.float32),
        "expected_counts": expected.astype(np.float32),
        "scale_factor": float(scale),
    }


# =============================================================================
# Metrics
# =============================================================================

def compute_psnr(gt: np.ndarray, recon: np.ndarray) -> float:
    mse = np.mean((gt.astype(np.float64) - recon.astype(np.float64)) ** 2)
    if mse < 1e-12:
        return 100.0
    data_range = float(gt.max() - gt.min())
    if data_range < 1e-12:
        return 0.0
    return float(10 * np.log10(data_range**2 / mse))


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
    ssim = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / \
           ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))
    return float(ssim)


# =============================================================================
# Image helpers
# =============================================================================

def _norm(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo + 1e-8)


def _save_png(arr: np.ndarray, path: Path, percentile_clip: bool = False) -> None:
    if percentile_clip and arr.max() > 0:
        lo, hi = np.percentile(arr[arr > 0], [1, 99])
        arr = np.clip(arr, lo, hi)
    Image.fromarray(
        np.clip(_norm(arr) * 255, 0, 255).astype(np.uint8), "L"
    ).save(str(path))


def _save_overview(x_true, sino_ideal, sino_meas, recon_fbp, path: Path) -> None:
    """4-panel overview: GT | ideal sinogram | measured sinogram | FBP recon."""
    th, tw = 128, 128

    def _r(a):
        pil = Image.fromarray(
            np.clip(_norm(a) * 255, 0, 255).astype(np.uint8), "L"
        )
        return np.array(pil.resize((tw, th), Image.LANCZOS)) / 255.0

    ov = np.zeros((th, 4 * tw), dtype=np.float32)
    ov[:, 0:tw] = _r(x_true)
    ov[:, tw : 2 * tw] = _r(sino_ideal)
    ov[:, 2 * tw : 3 * tw] = _r(sino_meas)
    ov[:, 3 * tw : 4 * tw] = _r(recon_fbp)
    _save_png(ov, path)


# =============================================================================
# Tier generation
# =============================================================================

def sample_mismatch(rng: np.random.Generator, spec: dict) -> dict:
    return {k: float(rng.uniform(v["min"], v["max"])) for k, v in spec.items()}


def generate_tier(
    tier: str,
    phantoms: list[tuple[np.ndarray, np.ndarray, str]],
    base_seed: int,
) -> None:
    """Generate one tier of the PET benchmark dataset."""
    spec_ranges = SPEC[tier]
    tier_dir = BENCHMARK_DIR / tier
    images_dir = tier_dir / "images"
    if tier_dir.exists():
        shutil.rmtree(tier_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    theta_deg = np.linspace(0, 180, N_ANGLES, endpoint=False).astype(np.float64)

    h5_path = tier_dir / f"pet_challenge_{tier}.h5"
    rng = np.random.default_rng(base_seed)
    true_specs: dict = {}
    all_psnrs: list[float] = []
    all_ssims: list[float] = []

    cr_lo, cr_hi = COUNT_RATE[tier]

    with h5py.File(h5_path, "w") as f:
        f.attrs["description"] = (
            f"PWM PET benchmark -- {tier} tier "
            f"(parallel-beam Radon + Poisson noise + attenuation "
            f"+ scatter + randoms + detector efficiency)"
        )
        f.attrs["spec_ranges"] = json.dumps(spec_ranges)
        f.attrs["geometry"] = json.dumps({
            "image_size": IMAGE_SIZE,
            "n_angles": N_ANGLES,
            "n_det": N_DET,
            "angle_range_deg": [0, 180],
            "fov_mm": FOV_MM,
            "pixel_size_mm": PIXEL_SIZE_MM,
        })
        f.attrs["forward_model"] = (
            "y_i ~ Poisson( eta_i * (a_i * [A * x]_i + s_i + r_i) )"
        )

        for idx, (activity, mu_map, scene_name) in enumerate(phantoms):
            key = f"sample_{idx:02d}"
            print(f"  [{tier}] Generating {key} ({scene_name})...", end="", flush=True)

            # Sample mismatch parameters
            mis = sample_mismatch(rng, spec_ranges)
            # Also sample count rate
            count_rate = float(rng.uniform(cr_lo, cr_hi))
            mis_full = {**mis, "count_rate_mcps": count_rate}
            true_specs[key] = mis_full

            # Forward model
            result = pet_forward_model(
                activity, mu_map, theta_deg,
                count_rate_mcps=count_rate,
                scatter_fraction=mis["scatter_fraction"],
                randoms_fraction=mis["randoms_fraction"],
                attenuation_error=mis["attenuation_error"],
                detector_efficiency_variation=mis["detector_efficiency_variation"],
                rng=rng,
            )

            sino_ideal = result["sinogram_ideal"]
            sino_measured = result["sinogram_measured"]
            n_det = sino_ideal.shape[1]

            # FBP reconstruction from measured sinogram
            attn_corr = result["attenuation_factors_used"]
            attn_corr_safe = np.where(attn_corr > 0.01, attn_corr, 0.01)

            # Correct: (measured - randoms - scatter) / attenuation / eta
            det_eff = result["detector_efficiency"][np.newaxis, :]
            det_eff_safe = np.where(det_eff > 0.5, det_eff, 1.0)
            sino_corrected = sino_measured / det_eff_safe
            sino_corrected = (sino_corrected - result["randoms"]
                              - result["scatter"])
            sino_corrected = np.maximum(sino_corrected, 0.0) / attn_corr_safe

            # Rescale back to activity units
            if result["scale_factor"] > 0:
                sino_corrected /= result["scale_factor"]

            recon_fbp = fbp_reconstruct(sino_corrected, theta_deg, IMAGE_SIZE)
            recon_fbp = np.maximum(recon_fbp, 0.0).astype(np.float32)

            psnr = compute_psnr(activity, recon_fbp)
            ssim = compute_ssim(activity, recon_fbp)
            all_psnrs.append(psnr)
            all_ssims.append(ssim)

            # Save to HDF5
            grp = f.create_group(key)
            grp.create_dataset("x_true", data=activity.astype(np.float32),
                               compression="gzip")
            grp.create_dataset("sinogram_ideal", data=sino_ideal,
                               compression="gzip")
            grp.create_dataset("sinogram_measured", data=sino_measured,
                               compression="gzip")
            grp.create_dataset("attenuation_map",
                               data=mu_map.astype(np.float32),
                               compression="gzip")
            grp.create_dataset("angles_deg",
                               data=theta_deg.astype(np.float32))
            grp.create_dataset("reconstruction_fbp", data=recon_fbp,
                               compression="gzip")
            grp.attrs["metadata"] = json.dumps({
                "scene": scene_name,
                "shape": list(activity.shape),
                "n_angles": N_ANGLES,
                "n_det": int(n_det),
                "psnr_fbp": float(psnr),
                "ssim_fbp": float(ssim),
            })
            grp.attrs["true_spec"] = json.dumps(mis_full)
            grp.attrs["spec_ranges"] = json.dumps(spec_ranges)

            # Save per-sample images
            sample_dir = images_dir / f"sample_{idx:02d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            _save_png(activity, sample_dir / "ground_truth.png")
            _save_png(sino_ideal, sample_dir / "sinogram_ideal.png")
            _save_png(sino_measured, sample_dir / "measurement.png")
            _save_png(recon_fbp, sample_dir / "reconstruction_fbp.png")
            _save_overview(activity, sino_ideal, sino_measured, recon_fbp,
                           sample_dir / "overview.png")
            with open(sample_dir / "spec.json", "w") as sf:
                json.dump({
                    "scene": scene_name,
                    "spec_ranges": spec_ranges,
                    "true_spec": mis_full,
                    "psnr_fbp": psnr,
                    "ssim_fbp": ssim,
                }, sf, indent=2)

            print(f"  PSNR={psnr:.2f} dB, SSIM={ssim:.3f}  "
                  f"counts={count_rate:.1f} Mcps  "
                  f"scatter={mis['scatter_fraction']:.2f}  "
                  f"randoms={mis['randoms_fraction']:.2f}  "
                  f"attn_err={mis['attenuation_error']:.3f}  "
                  f"det_var={mis['detector_efficiency_variation']:.3f}")

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


# =============================================================================
# Gallery image generation
# =============================================================================

def generate_gallery_images() -> None:
    """Generate gallery images for the platform benchmark page.

    Creates scene_00 through scene_03 with:
      gt.png, measurement_I.png, measurement_II.png,
      recon_I.png, recon_II.png, recon_III.png
    """
    gallery_base = (BENCHMARK_DIR.parent.parent.parent /
                    "platform" / "pwm_platform" / "static" / "img" /
                    "benchmark_gallery" / "pet")

    h5_path = BENCHMARK_DIR / "public" / "pet_challenge_public.h5"
    if not h5_path.exists():
        print("  [gallery] Public HDF5 not found, skipping gallery generation.")
        return

    # Pick diverse samples: brain(0), body(4), cardiac(8), brain variant(3)
    gallery_sample_indices = [0, 4, 8, 3]

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
            # recon_III.png -- |GT - FBP| difference
            diff = np.abs(x_true - recon_fbp)
            _save_png(diff, scene_dir / "recon_III.png")

            print(f"  [gallery] scene_{scene_idx:02d} saved -> {scene_dir}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    t0 = time.time()
    print("PET Benchmark Dataset Generator")
    print("=" * 68)
    print(f"Output : {BENCHMARK_DIR}")
    print(f"Geometry: {N_ANGLES} angles, {IMAGE_SIZE}x{IMAGE_SIZE} images, "
          f"FOV={FOV_MM:.0f} mm\n")

    # -- Public tier (12 samples) --
    print("Generating public tier (12 samples)...")
    public_phantoms = generate_phantoms_public(12)
    generate_tier("public", public_phantoms, base_seed=1000)

    # -- Dev tier (20 samples) --
    print("\nGenerating dev tier (20 samples)...")
    dev_phantoms = generate_phantoms_dev(20)
    generate_tier("dev", dev_phantoms, base_seed=5000)

    # -- Hidden tier (20 samples) --
    print("\nGenerating hidden tier (20 samples)...")
    hidden_phantoms = generate_phantoms_hidden(20)
    generate_tier("hidden", hidden_phantoms, base_seed=8000)

    # -- Gallery images --
    print("\nGenerating gallery images...")
    generate_gallery_images()

    elapsed = time.time() - t0
    print(f"\n{'=' * 68}")
    print(f"PET benchmark dataset generation complete! ({elapsed:.1f}s)")
    print(f"Output: {BENCHMARK_DIR}")
    print("=" * 68)


if __name__ == "__main__":
    main()
