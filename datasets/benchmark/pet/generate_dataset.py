#!/usr/bin/env python3
"""Generate PET (Positron Emission Tomography) benchmark dataset.

Forward model (2D PET):
    y_i ~ Poisson(a_i * [A * x]_i + r_i + s_i)

where:
    x       : activity map (ground truth, 256x256)
    A       : system matrix (parallel-beam Radon transform)
    a_i     : attenuation correction factors (from mu-map)
    r_i     : random coincidences (uniform background)
    s_i     : scatter contribution (smooth background)
    y_i     : measured sinogram (counts)

Geometry:
    256 angles over [0, pi)
    367 detector bins (default for 256x256 Radon)
    Parallel-beam geometry

Mismatch parameters:
    count_rate_mcps    : total count rate in Mcps (controls noise level)
    scatter_fraction   : scatter / total (0.30 - 0.55)
    randoms_fraction   : randoms / total (0.10 - 0.50)
    attenuation_error  : relative error in mu-map (0 - 10%)

Phantoms:
    Public  : 12 samples (brain, body, cardiac with diverse lesions)
    Dev     : 20 samples (augmented variants with shifted/rotated anatomy)
    Hidden  : 20 samples (adversarial: subtle lesions, extreme noise, attn errors)

Usage:
    python3 generate_dataset.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy.ndimage import rotate as nd_rotate, gaussian_filter, zoom as nd_zoom

BENCHMARK_DIR = Path(__file__).resolve().parent

# ── Geometry ──────────────────────────────────────────────────────────────────

IMAGE_SIZE = 256
N_ANGLES = 256
N_DET = 367  # default for 256x256 Radon transform (ceil(sqrt(2)*256))

# ── Mismatch ranges per tier ────────────────────────────────────────────────

SPEC = {
    "public": {
        "count_rate_mcps":    {"min": 2.0,  "max": 5.0,  "unit": "Mcps"},
        "scatter_fraction":   {"min": 0.30, "max": 0.40, "unit": ""},
        "randoms_fraction":   {"min": 0.10, "max": 0.25, "unit": ""},
        "attenuation_error":  {"min": 0.0,  "max": 0.03, "unit": "relative"},
    },
    "dev": {
        "count_rate_mcps":    {"min": 1.0,  "max": 5.0,  "unit": "Mcps"},
        "scatter_fraction":   {"min": 0.30, "max": 0.45, "unit": ""},
        "randoms_fraction":   {"min": 0.10, "max": 0.35, "unit": ""},
        "attenuation_error":  {"min": 0.0,  "max": 0.06, "unit": "relative"},
    },
    "hidden": {
        "count_rate_mcps":    {"min": 0.5,  "max": 5.0,  "unit": "Mcps"},
        "scatter_fraction":   {"min": 0.30, "max": 0.55, "unit": ""},
        "randoms_fraction":   {"min": 0.10, "max": 0.50, "unit": ""},
        "attenuation_error":  {"min": 0.0,  "max": 0.10, "unit": "relative"},
    },
}


# ── Radon Transform (numpy + scipy only) ────────────────────────────────────

def radon_transform(image: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Parallel-beam Radon transform using scipy.ndimage.rotate.

    Args:
        image: (H, W) float64 input image
        theta: array of projection angles in degrees

    Returns:
        sinogram: (len(theta), N_det) float64
    """
    image = image.astype(np.float64)
    H, W = image.shape
    # Pad to avoid clipping during rotation
    diag = int(np.ceil(np.sqrt(H**2 + W**2)))
    if diag % 2 != 0:
        diag += 1
    pad_h = (diag - H) // 2
    pad_w = (diag - W) // 2
    padded = np.zeros((diag, diag), dtype=np.float64)
    padded[pad_h:pad_h + H, pad_w:pad_w + W] = image

    sinogram = np.zeros((len(theta), diag), dtype=np.float64)
    for i, angle in enumerate(theta):
        rotated = nd_rotate(padded, angle, reshape=False, order=1,
                            mode='constant', cval=0.0)
        sinogram[i] = rotated.sum(axis=0)

    return sinogram


def _ramp_filter(n: int) -> np.ndarray:
    """Ram-Lak (ramp) filter in frequency domain for FBP."""
    freq = np.fft.fftfreq(n)
    filt = np.abs(freq) * 2.0  # Ram-Lak: |omega|
    # Apply Hamming window to reduce ringing
    hamming = 0.54 + 0.46 * np.cos(np.pi * freq / (np.abs(freq).max() + 1e-10))
    filt *= hamming
    return filt


def fbp_reconstruct(sinogram: np.ndarray, theta: np.ndarray,
                     output_size: int = IMAGE_SIZE) -> np.ndarray:
    """Filtered Back-Projection (FBP) reconstruction.

    Args:
        sinogram: (N_angles, N_det) float64
        theta: projection angles in degrees
        output_size: output image size

    Returns:
        recon: (output_size, output_size) float64
    """
    n_angles, n_det = sinogram.shape
    sinogram = sinogram.astype(np.float64)

    # Apply ramp filter to each projection
    # Pad to next power of 2 for FFT efficiency
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
        # Project grid onto detector
        t = x_grid * np.cos(angle_rad) + y_grid * np.sin(angle_rad) + det_center
        # Bilinear interpolation
        t0 = np.floor(t).astype(int)
        t1 = t0 + 1
        w = t - t0
        valid = (t0 >= 0) & (t1 < n_det)
        proj = filtered[i]
        vals = np.where(valid, (1 - w) * proj[np.clip(t0, 0, n_det - 1)] +
                        w * proj[np.clip(t1, 0, n_det - 1)], 0.0)
        recon += vals

    recon *= np.pi / (2 * n_angles)

    # Crop to output size
    crop_start = (diag - output_size) // 2
    recon = recon[crop_start:crop_start + output_size,
                  crop_start:crop_start + output_size]

    return np.maximum(recon, 0.0)


# ── Phantom Generators ───────────────────────────────────────────────────────

def _ellipse_mask(H: int, W: int, cx: float, cy: float,
                  a: float, b: float, angle_deg: float) -> np.ndarray:
    """Generate a binary ellipse mask (coordinates in [0, H) x [0, W)).
    Returns boolean array."""
    yy = np.linspace(-1.0, 1.0, H)[:, None]
    xx = np.linspace(-1.0, 1.0, W)[None, :]
    ca, sa = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
    yr = (yy - cy) * ca - (xx - cx) * sa
    xr = (yy - cy) * sa + (xx - cx) * ca
    return (xr / a)**2 + (yr / b)**2 <= 1.0


def make_brain_fdg_phantom(
    H: int, W: int, seed: int, variant: int = 0
) -> tuple[np.ndarray, np.ndarray, str]:
    """Generate a brain FDG-PET-like activity phantom + attenuation map.

    Returns:
        activity: (H, W) float64  [0, ~1]
        mu_map:   (H, W) float64  attenuation coefficients [cm^-1]
        name:     scene name
    """
    rng = np.random.default_rng(seed)

    activity = np.zeros((H, W), dtype=np.float64)
    mu_map = np.zeros((H, W), dtype=np.float64)

    # Skull outline
    skull = _ellipse_mask(H, W, 0.0, 0.0, 0.42 + variant * 0.01, 0.48 + variant * 0.005, 0)
    skull_inner = _ellipse_mask(H, W, 0.0, 0.0, 0.38 + variant * 0.01, 0.44 + variant * 0.005, 0)
    bone = skull & ~skull_inner

    # Brain tissue
    brain = skull_inner.copy()
    mu_map[bone.astype(bool)] = 0.15   # bone attenuation ~ 0.15 cm^-1
    mu_map[brain.astype(bool)] = 0.096  # soft tissue attenuation ~ 0.096 cm^-1

    # White matter: baseline activity ~ 0.25
    activity[brain.astype(bool)] = 0.25 + rng.uniform(-0.02, 0.02)

    # Gray matter (cortex): higher uptake ~ 1.0 (4x white matter)
    cortex_outer = _ellipse_mask(H, W, 0.0, 0.0, 0.37, 0.43, 0)
    cortex_inner = _ellipse_mask(H, W, 0.0, 0.0, 0.32, 0.38, 0)
    cortex = cortex_outer & ~cortex_inner & brain
    activity[cortex.astype(bool)] = 0.90 + rng.uniform(-0.05, 0.05)

    # Deep gray matter structures (caudate, putamen, thalamus)
    structures = [
        # (cx, cy, a, b, angle, uptake)
        (-0.10, 0.05, 0.04, 0.08, -10, 1.0),   # left caudate
        (0.10, 0.05, 0.04, 0.08, 10, 1.0),     # right caudate
        (-0.15, -0.02, 0.05, 0.03, 0, 0.95),   # left putamen
        (0.15, -0.02, 0.05, 0.03, 0, 0.95),    # right putamen
        (-0.06, -0.05, 0.04, 0.03, 0, 0.85),   # left thalamus
        (0.06, -0.05, 0.04, 0.03, 0, 0.85),    # right thalamus
    ]
    for cx, cy, a, b, ang, uptake in structures:
        # Small random offsets for variety
        cx += rng.uniform(-0.01, 0.01)
        cy += rng.uniform(-0.01, 0.01)
        mask = _ellipse_mask(H, W, cx, cy, a, b, ang + variant * 3) & brain
        activity[mask.astype(bool)] = uptake + rng.uniform(-0.05, 0.05)

    # Ventricles (CSF): cold regions
    vent_l = _ellipse_mask(H, W, -0.03, 0.05 + variant * 0.01, 0.02, 0.06, -5) & brain
    vent_r = _ellipse_mask(H, W, 0.03, 0.05 + variant * 0.01, 0.02, 0.06, 5) & brain
    activity[vent_l.astype(bool)] = 0.05
    activity[vent_r.astype(bool)] = 0.05

    # Hot spots (tumors) — 1-3 per phantom
    n_tumors = rng.integers(1, 4)
    for _ in range(n_tumors):
        tx = rng.uniform(-0.25, 0.25)
        ty = rng.uniform(-0.30, 0.30)
        tr = rng.uniform(0.015, 0.04)
        tumor = _ellipse_mask(H, W, tx, ty, tr, tr * rng.uniform(0.7, 1.3), rng.uniform(-30, 30)) & brain
        if tumor.sum() > 10:
            activity[tumor.astype(bool)] = rng.uniform(1.5, 3.0)  # hot lesion

    # Apply mild smoothing
    activity = gaussian_filter(activity, sigma=1.0)

    # Ensure activity is non-negative and normalized
    activity = np.clip(activity, 0.0, None)
    if activity.max() > 0:
        activity /= activity.max()

    return activity, mu_map, f"brain_fdg_{variant:02d}"


def make_body_phantom(
    H: int, W: int, seed: int, variant: int = 0
) -> tuple[np.ndarray, np.ndarray, str]:
    """Generate a body (torso) PET phantom with organs and lesions.

    Returns:
        activity: (H, W) float64
        mu_map:   (H, W) float64
        name:     scene name
    """
    rng = np.random.default_rng(seed)

    activity = np.zeros((H, W), dtype=np.float64)
    mu_map = np.zeros((H, W), dtype=np.float64)

    # Body outline (elliptical)
    body = _ellipse_mask(H, W, 0.0, 0.0, 0.40, 0.32, 0)
    mu_map[body.astype(bool)] = 0.096  # soft tissue
    activity[body.astype(bool)] = 0.15  # background uptake

    # Spine (posterior, high attenuation)
    spine = _ellipse_mask(H, W, 0.0, 0.22, 0.04, 0.03, 0) & body
    mu_map[spine.astype(bool)] = 0.15  # bone
    activity[spine.astype(bool)] = 0.05

    # Lungs (low attenuation, low uptake)
    lung_l = _ellipse_mask(H, W, -0.18, -0.02 + variant * 0.01, 0.12, 0.18, -5) & body
    lung_r = _ellipse_mask(H, W, 0.18, -0.02 + variant * 0.01, 0.12, 0.18, 5) & body
    mu_map[lung_l.astype(bool)] = 0.022  # inflated lung
    mu_map[lung_r.astype(bool)] = 0.022
    activity[lung_l.astype(bool)] = 0.03
    activity[lung_r.astype(bool)] = 0.03

    # Heart (moderate-high uptake)
    heart = _ellipse_mask(H, W, -0.05, -0.03, 0.08, 0.07, 15 + variant * 5) & body
    heart_inner = _ellipse_mask(H, W, -0.05, -0.03, 0.05, 0.04, 15 + variant * 5) & body
    myocardium = heart & ~heart_inner
    activity[myocardium.astype(bool)] = 0.70 + rng.uniform(-0.05, 0.10)
    activity[heart_inner.astype(bool)] = 0.20  # blood pool

    # Liver (right side, moderate uptake)
    liver = _ellipse_mask(H, W, 0.15, 0.08, 0.15, 0.10, -10) & body
    liver &= ~lung_r
    activity[liver.astype(bool)] = 0.45 + rng.uniform(-0.05, 0.05)
    mu_map[liver.astype(bool)] = 0.098

    # Kidneys
    kidney_l = _ellipse_mask(H, W, -0.15, 0.12, 0.04, 0.06, -10) & body
    kidney_r = _ellipse_mask(H, W, 0.15, 0.12, 0.04, 0.06, 10) & body
    activity[kidney_l.astype(bool)] = 0.55
    activity[kidney_r.astype(bool)] = 0.55

    # Lesions (1-4 tumors)
    n_tumors = rng.integers(1, 5)
    for _ in range(n_tumors):
        tx = rng.uniform(-0.30, 0.30)
        ty = rng.uniform(-0.25, 0.25)
        tr = rng.uniform(0.01, 0.035)
        tumor = _ellipse_mask(H, W, tx, ty, tr, tr * rng.uniform(0.6, 1.4),
                              rng.uniform(-45, 45)) & body
        if tumor.sum() > 5:
            activity[tumor.astype(bool)] = rng.uniform(1.5, 3.5)

    # Cold lesion (necrotic tumor / cyst)
    if rng.random() < 0.5:
        cx = rng.uniform(-0.20, 0.20)
        cy = rng.uniform(-0.15, 0.15)
        cr = rng.uniform(0.02, 0.04)
        cold = _ellipse_mask(H, W, cx, cy, cr, cr * 0.9, 0) & body
        activity[cold.astype(bool)] = 0.02

    activity = gaussian_filter(activity, sigma=0.8)
    activity = np.clip(activity, 0.0, None)
    if activity.max() > 0:
        activity /= activity.max()

    return activity, mu_map, f"body_{variant:02d}"


def make_cardiac_phantom(
    H: int, W: int, seed: int, variant: int = 0
) -> tuple[np.ndarray, np.ndarray, str]:
    """Generate a cardiac PET phantom (myocardial perfusion-like).

    Returns:
        activity: (H, W) float64
        mu_map:   (H, W) float64
        name:     scene name
    """
    rng = np.random.default_rng(seed)

    activity = np.zeros((H, W), dtype=np.float64)
    mu_map = np.zeros((H, W), dtype=np.float64)

    # Chest outline
    chest = _ellipse_mask(H, W, 0.0, 0.0, 0.42, 0.35, 0)
    mu_map[chest.astype(bool)] = 0.096
    activity[chest.astype(bool)] = 0.10

    # Lungs
    lung_l = _ellipse_mask(H, W, -0.20, -0.02, 0.13, 0.20, -3)
    lung_r = _ellipse_mask(H, W, 0.20, -0.02, 0.13, 0.20, 3)
    for lung in [lung_l & chest, lung_r & chest]:
        mu_map[lung.astype(bool)] = 0.022
        activity[lung.astype(bool)] = 0.03

    # Myocardium (prominent ring structure)
    heart_cx = -0.05 + variant * 0.01
    heart_cy = -0.02
    heart_outer = _ellipse_mask(H, W, heart_cx, heart_cy,
                                0.12 + variant * 0.005, 0.11, 10)
    heart_inner = _ellipse_mask(H, W, heart_cx, heart_cy,
                                0.07 + variant * 0.003, 0.06, 10)
    myocardium = (heart_outer & ~heart_inner) & chest
    blood_pool = heart_inner & chest

    # Uniform myocardial uptake (normal perfusion)
    base_myo = 0.90 + rng.uniform(-0.05, 0.05)
    activity[myocardium.astype(bool)] = base_myo
    activity[blood_pool.astype(bool)] = 0.25  # blood pool

    # Perfusion defect (0-2 segments with reduced uptake)
    n_defects = rng.integers(0, 3)
    for _ in range(n_defects):
        # Angular sector of myocardium
        angle_start = rng.uniform(0, 360)
        angle_span = rng.uniform(30, 90)
        y_coords, x_coords = np.where(myocardium)
        # Convert to angle from heart center
        ctr_y = int((heart_cy + 1.0) * H / 2)
        ctr_x = int((heart_cx + 1.0) * W / 2)
        angles = np.degrees(np.arctan2(y_coords - ctr_y, x_coords - ctr_x)) % 360
        in_sector = (angles >= angle_start) & (angles < angle_start + angle_span)
        if in_sector.sum() > 0:
            defect_uptake = rng.uniform(0.30, 0.65)
            activity[y_coords[in_sector], x_coords[in_sector]] = defect_uptake

    # Liver (appears at bottom of cardiac FOV)
    liver = _ellipse_mask(H, W, 0.10, 0.20, 0.20, 0.08, -5) & chest
    activity[liver.astype(bool)] = 0.50

    # Spine
    spine = _ellipse_mask(H, W, 0.0, 0.28, 0.04, 0.03, 0) & chest
    mu_map[spine.astype(bool)] = 0.15
    activity[spine.astype(bool)] = 0.05

    activity = gaussian_filter(activity, sigma=0.8)
    activity = np.clip(activity, 0.0, None)
    if activity.max() > 0:
        activity /= activity.max()

    return activity, mu_map, f"cardiac_{variant:02d}"


# ── Phantom diversity pool for each tier ─────────────────────────────────────

PHANTOM_GENERATORS = [make_brain_fdg_phantom, make_body_phantom, make_cardiac_phantom]


def generate_phantoms_public(n: int = 12) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Generate diverse public-tier phantoms: 4 brain + 4 body + 4 cardiac."""
    phantoms = []
    for i in range(4):
        phantoms.append(make_brain_fdg_phantom(IMAGE_SIZE, IMAGE_SIZE, seed=100 + i, variant=i))
    for i in range(4):
        phantoms.append(make_body_phantom(IMAGE_SIZE, IMAGE_SIZE, seed=200 + i, variant=i))
    for i in range(4):
        phantoms.append(make_cardiac_phantom(IMAGE_SIZE, IMAGE_SIZE, seed=300 + i, variant=i))
    return phantoms[:n]


def generate_phantoms_dev(n: int = 20) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Generate dev-tier phantoms with augmented diversity."""
    phantoms = []
    rng = np.random.default_rng(5000)
    for i in range(n):
        gen_fn = PHANTOM_GENERATORS[i % 3]
        activity, mu_map, name = gen_fn(IMAGE_SIZE, IMAGE_SIZE, seed=500 + i, variant=i)
        # Augment: rotation + flip + mild zoom
        angle = float(rng.uniform(15, 345))
        activity = nd_rotate(activity, angle, reshape=False, mode='constant', cval=0.0)
        mu_map = nd_rotate(mu_map, angle, reshape=False, mode='constant', cval=0.0)
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


def generate_phantoms_hidden(n: int = 20) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Generate hidden-tier phantoms with adversarial modifications."""
    phantoms = []
    rng = np.random.default_rng(8000)
    for i in range(n):
        gen_fn = PHANTOM_GENERATORS[i % 3]
        activity, mu_map, name = gen_fn(IMAGE_SIZE, IMAGE_SIZE, seed=800 + i, variant=i + 10)

        # Adversarial augmentation
        angle = float(rng.uniform(20, 340))
        activity = nd_rotate(activity, angle, reshape=False, mode='constant', cval=0.0)
        mu_map = nd_rotate(mu_map, angle, reshape=False, mode='constant', cval=0.0)
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
            r = rng.integers(2, 6)
            yy, xx = np.ogrid[-r:r+1, -r:r+1]
            circle = (yy**2 + xx**2 <= r**2).astype(np.float64)
            y0, y1 = max(0, cy - r), min(IMAGE_SIZE, cy + r + 1)
            x0, x1 = max(0, cx - r), min(IMAGE_SIZE, cx + r + 1)
            c_y0, c_y1 = r - (cy - y0), r + (y1 - cy)
            c_x0, c_x1 = r - (cx - x0), r + (x1 - cx)
            if activity[y0:y1, x0:x1].mean() > 0.1:  # only in active regions
                intensity = rng.uniform(1.5, 4.0)
                activity[y0:y1, x0:x1] = np.maximum(
                    activity[y0:y1, x0:x1],
                    circle[c_y0:c_y1, c_x0:c_x1] * intensity
                )

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


# ── PET Forward Model ────────────────────────────────────────────────────────

def pet_forward_model(
    activity: np.ndarray,
    mu_map: np.ndarray,
    theta_deg: np.ndarray,
    count_rate_mcps: float,
    scatter_fraction: float,
    randoms_fraction: float,
    attenuation_error: float,
    rng: np.random.Generator,
) -> dict:
    """Apply full PET forward model.

    y_i ~ Poisson(a_i * [A * x]_i + r_i + s_i)

    Args:
        activity:         (H, W) ground-truth activity map [0, 1]
        mu_map:           (H, W) attenuation map [cm^-1]
        theta_deg:        projection angles in degrees
        count_rate_mcps:  total count rate in mega-counts per second
        scatter_fraction: s / (a*Ax + r + s)
        randoms_fraction: r / (a*Ax + r + s)
        attenuation_error: relative error in mu-map
        rng:              random generator

    Returns:
        dict with sinogram_ideal, sinogram_measured, attenuation_factors, etc.
    """
    n_angles = len(theta_deg)

    # 1. Ideal sinogram (Radon transform of activity)
    sino_ideal = radon_transform(activity, theta_deg)
    sino_ideal = np.maximum(sino_ideal, 0.0)
    n_det = sino_ideal.shape[1]

    # 2. Attenuation factors: a_i = exp(-Radon(mu_map))
    sino_mu = radon_transform(mu_map, theta_deg)
    # Scale by pixel size (~0.86 mm for 220mm FOV / 256 px)
    pixel_size_cm = 22.0 / IMAGE_SIZE  # 220 mm = 22 cm, so pixel_size_cm ~ 0.086 cm
    sino_mu_physical = sino_mu * pixel_size_cm
    attn_factors_true = np.exp(-sino_mu_physical)

    # Apply attenuation error to get the "used" attenuation map
    if attenuation_error > 0:
        mu_map_err = mu_map * (1.0 + rng.uniform(-attenuation_error, attenuation_error,
                                                   mu_map.shape))
    else:
        mu_map_err = mu_map.copy()
    sino_mu_err = radon_transform(mu_map_err, theta_deg) * pixel_size_cm
    attn_factors_used = np.exp(-sino_mu_err)

    # 3. Scale sinogram to physical count level
    # total_counts = count_rate * acquisition_time (assume 1 second effective per line)
    # Scale ideal sinogram so total expected counts match count_rate
    total_expected_trues = count_rate_mcps * 1e6  # total true coincidences
    if sino_ideal.sum() > 0:
        scale = total_expected_trues / sino_ideal.sum()
    else:
        scale = 1.0
    sino_trues = sino_ideal * scale

    # 4. Attenuated trues: a_i * [A*x]_i (using TRUE attenuation)
    sino_atten = sino_trues * attn_factors_true

    # 5. Scatter: smooth background proportional to scatter_fraction
    # s = scatter_fraction * mean(attenuated_trues)
    mean_signal = sino_atten.mean() if sino_atten.mean() > 0 else 1.0
    # scatter_fraction is fraction of total, so s / (s + r + atten) = sf
    # We solve: s = sf / (1 - sf - rf) * mean_atten (approximately)
    denom = max(1.0 - scatter_fraction - randoms_fraction, 0.1)
    scatter_level = scatter_fraction / denom * mean_signal
    scatter = np.ones_like(sino_atten) * scatter_level
    # Add spatial variation to scatter (smooth)
    scatter_var = rng.standard_normal(sino_atten.shape) * scatter_level * 0.1
    scatter += gaussian_filter(scatter_var, sigma=[5.0, 10.0])
    scatter = np.maximum(scatter, 0.0)

    # 6. Randoms: uniform background
    randoms_level = randoms_fraction / denom * mean_signal
    randoms = np.ones_like(sino_atten) * randoms_level
    randoms += rng.standard_normal(sino_atten.shape) * randoms_level * 0.05
    randoms = np.maximum(randoms, 0.0)

    # 7. Expected counts (what detector measures before Poisson sampling)
    expected = sino_atten + scatter + randoms
    expected = np.maximum(expected, 0.01)  # avoid zero for Poisson

    # 8. Poisson sampling
    sino_measured = rng.poisson(expected).astype(np.float64)

    return {
        "sinogram_ideal": sino_ideal.astype(np.float32),
        "sinogram_measured": sino_measured.astype(np.float32),
        "attenuation_factors": attn_factors_true.astype(np.float32),
        "attenuation_factors_used": attn_factors_used.astype(np.float32),
        "scatter": scatter.astype(np.float32),
        "randoms": randoms.astype(np.float32),
        "expected_counts": expected.astype(np.float32),
        "scale_factor": float(scale),
    }


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


# ── Image helpers ────────────────────────────────────────────────────────────

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
    """4-panel overview: GT | ideal sino | measured sino | FBP recon."""
    th, tw = 128, 128

    def _r(a):
        pil = Image.fromarray(np.clip(_norm(a) * 255, 0, 255).astype(np.uint8), "L")
        return np.array(pil.resize((tw, th), Image.LANCZOS)) / 255.0

    ov = np.zeros((th, 4 * tw), dtype=np.float32)
    ov[:, 0:tw] = _r(x_true)
    ov[:, tw:2*tw] = _r(sino_ideal)
    ov[:, 2*tw:3*tw] = _r(sino_meas)
    ov[:, 3*tw:4*tw] = _r(recon_fbp)
    _save_png(ov, path)


# ── Tier generation ──────────────────────────────────────────────────────────

def sample_mismatch(rng: np.random.Generator, spec: dict) -> dict:
    return {k: float(rng.uniform(v["min"], v["max"])) for k, v in spec.items()}


def generate_tier(
    tier: str,
    phantoms: list[tuple[np.ndarray, np.ndarray, str]],
    base_seed: int,
) -> None:
    """Generate one tier of the PET benchmark."""
    spec_ranges = SPEC[tier]
    tier_dir = BENCHMARK_DIR / tier
    images_dir = tier_dir / "images"
    if tier_dir.exists():
        shutil.rmtree(tier_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    theta_deg = np.linspace(0, 180, N_ANGLES, endpoint=False).astype(np.float64)

    h5_path = tier_dir / f"pet_challenge_{tier}.h5"
    rng = np.random.default_rng(base_seed)
    true_specs = {}
    all_psnrs = []
    all_ssims = []

    with h5py.File(h5_path, "w") as f:
        f.attrs["description"] = (
            f"PWM PET benchmark -- {tier} tier "
            f"(parallel-beam Radon + Poisson noise + attenuation + scatter + randoms)"
        )
        f.attrs["spec_ranges"] = json.dumps(spec_ranges)
        f.attrs["geometry"] = json.dumps({
            "image_size": IMAGE_SIZE,
            "n_angles": N_ANGLES,
            "n_det": N_DET,
            "angle_range_deg": [0, 180],
            "fov_mm": 220.0,
            "pixel_size_mm": 220.0 / IMAGE_SIZE,
        })
        f.attrs["forward_model"] = (
            "y_i ~ Poisson(a_i * [A * x]_i + r_i + s_i)"
        )

        for idx, (activity, mu_map, scene_name) in enumerate(phantoms):
            key = f"sample_{idx:02d}"
            print(f"  [{tier}] Generating {key} ({scene_name})...", end="", flush=True)

            mis = sample_mismatch(rng, spec_ranges)
            true_specs[key] = mis

            # Apply forward model
            result = pet_forward_model(
                activity, mu_map, theta_deg,
                count_rate_mcps=mis["count_rate_mcps"],
                scatter_fraction=mis["scatter_fraction"],
                randoms_fraction=mis["randoms_fraction"],
                attenuation_error=mis["attenuation_error"],
                rng=rng,
            )

            sino_ideal = result["sinogram_ideal"]
            sino_measured = result["sinogram_measured"]
            n_det = sino_ideal.shape[1]

            # FBP reconstruction from measured sinogram
            # Simple approach: correct for attenuation + subtract estimated scatter/randoms,
            # then apply FBP
            attn_corr = result["attenuation_factors_used"]
            attn_corr_safe = np.where(attn_corr > 0.01, attn_corr, 0.01)

            # Corrected sinogram: (measured - randoms - scatter) / attenuation
            sino_corrected = (sino_measured - result["randoms"] - result["scatter"])
            sino_corrected = np.maximum(sino_corrected, 0.0) / attn_corr_safe

            # Rescale back to activity units
            if result["scale_factor"] > 0:
                sino_corrected /= result["scale_factor"]

            recon_fbp = fbp_reconstruct(sino_corrected, theta_deg, IMAGE_SIZE)
            recon_fbp = np.maximum(recon_fbp, 0.0).astype(np.float32)

            # Normalize for PSNR/SSIM computation
            gt_max = activity.max() if activity.max() > 0 else 1.0
            psnr = compute_psnr(activity, recon_fbp)
            ssim = compute_ssim(activity, recon_fbp)
            all_psnrs.append(psnr)
            all_ssims.append(ssim)

            # Save to HDF5
            grp = f.create_group(key)
            grp.create_dataset("x_true", data=activity.astype(np.float32),
                               compression="gzip")
            grp.create_dataset("sinogram_ideal", data=sino_ideal, compression="gzip")
            grp.create_dataset("sinogram_measured", data=sino_measured,
                               compression="gzip")
            grp.create_dataset("attenuation_map", data=mu_map.astype(np.float32),
                               compression="gzip")
            grp.create_dataset("angles_deg", data=theta_deg.astype(np.float32))
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
            grp.attrs["true_spec"] = json.dumps(mis)
            grp.attrs["spec_ranges"] = json.dumps(spec_ranges)

            # Save images
            sample_dir = images_dir / f"sample_{idx:02d}_{scene_name}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            _save_png(activity, sample_dir / "ground_truth.png")
            _save_png(sino_ideal, sample_dir / "sinogram_ideal.png")
            _save_png(sino_measured, sample_dir / "sinogram_measured.png")
            _save_png(recon_fbp, sample_dir / "reconstruction_fbp.png")
            _save_overview(activity, sino_ideal, sino_measured, recon_fbp,
                           sample_dir / "overview.png")
            with open(sample_dir / "spec.json", "w") as sf:
                json.dump({"scene": scene_name, "spec_ranges": spec_ranges,
                           "true_spec": mis, "psnr_fbp": psnr, "ssim_fbp": ssim},
                          sf, indent=2)

            print(f"  PSNR={psnr:.2f} dB, SSIM={ssim:.3f}  "
                  f"counts={mis['count_rate_mcps']:.1f} Mcps  "
                  f"scatter={mis['scatter_fraction']:.2f}  "
                  f"randoms={mis['randoms_fraction']:.2f}  "
                  f"attn_err={mis['attenuation_error']:.3f}")

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


# ── README writers ───────────────────────────────────────────────────────────

def _write_top_readme() -> None:
    txt = f"""# PET -- 2-D Parallel-Beam Emission Tomography

## Overview

Positron Emission Tomography (PET) benchmark with realistic physics:
Radon transform + attenuation + scatter + random coincidences + Poisson noise.

## Forward Model

```
y_i ~ Poisson(a_i * [A * x]_i + r_i + s_i)

where:
    x       : activity map (ground truth, 256x256)
    A       : system matrix (parallel-beam Radon transform, 256 angles)
    a_i     : attenuation correction factors (from mu-map)
    r_i     : random coincidences (uniform background)
    s_i     : scatter contribution (smooth background)
    y_i     : measured sinogram (counts)
```

## Geometry

| Parameter | Value |
|-----------|-------|
| IMAGE_SIZE | 256 x 256 |
| n_angles | 256 |
| n_det | 367 |
| angle_range | [0, 180) degrees |
| FOV | 220 mm |
| pixel_size | 0.86 mm/px |

## Mismatch Parameters

| Parameter | Description | Public | Dev | Hidden |
|-----------|-------------|--------|-----|--------|
| count_rate_mcps | Total count rate | 2-5 Mcps | 1-5 Mcps | 0.5-5 Mcps |
| scatter_fraction | Scatter / total | 0.30-0.40 | 0.30-0.45 | 0.30-0.55 |
| randoms_fraction | Randoms / total | 0.10-0.25 | 0.10-0.35 | 0.10-0.50 |
| attenuation_error | Relative mu-map error | 0-3% | 0-6% | 0-10% |

## Phantoms

| Type | Samples | Description |
|------|---------|-------------|
| Brain FDG | 4/tier | Modified Shepp-Logan with gray/white matter, lesions |
| Body | 4/tier | Torso with organs, lungs, heart, lesions |
| Cardiac | 4/tier | Myocardial perfusion with defects |

## Dataset Structure

```
pet/
+-- README.md
+-- generate_dataset.py
+-- public/    12 samples (GT + ideal sino + true spec visible)
+-- dev/       20 samples (blind eval, augmented variants)
+-- hidden/    20 samples (adversarial: micro-lesions, extreme params)
```

## HDF5 Structure (per sample)

```
sample_XX/
+-- x_true (256, 256) float32         # Ground truth activity map
+-- sinogram_ideal (256, 367) float32  # Clean Radon sinogram (no noise/scatter)
+-- sinogram_measured (256, 367) float32 # Measured sinogram (Poisson + scatter + randoms)
+-- attenuation_map (256, 256) float32 # Mu-map (attenuation coefficients)
+-- angles_deg (256,) float32          # Projection angles in degrees
+-- reconstruction_fbp (256, 256) float32 # FBP baseline reconstruction
```

## Scoring

```
Score = 0.4 * PSNR_norm + 0.4 * SSIM + 0.2 * Consistency
```

## References

1. Shepp, L.A. & Vardi, Y. (1982) "Maximum Likelihood Reconstruction for
   Emission Tomography," IEEE TMI.
2. Hudson, H.M. & Larkin, R.S. (1994) "Accelerated Image Reconstruction
   Using Ordered Subsets of Projection Data," IEEE TMI.
3. Reader, A.J. & Verhaeghe, J. (2014) "4D image reconstruction for
   emission tomography," PMB.
"""
    with open(BENCHMARK_DIR / "README.md", "w") as f:
        f.write(txt)


# ── Gallery image generation ────────────────────────────────────────────────

def generate_gallery_images() -> None:
    """Generate gallery images for the platform benchmark page.

    Creates scene_00 through scene_03 with:
      gt.png, measurement_I.png, measurement_II.png,
      recon_I.png, recon_II.png, recon_III.png
    """
    gallery_base = (BENCHMARK_DIR.parent.parent.parent /
                    "platform" / "pwm_platform" / "static" / "img" /
                    "benchmark_gallery" / "pet")

    # Load from public tier HDF5
    h5_path = BENCHMARK_DIR / "public" / "pet_challenge_public.h5"
    if not h5_path.exists():
        print("  [gallery] Public HDF5 not found, skipping gallery generation.")
        return

    # Pick diverse samples: brain (0), body (4), cardiac (8), brain variant (3)
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

            # gt.png — ground truth activity
            _save_png(x_true, scene_dir / "gt.png")

            # measurement_I.png — measured sinogram
            _save_png(sino_meas, scene_dir / "measurement_I.png")

            # measurement_II.png — ideal sinogram
            _save_png(sino_ideal, scene_dir / "measurement_II.png")

            # recon_I.png — FBP reconstruction
            _save_png(recon_fbp, scene_dir / "recon_I.png")

            # recon_II.png — attenuation map
            _save_png(attn_map, scene_dir / "recon_II.png")

            # recon_III.png — difference image |GT - FBP|
            diff = np.abs(x_true - recon_fbp)
            _save_png(diff, scene_dir / "recon_III.png")

            print(f"  [gallery] scene_{scene_idx:02d} images saved to {scene_dir}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("PET Benchmark Dataset Generator")
    print("=" * 68)
    print(f"Output: {BENCHMARK_DIR}")
    print(f"Geometry: {N_ANGLES} angles, {IMAGE_SIZE}x{IMAGE_SIZE} images\n")

    # ── Public tier (12 samples) ────────────────────────────────────────────
    print("Generating public tier (12 samples)...")
    public_phantoms = generate_phantoms_public(12)
    generate_tier("public", public_phantoms, base_seed=1000)

    # ── Dev tier (20 samples) ──────────────────────────────────────────────
    print("\nGenerating dev tier (20 samples)...")
    dev_phantoms = generate_phantoms_dev(20)
    generate_tier("dev", dev_phantoms, base_seed=5000)

    # ── Hidden tier (20 samples) ──────────────────────────────────────────
    print("\nGenerating hidden tier (20 samples)...")
    hidden_phantoms = generate_phantoms_hidden(20)
    generate_tier("hidden", hidden_phantoms, base_seed=8000)

    # ── README ──────────────────────────────────────────────────────────────
    _write_top_readme()

    # ── Gallery images ──────────────────────────────────────────────────────
    print("\nGenerating gallery images...")
    generate_gallery_images()

    print(f"\n{'=' * 68}")
    print("PET benchmark dataset generation complete!")
    print(f"Output: {BENCHMARK_DIR}")
    print("=" * 68)


if __name__ == "__main__":
    main()
