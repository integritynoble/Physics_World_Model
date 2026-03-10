#!/usr/bin/env python3
"""Generate Diffusion MRI benchmark datasets (public / dev / hidden).

Forward model:
    y = U_Omega * F * (S0 * exp(-b * ADC)) + n

where:
    S0      : proton density / T2-weighted baseline signal (256x256)
    b       : b-value (diffusion weighting factor, 0-3000 s/mm^2)
    ADC     : apparent diffusion coefficient map (tissue-specific, mm^2/s)
    F       : 2D Fourier transform
    U_Omega : k-space undersampling mask (variable-density Cartesian)
    n       : complex Gaussian noise (Rician in magnitude domain)

Brain slice phantoms with realistic ADC contrast:
    - White matter  : low ADC  ~ 0.7e-3 mm^2/s
    - Gray matter   : medium   ~ 1.0e-3 mm^2/s
    - CSF           : high     ~ 3.0e-3 mm^2/s
    - Fiber tracts  : anisotropic diffusion regions

Mismatch parameters:
    acceleration_factor     : k-space undersampling ratio
    noise_sigma             : complex Gaussian noise std (Rician in magnitude)
    b_value_error           : fractional error in assumed b-value
    eddy_current_distortion : eddy-current phase distortion amplitude (radians)

CPU baseline reconstruction:
    Zero-filled IFFT of undersampled k-space
    Expected baseline: ~20-26 dB PSNR

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
from scipy.ndimage import gaussian_filter, rotate as nd_rotate, zoom as nd_zoom

# ── Constants ─────────────────────────────────────────────────────────────────

IMAGE_SIZE = 256
BENCHMARK_DIR = Path(__file__).resolve().parent

# Tissue ADC values in mm^2/s (literature values at 37 C)
ADC_WM = 0.7e-3     # white matter
ADC_GM = 1.0e-3     # gray matter (cortex)
ADC_CSF = 3.0e-3    # cerebrospinal fluid
ADC_DEEP_GM = 0.8e-3  # deep gray matter (thalamus, caudate)
ADC_TUMOR = 0.5e-3   # restricted diffusion (e.g., acute stroke, tumor)
ADC_EDEMA = 1.5e-3   # vasogenic edema

# S0 (proton density) relative intensities
S0_WM = 0.70
S0_GM = 0.85
S0_CSF = 1.00
S0_DEEP_GM = 0.80
S0_SKULL = 0.05
S0_TUMOR = 0.75
S0_EDEMA = 0.90

# Default b-value for acquisition
B_VALUE_DEFAULT = 1000.0  # s/mm^2

# ── Mismatch ranges per tier ─────────────────────────────────────────────────

SPEC = {
    "public": {
        "acceleration_factor":     {"min": 2.0, "max": 4.0, "unit": "x"},
        "noise_sigma":             {"min": 0.01, "max": 0.03, "unit": ""},
        "b_value_error":           {"min": -0.05, "max": 0.05, "unit": "relative"},
        "eddy_current_distortion": {"min": 0.0, "max": 0.1, "unit": "radians"},
    },
    "dev": {
        "acceleration_factor":     {"min": 3.0, "max": 6.0, "unit": "x"},
        "noise_sigma":             {"min": 0.02, "max": 0.06, "unit": ""},
        "b_value_error":           {"min": -0.10, "max": 0.10, "unit": "relative"},
        "eddy_current_distortion": {"min": 0.0, "max": 0.3, "unit": "radians"},
    },
    "hidden": {
        "acceleration_factor":     {"min": 4.0, "max": 8.0, "unit": "x"},
        "noise_sigma":             {"min": 0.03, "max": 0.10, "unit": ""},
        "b_value_error":           {"min": -0.15, "max": 0.15, "unit": "relative"},
        "eddy_current_distortion": {"min": 0.0, "max": 0.5, "unit": "radians"},
    },
}


# ── Phantom Generators ───────────────────────────────────────────────────────

def _ellipse_mask(H: int, W: int, cx: float, cy: float,
                  a: float, b: float, angle_deg: float = 0.0) -> np.ndarray:
    """Binary ellipse mask in normalized [-1, 1] coordinates."""
    yy = np.linspace(-1.0, 1.0, H)[:, None]
    xx = np.linspace(-1.0, 1.0, W)[None, :]
    ca, sa = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
    yr = (yy - cy) * ca - (xx - cx) * sa
    xr = (yy - cy) * sa + (xx - cx) * ca
    return (xr / a) ** 2 + (yr / b) ** 2 <= 1.0


def _fiber_tract(H: int, W: int, cx: float, cy: float,
                 length: float, width: float, angle_deg: float) -> np.ndarray:
    """Elongated ellipse representing a fiber tract bundle."""
    return _ellipse_mask(H, W, cx, cy, width, length, angle_deg)


def make_brain_diffusion_phantom(
    H: int, W: int, seed: int, variant: int = 0,
    slice_level: str = "mid",
    pathology: str = "none",
) -> tuple[np.ndarray, np.ndarray, str]:
    """Generate a brain diffusion MRI phantom with S0 and ADC maps.

    Args:
        H, W: image dimensions
        seed: random seed for reproducibility
        variant: variant index for geometric diversity
        slice_level: "mid", "sup", or "inf" (axial slice level)
        pathology: "none", "stroke", "tumor", "ms"

    Returns:
        s0_map:  (H, W) float64 — proton density / baseline signal
        adc_map: (H, W) float64 — apparent diffusion coefficient map (mm^2/s)
        name:    scene description string
    """
    rng = np.random.default_rng(seed)

    s0_map = np.zeros((H, W), dtype=np.float64)
    adc_map = np.zeros((H, W), dtype=np.float64)

    # Slice-level-dependent geometry
    if slice_level == "sup":
        skull_a, skull_b = 0.40 + variant * 0.005, 0.44 + variant * 0.003
        vent_size = 0.015
    elif slice_level == "inf":
        skull_a, skull_b = 0.43 + variant * 0.005, 0.50 + variant * 0.003
        vent_size = 0.04
    else:  # mid
        skull_a, skull_b = 0.42 + variant * 0.005, 0.48 + variant * 0.003
        vent_size = 0.025

    # Skull
    skull_outer = _ellipse_mask(H, W, 0.0, 0.0, skull_a, skull_b, 0)
    skull_inner = _ellipse_mask(H, W, 0.0, 0.0,
                                skull_a - 0.04, skull_b - 0.04, 0)
    bone = skull_outer & ~skull_inner
    s0_map[bone] = S0_SKULL
    adc_map[bone] = 0.2e-3  # cortical bone — very low diffusivity

    # Brain parenchyma (white matter baseline)
    brain = skull_inner.copy()
    s0_map[brain] = S0_WM + rng.uniform(-0.02, 0.02)
    adc_map[brain] = ADC_WM + rng.uniform(-0.05e-3, 0.05e-3)

    # Gray matter cortex (outer band)
    cortex_outer = _ellipse_mask(H, W, 0.0, 0.0,
                                 skull_a - 0.05, skull_b - 0.05, 0)
    cortex_inner = _ellipse_mask(H, W, 0.0, 0.0,
                                 skull_a - 0.10, skull_b - 0.10, 0)
    cortex = cortex_outer & ~cortex_inner & brain
    s0_map[cortex] = S0_GM + rng.uniform(-0.03, 0.03)
    adc_map[cortex] = ADC_GM + rng.uniform(-0.05e-3, 0.05e-3)

    # Deep gray matter structures
    dgm_structures = [
        # (cx, cy, a, b, angle) — thalamus, caudate, putamen, globus pallidus
        (-0.08, -0.04, 0.04, 0.035, 0),    # left thalamus
        (0.08, -0.04, 0.04, 0.035, 0),     # right thalamus
        (-0.10, 0.06, 0.035, 0.07, -10),   # left caudate
        (0.10, 0.06, 0.035, 0.07, 10),     # right caudate
        (-0.16, -0.01, 0.05, 0.025, 0),    # left putamen
        (0.16, -0.01, 0.05, 0.025, 0),     # right putamen
        (-0.13, -0.02, 0.02, 0.02, 0),     # left globus pallidus
        (0.13, -0.02, 0.02, 0.02, 0),      # right globus pallidus
    ]
    for cx, cy, a, b, ang in dgm_structures:
        cx += rng.uniform(-0.008, 0.008)
        cy += rng.uniform(-0.008, 0.008)
        mask = _ellipse_mask(H, W, cx, cy, a, b, ang + variant * 2) & brain
        s0_map[mask] = S0_DEEP_GM + rng.uniform(-0.03, 0.03)
        adc_map[mask] = ADC_DEEP_GM + rng.uniform(-0.05e-3, 0.05e-3)

    # Lateral ventricles (CSF — high ADC, high S0)
    v_offset = variant * 0.005
    vent_l = _ellipse_mask(H, W, -0.04 + v_offset, 0.04,
                           vent_size, vent_size * 2.5, -5) & brain
    vent_r = _ellipse_mask(H, W, 0.04 - v_offset, 0.04,
                           vent_size, vent_size * 2.5, 5) & brain
    for v in [vent_l, vent_r]:
        s0_map[v] = S0_CSF
        adc_map[v] = ADC_CSF + rng.uniform(-0.2e-3, 0.2e-3)

    # Third ventricle (midline, small)
    v3 = _ellipse_mask(H, W, 0.0, 0.03, 0.008, 0.025, 0) & brain
    s0_map[v3] = S0_CSF
    adc_map[v3] = ADC_CSF

    # ── White matter fiber tracts (anisotropic diffusion regions) ──
    # These have low ADC (restricted) but distinctive directionality
    fiber_tracts = [
        # Corpus callosum (crossing midline)
        (0.0, 0.10, 0.18, 0.015, 0),
        # Internal capsule (left + right)
        (-0.10, 0.01, 0.015, 0.08, -15),
        (0.10, 0.01, 0.015, 0.08, 15),
        # Corona radiata (superior projections)
        (-0.07, 0.15, 0.02, 0.06, -8),
        (0.07, 0.15, 0.02, 0.06, 8),
        # Superior longitudinal fasciculus
        (-0.22, 0.02, 0.06, 0.015, 5 + variant * 2),
        (0.22, 0.02, 0.06, 0.015, -5 - variant * 2),
        # Cingulum bundle (around ventricles)
        (-0.05, 0.12, 0.03, 0.01, -20),
        (0.05, 0.12, 0.03, 0.01, 20),
    ]
    for cx, cy, a, b, ang in fiber_tracts:
        cx += rng.uniform(-0.005, 0.005)
        cy += rng.uniform(-0.005, 0.005)
        mask = _fiber_tract(H, W, cx, cy, a, b, ang) & brain
        # Fiber tracts: lower ADC than surrounding WM (restricted diffusion)
        s0_map[mask] = S0_WM + 0.05
        adc_map[mask] = 0.5e-3 + rng.uniform(-0.03e-3, 0.03e-3)

    # ── Pathology ──
    pathology_desc = pathology
    if pathology == "stroke":
        # Acute ischemic stroke: restricted diffusion (very low ADC)
        # Typically in MCA territory
        stroke_cx = rng.uniform(-0.20, -0.05)
        stroke_cy = rng.uniform(-0.10, 0.10)
        stroke_r = rng.uniform(0.03, 0.08)
        stroke_mask = _ellipse_mask(H, W, stroke_cx, stroke_cy,
                                    stroke_r, stroke_r * rng.uniform(0.6, 1.4),
                                    rng.uniform(-30, 30)) & brain
        if stroke_mask.sum() > 20:
            s0_map[stroke_mask] = S0_TUMOR
            adc_map[stroke_mask] = 0.3e-3 + rng.uniform(-0.05e-3, 0.05e-3)
            # Perilesional edema (high ADC)
            edema_mask = _ellipse_mask(
                H, W, stroke_cx, stroke_cy,
                stroke_r + 0.03, (stroke_r + 0.03) * 1.2,
                rng.uniform(-30, 30)
            ) & brain & ~stroke_mask
            s0_map[edema_mask] = S0_EDEMA
            adc_map[edema_mask] = ADC_EDEMA

    elif pathology == "tumor":
        # Brain tumor: heterogeneous ADC with necrotic core
        t_cx = rng.uniform(-0.15, 0.15)
        t_cy = rng.uniform(-0.10, 0.15)
        t_r = rng.uniform(0.04, 0.09)
        tumor_outer = _ellipse_mask(H, W, t_cx, t_cy,
                                    t_r, t_r * rng.uniform(0.8, 1.2),
                                    rng.uniform(-20, 20)) & brain
        tumor_core = _ellipse_mask(H, W, t_cx, t_cy,
                                   t_r * 0.5, t_r * 0.5 * rng.uniform(0.8, 1.2),
                                   rng.uniform(-20, 20)) & brain
        tumor_ring = tumor_outer & ~tumor_core
        if tumor_ring.sum() > 10:
            # Solid tumor: restricted diffusion
            s0_map[tumor_ring] = S0_TUMOR
            adc_map[tumor_ring] = ADC_TUMOR + rng.uniform(-0.1e-3, 0.1e-3)
            # Necrotic core: high ADC
            s0_map[tumor_core] = 0.60
            adc_map[tumor_core] = 2.0e-3 + rng.uniform(-0.3e-3, 0.3e-3)
            # Peritumoral edema
            edema_mask = _ellipse_mask(
                H, W, t_cx, t_cy,
                t_r + 0.04, (t_r + 0.04) * 1.1, 0
            ) & brain & ~tumor_outer
            s0_map[edema_mask] = S0_EDEMA
            adc_map[edema_mask] = ADC_EDEMA

    elif pathology == "ms":
        # Multiple sclerosis: scattered demyelination plaques
        n_plaques = rng.integers(3, 8)
        for _ in range(n_plaques):
            p_cx = rng.uniform(-0.25, 0.25)
            p_cy = rng.uniform(-0.15, 0.20)
            p_r = rng.uniform(0.01, 0.03)
            plaque = _ellipse_mask(H, W, p_cx, p_cy, p_r,
                                   p_r * rng.uniform(0.6, 1.4),
                                   rng.uniform(0, 180)) & brain
            if plaque.sum() > 5:
                # Active MS plaques: mildly restricted diffusion
                s0_map[plaque] = 0.80
                adc_map[plaque] = 0.6e-3 + rng.uniform(-0.1e-3, 0.1e-3)

    # Apply mild spatial smoothing for realism
    s0_map = gaussian_filter(s0_map, sigma=0.8)
    adc_map = gaussian_filter(adc_map, sigma=0.6)

    # Ensure physical bounds
    s0_map = np.clip(s0_map, 0.0, None)
    adc_map = np.clip(adc_map, 0.0, None)

    # Normalize S0 to [0, 1]
    if s0_map.max() > 0:
        s0_map /= s0_map.max()

    name = f"brain_{slice_level}_{pathology}_v{variant:02d}"
    return s0_map, adc_map, name


# ── Phantom Pool Generators per Tier ─────────────────────────────────────────

SLICE_LEVELS = ["mid", "sup", "inf"]
PATHOLOGIES = ["none", "stroke", "tumor", "ms"]


def generate_phantoms_public(n: int = 12) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """12 diverse public-tier phantoms: 3 slice levels x 4 pathologies."""
    phantoms = []
    idx = 0
    for sl in SLICE_LEVELS:
        for path in PATHOLOGIES:
            phantoms.append(
                make_brain_diffusion_phantom(
                    IMAGE_SIZE, IMAGE_SIZE, seed=100 + idx,
                    variant=idx, slice_level=sl, pathology=path
                )
            )
            idx += 1
    return phantoms[:n]


def generate_phantoms_dev(n: int = 20) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """20 dev-tier phantoms with augmented diversity."""
    phantoms = []
    rng = np.random.default_rng(5000)
    for i in range(n):
        sl = SLICE_LEVELS[i % 3]
        path = PATHOLOGIES[i % 4]
        s0, adc, name = make_brain_diffusion_phantom(
            IMAGE_SIZE, IMAGE_SIZE, seed=500 + i,
            variant=i, slice_level=sl, pathology=path
        )
        # Augment: rotation + flip
        angle = float(rng.uniform(-15, 15))
        s0 = nd_rotate(s0, angle, reshape=False, mode='constant', cval=0.0)
        adc = nd_rotate(adc, angle, reshape=False, mode='constant', cval=0.0)
        if rng.random() < 0.5:
            s0 = np.fliplr(s0)
            adc = np.fliplr(adc)
        # Mild zoom
        zf = float(rng.uniform(0.90, 1.10))
        if abs(zf - 1.0) > 0.01:
            s0 = _zoom_crop(s0, zf, IMAGE_SIZE)
            adc = _zoom_crop(adc, zf, IMAGE_SIZE)
        s0 = np.clip(s0, 0.0, None)
        adc = np.clip(adc, 0.0, None)
        if s0.max() > 0:
            s0 /= s0.max()
        phantoms.append((s0, adc, f"dev_{name}"))
    return phantoms


def generate_phantoms_hidden(n: int = 20) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """20 hidden-tier phantoms with adversarial modifications."""
    phantoms = []
    rng = np.random.default_rng(8000)
    for i in range(n):
        sl = SLICE_LEVELS[i % 3]
        path = PATHOLOGIES[i % 4]
        s0, adc, name = make_brain_diffusion_phantom(
            IMAGE_SIZE, IMAGE_SIZE, seed=800 + i,
            variant=i + 10, slice_level=sl, pathology=path
        )
        # Aggressive augmentation
        angle = float(rng.uniform(-25, 25))
        s0 = nd_rotate(s0, angle, reshape=False, mode='constant', cval=0.0)
        adc = nd_rotate(adc, angle, reshape=False, mode='constant', cval=0.0)
        if rng.random() < 0.7:
            s0 = np.fliplr(s0)
            adc = np.fliplr(adc)
        if rng.random() < 0.5:
            s0 = np.flipud(s0)
            adc = np.flipud(adc)
        zf = float(rng.uniform(0.80, 1.20))
        s0 = _zoom_crop(s0, zf, IMAGE_SIZE)
        adc = _zoom_crop(adc, zf, IMAGE_SIZE)

        # Add subtle micro-lesions (hard to detect)
        n_micro = rng.integers(2, 5)
        for _ in range(n_micro):
            cy = rng.integers(50, IMAGE_SIZE - 50)
            cx = rng.integers(50, IMAGE_SIZE - 50)
            r = rng.integers(2, 5)
            yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
            circle = (yy ** 2 + xx ** 2 <= r ** 2).astype(np.float64)
            y0 = max(0, cy - r)
            y1 = min(IMAGE_SIZE, cy + r + 1)
            x0 = max(0, cx - r)
            x1 = min(IMAGE_SIZE, cx + r + 1)
            c_y0 = r - (cy - y0)
            c_y1 = r + (y1 - cy)
            c_x0 = r - (cx - x0)
            c_x1 = r + (x1 - cx)
            patch = circle[c_y0:c_y1, c_x0:c_x1]
            if s0[y0:y1, x0:x1].mean() > 0.1:
                # Micro-stroke: very low ADC spot
                adc[y0:y1, x0:x1] = np.where(
                    patch > 0.5,
                    0.3e-3,
                    adc[y0:y1, x0:x1]
                )
                s0[y0:y1, x0:x1] = np.where(
                    patch > 0.5,
                    s0[y0:y1, x0:x1] * 0.9,
                    s0[y0:y1, x0:x1]
                )

        s0 = np.clip(s0, 0.0, None)
        adc = np.clip(adc, 0.0, None)
        if s0.max() > 0:
            s0 /= s0.max()
        phantoms.append((s0, adc, f"hidden_{name}"))
    return phantoms


def _zoom_crop(arr: np.ndarray, zoom_f: float, size: int) -> np.ndarray:
    """Zoom array and crop/pad to target size."""
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


# ── Forward Model ─────────────────────────────────────────────────────────────

def compute_diffusion_weighted_image(
    s0_map: np.ndarray, adc_map: np.ndarray, b_value: float
) -> np.ndarray:
    """Compute diffusion-weighted magnitude image.

    S = S0 * exp(-b * ADC)

    Args:
        s0_map: (H, W) proton density baseline signal
        adc_map: (H, W) apparent diffusion coefficient (mm^2/s)
        b_value: diffusion weighting factor (s/mm^2)

    Returns:
        dwi: (H, W) float64 diffusion-weighted image (ground truth)
    """
    return s0_map * np.exp(-b_value * adc_map)


def generate_cartesian_mask(
    H: int, W: int, acceleration: float,
    center_fraction: float = 0.08, seed: int = 42
) -> np.ndarray:
    """Generate variable-density Cartesian undersampling mask.

    Keeps center k-space lines (calibration region) and randomly
    samples outer lines with probability proportional to distance
    from center.
    """
    rng = np.random.RandomState(seed)
    mask = np.zeros((H, W), dtype=np.float32)

    # Always keep center lines (calibration)
    n_center = max(1, int(W * center_fraction))
    c = W // 2
    mask[:, c - n_center // 2: c + n_center // 2] = 1.0

    # Variable-density random sampling for outer lines
    n_total_lines = max(n_center, int(W / acceleration))
    n_random = n_total_lines - n_center

    # Build candidate list (non-center lines)
    available = [i for i in range(W)
                 if i < c - n_center // 2 or i >= c + n_center // 2]

    if n_random > 0 and len(available) > 0:
        # Variable-density probability: higher near center
        probs = np.array([1.0 / (abs(i - c) + 1) for i in available])
        probs /= probs.sum()
        chosen = rng.choice(available,
                            size=min(n_random, len(available)),
                            replace=False, p=probs)
        mask[:, chosen] = 1.0

    return mask


def apply_forward_model(
    s0_map: np.ndarray, adc_map: np.ndarray,
    b_value: float, acceleration: float,
    noise_sigma: float, b_value_error: float,
    eddy_current_amp: float, seed: int,
) -> dict:
    """Apply the diffusion MRI forward model with mismatch.

    Forward model:
        x_true = S0 * exp(-b * ADC)
        y = U_Omega * F(x_true * eddy_phase) + noise

    Args:
        s0_map: (H, W) proton density map
        adc_map: (H, W) ADC map (mm^2/s)
        b_value: nominal b-value (s/mm^2)
        acceleration: undersampling factor
        noise_sigma: complex Gaussian noise std
        b_value_error: fractional error in b-value
        eddy_current_amp: eddy-current distortion amplitude (radians)
        seed: random seed

    Returns:
        dict with x_true, y (undersampled k-space), H_ideal (mask), etc.
    """
    rng = np.random.RandomState(seed)
    H, W = s0_map.shape

    # Ground truth DWI (using TRUE b-value)
    x_true = compute_diffusion_weighted_image(s0_map, adc_map, b_value)

    # Apply eddy-current distortion as a spatially-varying phase
    if abs(eddy_current_amp) > 0.001:
        yy = np.linspace(-1, 1, H)[:, None]
        xx = np.linspace(-1, 1, W)[None, :]
        # Eddy current creates linear + quadratic phase distortion
        eddy_phase = eddy_current_amp * (
            0.5 * yy + 0.3 * xx + 0.2 * yy * xx
        )
        # Convert real image to complex with eddy phase
        x_complex = x_true * np.exp(1j * eddy_phase)
    else:
        x_complex = x_true.astype(np.complex128)

    # 2D Fourier transform to k-space
    kspace_full = np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(x_complex))
    ) / np.sqrt(H * W)

    # Undersampling mask
    mask = generate_cartesian_mask(H, W, acceleration, seed=seed + 1000)

    # Apply undersampling
    kspace_under = kspace_full * mask

    # Add complex Gaussian noise (only in sampled locations)
    noise = (rng.randn(H, W) + 1j * rng.randn(H, W)) * noise_sigma / np.sqrt(2)
    kspace_under = kspace_under + noise * mask

    # The "ideal" H matrix for participants is the mask
    # (participants know the mask but may not know exact eddy currents)
    return {
        "x_true": x_true.astype(np.float32),
        "y": kspace_under.astype(np.complex64),
        "H_ideal": mask.astype(np.float32),
        "kspace_full": kspace_full.astype(np.complex64),
        "s0_map": s0_map.astype(np.float32),
        "adc_map": adc_map.astype(np.float32),
    }


# ── Reconstruction (CPU Baseline) ────────────────────────────────────────────

def reconstruct_zero_filled(kspace_under: np.ndarray) -> np.ndarray:
    """Zero-filled IFFT reconstruction (simplest baseline).

    Args:
        kspace_under: (H, W) complex undersampled k-space

    Returns:
        recon: (H, W) float32 magnitude image
    """
    H, W = kspace_under.shape
    img = np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(kspace_under))
    ) * np.sqrt(H * W)
    return np.abs(img).astype(np.float32)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_psnr(gt: np.ndarray, recon: np.ndarray) -> float:
    gt = gt.astype(np.float64)
    recon = recon.astype(np.float64)
    mse = np.mean((gt - recon) ** 2)
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
           ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2))
    return float(ssim)


# ── Image helpers ─────────────────────────────────────────────────────────────

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


def _save_kspace_png(kspace: np.ndarray, path: Path) -> None:
    """Save k-space magnitude (log scale) as PNG."""
    mag = np.abs(kspace)
    log_mag = np.log1p(mag)
    _save_png(log_mag, path)


def _save_overview(x_true, kspace_mag, recon, adc_map, path: Path) -> None:
    """4-panel overview: GT DWI | k-space | ZF recon | ADC map."""
    th, tw = 128, 128

    def _r(a):
        pil = Image.fromarray(
            np.clip(_norm(a) * 255, 0, 255).astype(np.uint8), "L"
        )
        return np.array(pil.resize((tw, th), Image.LANCZOS)) / 255.0

    ov = np.zeros((th, 4 * tw), dtype=np.float32)
    ov[:, 0:tw] = _r(x_true)
    ov[:, tw:2 * tw] = _r(kspace_mag)
    ov[:, 2 * tw:3 * tw] = _r(recon)
    ov[:, 3 * tw:4 * tw] = _r(adc_map)
    _save_png(ov, path)


# ── Mismatch sampling ────────────────────────────────────────────────────────

def sample_mismatch(rng: np.random.Generator, spec: dict) -> dict:
    return {k: float(rng.uniform(v["min"], v["max"])) for k, v in spec.items()}


# ── Tier Generation ──────────────────────────────────────────────────────────

def generate_tier(
    tier: str,
    phantoms: list[tuple[np.ndarray, np.ndarray, str]],
    base_seed: int,
) -> Path:
    """Generate one tier of the diffusion MRI benchmark.

    Args:
        tier: "public", "dev", or "hidden"
        phantoms: list of (s0_map, adc_map, scene_name) tuples
        base_seed: base random seed for this tier

    Returns:
        Path to generated HDF5 file
    """
    spec_ranges = SPEC[tier]
    tier_dir = BENCHMARK_DIR / tier
    images_dir = tier_dir / "images"
    if tier_dir.exists():
        shutil.rmtree(tier_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    h5_path = tier_dir / f"diffusion_mri_challenge_{tier}.h5"
    rng = np.random.default_rng(base_seed)
    true_specs = {}
    all_psnrs = []
    all_ssims = []

    with h5py.File(h5_path, "w") as f:
        f.attrs["description"] = (
            f"PWM Diffusion MRI benchmark -- {tier} tier. "
            f"Forward model: y = U_Omega * F * (S0 * exp(-b * ADC)) + n. "
            f"Brain slice phantoms with ADC contrast and fiber tracts."
        )
        f.attrs["spec_ranges"] = json.dumps(spec_ranges)
        f.attrs["geometry"] = json.dumps({
            "image_size": IMAGE_SIZE,
            "b_value_nominal": B_VALUE_DEFAULT,
            "fov_mm": 240.0,
            "pixel_size_mm": 240.0 / IMAGE_SIZE,
            "field_strength_T": 3.0,
        })
        f.attrs["forward_model"] = (
            "y = U_Omega * F * (S0 * exp(-b * ADC)) + n"
        )

        for idx, (s0_map, adc_map, scene_name) in enumerate(phantoms):
            key = f"sample_{idx:02d}"
            print(f"  [{tier}] Generating {key} ({scene_name})...",
                  end="", flush=True)

            mis = sample_mismatch(rng, spec_ranges)
            true_specs[key] = mis

            # True b-value = nominal * (1 + error)
            b_value_true = B_VALUE_DEFAULT * (1.0 + mis["b_value_error"])

            # Forward model seed
            fwd_seed = base_seed + idx * 100 + 42

            result = apply_forward_model(
                s0_map, adc_map,
                b_value=b_value_true,
                acceleration=mis["acceleration_factor"],
                noise_sigma=mis["noise_sigma"],
                b_value_error=mis["b_value_error"],
                eddy_current_amp=mis["eddy_current_distortion"],
                seed=fwd_seed,
            )

            x_true = result["x_true"]
            y = result["y"]
            mask = result["H_ideal"]

            # CPU baseline: zero-filled IFFT
            recon = reconstruct_zero_filled(y)

            # Compute metrics
            psnr = compute_psnr(x_true, recon)
            ssim = compute_ssim(x_true, recon)
            all_psnrs.append(psnr)
            all_ssims.append(ssim)

            # Save to HDF5
            grp = f.create_group(key)
            grp.create_dataset("x_true", data=x_true, compression="gzip")
            grp.create_dataset("y", data=y, compression="gzip")
            grp.create_dataset("H_ideal", data=mask, compression="gzip")
            grp.create_dataset("s0_map",
                               data=result["s0_map"], compression="gzip")
            grp.create_dataset("adc_map",
                               data=result["adc_map"], compression="gzip")

            grp.attrs["metadata"] = json.dumps({
                "scene": scene_name,
                "shape": [IMAGE_SIZE, IMAGE_SIZE],
                "b_value_true": float(b_value_true),
                "psnr_zf": float(psnr),
                "ssim_zf": float(ssim),
            })
            grp.attrs["true_spec"] = json.dumps(mis)
            grp.attrs["spec_ranges"] = json.dumps(spec_ranges)

            # Save images
            sample_dir = images_dir / f"sample_{idx:02d}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            _save_png(x_true, sample_dir / "gt.png")
            _save_kspace_png(y, sample_dir / "measurement.png")
            _save_png(recon, sample_dir / "recon.png")
            _save_png(result["adc_map"], sample_dir / "adc_map.png")
            _save_png(mask, sample_dir / "mask.png")

            # Overview composite
            kspace_mag = np.log1p(np.abs(y))
            _save_overview(x_true, kspace_mag, recon, result["adc_map"],
                           sample_dir / "overview.png")

            # Per-sample spec
            with open(sample_dir / "spec.json", "w") as sf:
                json.dump({
                    "scene": scene_name,
                    "spec_ranges": spec_ranges,
                    "true_spec": mis,
                    "psnr_zf": psnr,
                    "ssim_zf": ssim,
                }, sf, indent=2)

            print(f"  PSNR={psnr:.2f} dB, SSIM={ssim:.3f}  "
                  f"accel={mis['acceleration_factor']:.1f}x  "
                  f"noise={mis['noise_sigma']:.3f}  "
                  f"b_err={mis['b_value_error']:.3f}  "
                  f"eddy={mis['eddy_current_distortion']:.3f}")

    # Save tier-level spec files
    with open(tier_dir / "spec.json", "w") as sf:
        json.dump(spec_ranges, sf, indent=2)
    with open(tier_dir / "true_spec.json", "w") as tf:
        json.dump(true_specs, tf, indent=2)

    mean_psnr = np.mean(all_psnrs) if all_psnrs else 0.0
    mean_ssim = np.mean(all_ssims) if all_ssims else 0.0
    print(f"  [{tier}] {len(phantoms)} samples | "
          f"Mean PSNR={mean_psnr:.2f} dB | Mean SSIM={mean_ssim:.3f}")
    print(f"  [{tier}] HDF5 -> {h5_path.name}")

    return h5_path


# ── Gallery Image Generation ─────────────────────────────────────────────────

def generate_gallery_images() -> None:
    """Generate gallery images for platform benchmark page.

    Creates scene_00 through scene_03 with:
        gt.png, measurement_I.png, measurement_II.png,
        recon_I.png, recon_II.png
    """
    gallery_base = (
        BENCHMARK_DIR.parent.parent.parent /
        "platform" / "pwm_platform" / "static" / "img" /
        "benchmark_gallery" / "diffusion_mri"
    )

    h5_path = BENCHMARK_DIR / "public" / "diffusion_mri_challenge_public.h5"
    if not h5_path.exists():
        print("  [gallery] Public HDF5 not found, skipping.")
        return

    # Pick 4 diverse samples: normal/stroke/tumor/ms
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
            mask = grp["H_ideal"][:]
            adc_map = grp["adc_map"][:]

            recon_zf = reconstruct_zero_filled(y)

            # gt.png — ground truth DWI
            _save_png(x_true, scene_dir / "gt.png")

            # measurement_I.png — undersampled k-space (log magnitude)
            _save_kspace_png(y, scene_dir / "measurement_I.png")

            # measurement_II.png — undersampling mask
            _save_png(mask, scene_dir / "measurement_II.png")

            # recon_I.png — zero-filled IFFT reconstruction
            _save_png(recon_zf, scene_dir / "recon_I.png")

            # recon_II.png — ADC map (key diffusion contrast)
            _save_png(adc_map, scene_dir / "recon_II.png")

            print(f"  [gallery] scene_{scene_idx:02d} -> {scene_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 68)
    print("Diffusion MRI Benchmark Dataset Generator")
    print("=" * 68)
    print(f"Output: {BENCHMARK_DIR}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Nominal b-value: {B_VALUE_DEFAULT} s/mm^2\n")

    # ── Public tier (12 samples) ──
    print("Generating public tier (12 samples)...")
    public_phantoms = generate_phantoms_public(12)
    generate_tier("public", public_phantoms, base_seed=0)

    # ── Dev tier (20 samples) ──
    print("\nGenerating dev tier (20 samples)...")
    dev_phantoms = generate_phantoms_dev(20)
    generate_tier("dev", dev_phantoms, base_seed=10000)

    # ── Hidden tier (20 samples) ──
    print("\nGenerating hidden tier (20 samples)...")
    hidden_phantoms = generate_phantoms_hidden(20)
    generate_tier("hidden", hidden_phantoms, base_seed=20000)

    # ── Gallery images ──
    print("\nGenerating gallery images...")
    generate_gallery_images()

    print(f"\n{'=' * 68}")
    print("Diffusion MRI benchmark dataset generation complete!")
    print(f"Output: {BENCHMARK_DIR}")
    print("=" * 68)


if __name__ == "__main__":
    main()
