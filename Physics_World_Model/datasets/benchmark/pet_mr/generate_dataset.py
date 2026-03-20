#!/usr/bin/env python3
"""Generate PET/MR multimodality benchmark dataset.

Combines MRI (anatomical reference) + PET (activity) into a realistic
multimodal inverse problem:
  - MRI provides soft-tissue contrast (proton density / T1-weighted)
  - PET activity is reconstructed using MR-based attenuation correction

Physics:
  y_mr  = M * FFT2(x_mr) + noise_mr        (undersampled k-space)
  y_pet = D(mu_mr) * R * x_pet + noise_pet  (PET with MR-derived attenuation)

  where:
    x_mr   : MR image (proton density / T1 contrast, 256x256)
    M      : Cartesian k-space undersampling mask (random lines)
    FFT2   : 2D Discrete Fourier Transform
    x_pet  : PET activity map (256x256)
    R      : Radon transform (parallel beam, 256 angles)
    D(mu)  : attenuation correction from MR-segmented mu-map
    mu_mr  : MR-derived attenuation map (less accurate than CT-based;
             no direct bone signal in standard MRI sequences)

Key differences from PET/CT:
  - MR provides soft tissue contrast (no direct attenuation measurement)
  - MR-based attenuation estimation is inherently less accurate (bone
    invisible in standard MR, requires atlas-based or UTE methods)
  - y_mr is k-space data (2D FFT), not a sinogram
  - MR undersampling via Cartesian random line mask

Mismatch parameters:
  mr_attenuation_error  : relative error in MR-derived mu-map (higher than CT)
  motion_shift          : inter-modality motion mismatch (pixels)
  b0_inhomogeneity      : off-resonance field in Hz
  pet_scatter_fraction  : scatter background in PET sinogram

Usage:
    cd datasets/benchmark/pet_mr
    python3 generate_dataset.py              # Generate all tiers
    python3 generate_dataset.py --tier public --seed 0
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, Any

import h5py
import numpy as np
from PIL import Image
from scipy.ndimage import (
    rotate as nd_rotate,
    shift as nd_shift,
    gaussian_filter,
    zoom as nd_zoom,
)

# Import radon_transform from PET generator
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pet.generate_dataset import radon_transform, fbp_reconstruct  # noqa: E402

BENCHMARK_DIR = Path(__file__).resolve().parent

# -- Geometry -----------------------------------------------------------------

IMAGE_SIZE = 256
N_ANGLES = 256           # PET projection angles
N_DET = 367              # Radon detector bins for 256x256
MR_ACCEL = 4.0           # MR undersampling factor (base)
MR_CENTER_FRAC = 0.08    # Fraction of center k-space always kept

# -- Mismatch ranges per tier -------------------------------------------------

SPEC = {
    "public": {
        "mr_attenuation_error": {"min": 0.0, "max": 0.08, "unit": "relative"},
        "motion_shift":         {"min": -1.0, "max": 1.0, "unit": "pixels"},
        "b0_inhomogeneity":     {"min": -10.0, "max": 10.0, "unit": "Hz"},
        "pet_scatter_fraction": {"min": 0.0, "max": 0.08, "unit": ""},
    },
    "dev": {
        "mr_attenuation_error": {"min": 0.0, "max": 0.15, "unit": "relative"},
        "motion_shift":         {"min": -3.0, "max": 3.0, "unit": "pixels"},
        "b0_inhomogeneity":     {"min": -25.0, "max": 25.0, "unit": "Hz"},
        "pet_scatter_fraction": {"min": 0.0, "max": 0.15, "unit": ""},
    },
    "hidden": {
        "mr_attenuation_error": {"min": 0.0, "max": 0.25, "unit": "relative"},
        "motion_shift":         {"min": -6.0, "max": 6.0, "unit": "pixels"},
        "b0_inhomogeneity":     {"min": -40.0, "max": 40.0, "unit": "Hz"},
        "pet_scatter_fraction": {"min": 0.0, "max": 0.25, "unit": ""},
    },
}


# -- Phantom generators -------------------------------------------------------

def _ellipse_mask(
    H: int, W: int, cx: float, cy: float,
    a: float, b: float, angle_deg: float = 0.0,
) -> np.ndarray:
    """Binary ellipse mask in normalised [-1,1] coordinates."""
    yy = np.linspace(-1.0, 1.0, H)[:, None]
    xx = np.linspace(-1.0, 1.0, W)[None, :]
    ca, sa = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
    yr = (yy - cy) * ca - (xx - cx) * sa
    xr = (yy - cy) * sa + (xx - cx) * ca
    return (xr / a) ** 2 + (yr / b) ** 2 <= 1.0


def make_brain_phantom(
    H: int, W: int, seed: int, variant: int = 0,
) -> dict:
    """Generate paired brain PET/MR phantoms.

    Returns dict with x_mr, x_pet, mu_map.

    MR image reflects proton density / T1 weighting:
      - CSF: high signal (~0.95)
      - Gray matter: moderate (~0.70)
      - White matter: lower (~0.55)
      - Bone: very low (~0.05, no proton signal in standard MR)
      - Air: zero

    PET image reflects FDG activity:
      - Gray matter: high (~0.90)
      - White matter: moderate (~0.25)
      - Ventricles (CSF): cold (~0.05)
      - Tumours: hot (1.5-3.0)

    mu_map from MR segmentation: soft tissue classification only
    (bone is typically underestimated in MR-based attenuation).
    """
    rng = np.random.default_rng(seed)

    x_mr = np.zeros((H, W), dtype=np.float64)
    x_pet = np.zeros((H, W), dtype=np.float64)
    mu_map = np.zeros((H, W), dtype=np.float64)

    # Skull outline
    skull = _ellipse_mask(H, W, 0.0, 0.0,
                          0.42 + variant * 0.01, 0.48 + variant * 0.005, 0)
    skull_inner = _ellipse_mask(H, W, 0.0, 0.0,
                                0.38 + variant * 0.01,
                                0.44 + variant * 0.005, 0)
    bone = skull & ~skull_inner
    brain = skull_inner.copy()

    # MR: bone is nearly invisible (very low signal)
    x_mr[bone] = 0.05 + rng.uniform(-0.01, 0.01)
    # MR: brain white matter baseline
    x_mr[brain] = 0.55 + rng.uniform(-0.03, 0.03)

    # mu_map: MR-based attenuation (bone UNDERESTIMATED -- key PET/MR challenge)
    # In CT-based mu, bone ~ 0.15 cm^-1 but MR cannot see bone directly
    # MR-based methods assign bone ~ soft tissue attenuation (large error)
    mu_map[bone] = 0.096  # assigned as soft tissue (error: true is ~0.15)
    mu_map[brain] = 0.096  # soft tissue correct

    # PET: white matter baseline
    x_pet[brain] = 0.25 + rng.uniform(-0.02, 0.02)

    # Gray matter cortex
    cortex_outer = _ellipse_mask(H, W, 0.0, 0.0, 0.37, 0.43, 0)
    cortex_inner = _ellipse_mask(H, W, 0.0, 0.0, 0.32, 0.38, 0)
    cortex = cortex_outer & ~cortex_inner & brain
    x_mr[cortex] = 0.70 + rng.uniform(-0.04, 0.04)   # gray matter MR
    x_pet[cortex] = 0.90 + rng.uniform(-0.05, 0.05)   # gray matter PET

    # Deep gray matter structures
    structures = [
        # (cx, cy, a, b, angle, mr_signal, pet_uptake)
        (-0.10, 0.05, 0.04, 0.08, -10, 0.65, 1.0),   # left caudate
        (0.10, 0.05, 0.04, 0.08, 10, 0.65, 1.0),      # right caudate
        (-0.15, -0.02, 0.05, 0.03, 0, 0.62, 0.95),    # left putamen
        (0.15, -0.02, 0.05, 0.03, 0, 0.62, 0.95),     # right putamen
        (-0.06, -0.05, 0.04, 0.03, 0, 0.60, 0.85),    # left thalamus
        (0.06, -0.05, 0.04, 0.03, 0, 0.60, 0.85),     # right thalamus
    ]
    for cx, cy, a, b, ang, mr_sig, pet_up in structures:
        cx += rng.uniform(-0.01, 0.01)
        cy += rng.uniform(-0.01, 0.01)
        mask = _ellipse_mask(H, W, cx, cy, a, b, ang + variant * 3) & brain
        x_mr[mask] = mr_sig + rng.uniform(-0.03, 0.03)
        x_pet[mask] = pet_up + rng.uniform(-0.05, 0.05)

    # Ventricles (CSF): bright on MR (T2/PD), cold on PET
    vent_l = _ellipse_mask(H, W, -0.03, 0.05 + variant * 0.01,
                           0.02, 0.06, -5) & brain
    vent_r = _ellipse_mask(H, W, 0.03, 0.05 + variant * 0.01,
                           0.02, 0.06, 5) & brain
    x_mr[vent_l] = 0.95 + rng.uniform(-0.02, 0.02)
    x_mr[vent_r] = 0.95 + rng.uniform(-0.02, 0.02)
    x_pet[vent_l] = 0.05
    x_pet[vent_r] = 0.05
    mu_map[vent_l] = 0.096  # CSF ~ water
    mu_map[vent_r] = 0.096

    # Tumours (hot on PET, variable on MR)
    n_tumors = rng.integers(1, 4)
    for _ in range(n_tumors):
        tx = rng.uniform(-0.25, 0.25)
        ty = rng.uniform(-0.30, 0.30)
        tr = rng.uniform(0.015, 0.04)
        tumor = _ellipse_mask(H, W, tx, ty, tr,
                              tr * rng.uniform(0.7, 1.3),
                              rng.uniform(-30, 30)) & brain
        if tumor.sum() > 10:
            x_pet[tumor] = rng.uniform(1.5, 3.0)
            # MR: tumours often hypo- or hyperintense
            x_mr[tumor] = rng.uniform(0.50, 0.85)

    # Smooth both images
    x_mr = gaussian_filter(x_mr, sigma=1.0)
    x_pet = gaussian_filter(x_pet, sigma=1.0)

    # Normalise
    x_mr = np.clip(x_mr, 0.0, None)
    if x_mr.max() > 0:
        x_mr /= x_mr.max()
    x_pet = np.clip(x_pet, 0.0, None)
    if x_pet.max() > 0:
        x_pet /= x_pet.max()

    return {
        "x_mr": x_mr,
        "x_pet": x_pet,
        "mu_map": mu_map,
        "scene_name": f"brain_{variant:02d}",
    }


def make_body_phantom(
    H: int, W: int, seed: int, variant: int = 0,
) -> dict:
    """Generate paired body (torso) PET/MR phantoms."""
    rng = np.random.default_rng(seed)

    x_mr = np.zeros((H, W), dtype=np.float64)
    x_pet = np.zeros((H, W), dtype=np.float64)
    mu_map = np.zeros((H, W), dtype=np.float64)

    # Body outline
    body = _ellipse_mask(H, W, 0.0, 0.0, 0.40, 0.32, 0)
    x_mr[body] = 0.40 + rng.uniform(-0.03, 0.03)   # soft tissue
    x_pet[body] = 0.15                                # background uptake
    mu_map[body] = 0.096

    # Spine (bone: invisible in MR, but high attenuation)
    spine = _ellipse_mask(H, W, 0.0, 0.22, 0.04, 0.03, 0) & body
    x_mr[spine] = 0.05   # bone: nearly invisible in MR
    x_pet[spine] = 0.05
    # MR-based mu: bone misclassified as soft tissue
    mu_map[spine] = 0.096  # should be ~0.15

    # Lungs (low signal in MR due to air)
    lung_l = _ellipse_mask(H, W, -0.18, -0.02 + variant * 0.01,
                           0.12, 0.18, -5) & body
    lung_r = _ellipse_mask(H, W, 0.18, -0.02 + variant * 0.01,
                           0.12, 0.18, 5) & body
    for lung in [lung_l, lung_r]:
        x_mr[lung] = 0.03    # very low MR signal (air)
        x_pet[lung] = 0.03
        mu_map[lung] = 0.022  # inflated lung

    # Heart
    heart = _ellipse_mask(H, W, -0.05, -0.03, 0.08, 0.07,
                          15 + variant * 5) & body
    heart_inner = _ellipse_mask(H, W, -0.05, -0.03, 0.05, 0.04,
                                15 + variant * 5) & body
    myocardium = heart & ~heart_inner
    x_mr[myocardium] = 0.65 + rng.uniform(-0.05, 0.05)
    x_pet[myocardium] = 0.70 + rng.uniform(-0.05, 0.10)
    x_mr[heart_inner] = 0.80  # blood pool (bright in MR)
    x_pet[heart_inner] = 0.20

    # Liver
    liver = _ellipse_mask(H, W, 0.15, 0.08, 0.15, 0.10, -10) & body
    liver &= ~lung_r
    x_mr[liver] = 0.55 + rng.uniform(-0.04, 0.04)
    x_pet[liver] = 0.45 + rng.uniform(-0.05, 0.05)
    mu_map[liver] = 0.098

    # Kidneys
    kidney_l = _ellipse_mask(H, W, -0.15, 0.12, 0.04, 0.06, -10) & body
    kidney_r = _ellipse_mask(H, W, 0.15, 0.12, 0.04, 0.06, 10) & body
    x_mr[kidney_l] = 0.60
    x_mr[kidney_r] = 0.60
    x_pet[kidney_l] = 0.55
    x_pet[kidney_r] = 0.55

    # Lesions
    n_tumors = rng.integers(1, 5)
    for _ in range(n_tumors):
        tx = rng.uniform(-0.30, 0.30)
        ty = rng.uniform(-0.25, 0.25)
        tr = rng.uniform(0.01, 0.035)
        tumor = _ellipse_mask(H, W, tx, ty, tr,
                              tr * rng.uniform(0.6, 1.4),
                              rng.uniform(-45, 45)) & body
        if tumor.sum() > 5:
            x_pet[tumor] = rng.uniform(1.5, 3.5)
            x_mr[tumor] = rng.uniform(0.45, 0.80)

    # Smooth
    x_mr = gaussian_filter(x_mr, sigma=0.8)
    x_pet = gaussian_filter(x_pet, sigma=0.8)
    x_mr = np.clip(x_mr, 0.0, None)
    x_pet = np.clip(x_pet, 0.0, None)
    if x_mr.max() > 0:
        x_mr /= x_mr.max()
    if x_pet.max() > 0:
        x_pet /= x_pet.max()

    return {
        "x_mr": x_mr,
        "x_pet": x_pet,
        "mu_map": mu_map,
        "scene_name": f"body_{variant:02d}",
    }


def make_cardiac_phantom(
    H: int, W: int, seed: int, variant: int = 0,
) -> dict:
    """Generate paired cardiac PET/MR phantoms."""
    rng = np.random.default_rng(seed)

    x_mr = np.zeros((H, W), dtype=np.float64)
    x_pet = np.zeros((H, W), dtype=np.float64)
    mu_map = np.zeros((H, W), dtype=np.float64)

    # Chest outline
    chest = _ellipse_mask(H, W, 0.0, 0.0, 0.42, 0.35, 0)
    x_mr[chest] = 0.40
    x_pet[chest] = 0.10
    mu_map[chest] = 0.096

    # Lungs
    lung_l = _ellipse_mask(H, W, -0.20, -0.02, 0.13, 0.20, -3) & chest
    lung_r = _ellipse_mask(H, W, 0.20, -0.02, 0.13, 0.20, 3) & chest
    for lung in [lung_l, lung_r]:
        x_mr[lung] = 0.03
        x_pet[lung] = 0.03
        mu_map[lung] = 0.022

    # Myocardium (prominent ring)
    heart_cx = -0.05 + variant * 0.01
    heart_cy = -0.02
    heart_outer = _ellipse_mask(H, W, heart_cx, heart_cy,
                                0.12 + variant * 0.005, 0.11, 10)
    heart_inner = _ellipse_mask(H, W, heart_cx, heart_cy,
                                0.07 + variant * 0.003, 0.06, 10)
    myocardium = (heart_outer & ~heart_inner) & chest
    blood_pool = heart_inner & chest

    base_myo = 0.90 + rng.uniform(-0.05, 0.05)
    x_mr[myocardium] = 0.65 + rng.uniform(-0.04, 0.04)
    x_pet[myocardium] = base_myo
    x_mr[blood_pool] = 0.85   # bright blood in MR
    x_pet[blood_pool] = 0.25

    # Perfusion defects
    n_defects = rng.integers(0, 3)
    for _ in range(n_defects):
        angle_start = rng.uniform(0, 360)
        angle_span = rng.uniform(30, 90)
        y_coords, x_coords = np.where(myocardium)
        ctr_y = int((heart_cy + 1.0) * H / 2)
        ctr_x = int((heart_cx + 1.0) * W / 2)
        angles = np.degrees(np.arctan2(y_coords - ctr_y,
                                       x_coords - ctr_x)) % 360
        in_sector = ((angles >= angle_start) &
                     (angles < angle_start + angle_span))
        if in_sector.sum() > 0:
            x_pet[y_coords[in_sector], x_coords[in_sector]] = \
                rng.uniform(0.30, 0.65)
            # Perfusion defect may also show on MR (edema)
            x_mr[y_coords[in_sector], x_coords[in_sector]] = \
                rng.uniform(0.55, 0.75)

    # Liver
    liver = _ellipse_mask(H, W, 0.10, 0.20, 0.20, 0.08, -5) & chest
    x_mr[liver] = 0.55
    x_pet[liver] = 0.50

    # Spine
    spine = _ellipse_mask(H, W, 0.0, 0.28, 0.04, 0.03, 0) & chest
    x_mr[spine] = 0.05   # bone invisible in MR
    x_pet[spine] = 0.05
    mu_map[spine] = 0.096  # MR-based: bone misclassified

    # Smooth and normalise
    x_mr = gaussian_filter(x_mr, sigma=0.8)
    x_pet = gaussian_filter(x_pet, sigma=0.8)
    x_mr = np.clip(x_mr, 0.0, None)
    x_pet = np.clip(x_pet, 0.0, None)
    if x_mr.max() > 0:
        x_mr /= x_mr.max()
    if x_pet.max() > 0:
        x_pet /= x_pet.max()

    return {
        "x_mr": x_mr,
        "x_pet": x_pet,
        "mu_map": mu_map,
        "scene_name": f"cardiac_{variant:02d}",
    }


# -- Phantom pool per tier ----------------------------------------------------

PHANTOM_GENERATORS = [make_brain_phantom, make_body_phantom,
                      make_cardiac_phantom]


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


def generate_phantoms_public(n: int = 12) -> list[dict]:
    """Diverse public phantoms: 4 brain + 4 body + 4 cardiac."""
    phantoms = []
    for i in range(4):
        phantoms.append(make_brain_phantom(IMAGE_SIZE, IMAGE_SIZE,
                                           seed=100 + i, variant=i))
    for i in range(4):
        phantoms.append(make_body_phantom(IMAGE_SIZE, IMAGE_SIZE,
                                          seed=200 + i, variant=i))
    for i in range(4):
        phantoms.append(make_cardiac_phantom(IMAGE_SIZE, IMAGE_SIZE,
                                              seed=300 + i, variant=i))
    return phantoms[:n]


def generate_phantoms_dev(n: int = 20) -> list[dict]:
    """Dev-tier phantoms with augmented diversity."""
    phantoms = []
    rng = np.random.default_rng(15000)
    for i in range(n):
        gen_fn = PHANTOM_GENERATORS[i % 3]
        data = gen_fn(IMAGE_SIZE, IMAGE_SIZE, seed=1500 + i, variant=i)
        # Augment: rotation + flip + zoom
        angle = float(rng.uniform(15, 345))
        for key in ("x_mr", "x_pet", "mu_map"):
            data[key] = nd_rotate(data[key], angle, reshape=False,
                                  mode="constant", cval=0.0)
        if rng.random() < 0.5:
            for key in ("x_mr", "x_pet", "mu_map"):
                data[key] = np.fliplr(data[key]).copy()
        zoom_f = float(rng.uniform(0.85, 1.15))
        if abs(zoom_f - 1.0) > 0.02:
            for key in ("x_mr", "x_pet", "mu_map"):
                data[key] = _zoom_crop(data[key], zoom_f, IMAGE_SIZE)
        # Re-normalise
        for key in ("x_mr", "x_pet"):
            data[key] = np.clip(data[key], 0.0, None)
            if data[key].max() > 0:
                data[key] /= data[key].max()
        data["scene_name"] = f"dev_{data['scene_name']}"
        phantoms.append(data)
    return phantoms


def generate_phantoms_hidden(n: int = 20) -> list[dict]:
    """Hidden-tier phantoms with adversarial modifications."""
    phantoms = []
    rng = np.random.default_rng(28000)
    for i in range(n):
        gen_fn = PHANTOM_GENERATORS[i % 3]
        data = gen_fn(IMAGE_SIZE, IMAGE_SIZE, seed=2800 + i, variant=i + 10)
        # Adversarial augmentation
        angle = float(rng.uniform(20, 340))
        for key in ("x_mr", "x_pet", "mu_map"):
            data[key] = nd_rotate(data[key], angle, reshape=False,
                                  mode="constant", cval=0.0)
        if rng.random() < 0.7:
            for key in ("x_mr", "x_pet", "mu_map"):
                data[key] = np.fliplr(data[key]).copy()
        if rng.random() < 0.5:
            for key in ("x_mr", "x_pet", "mu_map"):
                data[key] = np.flipud(data[key]).copy()
        zoom_f = float(rng.uniform(0.70, 1.30))
        for key in ("x_mr", "x_pet", "mu_map"):
            data[key] = _zoom_crop(data[key], zoom_f, IMAGE_SIZE)

        # Add subtle micro-lesions in PET (hard to detect)
        n_micro = rng.integers(2, 6)
        for _ in range(n_micro):
            cy = rng.integers(40, IMAGE_SIZE - 40)
            cx = rng.integers(40, IMAGE_SIZE - 40)
            r = rng.integers(2, 6)
            yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
            circle = (yy ** 2 + xx ** 2 <= r ** 2).astype(np.float64)
            y0, y1 = max(0, cy - r), min(IMAGE_SIZE, cy + r + 1)
            x0, x1 = max(0, cx - r), min(IMAGE_SIZE, cx + r + 1)
            c_y0, c_y1 = r - (cy - y0), r + (y1 - cy)
            c_x0, c_x1 = r - (cx - x0), r + (x1 - cx)
            if data["x_pet"][y0:y1, x0:x1].mean() > 0.1:
                intensity = rng.uniform(1.5, 4.0)
                data["x_pet"][y0:y1, x0:x1] = np.maximum(
                    data["x_pet"][y0:y1, x0:x1],
                    circle[c_y0:c_y1, c_x0:c_x1] * intensity,
                )

        # Re-normalise
        for key in ("x_mr", "x_pet"):
            data[key] = np.clip(data[key], 0.0, None)
            if data[key].max() > 0:
                data[key] /= data[key].max()
        data["scene_name"] = f"hidden_{data['scene_name']}"
        phantoms.append(data)
    return phantoms


# -- MR Forward Model ---------------------------------------------------------

def generate_cartesian_mask(
    H: int, W: int, acceleration: float,
    center_fraction: float = MR_CENTER_FRAC, seed: int = 42,
) -> np.ndarray:
    """Generate Cartesian (line) k-space undersampling mask."""
    rng = np.random.RandomState(seed)
    mask = np.zeros((H, W), dtype=np.float32)

    n_center = max(1, int(W * center_fraction))
    c = W // 2
    mask[:, c - n_center // 2: c + n_center // 2] = 1.0

    n_total_lines = max(n_center, int(W / acceleration))
    n_random = n_total_lines - n_center
    available = [i for i in range(W)
                 if i < c - n_center // 2 or i >= c + n_center // 2]
    if n_random > 0 and len(available) > 0:
        chosen = rng.choice(available,
                            size=min(n_random, len(available)),
                            replace=False)
        mask[:, chosen] = 1.0
    return mask


def mr_forward_model(
    x_mr: np.ndarray,
    acceleration: float,
    noise_sigma: float,
    b0_hz: float,
    seed: int,
) -> dict:
    """MR forward model: y_mr = M * FFT2(x_mr) + noise.

    Args:
        x_mr:         (H, W) real MR image [0, 1]
        acceleration: k-space undersampling factor
        noise_sigma:  additive complex Gaussian noise std
        b0_hz:        off-resonance field (Hz) for B0 inhomogeneity
        seed:         random seed

    Returns:
        dict with kspace_full, kspace_undersampled, mask
    """
    rng = np.random.RandomState(seed)
    H, W = x_mr.shape

    # Convert to complex image (proton density model, zero phase)
    img_complex = x_mr.astype(np.complex128)

    # Apply B0 inhomogeneity (smooth spatial phase modulation)
    if abs(b0_hz) > 0.1:
        y_coords = np.linspace(-1, 1, H)
        x_coords = np.linspace(-1, 1, W)
        yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")
        b0_map = b0_hz * (0.3 * yy + 0.2 * xx + 0.1 * yy * xx)
        te = 0.020  # echo time ~ 20 ms
        phase_map = np.exp(1j * 2 * np.pi * b0_map * te)
        img_complex = img_complex * phase_map

    # FFT2 to k-space (ortho convention)
    kspace_full = np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(img_complex))
    ) / np.sqrt(H * W)

    # Undersampling mask
    mask = generate_cartesian_mask(H, W, acceleration, seed=seed)

    # Undersampled k-space
    kspace_under = kspace_full * mask

    # Add complex Gaussian noise
    noise = (rng.randn(*kspace_under.shape) +
             1j * rng.randn(*kspace_under.shape)) * noise_sigma / np.sqrt(2)
    kspace_under = kspace_under + noise * mask

    return {
        "kspace_full": kspace_full.astype(np.complex64),
        "kspace_undersampled": kspace_under.astype(np.complex64),
        "mask": mask,
    }


def mr_reconstruct_zero_filled(kspace_under: np.ndarray) -> np.ndarray:
    """Zero-filled IFFT reconstruction. Returns magnitude image."""
    H, W = kspace_under.shape
    img = np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(kspace_under))
    ) * np.sqrt(H * W)
    return np.abs(img).astype(np.float32)


# -- PET Forward Model (MR-based attenuation) ---------------------------------

def pet_forward_with_mr_attenuation(
    x_pet: np.ndarray,
    mu_map: np.ndarray,
    theta_deg: np.ndarray,
    scatter_fraction: float,
    attenuation_error: float,
    rng: np.random.Generator,
) -> dict:
    """PET forward model with MR-derived attenuation correction.

    y_pet = D(mu_mr) * R * x_pet + scatter + Poisson noise

    MR-based attenuation is inherently less accurate than CT-based:
    - Bone is nearly invisible in standard MR sequences
    - Atlas/template-based methods introduce registration errors
    - The attenuation_error parameter models these combined effects

    Args:
        x_pet:             (H, W) PET activity map [0, 1]
        mu_map:            (H, W) MR-derived attenuation map [cm^-1]
        theta_deg:         projection angles in degrees
        scatter_fraction:  scatter / total fraction
        attenuation_error: relative error in MR-derived mu-map
        rng:               random generator

    Returns:
        dict with sinogram data and correction factors
    """
    # 1. Ideal sinogram
    sino_ideal = radon_transform(x_pet, theta_deg)
    sino_ideal = np.maximum(sino_ideal, 0.0)

    # 2. Attenuation factors from MR-derived mu-map
    sino_mu = radon_transform(mu_map, theta_deg)
    pixel_size_cm = 22.0 / IMAGE_SIZE  # ~0.086 cm
    sino_mu_physical = sino_mu * pixel_size_cm
    attn_factors_true = np.exp(-sino_mu_physical)

    # 3. Apply additional MR-specific attenuation error
    # This models: bone underestimation, air/tissue misclassification,
    # atlas registration errors, etc.
    if attenuation_error > 0:
        mu_map_err = mu_map * (1.0 + rng.uniform(
            -attenuation_error, attenuation_error, mu_map.shape))
        # Also add structured error (bone regions get extra error)
        # Bone pixels have mu~0.096 in MR-based (should be 0.15)
        # This is already built into the phantom; the error adds noise on top
    else:
        mu_map_err = mu_map.copy()
    sino_mu_err = radon_transform(mu_map_err, theta_deg) * pixel_size_cm
    attn_factors_used = np.exp(-sino_mu_err)

    # 4. Scale to physical count level
    total_counts = 2e6  # 2 Mcps baseline
    if sino_ideal.sum() > 0:
        scale = total_counts / sino_ideal.sum()
    else:
        scale = 1.0
    sino_trues = sino_ideal * scale

    # 5. Attenuated trues
    sino_atten = sino_trues * attn_factors_true

    # 6. Scatter (smooth background)
    mean_signal = max(sino_atten.mean(), 1.0)
    denom = max(1.0 - scatter_fraction, 0.1)
    scatter_level = scatter_fraction / denom * mean_signal
    scatter = np.ones_like(sino_atten) * scatter_level
    scatter_var = rng.standard_normal(sino_atten.shape) * scatter_level * 0.1
    scatter += gaussian_filter(scatter_var, sigma=[5.0, 10.0])
    scatter = np.maximum(scatter, 0.0)

    # 7. Expected counts and Poisson sampling
    expected = sino_atten + scatter
    expected = np.maximum(expected, 0.01)
    sino_measured = rng.poisson(expected).astype(np.float64)

    return {
        "sinogram_ideal": sino_ideal.astype(np.float32),
        "sinogram_measured": sino_measured.astype(np.float32),
        "attenuation_factors": attn_factors_true.astype(np.float32),
        "attenuation_factors_used": attn_factors_used.astype(np.float32),
        "scatter": scatter.astype(np.float32),
        "expected_counts": expected.astype(np.float32),
        "scale_factor": float(scale),
    }


# -- Metrics -------------------------------------------------------------------

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
    mu_x, mu_y = gt.mean(), recon.mean()
    var_x, var_y = gt.var(), recon.var()
    cov_xy = np.mean((gt - mu_x) * (recon - mu_y))
    ssim = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / \
           ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2))
    return float(ssim)


# -- Image helpers -------------------------------------------------------------

def _norm(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo + 1e-8)


def _save_png(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(
        np.clip(_norm(arr) * 255, 0, 255).astype(np.uint8), "L"
    ).save(str(path))


def _save_kspace_png(kspace: np.ndarray, path: Path) -> None:
    """Save k-space magnitude (log scale)."""
    mag = np.abs(kspace)
    log_mag = np.log1p(mag)
    _save_png(log_mag, path)


# -- Mismatch sampling --------------------------------------------------------

def sample_mismatch(rng: np.random.Generator, spec: dict) -> dict:
    return {k: float(rng.uniform(v["min"], v["max"]))
            for k, v in spec.items()}


# -- Tier generation -----------------------------------------------------------

def generate_tier(
    tier: str,
    phantoms: list[dict],
    base_seed: int,
) -> Path:
    """Generate one tier of the PET/MR benchmark.

    Returns path to the output HDF5 file.
    """
    spec_ranges = SPEC[tier]
    tier_dir = BENCHMARK_DIR / tier
    images_dir = tier_dir / "images"
    if tier_dir.exists():
        shutil.rmtree(tier_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    theta_deg = np.linspace(0, 180, N_ANGLES, endpoint=False).astype(np.float64)

    h5_path = tier_dir / f"pet_mr_challenge_{tier}.h5"
    rng = np.random.default_rng(base_seed)
    true_specs: dict = {}
    all_psnr_mr: list[float] = []
    all_psnr_pet: list[float] = []

    with h5py.File(h5_path, "w") as f:
        f.attrs["description"] = (
            f"PWM PET/MR benchmark -- {tier} tier. "
            f"Joint MR k-space + PET sinogram with MR-based attenuation. "
            f"Forward model: y_mr = M*FFT2(x_mr) + n, "
            f"y_pet = D(mu_mr)*R*x_pet + scatter + Poisson"
        )
        f.attrs["spec_ranges"] = json.dumps(spec_ranges)
        f.attrs["geometry"] = json.dumps({
            "image_size": IMAGE_SIZE,
            "n_angles_pet": N_ANGLES,
            "n_det_pet": N_DET,
            "mr_acceleration_base": MR_ACCEL,
            "angle_range_deg": [0, 180],
            "fov_mm": 220.0,
            "pixel_size_mm": 220.0 / IMAGE_SIZE,
        })
        f.attrs["forward_model"] = (
            "y_mr = M * FFT2(x_mr) + noise_mr; "
            "y_pet = D(mu_mr) * R * x_pet + scatter + Poisson"
        )

        for idx, phantom in enumerate(phantoms):
            key = f"sample_{idx:02d}"
            scene_name = phantom["scene_name"]
            print(f"  [{tier}] {key} ({scene_name})...", end="", flush=True)

            x_mr = phantom["x_mr"].copy()
            x_pet = phantom["x_pet"].copy()
            mu_map = phantom["mu_map"].copy()

            # Sample mismatch parameters
            mis = sample_mismatch(rng, spec_ranges)
            true_specs[key] = mis

            # Apply inter-modality motion shift (MR acquired at different time)
            shift_px = mis["motion_shift"]
            if abs(shift_px) > 0.1:
                # Shift the MR image relative to PET (misregistration)
                x_mr_shifted = nd_shift(x_mr, (shift_px, shift_px),
                                        order=1, mode="constant", cval=0.0)
            else:
                x_mr_shifted = x_mr.copy()

            # --- MR Forward Model ---
            # Acceleration varies slightly per sample
            mr_accel = MR_ACCEL + rng.uniform(-0.5, 0.5)
            mr_noise_sigma = rng.uniform(0.002, 0.010)
            mr_seed = base_seed + idx * 100 + 42

            mr_result = mr_forward_model(
                x_mr_shifted,
                acceleration=mr_accel,
                noise_sigma=mr_noise_sigma,
                b0_hz=mis["b0_inhomogeneity"],
                seed=mr_seed,
            )

            # MR zero-filled reconstruction for quality check
            recon_mr = mr_reconstruct_zero_filled(
                mr_result["kspace_undersampled"])
            gt_mr = np.abs(x_mr_shifted).astype(np.float32)
            # Normalise for metrics
            gt_max = gt_mr.max() if gt_mr.max() > 0 else 1.0
            psnr_mr = compute_psnr(gt_mr / gt_max, recon_mr / gt_max)
            all_psnr_mr.append(psnr_mr)

            # --- PET Forward Model (MR-based attenuation) ---
            pet_result = pet_forward_with_mr_attenuation(
                x_pet, mu_map, theta_deg,
                scatter_fraction=mis["pet_scatter_fraction"],
                attenuation_error=mis["mr_attenuation_error"],
                rng=rng,
            )

            # PET FBP reconstruction
            sino_meas = pet_result["sinogram_measured"]
            attn_corr = pet_result["attenuation_factors_used"]
            attn_safe = np.where(attn_corr > 0.01, attn_corr, 0.01)
            sino_corrected = (sino_meas - pet_result["scatter"])
            sino_corrected = np.maximum(sino_corrected, 0.0) / attn_safe
            if pet_result["scale_factor"] > 0:
                sino_corrected /= pet_result["scale_factor"]
            recon_pet = fbp_reconstruct(sino_corrected, theta_deg, IMAGE_SIZE)
            recon_pet = np.maximum(recon_pet, 0.0).astype(np.float32)
            psnr_pet = compute_psnr(x_pet, recon_pet)
            all_psnr_pet.append(psnr_pet)

            # --- Save to HDF5 ---
            grp = f.create_group(key)

            # Ground truth
            grp.create_dataset("x_mr", data=x_mr.astype(np.float32),
                               compression="gzip")
            grp.create_dataset("x_pet", data=x_pet.astype(np.float32),
                               compression="gzip")

            # Measurements
            # y_mr: undersampled k-space (complex)
            grp.create_dataset("y_mr",
                               data=mr_result["kspace_undersampled"],
                               compression="gzip")
            # y_pet: measured sinogram
            grp.create_dataset("y_pet",
                               data=pet_result["sinogram_measured"],
                               compression="gzip")

            # Supplementary data for reconstruction
            grp.create_dataset("mr_mask", data=mr_result["mask"],
                               compression="gzip")
            grp.create_dataset("mr_kspace_full",
                               data=mr_result["kspace_full"],
                               compression="gzip")
            grp.create_dataset("mu_map",
                               data=mu_map.astype(np.float32),
                               compression="gzip")
            grp.create_dataset("pet_attenuation_factors",
                               data=pet_result["attenuation_factors_used"],
                               compression="gzip")
            grp.create_dataset("pet_scatter_estimate",
                               data=pet_result["scatter"],
                               compression="gzip")
            grp.create_dataset("pet_angles_deg",
                               data=theta_deg.astype(np.float32))

            # Metadata
            grp.attrs["metadata"] = json.dumps({
                "scene": scene_name,
                "shape": [IMAGE_SIZE, IMAGE_SIZE],
                "n_angles_pet": N_ANGLES,
                "mr_acceleration": float(mr_accel),
                "mr_noise_sigma": float(mr_noise_sigma),
                "psnr_mr_zf": float(psnr_mr),
                "psnr_pet_fbp": float(psnr_pet),
            })
            grp.attrs["true_spec"] = json.dumps(mis)
            grp.attrs["spec_ranges"] = json.dumps(spec_ranges)

            # --- Save preview images ---
            sample_dir = images_dir / f"sample_{idx:02d}_{scene_name}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            _save_png(x_mr, sample_dir / "x_mr_gt.png")
            _save_png(x_pet, sample_dir / "x_pet_gt.png")
            _save_kspace_png(mr_result["kspace_undersampled"],
                             sample_dir / "kspace_undersampled.png")
            _save_png(pet_result["sinogram_measured"],
                      sample_dir / "sinogram_measured.png")
            _save_png(recon_mr, sample_dir / "recon_mr_zf.png")
            _save_png(recon_pet, sample_dir / "recon_pet_fbp.png")
            _save_png(mr_result["mask"], sample_dir / "mr_mask.png")

            with open(sample_dir / "spec.json", "w") as sf:
                json.dump({"scene": scene_name, "spec_ranges": spec_ranges,
                           "true_spec": mis,
                           "psnr_mr_zf": psnr_mr,
                           "psnr_pet_fbp": psnr_pet}, sf, indent=2)

            print(f"  MR-PSNR={psnr_mr:.2f} dB, PET-PSNR={psnr_pet:.2f} dB  "
                  f"attn_err={mis['mr_attenuation_error']:.3f}  "
                  f"shift={mis['motion_shift']:.1f}px  "
                  f"b0={mis['b0_inhomogeneity']:.1f}Hz  "
                  f"scatter={mis['pet_scatter_fraction']:.3f}")

    # Save tier spec files
    with open(tier_dir / "spec.json", "w") as sf:
        json.dump(spec_ranges, sf, indent=2)
    with open(tier_dir / "true_spec.json", "w") as tf:
        json.dump(true_specs, tf, indent=2)

    mean_psnr_mr = np.mean(all_psnr_mr) if all_psnr_mr else 0
    mean_psnr_pet = np.mean(all_psnr_pet) if all_psnr_pet else 0
    print(f"  [{tier}] {len(phantoms)} samples | "
          f"Mean MR-PSNR={mean_psnr_mr:.2f} dB | "
          f"Mean PET-PSNR={mean_psnr_pet:.2f} dB")
    print(f"  [{tier}] HDF5 -> {h5_path.name}")

    return h5_path


# -- Gallery images ------------------------------------------------------------

def generate_gallery_images() -> None:
    """Generate gallery images for the platform benchmark page."""
    gallery_base = (BENCHMARK_DIR.parent.parent.parent /
                    "platform" / "pwm_platform" / "static" / "img" /
                    "benchmark_gallery" / "pet_mr")

    h5_path = BENCHMARK_DIR / "public" / "pet_mr_challenge_public.h5"
    if not h5_path.exists():
        print("  [gallery] Public HDF5 not found, skipping gallery.")
        return

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
            x_mr = grp["x_mr"][:]
            x_pet = grp["x_pet"][:]
            y_mr = grp["y_mr"][:]
            y_pet = grp["y_pet"][:]

            # gt.png -- MR ground truth
            _save_png(x_mr, scene_dir / "gt.png")

            # measurement_I.png -- MR k-space (log magnitude)
            _save_kspace_png(y_mr, scene_dir / "measurement_I.png")

            # measurement_II.png -- PET sinogram
            _save_png(np.abs(y_pet), scene_dir / "measurement_II.png")

            # recon_I.png -- PET ground truth
            _save_png(x_pet, scene_dir / "recon_I.png")

            # recon_II.png -- MR zero-filled reconstruction
            recon_mr = mr_reconstruct_zero_filled(y_mr)
            _save_png(recon_mr, scene_dir / "recon_II.png")

            # recon_III.png -- mu-map
            if "mu_map" in grp:
                _save_png(grp["mu_map"][:], scene_dir / "recon_III.png")

            print(f"  [gallery] scene_{scene_idx:02d} saved to {scene_dir}")


# -- Main ---------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate PET/MR multimodality benchmark dataset.")
    parser.add_argument("--tier", type=str, default="all",
                        help="public|dev|hidden|all")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed base")
    args = parser.parse_args()

    print("=" * 68)
    print("PET/MR Multimodality Benchmark Dataset Generator")
    print("=" * 68)
    print(f"Output: {BENCHMARK_DIR}")
    print(f"Geometry: PET {N_ANGLES} angles, MR {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"MR forward: y_mr = M * FFT2(x_mr) + noise")
    print(f"PET forward: y_pet = D(mu_mr) * R * x_pet + scatter + Poisson")
    print()

    tiers = ["public", "dev", "hidden"] if args.tier == "all" else [args.tier]
    seed_offsets = {"public": 0, "dev": 10000, "hidden": 20000}

    h5_paths: list[Path] = []

    for tier in tiers:
        seed_base = args.seed + seed_offsets.get(tier, 0)

        if tier == "public":
            phantoms = generate_phantoms_public(12)
        elif tier == "dev":
            phantoms = generate_phantoms_dev(20)
        else:
            phantoms = generate_phantoms_hidden(20)

        print(f"\nGenerating {tier} tier ({len(phantoms)} samples)...")
        h5_path = generate_tier(tier, phantoms, base_seed=seed_base)
        h5_paths.append(h5_path)

    # Gallery images
    print("\nGenerating gallery images...")
    generate_gallery_images()

    print(f"\n{'=' * 68}")
    print("PET/MR benchmark dataset generation complete!")
    for p in h5_paths:
        print(f"  {p}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
