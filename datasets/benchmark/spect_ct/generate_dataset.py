#!/usr/bin/env python3
"""Generate SPECT/CT multimodality benchmark dataset.

Combines CT (attenuation map) + SPECT (activity) into realistic multimodal problem:
- CT provides anatomical reference and attenuation correction map (mu_map)
- SPECT activity is reconstructed using CT-based attenuation correction

Physics:
  y_ct   = Radon(x_ct) + Poisson noise                          (CT sinogram)
  y_spect = D(mu_map) * Radon_blur(x_spect) + scatter + Poisson (SPECT sinogram)

  where D(mu_map) is the depth-dependent attenuation factor derived from the CT,
  Radon_blur includes depth-dependent collimator-detector response (CDR) blur,
  and scatter is energy-dependent spatially-smooth background.

Key differences from PET/CT:
  1. SPECT uses wider collimator PSF (depth-dependent Gaussian blur along detector)
  2. SPECT has energy-dependent scatter (higher scatter fraction than PET)
  3. SPECT typically has lower count statistics (more Poisson noise)
  4. SPECT uses 128 angles (vs 180 for PET) over full 360-degree rotation

Mismatch parameters:
  ct_registration_shift : spatial shift between CT and SPECT (pixels)
  hu_to_mu_scale        : scaling factor for CT->mu conversion
  scatter_fraction      : scatter / total ratio in SPECT sinogram
  collimator_blur       : collimator PSF FWHM in pixels

Phantoms:
  Cardiac perfusion (Tc-99m sestamibi) -- primary SPECT application
  Brain perfusion (Tc-99m HMPAO)
  Bone scan (Tc-99m MDP)

Tiers:
  Public  : 12 samples (4 cardiac + 4 brain + 4 bone)  seed offset 0
  Dev     : 20 samples (augmented variants)             seed offset 10000
  Hidden  : 20 samples (adversarial: subtle defects)    seed offset 20000

Usage:
    cd datasets/benchmark/spect_ct
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
    gaussian_filter,
    zoom as nd_zoom,
    shift as nd_shift,
)

# Import radon_transform from the PET generator
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pet.generate_dataset import radon_transform  # noqa: E402

BENCHMARK_DIR = Path(__file__).resolve().parent

# -- Geometry -----------------------------------------------------------------

IMAGE_SIZE = 256
N_ANGLES_SPECT = 128       # SPECT: fewer angles than PET (128 vs 180/256)
N_ANGLES_CT = 180           # CT: standard 180 angles
ANGLE_RANGE_SPECT = 360.0   # SPECT: full 360-degree rotation
ANGLE_RANGE_CT = 180.0       # CT: 0-180 degrees

# Physical scale
FOV_MM = 400.0                          # body SPECT typical
PIXEL_SIZE_MM = FOV_MM / IMAGE_SIZE     # ~1.56 mm/px

# -- Seed offsets per tier (from project convention) --------------------------
TIER_SEED_OFFSETS = {"public": 0, "dev": 10000, "hidden": 20000}

# -- Mismatch ranges per tier -------------------------------------------------

SPEC = {
    "public": {
        "ct_registration_shift": {"min": -1, "max": 1, "unit": "pixels"},
        "hu_to_mu_scale": {"min": 0.97, "max": 1.03, "unit": "relative"},
        "scatter_fraction": {"min": 0.10, "max": 0.20, "unit": ""},
        "collimator_blur": {"min": 2.0, "max": 3.5, "unit": "pixels FWHM"},
    },
    "dev": {
        "ct_registration_shift": {"min": -3, "max": 3, "unit": "pixels"},
        "hu_to_mu_scale": {"min": 0.93, "max": 1.07, "unit": "relative"},
        "scatter_fraction": {"min": 0.08, "max": 0.28, "unit": ""},
        "collimator_blur": {"min": 1.5, "max": 4.5, "unit": "pixels FWHM"},
    },
    "hidden": {
        "ct_registration_shift": {"min": -5, "max": 5, "unit": "pixels"},
        "hu_to_mu_scale": {"min": 0.88, "max": 1.12, "unit": "relative"},
        "scatter_fraction": {"min": 0.05, "max": 0.35, "unit": ""},
        "collimator_blur": {"min": 1.0, "max": 6.0, "unit": "pixels FWHM"},
    },
}

# -- Tier sample counts -------------------------------------------------------
TIER_SAMPLE_COUNTS = {"public": 12, "dev": 20, "hidden": 20}


# -- Ellipse mask helper ------------------------------------------------------

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


# -- Zoom/crop utility --------------------------------------------------------

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


# -- Phantom Generators -------------------------------------------------------

def make_cardiac_perfusion_phantom(
    H: int, W: int, seed: int, variant: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Generate a cardiac perfusion SPECT/CT phantom (Tc-99m sestamibi).

    Returns:
        x_ct:     (H, W) float64 -- CT image (normalized attenuation) [0, 1]
        x_spect:  (H, W) float64 -- SPECT activity distribution [0, 1]
        mu_map:   (H, W) float64 -- attenuation coefficients [cm^-1]
        name:     scene name
    """
    rng = np.random.default_rng(seed)

    x_ct = np.zeros((H, W), dtype=np.float64)
    x_spect = np.zeros((H, W), dtype=np.float64)
    mu_map = np.zeros((H, W), dtype=np.float64)

    # Chest outline (torso cross-section)
    chest = _ellipse_mask(H, W, 0.0, 0.0, 0.44, 0.36, 0)
    x_ct[chest] = 0.30       # soft tissue CT value
    mu_map[chest] = 0.151    # soft tissue at 140 keV (Tc-99m)
    x_spect[chest] = 0.08    # low background uptake

    # Ribs/sternum (high attenuation bone)
    for rib_y in [-0.15, -0.05, 0.05, 0.15]:
        rib_l = _ellipse_mask(H, W, -0.38, rib_y, 0.04, 0.015, -5) & chest
        rib_r = _ellipse_mask(H, W, 0.38, rib_y, 0.04, 0.015, 5) & chest
        x_ct[rib_l] = 0.85
        x_ct[rib_r] = 0.85
        mu_map[rib_l] = 0.28   # bone at 140 keV
        mu_map[rib_r] = 0.28
        x_spect[rib_l] = 0.02
        x_spect[rib_r] = 0.02

    # Spine (posterior, dense bone)
    spine = _ellipse_mask(
        H, W, 0.0, 0.25 + variant * 0.005, 0.06, 0.05, 0) & chest
    x_ct[spine] = 0.90
    mu_map[spine] = 0.28
    x_spect[spine] = 0.04

    # Lungs (low attenuation, low uptake)
    lung_l = _ellipse_mask(
        H, W, -0.20, -0.02 + variant * 0.01, 0.14, 0.22, -4) & chest
    lung_r = _ellipse_mask(
        H, W, 0.20, -0.02 + variant * 0.01, 0.14, 0.22, 4) & chest
    for lung in [lung_l, lung_r]:
        x_ct[lung] = 0.05     # aerated lung on CT
        mu_map[lung] = 0.045  # inflated lung at 140 keV
        x_spect[lung] = 0.02

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

    base_myo_uptake = 0.95 + rng.uniform(-0.05, 0.05)
    x_spect[myocardium] = base_myo_uptake
    x_spect[lv_cavity] = 0.15   # blood pool
    x_ct[myocardium] = 0.35     # myocardium on CT
    x_ct[lv_cavity] = 0.25      # blood

    # Right ventricle (thinner wall, lower uptake)
    rv_outer = _ellipse_mask(
        H, W, heart_cx + 0.12, heart_cy + 0.02,
        0.06, 0.10, heart_angle + 10) & chest
    rv_inner = _ellipse_mask(
        H, W, heart_cx + 0.12, heart_cy + 0.02,
        0.04, 0.07, heart_angle + 10) & chest
    rv_wall = rv_outer & ~rv_inner & ~lv_outer
    x_spect[rv_wall] = 0.35 + rng.uniform(-0.05, 0.05)
    x_ct[rv_wall] = 0.32

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
        in_sector = (angles >= angle_start) & (angles < angle_start + angle_span)
        if in_sector.sum() > 0:
            defect_severity = rng.uniform(0.15, 0.65)
            x_spect[y_coords[in_sector], x_coords[in_sector]] = defect_severity

    # Liver (high uptake in sestamibi)
    liver = _ellipse_mask(H, W, 0.12, 0.12, 0.18, 0.08, -8) & chest
    liver = liver & ~lv_outer & ~rv_outer
    x_spect[liver] = 0.55 + rng.uniform(-0.10, 0.10)
    x_ct[liver] = 0.40
    mu_map[liver] = 0.158

    # Stomach (variable uptake)
    stomach = _ellipse_mask(H, W, -0.10, 0.15, 0.06, 0.04, 10) & chest
    x_spect[stomach] = rng.uniform(0.20, 0.60)
    x_ct[stomach] = 0.20

    # Smooth both
    x_ct = gaussian_filter(x_ct, sigma=0.8)
    x_spect = gaussian_filter(x_spect, sigma=0.8)
    x_ct = np.clip(x_ct, 0.0, None)
    x_spect = np.clip(x_spect, 0.0, None)
    if x_ct.max() > 0:
        x_ct /= x_ct.max()
    if x_spect.max() > 0:
        x_spect /= x_spect.max()

    return x_ct, x_spect, mu_map, f"cardiac_{variant:02d}"


def make_brain_perfusion_phantom(
    H: int, W: int, seed: int, variant: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Generate a brain perfusion SPECT/CT phantom (Tc-99m HMPAO).

    Returns:
        x_ct, x_spect, mu_map, name
    """
    rng = np.random.default_rng(seed)

    x_ct = np.zeros((H, W), dtype=np.float64)
    x_spect = np.zeros((H, W), dtype=np.float64)
    mu_map = np.zeros((H, W), dtype=np.float64)

    # Skull (bone)
    skull_outer = _ellipse_mask(
        H, W, 0.0, 0.0,
        0.40 + variant * 0.008, 0.46 + variant * 0.005, 0)
    skull_inner = _ellipse_mask(
        H, W, 0.0, 0.0,
        0.36 + variant * 0.008, 0.42 + variant * 0.005, 0)
    bone = skull_outer & ~skull_inner
    brain = skull_inner.copy()

    x_ct[bone] = 0.85      # dense bone on CT
    x_ct[brain] = 0.35     # brain tissue on CT
    mu_map[bone] = 0.28    # bone at 140 keV
    mu_map[brain] = 0.151  # soft tissue at 140 keV

    # White matter: moderate perfusion
    x_spect[brain] = 0.30 + rng.uniform(-0.03, 0.03)

    # Gray matter cortex: high perfusion
    cortex_outer = _ellipse_mask(H, W, 0.0, 0.0, 0.35, 0.41, 0)
    cortex_inner = _ellipse_mask(H, W, 0.0, 0.0, 0.29, 0.35, 0)
    cortex = cortex_outer & ~cortex_inner & brain
    x_spect[cortex] = 0.92 + rng.uniform(-0.05, 0.05)

    # Deep gray matter nuclei
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
        x_spect[structure] = uptake
    for structure in [thalamus_l, thalamus_r]:
        uptake = 0.82 + rng.uniform(-0.05, 0.05)
        x_spect[structure] = uptake

    # Cerebellum
    cerebellum = _ellipse_mask(
        H, W, 0.0, -0.30, 0.18, 0.08, 0) & brain
    x_spect[cerebellum] = 0.95 + rng.uniform(-0.05, 0.05)

    # Ventricles (CSF: no perfusion, visible on CT)
    vent_l = _ellipse_mask(
        H, W, -0.025, 0.05, 0.015, 0.055, -5) & brain
    vent_r = _ellipse_mask(
        H, W, 0.025, 0.05, 0.015, 0.055, 5) & brain
    x_spect[vent_l] = 0.05
    x_spect[vent_r] = 0.05
    x_ct[vent_l] = 0.10   # CSF dark on CT
    x_ct[vent_r] = 0.10

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
            x_spect[defect] = rng.uniform(0.15, 0.45)

    # Smooth
    x_ct = gaussian_filter(x_ct, sigma=1.0)
    x_spect = gaussian_filter(x_spect, sigma=1.0)
    x_ct = np.clip(x_ct, 0.0, None)
    x_spect = np.clip(x_spect, 0.0, None)
    if x_ct.max() > 0:
        x_ct /= x_ct.max()
    if x_spect.max() > 0:
        x_spect /= x_spect.max()

    return x_ct, x_spect, mu_map, f"brain_{variant:02d}"


def make_bone_scan_phantom(
    H: int, W: int, seed: int, variant: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Generate a bone scan SPECT/CT phantom (Tc-99m MDP).

    Bone scan is one of the most common nuclear medicine exams.
    High uptake in bone regions, with metastatic lesions as hot spots.

    Returns:
        x_ct, x_spect, mu_map, name
    """
    rng = np.random.default_rng(seed)

    x_ct = np.zeros((H, W), dtype=np.float64)
    x_spect = np.zeros((H, W), dtype=np.float64)
    mu_map = np.zeros((H, W), dtype=np.float64)

    # Body outline
    body = _ellipse_mask(H, W, 0.0, 0.0, 0.42, 0.34, 0)
    x_ct[body] = 0.30
    mu_map[body] = 0.151
    x_spect[body] = 0.10   # soft tissue background uptake

    # Spine (very high bone uptake)
    spine = _ellipse_mask(
        H, W, 0.0, 0.22 + variant * 0.005, 0.05, 0.04, 0) & body
    x_ct[spine] = 0.90
    mu_map[spine] = 0.28
    x_spect[spine] = 0.75 + rng.uniform(-0.05, 0.05)

    # Sternum
    sternum = _ellipse_mask(
        H, W, 0.0, -0.25, 0.02, 0.06, 0) & body
    x_ct[sternum] = 0.82
    mu_map[sternum] = 0.28
    x_spect[sternum] = 0.65 + rng.uniform(-0.05, 0.05)

    # Ribs (moderate bone uptake)
    for rib_y in [-0.18, -0.08, 0.02, 0.12]:
        rib_l = _ellipse_mask(
            H, W, -0.36, rib_y + variant * 0.003,
            0.05, 0.012, -8 + variant) & body
        rib_r = _ellipse_mask(
            H, W, 0.36, rib_y + variant * 0.003,
            0.05, 0.012, 8 - variant) & body
        x_ct[rib_l] = 0.80
        x_ct[rib_r] = 0.80
        mu_map[rib_l] = 0.28
        mu_map[rib_r] = 0.28
        x_spect[rib_l] = 0.55 + rng.uniform(-0.05, 0.05)
        x_spect[rib_r] = 0.55 + rng.uniform(-0.05, 0.05)

    # Lungs (low on both)
    lung_l = _ellipse_mask(
        H, W, -0.20, -0.03, 0.13, 0.20, -3) & body
    lung_r = _ellipse_mask(
        H, W, 0.20, -0.03, 0.13, 0.20, 3) & body
    for lung in [lung_l, lung_r]:
        x_ct[lung] = 0.05
        mu_map[lung] = 0.045
        x_spect[lung] = 0.03

    # Kidneys (high uptake due to MDP excretion)
    kidney_l = _ellipse_mask(
        H, W, -0.16, 0.10, 0.04, 0.06, -10) & body
    kidney_r = _ellipse_mask(
        H, W, 0.16, 0.10, 0.04, 0.06, 10) & body
    x_spect[kidney_l] = 0.80
    x_spect[kidney_r] = 0.80
    x_ct[kidney_l] = 0.38
    x_ct[kidney_r] = 0.38

    # Bladder (very high uptake)
    bladder = _ellipse_mask(
        H, W, 0.0, 0.25, 0.06, 0.04, 0) & body
    x_spect[bladder] = 0.95
    x_ct[bladder] = 0.28

    # Metastatic bone lesions (hot spots -- 1-4 lesions)
    n_mets = rng.integers(1, 5)
    for _ in range(n_mets):
        # Place preferentially near bone structures
        lx = rng.uniform(-0.30, 0.30)
        ly = rng.uniform(-0.20, 0.25)
        lr = rng.uniform(0.015, 0.04)
        lesion = _ellipse_mask(
            H, W, lx, ly, lr,
            lr * rng.uniform(0.6, 1.4),
            rng.uniform(-30, 30)) & body
        if lesion.sum() > 5:
            x_spect[lesion] = rng.uniform(1.2, 2.5)  # very hot
            x_ct[lesion] = rng.uniform(0.50, 0.75)    # sclerotic on CT

    # Smooth
    x_ct = gaussian_filter(x_ct, sigma=0.8)
    x_spect = gaussian_filter(x_spect, sigma=0.8)
    x_ct = np.clip(x_ct, 0.0, None)
    x_spect = np.clip(x_spect, 0.0, None)
    if x_ct.max() > 0:
        x_ct /= x_ct.max()
    if x_spect.max() > 0:
        x_spect /= x_spect.max()

    return x_ct, x_spect, mu_map, f"bone_{variant:02d}"


# -- Phantom diversity pools --------------------------------------------------

PHANTOM_GENERATORS = [
    make_cardiac_perfusion_phantom,
    make_brain_perfusion_phantom,
    make_bone_scan_phantom,
]


def generate_phantoms_public(n: int = 12):
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


def generate_phantoms_dev(n: int = 20):
    """Generate dev-tier phantoms with augmented diversity."""
    phantoms = []
    rng = np.random.default_rng(5000)
    for i in range(n):
        gen_fn = PHANTOM_GENERATORS[i % 3]
        x_ct, x_spect, mu_map, name = gen_fn(
            IMAGE_SIZE, IMAGE_SIZE, seed=500 + i, variant=i)
        # Augment: rotation + flip + mild zoom
        angle = float(rng.uniform(15, 345))
        x_ct = nd_rotate(x_ct, angle, reshape=False, mode='constant', cval=0.0)
        x_spect = nd_rotate(x_spect, angle, reshape=False, mode='constant', cval=0.0)
        mu_map = nd_rotate(mu_map, angle, reshape=False, mode='constant', cval=0.0)
        if rng.random() < 0.5:
            x_ct = np.fliplr(x_ct)
            x_spect = np.fliplr(x_spect)
            mu_map = np.fliplr(mu_map)
        zoom_f = float(rng.uniform(0.85, 1.15))
        if zoom_f != 1.0:
            x_ct = _zoom_crop(x_ct, zoom_f, IMAGE_SIZE)
            x_spect = _zoom_crop(x_spect, zoom_f, IMAGE_SIZE)
            mu_map = _zoom_crop(mu_map, zoom_f, IMAGE_SIZE)
        x_ct = np.clip(x_ct, 0.0, None)
        x_spect = np.clip(x_spect, 0.0, None)
        if x_ct.max() > 0:
            x_ct /= x_ct.max()
        if x_spect.max() > 0:
            x_spect /= x_spect.max()
        phantoms.append((x_ct, x_spect, mu_map, f"dev_{name}"))
    return phantoms


def generate_phantoms_hidden(n: int = 20):
    """Generate hidden-tier phantoms with adversarial modifications."""
    phantoms = []
    rng = np.random.default_rng(8000)
    for i in range(n):
        gen_fn = PHANTOM_GENERATORS[i % 3]
        x_ct, x_spect, mu_map, name = gen_fn(
            IMAGE_SIZE, IMAGE_SIZE, seed=800 + i, variant=i + 10)

        # Adversarial augmentation
        angle = float(rng.uniform(20, 340))
        x_ct = nd_rotate(x_ct, angle, reshape=False, mode='constant', cval=0.0)
        x_spect = nd_rotate(x_spect, angle, reshape=False, mode='constant', cval=0.0)
        mu_map = nd_rotate(mu_map, angle, reshape=False, mode='constant', cval=0.0)
        if rng.random() < 0.7:
            x_ct = np.fliplr(x_ct)
            x_spect = np.fliplr(x_spect)
            mu_map = np.fliplr(mu_map)
        if rng.random() < 0.5:
            x_ct = np.flipud(x_ct)
            x_spect = np.flipud(x_spect)
            mu_map = np.flipud(mu_map)

        # Aggressive zoom
        zoom_f = float(rng.uniform(0.70, 1.30))
        x_ct = _zoom_crop(x_ct, zoom_f, IMAGE_SIZE)
        x_spect = _zoom_crop(x_spect, zoom_f, IMAGE_SIZE)
        mu_map = _zoom_crop(mu_map, zoom_f, IMAGE_SIZE)

        # Add subtle micro-lesions (hard to detect)
        n_micro = rng.integers(2, 6)
        for _ in range(n_micro):
            cy = rng.integers(40, IMAGE_SIZE - 40)
            cx = rng.integers(40, IMAGE_SIZE - 40)
            r = rng.integers(2, 6)
            yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
            circle = (yy**2 + xx**2 <= r**2).astype(np.float64)
            y0, y1 = max(0, cy - r), min(IMAGE_SIZE, cy + r + 1)
            x0, x1 = max(0, cx - r), min(IMAGE_SIZE, cx + r + 1)
            c_y0, c_y1 = r - (cy - y0), r + (y1 - cy)
            c_x0, c_x1 = r - (cx - x0), r + (x1 - cx)
            if x_spect[y0:y1, x0:x1].mean() > 0.1:
                intensity = rng.uniform(1.5, 4.0)
                x_spect[y0:y1, x0:x1] = np.maximum(
                    x_spect[y0:y1, x0:x1],
                    circle[c_y0:c_y1, c_x0:c_x1] * intensity
                )

        x_ct = np.clip(x_ct, 0.0, None)
        x_spect = np.clip(x_spect, 0.0, None)
        if x_ct.max() > 0:
            x_ct /= x_ct.max()
        if x_spect.max() > 0:
            x_spect /= x_spect.max()
        phantoms.append((x_ct, x_spect, mu_map, f"hidden_{name}"))
    return phantoms


# -- SPECT/CT Forward Model ---------------------------------------------------

def ct_forward_model(
    x_ct: np.ndarray,
    theta_deg: np.ndarray,
    rng: np.random.Generator,
    ct_noise_lambda: float = 15.0,
) -> np.ndarray:
    """CT forward model: Radon transform + Poisson noise.

    Args:
        x_ct:            (H, W) CT image [0, 1]
        theta_deg:       projection angles in degrees
        rng:             random generator
        ct_noise_lambda: Poisson noise intensity

    Returns:
        y_ct: (N_angles, N_det) float32 measured CT sinogram
    """
    sino_ct = radon_transform(x_ct, theta_deg)
    # Scale to physical count level
    sino_ct = np.maximum(sino_ct, 0.0)
    # Add Poisson noise
    y_ct = sino_ct + rng.poisson(lam=ct_noise_lambda,
                                  size=sino_ct.shape).astype(np.float64)
    return y_ct.astype(np.float32)


def spect_forward_model(
    x_spect: np.ndarray,
    mu_map: np.ndarray,
    theta_deg: np.ndarray,
    collimator_blur_fwhm: float,
    scatter_fraction: float,
    count_rate: float,
    rng: np.random.Generator,
) -> dict:
    """SPECT forward model with depth-dependent attenuation and collimator blur.

    y_spect = D(mu_map) * Radon_blur(x_spect) + scatter + Poisson noise

    Args:
        x_spect:               (H, W) SPECT activity distribution [0, 1]
        mu_map:                (H, W) attenuation coefficients [cm^-1]
        theta_deg:             projection angles in degrees
        collimator_blur_fwhm:  collimator PSF FWHM in pixels
        scatter_fraction:      scatter / total ratio
        count_rate:            total expected counts (controls noise level)
        rng:                   random generator

    Returns:
        dict with sinogram_ideal, sinogram_measured, attenuation_factors, etc.
    """
    n_angles = len(theta_deg)
    pixel_size_cm = PIXEL_SIZE_MM / 10.0

    # 1. Ideal SPECT sinogram (Radon transform of activity)
    sino_ideal = radon_transform(x_spect, theta_deg)
    sino_ideal = np.maximum(sino_ideal, 0.0)
    n_det = sino_ideal.shape[1]

    # 2. Apply collimator blur (Gaussian blur along detector axis)
    # SPECT has wider PSF than PET due to collimator geometry
    collimator_sigma = collimator_blur_fwhm / 2.355  # FWHM to sigma
    if collimator_sigma > 0.3:
        sino_blurred = np.zeros_like(sino_ideal)
        for i in range(n_angles):
            sino_blurred[i] = gaussian_filter(
                sino_ideal[i], sigma=collimator_sigma)
    else:
        sino_blurred = sino_ideal.copy()

    # 3. Attenuation: D(mu_map) = exp(-Radon(mu_map))
    # Single-photon attenuation (one-sided, unlike PET two-sided)
    sino_mu = radon_transform(mu_map, theta_deg)
    sino_mu_physical = sino_mu * pixel_size_cm
    attn_factors = np.exp(-sino_mu_physical)

    # 4. Apply attenuation to blurred sinogram
    sino_atten = sino_blurred * attn_factors

    # 5. Scale to physical count level
    if sino_atten.sum() > 0:
        scale = count_rate / sino_atten.sum()
    else:
        scale = 1.0
    sino_scaled = sino_atten * scale

    # 6. Energy-dependent scatter (smooth, spatially varying)
    mean_signal = max(sino_scaled.mean(), 1e-6)
    denom = max(1.0 - scatter_fraction, 0.1)
    scatter_level = scatter_fraction / denom * mean_signal
    scatter = np.ones_like(sino_scaled) * scatter_level
    # Add spatial variation (smooth)
    scatter_noise = rng.standard_normal(sino_scaled.shape) * scatter_level * 0.15
    scatter += gaussian_filter(scatter_noise, sigma=[4.0, 8.0])
    scatter = np.maximum(scatter, 0.0)

    # 7. Expected counts
    expected = sino_scaled + scatter
    expected = np.maximum(expected, 0.01)

    # 8. Poisson sampling (SPECT has lower counts than PET)
    sino_measured = rng.poisson(expected).astype(np.float64)

    return {
        "sinogram_ideal": sino_ideal.astype(np.float32),
        "sinogram_blurred": sino_blurred.astype(np.float32),
        "sinogram_measured": sino_measured.astype(np.float32),
        "attenuation_factors": attn_factors.astype(np.float32),
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
    ssim = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / \
           ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))
    return float(ssim)


# -- Image helpers ------------------------------------------------------------

def _norm(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo + 1e-8)


def _save_png(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(
        np.clip(_norm(arr) * 255, 0, 255).astype(np.uint8), "L"
    ).save(str(path))


# -- Sample mismatch parameter sampling --------------------------------------

def sample_mismatch(rng: np.random.Generator, spec: dict) -> dict:
    """Sample mismatch parameters from spec ranges."""
    result = {}
    for k, v in spec.items():
        if k == "ct_registration_shift":
            # Integer shift
            result[k] = int(rng.integers(v["min"], v["max"] + 1))
        else:
            result[k] = float(rng.uniform(v["min"], v["max"]))
    return result


# -- Tier generation ----------------------------------------------------------

def generate_tier(
    tier: str,
    phantoms: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]],
    base_seed: int,
) -> None:
    """Generate one tier of the SPECT/CT benchmark."""
    spec_ranges = SPEC[tier]
    tier_dir = BENCHMARK_DIR / tier
    images_dir = tier_dir / "images"
    if tier_dir.exists():
        shutil.rmtree(tier_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    theta_spect = np.linspace(
        0, ANGLE_RANGE_SPECT, N_ANGLES_SPECT, endpoint=False).astype(np.float64)
    theta_ct = np.linspace(
        0, ANGLE_RANGE_CT, N_ANGLES_CT, endpoint=False).astype(np.float64)

    h5_path = tier_dir / f"spect_ct_challenge_{tier}.h5"
    rng = np.random.default_rng(base_seed)
    true_specs = {}

    # SPECT count rate: lower than PET (typical 0.5-2 Mcps vs 2-5 Mcps)
    spect_count_rate = 5e5  # 0.5 Mcps (lower statistics)

    with h5py.File(h5_path, "w") as f:
        f.attrs["description"] = (
            f"PWM SPECT/CT multimodality benchmark -- {tier} tier "
            f"(CT: Radon + Poisson | SPECT: attenuation * Radon_blur + scatter + Poisson)"
        )
        f.attrs["spec_ranges"] = json.dumps(spec_ranges)
        f.attrs["geometry"] = json.dumps({
            "image_size": IMAGE_SIZE,
            "n_angles_spect": N_ANGLES_SPECT,
            "n_angles_ct": N_ANGLES_CT,
            "angle_range_spect_deg": [0, ANGLE_RANGE_SPECT],
            "angle_range_ct_deg": [0, ANGLE_RANGE_CT],
            "fov_mm": FOV_MM,
            "pixel_size_mm": PIXEL_SIZE_MM,
        })
        f.attrs["forward_model"] = (
            "y_ct = Radon(x_ct) + Poisson; "
            "y_spect = D(mu_map) * Radon_blur(x_spect) + scatter + Poisson"
        )

        for idx, (x_ct, x_spect, mu_map, scene_name) in enumerate(phantoms):
            key = f"sample_{idx:02d}"
            print(f"  [{tier}] Generating {key} ({scene_name})...",
                  end="", flush=True)

            # Sample mismatch parameters
            mis = sample_mismatch(rng, spec_ranges)
            true_specs[key] = mis

            # Apply CT-SPECT registration shift
            ct_shift = mis["ct_registration_shift"]
            if ct_shift != 0:
                x_ct_shifted = nd_shift(
                    x_ct, (ct_shift, ct_shift), order=1, mode='constant')
            else:
                x_ct_shifted = x_ct.copy()

            # Derive mu_map from CT with hu_to_mu scaling
            mu_map_scaled = mu_map * mis["hu_to_mu_scale"]

            # CT forward model
            y_ct = ct_forward_model(x_ct_shifted, theta_ct, rng)

            # SPECT forward model
            spect_result = spect_forward_model(
                x_spect=x_spect,
                mu_map=mu_map_scaled,
                theta_deg=theta_spect,
                collimator_blur_fwhm=mis["collimator_blur"],
                scatter_fraction=mis["scatter_fraction"],
                count_rate=spect_count_rate,
                rng=rng,
            )
            y_spect = spect_result["sinogram_measured"]

            # Save to HDF5
            grp = f.create_group(key)
            grp.create_dataset("x_ct", data=x_ct_shifted.astype(np.float32),
                               compression="gzip")
            grp.create_dataset("x_spect", data=x_spect.astype(np.float32),
                               compression="gzip")
            grp.create_dataset("y_ct", data=y_ct, compression="gzip")
            grp.create_dataset("y_spect", data=y_spect.astype(np.float32),
                               compression="gzip")
            # Also save as x_true for generate_all_artifacts.py compatibility
            grp.create_dataset("x_true", data=x_spect.astype(np.float32),
                               compression="gzip")

            # Store metadata
            grp.attrs["metadata"] = json.dumps({
                "scene": scene_name,
                "shape": list(x_spect.shape),
                "n_angles_spect": N_ANGLES_SPECT,
                "n_angles_ct": N_ANGLES_CT,
                "n_det_spect": int(y_spect.shape[1]),
                "n_det_ct": int(y_ct.shape[1]),
            })
            grp.attrs["true_spec"] = json.dumps(mis)
            grp.attrs["spec_ranges"] = json.dumps(spec_ranges)
            grp.attrs.update({
                "theta_angles_spect": N_ANGLES_SPECT,
                "theta_angles_ct": N_ANGLES_CT,
                "n_detectors_spect": int(y_spect.shape[1]),
                "n_detectors_ct": int(y_ct.shape[1]),
                "collimator_blur": mis["collimator_blur"],
                "scatter_fraction": mis["scatter_fraction"],
                "attenuation_scale": mis["hu_to_mu_scale"],
            })

            # Save images
            sample_dir = images_dir / f"sample_{idx:02d}_{scene_name}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            _save_png(x_ct_shifted, sample_dir / "x_ct.png")
            _save_png(x_spect, sample_dir / "x_spect.png")
            _save_png(y_ct, sample_dir / "y_ct_sinogram.png")
            _save_png(y_spect, sample_dir / "y_spect_sinogram.png")

            print(f"  shift={ct_shift}  blur={mis['collimator_blur']:.1f}  "
                  f"scatter={mis['scatter_fraction']:.2f}  "
                  f"mu_scale={mis['hu_to_mu_scale']:.3f}")

    # Save spec files
    with open(tier_dir / "spec.json", "w") as sf:
        json.dump(spec_ranges, sf, indent=2)
    with open(tier_dir / "true_spec.json", "w") as tf:
        json.dump(true_specs, tf, indent=2)

    print(f"  [{tier}] {len(phantoms)} samples saved to {h5_path.name}")


# -- Gallery image generation -------------------------------------------------

def generate_gallery_images() -> None:
    """Generate gallery images for the platform benchmark page."""
    gallery_base = (BENCHMARK_DIR.parent.parent.parent /
                    "platform" / "pwm_platform" / "static" / "img" /
                    "benchmark_gallery" / "spect_ct")

    h5_path = BENCHMARK_DIR / "public" / "spect_ct_challenge_public.h5"
    if not h5_path.exists():
        print("  [gallery] Public HDF5 not found, skipping gallery generation.")
        return

    # Pick diverse samples: cardiac (0), brain (4), bone (8), cardiac variant (3)
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
            x_ct = grp["x_ct"][:]
            x_spect = grp["x_spect"][:]
            y_ct = grp["y_ct"][:]
            y_spect = grp["y_spect"][:]

            # gt.png -- SPECT activity (primary reconstruction target)
            _save_png(x_spect, scene_dir / "gt.png")
            # measurement_I.png -- measured SPECT sinogram
            _save_png(y_spect, scene_dir / "measurement_I.png")
            # measurement_II.png -- measured CT sinogram
            _save_png(y_ct, scene_dir / "measurement_II.png")
            # recon_I.png -- CT image
            _save_png(x_ct, scene_dir / "recon_I.png")
            # recon_II.png -- difference |x_ct - x_spect|
            diff = np.abs(x_ct.astype(np.float64) - x_spect.astype(np.float64))
            _save_png(diff, scene_dir / "recon_II.png")
            # recon_III.png -- overlay (CT structure + SPECT activity)
            overlay = 0.4 * _norm(x_ct) + 0.6 * _norm(x_spect)
            _save_png(overlay, scene_dir / "recon_III.png")

            print(f"  [gallery] scene_{scene_idx:02d} images saved to {scene_dir}")


# -- Main ---------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", type=str, default="all",
                        help="public|dev|hidden|all")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for sample generation")
    args = parser.parse_args()

    tiers = ["public", "dev", "hidden"] if args.tier == "all" else [args.tier]

    print("SPECT/CT Multimodality Benchmark Dataset Generator")
    print("=" * 68)
    print(f"Output: {BENCHMARK_DIR}")
    print(f"SPECT geometry: {N_ANGLES_SPECT} angles, {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"CT geometry:    {N_ANGLES_CT} angles, {IMAGE_SIZE}x{IMAGE_SIZE}\n")

    for tier in tiers:
        seed = args.seed + TIER_SEED_OFFSETS.get(tier, 0)

        if tier == "public":
            phantoms = generate_phantoms_public(12)
        elif tier == "dev":
            phantoms = generate_phantoms_dev(20)
        else:
            phantoms = generate_phantoms_hidden(20)

        print(f"\nGenerating {tier} tier ({len(phantoms)} samples)...")
        generate_tier(tier=tier, phantoms=phantoms, base_seed=seed)

    # Generate gallery images
    print("\nGenerating gallery images...")
    generate_gallery_images()

    print(f"\n{'=' * 68}")
    print("SPECT/CT benchmark dataset generation complete!")
    print(f"Output: {BENCHMARK_DIR}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
