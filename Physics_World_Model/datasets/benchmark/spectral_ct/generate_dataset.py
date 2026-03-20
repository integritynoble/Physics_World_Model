#!/usr/bin/env python3
"""Generate Spectral CT (Dual-Energy CT) benchmark dataset.

Forward model (2D dual-energy CT with material decomposition):
    y_low  = Poisson(I0_low  * exp(-Radon(mu_low(x))))  + readout_noise
    y_high = Poisson(I0_high * exp(-Radon(mu_high(x)))) + readout_noise

where:
    x_water, x_bone, x_iodine : material basis maps (ground truth, 256x256 each)
    mu_low  = alpha_w_low  * x_water + alpha_b_low  * x_bone + alpha_i_low  * x_iodine
    mu_high = alpha_w_high * x_water + alpha_b_high * x_bone + alpha_i_high * x_iodine

    alpha_*_{low,high} : mass attenuation coefficients [cm^2/g] at each energy
    Radon : parallel-beam Radon transform (256 angles, 367 detector bins)

Material attenuation coefficients (approximate, NIST XCOM):
    @ 80 kVp (effective ~56 keV):
        water:  0.2059 cm^2/g
        bone:   0.3260 cm^2/g
        iodine: 8.930  cm^2/g   (K-edge at 33.2 keV, high at 56 keV)
    @ 140 kVp (effective ~76 keV):
        water:  0.1823 cm^2/g
        bone:   0.2530 cm^2/g
        iodine: 3.650  cm^2/g

Densities (for converting concentration maps to linear attenuation):
    water:  1.0 g/cm^3
    bone:   1.9 g/cm^3 (cortical)
    iodine: contrast agent density ~ 4.93 g/cm^3 (pure), but used as concentration

The inverse problem: decompose material basis maps (water, bone, iodine)
from dual-energy sinogram measurements.

Mismatch parameters:
    energy_calibration_error : kVp shift (changes effective attenuation) [keV]
    scatter_fraction         : scatter / total signal fraction
    detector_crosstalk       : cross-energy leakage fraction
    beam_hardening           : polychromatic beam hardening coefficient

Phantoms:
    Public  : 12 samples (4 head + 4 abdomen + 4 chest)
    Dev     : 20 samples (augmented anatomy with diverse lesion patterns)
    Hidden  : 20 samples (adversarial: subtle iodine, extreme mismatch, micro-lesions)

Usage:
    python3 generate_dataset.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy.ndimage import rotate as nd_rotate, gaussian_filter, zoom as nd_zoom

# Import radon_transform from pet/generate_dataset.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pet"))
from generate_dataset import radon_transform  # noqa: E402

BENCHMARK_DIR = Path(__file__).resolve().parent

# -- Geometry -----------------------------------------------------------------

IMAGE_SIZE = 256
N_ANGLES = 256
N_DET = 367  # ceil(sqrt(2)*256)

# -- Material attenuation coefficients (cm^2/g) at two energies ---------------
# Based on NIST XCOM tabulated values for effective photon energies

# 80 kVp (effective ~56 keV)
ALPHA_WATER_LOW  = 0.2059   # cm^2/g
ALPHA_BONE_LOW   = 0.3260   # cm^2/g
ALPHA_IODINE_LOW = 8.930    # cm^2/g  (above K-edge)

# 140 kVp (effective ~76 keV)
ALPHA_WATER_HIGH  = 0.1823  # cm^2/g
ALPHA_BONE_HIGH   = 0.2530  # cm^2/g
ALPHA_IODINE_HIGH = 3.650   # cm^2/g

# Densities for converting maps to linear attenuation [g/cm^3]
RHO_WATER  = 1.00
RHO_BONE   = 1.90
# Iodine map is in mg/mL concentration; convert to g/cm^3 = mg/mL * 1e-3
IODINE_CONC_TO_DENSITY = 1e-3

# Photon counts (controls noise level)
I0_LOW  = 50_000.0   # 80 kVp has lower output
I0_HIGH = 100_000.0  # 140 kVp has higher output
SIGMA_READOUT = 5.0   # electronic readout noise

# Pixel size (for physical scaling of Radon line integrals)
FOV_CM = 25.0  # 25 cm FOV
PIXEL_SIZE_CM = FOV_CM / IMAGE_SIZE  # ~0.098 cm/px

# -- Mismatch spec ranges per tier --------------------------------------------

SPEC = {
    "public": {
        "energy_calibration_error": {"min": -1.0,  "max":  1.0,  "unit": "keV"},
        "scatter_fraction":         {"min":  0.02, "max":  0.08, "unit": ""},
        "detector_crosstalk":       {"min":  0.00, "max":  0.03, "unit": ""},
        "beam_hardening":           {"min":  0.00, "max":  0.05, "unit": ""},
    },
    "dev": {
        "energy_calibration_error": {"min": -2.0,  "max":  2.0,  "unit": "keV"},
        "scatter_fraction":         {"min":  0.02, "max":  0.12, "unit": ""},
        "detector_crosstalk":       {"min":  0.00, "max":  0.06, "unit": ""},
        "beam_hardening":           {"min":  0.00, "max":  0.10, "unit": ""},
    },
    "hidden": {
        "energy_calibration_error": {"min": -4.0,  "max":  4.0,  "unit": "keV"},
        "scatter_fraction":         {"min":  0.02, "max":  0.20, "unit": ""},
        "detector_crosstalk":       {"min":  0.00, "max":  0.10, "unit": ""},
        "beam_hardening":           {"min":  0.00, "max":  0.20, "unit": ""},
    },
}


# -- Ellipse mask helper -----------------------------------------------------

def _ellipse_mask(H: int, W: int, cx: float, cy: float,
                  a: float, b: float, angle_deg: float) -> np.ndarray:
    """Generate a binary ellipse mask. Coordinates in normalised [-1, 1]."""
    yy = np.linspace(-1.0, 1.0, H)[:, None]
    xx = np.linspace(-1.0, 1.0, W)[None, :]
    ca, sa = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
    yr = (yy - cy) * ca - (xx - cx) * sa
    xr = (yy - cy) * sa + (xx - cx) * ca
    return (xr / a)**2 + (yr / b)**2 <= 1.0


# -- Phantom Generators -------------------------------------------------------
# Each returns (x_water, x_bone, x_iodine, scene_name)
# Maps are in density units: water [g/cm^3], bone [g/cm^3], iodine [mg/mL]

def make_head_phantom(
    H: int, W: int, seed: int, variant: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Head phantom: skull bone ring, brain soft tissue, iodine contrast vessels."""
    rng = np.random.default_rng(seed)

    x_water = np.zeros((H, W), dtype=np.float64)
    x_bone = np.zeros((H, W), dtype=np.float64)
    x_iodine = np.zeros((H, W), dtype=np.float64)

    # Skull outline
    skull_outer = _ellipse_mask(H, W, 0.0, 0.0,
                                0.42 + variant * 0.008, 0.48 + variant * 0.005, 0)
    skull_inner = _ellipse_mask(H, W, 0.0, 0.0,
                                0.38 + variant * 0.008, 0.44 + variant * 0.005, 0)
    bone_ring = skull_outer & ~skull_inner
    brain = skull_inner.copy()

    # Bone in skull
    x_bone[bone_ring] = RHO_BONE * rng.uniform(0.85, 1.0)

    # Brain soft tissue (water-equivalent)
    x_water[brain] = RHO_WATER * rng.uniform(0.95, 1.05)

    # Gray matter regions (slightly higher water density)
    cortex_outer = _ellipse_mask(H, W, 0.0, 0.0, 0.37, 0.43, 0)
    cortex_inner = _ellipse_mask(H, W, 0.0, 0.0, 0.32, 0.38, 0)
    cortex = cortex_outer & ~cortex_inner & brain
    x_water[cortex] = RHO_WATER * rng.uniform(1.02, 1.08)

    # Ventricles (CSF, lower density water)
    vent_l = _ellipse_mask(H, W, -0.03, 0.05, 0.02, 0.06, -5) & brain
    vent_r = _ellipse_mask(H, W, 0.03, 0.05, 0.02, 0.06, 5) & brain
    x_water[vent_l] = RHO_WATER * 1.01  # CSF slightly different
    x_water[vent_r] = RHO_WATER * 1.01
    x_bone[vent_l] = 0.0
    x_bone[vent_r] = 0.0

    # Deep brain structures with some bone-like calcification
    structures = [
        (-0.10, 0.05, 0.04, 0.08, -10),   # left caudate
        (0.10, 0.05, 0.04, 0.08, 10),      # right caudate
        (-0.06, -0.05, 0.04, 0.03, 0),     # left thalamus
        (0.06, -0.05, 0.04, 0.03, 0),      # right thalamus
    ]
    for cx, cy, a, b, ang in structures:
        cx += rng.uniform(-0.01, 0.01)
        cy += rng.uniform(-0.01, 0.01)
        mask = _ellipse_mask(H, W, cx, cy, a, b, ang + variant * 3) & brain
        x_water[mask] = RHO_WATER * rng.uniform(1.0, 1.05)

    # Pineal gland calcification (tiny bone spot)
    pineal = _ellipse_mask(H, W, 0.0, -0.02, 0.01, 0.01, 0) & brain
    x_bone[pineal] = RHO_BONE * rng.uniform(0.2, 0.5)

    # Contrast-enhanced vessels (iodine)
    n_vessels = rng.integers(2, 5)
    for _ in range(n_vessels):
        vx = rng.uniform(-0.25, 0.25)
        vy = rng.uniform(-0.30, 0.30)
        vr = rng.uniform(0.008, 0.02)
        vessel = _ellipse_mask(H, W, vx, vy, vr, vr * rng.uniform(0.7, 1.3),
                               rng.uniform(-45, 45)) & brain
        if vessel.sum() > 3:
            x_iodine[vessel] = rng.uniform(2.0, 8.0)  # mg/mL

    # Enhancing tumour (iodine uptake)
    if rng.random() < 0.7:
        tx = rng.uniform(-0.20, 0.20)
        ty = rng.uniform(-0.25, 0.25)
        tr = rng.uniform(0.02, 0.05)
        tumour = _ellipse_mask(H, W, tx, ty, tr, tr * rng.uniform(0.7, 1.3),
                               rng.uniform(-30, 30)) & brain
        if tumour.sum() > 10:
            x_iodine[tumour] = rng.uniform(3.0, 12.0)  # contrast-enhancing tumour
            x_water[tumour] = RHO_WATER * rng.uniform(0.95, 1.02)

    # Smooth maps slightly
    x_water = gaussian_filter(x_water, sigma=0.8)
    x_bone = gaussian_filter(x_bone, sigma=0.5)
    x_iodine = gaussian_filter(x_iodine, sigma=0.6)

    return (np.maximum(x_water, 0.0), np.maximum(x_bone, 0.0),
            np.maximum(x_iodine, 0.0), f"head_{variant:02d}")


def make_abdomen_phantom(
    H: int, W: int, seed: int, variant: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Abdomen phantom: soft tissue organs, spine bone, iodine contrast in vessels/kidneys."""
    rng = np.random.default_rng(seed)

    x_water = np.zeros((H, W), dtype=np.float64)
    x_bone = np.zeros((H, W), dtype=np.float64)
    x_iodine = np.zeros((H, W), dtype=np.float64)

    # Body outline
    body = _ellipse_mask(H, W, 0.0, 0.0, 0.42, 0.34 + variant * 0.005, 0)
    x_water[body] = RHO_WATER * rng.uniform(0.95, 1.0)

    # Spine (posterior bone)
    spine = _ellipse_mask(H, W, 0.0, 0.22, 0.05, 0.04, 0) & body
    x_bone[spine] = RHO_BONE * rng.uniform(0.8, 1.0)
    x_water[spine] = RHO_WATER * 0.3  # less water in bone

    # Ribs (bilateral bone arcs)
    for side in [-1, 1]:
        for rib_y in [-0.15, -0.05, 0.05, 0.15]:
            rib = _ellipse_mask(H, W, side * 0.35, rib_y + variant * 0.01,
                                0.025, 0.06, side * 15) & body
            x_bone[rib] = RHO_BONE * rng.uniform(0.6, 0.9)
            x_water[rib] = RHO_WATER * 0.3

    # Liver (right side)
    liver = _ellipse_mask(H, W, 0.15, 0.05, 0.16, 0.12, -10) & body
    x_water[liver] = RHO_WATER * rng.uniform(1.02, 1.06)

    # Spleen (left side)
    spleen = _ellipse_mask(H, W, -0.20, 0.05, 0.07, 0.05, 10) & body
    x_water[spleen] = RHO_WATER * rng.uniform(1.02, 1.06)

    # Kidneys with contrast
    kidney_l = _ellipse_mask(H, W, -0.15, 0.12, 0.04, 0.06, -10) & body
    kidney_r = _ellipse_mask(H, W, 0.15, 0.12, 0.04, 0.06, 10) & body
    x_water[kidney_l] = RHO_WATER * 1.03
    x_water[kidney_r] = RHO_WATER * 1.03
    # Iodine contrast in kidneys (nephrographic phase)
    x_iodine[kidney_l] = rng.uniform(2.0, 5.0)
    x_iodine[kidney_r] = rng.uniform(2.0, 5.0)

    # Aorta (large vessel, iodine contrast)
    aorta = _ellipse_mask(H, W, -0.02, 0.12, 0.025, 0.025, 0) & body
    x_iodine[aorta] = rng.uniform(5.0, 15.0)  # highly enhanced
    x_water[aorta] = RHO_WATER * 1.0

    # Hepatic lesion with different iodine uptake
    if rng.random() < 0.6:
        lx = rng.uniform(0.05, 0.25)
        ly = rng.uniform(-0.05, 0.10)
        lr = rng.uniform(0.015, 0.04)
        lesion = _ellipse_mask(H, W, lx, ly, lr, lr * rng.uniform(0.7, 1.3),
                               rng.uniform(-20, 20)) & liver
        if lesion.sum() > 5:
            x_iodine[lesion] = rng.uniform(0.5, 3.0)  # hypo/hyper-enhancing
            x_water[lesion] = RHO_WATER * rng.uniform(0.90, 1.0)

    # Smooth
    x_water = gaussian_filter(x_water, sigma=0.8)
    x_bone = gaussian_filter(x_bone, sigma=0.5)
    x_iodine = gaussian_filter(x_iodine, sigma=0.6)

    return (np.maximum(x_water, 0.0), np.maximum(x_bone, 0.0),
            np.maximum(x_iodine, 0.0), f"abdomen_{variant:02d}")


def make_chest_phantom(
    H: int, W: int, seed: int, variant: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Chest phantom: lungs (low water), ribs/spine, mediastinal vessels with iodine."""
    rng = np.random.default_rng(seed)

    x_water = np.zeros((H, W), dtype=np.float64)
    x_bone = np.zeros((H, W), dtype=np.float64)
    x_iodine = np.zeros((H, W), dtype=np.float64)

    # Thorax outline
    thorax = _ellipse_mask(H, W, 0.0, 0.0, 0.43, 0.36 + variant * 0.005, 0)
    x_water[thorax] = RHO_WATER * rng.uniform(0.90, 0.98)

    # Spine
    spine = _ellipse_mask(H, W, 0.0, 0.26, 0.045, 0.04, 0) & thorax
    x_bone[spine] = RHO_BONE * rng.uniform(0.8, 1.0)
    x_water[spine] = RHO_WATER * 0.3

    # Sternum (anterior)
    sternum = _ellipse_mask(H, W, 0.0, -0.30, 0.03, 0.015, 0) & thorax
    x_bone[sternum] = RHO_BONE * rng.uniform(0.6, 0.8)
    x_water[sternum] = RHO_WATER * 0.3

    # Ribs
    for side in [-1, 1]:
        for rib_y in [-0.20, -0.10, 0.0, 0.10, 0.20]:
            rib = _ellipse_mask(H, W, side * 0.37, rib_y,
                                0.02, 0.05, side * 20) & thorax
            x_bone[rib] = RHO_BONE * rng.uniform(0.5, 0.8)
            x_water[rib] = RHO_WATER * 0.3

    # Lungs (very low water density, essentially air)
    lung_l = _ellipse_mask(H, W, -0.18, -0.02 + variant * 0.01,
                           0.13, 0.20, -5) & thorax
    lung_r = _ellipse_mask(H, W, 0.18, -0.02 + variant * 0.01,
                           0.13, 0.20, 5) & thorax
    x_water[lung_l] = RHO_WATER * rng.uniform(0.08, 0.15)  # inflated lung
    x_water[lung_r] = RHO_WATER * rng.uniform(0.08, 0.15)

    # Heart (myocardium = soft tissue + blood pool with iodine)
    heart_outer = _ellipse_mask(H, W, -0.05, -0.03,
                                0.11, 0.10, 15 + variant * 3) & thorax
    heart_inner = _ellipse_mask(H, W, -0.05, -0.03,
                                0.07, 0.06, 15 + variant * 3) & thorax
    myocardium = heart_outer & ~heart_inner
    blood_pool = heart_inner
    x_water[myocardium] = RHO_WATER * rng.uniform(1.02, 1.06)
    x_water[blood_pool] = RHO_WATER * 1.0
    x_iodine[blood_pool] = rng.uniform(3.0, 8.0)  # contrast in blood

    # Great vessels (aorta, pulmonary arteries)
    aorta = _ellipse_mask(H, W, 0.02, 0.05, 0.025, 0.025, 0) & thorax
    x_iodine[aorta] = rng.uniform(5.0, 12.0)
    pa = _ellipse_mask(H, W, -0.03, -0.08, 0.02, 0.02, 0) & thorax
    x_iodine[pa] = rng.uniform(4.0, 10.0)

    # Pulmonary nodule (small lesion in lung)
    if rng.random() < 0.6:
        side = rng.choice([-1, 1])
        nx = side * rng.uniform(0.08, 0.22)
        ny = rng.uniform(-0.15, 0.12)
        nr = rng.uniform(0.01, 0.03)
        lung_mask = lung_l if side < 0 else lung_r
        nodule = _ellipse_mask(H, W, nx, ny, nr, nr * rng.uniform(0.8, 1.2),
                               rng.uniform(-20, 20)) & lung_mask
        if nodule.sum() > 3:
            x_water[nodule] = RHO_WATER * rng.uniform(0.8, 1.0)
            if rng.random() < 0.5:
                x_iodine[nodule] = rng.uniform(1.0, 4.0)  # enhancing nodule
            if rng.random() < 0.3:
                x_bone[nodule] = RHO_BONE * rng.uniform(0.1, 0.3)  # calcified nodule

    # Smooth
    x_water = gaussian_filter(x_water, sigma=0.8)
    x_bone = gaussian_filter(x_bone, sigma=0.5)
    x_iodine = gaussian_filter(x_iodine, sigma=0.6)

    return (np.maximum(x_water, 0.0), np.maximum(x_bone, 0.0),
            np.maximum(x_iodine, 0.0), f"chest_{variant:02d}")


# -- Phantom pool for each tier -----------------------------------------------

PHANTOM_GENERATORS = [make_head_phantom, make_abdomen_phantom, make_chest_phantom]


def generate_phantoms_public(
    n: int = 12,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, str]]:
    """4 head + 4 abdomen + 4 chest phantoms."""
    phantoms = []
    for i in range(4):
        phantoms.append(make_head_phantom(IMAGE_SIZE, IMAGE_SIZE, seed=100 + i, variant=i))
    for i in range(4):
        phantoms.append(make_abdomen_phantom(IMAGE_SIZE, IMAGE_SIZE, seed=200 + i, variant=i))
    for i in range(4):
        phantoms.append(make_chest_phantom(IMAGE_SIZE, IMAGE_SIZE, seed=300 + i, variant=i))
    return phantoms[:n]


def generate_phantoms_dev(
    n: int = 20,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, str]]:
    """Dev-tier phantoms with augmented diversity."""
    phantoms = []
    rng = np.random.default_rng(5000)
    for i in range(n):
        gen_fn = PHANTOM_GENERATORS[i % 3]
        w, b, iod, name = gen_fn(IMAGE_SIZE, IMAGE_SIZE, seed=500 + i, variant=i)

        # Augment: rotation + flip + mild zoom
        angle = float(rng.uniform(15, 345))
        w = nd_rotate(w, angle, reshape=False, mode='constant', cval=0.0)
        b = nd_rotate(b, angle, reshape=False, mode='constant', cval=0.0)
        iod = nd_rotate(iod, angle, reshape=False, mode='constant', cval=0.0)

        if rng.random() < 0.5:
            w = np.fliplr(w)
            b = np.fliplr(b)
            iod = np.fliplr(iod)

        zoom_f = float(rng.uniform(0.85, 1.15))
        if zoom_f != 1.0:
            w = _zoom_crop(w, zoom_f, IMAGE_SIZE)
            b = _zoom_crop(b, zoom_f, IMAGE_SIZE)
            iod = _zoom_crop(iod, zoom_f, IMAGE_SIZE)

        phantoms.append((
            np.maximum(w, 0.0), np.maximum(b, 0.0),
            np.maximum(iod, 0.0), f"dev_{name}"
        ))
    return phantoms


def generate_phantoms_hidden(
    n: int = 20,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, str]]:
    """Hidden-tier phantoms with adversarial modifications."""
    phantoms = []
    rng = np.random.default_rng(8000)
    for i in range(n):
        gen_fn = PHANTOM_GENERATORS[i % 3]
        w, b, iod, name = gen_fn(IMAGE_SIZE, IMAGE_SIZE, seed=800 + i, variant=i + 10)

        # Aggressive spatial augmentation
        angle = float(rng.uniform(20, 340))
        w = nd_rotate(w, angle, reshape=False, mode='constant', cval=0.0)
        b = nd_rotate(b, angle, reshape=False, mode='constant', cval=0.0)
        iod = nd_rotate(iod, angle, reshape=False, mode='constant', cval=0.0)

        if rng.random() < 0.7:
            w = np.fliplr(w)
            b = np.fliplr(b)
            iod = np.fliplr(iod)
        if rng.random() < 0.5:
            w = np.flipud(w)
            b = np.flipud(b)
            iod = np.flipud(iod)

        zoom_f = float(rng.uniform(0.70, 1.30))
        w = _zoom_crop(w, zoom_f, IMAGE_SIZE)
        b = _zoom_crop(b, zoom_f, IMAGE_SIZE)
        iod = _zoom_crop(iod, zoom_f, IMAGE_SIZE)

        # Add subtle micro-lesions (hard to detect iodine spots)
        n_micro = rng.integers(2, 6)
        for _ in range(n_micro):
            cy = rng.integers(40, IMAGE_SIZE - 40)
            cx = rng.integers(40, IMAGE_SIZE - 40)
            r = rng.integers(2, 5)
            yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
            circle = (yy**2 + xx**2 <= r**2).astype(np.float64)
            y0, y1 = max(0, cy - r), min(IMAGE_SIZE, cy + r + 1)
            x0, x1 = max(0, cx - r), min(IMAGE_SIZE, cx + r + 1)
            c_y0 = r - (cy - y0)
            c_y1 = r + (y1 - cy)
            c_x0 = r - (cx - x0)
            c_x1 = r + (x1 - cx)
            patch = circle[c_y0:c_y1, c_x0:c_x1]
            # Micro iodine uptake (subtle enhancing lesion)
            if w[y0:y1, x0:x1].mean() > 0.1:
                iod[y0:y1, x0:x1] = np.maximum(
                    iod[y0:y1, x0:x1],
                    patch * rng.uniform(1.0, 5.0)
                )

        phantoms.append((
            np.maximum(w, 0.0), np.maximum(b, 0.0),
            np.maximum(iod, 0.0), f"hidden_{name}"
        ))
    return phantoms


def _zoom_crop(arr: np.ndarray, zoom_f: float, size: int) -> np.ndarray:
    """Zoom and crop/pad back to target size."""
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


# -- Dual-Energy CT Forward Model ---------------------------------------------

def compute_mu_map(
    x_water: np.ndarray,
    x_bone: np.ndarray,
    x_iodine: np.ndarray,
    energy: str,
    energy_shift_keV: float = 0.0,
) -> np.ndarray:
    """Compute linear attenuation map for a given energy.

    Args:
        x_water:  (H, W) water density map [g/cm^3]
        x_bone:   (H, W) bone density map [g/cm^3]
        x_iodine: (H, W) iodine concentration map [mg/mL]
        energy:   "low" (80 kVp) or "high" (140 kVp)
        energy_shift_keV: calibration error shifts effective energy

    Returns:
        mu: (H, W) linear attenuation map [cm^-1]
    """
    if energy == "low":
        # Energy calibration error: shift attenuation coefficients
        # Approximate: +1 keV shifts coefficients by ~-2% for photoelectric-dominated materials
        # and ~-0.5% for Compton-dominated
        shift_factor_pe = 1.0 - 0.02 * energy_shift_keV   # photoelectric (bone, iodine)
        shift_factor_c  = 1.0 - 0.005 * energy_shift_keV  # Compton (water)
        alpha_w = ALPHA_WATER_LOW * shift_factor_c
        alpha_b = ALPHA_BONE_LOW * shift_factor_pe
        alpha_i = ALPHA_IODINE_LOW * shift_factor_pe
    else:
        shift_factor_pe = 1.0 - 0.015 * energy_shift_keV
        shift_factor_c  = 1.0 - 0.004 * energy_shift_keV
        alpha_w = ALPHA_WATER_HIGH * shift_factor_c
        alpha_b = ALPHA_BONE_HIGH * shift_factor_pe
        alpha_i = ALPHA_IODINE_HIGH * shift_factor_pe

    mu = (alpha_w * x_water +
          alpha_b * x_bone +
          alpha_i * x_iodine * IODINE_CONC_TO_DENSITY)

    return np.maximum(mu, 0.0)


def spectral_ct_forward(
    x_water: np.ndarray,
    x_bone: np.ndarray,
    x_iodine: np.ndarray,
    theta_deg: np.ndarray,
    energy_calibration_error: float,
    scatter_fraction: float,
    detector_crosstalk: float,
    beam_hardening: float,
    rng: np.random.Generator,
) -> dict:
    """Full dual-energy CT forward model with mismatch effects.

    Returns dict with sinograms at both energies and intermediate results.
    """
    n_angles = len(theta_deg)

    # 1. Compute attenuation maps at both energies (with energy calibration error)
    mu_low = compute_mu_map(x_water, x_bone, x_iodine, "low",
                            energy_shift_keV=energy_calibration_error)
    mu_high = compute_mu_map(x_water, x_bone, x_iodine, "high",
                             energy_shift_keV=energy_calibration_error)

    # Also compute ideal attenuation maps (no calibration error) for reference
    mu_low_ideal = compute_mu_map(x_water, x_bone, x_iodine, "low", 0.0)
    mu_high_ideal = compute_mu_map(x_water, x_bone, x_iodine, "high", 0.0)

    # 2. Radon transform (line integrals of linear attenuation)
    # Scale by pixel size to get physical path length (cm)
    sino_low = radon_transform(mu_low, theta_deg) * PIXEL_SIZE_CM
    sino_high = radon_transform(mu_high, theta_deg) * PIXEL_SIZE_CM

    sino_low_ideal = radon_transform(mu_low_ideal, theta_deg) * PIXEL_SIZE_CM
    sino_high_ideal = radon_transform(mu_high_ideal, theta_deg) * PIXEL_SIZE_CM

    sino_low = np.maximum(sino_low, 0.0)
    sino_high = np.maximum(sino_high, 0.0)
    sino_low_ideal = np.maximum(sino_low_ideal, 0.0)
    sino_high_ideal = np.maximum(sino_high_ideal, 0.0)

    n_det = sino_low.shape[1]

    # 3. Beam hardening: p_eff = p + beta * p^2
    if beam_hardening > 0:
        sino_low = sino_low + beam_hardening * sino_low**2
        sino_high = sino_high + beam_hardening * sino_high**2 * 0.5  # less BH at higher energy

    # 4. Beer-Lambert: I = I0 * exp(-p)
    I_low = I0_LOW * np.exp(-sino_low)
    I_high = I0_HIGH * np.exp(-sino_high)

    # 5. Scatter contamination (smooth additive background)
    if scatter_fraction > 0:
        # Scatter proportional to total signal level
        scatter_low = scatter_fraction * I_low.mean() * np.ones_like(I_low)
        scatter_var = rng.standard_normal(I_low.shape) * scatter_low * 0.1
        scatter_low += gaussian_filter(scatter_var, sigma=[5.0, 8.0])
        scatter_low = np.maximum(scatter_low, 0.0)

        scatter_high = scatter_fraction * I_high.mean() * np.ones_like(I_high)
        scatter_var_h = rng.standard_normal(I_high.shape) * scatter_high * 0.1
        scatter_high += gaussian_filter(scatter_var_h, sigma=[5.0, 8.0])
        scatter_high = np.maximum(scatter_high, 0.0)

        I_low = I_low + scatter_low
        I_high = I_high + scatter_high

    # 6. Detector crosstalk (energy leakage between low and high channels)
    if detector_crosstalk > 0:
        I_low_ct = I_low * (1.0 - detector_crosstalk) + I_high * detector_crosstalk
        I_high_ct = I_high * (1.0 - detector_crosstalk) + I_low * detector_crosstalk
        I_low = I_low_ct
        I_high = I_high_ct

    # 7. Poisson noise + readout noise
    I_low_noisy = rng.poisson(np.maximum(I_low, 1.0)).astype(np.float64)
    I_low_noisy += rng.normal(0.0, SIGMA_READOUT, I_low_noisy.shape)
    I_low_noisy = np.maximum(I_low_noisy, 1.0)

    I_high_noisy = rng.poisson(np.maximum(I_high, 1.0)).astype(np.float64)
    I_high_noisy += rng.normal(0.0, SIGMA_READOUT, I_high_noisy.shape)
    I_high_noisy = np.maximum(I_high_noisy, 1.0)

    # 8. Convert back to line integrals (sinograms in nepers)
    y_low = -np.log(I_low_noisy / I0_LOW)
    y_high = -np.log(I_high_noisy / I0_HIGH)

    return {
        "y_low": y_low.astype(np.float32),
        "y_high": y_high.astype(np.float32),
        "sino_low_ideal": sino_low_ideal.astype(np.float32),
        "sino_high_ideal": sino_high_ideal.astype(np.float32),
        "mu_low_ideal": mu_low_ideal.astype(np.float32),
        "mu_high_ideal": mu_high_ideal.astype(np.float32),
    }


# -- Metrics ------------------------------------------------------------------

def compute_psnr(gt: np.ndarray, recon: np.ndarray) -> float:
    mse = np.mean((gt.astype(np.float64) - recon.astype(np.float64))**2)
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
    c1 = (0.01 * data_range)**2
    c2 = (0.03 * data_range)**2
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


# -- Naive material decomposition (baseline) -----------------------------------

def naive_material_decomposition(
    y_low: np.ndarray,
    y_high: np.ndarray,
    theta_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Naive 2-material decomposition using FBP + linear inversion.

    This is a baseline approach:
    1. FBP reconstruct at each energy to get mu_low, mu_high images
    2. Solve 2x2 system for water and bone at each pixel
    3. Estimate iodine from residual

    Returns (water_est, bone_est, iodine_est) all (H, W) float32.
    """
    from generate_dataset import radon_transform  # just to confirm import works
    # Use simple FBP from pet module
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pet"))
    from generate_dataset import fbp_reconstruct  # noqa: E402

    # FBP at each energy
    recon_low = fbp_reconstruct(y_low, theta_deg, IMAGE_SIZE)
    recon_high = fbp_reconstruct(y_high, theta_deg, IMAGE_SIZE)

    # Scale back to attenuation (approximate inverse of pixel_size scaling)
    recon_low = np.maximum(recon_low / PIXEL_SIZE_CM, 0.0)
    recon_high = np.maximum(recon_high / PIXEL_SIZE_CM, 0.0)

    # 2-material decomposition: solve [alpha_w_low, alpha_b_low; alpha_w_high, alpha_b_high] * [w; b] = [mu_low; mu_high]
    A = np.array([
        [ALPHA_WATER_LOW, ALPHA_BONE_LOW],
        [ALPHA_WATER_HIGH, ALPHA_BONE_HIGH],
    ])
    A_inv = np.linalg.inv(A)

    water_est = A_inv[0, 0] * recon_low + A_inv[0, 1] * recon_high
    bone_est = A_inv[1, 0] * recon_low + A_inv[1, 1] * recon_high

    # Estimate iodine from residual
    residual_low = recon_low - (ALPHA_WATER_LOW * water_est + ALPHA_BONE_LOW * bone_est)
    iodine_est = np.maximum(residual_low / (ALPHA_IODINE_LOW * IODINE_CONC_TO_DENSITY), 0.0)

    return (
        np.maximum(water_est, 0.0).astype(np.float32),
        np.maximum(bone_est, 0.0).astype(np.float32),
        iodine_est.astype(np.float32),
    )


# -- Tier generation -----------------------------------------------------------

def sample_mismatch(rng: np.random.Generator, spec: dict) -> dict:
    return {k: float(rng.uniform(v["min"], v["max"])) for k, v in spec.items()}


def generate_tier(
    tier: str,
    phantoms: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]],
    base_seed: int,
) -> None:
    """Generate one tier of the spectral CT benchmark."""
    spec_ranges = SPEC[tier]
    tier_dir = BENCHMARK_DIR / tier
    images_dir = tier_dir / "images"
    if tier_dir.exists():
        shutil.rmtree(tier_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    theta_deg = np.linspace(0, 180, N_ANGLES, endpoint=False).astype(np.float64)

    h5_path = tier_dir / f"spectral_ct_challenge_{tier}.h5"
    rng = np.random.default_rng(base_seed)
    true_specs = {}

    with h5py.File(h5_path, "w") as f:
        f.attrs["description"] = (
            f"PWM Spectral CT (Dual-Energy) benchmark -- {tier} tier "
            f"(parallel-beam Radon + Poisson noise + material decomposition)"
        )
        f.attrs["spec_ranges"] = json.dumps(spec_ranges)
        f.attrs["geometry"] = json.dumps({
            "image_size": IMAGE_SIZE,
            "n_angles": N_ANGLES,
            "n_det": N_DET,
            "angle_range_deg": [0, 180],
            "fov_cm": FOV_CM,
            "pixel_size_cm": PIXEL_SIZE_CM,
            "I0_low": I0_LOW,
            "I0_high": I0_HIGH,
            "energy_low_kVp": 80,
            "energy_high_kVp": 140,
        })
        f.attrs["forward_model"] = (
            "y_low = -log(Poisson(I0_low * exp(-Radon(mu_low))) / I0_low), "
            "y_high = -log(Poisson(I0_high * exp(-Radon(mu_high))) / I0_high), "
            "mu_low = alpha_w_low*x_water + alpha_b_low*x_bone + alpha_i_low*x_iodine, "
            "mu_high = alpha_w_high*x_water + alpha_b_high*x_bone + alpha_i_high*x_iodine"
        )
        f.attrs["attenuation_coefficients"] = json.dumps({
            "low_energy_kVp": 80,
            "high_energy_kVp": 140,
            "alpha_water_low": ALPHA_WATER_LOW,
            "alpha_bone_low": ALPHA_BONE_LOW,
            "alpha_iodine_low": ALPHA_IODINE_LOW,
            "alpha_water_high": ALPHA_WATER_HIGH,
            "alpha_bone_high": ALPHA_BONE_HIGH,
            "alpha_iodine_high": ALPHA_IODINE_HIGH,
        })

        for idx, (x_water, x_bone, x_iodine, scene_name) in enumerate(phantoms):
            key = f"sample_{idx:02d}"
            print(f"  [{tier}] Generating {key} ({scene_name})...", end="", flush=True)

            mis = sample_mismatch(rng, spec_ranges)
            true_specs[key] = mis

            # Forward model
            result = spectral_ct_forward(
                x_water, x_bone, x_iodine, theta_deg,
                energy_calibration_error=mis["energy_calibration_error"],
                scatter_fraction=mis["scatter_fraction"],
                detector_crosstalk=mis["detector_crosstalk"],
                beam_hardening=mis["beam_hardening"],
                rng=rng,
            )

            y_low = result["y_low"]
            y_high = result["y_high"]

            # Save to HDF5
            grp = f.create_group(key)

            # Ground truth material maps
            grp.create_dataset("x_water", data=x_water.astype(np.float32),
                               compression="gzip")
            grp.create_dataset("x_bone", data=x_bone.astype(np.float32),
                               compression="gzip")
            grp.create_dataset("x_iodine", data=x_iodine.astype(np.float32),
                               compression="gzip")

            # Composite x_true for compatibility (sum of weighted materials)
            x_true = x_water + x_bone + x_iodine * IODINE_CONC_TO_DENSITY
            grp.create_dataset("x_true", data=x_true.astype(np.float32),
                               compression="gzip")

            # Measured sinograms
            grp.create_dataset("y_low", data=y_low, compression="gzip")
            grp.create_dataset("y_high", data=y_high, compression="gzip")

            # Ideal sinograms (no mismatch, no noise)
            grp.create_dataset("sino_low_ideal", data=result["sino_low_ideal"],
                               compression="gzip")
            grp.create_dataset("sino_high_ideal", data=result["sino_high_ideal"],
                               compression="gzip")

            # Projection angles
            grp.create_dataset("angles_deg", data=theta_deg.astype(np.float32))

            # Metadata
            grp.attrs["metadata"] = json.dumps({
                "scene": scene_name,
                "shape": list(x_water.shape),
                "n_angles": N_ANGLES,
                "n_det": int(y_low.shape[1]),
                "materials": ["water", "bone", "iodine"],
                "water_range": [float(x_water.min()), float(x_water.max())],
                "bone_range": [float(x_bone.min()), float(x_bone.max())],
                "iodine_range": [float(x_iodine.min()), float(x_iodine.max())],
            })
            grp.attrs["true_spec"] = json.dumps(mis)
            grp.attrs["spec_ranges"] = json.dumps(spec_ranges)

            # Save images
            sample_dir = images_dir / f"sample_{idx:02d}_{scene_name}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            _save_png(x_water, sample_dir / "x_water.png")
            _save_png(x_bone, sample_dir / "x_bone.png")
            _save_png(x_iodine, sample_dir / "x_iodine.png")
            _save_png(x_true, sample_dir / "ground_truth.png")
            _save_png(y_low, sample_dir / "y_low.png")
            _save_png(y_high, sample_dir / "y_high.png")
            with open(sample_dir / "spec.json", "w") as sf:
                json.dump({
                    "scene": scene_name,
                    "spec_ranges": spec_ranges,
                    "true_spec": mis,
                }, sf, indent=2)

            print(f"  ecal={mis['energy_calibration_error']:.2f} keV  "
                  f"scatter={mis['scatter_fraction']:.3f}  "
                  f"xtalk={mis['detector_crosstalk']:.3f}  "
                  f"BH={mis['beam_hardening']:.3f}")

    # Save spec files
    with open(tier_dir / "spec.json", "w") as sf:
        json.dump(spec_ranges, sf, indent=2)
    with open(tier_dir / "true_spec.json", "w") as tf:
        json.dump(true_specs, tf, indent=2)

    print(f"  [{tier}] {len(phantoms)} samples -> {h5_path.name}")


# -- Gallery image generation --------------------------------------------------

def generate_gallery_images() -> None:
    """Generate gallery images for the platform benchmark page."""
    gallery_base = (BENCHMARK_DIR.parent.parent.parent /
                    "platform" / "pwm_platform" / "static" / "img" /
                    "benchmark_gallery" / "spectral_ct")

    h5_path = BENCHMARK_DIR / "public" / "spectral_ct_challenge_public.h5"
    if not h5_path.exists():
        print("  [gallery] Public HDF5 not found, skipping gallery generation.")
        return

    # Pick diverse samples: head(0), abdomen(4), chest(8), head variant(3)
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
            x_water = grp["x_water"][:]
            x_bone = grp["x_bone"][:]
            x_iodine = grp["x_iodine"][:]
            y_low = grp["y_low"][:]
            y_high = grp["y_high"][:]

            # gt.png -- composite ground truth
            _save_png(x_true, scene_dir / "gt.png")

            # measurement_I.png -- low energy sinogram
            _save_png(y_low, scene_dir / "measurement_I.png")

            # measurement_II.png -- high energy sinogram
            _save_png(y_high, scene_dir / "measurement_II.png")

            # recon_I.png -- water map
            _save_png(x_water, scene_dir / "recon_I.png")

            # recon_II.png -- bone map
            _save_png(x_bone, scene_dir / "recon_II.png")

            # recon_III.png -- iodine map
            _save_png(x_iodine, scene_dir / "recon_III.png")

            print(f"  [gallery] scene_{scene_idx:02d} saved to {scene_dir}")


# -- Main ---------------------------------------------------------------------

def main() -> None:
    print("Spectral CT (Dual-Energy) Benchmark Dataset Generator")
    print("=" * 68)
    print(f"Output: {BENCHMARK_DIR}")
    print(f"Geometry: {N_ANGLES} angles, {IMAGE_SIZE}x{IMAGE_SIZE} images")
    print(f"Materials: water, bone, iodine")
    print(f"Energies: 80 kVp (low) / 140 kVp (high)\n")

    # -- Public tier (12 samples) --
    print("Generating public tier (12 samples)...")
    public_phantoms = generate_phantoms_public(12)
    generate_tier("public", public_phantoms, base_seed=1000)

    # -- Dev tier (20 samples) --
    print("\nGenerating dev tier (20 samples)...")
    dev_phantoms = generate_phantoms_dev(20)
    generate_tier("dev", dev_phantoms, base_seed=10000 + 5000)

    # -- Hidden tier (20 samples) --
    print("\nGenerating hidden tier (20 samples)...")
    hidden_phantoms = generate_phantoms_hidden(20)
    generate_tier("hidden", hidden_phantoms, base_seed=20000 + 8000)

    # -- Gallery images --
    print("\nGenerating gallery images...")
    generate_gallery_images()

    print(f"\n{'=' * 68}")
    print("Spectral CT benchmark dataset generation complete!")
    print(f"Output: {BENCHMARK_DIR}")
    print("=" * 68)


if __name__ == "__main__":
    main()
