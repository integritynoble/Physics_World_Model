#!/usr/bin/env python3
"""Generate Optical Coherence Tomography (OCT) benchmark dataset.

Forward model (B-scan degradation):
    bscan_ideal(z, x) = PSF_axial(z) * reflectivity(z, x)
    bscan_atten(z, x) = bscan_ideal * exp(-rolloff * z)
    bscan_speckle(z, x) = bscan_atten * speckle_noise
    bscan_measured(z, x) = shift(bscan_speckle, motion) + shot_noise

where:
    reflectivity(z, x) -- 2D cross-sectional tissue reflectivity (ground truth)
    PSF_axial           -- axial point spread function (Gaussian, FWHM = coherence length)
    rolloff             -- depth-dependent signal attenuation (dB/mm)
    speckle_noise       -- multiplicative Rayleigh noise (coherent interference)
    motion              -- lateral shift artifact (px)
    shot_noise          -- additive Gaussian noise

Ground truth phantoms (256x256 B-scans):
    Retinal OCT layers: ILM, NFL, GCL, IPL, INL, OPL, ONL, IS/OS, RPE, choroid
    Pathological features: drusen, cysts, detachments
    Anterior segment: cornea, iris, lens, angle structures

Mismatch parameters:
    speckle_snr_db     : speckle noise level (22-35 dB public, 15-35 dB hidden)
    axial_psf_fwhm_um  : axial resolution (3-8 um public, 3-15 um hidden)
    motion_artifact_px  : lateral motion shift (0-3 px public, 0-10 px hidden)
    signal_rolloff_db   : depth-dependent signal falloff (2-6 dB/mm)

Tiers:
    public : 12 samples (4 normal retina + 4 pathological + 4 anterior segment)
    dev    : 20 samples (augmented, medium mismatch)
    hidden : 20 samples (adversarial, wide mismatch)

CPU reconstruction: Median filtering + bilateral denoising (speckle reduction)

Usage:
    cd datasets/benchmark/oct
    python3 generate_dataset.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_filter1d, median_filter, shift
from scipy.signal import fftconvolve

BENCHMARK_DIR = Path(__file__).resolve().parent
IMAGE_SIZE = 256

# -- Physics constants --------------------------------------------------------

PIXEL_SIZE_UM = 3.0          # um per pixel (axial and lateral)
DYNAMIC_RANGE_DB = 50.0      # log-compression dynamic range
CENTRE_WAVELENGTH_NM = 850.0 # broadband source centre wavelength

# -- Mismatch spec ranges per tier -------------------------------------------

SPEC = {
    "public": {
        "speckle_snr_db":     {"min": 22.0, "max": 35.0, "unit": "dB"},
        "axial_psf_fwhm_um":  {"min": 3.0,  "max": 8.0,  "unit": "um"},
        "motion_artifact_px": {"min": 0.0,  "max": 3.0,  "unit": "pixels"},
        "signal_rolloff_db":  {"min": 2.0,  "max": 6.0,  "unit": "dB/mm"},
    },
    "dev": {
        "speckle_snr_db":     {"min": 18.0, "max": 35.0, "unit": "dB"},
        "axial_psf_fwhm_um":  {"min": 3.0,  "max": 12.0, "unit": "um"},
        "motion_artifact_px": {"min": 0.0,  "max": 6.0,  "unit": "pixels"},
        "signal_rolloff_db":  {"min": 2.0,  "max": 8.0,  "unit": "dB/mm"},
    },
    "hidden": {
        "speckle_snr_db":     {"min": 15.0, "max": 35.0, "unit": "dB"},
        "axial_psf_fwhm_um":  {"min": 3.0,  "max": 15.0, "unit": "um"},
        "motion_artifact_px": {"min": 0.0,  "max": 10.0, "unit": "pixels"},
        "signal_rolloff_db":  {"min": 2.0,  "max": 10.0, "unit": "dB/mm"},
    },
}

# -- Retinal layer definitions ------------------------------------------------
# Each layer: (name, relative_depth_fraction, thickness_fraction, reflectivity)
# Reflectivity: 0.0 = transparent, 1.0 = maximum backscatter
# Bright layers: NFL (~0.85), IS/OS junction (~0.9), RPE (~0.95)
# Dark layers: ONL (~0.15), GCL (~0.25)

RETINAL_LAYERS = [
    # (name, depth_start_frac, thickness_frac, reflectivity, brightness_label)
    ("ILM",     0.15, 0.005, 0.70, "bright_boundary"),
    ("NFL",     0.155, 0.03,  0.85, "bright"),
    ("GCL",     0.185, 0.025, 0.30, "dark"),
    ("IPL",     0.21,  0.03,  0.50, "medium"),
    ("INL",     0.24,  0.025, 0.40, "medium_dark"),
    ("OPL",     0.265, 0.02,  0.55, "medium"),
    ("ONL",     0.285, 0.06,  0.15, "dark"),
    ("ELM",     0.345, 0.003, 0.60, "boundary"),
    ("IS/OS",   0.348, 0.008, 0.90, "very_bright"),
    ("OS",      0.356, 0.02,  0.35, "medium_dark"),
    ("RPE",     0.376, 0.015, 0.95, "very_bright"),
    ("BruchM",  0.391, 0.003, 0.65, "bright_boundary"),
    ("Choroid", 0.394, 0.12,  0.45, "medium"),
]


# -- Phantom generators -------------------------------------------------------

def _smooth_boundary(
    size: int,
    base_y: float,
    rng: np.random.Generator,
    amplitude: float = 3.0,
    n_harmonics: int = 5,
) -> np.ndarray:
    """Generate a smooth undulating boundary curve (1D, length=size).

    Creates natural-looking retinal layer curvature using superposition
    of sinusoidal harmonics with random phases.
    """
    x = np.linspace(0, 2 * np.pi, size, dtype=np.float64)
    curve = np.full(size, base_y, dtype=np.float64)

    # Global curvature (foveal depression / optic disc shape)
    global_freq = rng.uniform(0.3, 0.8)
    global_amp = rng.uniform(5.0, 15.0)
    global_phase = rng.uniform(0, 2 * np.pi)
    curve += global_amp * np.sin(global_freq * x + global_phase)

    # Local undulations
    for _ in range(n_harmonics):
        freq = rng.uniform(1.0, 6.0)
        amp = rng.uniform(0.5, amplitude)
        phase = rng.uniform(0, 2 * np.pi)
        curve += amp * np.sin(freq * x + phase)

    return curve


def _phantom_normal_retina(
    rng: np.random.Generator,
    size: int = IMAGE_SIZE,
    foveal: bool = False,
) -> np.ndarray:
    """Generate a normal retinal OCT B-scan phantom.

    Creates layered structure with characteristic bright/dark bands:
    - NFL, IS/OS, RPE are bright horizontal bands
    - ONL, GCL are dark layers
    - Natural foveal depression or optic disc curvature
    - Sub-pixel texture within layers
    """
    phantom = np.zeros((size, size), dtype=np.float64)

    # Background below retina (vitreous above, sclera below)
    phantom[:, :] = 0.02  # vitreous (dark)

    # Reference retinal position (top of retina)
    retina_top = size * 0.15
    if foveal:
        # Foveal depression: thinner at centre
        x_arr = np.arange(size, dtype=np.float64)
        fovea_x = size * rng.uniform(0.35, 0.65)
        fovea_width = rng.uniform(25.0, 45.0)
        fovea_depth = rng.uniform(8.0, 18.0)
        fovea_dip = fovea_depth * np.exp(-0.5 * ((x_arr - fovea_x) / fovea_width) ** 2)
    else:
        fovea_dip = np.zeros(size, dtype=np.float64)

    # Build each layer with smooth boundaries
    for name, depth_frac, thick_frac, reflectivity, _ in RETINAL_LAYERS:
        base_top = size * depth_frac
        base_bot = size * (depth_frac + thick_frac)

        top_curve = _smooth_boundary(size, base_top, rng, amplitude=2.0)
        bot_curve = _smooth_boundary(size, base_bot, rng, amplitude=2.0)

        # Apply foveal depression to inner layers (NFL thins at fovea)
        if name in ("NFL", "GCL", "IPL", "INL"):
            top_curve += fovea_dip * 0.3
            bot_curve += fovea_dip * 0.15
        elif name in ("OPL", "ONL", "ELM"):
            top_curve += fovea_dip * 0.1

        # Add intra-layer texture (sub-resolution scatterer variation)
        texture = gaussian_filter(
            rng.standard_normal((size, size)), sigma=2.0
        ) * reflectivity * 0.08

        for col in range(size):
            row_top = int(np.clip(top_curve[col], 0, size - 1))
            row_bot = int(np.clip(bot_curve[col], 0, size - 1))
            if row_top < row_bot:
                phantom[row_top:row_bot, col] = reflectivity
                phantom[row_top:row_bot, col] += texture[row_top:row_bot, col]

    # Below RPE/choroid: sclera (moderate reflectivity fading)
    sclera_top = int(size * 0.52)
    if sclera_top < size:
        depth = np.arange(size - sclera_top, dtype=np.float64)
        sclera_decay = 0.30 * np.exp(-depth / 40.0)
        phantom[sclera_top:, :] += sclera_decay[:, None]

    # Vitreous above retina (very dark, near zero)
    vit_bot = int(size * 0.14)
    phantom[:vit_bot, :] = 0.01 + rng.uniform(0.0, 0.005, (vit_bot, size))

    return np.clip(phantom, 0.0, 1.0).astype(np.float32)


def _phantom_pathological_retina(
    rng: np.random.Generator,
    size: int = IMAGE_SIZE,
    pathology: str = "drusen",
) -> np.ndarray:
    """Generate pathological retinal OCT phantom.

    Pathologies:
    - drusen: sub-RPE deposits (bumps elevating RPE)
    - cyst: intraretinal fluid pockets (dark voids in INL/ONL)
    - detachment: RPE or neurosensory detachment (fluid under RPE)
    - epiretinal_membrane: bright line above ILM with traction
    """
    # Start from normal retina
    phantom = _phantom_normal_retina(rng, size, foveal=(rng.random() > 0.5))

    if pathology == "drusen":
        # Sub-RPE deposits: bright bumps that push RPE upward
        n_drusen = rng.integers(3, 8)
        for _ in range(n_drusen):
            cx = rng.integers(size // 6, 5 * size // 6)
            width = rng.uniform(10.0, 35.0)
            height = rng.uniform(5.0, 15.0)
            # RPE region rows
            rpe_row = int(size * 0.376)
            x_arr = np.arange(size, dtype=np.float64)
            bump = height * np.exp(-0.5 * ((x_arr - cx) / width) ** 2)
            for col in range(size):
                bump_px = int(bump[col])
                if bump_px > 0:
                    row_start = max(0, rpe_row - bump_px)
                    # Drusen material (bright deposit)
                    phantom[row_start:rpe_row, col] = rng.uniform(0.7, 0.85)
                    # Push RPE up
                    if row_start > 2:
                        phantom[row_start - 2:row_start, col] = 0.95

    elif pathology == "cyst":
        # Intraretinal fluid-filled cysts (dark voids)
        n_cysts = rng.integers(2, 6)
        for _ in range(n_cysts):
            cx = rng.integers(size // 5, 4 * size // 5)
            cy = rng.integers(int(size * 0.20), int(size * 0.34))
            rx = rng.integers(8, 25)
            ry = rng.integers(5, 18)
            yy, xx = np.ogrid[:size, :size]
            dist = ((yy - cy) / max(ry, 1)) ** 2 + ((xx - cx) / max(rx, 1)) ** 2
            # Cyst interior (fluid = very dark)
            cyst_mask = dist <= 1.0
            phantom[cyst_mask] = rng.uniform(0.01, 0.04)
            # Bright border (cyst wall)
            wall_mask = (dist <= 1.3) & (dist > 1.0)
            phantom[wall_mask] = np.maximum(phantom[wall_mask], 0.65)

    elif pathology == "detachment":
        # Sub-RPE fluid (serous detachment)
        det_cx = rng.integers(size // 4, 3 * size // 4)
        det_width = rng.uniform(40.0, 80.0)
        det_height = rng.uniform(10.0, 25.0)
        rpe_row = int(size * 0.376)
        x_arr = np.arange(size, dtype=np.float64)
        det_curve = det_height * np.exp(-0.5 * ((x_arr - det_cx) / det_width) ** 2)
        for col in range(size):
            gap = int(det_curve[col])
            if gap > 0:
                # Fluid space under RPE (dark)
                fluid_top = rpe_row
                fluid_bot = min(rpe_row + gap, size)
                phantom[fluid_top:fluid_bot, col] = rng.uniform(0.02, 0.05)
                # Displaced RPE below fluid
                if fluid_bot < size - 2:
                    phantom[fluid_bot:fluid_bot + 3, col] = 0.90

    elif pathology == "epiretinal_membrane":
        # Bright membrane above ILM with traction folds
        ilm_row = int(size * 0.145)
        membrane_curve = _smooth_boundary(size, ilm_row - 5, rng, amplitude=3.0)
        n_folds = rng.integers(2, 5)
        x_arr = np.arange(size, dtype=np.float64)
        for _ in range(n_folds):
            fold_x = rng.integers(size // 6, 5 * size // 6)
            fold_w = rng.uniform(5.0, 15.0)
            fold_a = rng.uniform(2.0, 6.0)
            membrane_curve += fold_a * np.exp(-0.5 * ((x_arr - fold_x) / fold_w) ** 2)
        for col in range(size):
            row = int(np.clip(membrane_curve[col], 0, size - 1))
            if 0 < row < size - 1:
                phantom[row - 1:row + 2, col] = rng.uniform(0.70, 0.85)

    return np.clip(phantom, 0.0, 1.0).astype(np.float32)


def _phantom_anterior_segment(
    rng: np.random.Generator,
    size: int = IMAGE_SIZE,
) -> np.ndarray:
    """Generate anterior segment OCT phantom.

    Structures: cornea (bright), anterior chamber (dark), iris (bright),
    lens (layered), angle structures (trabecular meshwork).
    """
    phantom = np.zeros((size, size), dtype=np.float64)

    # Cornea: bright curved band at top
    x_arr = np.arange(size, dtype=np.float64)
    cornea_cx = size * rng.uniform(0.4, 0.6)
    cornea_radius = rng.uniform(200.0, 350.0)
    cornea_thick = rng.uniform(8.0, 14.0)

    # Anterior corneal surface
    cornea_ant = size * 0.08 + (x_arr - cornea_cx) ** 2 / (2 * cornea_radius)
    cornea_post = cornea_ant + cornea_thick

    for col in range(size):
        r_ant = int(np.clip(cornea_ant[col], 0, size - 1))
        r_post = int(np.clip(cornea_post[col], 0, size - 1))
        if r_ant < r_post:
            # Epithelium (bright)
            phantom[r_ant:min(r_ant + 2, r_post), col] = rng.uniform(0.75, 0.85)
            # Stroma (moderate)
            if r_ant + 2 < r_post - 2:
                phantom[r_ant + 2:r_post - 2, col] = rng.uniform(0.35, 0.50)
            # Endothelium (bright)
            phantom[max(r_ant, r_post - 2):r_post, col] = rng.uniform(0.70, 0.80)

    # Anterior chamber (dark, aqueous humor)
    ac_top = cornea_post.astype(int)
    ac_depth = rng.uniform(50.0, 80.0)
    for col in range(size):
        top = int(np.clip(ac_top[col], 0, size - 1))
        bot = int(np.clip(top + ac_depth, 0, size - 1))
        phantom[top:bot, col] = rng.uniform(0.01, 0.03)

    # Iris: bright structures extending from sides
    iris_row = int(np.mean(ac_top) + ac_depth * 0.85)
    iris_thick = rng.integers(4, 8)
    # Left iris wing
    iris_left_end = rng.integers(size // 3, size // 2 - 10)
    phantom[iris_row:iris_row + iris_thick, :iris_left_end] = rng.uniform(0.70, 0.90)
    # Right iris wing
    iris_right_start = rng.integers(size // 2 + 10, 2 * size // 3)
    phantom[iris_row:iris_row + iris_thick, iris_right_start:] = rng.uniform(0.70, 0.90)

    # Pupil (dark gap between iris wings)
    # Already dark from initialization

    # Lens: layered structure below iris
    lens_top = iris_row + iris_thick + rng.integers(3, 8)
    lens_cx = size * rng.uniform(0.4, 0.6)
    lens_radius = rng.uniform(120.0, 180.0)
    lens_thick = rng.uniform(25.0, 40.0)

    lens_ant_curve = lens_top + (x_arr - lens_cx) ** 2 / (2 * lens_radius)
    lens_post_curve = lens_ant_curve + lens_thick

    for col in range(size):
        lt = int(np.clip(lens_ant_curve[col], 0, size - 1))
        lb = int(np.clip(lens_post_curve[col], 0, size - 1))
        if lt < lb and lt > iris_row:
            # Lens capsule (bright)
            phantom[lt:min(lt + 2, lb), col] = rng.uniform(0.75, 0.85)
            # Lens cortex (moderate)
            if lt + 2 < lb - 2:
                phantom[lt + 2:lb - 2, col] = rng.uniform(0.20, 0.35)
            # Posterior capsule (bright)
            phantom[max(lt, lb - 2):lb, col] = rng.uniform(0.70, 0.80)

    # Angle structures (trabecular meshwork, Schlemm's canal)
    # Bright wedge-shaped region at iris root
    angle_regions = [(0, iris_left_end), (iris_right_start, size)]
    for x_start, x_end in angle_regions:
        for col in range(max(0, x_start - 5), min(size, x_end + 5)):
            for row in range(max(0, iris_row - 10), min(size, iris_row + iris_thick + 5)):
                if 0 <= row < size and 0 <= col < size:
                    if phantom[row, col] < 0.1:
                        phantom[row, col] = rng.uniform(0.30, 0.50)

    # Add sub-pixel texture
    texture = gaussian_filter(rng.standard_normal((size, size)), sigma=1.5) * 0.04
    phantom += texture

    return np.clip(phantom, 0.0, 1.0).astype(np.float32)


# -- Scene name tables --------------------------------------------------------

PUBLIC_SCENE_NAMES = [
    "normal_retina_01", "normal_retina_02",
    "normal_foveal_01", "normal_foveal_02",
    "drusen_01", "cyst_01", "detachment_01", "epiretinal_01",
    "anterior_seg_01", "anterior_seg_02",
    "anterior_seg_03", "anterior_seg_04",
]

DEV_SCENE_NAMES = [f"dev_oct_{i:02d}" for i in range(20)]
HIDDEN_SCENE_NAMES = [f"hidden_oct_{i:02d}" for i in range(20)]


# -- Phantom generation per tier ----------------------------------------------

def generate_public_phantoms(
    rng: np.random.Generator,
) -> list[tuple[str, np.ndarray]]:
    """Generate 12 public phantoms: 4 normal + 4 pathological + 4 anterior."""
    phantoms = []
    # 2 normal retina (extrafoveal)
    for i in range(2):
        x = _phantom_normal_retina(rng, foveal=False)
        phantoms.append((PUBLIC_SCENE_NAMES[i], x))
    # 2 normal foveal
    for i in range(2):
        x = _phantom_normal_retina(rng, foveal=True)
        phantoms.append((PUBLIC_SCENE_NAMES[2 + i], x))
    # 4 pathological
    pathologies = ["drusen", "cyst", "detachment", "epiretinal_membrane"]
    for i, path in enumerate(pathologies):
        x = _phantom_pathological_retina(rng, pathology=path)
        phantoms.append((PUBLIC_SCENE_NAMES[4 + i], x))
    # 4 anterior segment
    for i in range(4):
        x = _phantom_anterior_segment(rng)
        phantoms.append((PUBLIC_SCENE_NAMES[8 + i], x))
    return phantoms


def generate_dev_phantoms(
    rng: np.random.Generator,
) -> list[tuple[str, np.ndarray]]:
    """Generate 20 dev phantoms: mixed types with augmentation."""
    phantoms = []
    generators = [
        lambda r: _phantom_normal_retina(r, foveal=r.random() > 0.5),
        lambda r: _phantom_pathological_retina(
            r, pathology=r.choice(["drusen", "cyst", "detachment",
                                   "epiretinal_membrane"])),
        lambda r: _phantom_anterior_segment(r),
    ]
    for i in range(20):
        gen = generators[i % len(generators)]
        x = gen(rng)
        # Augmentation: flips
        if rng.random() < 0.5:
            x = np.fliplr(x).copy()
        if rng.random() < 0.3:
            x = np.flipud(x).copy()
        # Small random intensity scaling
        scale = rng.uniform(0.85, 1.15)
        x = np.clip(x * scale, 0.0, 1.0).astype(np.float32)
        phantoms.append((DEV_SCENE_NAMES[i], x))
    return phantoms


def generate_hidden_phantoms(
    rng: np.random.Generator,
) -> list[tuple[str, np.ndarray]]:
    """Generate 20 hidden phantoms: adversarial complex pathology."""
    phantoms = []
    for i in range(20):
        if i < 5:
            # Multi-pathology: drusen + cyst on same retina
            x = _phantom_pathological_retina(rng, pathology="drusen")
            # Add cysts on top
            n_cysts = rng.integers(2, 5)
            for _ in range(n_cysts):
                cx = rng.integers(IMAGE_SIZE // 5, 4 * IMAGE_SIZE // 5)
                cy = rng.integers(int(IMAGE_SIZE * 0.20), int(IMAGE_SIZE * 0.34))
                rx = rng.integers(6, 18)
                ry = rng.integers(4, 12)
                yy, xx = np.ogrid[:IMAGE_SIZE, :IMAGE_SIZE]
                dist = ((yy - cy) / max(ry, 1)) ** 2 + \
                       ((xx - cx) / max(rx, 1)) ** 2
                x[dist <= 1.0] = rng.uniform(0.01, 0.04)
        elif i < 10:
            # Severe detachment + epiretinal membrane
            x = _phantom_pathological_retina(rng, pathology="detachment")
            # Add ERM
            ilm_row = int(IMAGE_SIZE * 0.14)
            for col in range(IMAGE_SIZE):
                r = ilm_row - rng.integers(3, 8)
                if 0 < r < IMAGE_SIZE - 1:
                    x[r:r + 2, col] = rng.uniform(0.70, 0.85)
        elif i < 15:
            # Very low contrast retina (adversarial for despeckling)
            x = _phantom_normal_retina(rng, foveal=rng.random() > 0.5)
            x = x * rng.uniform(0.25, 0.45)  # reduce overall contrast
            x = np.clip(x, 0.0, 1.0).astype(np.float32)
        else:
            # Complex anterior segment with cataracts
            x = _phantom_anterior_segment(rng)
            # Add cataract opacity in lens region
            lens_region = (x > 0.15) & (x < 0.40)
            scatter = gaussian_filter(
                rng.standard_normal(x.shape), sigma=3.0
            ) * 0.15
            x[lens_region] += np.abs(scatter[lens_region])
            x = np.clip(x, 0.0, 1.0).astype(np.float32)

        # Augmentation
        if rng.random() < 0.7:
            x = np.fliplr(x).copy()
        if rng.random() < 0.4:
            x = np.flipud(x).copy()
        phantoms.append((HIDDEN_SCENE_NAMES[i], x))
    return phantoms


# -- Forward model ------------------------------------------------------------

def make_axial_psf(
    fwhm_um: float,
    pixel_size_um: float = PIXEL_SIZE_UM,
) -> np.ndarray:
    """Generate 1D axial PSF (Gaussian) for OCT.

    The coherence length of the broadband source determines axial resolution.
    FWHM = coherence length ~= 0.44 * lambda^2 / delta_lambda.

    Returns:
        psf_1d: 1D array, normalized to sum=1
    """
    sigma_um = fwhm_um / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_px = sigma_um / pixel_size_um
    k = max(3, int(6 * sigma_px) | 1)
    if k % 2 == 0:
        k += 1
    k = min(k, 31)
    half = k // 2
    z = np.arange(-half, half + 1, dtype=np.float64)
    psf = np.exp(-0.5 * (z / max(sigma_px, 0.3)) ** 2)
    psf /= psf.sum() + 1e-12
    return psf.astype(np.float32)


def apply_axial_psf(
    x_true: np.ndarray,
    psf_1d: np.ndarray,
) -> np.ndarray:
    """Apply axial (depth) PSF convolution to each A-scan (column).

    The PSF blurs along the depth (row) axis, simulating the finite
    coherence length of the broadband source.
    """
    # Convolve along axis 0 (depth) for each lateral position
    k = len(psf_1d)
    psf_2d = psf_1d.reshape(-1, 1).astype(np.float64)
    result = fftconvolve(
        x_true.astype(np.float64), psf_2d, mode="same"
    )
    return np.abs(result).astype(np.float32)


def apply_signal_rolloff(
    bscan: np.ndarray,
    rolloff_db_per_mm: float,
    pixel_size_um: float = PIXEL_SIZE_UM,
) -> np.ndarray:
    """Apply depth-dependent signal attenuation (sensitivity roll-off).

    In SD-OCT, deeper structures have lower SNR due to:
    - Limited spectral resolution of the spectrometer
    - Tissue scattering/absorption

    The roll-off is modelled as exponential decay with depth.
    """
    H = bscan.shape[0]
    depth_mm = np.arange(H, dtype=np.float64) * pixel_size_um / 1000.0
    # One-way attenuation (signal travels round-trip, so 2x)
    atten_db = 2.0 * rolloff_db_per_mm * depth_mm
    gain = 10.0 ** (-atten_db / 20.0)
    return (bscan * gain[:, None]).astype(np.float32)


def apply_speckle_noise(
    bscan: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply multiplicative speckle noise (Rayleigh) + additive shot noise.

    OCT speckle arises from coherent interference of backscattered waves
    from sub-resolution scatterers. Modelled as multiplicative Rayleigh
    noise with contrast controlled by snr_db.

    Also adds additive Gaussian shot noise.
    """
    H, W = bscan.shape

    # Multiplicative speckle: Rayleigh-distributed
    # snr_db controls the speckle contrast: higher SNR = less speckle
    speckle_sigma = 1.0 / (10.0 ** (snr_db / 20.0))

    # Generate complex field with random phase
    real_part = 1.0 + rng.normal(0, speckle_sigma, (H, W))
    imag_part = rng.normal(0, speckle_sigma, (H, W))
    speckle_envelope = np.sqrt(real_part ** 2 + imag_part ** 2)

    bscan_speckled = bscan.astype(np.float64) * speckle_envelope

    # Additive shot noise (Gaussian, proportional to signal)
    noise_level = np.mean(np.abs(bscan_speckled)) * 0.02
    shot_noise = rng.normal(0, max(noise_level, 1e-6), (H, W))
    bscan_noisy = bscan_speckled + shot_noise

    return np.maximum(bscan_noisy, 1e-10).astype(np.float32)


def apply_motion_artifact(
    bscan: np.ndarray,
    motion_px: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply lateral motion artifact (A-scan displacement).

    During B-scan acquisition, patient motion (eye saccades, breathing)
    causes lateral displacement of consecutive A-scans. Modelled as
    random column shifts with spatial correlation.
    """
    if motion_px < 0.5:
        return bscan

    H, W = bscan.shape
    result = bscan.copy()

    # Generate smooth random lateral shifts for each column
    raw_shifts = rng.normal(0, motion_px, W)
    # Smooth to create correlated motion (not random per-column)
    smooth_shifts = gaussian_filter1d(raw_shifts, sigma=8.0)

    for col in range(W):
        s = smooth_shifts[col]
        if abs(s) > 0.1:
            result[:, col] = shift(
                bscan[:, col], s, mode="nearest"
            ).astype(np.float32)

    return result


def log_compress(
    bscan: np.ndarray,
    dynamic_range_db: float = DYNAMIC_RANGE_DB,
) -> np.ndarray:
    """Log-compress B-scan to [0, 1] with given dynamic range.

    Standard OCT display: y = 20*log10(B/B_max), clipped to [-DR, 0] dB,
    then scaled to [0, 1].
    """
    bscan_pos = np.maximum(bscan, 1e-10)
    b_max = bscan_pos.max()
    y_dB = 20.0 * np.log10(bscan_pos / max(b_max, 1e-10))
    y_clipped = np.clip(y_dB, -dynamic_range_db, 0.0)
    y_norm = (y_clipped + dynamic_range_db) / dynamic_range_db
    return y_norm.astype(np.float32)


def forward_model(
    x_true: np.ndarray,
    axial_psf_fwhm_um: float,
    signal_rolloff_db: float,
    speckle_snr_db: float,
    motion_artifact_px: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Complete OCT B-scan forward model.

    Pipeline:
        1. Axial PSF convolution (coherence length)
        2. Depth-dependent attenuation (signal roll-off)
        3. Multiplicative speckle noise (Rayleigh)
        4. Motion artifact (lateral A-scan shifts)
        5. Additive shot noise

    Returns:
        bscan_ideal:    clean B-scan (PSF + rolloff only)
        bscan_measured: degraded B-scan (all effects)
    """
    # Axial PSF
    psf_1d = make_axial_psf(axial_psf_fwhm_um)
    bscan = apply_axial_psf(x_true, psf_1d)

    # Signal roll-off
    bscan = apply_signal_rolloff(bscan, signal_rolloff_db)

    # Ideal B-scan (no noise, no motion)
    bscan_ideal = bscan.copy()

    # Speckle noise
    bscan = apply_speckle_noise(bscan, speckle_snr_db, rng)

    # Motion artifact
    bscan = apply_motion_artifact(bscan, motion_artifact_px, rng)

    return bscan_ideal, bscan


# -- CPU reconstruction: Median + bilateral denoising -------------------------

def reconstruct_despeckle(
    bscan_measured: np.ndarray,
    median_size: int = 3,
    bilateral_sigma_space: float = 3.0,
    bilateral_sigma_intensity: float = 0.1,
) -> np.ndarray:
    """Simple speckle reduction reconstruction.

    Two-stage pipeline:
    1. Median filter (removes impulse-like speckle)
    2. Bilateral-like denoising via guided Gaussian filtering
       (preserves edges while smoothing)

    This is a baseline CPU-only reconstruction without any deep learning.
    """
    # Stage 1: Median filter
    denoised = median_filter(bscan_measured, size=median_size).astype(np.float64)

    # Stage 2: Edge-preserving smoothing (bilateral approximation)
    # Use structure tensor to detect edges
    smooth = gaussian_filter(denoised, sigma=bilateral_sigma_space)

    # Compute local intensity difference
    diff = np.abs(denoised - smooth)
    weight = np.exp(-diff ** 2 / (2.0 * bilateral_sigma_intensity ** 2))

    # Blend: preserve edges, smooth flat regions
    result = weight * smooth + (1.0 - weight) * denoised

    # Final gentle smoothing
    result = gaussian_filter(result, sigma=0.5)

    return np.clip(result, 0.0, None).astype(np.float32)


# -- Metrics ------------------------------------------------------------------

def compute_psnr(gt: np.ndarray, recon: np.ndarray) -> float:
    g = gt.astype(np.float64)
    r = recon.astype(np.float64)
    mse = float(np.mean((g - r) ** 2))
    if mse < 1e-12:
        return 100.0
    dr = float(g.max() - g.min())
    if dr < 1e-12:
        return 0.0
    return float(10.0 * np.log10(dr ** 2 / mse))


def compute_ssim(gt: np.ndarray, recon: np.ndarray) -> float:
    g = gt.astype(np.float64)
    r = recon.astype(np.float64)
    dr = float(g.max() - g.min())
    if dr < 1e-12:
        return 0.0
    c1 = (0.01 * dr) ** 2
    c2 = (0.03 * dr) ** 2
    mu_g, mu_r = g.mean(), r.mean()
    var_g, var_r = g.var(), r.var()
    cov = float(np.mean((g - mu_g) * (r - mu_r)))
    return float(((2 * mu_g * mu_r + c1) * (2 * cov + c2)) /
                 ((mu_g ** 2 + mu_r ** 2 + c1) * (var_g + var_r + c2)))


# -- Image helpers ------------------------------------------------------------

def _norm(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-8:
        return np.zeros_like(a, dtype=np.float32)
    return ((a - lo) / (hi - lo)).astype(np.float32)


def _save_png(arr: np.ndarray, path: Path) -> None:
    normed = np.clip(_norm(arr) * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(normed, "L")
    if img.size != (IMAGE_SIZE, IMAGE_SIZE):
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    img.save(str(path))


def _save_overview(
    x_true: np.ndarray,
    bscan_ideal: np.ndarray,
    bscan_measured: np.ndarray,
    recon: np.ndarray,
    path: Path,
) -> None:
    """Save 2x2 overview panel: GT | Ideal | Measured | Recon (256x256)."""
    th, tw = 128, 128

    def _r(a: np.ndarray) -> np.ndarray:
        pil = Image.fromarray(
            np.clip(_norm(a) * 255, 0, 255).astype(np.uint8), "L"
        )
        return np.array(pil.resize((tw, th), Image.LANCZOS))

    top = np.hstack([_r(x_true), _r(bscan_ideal)])
    bot = np.hstack([_r(bscan_measured), _r(recon)])
    ov = np.vstack([top, bot])
    Image.fromarray(ov, "L").save(str(path))


# -- Mismatch sampling -------------------------------------------------------

def sample_mismatch(rng: np.random.Generator, spec: dict) -> dict:
    result = {}
    for k, v in spec.items():
        lo, hi = v["min"], v["max"]
        if isinstance(lo, int) and isinstance(hi, int):
            result[k] = int(rng.integers(lo, hi + 1))
        else:
            result[k] = float(rng.uniform(lo, hi))
    return result


# -- Tier generator -----------------------------------------------------------

def generate_tier(
    tier: str,
    phantoms: list[tuple[str, np.ndarray]],
    base_seed: int,
) -> None:
    """Generate one tier of the OCT benchmark."""
    spec_ranges = SPEC[tier]
    tier_dir = BENCHMARK_DIR / tier
    images_dir = tier_dir / "images"
    if tier_dir.exists():
        shutil.rmtree(tier_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    h5_path = tier_dir / f"oct_challenge_{tier}.h5"
    rng = np.random.default_rng(base_seed)
    rows, true_specs = [], {}
    psnr_list, ssim_list = [], []

    with h5py.File(h5_path, "w") as f:
        f.attrs["description"] = (
            f"PWM OCT benchmark -- {tier} tier "
            f"(axial PSF + speckle + rolloff + motion)")
        f.attrs["spec_ranges"] = json.dumps(spec_ranges)
        f.attrs["geometry"] = json.dumps({
            "image_size": IMAGE_SIZE,
            "pixel_size_um": PIXEL_SIZE_UM,
            "fov_um": IMAGE_SIZE * PIXEL_SIZE_UM,
            "centre_wavelength_nm": CENTRE_WAVELENGTH_NM,
            "dynamic_range_dB": DYNAMIC_RANGE_DB,
        })

        for idx, (scene_name, x_true) in enumerate(phantoms):
            key = f"sample_{idx:02d}"
            mis = sample_mismatch(rng, spec_ranges)
            true_specs[key] = {**mis, "scene": scene_name}

            bscan_ideal, bscan_measured = forward_model(
                x_true,
                axial_psf_fwhm_um=mis["axial_psf_fwhm_um"],
                signal_rolloff_db=mis["signal_rolloff_db"],
                speckle_snr_db=mis["speckle_snr_db"],
                motion_artifact_px=mis["motion_artifact_px"],
                rng=rng,
            )

            recon = reconstruct_despeckle(bscan_measured)
            gt_norm = _norm(x_true)
            recon_norm = _norm(recon)
            psnr = compute_psnr(gt_norm, recon_norm)
            ssim = compute_ssim(gt_norm, recon_norm)
            psnr_list.append(psnr)
            ssim_list.append(ssim)

            grp = f.create_group(key)
            grp.create_dataset("x_true", data=x_true, compression="gzip")
            grp.create_dataset("bscan_ideal", data=bscan_ideal,
                               compression="gzip")
            grp.create_dataset("bscan_measured", data=bscan_measured,
                               compression="gzip")

            grp.attrs["metadata"] = json.dumps({
                "scene": scene_name,
                "shape": list(x_true.shape),
                "baseline_psnr_dB": round(psnr, 2),
                "baseline_ssim": round(ssim, 4),
            })
            grp.attrs["spec_ranges"] = json.dumps(spec_ranges)
            grp.attrs["true_spec"] = json.dumps({**mis, "scene": scene_name})

            sample_dir = images_dir / f"sample_{idx:02d}_{scene_name}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            _save_png(x_true, sample_dir / "ground_truth.png")
            _save_png(bscan_ideal, sample_dir / "bscan_ideal.png")
            _save_png(bscan_measured, sample_dir / "bscan_measured.png")
            _save_png(recon, sample_dir / "reconstruction.png")
            _save_overview(x_true, bscan_ideal, bscan_measured, recon,
                           sample_dir / "overview.png")

            with open(sample_dir / "spec.json", "w") as sf:
                json.dump({
                    "scene": scene_name,
                    "spec_ranges": spec_ranges,
                    "true_spec": mis,
                    "baseline_psnr_dB": round(psnr, 2),
                    "baseline_ssim": round(ssim, 4),
                }, sf, indent=2)

            rows.append((key, scene_name, mis, psnr, ssim))
            print(f"  [{tier}] {key} {scene_name}  "
                  f"speckle={mis['speckle_snr_db']:.1f}dB  "
                  f"psf={mis['axial_psf_fwhm_um']:.1f}um  "
                  f"motion={mis['motion_artifact_px']:.1f}px  "
                  f"rolloff={mis['signal_rolloff_db']:.1f}dB/mm  "
                  f"PSNR={psnr:.2f}dB  SSIM={ssim:.4f}")

    with open(tier_dir / "spec.json", "w") as sf:
        json.dump(spec_ranges, sf, indent=2)
    with open(tier_dir / "true_spec.json", "w") as tf:
        json.dump(true_specs, tf, indent=2)

    _write_tier_readme(tier, tier_dir, rows)
    print(f"  [{tier}] HDF5 -> {h5_path.name}  "
          f"avg PSNR={np.mean(psnr_list):.2f} dB  "
          f"avg SSIM={np.mean(ssim_list):.4f}")


# -- README writers -----------------------------------------------------------

def _write_tier_readme(tier: str, tier_dir: Path, rows: list) -> None:
    spec = SPEC[tier]
    if tier == "public":
        access = "Full (GT + true spec + ideal B-scan)"
        source = ("Synthetic retinal/anterior segment OCT phantoms: "
                  "normal retina, pathological, anterior segment (12 samples)")
    elif tier == "dev":
        access = "Blind (measured B-scan + spec ranges only)"
        source = "Augmented synthetic OCT phantoms -- mixed types (20 samples)"
    else:
        access = "Server-only"
        source = ("Adversarial synthetic OCT phantoms -- multi-pathology, "
                  "low contrast, complex anatomy (20 samples)")

    param_desc = {
        "speckle_snr_db": "Speckle noise SNR",
        "axial_psf_fwhm_um": "Axial PSF FWHM (resolution)",
        "motion_artifact_px": "Lateral motion artifact",
        "signal_rolloff_db": "Depth-dependent signal roll-off",
    }

    lines = [
        f"# OCT {tier.capitalize()} Tier\n\n",
        f"**Source:** {source}\n\n",
        f"**Access:** {access}\n\n",
        "## Mismatch Parameters\n\n",
        "| Parameter | Description | Range |\n",
        "|-----------|-------------|-------|\n",
    ]
    for k, v in spec.items():
        lo, hi, u = v["min"], v["max"], v.get("unit", "")
        lines.append(
            f"| `{k}` | {param_desc.get(k, k)} | [{lo}, {hi}] {u} |\n")

    lines += [
        "\n## Samples\n\n",
        "| Sample | Scene | Speckle (dB) | PSF (um) | Motion (px) | "
        "Rolloff (dB/mm) | PSNR (dB) | SSIM |\n",
        "|--------|-------|-------------|----------|-------------|"
        "-----------------|-----------|------|\n",
    ]
    for key, scene, mis, psnr, ssim in rows:
        lines.append(
            f"| {key} | {scene}"
            f" | {mis['speckle_snr_db']:.1f}"
            f" | {mis['axial_psf_fwhm_um']:.1f}"
            f" | {mis['motion_artifact_px']:.1f}"
            f" | {mis['signal_rolloff_db']:.1f}"
            f" | {psnr:.2f}"
            f" | {ssim:.4f} |\n")

    lines += [
        "\n## HDF5 Datasets per Sample\n\n",
        "| Key | Shape | Dtype | Description |\n",
        "|-----|-------|-------|-------------|\n",
        "| `x_true` | (256, 256) | float32 | "
        "Ground-truth tissue reflectivity (B-scan) |\n",
        "| `bscan_ideal` | (256, 256) | float32 | "
        "Clean B-scan (PSF + rolloff only) |\n",
        "| `bscan_measured` | (256, 256) | float32 | "
        "Degraded B-scan (speckle + motion + noise) |\n",
    ]

    with open(tier_dir / "README.md", "w") as f:
        f.writelines(lines)


def _write_top_readme() -> None:
    txt = """# OCT (Optical Coherence Tomography) Benchmark Dataset

## Overview

Optical Coherence Tomography B-scan benchmark with axial PSF convolution,
multiplicative speckle noise, depth-dependent signal roll-off, and motion
artifacts. Uses synthetic retinal and anterior segment phantoms with
realistic layered tissue structure.

## Forward Model

```
bscan_ideal(z, x)    = PSF_axial(z) * reflectivity(z, x)
bscan_atten(z, x)    = bscan_ideal * exp(-rolloff * z)
bscan_speckle(z, x)  = bscan_atten * speckle_noise (Rayleigh)
bscan_measured(z, x)  = shift(bscan_speckle, motion) + shot_noise

where:
  reflectivity(z, x) -- 2D cross-sectional tissue reflectivity (ground truth)
  PSF_axial           -- Gaussian axial PSF (FWHM = coherence length)
  rolloff             -- depth-dependent signal attenuation (dB/mm)
  speckle_noise       -- multiplicative Rayleigh noise from coherent interference
  motion              -- lateral A-scan displacement (eye saccades)
  shot_noise          -- additive Gaussian noise
```

## Imaging Parameters

| Parameter | Value |
|-----------|-------|
| Centre wavelength | 850 nm |
| Pixel size | 3.0 um |
| Image size | 256 x 256 px |
| FOV | 768 um (axial) x 768 um (lateral) |
| Dynamic range | 50 dB |

## Mismatch Parameters (ThetaSpace)

| Knob | Symbol | Description | Public | Dev | Hidden |
|------|--------|-------------|--------|-----|--------|
| `speckle_snr_db` | SNR_s | Speckle noise level | 22-35 dB | 18-35 dB | 15-35 dB |
| `axial_psf_fwhm_um` | FWHM_z | Axial resolution | 3-8 um | 3-12 um | 3-15 um |
| `motion_artifact_px` | d_motion | Lateral motion | 0-3 px | 0-6 px | 0-10 px |
| `signal_rolloff_db` | alpha_z | Depth signal falloff | 2-6 dB/mm | 2-8 dB/mm | 2-10 dB/mm |

## Phantom Types

| Type | Description | Tier |
|------|-------------|------|
| Normal retina | 10 layered structures (ILM to choroid) with natural curvature | Public, Dev |
| Foveal retina | Normal retina with foveal depression (thinned inner layers) | Public |
| Drusen | Sub-RPE deposits elevating RPE layer | Public, Hidden |
| Intraretinal cysts | Fluid-filled dark voids in INL/ONL | Public, Hidden |
| Serous detachment | Sub-RPE fluid separating RPE from Bruch membrane | Public, Hidden |
| Epiretinal membrane | Bright membrane above ILM with traction folds | Public, Hidden |
| Anterior segment | Cornea, anterior chamber, iris, lens, angle structures | Public, Dev, Hidden |
| Multi-pathology | Drusen + cysts on same retina | Hidden |
| Low contrast | Very low contrast retina (adversarial) | Hidden |
| Cataract | Anterior segment with lens opacity | Hidden |

## Retinal Layer Structure

```
Vitreous (dark)
  ILM ----------- Inner Limiting Membrane (bright boundary)
  NFL ----------- Nerve Fiber Layer (BRIGHT)
  GCL ----------- Ganglion Cell Layer (dark)
  IPL ----------- Inner Plexiform Layer (medium)
  INL ----------- Inner Nuclear Layer (medium-dark)
  OPL ----------- Outer Plexiform Layer (medium)
  ONL ----------- Outer Nuclear Layer (DARK)
  ELM ----------- External Limiting Membrane (boundary)
  IS/OS --------- Inner/Outer Segment junction (VERY BRIGHT)
  OS  ----------- Outer Segments (medium-dark)
  RPE ----------- Retinal Pigment Epithelium (VERY BRIGHT)
  Bruch's ------- Bruch Membrane (bright boundary)
  Choroid ------- Choroidal tissue (medium, depth-attenuated)
Sclera (fading)
```

## Dataset Structure

```
oct/
+-- README.md
+-- generate_dataset.py
+-- public/    12 samples (4 normal + 4 pathological + 4 anterior)
|   +-- oct_challenge_public.h5
|   +-- spec.json / true_spec.json
|   +-- images/sample_XX_*/
+-- dev/       20 samples (augmented, medium mismatch)
|   +-- oct_challenge_dev.h5
|   +-- spec.json / true_spec.json
|   +-- images/sample_XX_*/
+-- hidden/    20 samples (adversarial, wide mismatch)
    +-- oct_challenge_hidden.h5
    +-- spec.json / true_spec.json
    +-- images/sample_XX_*/
```

## HDF5 Structure (per sample)

```
sample_XX/
+-- x_true (256, 256) float32         -- Ground-truth tissue reflectivity
+-- bscan_ideal (256, 256) float32    -- Clean B-scan (PSF + rolloff)
+-- bscan_measured (256, 256) float32 -- Degraded B-scan (all effects)
```

## CPU Reconstruction

Median filtering + bilateral denoising (edge-preserving speckle reduction):
  1. Median filter (size=3) to remove impulse-like speckle
  2. Bilateral-like smoothing via guided Gaussian filtering
  3. Edge detection via local intensity difference weighting

## Scoring

```
Score = 0.4 * PSNR_norm + 0.4 * SSIM + 0.2 * Consistency
```

## References

- Huang et al. (1991) "Optical coherence tomography," Science 254:1178-1181.
- Drexler & Fujimoto (2008) "Optical Coherence Tomography," Springer.
- Maggioni et al. (2012) "BM3D for OCT," IEEE Trans. Image Processing 21:1715-1728.
- Hu et al. (2020) "speckle2void," Biomedical Optics Express 11:817-830.
"""
    with open(BENCHMARK_DIR / "README.md", "w") as f:
        f.write(txt)


# -- Gallery image generation -------------------------------------------------

def generate_gallery_images(
    phantoms: list[tuple[str, np.ndarray]],
    gallery_dir: Path,
    n_scenes: int = 4,
    seed: int = 6666,
) -> None:
    """Generate gallery images for the platform benchmark page.

    Each scene: gt.png, measurement_I.png, measurement_II.png,
                recon_I.png, recon_II.png, recon_III.png
    All 256x256 grayscale PNGs.
    """
    rng = np.random.default_rng(seed)

    # Use diverse phantom types for the 4 gallery scenes:
    #   scene_00: normal retina (index 0)
    #   scene_01: pathological drusen (index 4)
    #   scene_02: anterior segment (index 8)
    #   scene_03: foveal retina (index 2)
    gallery_indices = [0, 4, 8, 2]

    for si, pidx in enumerate(gallery_indices[:n_scenes]):
        if pidx >= len(phantoms):
            pidx = si % len(phantoms)
        scene_name, x_true = phantoms[pidx]
        scene_dir = gallery_dir / f"scene_{si:02d}"
        scene_dir.mkdir(parents=True, exist_ok=True)

        _save_png(x_true, scene_dir / "gt.png")

        # Measurement I: mild conditions
        _, bm1 = forward_model(
            x_true,
            axial_psf_fwhm_um=4.0,
            signal_rolloff_db=3.0,
            speckle_snr_db=30.0,
            motion_artifact_px=1.0,
            rng=rng,
        )
        _save_png(bm1, scene_dir / "measurement_I.png")

        # Measurement II: heavy degradation
        _, bm2 = forward_model(
            x_true,
            axial_psf_fwhm_um=10.0,
            signal_rolloff_db=6.0,
            speckle_snr_db=18.0,
            motion_artifact_px=5.0,
            rng=rng,
        )
        _save_png(bm2, scene_dir / "measurement_II.png")

        # Recon I: despeckle from mild
        r1 = reconstruct_despeckle(bm1, median_size=3)
        _save_png(r1, scene_dir / "recon_I.png")

        # Recon II: despeckle from heavy
        r2 = reconstruct_despeckle(bm2, median_size=5)
        _save_png(r2, scene_dir / "recon_II.png")

        # Recon III: aggressive smoothing
        r3 = reconstruct_despeckle(bm2, median_size=7,
                                    bilateral_sigma_space=5.0,
                                    bilateral_sigma_intensity=0.15)
        _save_png(r3, scene_dir / "recon_III.png")

        print(f"  Gallery scene_{si:02d} ({scene_name}): 6 images saved")


# -- Main --------------------------------------------------------------------

def main() -> None:
    print("OCT (Optical Coherence Tomography) Benchmark Dataset Generator")
    print("=" * 65)
    print(f"Output: {BENCHMARK_DIR}\n")

    # -- Public tier (12 samples) ---------------------------------------------
    print("Generating public tier (12 samples)...")
    rng_pub = np.random.default_rng(1000)
    public_phantoms = generate_public_phantoms(rng_pub)
    generate_tier("public", public_phantoms, base_seed=1000)

    # -- Dev tier (20 samples) ------------------------------------------------
    print("\nGenerating dev tier (20 samples, augmented)...")
    rng_dev = np.random.default_rng(2000)
    dev_phantoms = generate_dev_phantoms(rng_dev)
    generate_tier("dev", dev_phantoms, base_seed=2000)

    # -- Hidden tier (20 samples) ---------------------------------------------
    print("\nGenerating hidden tier (20 samples, adversarial)...")
    rng_hid = np.random.default_rng(3000)
    hidden_phantoms = generate_hidden_phantoms(rng_hid)
    generate_tier("hidden", hidden_phantoms, base_seed=3000)

    # -- Gallery images -------------------------------------------------------
    gallery_dir = (Path(__file__).resolve().parent.parent.parent.parent /
                   "platform" / "pwm_platform" / "static" / "img" /
                   "benchmark_gallery" / "oct")
    print(f"\nGenerating gallery images at {gallery_dir}...")
    generate_gallery_images(public_phantoms, gallery_dir, n_scenes=4)

    # -- Top-level README -----------------------------------------------------
    _write_top_readme()

    print(f"\n{'=' * 65}")
    print(f"Done -- OCT benchmark ready at {BENCHMARK_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
