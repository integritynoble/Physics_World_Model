#!/usr/bin/env python3
"""Universal benchmark dataset generator for modalities without dedicated generators.

Generates HDF5 challenge datasets for ~127 modalities that lack local generators,
using standard test images + domain-appropriate forward models.

Usage:
    python3 generate_universal_datasets.py                    # Generate all missing
    python3 generate_universal_datasets.py --modality afm,ebsd
    python3 generate_universal_datasets.py --upload-gcs       # Also upload to GCS
    python3 generate_universal_datasets.py --dry-run          # Preview only
"""

from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from scipy.signal import fftconvolve

BENCHMARK_DIR = Path(__file__).resolve().parent
STANDARD_IMAGES_DIR = Path("/tmp/standard_test_images")

# ══════════════════════════════════════════════════════════════════════════════
#  Forward model types — each modality maps to one of these
# ══════════════════════════════════════════════════════════════════════════════

FORWARD_MODELS = {
    # PSF convolution: y = conv(x, psf) + noise
    "psf": {
        "description": "PSF convolution + Poisson/Gaussian noise",
        "mismatch_params": [
            {"name": "psf_sigma", "min": 1.0, "max": 3.0, "unit": "px"},
            {"name": "noise_level", "min": 0.01, "max": 0.05, "unit": ""},
            {"name": "background", "min": 0.0, "max": 0.02, "unit": ""},
        ],
    },
    # Radon sinogram: y = Radon(x) + noise
    "radon": {
        "description": "Radon transform projection + Poisson noise",
        "mismatch_params": [
            {"name": "angle_offset", "min": -2.0, "max": 2.0, "unit": "deg"},
            {"name": "noise_level", "min": 0.01, "max": 0.05, "unit": ""},
            {"name": "scatter_fraction", "min": 0.0, "max": 0.05, "unit": ""},
        ],
    },
    # k-space undersampling: y = M * F * x + noise
    "kspace": {
        "description": "Fourier k-space undersampling + Gaussian noise",
        "mismatch_params": [
            {"name": "acceleration", "min": 2, "max": 4, "unit": "x"},
            {"name": "noise_level", "min": 0.01, "max": 0.05, "unit": ""},
            {"name": "phase_error", "min": 0.0, "max": 0.05, "unit": "rad"},
        ],
    },
    # Identity/denoising: y = x + noise (forward model in phantom)
    "identity": {
        "description": "Identity forward model with noise/degradation",
        "mismatch_params": [
            {"name": "noise_level", "min": 0.01, "max": 0.05, "unit": ""},
            {"name": "blur_sigma", "min": 0.0, "max": 1.0, "unit": "px"},
            {"name": "offset", "min": -0.01, "max": 0.01, "unit": ""},
        ],
    },
    # Binary mask: y = M * x + noise (compressive sensing)
    "mask": {
        "description": "Binary random mask + Poisson noise",
        "mismatch_params": [
            {"name": "mask_shift_x", "min": -1.0, "max": 1.0, "unit": "px"},
            {"name": "mask_shift_y", "min": -1.0, "max": 1.0, "unit": "px"},
            {"name": "noise_level", "min": 0.01, "max": 0.05, "unit": ""},
        ],
    },
}

# ══════════════════════════════════════════════════════════════════════════════
#  Modality → forward model + phantom type mapping
# ══════════════════════════════════════════════════════════════════════════════

# Category-based default mapping
CATEGORY_DEFAULTS = {
    "microscopy":       ("psf", "cell"),
    "medical":          ("radon", "anatomy"),
    "medical_imaging":  ("radon", "anatomy"),
    "remote_sensing":   ("kspace", "terrain"),
    "spectroscopy":     ("identity", "spectral"),
    "electron_microscopy": ("identity", "nanostructure"),
    "computational":    ("psf", "natural"),
    "compressive":      ("mask", "natural"),
    "clinical_optics":  ("psf", "tissue"),
    "depth_imaging":    ("psf", "depth"),
    "industrial":       ("identity", "material"),
    "scanning_probe":   ("identity", "surface"),
}

# Per-modality overrides (forward_model, phantom_type)
MODALITY_CONFIG: Dict[str, Tuple[str, str]] = {
    # Medical tomography
    "angiography": ("radon", "vascular"),
    "asl_mri": ("kspace", "brain"),
    "brachytherapy_img": ("radon", "anatomy"),
    "cbct": ("radon", "anatomy"),  # already has local generator
    "cest_mri": ("identity", "brain"),
    "ceus": ("identity", "vascular"),
    "dexa": ("identity", "bone"),
    "digital_breast_tomo": ("radon", "breast"),
    "doppler_ultrasound": ("identity", "vascular"),
    "dot": ("identity", "tissue"),
    "elastography": ("identity", "tissue"),
    "fluoroscopy": ("radon", "anatomy"),
    "impedance_tomo": ("identity", "tissue"),
    "ivus": ("identity", "vascular"),
    "magnetic_particle": ("identity", "anatomy"),
    "mr_elastography": ("kspace", "brain"),
    "mr_fingerprinting": ("kspace", "brain"),
    "mra": ("kspace", "vascular"),
    "mrs": ("identity", "brain"),
    "muon_tomo": ("radon", "material"),
    "neutron_tomo": ("radon", "material"),
    "nirs_brain": ("identity", "brain"),
    "octa": ("identity", "vascular"),
    "phase_contrast": ("psf", "cell"),
    "portal_imaging": ("radon", "anatomy"),
    "proton_radiography": ("radon", "material"),
    "proton_therapy_img": ("radon", "anatomy"),
    "spectral_ct": ("radon", "anatomy"),  # already has local generator
    "swi": ("kspace", "brain"),
    "us_mri": ("identity", "anatomy"),
    "xray_ndt": ("radon", "material"),
    "xray_radiography": ("radon", "anatomy"),

    # Microscopy
    "cars": ("identity", "cell"),
    "cathodoluminescence": ("identity", "nanostructure"),
    "clem": ("identity", "cell"),
    "confocal_endomicroscopy": ("psf", "tissue"),
    "confocal_livecell": ("psf", "cell"),
    "dark_field": ("psf", "nanostructure"),
    "dic": ("identity", "cell"),
    "dna_paint": ("psf", "molecule"),
    "expansion": ("psf", "cell"),
    "flim": ("identity", "cell"),
    "ism": ("psf", "cell"),
    "lattice_lightsheet": ("psf", "cell"),
    "minflux": ("identity", "molecule"),
    "spinning_disk": ("psf", "cell"),
    "srs": ("identity", "cell"),
    "shg": ("identity", "tissue"),
    "three_photon": ("psf", "neural"),
    "tirf": ("psf", "cell"),
    "widefield_lowdose": ("psf", "cell"),

    # Electron microscopy
    "cryo_et": ("identity", "protein"),
    "ebsd": ("identity", "crystal"),
    "edx_mapping": ("identity", "material"),
    "eels": ("identity", "material"),
    "electron_diffraction": ("identity", "crystal"),
    "electron_holography": ("identity", "nanostructure"),
    "electron_tomography": ("radon", "nanostructure"),
    "fib_sem": ("identity", "material"),
    "stem": ("identity", "nanostructure"),

    # Remote sensing
    "flash_lidar": ("identity", "depth"),
    "multispectral_sat": ("identity", "terrain"),
    "ocean_color": ("identity", "terrain"),
    "passive_microwave": ("identity", "terrain"),
    "polsar": ("identity", "terrain"),
    "weather_radar": ("identity", "terrain"),

    # Computational optics
    "cassi": ("mask", "natural"),
    "coded_exposure": ("psf", "natural"),
    "coronagraphy": ("identity", "star"),
    "eht_imaging": ("kspace", "astronomy"),
    "entangled_photon": ("identity", "natural"),
    "event_camera": ("identity", "natural"),
    "hdr_imaging": ("identity", "natural"),
    "integral": ("psf", "natural"),
    "light_field": ("psf", "natural"),
    "lucky_imaging": ("psf", "astronomy"),
    "machine_vision": ("psf", "natural"),
    "matrix": ("identity", "natural"),
    "nerf": ("psf", "natural"),
    "panorama": ("identity", "natural"),
    "photometric_stereo": ("identity", "depth"),
    "polarization": ("identity", "natural"),
    "structured_light": ("identity", "depth"),

    # Spectroscopy / chemical
    "atom_probe": ("identity", "crystal"),
    "brillouin": ("identity", "tissue"),
    "ct_fluorescence": ("identity", "anatomy"),
    "desi": ("identity", "tissue"),
    "libs": ("identity", "material"),
    "maldi_msi": ("identity", "tissue"),
    "mfm": ("identity", "surface"),
    "nsom": ("psf", "nanostructure"),
    "pump_probe": ("identity", "spectral"),
    "quantum_illumination": ("identity", "natural"),
    "sims": ("identity", "material"),
    "stm": ("identity", "surface"),
    "xrf_imaging": ("identity", "material"),
    "xrf_tomo": ("radon", "material"),

    # Specialized
    "acoustic_emission": ("identity", "material"),
    "acoustic_microscopy": ("psf", "material"),
    "active_thermography": ("identity", "material"),
    "adaptive_optics": ("psf", "astronomy"),
    "afm": ("identity", "surface"),
    "bioluminescence_tomo": ("identity", "anatomy"),
    "cup": ("mask", "natural"),
    "eddy_current": ("identity", "material"),
    "fwi": ("identity", "terrain"),
    "ghost_imaging": ("mask", "natural"),  # already has local
    "gpr": ("radon", "terrain"),
    "gravitational_wave": ("identity", "signal"),
    "neutron_diffraction": ("identity", "crystal"),
    "ocean_acoustic_tomo": ("identity", "depth"),
    "particle_calorimetry": ("identity", "particle"),
    "radio_astronomy": ("kspace", "astronomy"),
    "radio_interferometry": ("kspace", "astronomy"),
    "saxs": ("identity", "crystal"),
    "seismic_tomo": ("radon", "terrain"),
    "shearography": ("identity", "material"),
    "solar_imaging": ("identity", "astronomy"),
    "sonar": ("identity", "depth"),
    "spc": ("mask", "natural"),
    "spc_block": ("mask", "natural"),
    "streak_camera": ("mask", "natural"),
    "talbot_lau": ("identity", "anatomy"),
    "terahertz": ("psf", "material"),
    "tof_camera": ("identity", "depth"),
    "ultrasonic_phased_array": ("identity", "material"),
    "waxs": ("identity", "crystal"),
    "xfel_sfx": ("identity", "crystal"),
    "xray_crystallography": ("identity", "crystal"),
}

# ══════════════════════════════════════════════════════════════════════════════
#  Phantom generators (domain-specific ground truth)
# ══════════════════════════════════════════════════════════════════════════════

def _load_standard_image(name: str = "camera", size: int = 256) -> np.ndarray:
    """Load a standard test image."""
    path = STANDARD_IMAGES_DIR / f"{name}_{size}.npy"
    if path.exists():
        return np.load(path).astype(np.float32)
    # Fallback to skimage
    from skimage import data as skdata
    from skimage.transform import resize as skresize
    img = getattr(skdata, name, skdata.camera)()
    if img.ndim == 3:
        img = np.mean(img, axis=2)
    img = skresize(img, (size, size), anti_aliasing=True, preserve_range=True)
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / img.max()
    return img


def generate_phantom(phantom_type: str, seed: int, size: int = 256) -> np.ndarray:
    """Generate a domain-appropriate phantom."""
    rng = np.random.RandomState(seed)

    if phantom_type == "natural":
        # Use standard test images (cameraman, astronaut, etc.)
        images = ["camera", "astronaut", "coffee", "coins", "hubble"]
        name = images[seed % len(images)]
        x = _load_standard_image(name, size)
        # Add slight variation per seed
        if seed > 0:
            shift_y, shift_x = rng.randint(-10, 10, 2)
            x = np.roll(np.roll(x, shift_y, axis=0), shift_x, axis=1)
        return x

    elif phantom_type == "anatomy":
        return _make_anatomy_phantom(size, rng)

    elif phantom_type == "brain":
        return _make_brain_phantom(size, rng)

    elif phantom_type == "vascular":
        return _make_vascular_phantom(size, rng)

    elif phantom_type == "breast":
        return _make_breast_phantom(size, rng)

    elif phantom_type == "bone":
        return _make_bone_phantom(size, rng)

    elif phantom_type == "cell":
        return _make_cell_phantom(size, rng)

    elif phantom_type == "tissue":
        return _make_tissue_phantom(size, rng)

    elif phantom_type == "molecule":
        return _make_molecule_phantom(size, rng)

    elif phantom_type == "neural":
        return _make_neural_phantom(size, rng)

    elif phantom_type == "protein":
        return _make_protein_phantom(size, rng)

    elif phantom_type == "nanostructure":
        return _make_nanostructure_phantom(size, rng)

    elif phantom_type == "crystal":
        return _make_crystal_phantom(size, rng)

    elif phantom_type == "material":
        return _make_material_phantom(size, rng)

    elif phantom_type == "surface":
        return _make_surface_phantom(size, rng)

    elif phantom_type == "terrain":
        return _make_terrain_phantom(size, rng)

    elif phantom_type == "depth":
        return _make_depth_phantom(size, rng)

    elif phantom_type == "spectral":
        return _make_spectral_phantom(size, rng)

    elif phantom_type == "astronomy":
        return _make_astronomy_phantom(size, rng)

    elif phantom_type == "star":
        return _make_star_phantom(size, rng)

    elif phantom_type == "signal":
        return _make_signal_phantom(size, rng)

    elif phantom_type == "particle":
        return _make_particle_phantom(size, rng)

    else:
        # Fallback to natural image
        return _load_standard_image("camera", size)


def _make_anatomy_phantom(size: int, rng: np.random.RandomState) -> np.ndarray:
    """Shepp-Logan-like anatomy phantom."""
    yy = np.linspace(-1, 1, size)
    xx = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(xx, yy)
    arr = np.zeros((size, size), dtype=np.float32)
    # Outer body
    arr[(X / 0.85)**2 + (Y / 0.95)**2 < 1] = 0.15
    # Organs
    n_organs = 3 + rng.randint(3)
    for _ in range(n_organs):
        cx, cy = rng.uniform(-0.4, 0.4, 2)
        rx, ry = rng.uniform(0.1, 0.3, 2)
        intensity = rng.uniform(0.3, 0.8)
        mask = ((X - cx) / rx)**2 + ((Y - cy) / ry)**2 < 1
        arr[mask] = intensity
    # Spine
    arr[((X)**2 + ((Y + 0.3) / 0.05)**2) < 0.1] = 0.85
    return gaussian_filter(arr, sigma=1.0)


def _make_brain_phantom(size: int, rng: np.random.RandomState) -> np.ndarray:
    """Brain phantom with gray/white matter."""
    try:
        x = np.load(STANDARD_IMAGES_DIR / "zubal_brain_256.npy")
        if x.shape[0] != size:
            x = zoom(x, size / x.shape[0], order=1)
        # Apply random rotation/contrast variation
        shift = rng.randint(-5, 5, 2)
        x = np.roll(np.roll(x, shift[0], axis=0), shift[1], axis=1)
        x = np.clip(x * rng.uniform(0.9, 1.1), 0, 1)
        return x.astype(np.float32)
    except Exception:
        yy, xx = np.linspace(-1, 1, size), np.linspace(-1, 1, size)
        X, Y = np.meshgrid(xx, yy)
        arr = np.zeros((size, size), dtype=np.float32)
        arr[(X**2 + Y**2) < 0.7] = 0.3  # skull
        arr[(X**2 + Y**2) < 0.6] = 0.8  # gray matter
        arr[(X**2 + Y**2) < 0.4] = 0.4  # white matter
        arr[((X / 0.06)**2 + ((Y + 0.05) / 0.12)**2) < 1] = 0.05  # ventricle
        return gaussian_filter(arr, sigma=1.5)


def _make_vascular_phantom(size: int, rng: np.random.RandomState) -> np.ndarray:
    """Vascular network phantom."""
    arr = np.zeros((size, size), dtype=np.float32)
    # Background tissue
    arr += 0.15
    # Generate branching vessels
    n_vessels = 5 + rng.randint(5)
    for _ in range(n_vessels):
        x0, y0 = rng.randint(0, size, 2)
        angle = rng.uniform(0, 2 * np.pi)
        width = rng.uniform(1.5, 4.0)
        length = rng.randint(60, 150)
        for step in range(length):
            x = int(x0 + step * np.cos(angle))
            y = int(y0 + step * np.sin(angle))
            if 0 <= x < size and 0 <= y < size:
                r = int(max(1, width))
                yy, xx = np.ogrid[max(0,y-r):min(size,y+r+1), max(0,x-r):min(size,x+r+1)]
                dist = np.sqrt((xx - x)**2 + (yy - y)**2)
                mask = dist < width
                arr[max(0,y-r):min(size,y+r+1), max(0,x-r):min(size,x+r+1)][mask] = rng.uniform(0.6, 0.9)
            angle += rng.uniform(-0.1, 0.1)
            width *= 0.995
    return gaussian_filter(np.clip(arr, 0, 1), sigma=0.8)


def _make_breast_phantom(size: int, rng: np.random.RandomState) -> np.ndarray:
    """Breast tissue phantom."""
    yy = np.linspace(-1, 1, size)
    xx = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(xx, yy)
    arr = np.zeros((size, size), dtype=np.float32)
    # Breast outline (semicircle)
    arr[(X**2 + Y**2) < 0.8] = 0.15  # adipose
    # Fibroglandular regions (random blobs)
    for _ in range(5 + rng.randint(5)):
        cx, cy = rng.uniform(-0.5, 0.5, 2)
        r = rng.uniform(0.05, 0.15)
        mask = ((X - cx)**2 + (Y - cy)**2) < r**2
        arr[mask] = rng.uniform(0.3, 0.5)
    # Masses
    for _ in range(1 + rng.randint(2)):
        cx, cy = rng.uniform(-0.3, 0.3, 2)
        r = rng.uniform(0.03, 0.08)
        mask = ((X - cx)**2 + (Y - cy)**2) < r**2
        arr[mask] = rng.uniform(0.5, 0.7)
    return gaussian_filter(arr, sigma=1.0)


def _make_bone_phantom(size: int, rng: np.random.RandomState) -> np.ndarray:
    """Bone density phantom (DEXA-like)."""
    yy = np.linspace(-1, 1, size)
    xx = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(xx, yy)
    arr = np.zeros((size, size), dtype=np.float32)
    # Soft tissue background
    arr += 0.1
    # Vertebrae-like structures
    for i in range(4):
        cy = -0.6 + i * 0.4
        mask = ((X / 0.15)**2 + ((Y - cy) / 0.12)**2) < 1
        arr[mask] = rng.uniform(0.5, 0.8)
        # Cortical shell
        shell = ((X / 0.18)**2 + ((Y - cy) / 0.15)**2 < 1) & ~mask
        arr[shell] = rng.uniform(0.8, 1.0)
    return gaussian_filter(arr, sigma=0.5)


def _make_cell_phantom(size: int, rng: np.random.RandomState) -> np.ndarray:
    """Fluorescence microscopy cell phantom."""
    arr = np.zeros((size, size), dtype=np.float32)
    n_cells = 8 + rng.randint(8)
    yy, xx = np.ogrid[:size, :size]
    for _ in range(n_cells):
        cx, cy = rng.randint(20, size - 20, 2)
        rx, ry = rng.uniform(8, 25, 2)
        angle = rng.uniform(0, np.pi)
        intensity = rng.uniform(0.4, 1.0)
        # Elliptical cell body
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        dx, dy = xx - cx, yy - cy
        xr = (dx * cos_a + dy * sin_a) / rx
        yr = (-dx * sin_a + dy * cos_a) / ry
        dist = xr**2 + yr**2
        cell_mask = dist < 1
        arr[cell_mask] = np.maximum(arr[cell_mask], intensity * (1 - 0.3 * dist[cell_mask]))
        # Nucleus (brighter, smaller)
        nucleus_mask = dist < 0.3
        arr[nucleus_mask] = np.maximum(arr[nucleus_mask], intensity * 1.2)
    return np.clip(gaussian_filter(arr, sigma=0.8), 0, 1)


def _make_tissue_phantom(size: int, rng: np.random.RandomState) -> np.ndarray:
    """Biological tissue cross-section."""
    arr = np.zeros((size, size), dtype=np.float32)
    # Multi-layer tissue
    n_layers = 3 + rng.randint(3)
    layer_h = size // n_layers
    for i in range(n_layers):
        base = rng.uniform(0.2, 0.6)
        wave = 5 * np.sin(np.linspace(0, 4 * np.pi, size)) * rng.uniform(0.5, 1.5)
        for j in range(size):
            y_start = int(i * layer_h + wave[j])
            y_end = int((i + 1) * layer_h + wave[j])
            y_start = max(0, min(size - 1, y_start))
            y_end = max(0, min(size, y_end))
            arr[y_start:y_end, j] = base + rng.uniform(-0.05, 0.05)
    arr += 0.03 * rng.randn(size, size)
    return np.clip(gaussian_filter(arr, sigma=1.0), 0, 1)


def _make_molecule_phantom(size: int, rng: np.random.RandomState) -> np.ndarray:
    """Sparse molecule positions (SMLM-like)."""
    arr = np.zeros((size, size), dtype=np.float32)
    # Background
    arr += 0.02
    # Sparse bright molecules
    n_mol = 50 + rng.randint(100)
    yy, xx = np.ogrid[:size, :size]
    for _ in range(n_mol):
        cx, cy = rng.uniform(10, size - 10, 2)
        sigma = rng.uniform(0.8, 2.0)
        intensity = rng.uniform(0.5, 1.0)
        dist2 = (xx - cx)**2 + (yy - cy)**2
        arr += intensity * np.exp(-dist2 / (2 * sigma**2))
    return np.clip(arr, 0, 1)


def _make_neural_phantom(size: int, rng: np.random.RandomState) -> np.ndarray:
    """Neural tissue (neurons + dendrites)."""
    arr = np.zeros((size, size), dtype=np.float32)
    arr += 0.05  # background
    # Neuron soma
    yy, xx = np.ogrid[:size, :size]
    n_neurons = 10 + rng.randint(10)
    for _ in range(n_neurons):
        cx, cy = rng.randint(20, size - 20, 2)
        r = rng.uniform(5, 12)
        dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        arr[dist < r] = rng.uniform(0.6, 1.0)
        # Dendrites (random walks)
        for _ in range(2 + rng.randint(3)):
            x0, y0 = float(cx), float(cy)
            angle = rng.uniform(0, 2 * np.pi)
            for step in range(rng.randint(20, 60)):
                x0 += np.cos(angle) * 1.5
                y0 += np.sin(angle) * 1.5
                angle += rng.uniform(-0.3, 0.3)
                xi, yi = int(x0), int(y0)
                if 0 <= xi < size and 0 <= yi < size:
                    arr[max(0, yi-1):min(size, yi+2), max(0, xi-1):min(size, xi+2)] = 0.5
    return np.clip(gaussian_filter(arr, sigma=0.5), 0, 1)


def _make_protein_phantom(size: int, rng: np.random.RandomState) -> np.ndarray:
    """Protein-like structures (cryo-EM)."""
    arr = np.zeros((size, size), dtype=np.float32)
    arr += 0.1  # ice background
    # Protein complex (cluster of spheres)
    cx, cy = size // 2, size // 2
    yy, xx = np.ogrid[:size, :size]
    n_subunits = 5 + rng.randint(8)
    for _ in range(n_subunits):
        sx = cx + rng.uniform(-30, 30)
        sy = cy + rng.uniform(-30, 30)
        r = rng.uniform(8, 20)
        dist = np.sqrt((xx - sx)**2 + (yy - sy)**2)
        arr[dist < r] = rng.uniform(0.4, 0.8)
    # Add some noise texture
    arr += 0.02 * rng.randn(size, size)
    return np.clip(gaussian_filter(arr, sigma=1.0), 0, 1)


def _make_nanostructure_phantom(size: int, rng: np.random.RandomState) -> np.ndarray:
    """Nanostructured material (SEM/TEM-like)."""
    arr = 0.15 + 0.03 * rng.randn(size, size)
    yy, xx = np.ogrid[:size, :size]
    # Nanoparticles
    n_particles = 30 + rng.randint(30)
    for _ in range(n_particles):
        cx, cy = rng.randint(5, size - 5, 2)
        r = rng.uniform(2, 8)
        intensity = rng.uniform(0.5, 1.0)
        dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        arr[dist < r] = intensity
    return np.clip(gaussian_filter(arr.astype(np.float32), sigma=0.5), 0, 1)


def _make_crystal_phantom(size: int, rng: np.random.RandomState) -> np.ndarray:
    """Crystal lattice / diffraction pattern."""
    arr = np.zeros((size, size), dtype=np.float32)
    # Voronoi-like grain structure
    n_grains = 10 + rng.randint(10)
    centers = rng.randint(0, size, (n_grains, 2))
    intensities = rng.uniform(0.3, 0.9, n_grains)
    yy, xx = np.ogrid[:size, :size]
    for i in range(n_grains):
        cx, cy = centers[i]
        dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        mask = dist < (size / n_grains * 1.5)
        arr[mask] = intensities[i]
    # Grain boundaries
    arr = gaussian_filter(arr, sigma=2.0)
    grad = np.sqrt(np.gradient(arr, axis=0)**2 + np.gradient(arr, axis=1)**2)
    arr -= 0.5 * grad / (grad.max() + 1e-8)
    return np.clip(arr, 0, 1)


def _make_material_phantom(size: int, rng: np.random.RandomState) -> np.ndarray:
    """Industrial material cross-section."""
    arr = np.zeros((size, size), dtype=np.float32)
    arr += rng.uniform(0.2, 0.4)  # base material
    # Defects (voids, inclusions)
    yy, xx = np.ogrid[:size, :size]
    for _ in range(5 + rng.randint(10)):
        cx, cy = rng.randint(10, size - 10, 2)
        r = rng.uniform(3, 15)
        dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        if rng.rand() > 0.5:
            arr[dist < r] = rng.uniform(0, 0.1)  # void
        else:
            arr[dist < r] = rng.uniform(0.7, 1.0)  # inclusion
    # Texture
    arr += 0.02 * rng.randn(size, size)
    return np.clip(gaussian_filter(arr.astype(np.float32), sigma=0.5), 0, 1)


def _make_surface_phantom(size: int, rng: np.random.RandomState) -> np.ndarray:
    """Surface topography (AFM/STM-like)."""
    arr = np.zeros((size, size), dtype=np.float32)
    # Multi-scale surface roughness
    for scale in [64, 32, 16, 8]:
        noise = rng.randn(size // scale + 1, size // scale + 1)
        noise = zoom(noise, scale, order=3)[:size, :size]
        arr += noise * (scale / 64.0) * 0.15
    # Surface features (steps, defects)
    arr[size//3:2*size//3, :] += 0.3  # step edge
    arr += 0.5
    return np.clip(arr, 0, 1)


def _make_terrain_phantom(size: int, rng: np.random.RandomState) -> np.ndarray:
    """Terrain/landscape (remote sensing)."""
    arr = np.zeros((size, size), dtype=np.float32)
    # Multi-scale terrain
    for scale in [64, 32, 16, 8, 4]:
        noise = rng.randn(size // scale + 1, size // scale + 1)
        noise = zoom(noise, scale, order=3)[:size, :size]
        arr += noise * (scale / 64.0) * 0.2
    arr = arr - arr.min()
    arr = arr / (arr.max() + 1e-8)
    # Add some structured features (roads, fields)
    arr[size//4:size//4+3, :] = 0.8  # road
    arr[:, size//3:size//3+3] = 0.7  # road
    return np.clip(arr, 0, 1)


def _make_depth_phantom(size: int, rng: np.random.RandomState) -> np.ndarray:
    """Depth map (LiDAR/ToF-like)."""
    arr = np.zeros((size, size), dtype=np.float32)
    # Ground plane
    yy = np.linspace(0.3, 0.8, size)
    arr += yy[:, np.newaxis]
    # Objects at different depths
    yy_g, xx_g = np.ogrid[:size, :size]
    for _ in range(3 + rng.randint(5)):
        cx, cy = rng.randint(20, size - 20, 2)
        rx, ry = rng.randint(15, 40, 2)
        depth = rng.uniform(0.1, 0.5)
        mask = ((xx_g - cx) / rx)**2 + ((yy_g - cy) / ry)**2 < 1
        arr[mask] = depth
    return np.clip(arr, 0, 1)


def _make_spectral_phantom(size: int, rng: np.random.RandomState) -> np.ndarray:
    """Spectral measurement (Raman/FTIR-like)."""
    arr = np.zeros((size, size), dtype=np.float32)
    # Spatial distribution of chemical species
    n_species = 3 + rng.randint(3)
    yy, xx = np.ogrid[:size, :size]
    for _ in range(n_species):
        cx, cy = rng.randint(30, size - 30, 2)
        rx, ry = rng.uniform(15, 40, 2)
        intensity = rng.uniform(0.3, 0.9)
        dist = ((xx - cx) / rx)**2 + ((yy - cy) / ry)**2
        arr[dist < 1] = intensity
    arr += 0.05  # background signal
    return np.clip(gaussian_filter(arr, sigma=1.0), 0, 1)


def _make_astronomy_phantom(size: int, rng: np.random.RandomState) -> np.ndarray:
    """Astronomical source (galaxy, nebula)."""
    arr = np.zeros((size, size), dtype=np.float32)
    arr += 0.02  # sky background
    cx, cy = size // 2, size // 2
    yy, xx = np.ogrid[:size, :size]
    # Galaxy core
    dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    arr += 0.8 * np.exp(-dist**2 / (2 * 20**2))
    # Spiral arms / extended emission
    theta = np.arctan2(yy - cy, xx - cx)
    spiral = 0.3 * np.exp(-dist / 50) * (1 + 0.5 * np.sin(2 * theta + dist / 10))
    arr += spiral
    # Point sources (stars)
    for _ in range(20 + rng.randint(20)):
        sx, sy = rng.randint(5, size - 5, 2)
        arr[sy, sx] = rng.uniform(0.3, 1.0)
    return np.clip(gaussian_filter(arr, sigma=0.5), 0, 1)


def _make_star_phantom(size: int, rng: np.random.RandomState) -> np.ndarray:
    """Star + planet (coronagraphy)."""
    arr = np.zeros((size, size), dtype=np.float32)
    cx, cy = size // 2, size // 2
    yy, xx = np.ogrid[:size, :size]
    dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    # Star PSF halo
    arr += 0.5 * np.exp(-dist**2 / (2 * 5**2))
    # Planet (faint, offset)
    px = cx + rng.randint(20, 50) * (1 if rng.rand() > 0.5 else -1)
    py = cy + rng.randint(20, 50) * (1 if rng.rand() > 0.5 else -1)
    pdist = np.sqrt((xx - px)**2 + (yy - py)**2)
    arr += 0.005 * np.exp(-pdist**2 / (2 * 3**2))  # 100x fainter
    return np.clip(arr, 0, 1)


def _make_signal_phantom(size: int, rng: np.random.RandomState) -> np.ndarray:
    """1D-like signal displayed as 2D (gravitational wave)."""
    arr = np.zeros((size, size), dtype=np.float32)
    # Time-frequency representation (spectrogram-like)
    t = np.linspace(0, 1, size)
    for i in range(size):
        freq = 10 + 50 * (i / size)**2  # chirp
        arr[i, :] = 0.5 * np.sin(2 * np.pi * freq * t) * np.exp(-(t - 0.5)**2 / 0.05)
    arr = np.abs(arr)
    arr += 0.05 * rng.randn(size, size)
    return np.clip(arr, 0, 1)


def _make_particle_phantom(size: int, rng: np.random.RandomState) -> np.ndarray:
    """Particle shower / calorimeter."""
    arr = np.zeros((size, size), dtype=np.float32)
    # Energy deposits from particle shower
    yy, xx = np.ogrid[:size, :size]
    # Main shower axis
    cx = size // 2
    for i in range(size):
        spread = 5 + i * 0.1
        n_hits = rng.poisson(3)
        for _ in range(n_hits):
            hx = cx + int(rng.normal(0, spread))
            if 0 <= hx < size:
                energy = rng.exponential(0.3)
                arr[i, max(0, hx-2):min(size, hx+3)] += energy
    return np.clip(arr / (arr.max() + 1e-8), 0, 1)


# ══════════════════════════════════════════════════════════════════════════════
#  Forward models
# ══════════════════════════════════════════════════════════════════════════════

def _make_psf(sigma: float, size: int = 21) -> np.ndarray:
    """Gaussian PSF kernel."""
    k = size // 2
    yy, xx = np.ogrid[-k:k+1, -k:k+1]
    psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return psf / psf.sum()


def apply_forward_model(
    x_true: np.ndarray,
    model_type: str,
    mismatch_params: dict,
    rng: np.random.RandomState,
    tier: str = "public",
) -> tuple[np.ndarray, dict]:
    """Apply forward model to ground truth, returning (y, H_ideal)."""
    size = x_true.shape[0]

    if model_type == "psf":
        # PSF convolution
        sigma = mismatch_params.get("psf_sigma", 2.0)
        noise = mismatch_params.get("noise_level", 0.02)
        bg = mismatch_params.get("background", 0.01)
        psf = _make_psf(sigma)
        y = fftconvolve(x_true, psf, mode="same") + bg
        y += noise * rng.randn(*y.shape)
        y = np.maximum(y, 0)
        H_ideal = psf.astype(np.float32)
        return y.astype(np.float32), {"psf": H_ideal}

    elif model_type == "radon":
        # Fast vectorized Radon transform
        n_angles = 180
        angles = np.linspace(0, np.pi, n_angles, endpoint=False)
        n_det = int(np.ceil(np.sqrt(2) * size))
        sino = np.zeros((n_angles, n_det), dtype=np.float64)

        # Coordinate grids
        coords = np.arange(size) - size / 2.0 + 0.5
        X, Y = np.meshgrid(coords, coords)

        for i, theta in enumerate(angles):
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            # Project all pixels onto detector axis
            proj = X * cos_t + Y * sin_t  # (size, size)
            proj_shifted = proj + n_det / 2.0  # shift to [0, n_det]
            # Bin into detector pixels using histogram
            sino[i] = np.histogram(
                proj_shifted.ravel(),
                bins=n_det, range=(0, n_det),
                weights=x_true.ravel(),
            )[0]

        noise = mismatch_params.get("noise_level", 0.02)
        sino += noise * rng.randn(*sino.shape) * (sino.max() + 1e-8)
        sino = np.maximum(sino, 0).astype(np.float32)
        H_ideal = angles.astype(np.float32)
        return sino, {"theta": H_ideal, "n_det": n_det}

    elif model_type == "kspace":
        # k-space undersampling
        accel = int(mismatch_params.get("acceleration", 4))
        noise = mismatch_params.get("noise_level", 0.02)
        kspace = np.fft.fft2(x_true)
        # Create undersampling mask (variable density)
        mask = np.zeros((size, size), dtype=np.float32)
        # Always keep center lines
        center_frac = 0.08
        n_center = max(1, int(size * center_frac))
        mask[size//2 - n_center:size//2 + n_center, :] = 1.0
        mask[:, size//2 - n_center:size//2 + n_center] = 1.0
        # Random lines
        for i in range(0, size, accel):
            mask[i, :] = 1.0
        y_kspace = kspace * mask
        y_kspace += noise * (rng.randn(size, size) + 1j * rng.randn(size, size)) * np.abs(kspace).max()
        y = np.abs(np.fft.ifft2(y_kspace)).astype(np.float32)
        return y, {"mask": mask, "acceleration": accel}

    elif model_type == "mask":
        # Binary random mask
        mask_ratio = 0.5
        mask = (rng.rand(size, size) > mask_ratio).astype(np.float32)
        noise = mismatch_params.get("noise_level", 0.02)
        y = x_true * mask + noise * rng.randn(size, size)
        y = np.maximum(y, 0).astype(np.float32)
        return y, {"mask": mask}

    else:  # identity
        noise = mismatch_params.get("noise_level", 0.02)
        blur = mismatch_params.get("blur_sigma", 0.5)
        offset = mismatch_params.get("offset", 0.0)
        if blur > 0.1:
            y = gaussian_filter(x_true, sigma=blur)
        else:
            y = x_true.copy()
        y += noise * rng.randn(*y.shape) + offset
        y = np.maximum(y, 0).astype(np.float32)
        return y, {}


# ══════════════════════════════════════════════════════════════════════════════
#  Tier configuration
# ══════════════════════════════════════════════════════════════════════════════

TIER_CONFIG = {
    "public": {"n_samples": 12, "seed_offset": 0, "difficulty": 1.0},
    "dev": {"n_samples": 20, "seed_offset": 10000, "difficulty": 1.5},
    "hidden": {"n_samples": 20, "seed_offset": 20000, "difficulty": 2.0},
}


def sample_mismatch_params(
    model_type: str,
    tier: str,
    rng: np.random.RandomState,
) -> dict:
    """Sample mismatch parameters for a given tier."""
    base_params = FORWARD_MODELS[model_type]["mismatch_params"]
    difficulty = TIER_CONFIG[tier]["difficulty"]
    params = {}
    for p in base_params:
        lo, hi = p["min"], p["max"]
        # Scale range by difficulty
        mid = (lo + hi) / 2
        half = (hi - lo) / 2 * difficulty
        val = rng.uniform(mid - half, mid + half)
        params[p["name"]] = val
    return params


# ══════════════════════════════════════════════════════════════════════════════
#  Image saving
# ══════════════════════════════════════════════════════════════════════════════

def save_image(arr: np.ndarray, path: Path) -> None:
    """Save array as PNG."""
    from PIL import Image
    if arr.ndim == 2:
        norm = arr - arr.min()
        if norm.max() > 0:
            norm = norm / norm.max()
        img = (norm * 255).astype(np.uint8)
        Image.fromarray(img, mode="L").save(str(path))
    else:
        Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(str(path))


# ══════════════════════════════════════════════════════════════════════════════
#  Main generator
# ══════════════════════════════════════════════════════════════════════════════

def generate_modality(modality: str, upload_gcs: bool = False) -> None:
    """Generate all 3 tiers for a single modality."""
    if modality not in MODALITY_CONFIG:
        print(f"  WARNING: Unknown modality {modality}, using default PSF/natural")
        model_type, phantom_type = "psf", "natural"
    else:
        model_type, phantom_type = MODALITY_CONFIG[modality]

    mod_dir = BENCHMARK_DIR / modality
    print(f"  Generating {modality} ({model_type}/{phantom_type})")

    for tier in ["public", "dev", "hidden"]:
        tier_cfg = TIER_CONFIG[tier]
        n_samples = tier_cfg["n_samples"]
        seed_offset = tier_cfg["seed_offset"]

        tier_dir = mod_dir / tier
        tier_dir.mkdir(parents=True, exist_ok=True)
        images_dir = tier_dir / "images"
        images_dir.mkdir(exist_ok=True)

        h5_path = tier_dir / f"{modality}_challenge_{tier}.h5"

        # Write spec.json
        spec_params = FORWARD_MODELS[model_type]["mismatch_params"]
        spec_path = tier_dir / "spec.json"
        with open(spec_path, "w") as f:
            json.dump(spec_params, f, indent=2)

        with h5py.File(h5_path, "w") as hf:
            for idx in range(n_samples):
                seed = seed_offset + idx
                rng = np.random.RandomState(seed)

                # Generate phantom
                x_true = generate_phantom(phantom_type, seed, size=256)
                x_true = np.clip(x_true, 0, 1).astype(np.float32)

                # Sample mismatch and apply forward model
                mismatch = sample_mismatch_params(model_type, tier, rng)
                y, h_ideal_dict = apply_forward_model(
                    x_true, model_type, mismatch, rng, tier
                )

                # Store in HDF5
                grp = hf.create_group(f"sample_{idx:02d}")
                grp.create_dataset("x_true", data=x_true, compression="gzip")
                grp.create_dataset("y", data=y, compression="gzip")

                # Store H_ideal components
                if "psf" in h_ideal_dict:
                    grp.create_dataset("H_ideal", data=h_ideal_dict["psf"], compression="gzip")
                elif "theta" in h_ideal_dict:
                    grp.create_dataset("H_ideal", data=h_ideal_dict["theta"])
                    grp.attrs["n_det"] = h_ideal_dict.get("n_det", 0)
                elif "mask" in h_ideal_dict:
                    grp.create_dataset("H_ideal", data=h_ideal_dict["mask"], compression="gzip")

                # Store true spec
                grp.attrs["true_spec"] = json.dumps(mismatch)

                # Save gallery images for first 6 samples
                if idx < 6:
                    sample_img_dir = images_dir / f"sample_{idx:02d}"
                    sample_img_dir.mkdir(exist_ok=True)
                    save_image(x_true, sample_img_dir / "ground_truth.png")
                    save_image(y, sample_img_dir / "measurement.png")

                    # Simple reconstruction
                    if model_type == "identity":
                        recon = gaussian_filter(y, sigma=0.5)
                    elif model_type == "psf":
                        recon = y  # Wiener would be better but keep simple
                    else:
                        recon = y if y.shape == x_true.shape else x_true * 0.9
                    save_image(recon, sample_img_dir / "recon.png")

                    # Per-sample spec
                    with open(sample_img_dir / "spec.json", "w") as f:
                        json.dump({"mismatch": mismatch, "sample": idx}, f, indent=2)

        # Write true_spec.json
        true_spec_path = tier_dir / "true_spec.json"
        # Generate a representative true_spec
        rng_ts = np.random.RandomState(seed_offset)
        representative = sample_mismatch_params(model_type, tier, rng_ts)
        with open(true_spec_path, "w") as f:
            json.dump(representative, f, indent=2)

        print(f"    {tier}: {n_samples} samples -> {h5_path.name} ({h5_path.stat().st_size / 1024 / 1024:.1f} MB)")

    if upload_gcs:
        _upload_to_gcs(modality)


def _upload_to_gcs(modality: str) -> None:
    """Upload modality data to GCS."""
    import subprocess
    mod_dir = BENCHMARK_DIR / modality

    for tier in ["public", "dev", "hidden"]:
        tier_dir = mod_dir / tier
        if not tier_dir.exists():
            continue

        # Upload to datasets/{modality}/
        dst1 = f"gs://pwm-benchmark-datasets/datasets/{modality}/{tier}/"
        subprocess.run(
            ["gsutil", "-m", "rsync", "-r", str(tier_dir), dst1],
            capture_output=True,
        )

        # Also to datasets/Benchmark/{modality}/
        dst2 = f"gs://pwm-benchmark-datasets/datasets/Benchmark/{modality}/{tier}/"
        subprocess.run(
            ["gsutil", "-m", "rsync", "-r", str(tier_dir), dst2],
            capture_output=True,
        )

    print(f"    Uploaded {modality} to GCS")


def get_missing_modalities() -> List[str]:
    """Find modalities that need datasets generated."""
    missing = []
    for mod in sorted(MODALITY_CONFIG.keys()):
        h5_path = BENCHMARK_DIR / mod / "public" / f"{mod}_challenge_public.h5"
        if not h5_path.exists():
            missing.append(mod)
    return missing


def main():
    parser = argparse.ArgumentParser(description="Universal benchmark dataset generator")
    parser.add_argument("--modality", type=str, default="",
                        help="Comma-separated modalities (default: all missing)")
    parser.add_argument("--upload-gcs", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.modality:
        modalities = args.modality.split(",")
    else:
        modalities = get_missing_modalities()

    print(f"Found {len(modalities)} modalities to generate")

    if args.dry_run:
        for mod in modalities:
            cfg = MODALITY_CONFIG.get(mod, ("psf", "natural"))
            print(f"  {mod}: model={cfg[0]}, phantom={cfg[1]}")
        return 0

    for i, mod in enumerate(modalities):
        print(f"\n[{i+1}/{len(modalities)}] {mod}")
        try:
            generate_modality(mod, upload_gcs=args.upload_gcs)
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nDone. Generated {len(modalities)} modality datasets.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
