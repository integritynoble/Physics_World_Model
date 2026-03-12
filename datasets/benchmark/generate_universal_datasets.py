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
    # ── New physics-correct forward models (added for mechanism verification) ──
    # Fourier magnitude: y = |F(x)|² (diffraction, crystallography)
    "fourier_mag": {
        "description": "Fourier magnitude (diffraction pattern) + Poisson noise",
        "mismatch_params": [
            {"name": "noise_level", "min": 0.01, "max": 0.05, "unit": ""},
            {"name": "beam_energy_offset", "min": -0.02, "max": 0.02, "unit": "keV"},
            {"name": "detector_offset", "min": -1.0, "max": 1.0, "unit": "px"},
        ],
    },
    # Interference: y = |R + O*exp(iφ)|² (holography, DIC, shearography)
    "interference": {
        "description": "Interference pattern from phase object + reference wave",
        "mismatch_params": [
            {"name": "noise_level", "min": 0.01, "max": 0.05, "unit": ""},
            {"name": "fringe_spacing", "min": 8.0, "max": 16.0, "unit": "px"},
            {"name": "visibility", "min": 0.7, "max": 1.0, "unit": ""},
        ],
    },
    # Diffusion Green's function: y = G·x (DOT, BLT, fNIRS, EIT)
    "diffusion": {
        "description": "Diffusion equation Green's function forward model",
        "mismatch_params": [
            {"name": "noise_level", "min": 0.01, "max": 0.05, "unit": ""},
            {"name": "diffusion_coeff", "min": 0.5, "max": 2.0, "unit": "mm²/s"},
            {"name": "absorption", "min": 0.01, "max": 0.1, "unit": "1/mm"},
        ],
    },
    # Spectral model: y(ω) = Σ c·L(ω-ω₀) (MRS, CEST, Brillouin, CARS, SRS, SHG, etc.)
    "spectral": {
        "description": "Spectral peaks / nonlinear optical signal model",
        "mismatch_params": [
            {"name": "noise_level", "min": 0.01, "max": 0.05, "unit": ""},
            {"name": "linewidth", "min": 1.0, "max": 5.0, "unit": "channels"},
            {"name": "baseline_drift", "min": -0.02, "max": 0.02, "unit": ""},
        ],
    },
    # Temporal decay: y(t) = Σ a·exp(-t/τ) (FLIM, pump-probe)
    "temporal_decay": {
        "description": "Exponential temporal decay + IRF convolution",
        "mismatch_params": [
            {"name": "noise_level", "min": 0.01, "max": 0.05, "unit": ""},
            {"name": "irf_width", "min": 0.5, "max": 2.0, "unit": "ns"},
            {"name": "background", "min": 0.0, "max": 0.02, "unit": ""},
        ],
    },
    # Polarimetric: y_k = S₀ + S₁·cos(2θ_k) (polarization, PolSAR)
    "polarimetric": {
        "description": "Malus's law polarimetric measurement at multiple analyzer angles",
        "mismatch_params": [
            {"name": "noise_level", "min": 0.01, "max": 0.05, "unit": ""},
            {"name": "angle_offset", "min": -2.0, "max": 2.0, "unit": "deg"},
            {"name": "diattenuation", "min": 0.9, "max": 1.0, "unit": ""},
        ],
    },
    # Lambertian rendering: y_k = ρ·max(n·l_k, 0) (photometric stereo, structured light)
    "lambertian": {
        "description": "Lambertian surface rendering under directional illumination",
        "mismatch_params": [
            {"name": "noise_level", "min": 0.01, "max": 0.05, "unit": ""},
            {"name": "ambient", "min": 0.02, "max": 0.1, "unit": ""},
            {"name": "specular", "min": 0.0, "max": 0.1, "unit": ""},
        ],
    },
    # Multi-exposure: y_k = clip(g(t_k·x), 0, 1) (HDR, panorama)
    "multi_exposure": {
        "description": "Multi-exposure bracketed capture with saturation",
        "mismatch_params": [
            {"name": "noise_level", "min": 0.01, "max": 0.05, "unit": ""},
            {"name": "gamma", "min": 1.8, "max": 2.4, "unit": ""},
            {"name": "exposure_ratio", "min": 2.0, "max": 4.0, "unit": "stops"},
        ],
    },
    # Temporal shearing/3D→2D: y = Σ_t M(t)·x_t (CUP, streak, Doppler)
    "temporal_shear": {
        "description": "Temporal shearing of 3D(x,y,t) datacube to 2D snapshot",
        "mismatch_params": [
            {"name": "noise_level", "min": 0.01, "max": 0.05, "unit": ""},
            {"name": "shear_rate", "min": 0.8, "max": 1.2, "unit": "px/frame"},
            {"name": "encoding_ratio", "min": 0.3, "max": 0.7, "unit": ""},
        ],
    },
    # Emissivity/radiative transfer: T_B = ε·T + atm (passive MW, multispectral, ocean color)
    "emissivity": {
        "description": "Thermal emissivity / radiative transfer model",
        "mismatch_params": [
            {"name": "noise_level", "min": 0.01, "max": 0.05, "unit": ""},
            {"name": "atmospheric_opacity", "min": 0.1, "max": 0.5, "unit": ""},
            {"name": "calibration_offset", "min": -0.02, "max": 0.02, "unit": "K"},
        ],
    },
    # Born approximation wave: y = G⊛x (FWI, sonar, ocean acoustic)
    "wave_born": {
        "description": "Born-approximation scattered wavefield",
        "mismatch_params": [
            {"name": "noise_level", "min": 0.01, "max": 0.05, "unit": ""},
            {"name": "frequency", "min": 5.0, "max": 20.0, "unit": "Hz"},
            {"name": "damping", "min": 0.01, "max": 0.1, "unit": ""},
        ],
    },
    # System matrix: y = A·x (MPI, eddy current, impedance)
    "system_matrix": {
        "description": "Dense system matrix linear forward model",
        "mismatch_params": [
            {"name": "noise_level", "min": 0.01, "max": 0.05, "unit": ""},
            {"name": "condition_number", "min": 10.0, "max": 100.0, "unit": ""},
            {"name": "sparsity", "min": 0.3, "max": 0.7, "unit": ""},
        ],
    },
    # Radon + CTF: y_θ = CTF·Radon_θ(x) (cryo-ET)
    "radon_ctf": {
        "description": "Tilt-series Radon projection with CTF modulation",
        "mismatch_params": [
            {"name": "noise_level", "min": 0.02, "max": 0.08, "unit": ""},
            {"name": "defocus", "min": 1.0, "max": 5.0, "unit": "μm"},
            {"name": "tilt_range", "min": 50.0, "max": 70.0, "unit": "deg"},
        ],
    },
    # Heat diffusion: T(t) = Q/√(αt) · f(d) (active thermography, elastography)
    "heat_diffusion": {
        "description": "1D heat diffusion for thermal/elastic response",
        "mismatch_params": [
            {"name": "noise_level", "min": 0.01, "max": 0.05, "unit": ""},
            {"name": "diffusivity", "min": 0.5, "max": 5.0, "unit": "mm²/s"},
            {"name": "surface_loss", "min": 0.0, "max": 0.1, "unit": ""},
        ],
    },
    # Beamforming: y = DAS(x) (ultrasonic phased array, IVUS)
    "beamform": {
        "description": "Delay-and-sum beamforming forward model",
        "mismatch_params": [
            {"name": "noise_level", "min": 0.01, "max": 0.05, "unit": ""},
            {"name": "speed_of_sound", "min": 1450.0, "max": 1550.0, "unit": "m/s"},
            {"name": "aperture_fraction", "min": 0.3, "max": 0.8, "unit": ""},
        ],
    },
    # Coronagraph: y = PSF_res⊛star + planet (coronagraphy)
    "coronagraph": {
        "description": "Coronagraphic starlight suppression + residual speckle",
        "mismatch_params": [
            {"name": "noise_level", "min": 0.01, "max": 0.05, "unit": ""},
            {"name": "contrast_ratio", "min": 1e-4, "max": 1e-3, "unit": ""},
            {"name": "speckle_rms", "min": 0.01, "max": 0.05, "unit": ""},
        ],
    },
    # Event camera: events when |Δlog(I)| > C
    "event_threshold": {
        "description": "Event camera log-intensity change threshold model",
        "mismatch_params": [
            {"name": "noise_level", "min": 0.01, "max": 0.05, "unit": ""},
            {"name": "threshold", "min": 0.1, "max": 0.3, "unit": "log"},
            {"name": "refractory_period", "min": 0.0, "max": 0.01, "unit": "ms"},
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
    "cest_mri": ("spectral", "brain"),
    "ceus": ("identity", "vascular"),
    "dexa": ("radon", "bone"),
    "digital_breast_tomo": ("radon", "breast"),
    "doppler_ultrasound": ("temporal_shear", "vascular"),
    "dot": ("diffusion", "tissue"),
    "elastography": ("heat_diffusion", "tissue"),
    "fluoroscopy": ("radon", "anatomy"),
    "impedance_tomo": ("diffusion", "tissue"),
    "ivus": ("beamform", "vascular"),
    "magnetic_particle": ("system_matrix", "anatomy"),
    "mr_elastography": ("kspace", "brain"),
    "mr_fingerprinting": ("kspace", "brain"),
    "mra": ("kspace", "vascular"),
    "mrs": ("spectral", "brain"),
    "muon_tomo": ("radon", "material"),
    "neutron_tomo": ("radon", "material"),
    "nirs_brain": ("diffusion", "brain"),
    "octa": ("psf", "vascular"),
    "phase_contrast": ("psf", "cell"),
    "portal_imaging": ("radon", "anatomy"),
    "proton_radiography": ("radon", "material"),
    "proton_therapy_img": ("radon", "anatomy"),
    "spectral_ct": ("radon", "anatomy"),  # already has local generator
    "swi": ("kspace", "brain"),
    "us_mri": ("kspace", "anatomy"),
    "xray_ndt": ("radon", "material"),
    "xray_radiography": ("radon", "anatomy"),

    # Microscopy
    "cars": ("spectral", "cell"),
    "cathodoluminescence": ("spectral", "nanostructure"),
    "clem": ("psf", "cell"),
    "confocal_endomicroscopy": ("psf", "tissue"),
    "confocal_livecell": ("psf", "cell"),
    "dark_field": ("psf", "nanostructure"),
    "dic": ("interference", "cell"),
    "dna_paint": ("psf", "molecule"),
    "expansion": ("psf", "cell"),
    "flim": ("temporal_decay", "cell"),
    "ism": ("psf", "cell"),
    "lattice_lightsheet": ("psf", "cell"),
    "minflux": ("psf", "molecule"),
    "spinning_disk": ("psf", "cell"),
    "srs": ("spectral", "cell"),
    "shg": ("spectral", "tissue"),
    "three_photon": ("psf", "neural"),
    "tirf": ("psf", "cell"),
    "widefield_lowdose": ("psf", "cell"),

    # Electron microscopy
    "cryo_et": ("radon_ctf", "protein"),
    "ebsd": ("fourier_mag", "crystal"),
    "edx_mapping": ("identity", "material"),
    "eels": ("identity", "material"),
    "electron_diffraction": ("fourier_mag", "crystal"),
    "electron_holography": ("interference", "nanostructure"),
    "electron_tomography": ("radon", "nanostructure"),
    "fib_sem": ("identity", "material"),
    "stem": ("identity", "nanostructure"),

    # Remote sensing
    "flash_lidar": ("radon", "depth"),
    "multispectral_sat": ("emissivity", "terrain"),
    "ocean_color": ("emissivity", "terrain"),
    "passive_microwave": ("emissivity", "terrain"),
    "polsar": ("polarimetric", "terrain"),
    "weather_radar": ("emissivity", "terrain"),

    # Computational optics
    "cassi": ("mask", "natural"),
    "coded_exposure": ("psf", "natural"),
    "coronagraphy": ("coronagraph", "star"),
    "eht_imaging": ("kspace", "astronomy"),
    "entangled_photon": ("mask", "natural"),
    "event_camera": ("event_threshold", "natural"),
    "hdr_imaging": ("multi_exposure", "natural"),
    "integral": ("psf", "natural"),
    "light_field": ("psf", "natural"),
    "lucky_imaging": ("psf", "astronomy"),
    "machine_vision": ("psf", "natural"),
    "matrix": ("mask", "natural"),
    "nerf": ("psf", "natural"),
    "panorama": ("multi_exposure", "natural"),
    "photometric_stereo": ("lambertian", "depth"),
    "polarization": ("polarimetric", "natural"),
    "structured_light": ("lambertian", "depth"),

    # Spectroscopy / chemical
    "atom_probe": ("identity", "crystal"),
    "brillouin": ("spectral", "tissue"),
    "ct_fluorescence": ("diffusion", "anatomy"),
    "desi": ("identity", "tissue"),
    "libs": ("identity", "material"),
    "maldi_msi": ("identity", "tissue"),
    "mfm": ("system_matrix", "surface"),
    "nsom": ("psf", "nanostructure"),
    "pump_probe": ("temporal_decay", "spectral"),
    "quantum_illumination": ("mask", "natural"),
    "sims": ("identity", "material"),
    "stm": ("identity", "surface"),
    "xrf_imaging": ("identity", "material"),
    "xrf_tomo": ("radon", "material"),

    # Specialized
    "acoustic_emission": ("identity", "material"),
    "acoustic_microscopy": ("psf", "material"),
    "active_thermography": ("heat_diffusion", "material"),
    "adaptive_optics": ("psf", "astronomy"),
    "afm": ("identity", "surface"),
    "bioluminescence_tomo": ("diffusion", "anatomy"),
    "cup": ("temporal_shear", "natural"),
    "eddy_current": ("system_matrix", "material"),
    "fwi": ("wave_born", "terrain"),
    "ghost_imaging": ("mask", "natural"),  # already has local
    "gpr": ("radon", "terrain"),
    "gravitational_wave": ("spectral", "signal"),
    "neutron_diffraction": ("fourier_mag", "crystal"),
    "ocean_acoustic_tomo": ("wave_born", "depth"),
    "particle_calorimetry": ("identity", "particle"),
    "radio_astronomy": ("kspace", "astronomy"),
    "radio_interferometry": ("kspace", "astronomy"),
    "saxs": ("fourier_mag", "crystal"),
    "seismic_tomo": ("radon", "terrain"),
    "shearography": ("interference", "material"),
    "solar_imaging": ("identity", "astronomy"),
    "sonar": ("wave_born", "depth"),
    "spc": ("mask", "natural"),
    "spc_block": ("mask", "natural"),
    "streak_camera": ("temporal_shear", "natural"),
    "talbot_lau": ("interference", "anatomy"),
    "terahertz": ("psf", "material"),
    "tof_camera": ("identity", "depth"),
    "ultrasonic_phased_array": ("beamform", "material"),
    "waxs": ("fourier_mag", "crystal"),
    "xfel_sfx": ("fourier_mag", "crystal"),
    "xray_crystallography": ("fourier_mag", "crystal"),
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


    elif model_type == "fourier_mag":
        # Fourier magnitude (diffraction): y = |F(x)|² + noise
        noise = mismatch_params.get("noise_level", 0.02)
        kspace = np.fft.fftshift(np.fft.fft2(x_true))
        y = np.abs(kspace)**2
        # Normalize to [0, 1] range for storage
        y = y / (y.max() + 1e-8)
        # Log scale for better dynamic range (like real detectors)
        y = np.log1p(y * 1000) / np.log1p(1000)
        y += noise * rng.randn(*y.shape)
        y = np.maximum(y, 0).astype(np.float32)
        return y, {"forward_type": "fourier_magnitude"}

    elif model_type == "interference":
        # Interference: y = |R + O·exp(i·2π·x·f)|²
        noise = mismatch_params.get("noise_level", 0.02)
        fringe = mismatch_params.get("fringe_spacing", 12.0)
        vis = mismatch_params.get("visibility", 0.9)
        # Treat x as phase object
        phase = x_true * 2 * np.pi  # Scale to [0, 2π]
        # Reference wave (off-axis tilted plane wave)
        yy = np.arange(size).reshape(-1, 1)
        xx = np.arange(size).reshape(1, -1)
        ref_phase = 2 * np.pi * xx / fringe
        # Interference
        obj = vis * np.exp(1j * phase)
        ref = np.exp(1j * ref_phase)
        y = np.abs(ref + obj)**2
        y = y / (y.max() + 1e-8)
        y += noise * rng.randn(*y.shape)
        y = np.maximum(y, 0).astype(np.float32)
        return y, {"fringe_spacing": float(fringe)}

    elif model_type == "diffusion":
        # Diffusion Green's function: y_boundary = G · x_interior
        noise = mismatch_params.get("noise_level", 0.02)
        D = mismatch_params.get("diffusion_coeff", 1.0)
        mu_a = mismatch_params.get("absorption", 0.05)
        # Build Green's function kernel (diffusion PSF with exponential decay)
        yy = np.arange(size).reshape(-1, 1) - size / 2
        xx = np.arange(size).reshape(1, -1) - size / 2
        r = np.sqrt(xx**2 + yy**2) + 1e-8
        # 2D diffusion Green's function with absorption
        eff_atten = np.sqrt(mu_a / D)
        G = np.exp(-eff_atten * r) / (2 * np.pi * D * r + 1e-8)
        G = G / (G.sum() + 1e-8)
        # Forward: boundary measurement = convolution
        y = fftconvolve(x_true, G, mode="same")
        y = y / (y.max() + 1e-8)
        y += noise * rng.randn(*y.shape)
        y = np.maximum(y, 0).astype(np.float32)
        return y, {"diffusion_coeff": float(D), "absorption": float(mu_a)}

    elif model_type == "spectral":
        # Spectral model: y(ω) = Σ c·L(ω-ω₀, Γ) per pixel → 2D spectral image
        noise = mismatch_params.get("noise_level", 0.02)
        lw = mismatch_params.get("linewidth", 3.0)
        baseline = mismatch_params.get("baseline_drift", 0.0)
        # Generate spectral measurement: treat rows as spatial, columns as spectral channels
        n_peaks = 5
        peak_positions = rng.uniform(0.1, 0.9, n_peaks) * size
        peak_widths = np.full(n_peaks, lw)
        channels = np.arange(size).reshape(1, -1)
        # Each pixel's spectrum is sum of Lorentzian peaks weighted by local concentration
        spectrum_template = np.zeros((1, size), dtype=np.float64)
        for k in range(n_peaks):
            spectrum_template += 1.0 / (1 + ((channels - peak_positions[k]) / peak_widths[k])**2)
        # Weight by spatial content (average along columns for concentration)
        concentration = x_true  # Each row is a spatial location
        y = concentration * spectrum_template  # Broadcasting: (size, size)
        y = y / (y.max() + 1e-8) + baseline
        y += noise * rng.randn(*y.shape)
        y = np.maximum(y, 0).astype(np.float32)
        return y, {"n_peaks": n_peaks, "peak_positions": peak_positions.tolist()}

    elif model_type == "temporal_decay":
        # Exponential decay: y(t) = Σ a·exp(-t/τ) ⊛ IRF
        noise = mismatch_params.get("noise_level", 0.02)
        irf_w = mismatch_params.get("irf_width", 1.0)
        bg = mismatch_params.get("background", 0.01)
        # Treat columns as time bins, rows as spatial
        t = np.linspace(0, 10, size).reshape(1, -1)  # ns
        # Lifetime map from image: τ = 1 + 4*x (range 1-5 ns)
        tau = 1.0 + 4.0 * x_true[:, :1]  # (size, 1) — one lifetime per row
        amplitude = x_true.mean(axis=1, keepdims=True)
        # Decay curves
        y = amplitude * np.exp(-t / (tau + 1e-8)) + bg
        # Convolve with IRF (Gaussian)
        irf = np.exp(-0.5 * (t - 2.0)**2 / (irf_w**2 + 1e-8))
        irf = irf / (irf.sum() + 1e-8)
        from scipy.signal import fftconvolve as fftconv
        for i in range(size):
            y[i] = fftconv(y[i], irf[0], mode="same")
        y = y / (y.max() + 1e-8)
        y += noise * rng.randn(*y.shape)
        y = np.maximum(y, 0).astype(np.float32)
        return y, {"irf_width": float(irf_w)}

    elif model_type == "polarimetric":
        # Malus's law: y_k = 0.5*(S₀ + S₁·cos(2θ_k) + S₂·sin(2θ_k))
        noise = mismatch_params.get("noise_level", 0.02)
        angle_off = mismatch_params.get("angle_offset", 0.0)
        # 4 analyzer angles (0°, 45°, 90°, 135°) → pack into 2D
        S0 = x_true  # Total intensity
        S1 = x_true * 0.5 * np.cos(2 * np.pi * x_true)  # DOP × cos(2χ)
        S2 = x_true * 0.5 * np.sin(2 * np.pi * x_true)  # DOP × sin(2χ)
        angles = np.array([0, 45, 90, 135]) + angle_off
        half_h = size // 2
        half_w = size // 2
        y = np.zeros((size, size), dtype=np.float64)
        for k, theta in enumerate(angles):
            theta_rad = np.deg2rad(theta)
            I_k = 0.5 * (S0 + S1 * np.cos(2 * theta_rad) + S2 * np.sin(2 * theta_rad))
            # Place in quadrant
            r0 = (k // 2) * half_h
            c0 = (k % 2) * half_w
            y[r0:r0+half_h, c0:c0+half_w] = I_k[:half_h, :half_w]
        y = y / (y.max() + 1e-8)
        y += noise * rng.randn(*y.shape)
        y = np.maximum(y, 0).astype(np.float32)
        return y, {"analyzer_angles": [0, 45, 90, 135]}

    elif model_type == "lambertian":
        # Lambertian: y_k = ρ·max(n·l_k, 0) for K light directions
        noise = mismatch_params.get("noise_level", 0.02)
        ambient = mismatch_params.get("ambient", 0.05)
        # Compute surface normals from depth map (x_true treated as height map)
        dz_dx = np.gradient(x_true, axis=1)
        dz_dy = np.gradient(x_true, axis=0)
        # Normal = (-dz/dx, -dz/dy, 1) normalized
        norm_factor = np.sqrt(dz_dx**2 + dz_dy**2 + 1)
        nx, ny, nz = -dz_dx / norm_factor, -dz_dy / norm_factor, 1.0 / norm_factor
        albedo = x_true
        # 4 light directions (pack into quadrants)
        lights = [(0.5, 0.5, 0.7), (-0.5, 0.5, 0.7), (0.5, -0.5, 0.7), (-0.5, -0.5, 0.7)]
        half_h, half_w = size // 2, size // 2
        y = np.zeros((size, size), dtype=np.float64)
        for k, (lx, ly, lz) in enumerate(lights):
            ln = np.sqrt(lx**2 + ly**2 + lz**2)
            lx, ly, lz = lx/ln, ly/ln, lz/ln
            shade = np.maximum(nx*lx + ny*ly + nz*lz, 0) * albedo + ambient
            r0 = (k // 2) * half_h
            c0 = (k % 2) * half_w
            y[r0:r0+half_h, c0:c0+half_w] = shade[:half_h, :half_w]
        y = y / (y.max() + 1e-8)
        y += noise * rng.randn(*y.shape)
        y = np.maximum(y, 0).astype(np.float32)
        return y, {"n_lights": 4, "light_dirs": lights}

    elif model_type == "multi_exposure":
        # Multi-exposure: y_k = clip(gamma(t_k · x), 0, 1)
        noise = mismatch_params.get("noise_level", 0.02)
        gamma = mismatch_params.get("gamma", 2.2)
        exp_ratio = mismatch_params.get("exposure_ratio", 3.0)
        # 4 exposures packed into quadrants
        exposures = [1.0/exp_ratio**2, 1.0/exp_ratio, 1.0, exp_ratio]
        half_h, half_w = size // 2, size // 2
        y = np.zeros((size, size), dtype=np.float64)
        for k, t_k in enumerate(exposures):
            exposed = np.clip(t_k * x_true, 0, 1)
            # Apply camera response (gamma)
            ldr = np.clip(exposed ** (1.0 / gamma), 0, 1)
            ldr += noise * rng.randn(*ldr.shape)
            r0 = (k // 2) * half_h
            c0 = (k % 2) * half_w
            y[r0:r0+half_h, c0:c0+half_w] = ldr[:half_h, :half_w]
        y = np.maximum(y, 0).astype(np.float32)
        return y, {"exposures": exposures, "gamma": float(gamma)}

    elif model_type == "temporal_shear":
        # 3D(x,y,t) → 2D: y = Σ_t C(t) · S(t) · x_frame(t)
        noise = mismatch_params.get("noise_level", 0.02)
        shear = mismatch_params.get("shear_rate", 1.0)
        encoding = mismatch_params.get("encoding_ratio", 0.5)
        # Generate a synthetic 3D datacube from x_true (use shifted versions as temporal frames)
        n_frames = 8
        y = np.zeros((size, size), dtype=np.float64)
        for t in range(n_frames):
            # Each frame: shifted version of x_true (simulating temporal evolution)
            shift_y = int(t * shear * 3)
            frame = np.roll(x_true, shift_y, axis=0) * (1 - 0.05 * t)
            # Spatial encoding mask (random binary, different per frame)
            rng_t = np.random.RandomState(rng.randint(0, 100000))
            code = (rng_t.rand(size, size) > (1 - encoding)).astype(np.float32)
            # Temporal shearing: shift frame in detector by t*shear_rate pixels
            shear_shift = int(t * shear * size / n_frames)
            coded_frame = np.roll(frame * code, shear_shift, axis=1)
            y += coded_frame
        y = y / (y.max() + 1e-8)
        y += noise * rng.randn(*y.shape)
        y = np.maximum(y, 0).astype(np.float32)
        return y, {"n_frames": n_frames, "shear_rate": float(shear)}

    elif model_type == "emissivity":
        # Radiative transfer: y = ε·T_surface + (1-ε)·T_atm + path_radiance
        noise = mismatch_params.get("noise_level", 0.02)
        tau_atm = mismatch_params.get("atmospheric_opacity", 0.3)
        cal_off = mismatch_params.get("calibration_offset", 0.0)
        # Treat x_true as surface property (reflectance/emissivity map)
        emissivity = 0.3 + 0.7 * x_true  # Emissivity range [0.3, 1.0]
        T_surface = 0.5 + 0.5 * x_true  # Surface temperature (normalized)
        T_atm = 0.3  # Atmospheric temperature (constant)
        # At-sensor radiance
        transmittance = np.exp(-tau_atm)
        y = transmittance * emissivity * T_surface + (1 - transmittance) * T_atm + cal_off
        y = y / (y.max() + 1e-8)
        y += noise * rng.randn(*y.shape)
        y = np.maximum(y, 0).astype(np.float32)
        return y, {"atmospheric_opacity": float(tau_atm)}

    elif model_type == "wave_born":
        # Born approximation: y = G_wave ⊛ (Δv · x) where G is wave Green's function
        noise = mismatch_params.get("noise_level", 0.02)
        freq = mismatch_params.get("frequency", 10.0)
        damping = mismatch_params.get("damping", 0.05)
        # Wavenumber
        k = 2 * np.pi * freq / 100  # Normalized
        yy = np.arange(size).reshape(-1, 1) - size / 2
        xx = np.arange(size).reshape(1, -1) - size / 2
        r = np.sqrt(xx**2 + yy**2) + 1e-8
        # 2D Green's function (Hankel ~ exp(ikr)/sqrt(r))
        G = np.exp(1j * k * r - damping * r) / np.sqrt(r + 1e-8)
        G = G / (np.abs(G).sum() + 1e-8)
        # Scattered field
        scattered = fftconvolve(x_true, np.real(G), mode="same")
        # Take magnitude (receivers see amplitude)
        y = np.abs(scattered)
        y = y / (y.max() + 1e-8)
        y += noise * rng.randn(*y.shape)
        y = np.maximum(y, 0).astype(np.float32)
        return y, {"frequency": float(freq), "damping": float(damping)}

    elif model_type == "system_matrix":
        # Dense system matrix: y = A·x_vec (reshaped back to 2D)
        noise = mismatch_params.get("noise_level", 0.02)
        cond = mismatch_params.get("condition_number", 50.0)
        sparsity = mismatch_params.get("sparsity", 0.5)
        # Downsample for tractability: work on 64x64 blocks
        block = 4
        small_size = size // block
        x_small = x_true.reshape(small_size, block, small_size, block).mean(axis=(1, 3))
        n_px = small_size * small_size
        n_meas = n_px  # Same number of measurements
        # Generate structured random system matrix
        A = rng.randn(n_meas, n_px).astype(np.float32)
        # Apply sparsity mask
        A *= (rng.rand(n_meas, n_px) < sparsity).astype(np.float32)
        # Normalize rows
        row_norms = np.sqrt((A**2).sum(axis=1, keepdims=True)) + 1e-8
        A = A / row_norms
        # Forward
        x_vec = x_small.ravel()
        y_vec = A @ x_vec
        y_small = y_vec.reshape(small_size, small_size)
        # Upsample back
        y = np.kron(y_small, np.ones((block, block)))
        y = y / (np.abs(y).max() + 1e-8)
        y += noise * rng.randn(*y.shape)
        y = np.maximum(y, 0).astype(np.float32)
        return y, {"n_measurements": n_meas, "sparsity": float(sparsity)}

    elif model_type == "radon_ctf":
        # Radon + CTF: y_θ = CTF(Radon_θ(x)) — cryo-ET tilt series
        noise = mismatch_params.get("noise_level", 0.05)
        defocus = mismatch_params.get("defocus", 3.0)
        tilt_range = mismatch_params.get("tilt_range", 60.0)
        n_tilts = 41
        max_angle = np.deg2rad(tilt_range)
        angles = np.linspace(-max_angle, max_angle, n_tilts)
        n_det = int(np.ceil(np.sqrt(2) * size))
        sino = np.zeros((n_tilts, n_det), dtype=np.float64)
        coords = np.arange(size) - size / 2.0 + 0.5
        X, Y = np.meshgrid(coords, coords)
        for i, theta in enumerate(angles):
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            proj = X * cos_t + Y * sin_t
            proj_shifted = proj + n_det / 2.0
            sino[i] = np.histogram(proj_shifted.ravel(), bins=n_det, range=(0, n_det), weights=x_true.ravel())[0]
        # Apply CTF: oscillating contrast transfer
        freq = np.fft.fftfreq(n_det)
        ctf = np.sin(np.pi * defocus * freq**2 * 1000)  # Simplified CTF
        for i in range(n_tilts):
            sino_f = np.fft.fft(sino[i])
            sino[i] = np.real(np.fft.ifft(sino_f * ctf))
        sino += noise * rng.randn(*sino.shape) * (np.abs(sino).max() + 1e-8)
        sino = sino.astype(np.float32)
        return sino, {"theta": angles.astype(np.float32), "n_det": n_det, "defocus": float(defocus)}

    elif model_type == "heat_diffusion":
        # 1D heat diffusion: T(x,t) response to impulsive heating
        noise = mismatch_params.get("noise_level", 0.02)
        alpha = mismatch_params.get("diffusivity", 2.0)
        loss = mismatch_params.get("surface_loss", 0.05)
        # Treat x_true as defect depth map → surface temperature response
        # T(t) ∝ 1/√(t) · [1 - f(d/√(αt))]
        # Map to 2D: rows = spatial, columns = time steps
        n_times = size
        t = np.linspace(0.1, 10.0, n_times).reshape(1, -1)  # time in seconds
        # Defect depth from image intensity (deeper = lower intensity)
        depth = 0.5 + 2.0 * x_true[:, :1]  # (size, 1) depth in mm
        # Surface temperature response
        T_surface = (1.0 / np.sqrt(np.pi * alpha * t)) * np.exp(-depth**2 / (4 * alpha * t + 1e-8))
        T_surface *= np.exp(-loss * t)  # Surface heat loss
        y = T_surface
        y = y / (y.max() + 1e-8)
        y += noise * rng.randn(*y.shape)
        y = np.maximum(y, 0).astype(np.float32)
        return y, {"diffusivity": float(alpha)}

    elif model_type == "beamform":
        # Delay-and-sum beamforming: y(r,θ) = Σ_n w_n · x(r_n(delay))
        noise = mismatch_params.get("noise_level", 0.02)
        c = mismatch_params.get("speed_of_sound", 1500.0)
        aperture = mismatch_params.get("aperture_fraction", 0.5)
        # Simulate ultrasound B-mode: PSF varies with depth (axial resolution degrades)
        yy = np.arange(size).reshape(-1, 1).astype(np.float64)  # depth axis
        xx = np.arange(size).reshape(1, -1).astype(np.float64)  # lateral axis
        # Depth-dependent PSF (lateral resolution degrades with depth)
        y = np.zeros((size, size), dtype=np.float64)
        for depth_idx in range(size):
            depth_frac = (depth_idx + 1) / size
            # Lateral PSF width increases with depth (beam divergence)
            sigma_lat = 1.0 + 3.0 * depth_frac * (1 - aperture)
            kernel_w = int(6 * sigma_lat) + 1
            if kernel_w % 2 == 0:
                kernel_w += 1
            kernel = np.exp(-0.5 * (np.arange(kernel_w) - kernel_w // 2)**2 / (sigma_lat**2 + 1e-8))
            kernel /= kernel.sum()
            row = np.convolve(x_true[depth_idx], kernel, mode="same")
            # Depth attenuation
            row *= np.exp(-0.02 * depth_idx)
            y[depth_idx] = row
        # Add speckle
        speckle = np.abs(rng.randn(*y.shape) + 1j * rng.randn(*y.shape))
        y = y * (0.5 + 0.5 * speckle)
        y = y / (y.max() + 1e-8)
        y += noise * rng.randn(*y.shape)
        y = np.maximum(y, 0).astype(np.float32)
        return y, {"speed_of_sound": float(c)}

    elif model_type == "coronagraph":
        # Coronagraph: residual starlight PSF + planet signals
        noise = mismatch_params.get("noise_level", 0.02)
        contrast = mismatch_params.get("contrast_ratio", 5e-4)
        speckle_rms = mismatch_params.get("speckle_rms", 0.03)
        # Star at center with coronagraph suppression
        yy = np.arange(size).reshape(-1, 1) - size / 2
        xx = np.arange(size).reshape(1, -1) - size / 2
        r = np.sqrt(xx**2 + yy**2) + 1e-8
        # Residual stellar halo (Airy-like + speckles)
        stellar_halo = contrast / (r**2 + 1)
        # Random speckle field (quasi-static)
        speckle = speckle_rms * rng.randn(size, size)
        speckle = gaussian_filter(speckle, sigma=3.0)
        # Planet signals from x_true (faint point sources)
        y = stellar_halo + np.abs(speckle) + x_true * 0.1
        y = y / (y.max() + 1e-8)
        y += noise * rng.randn(*y.shape)
        y = np.maximum(y, 0).astype(np.float32)
        return y, {"contrast_ratio": float(contrast)}

    elif model_type == "event_threshold":
        # Event camera: binary events where |Δlog(I)| > threshold
        noise = mismatch_params.get("noise_level", 0.02)
        threshold = mismatch_params.get("threshold", 0.2)
        # Generate two frames (current and reference) from x_true
        I_ref = np.clip(x_true, 0.01, 1.0)
        # Simulated motion: shift x_true slightly
        I_cur = np.roll(x_true, 3, axis=1) * rng.uniform(0.9, 1.1)
        I_cur = np.clip(I_cur, 0.01, 1.0)
        # Log intensity change
        delta_log = np.log(I_cur) - np.log(I_ref)
        # Events: +1 for ON (positive change), -1 for OFF (negative change), 0 otherwise
        events = np.zeros_like(delta_log)
        events[delta_log > threshold] = 1.0
        events[delta_log < -threshold] = 0.5  # Encode OFF events as 0.5
        # Add noise
        y = events + noise * rng.randn(*events.shape)
        y = np.clip(y, 0, 1).astype(np.float32)
        return y, {"threshold": float(threshold)}

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
                    if "defocus" in h_ideal_dict:
                        grp.attrs["defocus"] = h_ideal_dict["defocus"]
                elif "mask" in h_ideal_dict:
                    grp.create_dataset("H_ideal", data=h_ideal_dict["mask"], compression="gzip")
                # Store any other metadata from h_ideal_dict as attributes
                for key, val in h_ideal_dict.items():
                    if key not in ("psf", "theta", "mask", "n_det", "defocus"):
                        try:
                            if isinstance(val, (int, float, str)):
                                grp.attrs[key] = val
                            elif isinstance(val, list) and len(val) < 100:
                                grp.attrs[key] = str(val)
                        except Exception:
                            pass

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
                    elif model_type in ("psf", "diffusion", "wave_born", "emissivity",
                                        "beamform", "coronagraph", "event_threshold"):
                        recon = y  # Same shape, direct display
                    elif model_type in ("spectral", "temporal_decay", "polarimetric",
                                        "lambertian", "multi_exposure", "heat_diffusion",
                                        "fourier_mag", "interference", "temporal_shear",
                                        "system_matrix"):
                        recon = y  # 2D output, displayable
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
