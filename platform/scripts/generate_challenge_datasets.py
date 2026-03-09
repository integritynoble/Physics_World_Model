#!/usr/bin/env python3
"""Generate HDF5 challenge datasets for the Blind Reconstruction Challenge.

Creates Public, Dev, and Hidden HDF5 files for each variant.
Each tier uses DIFFERENT underlying datasets (different ground truth images)
via ``tier_data_sources``, plus different mismatch realizations
(different true_spec values + different noise seeds).

Supports all 168+ modalities via a generic generation pipeline that:
  1. Resolves ground-truth phantoms from the dataset registry / downloaders
  2. Applies physics-accurate forward models via 7 category runners
  3. Applies mismatch perturbations derived from the variant's spec_ranges
  4. Adds category-appropriate noise (Gaussian, Poisson, Poisson-Gaussian, speckle)

Public HDF5 schema (what contestants download — includes ground truth):
    /sample_{nn}/y           — measurements (corrupted by mismatch + noise)
    /sample_{nn}/H_ideal     — ideal operator components
    /sample_{nn}/spec_ranges — JSON string with mismatch ranges
    /sample_{nn}/metadata    — JSON string (scene name, dimensions, noise model)
    /sample_{nn}/x_true      — ground truth signal
    /sample_{nn}/true_spec   — JSON string with exact mismatch params

Dev HDF5 schema (contestants download — no ground truth):
    /sample_{nn}/y           — measurements (corrupted by mismatch + noise)
    /sample_{nn}/H_ideal     — ideal operator components
    /sample_{nn}/spec_ranges — JSON string with mismatch ranges
    /sample_{nn}/metadata    — JSON string (scene name, dimensions, noise model)

Hidden HDF5 schema (server-side only — includes ground truth for eval):
    /sample_{nn}/...         — same as Public (full data for server-side evaluation)

Usage:
    python scripts/generate_challenge_datasets.py --variant sd_cassi
    python scripts/generate_challenge_datasets.py --variant all
    python scripts/generate_challenge_datasets.py --variant all-challenge
    python scripts/generate_challenge_datasets.py --variant all-challenge --category microscopy
    python scripts/generate_challenge_datasets.py --variant ct --upload-gcs
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import h5py
import numpy as np

# Add parent and project root to path so we can import everything
_SCRIPT_DIR = Path(__file__).resolve().parent
_PLATFORM_DIR = _SCRIPT_DIR.parent
_PROJECT_ROOT = _PLATFORM_DIR.parent
sys.path.insert(0, str(_PLATFORM_DIR))
sys.path.insert(0, str(_PROJECT_ROOT))

from pwm_platform.services.benchmark_database._challenge_data import CHALLENGE_CONFIG

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  Category → Runner mapping
# ══════════════════════════════════════════════════════════════════════════════

# Maps the 21 category slugs (from _challenge_data.py / variant_registry) to
# one of 7 runner types that determines the forward model + H_ideal storage.
_CATEGORY_TO_RUNNER: dict[str, str] = {
    # Radon-based (sinogram projection)
    "medical":                    "radon",
    "particle_imaging":           "radon",
    # k-space-based (Fourier undersampling)
    "remote_sensing":             "kspace",
    # PSF-based (convolution with point-spread function)
    "microscopy":                 "psf",
    "clinical_optics":            "psf",
    "coherent":                   "psf",
    "computational":              "psf",
    "computational_photography":  "psf",
    "depth_imaging":              "psf",
    "spectroscopy":               "psf",
    "astronomy":                  "psf",
    "neural_rendering":           "psf",
    "experimental_science":       "psf",
    "scientific_instrumentation": "psf",
    "multi_modal_fusion":         "psf",
    "industrial_inspection":      "psf",
    "medical_ultrasound":         "psf",
    # CTF-based (contrast transfer function)
    "electron_microscopy":        "ctf",
    # Mask-based (binary coded aperture)
    "compressive":                "mask",
    "ultrafast":                  "mask",
    "quantum":                    "mask",
    # Scanning probe (tip convolution)
    "scanning_probe":             "tip",
}

# ── Variant-level runner overrides ───────────────────────────────────────────
# Some variants in a broad category need a different forward model.
_VARIANT_TO_RUNNER: dict[str, str] = {
    # DEXA is dual-energy projection, not CT sinogram
    "dexa": "dual_energy",
    # Mammography is 2D projection, not tomographic
    "mammography": "projection",
    # Fluoroscopy is real-time 2D projection
    "fluoroscopy": "projection",
    # X-ray radiography is 2D projection
    "xray_radiography": "projection",
    # ASL MRI is k-space undersampled MRI, NOT Radon-based (medical category default)
    "asl_mri": "kspace",
    # Atom Probe Tomography: position-sensitive ToF detector — use PSF runner
    # The "forward model" maps composition map -> detector hits via electrostatic
    # trajectory.  We approximate this with a PSF-based convolution (detector blur)
    # plus Poisson noise; the dedicated phantom handles microstructure.
    "atom_probe": "psf",
    # Bioluminescence Tomography: photon diffusion in tissue maps source -> surface flux.
    # The steady-state diffusion equation acts as a spatial low-pass filter (Green's
    # function convolution).  PSF runner approximates the diffusion blurring kernel.
    "bioluminescence_tomo": "psf",
    # Brachytherapy Imaging: multi-view X-ray Radon projection of I-125 seeds in tissue.
    # Use radon runner since the forward model is Radon-transform based.
    "brachytherapy_img": "radon",
    # Brillouin Microscopy: VIPA spectrometer — y is already the spectral measurement,
    # not a CT sinogram or k-space undersampling.  Identity runner applies minimal noise.
    "brillouin": "identity",
    # CARS Microscopy: CARS signal with NRB — y is the measured CARS intensity,
    # reconstruction recovers Im[chi^(3)].  Identity runner applies minimal noise.
    "cars": "identity",
    # Cathodoluminescence: CL intensity map with PSF broadening and PMT shot noise.
    # y is the blurred/noisy measurement; identity runner applies minimal noise.
    "cathodoluminescence": "identity",
    # CBCT: cone-beam CT uses Radon-transform based projection (FDK geometry).
    # Use radon runner since the forward model is Radon-transform based.
    "cbct": "radon",
    # CEST MRI: z-spectrum acquisition maps APT signal via exchange saturation.
    # y is the measured z-spectrum slice; identity runner applies minimal noise.
    "cest_mri": "identity",
    # CEUS: microbubble contrast-enhanced ultrasound — y is the combined B-mode +
    # contrast measurement; identity runner applies minimal noise.
    "ceus": "identity",
    # CLEM: correlative light and electron microscopy — y is the FM fluorescence image,
    # x_true is the EM ultrastructural image; identity runner applies minimal noise.
    "clem": "identity",
    # Coded exposure: flutter shutter deconvolution — y is the coded-blurred measurement,
    # x_true is the sharp ground truth frame; identity runner applies minimal noise.
    "coded_exposure": "identity",
    # Confocal 3D: optical sectioning deconvolution — y is the blurred+noisy max-projection,
    # x_true is the ground truth max-projection; identity runner applies minimal noise.
    "confocal_3d": "identity",
    # Confocal laser endomicroscopy: CLE mucosal imaging — y is the fibre-artefact/speckle
    # corrupted measurement; identity runner applies minimal noise.
    "confocal_endomicroscopy": "identity",
    # Confocal live-cell: low-dose fluorescence time-lapse denoising — y is the Poisson/Gaussian
    # noisy measurement; identity runner applies minimal noise.
    "confocal_livecell": "identity",
    # Coronagraphy: post-coronagraph focal-plane image with residual stellar speckles.
    # Reconstruction recovers the planet signal from the speckle background.
    # Identity runner applies minimal noise; phantom handles the full forward model.
    "coronagraphy": "identity",
    # Cryo-EM single-particle: CTF corruption and low-dose Poisson noise are handled
    # by the phantom generator; identity runner applies minimal additional noise.
    "cryo_em": "identity",
    # Cryo-ET cellular tomography: missing-wedge corruption is handled by the phantom
    # generator; identity runner applies minimal additional noise.
    "cryo_et": "identity",
    # CT: parallel-beam X-ray CT uses Radon-transform based projection.
    # Use radon runner since the forward model is sinogram/Radon-transform based.
    "ct": "radon",
    # XRF-CT: X-ray fluorescence CT — y is the Poisson-noisy fluorescence emission map
    # with Compton scatter background; identity runner applies minimal additional noise.
    "ct_fluorescence": "identity",
    # CUP: Compressed Ultrafast Photography — y is the compressed temporal measurement,
    # x_true is the dynamic scene frame; identity runner applies minimal additional noise.
    "cup": "identity",
    # Dark-field: sparse sub-wavelength particle scattering — y is Poisson+Gaussian noisy
    # measurement; identity runner applies the noise model defined in the phantom generator.
    "dark_field": "identity",
    # DESI-MSI: desorption electrospray ionization mass spectrometry imaging — y is the
    # multiplicative lognormal + Gaussian noisy measurement of lipid/metabolite spatial
    # distribution; identity runner applies the noise model defined in the phantom generator.
    "desi": "identity",
    # DIC: Differential Interference Contrast microscopy — y is the gradient-based intensity
    # measurement from the DIC shear kernel; identity runner applies minimal additional noise.
    "dic": "identity",
    # Diffusion MRI: k-space undersampled DTI acquisition — y is the undersampled k-space
    # reconstruction; kspace runner matches the FA map forward model.
    "diffusion_mri": "kspace",
    # Digital Breast Tomosynthesis: limited-angle Radon-transform projection (11 angles,
    # ±25°) with Poisson noise and FBP back-projection.  Radon runner matches the
    # limited-angle tomosynthesis forward model.
    "digital_breast_tomo": "radon",
    # DNA-PAINT: stochastic blinking forward model with Gaussian PSF — y is the
    # widefield diffraction-limited accumulation image; identity runner applies
    # minimal additional noise to the phantom's blinking accumulation.
    "dna_paint": "identity",
    # Doppler Ultrasound: Doppler frequency shift + speckle noise forward model —
    # y is the noisy Doppler measurement; identity runner applies minimal additional
    # noise to the phantom's speckle-corrupted Doppler signal.
    "doppler_ultrasound": "identity",
    # DOT: diffuse optical tomography — Born approximation boundary measurements
    # map absorption coefficient map to boundary flux; identity runner applies
    # minimal additional noise to the phantom's Born-approximation reconstruction.
    "dot": "identity",
    # EBSD: Kikuchi pattern degradation is handled by the phantom generator
    # (Voronoi grain boundary blur + Poisson shot noise); identity runner applies
    # minimal additional noise to the phantom's orientation map.
    "ebsd": "identity",
    # Eddy current: EM induction forward model is handled by the phantom generator
    # (blurred gradient of conductivity map + Gaussian noise); identity runner
    # applies minimal additional noise to the phantom's impedance signal map.
    "eddy_current": "identity",
    # EDX mapping: Poisson counting statistics and X-ray background (Bremsstrahlung)
    # are handled by the phantom generator; identity runner applies minimal
    # additional noise to the phantom's count map.
    "edx_mapping": "identity",
    # EELS: Poisson shot noise and multiple-scattering convolution are handled
    # by the phantom generator; identity runner applies minimal additional noise
    # to the phantom's chemical phase map.
    "eels": "identity",
    # EHT/VLBI: sparse u-v sampling and thermal noise handled by the phantom
    # generator; identity runner applies no additional degradation.
    "eht_imaging": "identity",
    # Elastography: shear wave displacement model and noise embedded in phantom;
    # identity runner applies no additional degradation.
    "elastography": "identity",
    # Electron Diffraction: Debye-Scherrer ring pattern with Poisson shot noise and
    # dynamic scattering are handled by the phantom generator; identity runner applies
    # no additional degradation.
    "electron_diffraction": "identity",
    # Electron Holography: off-axis fringe pattern with phase modulation and shot noise
    # are handled by the phantom generator; identity runner applies no additional degradation.
    "electron_holography": "identity",
    # Electron Tomography: limited-angle tilt series and back-projection with missing wedge
    # are handled by the phantom generator; identity runner applies no additional degradation.
    "electron_tomography": "identity",
}


# ── Data loaders──────────────────────────────────────────────────────────────


def _load_mat_scene(path: Path, key: str | None = None) -> np.ndarray:
    """Load a .mat file and return the signal array."""
    import scipy.io as sio

    data = sio.loadmat(str(path))
    if key is not None:
        return np.array(data[key], dtype=np.float64)
    # Try known keys first
    for k in ("img", "orig", "image"):
        if k in data:
            return np.array(data[k], dtype=np.float64)
    # Auto-detect: pick the largest non-metadata array
    candidates = {
        k: v for k, v in data.items()
        if not k.startswith("_") and isinstance(v, np.ndarray)
    }
    if not candidates:
        raise ValueError(f"No arrays found in {path}")
    best = max(candidates, key=lambda k: candidates[k].size)
    return np.array(candidates[best], dtype=np.float64)


def _load_tif_image(path: Path, target_size: tuple[int, int] | None = (256, 256)) -> np.ndarray:
    """Load a .tif image and return as float64 in [0, 1].

    Resizes to target_size if provided and image dimensions don't match.
    """
    from PIL import Image

    img = Image.open(path).convert("L")
    if target_size is not None and img.size != target_size:
        img = img.resize(target_size, Image.LANCZOS)
    return np.array(img, dtype=np.float64) / 255.0


# ── Ground truth resolver ────────────────────────────────────────────────────

def _get_variant_category(variant_key: str) -> str:
    """Look up the category for a variant, checking multiple sources."""
    try:
        from pwm_platform.services.benchmark_database._variant_registry import VARIANT_REGISTRY
        entry = VARIANT_REGISTRY.get(variant_key)
        if entry:
            return entry["category"]
    except ImportError:
        pass
    try:
        from pwm_platform.services.benchmark_database._modality_catalog import MODALITY_CATALOG
        entry = MODALITY_CATALOG.get(variant_key)
        if entry:
            return entry["category"]
    except ImportError:
        pass
    return "microscopy"  # safe default


def _get_runner_type(variant_key: str) -> str:
    """Determine the runner type for a variant.

    Checks variant-level overrides first, then falls back to category mapping.
    """
    if variant_key in _VARIANT_TO_RUNNER:
        return _VARIANT_TO_RUNNER[variant_key]
    category = _get_variant_category(variant_key)
    return _CATEGORY_TO_RUNNER.get(category, "psf")


def _crop_or_resize_2d(arr: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """Crop or resize an array to the target 2D spatial shape."""
    from PIL import Image

    th, tw = target_shape[0], target_shape[1]

    if arr.ndim == 2:
        if arr.shape == (th, tw):
            return arr
        img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
        img = img.resize((tw, th), Image.LANCZOS)
        return np.array(img, dtype=np.float64) / 255.0
    elif arr.ndim == 3:
        # 3D: resize each channel/slice
        H, W, C = arr.shape
        if H == th and W == tw:
            return arr
        out = np.zeros((th, tw, C), dtype=np.float64)
        for c in range(C):
            sl = arr[:, :, c]
            img = Image.fromarray((np.clip(sl, 0, 1) * 255).astype(np.uint8))
            img = img.resize((tw, th), Image.LANCZOS)
            out[:, :, c] = np.array(img, dtype=np.float64) / 255.0
        return out
    return arr


def _resolve_ground_truth(
    variant_key: str,
    signal_shape: tuple[int, ...],
    seed: int = 42,
    data_root: Path | None = None,
) -> np.ndarray:
    """Resolve a ground-truth signal for a variant.

    Priority: DATASET_REGISTRY generated phantoms (always available),
    then category-specific fallback phantoms.
    """
    try:
        from benchmarks.datasets.registry import get_datasets_for_modality
        from benchmarks.datasets.downloaders import (
            generate_medical_phantom, generate_em_phantom, generate_surface,
            generate_oct_phantom, generate_smlm_phantom, generate_depth_map,
            generate_test_scene, generate_star_field, generate_resolution_target,
            generate_diffraction_pattern, generate_elemental_map,
            generate_ndt_phantom, generate_velocity_model,
            generate_ae_source_map, generate_sam_phantom,
            generate_thermography_phantom, generate_ao_wavefront, generate_afm_surface,
            generate_angiography_vessel_phantom, generate_asl_perfusion_phantom,
            generate_apt_composition_map, generate_blt_source_phantom,
            generate_brachytherapy_seed_phantom,
            generate_brillouin_vipa_phantom,
            generate_cars_raman_phantom,
            generate_cathodoluminescence_phantom,
            generate_cbct_head_phantom,
            generate_cest_mri_phantom,
            generate_ceus_phantom,
            generate_clem_phantom,
            generate_coded_exposure_phantom,
            generate_confocal_3d_phantom,
            generate_confocal_endomicroscopy_phantom,
            generate_confocal_livecell_phantom,
            generate_coronagraphy_phantom,
            generate_cryo_em_phantom,
            generate_cryo_et_phantom,
            generate_ct_phantom,
            generate_ct_fluorescence_phantom,
            generate_cup_phantom,
            generate_dark_field_phantom,
            generate_dexa_phantom,
            generate_desi_phantom,
            generate_dic_phantom,
            generate_diffusion_mri_phantom,
            generate_digital_breast_tomo_phantom,
            generate_dna_paint_phantom,
            generate_doppler_ultrasound_phantom,
            generate_dot_phantom,
            generate_ebsd_phantom,
            generate_eddy_current_phantom,
            generate_eels_phantom,
            generate_eht_imaging_phantom,
            generate_elastography_phantom,
            generate_electron_diffraction_phantom,
            generate_electron_holography_phantom,
            generate_electron_tomography_phantom,
        )

        # Look up registry entries for this modality
        entries = get_datasets_for_modality(variant_key)
        # Sort by source priority: generated first (always available)
        _SOURCE_PRIORITY = {"generated": 0, "synthetic_web": 1, "web": 2, "experimental": 3}
        entries.sort(key=lambda e: _SOURCE_PRIORITY.get(e.source_type, 4))

        # Use the first "generated" entry (guaranteed to work without downloads)
        for entry in entries:
            if entry.source_type == "generated":
                target = tuple(signal_shape[:2])
                _GENERATOR_MAP = {
                    "generate_medical_phantom": generate_medical_phantom,
                    "generate_em_phantom": generate_em_phantom,
                    "generate_surface": generate_surface,
                    "generate_oct_phantom": generate_oct_phantom,
                    "generate_smlm_phantom": generate_smlm_phantom,
                    "generate_depth_map": generate_depth_map,
                    "generate_test_scene": generate_test_scene,
                    "generate_star_field": generate_star_field,
                    "generate_resolution_target": generate_resolution_target,
                    "generate_diffraction_pattern": generate_diffraction_pattern,
                    "generate_elemental_map": generate_elemental_map,
                    "generate_ndt_phantom": generate_ndt_phantom,
                    "generate_velocity_model": generate_velocity_model,
                    "generate_ae_source_map": generate_ae_source_map,
                    "generate_sam_phantom": generate_sam_phantom,
                    "generate_thermography_phantom": generate_thermography_phantom,
                    "generate_ao_wavefront": generate_ao_wavefront,
                    "generate_afm_surface": generate_afm_surface,
                    "generate_angiography_vessel_phantom": generate_angiography_vessel_phantom,
                    "generate_asl_perfusion_phantom": generate_asl_perfusion_phantom,
                    "generate_apt_composition_map": generate_apt_composition_map,
                    "generate_blt_source_phantom": generate_blt_source_phantom,
                    "generate_brachytherapy_seed_phantom": generate_brachytherapy_seed_phantom,
                    "generate_brillouin_vipa_phantom": generate_brillouin_vipa_phantom,
                    "generate_cars_raman_phantom": generate_cars_raman_phantom,
                    "generate_cathodoluminescence_phantom": generate_cathodoluminescence_phantom,
                    "generate_cbct_head_phantom": generate_cbct_head_phantom,
                    "generate_cest_mri_phantom": generate_cest_mri_phantom,
                    "generate_ceus_phantom": generate_ceus_phantom,
                    "generate_clem_phantom": generate_clem_phantom,
                    "generate_coded_exposure_phantom": generate_coded_exposure_phantom,
                    "generate_confocal_3d_phantom": generate_confocal_3d_phantom,
                    "generate_confocal_endomicroscopy_phantom": generate_confocal_endomicroscopy_phantom,
                    "generate_confocal_livecell_phantom": generate_confocal_livecell_phantom,
                    "generate_coronagraphy_phantom": generate_coronagraphy_phantom,
                    "generate_cryo_em_phantom": generate_cryo_em_phantom,
                    "generate_cryo_et_phantom": generate_cryo_et_phantom,
                    "generate_ct_phantom": generate_ct_phantom,
                    "generate_ct_fluorescence_phantom": generate_ct_fluorescence_phantom,
                    "generate_cup_phantom": generate_cup_phantom,
                    "generate_dark_field_phantom": generate_dark_field_phantom,
                    "generate_dexa_phantom": generate_dexa_phantom,
                    "generate_desi_phantom": generate_desi_phantom,
                    "generate_dic_phantom": generate_dic_phantom,
                    "generate_diffusion_mri_phantom": generate_diffusion_mri_phantom,
                    "generate_digital_breast_tomo_phantom": generate_digital_breast_tomo_phantom,
                    "generate_dna_paint_phantom": generate_dna_paint_phantom,
                    "generate_doppler_ultrasound_phantom": generate_doppler_ultrasound_phantom,
                    "generate_dot_phantom": generate_dot_phantom,
                    "generate_ebsd_phantom": generate_ebsd_phantom,
                    "generate_eddy_current_phantom": generate_eddy_current_phantom,
                    "generate_eels_phantom": generate_eels_phantom,
                    "generate_eht_imaging_phantom": generate_eht_imaging_phantom,
                    "generate_elastography_phantom": generate_elastography_phantom,
                    "generate_electron_diffraction_phantom": generate_electron_diffraction_phantom,
                    "generate_electron_holography_phantom": generate_electron_holography_phantom,
                    "generate_electron_tomography_phantom": generate_electron_tomography_phantom,
                }
                gen_fn = _GENERATOR_MAP.get(entry.converter)
                if gen_fn:
                    result = gen_fn(target_shape=target, seed=seed)
                    # Some generators return list[dict]; extract x_true from first sample
                    if isinstance(result, list):
                        arr = result[0]["x_true"] if result else np.zeros(target, dtype=np.float32)
                    else:
                        arr = result
                    return _crop_or_resize_2d(arr.astype(np.float64), signal_shape)

    except ImportError:
        logger.debug("benchmarks.datasets not available, using fallback phantoms")

    # Fallback: generate a phantom based on category
    return _generate_fallback_phantom(variant_key, signal_shape, seed)


def _generate_fallback_phantom(
    variant_key: str,
    signal_shape: tuple[int, ...],
    seed: int,
) -> np.ndarray:
    """Generate a category-appropriate fallback phantom."""
    rng = np.random.RandomState(seed)
    H = signal_shape[0]
    W = signal_shape[1] if len(signal_shape) > 1 else H

    runner_type = _get_runner_type(variant_key)

    if runner_type == "dual_energy":
        # DEXA: bone + soft tissue material maps
        return _make_dexa_phantom(H, W, rng)
    elif runner_type == "projection":
        # 2D X-ray projection (mammography, fluoroscopy, radiography)
        arr = _make_shepp_logan(H, W)  # anatomical phantom
    elif runner_type == "radon":
        # Shepp-Logan-like phantom
        arr = _make_shepp_logan(H, W)
    elif runner_type == "kspace":
        # Brain-like phantom for MRI/SAR
        arr = _make_brain_phantom(H, W, rng)
    elif runner_type == "ctf":
        # Particle phantom for EM
        arr = _make_particle_phantom(H, W, rng)
    elif runner_type == "mask":
        # Spectral/textured scene for compressive
        arr = _make_test_scene(H, W, rng)
    elif runner_type == "tip":
        # Surface topography
        arr = _make_surface(H, W, rng)
    else:
        # Generic cell/microscopy phantom
        arr = _make_cell_phantom(H, W, rng)

    # Handle 3D shapes (e.g., medical volumes [128, 128, 64])
    if len(signal_shape) == 3:
        D = signal_shape[2]
        volume = np.stack([arr * (0.8 + 0.2 * np.sin(np.pi * d / D))
                           for d in range(D)], axis=-1)
        return volume

    return arr


def _make_shepp_logan(H: int, W: int) -> np.ndarray:
    """Simple Shepp-Logan phantom."""
    yy = np.linspace(-1, 1, H)
    xx = np.linspace(-1, 1, W)
    X, Y = np.meshgrid(xx, yy)
    arr = np.zeros((H, W), dtype=np.float64)
    arr[(X / 0.85)**2 + (Y / 0.95)**2 < 1] = 0.15
    arr[((X - 0.2) / 0.25)**2 + ((Y + 0.1) / 0.35)**2 < 1] = 0.6
    arr[((X + 0.25) / 0.20)**2 + ((Y + 0.05) / 0.30)**2 < 1] = 0.45
    arr[((X + 0.05) / 0.15)**2 + ((Y - 0.35) / 0.20)**2 < 1] = 0.7
    arr[(X / 0.08)**2 + ((Y + 0.05) / 0.15)**2 < 1] = 0.05
    return np.clip(arr, 0, 1)


def _make_brain_phantom(H: int, W: int, rng: np.random.RandomState) -> np.ndarray:
    """Brain-like phantom."""
    yy = np.linspace(-1, 1, H)
    xx = np.linspace(-1, 1, W)
    X, Y = np.meshgrid(xx, yy)
    arr = np.zeros((H, W), dtype=np.float64)
    arr[(X**2 + Y**2) < 0.8] = 0.2
    arr[((X**2 + Y**2) < 0.6) & ((X**2 + Y**2) > 0.35)] = 0.8
    arr[(X**2 + Y**2) < 0.35] = 0.4
    arr[((X / 0.08)**2 + ((Y + 0.05) / 0.15)**2) < 1] = 0.05
    arr[((X / 0.08)**2 + ((Y - 0.05) / 0.15)**2) < 1] = 0.05
    return np.clip(arr, 0, 1)


def _make_particle_phantom(H: int, W: int, rng: np.random.RandomState) -> np.ndarray:
    """Nanoparticle phantom for electron microscopy."""
    arr = 0.1 + 0.02 * rng.randn(H, W)
    yy, xx = np.ogrid[:H, :W]
    for _ in range(40):
        cx, cy = rng.randint(5, W - 5), rng.randint(5, H - 5)
        r = rng.uniform(2, max(3, min(H, W) / 30))
        intensity = rng.uniform(0.5, 1.0)
        dist = np.sqrt((xx - cx)**2 + (yy - cy)**2).astype(np.float64)
        arr += intensity * np.exp(-dist**2 / (2 * (r / 2.5)**2))
    return np.clip(arr / max(arr.max(), 1e-8), 0, 1)


def _make_test_scene(H: int, W: int, rng: np.random.RandomState) -> np.ndarray:
    """High-contrast test scene."""
    arr = np.zeros((H, W), dtype=np.float64)
    qh, qw = H // 2, W // 2
    block = max(4, min(16, H // 8))
    for i in range(qh // block):
        for j in range(qw // block):
            if (i + j) % 2 == 0:
                arr[i * block:(i + 1) * block, j * block:(j + 1) * block] = 0.9
    arr[:qh, qw:] = np.linspace(0, 1, W - qw)[np.newaxis, :]
    for _ in range(10):
        cx, cy = rng.randint(0, W), rng.randint(0, H)
        r = rng.randint(5, max(6, min(H, W) // 8))
        yy, xx = np.ogrid[:H, :W]
        mask = (xx - cx)**2 + (yy - cy)**2 < r**2
        arr[mask] = rng.uniform(0.3, 1.0)
    return np.clip(arr, 0, 1)


def _make_surface(H: int, W: int, rng: np.random.RandomState) -> np.ndarray:
    """Fractal surface topography."""
    freq_x = np.fft.fftfreq(W)
    freq_y = np.fft.fftfreq(H)
    FX, FY = np.meshgrid(freq_x, freq_y)
    radius = np.sqrt(FX**2 + FY**2)
    radius[0, 0] = 1.0
    power = 1.0 / (radius ** 2)
    phase = rng.uniform(0, 2 * np.pi, (H, W))
    fft_data = np.sqrt(power) * np.exp(1j * phase)
    surface = np.real(np.fft.ifft2(fft_data))
    surface[:, W // 2:] += 0.3
    smin, smax = surface.min(), surface.max()
    if smax - smin > 1e-8:
        surface = (surface - smin) / (smax - smin)
    return surface.astype(np.float64)


def _make_dexa_phantom(H: int, W: int, rng: np.random.RandomState) -> np.ndarray:
    """DEXA phantom: bone mineral density + soft tissue thickness maps.

    Returns shape (H, W, 2) where channel 0 = bone, channel 1 = soft tissue.
    Simulates a lumbar spine / hip DEXA scan geometry with:
    - Vertebral bodies or femoral head as bone structures
    - Surrounding soft tissue of varying thickness
    """
    yy = np.linspace(-1, 1, H)
    xx = np.linspace(-1, 1, W)
    X, Y = np.meshgrid(xx, yy)

    bone = np.zeros((H, W), dtype=np.float64)
    tissue = np.ones((H, W), dtype=np.float64) * 0.3  # uniform soft tissue background

    # Simulate spine-like bone structures (vertebral bodies)
    n_vertebrae = rng.randint(3, 6)
    y_positions = np.linspace(-0.6, 0.6, n_vertebrae)
    for yp in y_positions:
        # Vertebral body: rectangular with rounded edges
        vw = rng.uniform(0.15, 0.25)  # width
        vh = rng.uniform(0.08, 0.15)  # height
        density = rng.uniform(0.5, 1.0)  # bone density
        x_off = rng.uniform(-0.05, 0.05)  # slight lateral offset
        mask = (np.abs(X - x_off) < vw) & (np.abs(Y - yp) < vh)
        bone[mask] = density

        # Spinous process (posterior element)
        sp_mask = (np.abs(X - x_off - 0.3) < 0.04) & (np.abs(Y - yp) < vh * 0.6)
        bone[sp_mask] = density * 0.7

    # Add pelvis/hip bone region
    pelvis_y = rng.uniform(0.5, 0.7)
    for side in [-1, 1]:
        px = side * rng.uniform(0.25, 0.4)
        pr = rng.uniform(0.12, 0.2)
        dist = np.sqrt((X - px)**2 + (Y - pelvis_y)**2)
        pelvis_mask = dist < pr
        bone[pelvis_mask] = rng.uniform(0.6, 0.9)
        # Femoral head (small dense circle)
        fh_dist = np.sqrt((X - px)**2 + (Y - pelvis_y - pr * 0.8)**2)
        fh_mask = fh_dist < pr * 0.35
        bone[fh_mask] = rng.uniform(0.8, 1.0)

    # Soft tissue varies spatially (thicker around abdomen)
    tissue += 0.3 * np.exp(-X**2 / 0.5) * np.exp(-(Y - 0.1)**2 / 0.8)
    tissue += 0.05 * rng.randn(H, W)
    tissue = np.clip(tissue, 0.05, 1.0)

    # Where bone is present, soft tissue is partially displaced
    tissue = tissue * (1.0 - 0.5 * bone)

    bone = np.clip(bone, 0, 1)
    tissue = np.clip(tissue, 0, 1)

    return np.stack([bone, tissue], axis=-1)


def _make_cell_phantom(H: int, W: int, rng: np.random.RandomState) -> np.ndarray:
    """Fluorescence microscopy cell phantom."""
    arr = np.zeros((H, W), dtype=np.float64)
    n_cells = rng.randint(5, 11)
    for _ in range(n_cells):
        cx, cy = rng.randint(20, W - 20), rng.randint(20, H - 20)
        rx, ry = rng.randint(10, max(11, W // 6)), rng.randint(10, max(11, H // 6))
        intensity = rng.uniform(0.4, 1.0)
        yy, xx = np.ogrid[:H, :W]
        mask = ((xx - cx) / max(rx, 1))**2 + ((yy - cy) / max(ry, 1))**2 < 1
        arr = np.maximum(arr, intensity * mask.astype(np.float64))
    return arr


# ══════════════════════════════════════════════════════════════════════════════
#  Per-tier ground truth resolution
# ══════════════════════════════════════════════════════════════════════════════

# Tier seed offsets ensure different phantom realizations per tier
_TIER_SEED_OFFSETS: dict[str, int] = {
    "public": 0,
    "dev": 10000,
    "hidden": 20000,
}


def _load_scenes_from_directory(
    dir_path: Path,
    fmt: str,
    signal_shape: tuple[int, ...],
    max_scenes: int,
    seed: int = 42,
) -> list[np.ndarray]:
    """Load ground-truth scenes from a local directory.

    Supports .mat, .tif, .png, .jpg formats. Returns a list of arrays
    cropped/resized to signal_shape.
    """
    ext_map = {
        "mat": ["*.mat"],
        "tif": ["*.tif", "*.tiff"],
        "png": ["*.png"],
        "jpg": ["*.jpg", "*.jpeg"],
    }
    patterns = ext_map.get(fmt, [f"*.{fmt}"])

    files = []
    for pat in patterns:
        files.extend(sorted(dir_path.glob(pat)))
    if not files:
        logger.warning("No %s files found in %s", fmt, dir_path)
        return []

    # Deterministic shuffle to avoid always using the same subset
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(files))[:max_scenes]

    scenes = []
    for idx in sorted(indices):
        fpath = files[idx]
        try:
            if fmt == "mat":
                arr = _load_mat_scene(fpath)
            else:
                arr = _load_tif_image(fpath, target_size=None)
            # Normalize to [0, 1]
            if arr.max() > 1.0:
                arr = arr / max(arr.max(), 1e-8)
            arr = np.clip(arr, 0, 1).astype(np.float64)
            arr = _crop_or_resize_2d(arr, signal_shape)
            scenes.append(arr)
        except Exception as e:
            logger.warning("Failed to load %s: %s", fpath, e)
            continue

    return scenes


def _load_scenes_from_registry(
    registry_id: str,
    signal_shape: tuple[int, ...],
    max_scenes: int,
    seed: int = 42,
    data_root: Path | None = None,
) -> list[np.ndarray]:
    """Load ground-truth scenes from a dataset registry entry.

    Downloads if needed, then extracts and crops scenes.
    """
    try:
        from benchmarks.datasets.registry import DATASET_REGISTRY
        from benchmarks.datasets.downloaders import (
            download_file, convert_mat, CACHE_ROOT,
        )
    except ImportError:
        logger.warning("Cannot import registry/downloaders for %s", registry_id)
        return []

    entry = DATASET_REGISTRY.get(registry_id)
    if entry is None:
        logger.warning("Registry entry not found: %s", registry_id)
        return []

    # For generated entries, use the generator directly
    if entry.source_type == "generated":
        return _load_scenes_from_generator(
            entry.converter, signal_shape, max_scenes, seed,
        )

    # For web entries, try to use cached data or download
    cache_dir = CACHE_ROOT / registry_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    npy_path = cache_dir / "data.npy"

    if npy_path.exists():
        data = np.load(str(npy_path))
    else:
        # Try downloading
        try:
            ext = entry.format
            if ext == "zip":
                ext = "zip"
            dl_path = cache_dir / f"download.{ext}"
            download_file(entry.url, dl_path)

            # Convert based on format
            if entry.format == "mat":
                data = _load_mat_scene(dl_path, key=entry.mat_key)
            elif entry.format == "mat_v73":
                import h5py as h5
                with h5.File(dl_path, "r") as hf:
                    key = entry.mat_key or list(hf.keys())[0]
                    data = np.array(hf[key], dtype=np.float64)
            elif entry.format == "hdf5":
                import h5py as h5
                with h5.File(dl_path, "r") as hf:
                    key = entry.mat_key or "data"
                    if key in hf:
                        data = np.array(hf[key], dtype=np.float64)
                    else:
                        # Take first dataset
                        first_key = list(hf.keys())[0]
                        data = np.array(hf[first_key], dtype=np.float64)
            elif entry.format == "zip":
                # Extract images from ZIP and load them
                import zipfile
                from PIL import Image
                extract_dir = cache_dir / "extracted"
                extract_dir.mkdir(exist_ok=True)
                with zipfile.ZipFile(dl_path, "r") as zf:
                    zf.extractall(extract_dir)
                # Find all image files
                img_files = []
                for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
                    img_files.extend(sorted(extract_dir.rglob(ext)))
                if not img_files:
                    logger.warning("No images found in ZIP for %s", registry_id)
                    return []
                # Load first image as representative data
                rng_z = np.random.RandomState(seed)
                chosen = rng_z.permutation(len(img_files))[:max(max_scenes, 1)]
                all_imgs = []
                for ci in sorted(chosen):
                    img = Image.open(img_files[ci]).convert("L")
                    arr = np.array(img, dtype=np.float64) / 255.0
                    all_imgs.append(arr)
                if all_imgs:
                    # Stack into 3D array (scenes along last axis)
                    target = tuple(signal_shape[:2])
                    resized = []
                    for im in all_imgs:
                        im = _crop_or_resize_2d(im, signal_shape)
                        resized.append(im)
                    # Return scenes directly instead of going through numpy cache
                    return resized[:max_scenes]
                return []
            else:
                logger.warning("Unsupported format %s for registry %s", entry.format, registry_id)
                return []

            # Save cache
            np.save(str(npy_path), data)
        except Exception as e:
            logger.warning("Failed to download/convert %s: %s", registry_id, e)
            return []

    # Normalize
    if data.max() > 1.0:
        data = data / max(data.max(), 1e-8)
    data = np.clip(data, 0, 1).astype(np.float64)

    # Extract scenes by slicing
    scenes = []
    rng = np.random.RandomState(seed)
    if data.ndim == 3 and len(signal_shape) >= 2:
        # 3D cube: extract 2D slices or spectral crops
        n_slices = data.shape[2] if data.ndim == 3 else 1
        if len(signal_shape) == 3 and signal_shape[2] <= n_slices:
            # Need spectral subcubes
            n_bands = signal_shape[2]
            for _ in range(max_scenes):
                start = rng.randint(0, max(1, n_slices - n_bands))
                subcube = data[:, :, start:start + n_bands]
                subcube = _crop_or_resize_2d(subcube, signal_shape)
                scenes.append(subcube)
        else:
            # Extract 2D slices
            indices = rng.permutation(n_slices)[:max_scenes]
            for idx in sorted(indices):
                sl = data[:, :, idx]
                sl = _crop_or_resize_2d(sl, signal_shape)
                scenes.append(sl)
    elif data.ndim == 2:
        # Single 2D image: augment to create multiple scenes
        data = _crop_or_resize_2d(data, signal_shape)
        for i in range(max_scenes):
            scenes.append(_augment_scene(data, seed + i))
    else:
        data = _crop_or_resize_2d(data, signal_shape)
        scenes.append(data)

    return scenes[:max_scenes]


def _load_synthetic_source_pool(data_root: Path | None = None) -> list[np.ndarray]:
    """Load multi-domain source pool for synthetic scene generation.

    Sources: BSDS400, BrainImages, TSA hyperspectral, CACTI video, real MRI.
    """
    if data_root is None:
        data_root = _find_data_root()
    datasets_root = data_root

    from scripts.generate_synthetic_scenes import load_all_sources
    return load_all_sources(datasets_root)


# Cached source pool (loaded once per process)
_SYNTHETIC_SOURCE_POOL: list[np.ndarray] | None = None


def _get_synthetic_source_pool(data_root: Path | None = None) -> list[np.ndarray]:
    global _SYNTHETIC_SOURCE_POOL
    if _SYNTHETIC_SOURCE_POOL is None:
        _SYNTHETIC_SOURCE_POOL = _load_synthetic_source_pool(data_root)
        logger.info("Loaded %d source images for synthetic generation",
                     len(_SYNTHETIC_SOURCE_POOL))
    return _SYNTHETIC_SOURCE_POOL


def _load_scenes_from_generator(
    generator_name: str,
    signal_shape: tuple[int, ...],
    max_scenes: int,
    seed: int,
    data_root: Path | None = None,
) -> list[np.ndarray]:
    """Generate scenes using a named generator function."""

    # Special case: complex synthetic scene generator
    if generator_name == "generate_synthetic_scene":
        from scripts.generate_synthetic_scenes import generate_synthetic_scene
        source_pool = _get_synthetic_source_pool(data_root)
        if len(source_pool) < 5:
            logger.warning("Synthetic generator needs >=5 source images, got %d",
                           len(source_pool))
            return []
        target_size = signal_shape[0]
        scenes = []
        for i in range(max_scenes):
            scene_seed = seed + i * 137  # Match the CLI script's seed spacing
            arr = generate_synthetic_scene(source_pool, target_size, scene_seed)
            arr = arr.astype(np.float64)
            arr = _crop_or_resize_2d(arr, signal_shape)
            scenes.append(np.clip(arr, 0, 1))
        return scenes

    try:
        from benchmarks.datasets.downloaders import (
            generate_medical_phantom, generate_em_phantom, generate_surface,
            generate_oct_phantom, generate_smlm_phantom, generate_depth_map,
            generate_test_scene, generate_star_field, generate_resolution_target,
            generate_diffraction_pattern, generate_elemental_map,
            generate_ndt_phantom, generate_velocity_model,
            generate_ae_source_map, generate_sam_phantom,
            generate_thermography_phantom, generate_ao_wavefront, generate_afm_surface,
            generate_angiography_vessel_phantom, generate_asl_perfusion_phantom,
            generate_apt_composition_map, generate_blt_source_phantom,
            generate_brachytherapy_seed_phantom,
            generate_brillouin_vipa_phantom,
            generate_cars_raman_phantom,
            generate_cathodoluminescence_phantom,
            generate_cbct_head_phantom,
            generate_cest_mri_phantom,
            generate_ceus_phantom,
            generate_clem_phantom,
            generate_coded_exposure_phantom,
            generate_confocal_3d_phantom,
            generate_confocal_endomicroscopy_phantom,
            generate_confocal_livecell_phantom,
            generate_coronagraphy_phantom,
            generate_cryo_em_phantom,
            generate_cryo_et_phantom,
            generate_ct_phantom,
            generate_ct_fluorescence_phantom,
            generate_cup_phantom,
            generate_dark_field_phantom,
            generate_dexa_phantom,
            generate_desi_phantom,
            generate_dic_phantom,
            generate_diffusion_mri_phantom,
            generate_digital_breast_tomo_phantom,
            generate_dna_paint_phantom,
            generate_doppler_ultrasound_phantom,
            generate_dot_phantom,
            generate_ebsd_phantom,
            generate_eddy_current_phantom,
            generate_edx_mapping_phantom,
            generate_eels_phantom,
            generate_eht_imaging_phantom,
            generate_elastography_phantom,
            generate_electron_diffraction_phantom,
            generate_electron_holography_phantom,
            generate_electron_tomography_phantom,
        )
    except ImportError:
        return []

    gen_map = {
        "generate_medical_phantom": generate_medical_phantom,
        "generate_em_phantom": generate_em_phantom,
        "generate_surface": generate_surface,
        "generate_oct_phantom": generate_oct_phantom,
        "generate_smlm_phantom": generate_smlm_phantom,
        "generate_depth_map": generate_depth_map,
        "generate_test_scene": generate_test_scene,
        "generate_star_field": generate_star_field,
        "generate_resolution_target": generate_resolution_target,
        "generate_diffraction_pattern": generate_diffraction_pattern,
        "generate_elemental_map": generate_elemental_map,
        "generate_ndt_phantom": generate_ndt_phantom,
        "generate_velocity_model": generate_velocity_model,
        "generate_ae_source_map": generate_ae_source_map,
        "generate_sam_phantom": generate_sam_phantom,
        "generate_thermography_phantom": generate_thermography_phantom,
        "generate_ao_wavefront": generate_ao_wavefront,
        "generate_afm_surface": generate_afm_surface,
        "generate_angiography_vessel_phantom": generate_angiography_vessel_phantom,
        "generate_asl_perfusion_phantom": generate_asl_perfusion_phantom,
        "generate_apt_composition_map": generate_apt_composition_map,
        "generate_blt_source_phantom": generate_blt_source_phantom,
        "generate_brachytherapy_seed_phantom": generate_brachytherapy_seed_phantom,
        "generate_brillouin_vipa_phantom": generate_brillouin_vipa_phantom,
        "generate_cars_raman_phantom": generate_cars_raman_phantom,
        "generate_cathodoluminescence_phantom": generate_cathodoluminescence_phantom,
        "generate_cbct_head_phantom": generate_cbct_head_phantom,
        "generate_cest_mri_phantom": generate_cest_mri_phantom,
        "generate_ceus_phantom": generate_ceus_phantom,
        "generate_clem_phantom": generate_clem_phantom,
        "generate_coded_exposure_phantom": generate_coded_exposure_phantom,
        "generate_confocal_3d_phantom": generate_confocal_3d_phantom,
        "generate_confocal_endomicroscopy_phantom": generate_confocal_endomicroscopy_phantom,
        "generate_confocal_livecell_phantom": generate_confocal_livecell_phantom,
        "generate_coronagraphy_phantom": generate_coronagraphy_phantom,
        "generate_cryo_em_phantom": generate_cryo_em_phantom,
        "generate_cryo_et_phantom": generate_cryo_et_phantom,
        "generate_ct_phantom": generate_ct_phantom,
        "generate_ct_fluorescence_phantom": generate_ct_fluorescence_phantom,
        "generate_cup_phantom": generate_cup_phantom,
        "generate_dark_field_phantom": generate_dark_field_phantom,
        "generate_dexa_phantom": generate_dexa_phantom,
        "generate_desi_phantom": generate_desi_phantom,
        "generate_dic_phantom": generate_dic_phantom,
        "generate_diffusion_mri_phantom": generate_diffusion_mri_phantom,
        "generate_digital_breast_tomo_phantom": generate_digital_breast_tomo_phantom,
        "generate_dna_paint_phantom": generate_dna_paint_phantom,
        "generate_doppler_ultrasound_phantom": generate_doppler_ultrasound_phantom,
        "generate_dot_phantom": generate_dot_phantom,
        "generate_ebsd_phantom": generate_ebsd_phantom,
        "generate_eddy_current_phantom": generate_eddy_current_phantom,
        "generate_edx_mapping_phantom": generate_edx_mapping_phantom,
        "generate_eels_phantom": generate_eels_phantom,
        "generate_eht_imaging_phantom": generate_eht_imaging_phantom,
        "generate_elastography_phantom": generate_elastography_phantom,
        "generate_electron_diffraction_phantom": generate_electron_diffraction_phantom,
        "generate_electron_holography_phantom": generate_electron_holography_phantom,
        "generate_electron_tomography_phantom": generate_electron_tomography_phantom,
    }

    gen_fn = gen_map.get(generator_name)
    if gen_fn is None:
        logger.warning("Unknown generator: %s", generator_name)
        return []

    target = tuple(signal_shape[:2])
    scenes = []
    for i in range(max_scenes):
        result = gen_fn(target_shape=target, seed=seed + i)
        # Some generators return list[dict]; extract x_true from first sample
        if isinstance(result, list):
            arr = result[0]["x_true"] if result else np.zeros(target, dtype=np.float32)
        else:
            arr = result
        arr = arr.astype(np.float64)
        arr = _crop_or_resize_2d(arr, signal_shape)
        if arr.max() > 0:
            arr = arr / max(arr.max(), 1e-8)
        scenes.append(np.clip(arr, 0, 1))

    return scenes


def _augment_scene(x: np.ndarray, seed: int) -> np.ndarray:
    """Augment a scene with deterministic flip/rotate/crop to create variety."""
    rng = np.random.RandomState(seed)
    out = x.copy()

    # Random flip
    if rng.rand() > 0.5:
        out = np.flip(out, axis=0).copy()
    if rng.rand() > 0.5:
        out = np.flip(out, axis=1).copy()

    # Random 90-degree rotation
    k = rng.randint(0, 4)
    if k > 0:
        if out.ndim == 2:
            out = np.rot90(out, k=k).copy()
        else:
            out = np.rot90(out, k=k, axes=(0, 1)).copy()

    return out


def _select_bands(cube: np.ndarray, n_bands: int, seed: int) -> np.ndarray:
    """Select n_bands spectral bands from a hyperspectral cube."""
    if cube.ndim != 3 or cube.shape[2] <= n_bands:
        return cube
    rng = np.random.RandomState(seed)
    total = cube.shape[2]
    # Evenly spaced with random offset
    offset = rng.randint(0, max(1, total - n_bands))
    step = max(1, (total - offset) // n_bands)
    indices = list(range(offset, min(total, offset + step * n_bands), step))[:n_bands]
    return cube[:, :, indices]


def _resolve_tier_ground_truth(
    variant_key: str,
    tier_name: str,
    signal_shape: tuple[int, ...],
    scene_index: int,
    seed: int,
    tier_data_source: dict | None = None,
    data_root: Path | None = None,
) -> np.ndarray:
    """Resolve a ground-truth signal for a specific tier.

    Resolution chain:
    1. tier_data_source with "path" → load from local directory
    2. tier_data_source with "registry_id" → load from web/downloaded dataset
    3. tier_data_source with "generator" → call named generator with seed offset
    4. Fallback: call _resolve_ground_truth() with tier-offset seed

    Each tier gets different data, preventing memorization attacks.
    """
    tier_offset = _TIER_SEED_OFFSETS.get(tier_name, 0)
    effective_seed = seed + tier_offset

    if tier_data_source is not None:
        # 1. Local path
        if "path" in tier_data_source:
            if data_root is None:
                data_root = _find_data_root()
            dir_path = data_root.parent / tier_data_source["path"]
            if not dir_path.exists():
                # Try relative to project root
                dir_path = _find_data_root().parent / tier_data_source["path"]
            fmt = tier_data_source.get("format", "mat")
            scenes = _load_scenes_from_directory(
                dir_path, fmt, signal_shape,
                max_scenes=scene_index + 1,
                seed=effective_seed,
            )
            if scenes and scene_index < len(scenes):
                return scenes[scene_index]

        # 2. Registry ID
        if "registry_id" in tier_data_source:
            scenes = _load_scenes_from_registry(
                tier_data_source["registry_id"],
                signal_shape,
                max_scenes=scene_index + 1,
                seed=effective_seed,
                data_root=data_root,
            )
            if scenes and scene_index < len(scenes):
                return scenes[scene_index]

        # 3. Named generator
        if "generator" in tier_data_source:
            gen_seed_offset = tier_data_source.get("seed_offset", tier_offset)
            gen_seed = seed + gen_seed_offset + scene_index
            scenes = _load_scenes_from_generator(
                tier_data_source["generator"],
                signal_shape,
                max_scenes=1,
                seed=gen_seed,
            )
            if scenes:
                return scenes[0]

    # 4. Fallback: original resolver with tier-offset seed
    return _resolve_ground_truth(
        variant_key, signal_shape,
        seed=effective_seed + scene_index,
        data_root=data_root,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Generic forward models (7 runner types)
# ══════════════════════════════════════════════════════════════════════════════


def _forward_radon(x: np.ndarray, n_angles: int = 180) -> tuple[np.ndarray, np.ndarray]:
    """Radon transform (sinogram). Returns (sinogram, angles_array)."""
    H, W = x.shape[:2]
    # If 3D, take the central slice for the forward model
    if x.ndim == 3:
        x_2d = x[:, :, x.shape[2] // 2]
    else:
        x_2d = x

    angles = np.linspace(0, 180, n_angles, endpoint=False)
    sinogram = np.zeros((n_angles, max(H, W)), dtype=np.float64)
    center = np.array([H / 2.0, W / 2.0])

    for i, angle in enumerate(angles):
        theta = np.deg2rad(angle)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        # Project each row/col
        for r in range(max(H, W)):
            s = r - max(H, W) / 2.0
            total = 0.0
            count = 0
            for t_idx in range(max(H, W)):
                t = t_idx - max(H, W) / 2.0
                yi = int(center[0] + s * cos_t - t * sin_t)
                xi = int(center[1] + s * sin_t + t * cos_t)
                if 0 <= yi < H and 0 <= xi < W:
                    total += x_2d[yi, xi]
                    count += 1
            sinogram[i, r] = total

    return sinogram, angles


def _forward_radon_fast(x: np.ndarray, n_angles: int = 180) -> tuple[np.ndarray, np.ndarray]:
    """Fast vectorized Radon transform. Returns (sinogram, angles_array)."""
    if x.ndim == 3:
        x_2d = x[:, :, x.shape[2] // 2]
    else:
        x_2d = x.copy()

    H, W = x_2d.shape
    diag = int(np.ceil(np.sqrt(H**2 + W**2)))
    angles = np.linspace(0, 180, n_angles, endpoint=False)

    # Pad to square diagonal
    pad_h = (diag - H) // 2
    pad_w = (diag - W) // 2
    padded = np.pad(x_2d, ((pad_h, diag - H - pad_h), (pad_w, diag - W - pad_w)))

    sinogram = np.zeros((n_angles, diag), dtype=np.float64)
    from scipy.ndimage import rotate as ndrotate
    for i, angle in enumerate(angles):
        rotated = ndrotate(padded, -angle, reshape=False, order=1, mode="constant")
        sinogram[i] = rotated.sum(axis=0)

    return sinogram, angles


def _forward_kspace(x: np.ndarray, acceleration: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """k-space undersampling. Returns (undersampled_kspace_magnitude, mask)."""
    if x.ndim == 3:
        x_2d = x[:, :, x.shape[2] // 2]
    else:
        x_2d = x.copy()

    H, W = x_2d.shape
    kspace = np.fft.fftshift(np.fft.fft2(x_2d))

    # Cartesian undersampling mask with ACS lines
    mask = np.zeros((H, W), dtype=np.float64)
    acs_lines = max(4, H // 16)
    center = H // 2
    mask[center - acs_lines // 2:center + acs_lines // 2, :] = 1.0
    # Uniform undersampling
    step = max(1, acceleration)
    mask[::step, :] = 1.0

    undersampled = kspace * mask
    y = np.log1p(np.abs(undersampled))
    return y, mask


def _forward_psf(x: np.ndarray, sigma: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """PSF blur convolution. Returns (blurred, psf_kernel)."""
    from scipy.signal import fftconvolve

    if x.ndim == 3:
        x_2d = x[:, :, x.shape[2] // 2]
    else:
        x_2d = x.copy()

    ksize = max(3, int(6 * sigma) | 1)  # ensure odd
    half = ksize // 2
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    psf /= psf.sum()

    blurred = fftconvolve(x_2d, psf, mode="same")
    return np.clip(blurred, 0, None), psf


def _forward_ctf(x: np.ndarray, defocus_nm: float = 1000.0) -> tuple[np.ndarray, np.ndarray]:
    """Contrast Transfer Function modulation. Returns (modulated, ctf_params)."""
    if x.ndim == 3:
        x_2d = x[:, :, x.shape[2] // 2]
    else:
        x_2d = x.copy()

    H, W = x_2d.shape
    voltage_kv = 300.0
    cs_mm = 2.7
    pixel_nm = 1.0
    amplitude_contrast = 0.07

    # Electron wavelength
    lam_nm = 0.01226 / np.sqrt(voltage_kv + 0.978e-3 * voltage_kv**2)
    cs_nm = cs_mm * 1e6

    freq_x = np.fft.fftfreq(W, d=pixel_nm)
    freq_y = np.fft.fftfreq(H, d=pixel_nm)
    FX, FY = np.meshgrid(freq_x, freq_y)
    s2 = FX**2 + FY**2
    s2[0, 0] = 1e-20

    chi = np.pi * lam_nm * defocus_nm * s2 - 0.5 * np.pi * cs_nm * lam_nm**3 * s2**2
    ctf = -np.sqrt(1 - amplitude_contrast**2) * np.sin(chi) - amplitude_contrast * np.cos(chi)

    ft = np.fft.fft2(x_2d)
    modulated = np.real(np.fft.ifft2(ft * np.fft.ifftshift(ctf)))
    ctf_params = np.array([defocus_nm, cs_mm, voltage_kv, pixel_nm, amplitude_contrast])

    return modulated, ctf_params


def _forward_mask(x: np.ndarray, density: float = 0.5, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Binary mask coding. Returns (coded_measurement, mask)."""
    if x.ndim == 3:
        # For spectral cubes: sum coded bands (CASSI-like)
        H, W, C = x.shape
        rng_m = np.random.default_rng(seed)
        mask = (rng_m.random((H, W)) > (1 - density)).astype(np.float64)
        y = np.zeros((H, W), dtype=np.float64)
        for c in range(C):
            y += mask * x[:, :, c]
        y /= C
        return y, mask
    else:
        H, W = x.shape
        rng_m = np.random.default_rng(seed)
        mask = (rng_m.random((H, W)) > (1 - density)).astype(np.float64)
        y = mask * x
        return y, mask


def _forward_tip(x: np.ndarray, tip_radius: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Scanning probe tip convolution. Returns (convolved, tip_kernel)."""
    from scipy.signal import fftconvolve

    if x.ndim == 3:
        x_2d = x[:, :, x.shape[2] // 2]
    else:
        x_2d = x.copy()

    ksize = 2 * tip_radius + 1
    half = tip_radius
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    r = np.sqrt(xx**2 + yy**2)
    tip = np.zeros_like(r)
    mask = r <= tip_radius
    tip[mask] = r[mask]**2 / (2.0 * tip_radius)
    tip = tip.max() - tip  # invert: center is highest
    tip[~mask] = 0
    tip /= max(tip.sum(), 1e-8)

    convolved = fftconvolve(x_2d, tip, mode="same")
    return convolved, tip


def _forward_dual_energy(
    x: np.ndarray,
    mu_bone: tuple[float, float] = (0.55, 0.30),
    mu_tissue: tuple[float, float] = (0.20, 0.18),
) -> tuple[np.ndarray, np.ndarray]:
    """Dual-energy X-ray projection (DEXA).

    The ground truth x has shape (H, W, 2) where:
      x[:,:,0] = bone thickness map (BMD proxy)
      x[:,:,1] = soft tissue thickness map

    Returns:
      y: (H, W, 2) — log-attenuation images at low and high energy
      H_ideal: (2, 2) — attenuation coefficient matrix
        [[mu_bone_low, mu_tissue_low],
         [mu_bone_high, mu_tissue_high]]

    The forward model: y_e(i,j) = mu_bone(E) * t_bone(i,j) + mu_tissue(E) * t_tissue(i,j)
    """
    if x.ndim == 2:
        # If 2D, treat as bone density only; synthesize soft tissue as complement
        bone = x.copy()
        tissue = 1.0 - 0.3 * bone + 0.1  # soft tissue background
        x = np.stack([bone, tissue], axis=-1)

    H, W = x.shape[:2]
    bone_map = x[:, :, 0]  # bone thickness
    tissue_map = x[:, :, 1]  # soft tissue thickness

    # Attenuation coefficient matrix (2 energies × 2 materials)
    A = np.array([[mu_bone[0], mu_tissue[0]],   # low energy
                  [mu_bone[1], mu_tissue[1]]])   # high energy

    # Log-attenuation at each energy
    y = np.zeros((H, W, 2), dtype=np.float64)
    y[:, :, 0] = A[0, 0] * bone_map + A[0, 1] * tissue_map  # low energy
    y[:, :, 1] = A[1, 0] * bone_map + A[1, 1] * tissue_map  # high energy

    return y, A


def _forward_projection(
    x: np.ndarray,
    scatter_frac: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """2D X-ray projection (radiography, mammography, fluoroscopy).

    Simulates Beer-Lambert attenuation: I = I_0 * exp(-integral(mu * t)).
    For a 2D phantom, the projection is the image itself (no tomographic geometry).

    Returns:
      y: (H, W) — log-attenuation projection image
      H_ideal: (2,) — [I_0, scatter_fraction]
    """
    if x.ndim == 3:
        # If 3D, sum along depth to get projection
        x_proj = x.sum(axis=-1)
        xmax = x_proj.max()
        if xmax > 0:
            x_proj = x_proj / xmax
    else:
        x_proj = x.copy()

    # Log-attenuation: y = -ln(I/I0) = mu * t (proportional to thickness/density)
    y = x_proj.copy()

    H_ideal = np.array([1.0, scatter_frac])
    return y, H_ideal


# ── Forward model dispatch ───────────────────────────────────────────────────


def _apply_forward_model(
    runner_type: str,
    x: np.ndarray,
    rng: np.random.Generator,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply physics forward model and return (measurement, H_ideal)."""
    if runner_type == "radon":
        return _forward_radon_fast(x, n_angles=180)
    elif runner_type == "kspace":
        return _forward_kspace(x, acceleration=4)
    elif runner_type == "psf":
        return _forward_psf(x, sigma=2.0)
    elif runner_type == "ctf":
        return _forward_ctf(x, defocus_nm=1000.0)
    elif runner_type == "mask":
        return _forward_mask(x, density=0.5, seed=seed)
    elif runner_type == "tip":
        return _forward_tip(x, tip_radius=5)
    elif runner_type == "dual_energy":
        return _forward_dual_energy(x)
    elif runner_type == "projection":
        return _forward_projection(x)
    elif runner_type == "identity":
        # Identity forward model: y = x + small Gaussian noise, H_ideal = I
        H_size = min(x.size, 2048)
        H_ideal = np.eye(H_size, dtype=np.float32)
        noise = rng.standard_normal(x.shape).astype(np.float32) * 0.01
        return (x + noise).astype(np.float32), H_ideal
    else:
        # Default: PSF-based
        return _forward_psf(x, sigma=2.0)


# ── Generic mismatch application ────────────────────────────────────────────


def _apply_generic_mismatch(
    y: np.ndarray,
    H_ideal: np.ndarray,
    true_spec: dict,
    runner_type: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply mismatch perturbations based on true_spec parameter names.

    Classifies each mismatch param by name pattern and applies the
    appropriate perturbation to the measurement.
    """
    from scipy.ndimage import shift as ndshift, gaussian_filter

    y_out = y.copy()

    for name, value in true_spec.items():
        name_lower = name.lower()

        # Shift-type params: translate the measurement
        if any(k in name_lower for k in ("dx", "dy", "shift", "offset", "displacement", "cor_shift")):
            if "dx" in name_lower or "x_shift" in name_lower or "lateral" in name_lower:
                if y_out.ndim >= 2:
                    y_out = ndshift(y_out, [0, float(value)], order=1, mode="nearest")
            elif "dy" in name_lower or "y_shift" in name_lower:
                if y_out.ndim >= 2:
                    y_out = ndshift(y_out, [float(value), 0], order=1, mode="nearest")
            elif "offset" in name_lower:
                y_out = y_out + float(value)

        # Rotation-type params
        elif any(k in name_lower for k in ("rotation", "theta", "angle", "tilt")):
            if y_out.ndim == 2 and abs(float(value)) > 1e-6:
                from scipy.ndimage import rotate as ndrotate
                y_out = ndrotate(y_out, float(value), reshape=False, order=1, mode="nearest")

        # Blur-type params
        elif any(k in name_lower for k in ("blur", "sigma", "psf", "fwhm", "defocus")):
            if y_out.ndim >= 2 and abs(float(value)) > 1e-6:
                y_out = gaussian_filter(y_out, sigma=abs(float(value)))

        # Gain/scale-type params
        elif any(k in name_lower for k in ("gain", "scale", "amplitude", "intensity", "alpha")):
            y_out = y_out * float(value)

        # Phase-type params (for k-space/SAR)
        elif any(k in name_lower for k in ("phase", "dispersion")):
            # Add phase perturbation as multiplicative modulation
            if y_out.ndim == 2:
                H_y, W_y = y_out.shape
                phase_mod = 1.0 + float(value) * np.sin(
                    np.linspace(0, 2 * np.pi, W_y)[np.newaxis, :] *
                    np.linspace(0, 2 * np.pi, H_y)[:, np.newaxis]
                )
                y_out = y_out * phase_mod

        # Decay-type params
        elif any(k in name_lower for k in ("decay", "attenuation", "damping")):
            if y_out.ndim == 2:
                decay = np.exp(-abs(float(value)) * np.arange(y_out.shape[-1]))[np.newaxis, :]
                y_out = y_out * decay

        # Noise-type params (additional noise floor)
        elif any(k in name_lower for k in ("noise", "dark_current", "readout")):
            y_out = y_out + rng.normal(0, abs(float(value)), y_out.shape)

        # Drift-type params
        elif any(k in name_lower for k in ("drift", "jitter", "vibration")):
            if y_out.ndim >= 2:
                y_out = ndshift(y_out, [float(value) * 0.5] * min(y_out.ndim, 2),
                                order=1, mode="nearest")

    return y_out


# ── Mismatch application ─────────────────────────────────────────────────────


def _apply_cassi_mismatch(
    x: np.ndarray, mask: np.ndarray, true_spec: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Apply CASSI mismatch: subpixel mask shift, rotation, modified dispersion.

    Returns (y_mismatch, H_ideal_mask).
    """
    from scipy.ndimage import shift, rotate

    H, W, L = x.shape

    # Generate ideal coded aperture mask if not provided
    if mask is None:
        rng = np.random.default_rng(42)
        mask = (rng.random((H, W)) > 0.5).astype(np.float64)

    # Ideal measurement (no mismatch)
    # CASSI: y = sum_l mask * shift(x_l, l*dispersion)
    ideal_dispersion = 2.0  # nominal slope

    # Apply mismatch to mask: subpixel shift + rotation
    dx = true_spec["mask_dx"]
    dy = true_spec["mask_dy"]
    rot = true_spec["mask_rotation"]
    mismatch_mask = shift(mask, [dy, dx], order=1, mode="constant")
    if abs(rot) > 1e-6:
        mismatch_mask = rotate(mismatch_mask, rot, reshape=False, order=1, mode="constant")

    # Apply mismatch dispersion
    mismatch_slope = true_spec["dispersion_slope"]
    mismatch_axis = true_spec["dispersion_axis"]

    # Generate mismatched measurement
    y = np.zeros((H, W + (L - 1) * int(np.ceil(mismatch_slope))), dtype=np.float64)
    for l in range(L):
        disp = mismatch_slope * l
        disp_int = int(np.floor(disp))
        disp_frac = disp - disp_int
        coded = mismatch_mask * x[:, :, l]
        # Sub-pixel dispersion via linear interpolation
        y[:, disp_int:disp_int + W] += coded * (1 - disp_frac)
        if disp_frac > 0:
            y[:, disp_int + 1:disp_int + 1 + W] += coded * disp_frac

    return y, mask


def _apply_cacti_mismatch(
    x: np.ndarray, mask: np.ndarray, true_spec: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Apply CACTI mismatch: mask shift/rotation/blur, temporal clock offset, gain/offset drift.

    Returns (y_mismatch, H_ideal_mask).
    """
    from scipy.ndimage import shift, rotate, gaussian_filter

    H, W, T = x.shape

    # Generate ideal temporal mask if not provided
    if mask is None:
        rng = np.random.default_rng(42)
        mask = (rng.random((H, W, T)) > 0.5).astype(np.float64)

    # Apply mask mismatch
    mismatch_mask = mask.copy()
    dx = true_spec["mask_dx"]
    dy = true_spec["mask_dy"]
    rot = true_spec["mask_rotation"]
    blur = true_spec["mask_blur"]

    for t in range(T):
        frame = mismatch_mask[:, :, t]
        frame = shift(frame, [dy, dx], order=1, mode="constant")
        if abs(rot) > 1e-6:
            frame = rotate(frame, rot, reshape=False, order=1, mode="constant")
        if blur > 0:
            frame = gaussian_filter(frame, sigma=blur)
        mismatch_mask[:, :, t] = frame

    # Generate measurement: y = sum_t mask_t * x_t + gain/offset drift
    gain = true_spec["gain_drift"]
    offset = true_spec["offset_drift"]
    y = np.zeros((H, W), dtype=np.float64)
    for t in range(T):
        y += mismatch_mask[:, :, t] * x[:, :, t]
    y = gain * y + offset

    return y, mask


def _apply_spc_mismatch_block(
    x: np.ndarray, phi: np.ndarray, true_spec: dict, block_size: int = 33
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply SPC mismatch on block-by-block basis with gain decay + noise.

    The image is split into non-overlapping blocks of block_size x block_size.
    Each block is measured independently with the same phi matrix.

    Returns (y_all_blocks, H_ideal_matrix, x_blocks_flat).
    """
    H, W = x.shape
    n = block_size * block_size

    # Generate sensing matrix if not provided
    if phi is None:
        rng = np.random.default_rng(42)
        m = n // 4  # 25% compression ratio (272 measurements for 1089 pixels)
        phi = rng.standard_normal((m, n)).astype(np.float64)
        phi /= np.linalg.norm(phi, axis=1, keepdims=True)

    m = phi.shape[0]

    # Process each block
    n_blocks_h = H // block_size
    n_blocks_w = W // block_size
    total_blocks = n_blocks_h * n_blocks_w

    y_all = np.zeros((total_blocks, m), dtype=np.float64)
    alpha = true_spec["gain_decay_alpha"]
    decay = np.exp(-alpha * np.arange(m))

    for bh in range(n_blocks_h):
        for bw in range(n_blocks_w):
            b_idx = bh * n_blocks_w + bw
            block = x[bh*block_size:(bh+1)*block_size, bw*block_size:(bw+1)*block_size]
            x_flat = block.flatten()
            y_ideal = phi @ x_flat
            y_all[b_idx] = decay * y_ideal

    return y_all, phi


# ── Noise application ─────────────────────────────────────────────────────────


def _add_noise(y: np.ndarray, noise_model: str, noise_params: dict, rng: np.random.Generator) -> np.ndarray:
    """Add noise to measurements.

    Supported noise models:
      - poisson_gaussian: Combined Poisson shot noise + Gaussian read noise
      - gaussian: Additive white Gaussian noise
      - poisson: Pure Poisson shot noise (medical, EM, astronomy)
      - speckle: Rayleigh-distributed multiplicative noise (SAR, ultrasound)
    """
    if noise_model == "poisson_gaussian":
        alpha = noise_params.get("poisson_alpha", 1.0)
        sigma = noise_params.get("gaussian_sigma", 0.01)
        # Poisson component (scaled)
        y_pos = np.maximum(y, 0)
        if alpha > 0 and y_pos.max() > 0:
            y_noisy = rng.poisson(np.maximum(y_pos / alpha, 0.001)).astype(np.float64) * alpha
        else:
            y_noisy = y.copy()
        # Gaussian component
        y_noisy += rng.normal(0, sigma, y.shape)
        return y_noisy
    elif noise_model == "gaussian":
        sigma = noise_params.get("sigma", 0.03)
        return y + rng.normal(0, sigma, y.shape)
    elif noise_model == "poisson":
        # Pure Poisson shot noise (for medical, EM, astronomy)
        peak = noise_params.get("peak_counts", 1000.0)
        y_pos = np.maximum(y, 0)
        scale = peak / max(y_pos.max(), 1e-8)
        y_scaled = y_pos * scale
        y_noisy = rng.poisson(np.maximum(y_scaled, 0.001)).astype(np.float64) / scale
        return y_noisy
    elif noise_model == "speckle":
        # Rayleigh-distributed multiplicative noise (SAR, ultrasound)
        n_looks = noise_params.get("n_looks", 4)
        # Speckle: y_noisy = y * (sum of n_looks exponential(1) / n_looks)
        speckle = np.zeros_like(y)
        for _ in range(int(n_looks)):
            speckle += rng.exponential(1.0, y.shape)
        speckle /= n_looks
        return y * speckle
    else:
        # Unknown model: add mild Gaussian noise as safety fallback
        return y + rng.normal(0, 0.02, y.shape)


# ── HDF5 writer ───────────────────────────────────────────────────────────────


def _write_sample(
    grp: h5py.Group,
    y: np.ndarray,
    H_ideal: np.ndarray,
    spec_ranges: list[dict],
    metadata: dict,
    x_true: np.ndarray | None = None,
    true_spec: dict | None = None,
):
    """Write a single sample to an HDF5 group."""
    grp.create_dataset("y", data=y, compression="gzip", compression_opts=4)
    grp.create_dataset("H_ideal", data=H_ideal, compression="gzip", compression_opts=4)
    grp.attrs["spec_ranges"] = json.dumps(spec_ranges)
    grp.attrs["metadata"] = json.dumps(metadata)

    # Ground-truth fields (included in Public + Hidden tiers)
    if x_true is not None:
        grp.create_dataset("x_true", data=x_true, compression="gzip", compression_opts=4)
    if true_spec is not None:
        grp.attrs["true_spec"] = json.dumps(true_spec)


# ── Per-variant generators ────────────────────────────────────────────────────


def _find_data_root() -> Path:
    """Find the project datasets directory."""
    # Try relative to script location (platform/scripts/ -> ../../datasets/)
    candidates = [
        Path(__file__).resolve().parent.parent.parent / "datasets",
        Path.cwd() / "datasets",
        Path.cwd().parent / "datasets",
    ]
    for p in candidates:
        if p.is_dir():
            return p
    raise FileNotFoundError(
        "Cannot find datasets/ directory. "
        "Run from the project root or set --data-root."
    )


def _include_ground_truth(tier_name: str, visible_data: list[str]) -> bool:
    """Determine whether to include ground truth in the HDF5 file.

    Internal files (local) always include x_true for scoring.
    Public-facing files on GCS should be post-processed to strip x_true
    from dev/hidden tiers before upload.
    """
    return True


def _generate_cassi(cfg: dict, output_dir: Path, data_root: Path | None = None):
    """Generate SD-CASSI challenge datasets (3 tiers).

    Each tier uses different hyperspectral scenes:
      - Public: KAIST 10 scenes (TSA_simu_data/Truth/)
      - Dev: Pavia University HS (web registry) or generated
      - Hidden: PnP-CASSI-Dataset crops
    """
    if data_root is None:
        data_root = _find_data_root()
    mask_path = data_root / "TSA_simu_data" / "mask.mat"

    scenes = cfg["scenes"]
    signal_shape = tuple(cfg.get("signal_shape", [256, 256, 28]))
    tier_data_sources = cfg.get("tier_data_sources", {})

    # Load real CASSI coded aperture mask
    mask = None
    if mask_path.exists():
        import scipy.io as sio
        mask_data = sio.loadmat(str(mask_path))
        mask = np.array(mask_data["mask"], dtype=np.float64)
        logger.info("Loaded real CASSI mask (%s) from %s", mask.shape, mask_path)

    for tier_name, tier_cfg in cfg["tiers"].items():
        tier_true_spec = tier_cfg["true_spec"]
        tier_seed = tier_cfg["seed"]
        visible_data = tier_cfg["visible_data"]
        tier_source = tier_data_sources.get(tier_name)

        rng = np.random.default_rng(tier_seed)

        out_path = output_dir / f"sd_cassi_challenge_{tier_name}.h5"
        logger.info("Generating %s -> %s (tier_source=%s)", tier_name, out_path,
                     tier_source.get("type", "?") if tier_source else "default")

        with h5py.File(out_path, "w") as f:
            f.attrs["variant"] = "sd_cassi"
            f.attrs["tier"] = tier_name
            f.attrs["version"] = "1.0"

            for i, scene_id in enumerate(scenes):
                # Resolve tier-specific ground truth
                try:
                    x = _resolve_tier_ground_truth(
                        "sd_cassi", tier_name, signal_shape,
                        scene_index=i, seed=tier_seed + i,
                        tier_data_source=tier_source,
                        data_root=data_root,
                    )
                except Exception as e:
                    logger.warning("Tier %s scene %d: tier GT failed (%s), trying original", tier_name, i, e)
                    # Fall back to original KAIST loading
                    truth_dir = data_root / "TSA_simu_data" / "Truth"
                    scene_path = truth_dir / f"scene{scene_id:02d}.mat"
                    if scene_path.exists():
                        x = _load_mat_scene(scene_path)
                    else:
                        logger.warning("Scene file not found: %s, skipping", scene_path)
                        continue

                # Normalize to [0, 1]
                if x.max() > 0:
                    x = x / x.max()

                # Ensure correct spectral dimension
                if x.ndim == 2 and len(signal_shape) == 3:
                    # Expand 2D to 3D by repeating with variation
                    n_bands = signal_shape[2]
                    x_3d = np.stack([x * (0.8 + 0.2 * np.sin(np.pi * b / n_bands))
                                     for b in range(n_bands)], axis=-1)
                    x = x_3d
                elif x.ndim == 3 and x.shape[2] != signal_shape[2]:
                    x = _select_bands(x, signal_shape[2], seed=tier_seed + i)

                x = _crop_or_resize_2d(x, signal_shape)

                # Generate random mask if real one not available
                if mask is None:
                    H, W = x.shape[:2]
                    mask = (rng.random((H, W)) > 0.5).astype(np.float64)

                y, H_ideal = _apply_cassi_mismatch(x, mask, tier_true_spec)
                y = _add_noise(y, cfg["noise_model"], cfg["noise_params"], rng)

                include_gt = _include_ground_truth(tier_name, visible_data)

                grp = f.create_group(f"sample_{i:02d}")
                _write_sample(
                    grp, y, H_ideal, cfg["spec_ranges"],
                    metadata={
                        "scene": f"scene{scene_id:02d}",
                        "shape": list(x.shape),
                        "noise_model": cfg["noise_model"],
                    },
                    x_true=x if include_gt else None,
                    true_spec=tier_true_spec if include_gt else None,
                )

        logger.info("Written %d samples to %s", len(scenes), out_path)


def _cacti_scene_filename(scene_name: str) -> str:
    """Map CACTI scene name to actual .mat filename."""
    _MAP = {
        "kobe": "kobe_cacti.mat",
        "traffic": "traffic_cacti.mat",
        "runner": "runner8_cacti.mat",
        "drop": "drop8_cacti.mat",
        "crash": "crash32_cacti.mat",
        "aerial": "aerial32_cacti.mat",
    }
    return _MAP.get(scene_name, f"{scene_name}_cacti.mat")


def _load_cacti_real_data(
    real_data_dir: Path, n_frames: int, max_scenes: int, seed: int,
) -> list[np.ndarray]:
    """Load real CACTI data from the real_data directory.

    Real data is organized as: real_data/cr{N}/meas_{object}_cr_{N}.mat
    We extract measurements from the lowest CR (cr10) for best quality.
    """
    import scipy.io as sio

    cr_dir = real_data_dir / "cr10"
    if not cr_dir.exists():
        # Try any CR directory
        cr_dirs = sorted(real_data_dir.glob("cr*"))
        cr_dir = cr_dirs[0] if cr_dirs else real_data_dir

    mat_files = sorted(cr_dir.glob("meas_*.mat"))
    if not mat_files:
        return []

    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(mat_files))[:max_scenes]

    scenes = []
    for idx in sorted(indices):
        fpath = mat_files[idx]
        try:
            mat = sio.loadmat(str(fpath))
            # Real CACTI .mat files have 'meas' key with measurement data
            # We need to create a synthetic video from the measurement
            for key in ("orig", "meas", "meas_real"):
                if key in mat:
                    data = np.array(mat[key], dtype=np.float64)
                    break
            else:
                continue

            if data.max() > 1.0:
                data = data / max(data.max(), 1e-8)
            data = np.clip(data, 0, 1)

            # Ensure n_frames temporal dimension
            if data.ndim == 2:
                # Single measurement: create video by tiling with variation
                H, W = data.shape
                x = np.stack([data * (0.85 + 0.15 * np.sin(np.pi * t / n_frames))
                              for t in range(n_frames)], axis=-1)
            elif data.ndim == 3:
                if data.shape[2] > n_frames:
                    data = data[:, :, :n_frames]
                elif data.shape[2] < n_frames:
                    # Pad by repeating frames
                    reps = (n_frames + data.shape[2] - 1) // data.shape[2]
                    data = np.tile(data, (1, 1, reps))[:, :, :n_frames]
                x = data
            else:
                continue

            scenes.append(x)
        except Exception as e:
            logger.warning("Failed to load CACTI real data %s: %s", fpath, e)

    return scenes


def _generate_cacti(cfg: dict, output_dir: Path, data_root: Path | None = None):
    """Generate CACTI challenge datasets (3 tiers).

    Each tier uses different video scenes:
      - Public: 6 simulation videos (CACTI/simulation/)
      - Dev: 4 real objects (CACTI/real_data/)
      - Hidden: Generated video phantoms
    """
    if data_root is None:
        data_root = _find_data_root()
    sim_dir = data_root / "CACTI" / "simulation"

    scenes = cfg["scenes"]
    n_frames = 8  # Challenge uses 8-frame videos
    signal_shape = tuple(cfg.get("signal_shape", [256, 256, 8]))
    tier_data_sources = cfg.get("tier_data_sources", {})

    for tier_name, tier_cfg in cfg["tiers"].items():
        tier_true_spec = tier_cfg["true_spec"]
        tier_seed = tier_cfg["seed"]
        visible_data = tier_cfg["visible_data"]
        tier_source = tier_data_sources.get(tier_name)

        rng = np.random.default_rng(tier_seed)
        mask = None

        out_path = output_dir / f"cacti_challenge_{tier_name}.h5"
        logger.info("Generating %s -> %s (tier_source=%s)", tier_name, out_path,
                     tier_source.get("type", "?") if tier_source else "default")

        # Pre-load tier-specific scenes if not using default simulation data
        tier_scenes = None
        if tier_source and tier_source.get("type") != "experimental" or (
            tier_source and "path" in tier_source and "simulation" not in tier_source["path"]
        ):
            if tier_source and "path" in tier_source:
                real_dir = data_root.parent / tier_source["path"]
                if not real_dir.exists():
                    real_dir = _find_data_root().parent / tier_source["path"]
                if real_dir.exists():
                    tier_scenes = _load_cacti_real_data(
                        real_dir, n_frames, len(scenes), tier_seed,
                    )
                    logger.info("  Loaded %d real CACTI scenes for tier %s", len(tier_scenes), tier_name)
            elif tier_source and "generator" in tier_source:
                seed_offset = tier_source.get("seed_offset", 0)
                tier_scenes = []
                for si in range(len(scenes)):
                    gen_seed = tier_seed + seed_offset + si
                    x_2d = _generate_fallback_phantom("cacti", signal_shape[:2], gen_seed)
                    # Expand to video frames
                    x = np.stack([x_2d * (0.85 + 0.15 * np.sin(np.pi * t / n_frames))
                                  for t in range(n_frames)], axis=-1)
                    tier_scenes.append(x)
                logger.info("  Generated %d phantom scenes for tier %s", len(tier_scenes), tier_name)

        with h5py.File(out_path, "w") as f:
            f.attrs["variant"] = "cacti"
            f.attrs["tier"] = tier_name
            f.attrs["version"] = "1.0"

            for i, scene_name in enumerate(scenes):
                if tier_scenes is not None and i < len(tier_scenes):
                    # Use tier-specific scene
                    x = tier_scenes[i]
                    # Crop/resize to signal shape
                    x = _crop_or_resize_2d(x, signal_shape)
                else:
                    # Default: load from simulation directory
                    scene_path = sim_dir / _cacti_scene_filename(scene_name)
                    if not scene_path.exists():
                        logger.warning("Scene file not found: %s, skipping", scene_path)
                        continue

                    import scipy.io as sio
                    mat = sio.loadmat(str(scene_path))

                    # Load video: take first n_frames from 'orig'
                    orig = np.array(mat["orig"], dtype=np.float64)
                    if orig.shape[2] > n_frames:
                        orig = orig[:, :, :n_frames]
                    # Normalize to [0, 1]
                    if orig.max() > 1.0:
                        orig = orig / 255.0
                    x = np.clip(orig, 0, 1)

                    # Use real mask from data (first scene sets H_ideal for all)
                    if mask is None:
                        if "mask" in mat:
                            mask = np.array(mat["mask"], dtype=np.float64)
                            if mask.shape[2] > n_frames:
                                mask = mask[:, :, :n_frames]

                # Ensure mask exists
                if mask is None:
                    H, W = x.shape[:2]
                    mask = (rng.random((H, W, n_frames)) > 0.5).astype(np.float64)

                # Normalize
                if x.max() > 1.0:
                    x = x / max(x.max(), 1e-8)
                x = np.clip(x, 0, 1)

                y, H_ideal = _apply_cacti_mismatch(x, mask, tier_true_spec)
                y = _add_noise(y, cfg["noise_model"], cfg["noise_params"], rng)

                include_gt = _include_ground_truth(tier_name, visible_data)

                grp = f.create_group(f"sample_{i:02d}")
                _write_sample(
                    grp, y, H_ideal, cfg["spec_ranges"],
                    metadata={
                        "scene": scene_name,
                        "shape": list(x.shape),
                        "noise_model": cfg["noise_model"],
                    },
                    x_true=x if include_gt else None,
                    true_spec=tier_true_spec if include_gt else None,
                )

        logger.info("Written %d samples to %s", len(scenes), out_path)


def _load_spc_images_from_dir(
    dir_path: Path, fmt: str, max_images: int, seed: int,
    target_size: tuple[int, int] | None = (256, 256),
) -> list[np.ndarray]:
    """Load grayscale images from a directory for SPC processing."""
    ext_map = {"tif": ["*.tif", "*.tiff"], "png": ["*.png"], "jpg": ["*.jpg", "*.jpeg"]}
    patterns = ext_map.get(fmt, [f"*.{fmt}"])

    files = []
    for pat in patterns:
        files.extend(sorted(dir_path.glob(pat)))
    if not files:
        return []

    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(files))[:max_images]

    images = []
    for idx in sorted(indices):
        try:
            img = _load_tif_image(files[idx], target_size=target_size)
            images.append(img)
        except Exception as e:
            logger.warning("Failed to load %s: %s", files[idx], e)

    return images


def _generate_spc(variant_key: str, cfg: dict, output_dir: Path, data_root: Path | None = None):
    """Generate SPC challenge datasets (block-based, 3 tiers).

    Images are processed in 33x33 non-overlapping blocks.
    Each block uses the same sensing matrix phi (272 x 1089).
    The phi matrix is stored once per sample as H_ideal.

    Each tier uses different image datasets:
      - Public: Set11 images
      - Dev: BSDS400 images
      - Hidden: BrainImages_test
    """
    if data_root is None:
        data_root = _find_data_root()

    scenes = cfg["scenes"]
    block_size = 33
    tier_data_sources = cfg.get("tier_data_sources", {})

    # Load ISTA-Net's trained Phi matrix (used in InverseNet paper)
    # This ensures compatibility with pretrained deep unfolding methods.
    n = block_size * block_size  # 1089
    m = n // 4  # 272
    ista_phi_path = (
        Path(__file__).resolve().parent.parent.parent
        / "papers" / "inversenet" / "data" / "spc" / "sampling_matrix"
        / "phi_0_25_1089.mat"
    )
    if ista_phi_path.exists():
        import scipy.io as sio
        phi_mat = sio.loadmat(str(ista_phi_path))
        phi_key = [k for k in phi_mat.keys() if not k.startswith("__")][0]
        phi = phi_mat[phi_key].astype(np.float64)
        logger.info("SPC sensing matrix: loaded ISTA-Net Phi from %s %s", ista_phi_path.name, phi.shape)
    else:
        # Fallback: random Gaussian if ISTA-Net Phi not available
        rng_phi = np.random.default_rng(42)
        phi = rng_phi.standard_normal((m, n)).astype(np.float64)
        phi /= np.linalg.norm(phi, axis=1, keepdims=True)
        logger.info("SPC sensing matrix: random Gaussian %s (ISTA-Net Phi not found)", phi.shape)

    for tier_name, tier_cfg in cfg["tiers"].items():
        tier_true_spec = tier_cfg["true_spec"]
        tier_seed = tier_cfg["seed"]
        visible_data = tier_cfg["visible_data"]
        tier_source = tier_data_sources.get(tier_name)

        rng = np.random.default_rng(tier_seed)

        # Resolve tier-specific image source
        tier_images = []
        if tier_source and "generator" in tier_source:
            # Use named generator for simulated scenes (dev/hidden tiers)
            gen_seed_offset = tier_source.get("seed_offset", 0)
            gen_seed = tier_seed + gen_seed_offset
            signal_shape = tuple(cfg.get("signal_shape", [256, 256]))
            tier_images = _load_scenes_from_generator(
                tier_source["generator"],
                signal_shape,
                max_scenes=len(scenes),
                seed=gen_seed,
                data_root=data_root,
            )
            img_dir = f"<generator:{tier_source['generator']}>"
            logger.info("  Generated %d simulated scenes for tier %s", len(tier_images), tier_name)
        elif tier_source and "path" in tier_source:
            img_dir = data_root.parent / tier_source["path"]
            if not img_dir.exists():
                img_dir = _find_data_root().parent / tier_source["path"]
            img_fmt = tier_source.get("format", "tif")
            signal_shape = tuple(cfg.get("signal_shape", [256, 256]))
            tier_images = _load_spc_images_from_dir(
                img_dir, img_fmt, len(scenes), tier_seed,
                target_size=(signal_shape[0], signal_shape[1]),
            )
        else:
            img_dir = data_root / "SPC" / "Set11"
            img_fmt = "tif"
            tier_images = _load_spc_images_from_dir(
                img_dir, img_fmt, len(scenes), tier_seed,
            )

        # Fallback to Set11 if tier source produced no images
        if not tier_images:
            img_dir = data_root / "SPC" / "Set11"
            tier_images = _load_spc_images_from_dir(img_dir, "tif", len(scenes), tier_seed)

        if not tier_images:
            logger.warning("No images found for %s tier %s, skipping", variant_key, tier_name)
            continue

        out_path = output_dir / f"{variant_key}_challenge_{tier_name}.h5"
        logger.info("Generating %s -> %s (images from %s)", tier_name, out_path, img_dir)

        with h5py.File(out_path, "w") as f:
            f.attrs["variant"] = variant_key
            f.attrs["tier"] = tier_name
            f.attrs["version"] = "1.0"
            f.attrs["block_size"] = block_size

            for i, scene_id in enumerate(scenes):
                idx = i % len(tier_images)
                x = tier_images[idx]

                # Crop image to multiple of block_size
                H, W = x.shape
                H_crop = (H // block_size) * block_size
                W_crop = (W // block_size) * block_size
                if H_crop == 0 or W_crop == 0:
                    # Image too small: resize to 256x256
                    x = _crop_or_resize_2d(x, (256, 256))
                    H, W = x.shape
                    H_crop = (H // block_size) * block_size
                    W_crop = (W // block_size) * block_size
                x = x[:H_crop, :W_crop]

                y, _ = _apply_spc_mismatch_block(x, phi, tier_true_spec, block_size)
                # Add noise to each block's measurements
                y = _add_noise(y, cfg["noise_model"], cfg["noise_params"], rng)

                include_gt = _include_ground_truth(tier_name, visible_data)

                scene_name = f"scene_{i:02d}" if i >= len(tier_images) else tier_images[idx].__class__.__name__
                grp = f.create_group(f"sample_{i:02d}")
                _write_sample(
                    grp, y, phi, cfg["spec_ranges"],
                    metadata={
                        "scene": f"scene_{i:02d}",
                        "shape": list(x.shape),
                        "block_size": block_size,
                        "noise_model": cfg["noise_model"],
                    },
                    x_true=x if include_gt else None,
                    true_spec=tier_true_spec if include_gt else None,
                )

        logger.info("Written %d samples to %s", len(scenes), out_path)


# ══════════════════════════════════════════════════════════════════════════════
#  Generic generator (handles all non-hand-crafted variants)
# ══════════════════════════════════════════════════════════════════════════════


def _generate_generic(variant_key: str, cfg: dict, output_dir: Path, data_root: Path | None = None):
    """Generate challenge datasets for any variant using the generic pipeline.

    Works for all 168+ modalities by:
      1. Resolving per-tier ground-truth via tier_data_sources
      2. Applying physics-accurate forward models via runner type
      3. Applying mismatch perturbations from true_spec
      4. Adding category-appropriate noise
    """
    runner_type = _get_runner_type(variant_key)
    signal_shape = tuple(cfg.get("signal_shape", [256, 256]))
    scene_count = cfg.get("scene_count", 5)
    noise_model = cfg.get("noise_model", "gaussian")
    noise_params = cfg.get("noise_params", {})
    tier_data_sources = cfg.get("tier_data_sources", {})

    logger.info(
        "Generic generator: %s  runner=%s  shape=%s  scenes=%d  noise=%s  tier_sources=%s",
        variant_key, runner_type, signal_shape, scene_count, noise_model,
        "yes" if tier_data_sources else "no",
    )

    for tier_name, tier_cfg in cfg["tiers"].items():
        tier_true_spec = tier_cfg["true_spec"]
        tier_seed = tier_cfg["seed"]
        visible_data = tier_cfg["visible_data"]
        tier_source = tier_data_sources.get(tier_name)

        rng = np.random.default_rng(tier_seed)

        out_path = output_dir / f"{variant_key}_challenge_{tier_name}.h5"
        logger.info("  Generating %s -> %s", tier_name, out_path)

        with h5py.File(out_path, "w") as f:
            f.attrs["variant"] = variant_key
            f.attrs["tier"] = tier_name
            f.attrs["version"] = "1.0"
            f.attrs["runner_type"] = runner_type

            written = 0
            for i in range(scene_count):
                scene_seed = tier_seed + i

                # 1. Get tier-specific ground truth
                try:
                    x = _resolve_tier_ground_truth(
                        variant_key, tier_name, signal_shape,
                        scene_index=i, seed=scene_seed,
                        tier_data_source=tier_source,
                        data_root=data_root,
                    )
                except Exception as e:
                    logger.warning("  Scene %d: ground truth failed (%s), using fallback", i, e)
                    tier_offset = _TIER_SEED_OFFSETS.get(tier_name, 0)
                    x = _generate_fallback_phantom(variant_key, signal_shape, scene_seed + tier_offset)

                # Normalize to [0, 1]
                xmax = x.max()
                if xmax > 0:
                    x = x / xmax
                x = np.clip(x, 0, 1)

                # 2. Apply forward model
                y, H_ideal = _apply_forward_model(runner_type, x, rng, seed=42 + i)

                # 3. Apply mismatch
                y = _apply_generic_mismatch(y, H_ideal, tier_true_spec, runner_type, rng)

                # 4. Add noise
                y = _add_noise(y, noise_model, noise_params, rng)

                # 5. Write sample
                include_gt = _include_ground_truth(tier_name, visible_data)

                grp = f.create_group(f"sample_{i:02d}")
                _write_sample(
                    grp, y, H_ideal, cfg["spec_ranges"],
                    metadata={
                        "scene": f"scene_{i:02d}",
                        "shape": list(x.shape),
                        "runner_type": runner_type,
                        "noise_model": noise_model,
                    },
                    x_true=x if include_gt else None,
                    true_spec=tier_true_spec if include_gt else None,
                )
                written += 1

        logger.info("  Written %d samples to %s", written, out_path)


# ── Variant dispatch ──────────────────────────────────────────────────────────

# Hand-crafted generators for the original 4 variants (real datasets)
_GENERATORS = {
    "sd_cassi": lambda cfg, out, dr: _generate_cassi(cfg, out, dr),
    "cacti": lambda cfg, out, dr: _generate_cacti(cfg, out, dr),
    "spc_block": lambda cfg, out, dr: _generate_spc("spc_block", cfg, out, dr),
    "spc_kronecker": lambda cfg, out, dr: _generate_spc("spc_kronecker", cfg, out, dr),
}


def generate_variant(variant_key: str, output_dir: Path, data_root: Path | None = None):
    """Generate challenge datasets for a single variant.

    Uses hand-crafted generators for the 4 original variants (sd_cassi, cacti,
    spc_block, spc_kronecker) and falls through to the generic pipeline for
    all other variants.
    """
    cfg = CHALLENGE_CONFIG.get(variant_key)
    if cfg is None:
        raise ValueError(f"No challenge config for variant: {variant_key}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Try hand-crafted generator first
    gen = _GENERATORS.get(variant_key)
    if gen is not None:
        gen(cfg, output_dir, data_root)
        return

    # Fall through to generic generator
    _generate_generic(variant_key, cfg, output_dir, data_root)


# ── GCS upload ───────────────────────────────────────────────────────────────


def _get_gcs_bucket():
    """Get the GCS bucket handle, or None if unavailable."""
    try:
        from google.cloud import storage
    except ImportError:
        logger.error("google-cloud-storage not installed. Run: pip install google-cloud-storage")
        return None

    bucket_name = "pwm-benchmark-datasets"
    try:
        client = storage.Client()
        return client.bucket(bucket_name)
    except Exception as e:
        logger.error("Cannot connect to GCS bucket %s: %s", bucket_name, e)
        return None


def _upload_to_gcs(output_dir: Path):
    """Upload all HDF5 files in output_dir to GCS bucket."""
    bucket = _get_gcs_bucket()
    if bucket is None:
        return

    prefix = "challenge-data/v1.0/"
    h5_files = sorted(output_dir.glob("*.h5"))
    logger.info("Uploading %d files to gs://%s/%s", len(h5_files), bucket.name, prefix)

    for h5_path in h5_files:
        blob_name = prefix + h5_path.name
        blob = bucket.blob(blob_name)
        logger.info("  Uploading %s -> gs://%s/%s", h5_path.name, bucket.name, blob_name)
        blob.upload_from_filename(str(h5_path))

    logger.info("GCS upload complete.")


def _upload_file_to_gcs(h5_path: Path, bucket) -> bool:
    """Upload a single HDF5 file to GCS. Returns True on success."""
    prefix = "challenge-data/v1.0/"
    blob_name = prefix + h5_path.name
    blob = bucket.blob(blob_name)
    try:
        logger.info("  Uploading %s -> gs://%s/%s", h5_path.name, bucket.name, blob_name)
        blob.upload_from_filename(str(h5_path))
        return True
    except Exception as e:
        logger.error("  Upload failed for %s: %s", h5_path.name, e)
        return False


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Generate HDF5 challenge datasets for the Blind Reconstruction Challenge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --variant sd_cassi              # Original 4 variants (real data)
  %(prog)s --variant ct                    # Single generic variant
  %(prog)s --variant all                   # All variants with challenge configs
  %(prog)s --variant all-challenge         # Same as 'all' (explicit)
  %(prog)s --variant all --category microscopy  # Filter by category
  %(prog)s --variant all --upload-gcs      # Generate + upload to GCS (keep local)
  %(prog)s --variant all --gcs-only        # Generate + upload to GCS (no local storage)
  %(prog)s --variant all --dry-run         # List variants without generating
""",
    )
    parser.add_argument(
        "--variant",
        required=True,
        help=(
            "Variant key (e.g., sd_cassi, ct, mri, widefield) or "
            "'all' / 'all-challenge' for all configured variants"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "pwm_platform" / "static" / "benchmark-data" / "challenge-data" / "v1.0",
        help="Output directory for HDF5 files (default: static/benchmark-data/challenge-data/v1.0/)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root directory containing source datasets (auto-detected if not set)",
    )
    parser.add_argument(
        "--category",
        default=None,
        help="Filter variants by category (e.g., microscopy, medical, electron_microscopy)",
    )
    parser.add_argument(
        "--upload-gcs",
        action="store_true",
        help="Upload generated HDF5 files to GCS bucket after generation (keeps local copies)",
    )
    parser.add_argument(
        "--gcs-only",
        action="store_true",
        help=(
            "Generate → upload to GCS → delete local files. "
            "No local storage; downloads served via the /gcs/ proxy from GCS."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List variants that would be generated without actually generating",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # --gcs-only implies --upload-gcs
    if args.gcs_only:
        args.upload_gcs = True

    # Ensure CHALLENGE_CONFIG is fully populated (auto-generated entries)
    try:
        from pwm_platform.services.benchmark_database import CHALLENGE_CONFIG as _full_config
        # The import triggers auto-generation in __init__.py
        # Merge any auto-generated entries back
        for k, v in _full_config.items():
            if k not in CHALLENGE_CONFIG:
                CHALLENGE_CONFIG[k] = v
        logger.info("Loaded %d challenge configs (including auto-generated)", len(CHALLENGE_CONFIG))
    except ImportError:
        logger.info("Using %d challenge configs from _challenge_data.py only", len(CHALLENGE_CONFIG))

    # Resolve variant list
    if args.variant in ("all", "all-challenge"):
        variants = sorted(CHALLENGE_CONFIG.keys())
    else:
        variants = [args.variant]

    # Filter by category if specified
    if args.category:
        cat_filter = args.category.lower().replace(" ", "_")
        filtered = []
        for v in variants:
            try:
                cat = _get_variant_category(v)
                if cat.lower().replace(" ", "_") == cat_filter:
                    filtered.append(v)
            except Exception:
                pass
        logger.info("Category filter '%s': %d / %d variants", args.category, len(filtered), len(variants))
        variants = filtered

    if not variants:
        logger.error("No variants to generate. Check --variant and --category flags.")
        return

    # Dry run: just list
    if args.dry_run:
        print(f"\n{'Variant':<35} {'Runner':<10} {'Scenes':<7} {'Noise':<18} {'Shape'}")
        print("=" * 100)
        for v in variants:
            cfg = CHALLENGE_CONFIG.get(v)
            if cfg:
                runner = _get_runner_type(v)
                shape = cfg.get("signal_shape", [256, 256])
                scenes = cfg.get("scene_count", "?")
                noise = cfg.get("noise_model", "?")
                print(f"{v:<35} {runner:<10} {scenes:<7} {noise:<18} {shape}")
        print(f"\nTotal: {len(variants)} variants × 3 tiers = {len(variants) * 3} HDF5 files")
        return

    # For --gcs-only: use a temp directory, verify GCS is reachable first
    import tempfile

    gcs_bucket = None
    if args.gcs_only:
        gcs_bucket = _get_gcs_bucket()
        if gcs_bucket is None:
            logger.error("--gcs-only requires a working GCS connection. Aborting.")
            return
        # Use a temp directory instead of the default output-dir
        tmp_dir = Path(tempfile.mkdtemp(prefix="pwm_challenge_"))
        output_dir = tmp_dir
        logger.info("GCS-only mode: generating to temp dir %s", tmp_dir)
    else:
        output_dir = args.output_dir

    # Generate (and upload per-variant for --gcs-only)
    successes = 0
    failures = 0
    uploaded = 0
    for i, v in enumerate(variants, 1):
        logger.info("=== [%d/%d] Generating challenge datasets for %s ===", i, len(variants), v)
        try:
            generate_variant(v, output_dir, args.data_root)
            successes += 1

            # --gcs-only: upload this variant's files immediately, then delete
            if args.gcs_only and gcs_bucket is not None:
                variant_files = sorted(output_dir.glob(f"{v}_challenge_*.h5"))
                for h5_path in variant_files:
                    if _upload_file_to_gcs(h5_path, gcs_bucket):
                        uploaded += 1
                    h5_path.unlink()  # delete local copy regardless

        except FileNotFoundError as e:
            logger.error("Skipping %s: %s", v, e)
            failures += 1
        except Exception:
            logger.exception("Failed to generate %s", v)
            failures += 1

    logger.info("Done. %d succeeded, %d failed out of %d variants.", successes, failures, len(variants))

    if args.gcs_only:
        logger.info("GCS-only: uploaded %d files, no local copies kept.", uploaded)
        # Clean up the temp directory
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
    elif args.upload_gcs and successes > 0:
        # Batch upload after all generation (keeps local copies)
        _upload_to_gcs(output_dir)


if __name__ == "__main__":
    main()
