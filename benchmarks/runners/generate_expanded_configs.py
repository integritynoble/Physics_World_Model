#!/usr/bin/env python3
"""Generate expanded benchmark configs for all 168 modalities.

Reads the existing single-modality configs from benchmarks/configs/
and the modality registry to produce expanded configs with all variants,
sizes, noise levels, and mismatch levels.

Usage:
    python3 -m benchmarks.runners.generate_expanded_configs
    python3 -m benchmarks.runners.generate_expanded_configs --modality mri
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CONFIGS_DIR = ROOT / "benchmarks" / "configs"
EXPANDED_DIR = ROOT / "benchmarks" / "expanded_configs"
EXPANDED_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Modality variant registry
# Maps modality_id -> list of (variant_id, variant_name, dag, optical_elements, reference)
# ============================================================
VARIANT_REGISTRY = {
    # --- Compressive Imaging ---
    "spc": [
        ("hadamard_spc", "Hadamard SPC", "Src -> M(DMD_Hadamard) -> Sigma -> D(photodiode)",
         ["LED source", "DMD (TI DLP, Hadamard patterns)", "Collection lens", "Single photodiode"],
         "Duarte et al., IEEE SPM 2008"),
        ("fourier_spc", "Fourier SPC", "Src -> M(DMD_sinusoidal) -> Sigma -> D(photodiode)",
         ["LED source", "DMD (sinusoidal fringe patterns)", "Collection lens", "Photodiode"],
         "Fourier single-pixel imaging"),
        ("gaussian_spc", "Gaussian Random SPC", "Src -> M(DMD_random) -> Sigma -> D(photodiode)",
         ["LED source", "DMD (Gaussian random patterns)", "Collection lens", "Photodiode"],
         "CS-theory optimal; RIP guarantee"),
        ("hyperspectral_spc", "Hyperspectral SPC", "Src(broadband) -> M(DMD) -> Sigma -> D(spectrometer)",
         ["Broadband source", "DMD", "Collection lens", "Spectrometer (2048 channels)"],
         "OpenSpyrit SPIHIM"),
        ("adaptive_spc", "Adaptive SPC", "Src -> M(DMD_wavelet) -> Sigma -> D(photodiode)",
         ["LED source", "DMD (adaptive wavelet patterns)", "Collection lens", "Photodiode"],
         "Rousset et al., IEEE TCI 2017"),
        ("ghost_spc", "Compressive Ghost Imaging", "Src(thermal/SPDC) -> M(correlations) -> Sigma -> D(bucket)",
         ["Thermal/SPDC source", "Spatial correlations", "Bucket detector"],
         "Ghost imaging variant"),
    ],
    "cacti": [
        ("grayscale_cacti", "Grayscale CACTI", "Src -> M(shifting_binary_mask) -> Sigma(temporal) -> D(CCD)",
         ["Scene", "Binary mask on translation stage", "CCD sensor"],
         "Llull et al., Optics Express 2013"),
        ("color_cacti", "Color CACTI (Bayer)", "Src -> M(shifting_mask) -> Sigma(temporal) -> D(Bayer_CCD)",
         ["Scene", "Binary mask", "Bayer CFA", "CCD sensor"],
         "Color video reconstruction"),
        ("dual_mask_cacti", "Dual-mask CACTI", "Src -> M(mask1) -> M(mask2) -> Sigma -> D",
         ["Scene", "Mask 1", "Mask 2 (different plane)", "CCD sensor"],
         "Additional spatial diversity"),
        ("coded_exposure_sci", "Coded Exposure SCI", "Src -> M(electronic_shutter) -> Sigma -> D",
         ["Scene", "Electronic shutter modulation (no moving parts)", "CCD sensor"],
         "No mechanical mask"),
        ("spectral_temporal_cacti", "Spectral-temporal CACTI", "Src -> M(mask) -> W(prism) -> Sigma -> D",
         ["Scene", "Coded mask", "Disperser (prism)", "CCD sensor"],
         "Hyperspectral video"),
    ],
    "matrix": [
        ("dense_gaussian", "Dense Gaussian Matrix", "Src -> M(dense_gaussian) -> D",
         ["Signal source", "Dense Gaussian measurement matrix", "Detector"], "CS theory"),
        ("sparse_binary", "Sparse Binary Matrix", "Src -> M(sparse_binary) -> D",
         ["Signal source", "Sparse 0/1 measurement matrix", "Detector"], "Sparse sensing"),
        ("partial_dct", "Structured DCT Matrix", "Src -> M(partial_DCT) -> D",
         ["Signal source", "Partial DCT measurement matrix", "Detector"], "DCT-based CS"),
        ("learned_matrix", "Learned Matrix", "Src -> M(learned) -> D",
         ["Signal source", "End-to-end optimized measurement matrix", "Detector"], "Learned CS"),
    ],
    # --- Medical Imaging ---
    "mri": [
        ("cartesian_single", "Cartesian Single-Coil MRI", "Src(RF) -> M(tissue) -> F(k-space_Cartesian) -> S(uniform) -> D(ADC)",
         ["RF excitation coil", "Single receive coil", "ADC"], "Basic MRI"),
        ("cartesian_sense", "Cartesian Multi-Coil SENSE", "Src(RF) -> M(coil_8ch) -> F(k-space) -> S(undersampled) -> D(ADC_8ch)",
         ["RF excitation coil", "8-32 channel receive coil array", "SENSE reconstruction"], "Pruessmann et al., MRM 1999"),
        ("cartesian_grappa", "Cartesian Multi-Coil GRAPPA", "Src(RF) -> M(coil_8ch) -> F(k-space) -> S(undersampled_ACS) -> D(ADC_8ch)",
         ["RF excitation coil", "Multi-channel coil array", "ACS calibration lines"], "Griswold et al., MRM 2002"),
        ("radial", "Radial Trajectory MRI", "Src(RF) -> M(coil) -> F(k-space_radial) -> S(golden_angle) -> D(ADC)",
         ["RF excitation coil", "Receive coil", "Radial gradient waveforms"], "Golden-angle radial"),
        ("spiral", "Spiral Trajectory MRI", "Src(RF) -> M(coil) -> F(k-space_spiral) -> S(Archimedean) -> D(ADC)",
         ["RF excitation coil", "Receive coil", "Spiral gradient waveforms"], "Archimedean spiral"),
        ("volumetric_3d", "3D Volumetric MRI", "Src(RF) -> M(coil_32ch) -> F(k-space_3D) -> S(undersampled_3D) -> D(ADC_32ch)",
         ["RF excitation coil", "32-channel coil array", "3D encoding gradients"], "3D acquisition"),
    ],
    "pet": [
        ("pet_2d", "2D PET", "Src(annihilation) -> Pi(LOR_2D) -> D(BGO)",
         ["Annihilation source (511 keV)", "Septa (inter-ring)", "BGO scintillator ring"], "2D PET with septa"),
        ("pet_3d", "3D PET", "Src(annihilation) -> Pi(LOR_3D) -> D(LYSO)",
         ["Annihilation source", "No septa", "LYSO scintillator ring"], "3D PET, no septa"),
        ("tof_pet", "Time-of-Flight PET", "Src(annihilation) -> Pi(LOR_TOF) -> D(LYSO_fast)",
         ["Annihilation source", "Fast LYSO scintillator (200-400 ps)", "TOF electronics"], "TOF-PET"),
        ("listmode_pet", "List-mode PET", "Src(annihilation) -> Pi(LOR_listmode) -> D(LYSO)",
         ["Annihilation source", "List-mode acquisition electronics", "LYSO detector"], "Event-by-event"),
        ("total_body_pet", "Total-Body PET", "Src(annihilation) -> Pi(LOR_extended) -> D(LYSO_2m)",
         ["Annihilation source", ">1m axial FOV detector ring", "LYSO scintillator"], "uExplorer/Quadra"),
    ],
    "ultrasound": [
        ("focused_us", "Focused Transmit B-mode", "Src(piezo_128ch) -> P(wave_focused) -> D(128ch_ADC)",
         ["128-element linear array (5 MHz)", "Focused transmit beam", "128-channel ADC"], "Standard clinical US"),
        ("planewave_us", "Plane-Wave Ultrasound", "Src(piezo_128ch) -> P(wave_planar) -> D(128ch_ADC)",
         ["128-element linear array", "Unfocused plane wave", "Coherent compounding"], "Ultrafast US"),
        ("diverging_us", "Diverging Wave Ultrasound", "Src(phased_array) -> P(wave_diverging) -> D(64ch_ADC)",
         ["64-element phased array", "Virtual apex diverging wave", "ADC"], "Cardiac ultrafast"),
        ("synthetic_aperture_us", "Synthetic Aperture US", "Src(single_element) -> P(wave) -> D(128ch)",
         ["Single-element transmit", "Full synthetic aperture receive", "128-channel ADC"], "Maximum flexibility"),
        ("3d_us", "3D/4D Ultrasound", "Src(matrix_array) -> P(wave_3D) -> D(matrix_ADC)",
         ["2D matrix array transducer", "3D beam steering", "Volumetric ADC"], "Volumetric imaging"),
    ],
    # --- Microscopy ---
    "sim": [
        ("2d_sim", "2D-SIM (linear)", "Src(laser) -> M(grating_3ang_3phase) -> C(PSF_NA1.49) -> D(sCMOS)",
         ["Laser (488/561 nm)", "SLM or grating (3 angles x 3 phases = 9 frames)", "Objective 100x/1.49 oil", "sCMOS (6.5um)"],
         "Standard 2D-SIM"),
        ("3d_sim", "3D-SIM", "Src(laser) -> M(3D_pattern_3ang_5phase) -> C(PSF_3D) -> D(sCMOS)",
         ["Laser", "SLM (3 angles x 5 phases = 15 frames, axial modulation)", "Objective 100x/1.49", "sCMOS"],
         "2x lateral + 3x axial resolution"),
        ("nl_sim", "Nonlinear SIM (NL-SIM)", "Src(high_power_laser) -> M(saturated_pattern) -> C(PSF) -> D(sCMOS)",
         ["High-power laser", "Photoswitchable fluorophores", "SLM (>25 frames)", "sCMOS"],
         ">2x resolution, 5-7 harmonics"),
        ("lattice_sim", "Lattice SIM", "Src(laser) -> M(lattice_spot_pattern) -> C(PSF) -> D(sCMOS)",
         ["Laser", "Lattice spot pattern (not grating)", "Objective", "sCMOS"],
         "Zeiss Lattice SIM; faster, lower phototoxicity"),
        ("lls_sim", "Lattice Light-Sheet SIM", "Src(Bessel_lattice) -> M(lattice_pattern) -> C(PSF_lightsheet) -> D(sCMOS)",
         ["Bessel beam lattice light-sheet", "SIM illumination pattern", "Orthogonal detection", "sCMOS"],
         "~120nm lat, ~160nm axial"),
        ("isim", "Instant SIM (iSIM)", "Src(laser) -> M(microlens_array) -> C(PSF_reassigned) -> D(sCMOS)",
         ["Laser", "Spinning disk with microlens array", "Pixel reassignment optics", "sCMOS"],
         "Real-time; ~1.4x resolution gain"),
        ("open_sim", "openSIM", "Src(LED) -> M(DMD_pattern) -> C(PSF) -> D(sCMOS)",
         ["LED", "DMD pattern generator (UC2 compatible)", "Objective", "sCMOS"],
         "Open-hardware add-on"),
    ],
    "widefield": [
        ("standard_wf", "Standard Widefield", "Src(LED_470nm) -> C(PSF_Airy_NA1.3) -> D(sCMOS_16bit)",
         ["LED (470 nm)", "Objective 40x/1.3 oil", "Tube lens", "sCMOS (6.5um, 16-bit)"],
         "Standard widefield fluorescence"),
        ("low_dose_wf", "Low-Dose Widefield", "Src(LED_low) -> C(PSF_Airy) -> D(sCMOS)",
         ["LED (reduced power: 10-500 photons/px)", "Objective", "sCMOS"],
         "Photon-starved regime"),
        ("deconv_wf", "Deconvolution Widefield", "Src(LED) -> C(PSF_measured) -> D(sCMOS) + RL_deconv",
         ["LED", "Objective", "sCMOS", "Richardson-Lucy or CARE post-processing"],
         "Restores 3D optical sections"),
    ],
    "lightsheet": [
        ("spim", "SPIM (OpenSPIM)", "Src(laser_sheet) -> C(PSF_sheet_5um) -> D(sCMOS)",
         ["Cylindrical lens (sheet generation)", "Detection objective (perpendicular)", "sCMOS"],
         "Single-plane illumination"),
        ("dispim", "diSPIM", "[Src1->C->D1; Src2->C->D2]",
         ["Dual perpendicular light sheets", "Two objectives", "Two sCMOS cameras"],
         "Isotropic resolution via joint deconvolution"),
        ("llsm", "Lattice Light-Sheet (LLSM)", "Src(Bessel_lattice) -> C(PSF_Bessel) -> D(sCMOS)",
         ["Bessel beam lattice generator", "Excitation/detection objectives", "sCMOS"],
         "Chen lab, Janelia; near-isotropic"),
        ("opm", "Oblique Plane Microscope (OPM)", "Src(oblique_sheet_45deg) -> C(PSF) -> D(sCMOS)",
         ["Single objective", "Tilted light sheet (35-45 deg)", "sCMOS"],
         "Single-objective design"),
        ("mesospim", "mesoSPIM", "Src(laser_sheet_wide) -> C(PSF) -> D(sCMOS)",
         ["Wide laser sheet", "Low-magnification objective", "Large-FOV sCMOS"],
         "Cleared-tissue, cm-scale FOV"),
        ("exa_spim", "ExA-SPIM", "Src(laser_sheet) -> C(PSF) -> D(sCMOS)",
         ["Laser sheet", "Expansion microscopy sample", "High-res sCMOS"],
         "Expansion + SPIM; whole mouse brain"),
    ],
    "palm_storm": [
        ("2d_palm", "2D-PALM", "Src(405nm_activation) -> M(photoactivatable_FP) -> D(EMCCD)",
         ["405 nm activation laser", "Photoactivatable fluorescent protein", "EMCCD (16um pixel)"],
         "Photoactivatable localization microscopy"),
        ("2d_storm", "2D-STORM", "Src(647nm) -> M(photoswitchable_dye) -> D(EMCCD)",
         ["647 nm excitation laser", "Cy5/Alexa647 photoswitchable dyes", "EMCCD + TIRF illumination"],
         "Stochastic optical reconstruction microscopy"),
        ("3d_storm_astigmatic", "3D-STORM (Astigmatic)", "Src(647nm) -> M(dye) -> C(cylindrical_lens) -> D(EMCCD)",
         ["647 nm laser", "Photoswitchable dye", "Cylindrical lens for astigmatism", "EMCCD"],
         "Astigmatic z-encoding; +/-600nm axial range"),
        ("3d_storm_dh", "3D-STORM (Double-Helix)", "Src(647nm) -> M(dye) -> C(DH_PSF_mask) -> D(EMCCD)",
         ["647 nm laser", "Photoswitchable dye", "Double-helix PSF phase mask", "EMCCD"],
         "Double-helix PSF for z-encoding"),
    ],
    "fpm": [
        ("led_array_fpm", "LED Array FPM", "Src(LED_15x15) -> M(thin_sample) -> P(objective_pupil_4x) -> D(sCMOS)",
         ["15x15 LED array (4mm pitch, 80mm distance)", "Thin sample", "4x/0.1NA objective", "sCMOS"],
         "Zheng et al., Nature Photonics 2013"),
        ("dome_fpm", "Dome Illumination FPM", "Src(LED_dome) -> M(sample) -> P(pupil) -> D(sCMOS)",
         ["Hemispherical LED dome", "Sample", "Objective", "sCMOS"],
         "Dark-field angles included"),
        ("3d_fpm", "3D FPM", "Src(LED_array) -> M(thick_sample_multislice) -> P(pupil) -> D(sCMOS)",
         ["LED array", "Thick sample (multi-slice model)", "Objective", "sCMOS"],
         "Volumetric reconstruction"),
        ("multispectral_fpm", "Multispectral FPM", "Src(RGB_LED_array) -> M(sample) -> P(pupil) -> D(sCMOS)",
         ["RGB LED array", "Sample", "Objective", "sCMOS"],
         "Spectral imaging via wavelength multiplexing"),
    ],
    # --- Coherent Imaging ---
    "ptychography": [
        ("farfield_xray", "Far-field X-ray Ptychography", "Src(synchrotron_10keV) -> M(probe) -> P(Fresnel) -> D(CCD_2048)",
         ["Synchrotron X-ray (10 keV)", "Focused probe (zone plate)", "Area detector (2048x2048)"],
         "PtychoNN benchmark"),
        ("nearfield_ptycho", "Near-field Ptychography", "Src(synchrotron) -> M(probe) -> P(Fresnel_short) -> D(CCD)",
         ["Synchrotron source", "Probe", "Short propagation distance", "Detector"],
         "Relaxed overlap; Fresnel zone"),
        ("bragg_ptycho", "Bragg Ptychography", "Src(synchrotron) -> M(crystal_probe) -> R(Bragg) -> D(CCD)",
         ["Synchrotron source", "Focused probe", "Crystal sample at Bragg angle", "Area detector"],
         "3D strain mapping"),
        ("electron_ptycho", "Electron Ptychography (4D-STEM)", "Src(e_gun_200keV) -> M(STEM_probe) -> P(elastic_scatter) -> D(pixelated_256x256)",
         ["Electron gun (200 keV)", "STEM probe", "Pixelated detector (256x256)"],
         "4D dataset: scan(x,y) + diff(kx,ky)"),
    ],
    "holography": [
        ("offaxis_dhm", "Off-axis DHM", "Src(HeNe_633nm) -> P(sample) + P(ref_tilted_3deg) -> Sigma(interference) -> D(CCD)",
         ["HeNe laser (633 nm)", "Sample arm", "Tilted reference beam (3 deg)", "CCD (interference)"],
         "Single-shot; carrier frequency separation"),
        ("inline_gabor", "In-line (Gabor) Holography", "Src(laser) -> P(sample) -> D(CCD)",
         ["Laser source", "Sample (in-line)", "CCD (no separate reference)"],
         "Coaxial; twin image artifact"),
        ("lensless_dhm", "Lensless DHM (DLHM)", "Src(point_source) -> P(Fresnel) -> D(CCD_no_lens)",
         ["Point source (LED or fiber)", "Free-space Fresnel propagation", "Bare CCD sensor (no lens)"],
         "No objective; long working distance"),
        ("electron_holography", "Electron Holography", "Src(e_gun) -> P(biprism) -> D(CCD)",
         ["Electron gun (TEM)", "Electrostatic biprism", "CCD"],
         "Electric/magnetic field measurement"),
    ],
    "lensless": [
        ("diffusercam", "DiffuserCam", "Src(scene) -> C(random_diffuser_PSF) -> D(CMOS)",
         ["Scene", "Polycarbonate diffuser (random PSF)", "Bare CMOS sensor"],
         "Antipa et al., Optica 2018"),
        ("phlatcam", "PhlatCam", "Src(scene) -> C(designed_spiral_phase_mask) -> D(CMOS)",
         ["Scene", "Optimized spiral phase mask", "CMOS sensor"],
         "Boominathan et al., IEEE TPAMI 2020"),
        ("flatcam", "FlatCam/FlatScope", "Src(scene) -> M(binary_amplitude_mask) -> D(CMOS)",
         ["Scene", "Random binary amplitude mask", "CMOS (mask-sensor distance ~0)"],
         "Ultra-thin camera"),
        ("spectral_diffusercam", "SpectralDiffuserCam", "Src(scene) -> C(diffuser) -> W(spectral_CFA) -> D(CMOS)",
         ["Scene", "Diffuser", "Spectral filter array", "CMOS"],
         "Hyperspectral single-shot"),
    ],
    # --- Neural Rendering ---
    "nerf": [
        ("nerf_original", "NeRF (original MLP)", "Src(scene) -> M(volume_density) -> P(ray_marching) -> D(camera)",
         ["Scene", "MLP (1.2M params)", "Volume rendering + positional encoding", "Camera"],
         "Mildenhall et al., ECCV 2020"),
        ("mipnerf360", "Mip-NeRF 360", "Src(scene) -> M(volume) -> P(cone_casting) -> D(camera)",
         ["Scene", "Anti-aliased MLP (9M params)", "Cone casting", "Camera"],
         "Barron et al., CVPR 2022"),
        ("instant_ngp", "Instant-NGP", "Src(scene) -> M(hash_grid) -> P(ray_marching) -> D(camera)",
         ["Scene", "Hash encoding + small MLP (5M params)", "Volume rendering", "Camera"],
         "Muller et al., SIGGRAPH 2022"),
        ("nerf_minus", "NeRF-- (no poses)", "Src(scene) -> M(volume) -> P(ray_marching) -> D(camera) + pose_solver",
         ["Scene", "MLP", "Volume rendering", "Camera", "Joint pose estimation"],
         "No known camera poses"),
        ("zip_nerf", "Zip-NeRF", "Src(scene) -> M(hash_grid) -> P(cone_casting) -> D(camera)",
         ["Scene", "Mip-NeRF + hash grid encoding", "Cone casting", "Camera"],
         "State of art quality"),
    ],
    "gaussian_splatting": [
        ("3dgs_original", "3D Gaussian Splatting", "Src(scene) -> M(3D_Gaussians) -> P(diff_rasterizer) -> D(camera)",
         ["Scene", "100K-5M 3D Gaussian splats", "Differentiable tile-based rasterizer", "Camera"],
         "Kerbl et al., SIGGRAPH 2023"),
        ("2dgs", "2D Gaussian Splatting", "Src(scene) -> M(2D_disk_Gaussians) -> P(diff_rasterizer) -> D(camera)",
         ["Scene", "2D disk primitives", "Rasterizer", "Camera"],
         "Better surface reconstruction"),
        ("scaffold_gs", "Scaffold-GS", "Src(scene) -> M(anchor_Gaussians_voxel) -> P(diff_rasterizer) -> D(camera)",
         ["Scene", "Voxel scaffold + anchor Gaussians", "Rasterizer", "Camera"],
         "Compact; fewer Gaussians"),
        ("mip_splatting", "Mip-Splatting", "Src(scene) -> M(3D_Gaussians_smoothed) -> P(diff_rasterizer) -> D(camera)",
         ["Scene", "3D Gaussians + 3D smoothing filter", "Rasterizer", "Camera"],
         "Anti-aliased"),
        ("4dgs", "4D Gaussian Splatting", "Src(dynamic_scene) -> M(4D_Gaussians) -> P(diff_rasterizer) -> D(camera)",
         ["Dynamic scene", "4D Gaussians (space + time)", "Temporal rasterizer", "Camera"],
         "Dynamic scenes"),
    ],
}

# Standard mismatch levels for all modalities
STANDARD_MISMATCH_LEVELS = {
    "M0_nominal": {"description": "No mismatch - perfect forward model", "n_params_perturbed": 0},
    "M1_single": {"description": "Single parameter perturbed", "n_params_perturbed": 1},
    "M2_compound": {"description": "3+ parameters simultaneously perturbed", "n_params_perturbed": 3},
    "M3_real": {"description": "Real calibration/experimental errors", "n_params_perturbed": "all"},
    "M4_adversarial": {"description": "Worst-case mismatch optimized to max failure", "n_params_perturbed": "all"},
}

# Standard noise levels
STANDARD_NOISE_LEVELS = {
    "clean": {"label": "Clean", "snr_db": 60},
    "low": {"label": "Low noise", "snr_db": 40},
    "medium": {"label": "Medium noise", "snr_db": 30},
    "high": {"label": "High noise", "snr_db": 20},
}

# Standard image sizes by category
STANDARD_SIZES = {
    "Compressive Imaging": {
        "small": {"x_shape": [128, 128], "label": "Small"},
        "standard": {"x_shape": [256, 256], "label": "Standard"},
        "large": {"x_shape": [512, 512], "label": "Large"},
    },
    "Microscopy": {
        "small": {"x_shape": [128, 128], "label": "Small"},
        "standard": {"x_shape": [256, 256], "label": "Standard"},
        "large": {"x_shape": [512, 512], "label": "Large"},
        "xlarge": {"x_shape": [1024, 1024], "label": "XLarge"},
    },
    "Medical Imaging": {
        "small": {"x_shape": [256, 256], "label": "Small"},
        "standard": {"x_shape": [512, 512], "label": "Standard"},
    },
    "default": {
        "small": {"x_shape": [128, 128], "label": "Small"},
        "standard": {"x_shape": [256, 256], "label": "Standard"},
        "large": {"x_shape": [512, 512], "label": "Large"},
    },
}


def generate_expanded_config(modality_id: str) -> dict | None:
    """Generate expanded config for a modality.

    Reads the existing single config, enriches with variants from
    VARIANT_REGISTRY, and adds standard noise/mismatch levels.
    """
    single_config_path = CONFIGS_DIR / f"{modality_id}.yaml"
    if not single_config_path.exists():
        logger.warning(f"No single config for {modality_id}, skipping")
        return None

    with open(single_config_path) as f:
        base = yaml.safe_load(f)

    if not base:
        return None

    category = base.get("category", "default")

    # Build expanded config
    expanded = {
        "modality_id": modality_id,
        "display_name": base.get("display_name", modality_id),
        "category": category,
        "carrier": base.get("carrier", "Photon"),
        "maturity": base.get("maturity", "M0"),
    }

    # Variants
    if modality_id in VARIANT_REGISTRY:
        variants = {}
        for vid, vname, dag, elements, ref in VARIANT_REGISTRY[modality_id]:
            variants[vid] = {
                "id": vid,
                "name": vname,
                "dag": dag,
                "optical_elements": elements,
                "reference": ref,
            }
        expanded["variants"] = variants
    else:
        # Create a single "standard" variant from the base config
        expanded["variants"] = {
            "standard": {
                "id": "standard",
                "name": base.get("display_name", modality_id),
                "dag": base.get("canonical_dag", ""),
                "optical_elements": [],
                "reference": "",
            }
        }

    # Image sizes
    sizes = STANDARD_SIZES.get(category, STANDARD_SIZES["default"])
    expanded["image_sizes"] = sizes

    # Noise levels
    expanded["noise_levels"] = STANDARD_NOISE_LEVELS

    # Mismatch params (from base config)
    expanded["mismatch_params"] = base.get("mismatch_params", [])

    # Mismatch levels
    expanded["mismatch_levels"] = STANDARD_MISMATCH_LEVELS

    # Data sources
    ds = base.get("data_source", {})
    sources = []
    if ds.get("dataset_url"):
        sources.append({
            "id": ds.get("dataset_id", f"{modality_id}_web"),
            "type": "web",
            "label": "WEB",
            "url": ds["dataset_url"],
            "citation": ds.get("citation", ""),
            "license": ds.get("license", ""),
            "applies_to": "all_variants",
        })
    sources.append({
        "id": f"{modality_id}_generated",
        "type": "generated",
        "label": "GEN",
        "description": f"Synthetically generated {modality_id} data",
        "applies_to": "all_variants",
    })
    expanded["data_sources"] = sources

    # Compute totals
    n_v = len(expanded["variants"])
    n_s = len(expanded["image_sizes"])
    n_n = len(expanded["noise_levels"])
    n_m = len(expanded["mismatch_levels"])
    b1 = n_v * 4 * 3  # variants x difficulties x rounds
    b234 = n_v * n_s * n_n * n_m
    expanded["total_cases"] = {
        "B1": b1,
        "B2": b234,
        "B3": b234,
        "B4": b234,
        "grand_total": b1 + 3 * b234,
    }

    return expanded


def main():
    parser = argparse.ArgumentParser(description="Generate expanded benchmark configs")
    parser.add_argument("--modality", type=str, help="Single modality (default: all in VARIANT_REGISTRY)")
    parser.add_argument("--all-configs", action="store_true", help="Generate for all 168 modalities (even without variants)")
    args = parser.parse_args()

    if args.modality:
        modality_ids = [args.modality]
    elif args.all_configs:
        modality_ids = sorted(p.stem for p in CONFIGS_DIR.glob("*.yaml") if p.stem != "_template")
    else:
        modality_ids = sorted(VARIANT_REGISTRY.keys())

    generated = 0
    total_cases = 0

    for mid in modality_ids:
        # Skip if already has a hand-written expanded config
        expanded_path = EXPANDED_DIR / f"{mid}_expanded.yaml"
        if expanded_path.exists():
            logger.info(f"  {mid}: already exists (hand-written), skipping")
            continue

        expanded = generate_expanded_config(mid)
        if expanded is None:
            continue

        with open(expanded_path, "w") as f:
            yaml.dump(expanded, f, default_flow_style=False, sort_keys=False, width=120)

        n = expanded["total_cases"]["grand_total"]
        total_cases += n
        generated += 1
        logger.info(f"  {mid}: {len(expanded['variants'])} variants, {n} cases -> {expanded_path.name}")

    logger.info(f"\nGenerated {generated} expanded configs, {total_cases} total cases")


if __name__ == "__main__":
    main()
