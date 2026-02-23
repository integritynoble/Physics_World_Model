"""
Default Experimental Setups — popular benchmark configurations for all 64 modalities.

Each modality gets a standard experimental setup drawn from canonical papers
and benchmark datasets. These are auto-populated when a run is created
so users get realistic, publication-grade configurations out of the box.
"""

from __future__ import annotations

# ── Default experimental setups for all 64 modalities ─────────────────────
# Each value is a dict of standard experimental parameters.
# Sources: fastMRI, CAVE, KITTI, HCP, PICMUS, LoDoPaB-CT, etc.

DEFAULT_EXPERIMENT_SPEC: dict[str, dict] = {

    # ══════════════════════════════════════════════════════════════════════
    # COMPRESSIVE
    # ══════════════════════════════════════════════════════════════════════

    "cassi": {
        "setup_name": "CASSI Hyperspectral Snapshot",
        "reference": "CAVE / KAIST benchmark (MST++, TSA-Net)",
        "spectral_bands": 28,
        "spatial_resolution": "256x256",
        "wavelength_range_nm": "450-650",
        "compression_ratio": 28,
        "coded_aperture": "binary random mask",
        "disperser": "single-disperser (SD-CASSI)",
        "reconstruction": "MST / GAP-TV",
        "dataset": "CAVE (32 scenes), KAIST (30 scenes)",
    },
    "cacti": {
        "setup_name": "CACTI Video Compressive Sensing",
        "reference": "PnP-SCI, STFormer, EfficientSCI",
        "frames_per_snapshot": 8,
        "spatial_resolution": "256x256",
        "compression_ratio": 8,
        "coded_aperture": "shifting binary mask",
        "equivalent_fps": 1200,
        "reconstruction": "GAP-TV / PnP-FFDNet",
        "dataset": "Kobe, Runner, Drop, Traffic (grayscale)",
    },
    "spc": {
        "setup_name": "Single-Pixel Camera",
        "reference": "Rice SPC, InverseNet-SPC",
        "spatial_resolution": "64x64",
        "sampling_ratio": 0.25,
        "sensing_matrix": "Walsh-Hadamard",
        "detector": "single Si photodiode",
        "modulator": "DMD (digital micromirror device)",
        "reconstruction": "PnP-FISTA / ISTA-Net",
        "dataset": "MNIST, Set11, BSD68",
    },
    "matrix": {
        "setup_name": "Generic Compressive Matrix Sensing",
        "reference": "Compressed sensing theory (Candes & Tao)",
        "matrix_size": "256x256",
        "sampling_ratio": 0.25,
        "sensing_matrix": "Gaussian random",
        "rank_assumption": "low-rank",
        "reconstruction": "FISTA-L2 / ADMM",
    },

    # ══════════════════════════════════════════════════════════════════════
    # MEDICAL — CT / CBCT
    # ══════════════════════════════════════════════════════════════════════

    "ct": {
        "setup_name": "Sparse-View / Low-Dose CT",
        "reference": "LoDoPaB-CT (Scientific Data 2021), AAPM Low-Dose CT Challenge",
        "image_size": "512x512",
        "num_views": 60,
        "full_dose_views": 1000,
        "detector_pixels": 736,
        "kVp": 120,
        "dose_level": "25% of full dose (quarter-dose)",
        "reconstruction": "FBP + DL denoising / end-to-end",
        "dataset": "LoDoPaB-CT, DeepLesion",
    },
    "cbct": {
        "setup_name": "Cone-Beam CT",
        "reference": "ICASSP 2024 CBCT Challenge",
        "image_size": "512x512",
        "projection_views": 360,
        "sparse_views": 20,
        "detector_size": "150x150 px",
        "pixel_pitch_mm": 0.4,
        "kVp": 90,
        "tube_current_mA": 8,
        "voxel_size_mm": 0.3,
        "reconstruction": "FDK / iterative",
    },

    # ══════════════════════════════════════════════════════════════════════
    # MEDICAL — MRI family
    # ══════════════════════════════════════════════════════════════════════

    "mri": {
        "setup_name": "Accelerated MRI Reconstruction",
        "reference": "fastMRI (Zbontar et al., Radiology AI 2019)",
        "anatomy": "knee / brain",
        "matrix_size": "320x320",
        "field_strength_T": 3.0,
        "receive_coils": 15,
        "acceleration_factor": 4,
        "k_space_sampling": "variable-density random Cartesian",
        "center_fraction": 0.08,
        "sequence": "TSE (turbo spin echo)",
        "reconstruction": "SENSE / E2E-VarNet",
        "dataset": "fastMRI (knee: 1594 volumes, brain: 6970 volumes)",
    },
    "fmri": {
        "setup_name": "Functional MRI (BOLD)",
        "reference": "Human Connectome Project (HCP) 3T protocol",
        "field_strength_T": 3.0,
        "voxel_size_mm": "2x2x2",
        "TR_s": 0.72,
        "TE_ms": 33.1,
        "matrix_size": "104x90",
        "slices": 72,
        "multiband_factor": 8,
        "sequence": "gradient-echo EPI",
        "paradigm": "resting-state / task-based",
        "dataset": "HCP 3T (1200 subjects)",
    },
    "diffusion_mri": {
        "setup_name": "Diffusion Tensor Imaging (DTI)",
        "reference": "HCP diffusion protocol",
        "field_strength_T": 3.0,
        "b_values": [0, 1000],
        "diffusion_directions": 64,
        "matrix_size": "128x128",
        "TR_ms": 4000,
        "TE_ms": 80,
        "voxel_size_mm": "2x2x2",
        "sequence": "single-shot EPI DWI",
        "reconstruction": "weighted least squares / RESTORE",
        "dataset": "HCP, UK Biobank",
    },
    "mrs": {
        "setup_name": "Single-Voxel MR Spectroscopy",
        "reference": "MRSinMRS consensus (NMR Biomed 2021)",
        "field_strength_T": 3.0,
        "sequence": "PRESS",
        "TE_ms": 30,
        "TR_ms": 2000,
        "voxel_size_cm3": "2x2x2 (8 mL)",
        "transients": 64,
        "metabolites": ["NAA", "Cr", "Cho", "Glx", "mI"],
        "fitting": "LCModel",
    },

    # ══════════════════════════════════════════════════════════════════════
    # MEDICAL — Nuclear
    # ══════════════════════════════════════════════════════════════════════

    "pet": {
        "setup_name": "FDG-PET Whole-Body",
        "reference": "GE Discovery MI / Siemens Biograph mCT",
        "matrix_size": "256x256",
        "reconstruction": "3D TOF-OSEM",
        "iterations": 3,
        "subsets": 17,
        "post_filter_fwhm_mm": 5.0,
        "isotope": "18F-FDG",
        "scan_duration_min": 10,
        "administered_activity_MBq": 370,
        "dataset": "TCIA, AutoPET Challenge",
    },
    "spect": {
        "setup_name": "Tc-99m Myocardial Perfusion SPECT",
        "reference": "EJNMMI Physics 2021 optimization",
        "matrix_size": "64x64",
        "projections": 64,
        "reconstruction": "OSEM (AC+SC+RR)",
        "iterations": 8,
        "subsets": 8,
        "post_filter_fwhm_mm": 8.0,
        "isotope": "99mTc",
        "application": "myocardial perfusion imaging",
        "acquisition_time_per_view_s": 20,
    },

    # ══════════════════════════════════════════════════════════════════════
    # MEDICAL — Ultrasound
    # ══════════════════════════════════════════════════════════════════════

    "ultrasound": {
        "setup_name": "Ultrafast Plane-Wave Compounding",
        "reference": "PICMUS Challenge (IEEE IUS)",
        "probe": "128-element linear array",
        "center_frequency_MHz": 5.21,
        "plane_wave_angles": 11,
        "compound_frame_rate_Hz": 1000,
        "imaging_depth_mm": 40,
        "speed_of_sound_m_s": 1540,
        "lateral_resolution_mm": 0.3,
        "axial_resolution_mm": 0.15,
        "dataset": "PICMUS",
    },
    "doppler_ultrasound": {
        "setup_name": "Color Doppler Flow Imaging",
        "reference": "Clinical Doppler standards",
        "probe_frequency_MHz": 5.0,
        "PRF_kHz": 10.0,
        "ensemble_length": 16,
        "wall_filter": "polynomial regression / SVD clutter filter",
        "velocity_range_cm_s": "0-200",
        "application": "carotid / renal flow",
    },
    "elastography": {
        "setup_name": "Supersonic Shear-Wave Elastography",
        "reference": "Supersonic Imagine Aixplorer",
        "probe_frequency_MHz": 4.0,
        "push_frequency_Hz": 50,
        "shear_wave_speed_m_s": "1-5 (tissue-dependent)",
        "method": "ARFI / supersonic shear imaging",
        "stiffness_range_kPa": "1-75",
        "application": "liver fibrosis staging",
    },

    # ══════════════════════════════════════════════════════════════════════
    # MEDICAL — X-ray family
    # ══════════════════════════════════════════════════════════════════════

    "fluoroscopy": {
        "setup_name": "Interventional Fluoroscopy",
        "reference": "C-arm flat-panel detector standard",
        "image_size": "1024x1024",
        "kVp": 70,
        "frame_rate_fps": 15,
        "dose_per_frame_uGy": 1.0,
        "detector_size_cm": "30x30",
        "detector_type": "flat-panel (CsI + aSi)",
    },
    "angiography": {
        "setup_name": "Digital Subtraction Angiography (DSA)",
        "reference": "Clinical DSA standard",
        "image_size": "1024x1024",
        "kVp": 80,
        "frame_rate_fps": 30,
        "contrast_agent": "iodinated (Iopamidol)",
        "injection_rate_mL_s": 4.0,
        "detector": "flat-panel (CsI)",
        "application": "cerebral / coronary",
    },
    "xray_radiography": {
        "setup_name": "Digital Chest Radiography",
        "reference": "ASRT best-practices; CheXpert dataset",
        "image_size": "2048x2048",
        "pixel_pitch_mm": 0.1,
        "kVp": 120,
        "mAs": 4.0,
        "SID_cm": 180,
        "detector": "flat-panel (CsI + aSi TFT)",
        "dataset": "CheXpert, MIMIC-CXR, NIH ChestX-ray14",
    },
    "mammography": {
        "setup_name": "Full-Field Digital Mammography (FFDM)",
        "reference": "VinDr-Mammo (Scientific Data 2023), CBIS-DDSM",
        "image_size": "2294x1914",
        "kVp": 28,
        "target_filter": "W/Rh",
        "mAs": 60,
        "detector": "flat-panel amorphous selenium",
        "dataset": "VinDr-Mammo, CBIS-DDSM, INbreast",
    },
    "dexa": {
        "setup_name": "Dual-Energy X-ray Absorptiometry",
        "reference": "Hologic Discovery / GE Lunar iDXA",
        "energies_kVp": [70, 140],
        "pixel_size_mm": 0.5,
        "scan_time_s": 30,
        "dose_uSv": 1,
        "output": "BMD (g/cm2), T-score",
        "sites": "lumbar spine, proximal femur",
    },

    # ══════════════════════════════════════════════════════════════════════
    # MEDICAL — Optical / Diffuse
    # ══════════════════════════════════════════════════════════════════════

    "dot": {
        "setup_name": "CW Frequency-Multiplexed DOT",
        "reference": "Frontiers in Physics multi-wavelength DOT",
        "wavelengths_nm": [685, 785, 830],
        "source_positions": 20,
        "detector_positions": 32,
        "total_SD_pairs": 640,
        "geometry": "circular array (breast) / reflectance (brain)",
        "reconstruction_grid_mm": "1x1x1 voxels, 50x50x30 mm volume",
        "modulation": "continuous-wave",
        "reconstruction": "Born approximation / diffusion model",
    },
    "photoacoustic": {
        "setup_name": "Linear-Array Photoacoustic Tomography",
        "reference": "OADAT dataset (J Biophotonics 2024)",
        "laser_wavelengths_nm": [700, 800, 900],
        "pulse_duration_ns": 6,
        "pulse_repetition_Hz": 10,
        "transducer": "128-element linear array",
        "center_frequency_MHz": 7.5,
        "bandwidth_percent": 80,
        "element_pitch_mm": 0.2,
        "sampling_rate_MHz": 31.2,
        "reconstruction_grid_um": 30,
        "reconstruction": "DAS / model-based",
        "dataset": "OADAT",
    },

    # ══════════════════════════════════════════════════════════════════════
    # COHERENT
    # ══════════════════════════════════════════════════════════════════════

    "ptychography": {
        "setup_name": "Synchrotron X-ray Ptychography",
        "reference": "Diamond / APS / ESRF beamlines",
        "photon_energy_keV": 12.4,
        "wavelength_nm": 0.1,
        "detector": "512x512 Eiger / Pilatus",
        "pixel_size_um": 75,
        "probe_size_um": 1.0,
        "step_size_nm": 200,
        "overlap_ratio": 0.70,
        "propagation_distance_m": 2.1,
        "achieved_resolution_nm": 10,
        "reconstruction": "ePIE / difference map",
    },
    "holography": {
        "setup_name": "Off-Axis Digital Holographic Microscopy",
        "reference": "Lyncee Tec DHM / quantitative phase imaging",
        "wavelength_nm": 532,
        "pixel_size_um": 3.45,
        "sensor": "sCMOS 2048x2048",
        "propagation_distance_mm": 100,
        "coherence_length_mm": ">1 (laser source)",
        "reconstruction": "angular spectrum method",
        "application": "quantitative phase imaging",
    },
    "phase_retrieval": {
        "setup_name": "Coherent Diffractive Imaging (CDI)",
        "reference": "Miao / Zuo group; Science 1999",
        "accelerating_voltage_kV": 300,
        "wavelength_pm": 1.97,
        "detector": "2k x 2k CCD / direct electron",
        "oversampling_ratio": 4,
        "reconstruction": "HIO + ER (hybrid input-output + error reduction)",
        "resolution_nm": 2,
    },

    # ══════════════════════════════════════════════════════════════════════
    # MICROSCOPY
    # ══════════════════════════════════════════════════════════════════════

    "widefield": {
        "setup_name": "Widefield Epifluorescence Microscopy",
        "reference": "Standard fluorescence deconvolution setup",
        "objective_NA": 1.4,
        "magnification": "60x",
        "pixel_size_nm": 65,
        "excitation_nm": 488,
        "emission_nm": 520,
        "exposure_ms": 100,
        "sensor": "sCMOS 2048x2048",
        "reconstruction": "Richardson-Lucy deconvolution",
    },
    "widefield_lowdose": {
        "setup_name": "Low-Dose Widefield Fluorescence Imaging",
        "reference": "Photon-limited microscopy benchmarks",
        "objective_NA": 1.4,
        "magnification": "60x",
        "pixel_size_nm": 65,
        "excitation_nm": 488,
        "emission_nm": 520,
        "exposure_ms": 5,
        "photon_budget": "50-200 photons/pixel",
        "sensor": "sCMOS 2048x2048",
        "reconstruction": "PnP-HQS / Noise2Void",
    },
    "confocal_livecell": {
        "setup_name": "Confocal Live-Cell Time-Lapse",
        "reference": "Zeiss LSM 880 / Nikon A1R standard",
        "objective_NA": 1.4,
        "magnification": "63x",
        "pixel_size_nm": 80,
        "excitation_nm": 488,
        "pinhole_AU": 1.0,
        "dwell_time_us": 2,
        "frame_interval_s": 5,
        "time_points": 200,
        "image_size": "512x512",
    },
    "confocal_3d": {
        "setup_name": "Confocal 3D Z-Stack",
        "reference": "Zeiss LSM 880 / Leica SP8 standard",
        "objective_NA": 1.4,
        "magnification": "63x",
        "pixel_size_nm": 80,
        "excitation_nm": 488,
        "pinhole_AU": 1.0,
        "dwell_time_us": 8,
        "lateral_resolution_nm": 180,
        "z_step_nm": 300,
        "z_slices": 64,
        "image_size": "512x512",
        "reconstruction": "Richardson-Lucy 3D deconvolution",
    },
    "sim": {
        "setup_name": "2D-SIM Structured Illumination",
        "reference": "Zeiss Elyra 7 / canonical 3-angle 5-phase SIM",
        "objective_NA": 1.49,
        "magnification": "100x",
        "pixel_size_nm": 32,
        "excitation_nm": 488,
        "orientations": 3,
        "phases_per_orientation": 5,
        "raw_images": 15,
        "achieved_resolution_nm": 110,
        "reconstruction": "Wiener-SIM / fairSIM",
    },
    "lightsheet": {
        "setup_name": "Light-Sheet Fluorescence (SPIM)",
        "reference": "Zeiss Lightsheet 7 / OpenSPIM standard",
        "detection_NA": 0.8,
        "illumination_NA": 0.1,
        "magnification": "16x",
        "pixel_size_nm": 406,
        "sheet_thickness_um": 5,
        "excitation_nm": 488,
        "frame_rate_fps": 10,
        "sample": "zebrafish embryo / cleared tissue",
        "reconstruction": "deconvolution + destriping",
    },
    "two_photon": {
        "setup_name": "Two-Photon Deep-Tissue Imaging",
        "reference": "Denk et al., Science 1990; in vivo neuroscience",
        "objective_NA": 0.95,
        "magnification": "20x (water immersion)",
        "pixel_size_nm": 330,
        "excitation_nm": 920,
        "pulse_width_fs": 100,
        "repetition_rate_MHz": 80,
        "dwell_time_us": 2,
        "imaging_depth_um": 500,
        "laser": "Ti:Sapphire (Coherent Chameleon)",
    },
    "sted": {
        "setup_name": "STED Super-Resolution Nanoscopy",
        "reference": "Abberior STEDYCON / Leica TCS SP8 STED",
        "objective_NA": 1.4,
        "magnification": "100x (oil immersion)",
        "pixel_size_nm": 20,
        "excitation_nm": 640,
        "STED_depletion_nm": 775,
        "STED_power_mW": 200,
        "achieved_resolution_nm": 50,
        "dwell_time_us": 20,
    },
    "tirf": {
        "setup_name": "TIRF Evanescent-Wave Microscopy",
        "reference": "Nikon Eclipse Ti2 TIRF / Olympus cellTIRF",
        "objective_NA": 1.49,
        "magnification": "100x (oil immersion)",
        "pixel_size_nm": 65,
        "excitation_nm": 488,
        "evanescent_depth_nm": 100,
        "exposure_ms": 30,
        "frame_rate_fps": 33,
    },
    "flim": {
        "setup_name": "TCSPC Fluorescence Lifetime Imaging",
        "reference": "Becker & Hickl SPC-150N / PicoQuant",
        "objective_NA": 1.3,
        "pixel_size_nm": 100,
        "image_size": "256x256",
        "TCSPC_channels": 256,
        "time_resolution_ps": 50,
        "IRF_FWHM_ps": 25,
        "repetition_rate_MHz": 40,
        "lifetime_range_ns": "0.5-10",
        "analysis": "phasor / bi-exponential fit",
    },
    "fpm": {
        "setup_name": "Fourier Ptychographic Microscopy",
        "reference": "Zheng et al., Nature Photonics 2013",
        "objective_NA": 0.13,
        "magnification": "4x",
        "synthetic_NA": 0.50,
        "LED_array": "15x15",
        "num_images": 225,
        "pixel_size_um": 1.56,
        "wavelength_nm": 530,
        "illumination_NA_max": 0.36,
        "reconstruction": "sequential phase retrieval / DPC",
    },
    "palm_storm": {
        "setup_name": "SMLM Super-Resolution (PALM/STORM)",
        "reference": "Nat. Rev. Methods Primers 2021; ThunderSTORM",
        "objective_NA": 1.49,
        "magnification": "100x",
        "camera_pixel_nm": 100,
        "reconstruction_pixel_nm": 25,
        "excitation_nm": 640,
        "activation_nm": 405,
        "exposure_ms": 20,
        "total_frames": 10000,
        "frame_rate_fps": 50,
        "achieved_resolution_nm": 20,
        "reconstruction": "ThunderSTORM / DECODE",
    },
    "polarization": {
        "setup_name": "Mueller Matrix Polarization Microscopy",
        "reference": "Mehta et al., Opt. Express; LC-PolScope",
        "objective_NA": 1.3,
        "magnification": "60x",
        "pixel_size_nm": 110,
        "polarization_states": 4,
        "wavelength_nm": 546,
        "analyzer": "rotating quarter-wave + linear polarizer",
        "application": "birefringence / collagen mapping",
    },
    "lensless": {
        "setup_name": "DiffuserCam Lensless Imaging",
        "reference": "Antipa et al., Optica 2018; FlatCam",
        "sensor": "2592x1944 CMOS",
        "pixel_pitch_um": 1.67,
        "diffuser_to_sensor_mm": 2.5,
        "diffuser_type": "optical diffuser / coded mask",
        "reconstruction": "ADMM / FISTA",
    },

    # ══════════════════════════════════════════════════════════════════════
    # ELECTRON MICROSCOPY
    # ══════════════════════════════════════════════════════════════════════

    "sem": {
        "setup_name": "Scanning Electron Microscopy",
        "reference": "FEI / JEOL standard SE2 imaging",
        "accelerating_voltage_kV": 10,
        "beam_current_nA": 0.54,
        "working_distance_mm": 10,
        "pixel_size_nm": 7.1,
        "magnification": "20,000x",
        "detector": "SE2 (secondary electron)",
        "image_size": "1024x768",
    },
    "tem": {
        "setup_name": "High-Resolution TEM (HRTEM)",
        "reference": "JEOL JEM-3000F / FEI Titan",
        "accelerating_voltage_kV": 300,
        "pixel_size_pm": 50,
        "magnification": "1,000,000x",
        "detector": "Gatan K3 direct electron (4k x 4k)",
        "spatial_resolution_pm": 50,
        "dose_e_per_A2": 30,
        "reconstruction": "CTF correction",
    },
    "stem": {
        "setup_name": "HAADF-STEM Imaging",
        "reference": "Aberration-corrected STEM (Nion / JEOL)",
        "accelerating_voltage_kV": 200,
        "convergence_angle_mrad": 9.6,
        "beam_current_pA": 10,
        "pixel_size_nm": 0.07,
        "HAADF_inner_angle_mrad": 80,
        "HAADF_outer_angle_mrad": 200,
        "image_size": "512x512",
    },
    "electron_tomography": {
        "setup_name": "STEM Electron Tomography",
        "reference": "Nat. Scientific Data nanomaterial tomo",
        "accelerating_voltage_kV": 200,
        "tilt_range_deg": "[-70, +70]",
        "tilt_increment_deg": 2.0,
        "detector": "HAADF-STEM",
        "pixel_size_nm": 0.71,
        "total_dose_e_per_nm2": 39000,
        "reconstruction": "SIRT / WBP (weighted back-projection)",
    },
    "electron_diffraction": {
        "setup_name": "4D-STEM Scanning Diffraction",
        "reference": "Ophus, Microscopy & Microanalysis 2019",
        "accelerating_voltage_kV": 200,
        "convergence_angle_mrad": 1.5,
        "step_size_nm": 1.0,
        "detector": "Medipix3 / Merlin (256x256 px)",
        "exposure_ms": 1,
        "camera_length_mm": 580,
        "reconstruction": "ptychographic phase retrieval / WDD",
    },
    "electron_holography": {
        "setup_name": "Off-Axis Electron Holography",
        "reference": "Dunin-Borkowski et al., electron biprism",
        "accelerating_voltage_kV": 300,
        "wavelength_pm": 1.97,
        "detector": "Gatan Orius CCD (2k x 2k)",
        "biprism_voltage_V": 150,
        "exposure_s": 2,
        "fringe_spacing_nm": 0.3,
        "reconstruction": "Fourier filtering + phase unwrapping",
        "application": "magnetic / electric field mapping",
    },
    "ebsd": {
        "setup_name": "Electron Backscatter Diffraction",
        "reference": "Oxford Symmetry / EDAX Hikari",
        "accelerating_voltage_kV": 20,
        "sample_tilt_deg": 70,
        "step_size_um": 0.5,
        "camera_resolution": "446x446",
        "exposure_ms": 10,
        "indexing": "Hough transform / dictionary indexing",
        "output": "grain orientation map (IPF), misorientation",
    },
    "eels": {
        "setup_name": "STEM-EELS Spectrum Imaging",
        "reference": "Gatan Quantum GIF / Enfinium",
        "accelerating_voltage_kV": 100,
        "energy_resolution_eV": 0.3,
        "dispersion_eV_per_ch": 0.1,
        "collection_angle_mrad": 30,
        "dwell_time_ms": 50,
        "spectrum_range": "core-loss + low-loss",
        "analysis": "elemental mapping, ELNES fine structure",
    },

    # ══════════════════════════════════════════════════════════════════════
    # CLINICAL OPTICS
    # ══════════════════════════════════════════════════════════════════════

    "oct": {
        "setup_name": "Spectral-Domain OCT (Retinal)",
        "reference": "Zeiss Cirrus HD-OCT / Heidelberg Spectralis",
        "wavelength_nm": 840,
        "bandwidth_nm": 45,
        "axial_resolution_um": 5,
        "lateral_resolution_um": 15,
        "A_scan_rate_kHz": 40,
        "scan_width_mm": 6.0,
        "B_scan_lines": 512,
        "A_scans_per_B": 512,
        "SNR_dB": 98,
    },
    "octa": {
        "setup_name": "OCT Angiography (Retinal Vasculature)",
        "reference": "Zeiss AngioPlex / Optovue AngioVue",
        "wavelength_nm": 840,
        "A_scan_rate_kHz": 68,
        "scan_pattern": "6x6 mm",
        "repeated_B_scans": 4,
        "en_face_resolution_um": 15,
        "algorithm": "SSADA / OCTA ratio",
    },
    "fundus": {
        "setup_name": "Digital Fundus Photography",
        "reference": "EyePACS / DRIVE / MESSIDOR datasets",
        "image_size": "2124x2056",
        "field_of_view_deg": 45,
        "wavelength_range_nm": "500-700",
        "flash_exposure_ms": 0.04,
        "dataset": "EyePACS, DRIVE, MESSIDOR, APTOS",
    },
    "endoscopy": {
        "setup_name": "White-Light Fiber Endoscopy",
        "reference": "Clinical GI endoscopy standard",
        "resolution": "1920x1080 (HD)",
        "frame_rate_fps": 60,
        "field_of_view_deg": 140,
        "working_channel_mm": 3.7,
        "wavelength_range_nm": "400-700 (white light)",
        "dataset": "Kvasir, CVC-ClinicDB, HyperKvasir",
    },

    # ══════════════════════════════════════════════════════════════════════
    # COMPUTATIONAL / COMPUTATIONAL PHOTOGRAPHY
    # ══════════════════════════════════════════════════════════════════════

    "light_field": {
        "setup_name": "Micro-Lens Array Light Field Camera",
        "reference": "HCI Light Field Benchmark / Stanford Archive",
        "micro_lens_pitch_um": 14,
        "angular_resolution": "9x9 (HCI) / 15x15 (Lytro Illum)",
        "total_sensor_px": "7728x5368",
        "spatial_per_view": "434x625",
        "dataset": "HCI 4D LF Benchmark, Stanford Lego Gantry",
    },
    "integral": {
        "setup_name": "Integral Photography / Fly-Eye Lens",
        "reference": "Lippmann integral photography; micro-lens arrays",
        "micro_lens_pitch_mm": 1.0,
        "micro_lens_NA": 0.16,
        "sensor_pixel_um": 5.5,
        "pixels_per_lens": "20x20",
        "reconstruction": "3D focal-stack / refocusing",
    },
    "panorama": {
        "setup_name": "Multi-Focus Panoramic Fusion",
        "reference": "Focus stacking / extended DOF",
        "image_size": "4096x2048 (equirectangular)",
        "focus_planes": 6,
        "overlap_percent": 30,
        "fusion": "Laplacian pyramid / wavelet fusion",
        "application": "all-in-focus / extended depth of field",
    },

    # ══════════════════════════════════════════════════════════════════════
    # NEURAL RENDERING
    # ══════════════════════════════════════════════════════════════════════

    "nerf": {
        "setup_name": "NeRF Novel View Synthesis",
        "reference": "Mildenhall et al., ECCV 2020; NeRF Blender Synthetic",
        "training_views": 100,
        "test_views": 200,
        "image_resolution": "800x800",
        "scene_type": "object-centric 360 deg (Blender synthetic)",
        "evaluation": "PSNR / SSIM / LPIPS",
        "architecture": "positional encoding + MLP",
        "dataset": "Blender Synthetic (8 scenes), LLFF (8 forward-facing)",
    },
    "gaussian_splatting": {
        "setup_name": "3D Gaussian Splatting",
        "reference": "Kerbl et al., SIGGRAPH 2023",
        "training_views": "24-300 (scene-dependent)",
        "image_resolution": "~1600x1200",
        "initialization": "SfM point cloud (COLMAP)",
        "rendering_fps": 30,
        "scene_type": "unbounded indoor / outdoor",
        "evaluation": "PSNR / SSIM / LPIPS",
        "dataset": "Mip-NeRF360, Tanks & Temples, Deep Blending",
    },

    # ══════════════════════════════════════════════════════════════════════
    # DEPTH IMAGING
    # ══════════════════════════════════════════════════════════════════════

    "tof_camera": {
        "setup_name": "Consumer Time-of-Flight Depth Camera",
        "reference": "Intel RealSense L515 / Microsoft Azure Kinect",
        "depth_resolution": "640x480",
        "range_m": "0.1-6.0",
        "frame_rate_fps": 30,
        "wavelength_nm": 850,
        "depth_accuracy_mm": 2.0,
        "modulation": "AMCW (amplitude-modulated continuous wave)",
    },
    "structured_light": {
        "setup_name": "Structured-Light 3D Scanning",
        "reference": "Intel RealSense SR305 / Kinect v1",
        "pattern": "pseudorandom IR dot pattern",
        "wavelength_nm": 850,
        "range_m": "0.2-1.5",
        "depth_resolution": "640x480",
        "accuracy_mm": 1.0,
        "frame_rate_fps": 30,
    },
    "lidar": {
        "setup_name": "Automotive LiDAR (KITTI Benchmark)",
        "reference": "Velodyne HDL-64E on KITTI",
        "channels": 64,
        "range_m": 120,
        "horizontal_FOV_deg": 360,
        "vertical_FOV_deg": 27,
        "horizontal_resolution_deg": 0.08,
        "vertical_resolution_deg": 0.4,
        "rotation_rate_Hz": 10,
        "wavelength_nm": 905,
        "points_per_second": 2200000,
        "dataset": "KITTI, nuScenes, Waymo Open",
    },

    # ══════════════════════════════════════════════════════════════════════
    # REMOTE SENSING
    # ══════════════════════════════════════════════════════════════════════

    "sar": {
        "setup_name": "Sentinel-1 IW Mode SAR",
        "reference": "ESA Copernicus / Google Earth Engine",
        "frequency_band": "C-band (5.405 GHz)",
        "wavelength_cm": 5.6,
        "mode": "IW (Interferometric Wide Swath)",
        "spatial_resolution_m": "5 (range) x 20 (azimuth)",
        "swath_km": 250,
        "polarization": "VV + VH (dual-pol)",
        "incidence_angle_deg": "29.1-46.0",
        "revisit_days": 6,
        "product": "GRD (ground range detected)",
    },
    "sonar": {
        "setup_name": "Side-Scan Sonar Seabed Mapping",
        "reference": "S3Simulator (2024), UATD dataset",
        "frequency_kHz": 900,
        "range_m": 100,
        "resolution_m": 0.1,
        "swath_m": 200,
        "platform": "AUV / towed body",
        "application": "seabed mapping / mine detection",
    },

    # ══════════════════════════════════════════════════════════════════════
    # PARTICLE IMAGING
    # ══════════════════════════════════════════════════════════════════════

    "neutron_tomo": {
        "setup_name": "Thermal Neutron Radiography / Tomography",
        "reference": "IAEA; PSI ICON beamline",
        "neutron_energy_eV": 0.025,
        "energy_type": "thermal",
        "detector": "LiF/ZnS scintillator + CCD",
        "pixel_size_um": 100,
        "image_size": "2048x2048",
        "exposure_s": 10,
        "flux_n_per_cm2_s": 1e8,
        "facility": "research reactor / spallation source",
    },
    "proton_radiography": {
        "setup_name": "Proton CT / Radiography",
        "reference": "Medical Physics 2024 (Liu et al.)",
        "proton_energy_MeV": 200,
        "detector": "scintillating fiber tracker + residual energy",
        "tracker_planes": 4,
        "image_matrix": "256x256",
        "projections": 360,
        "RSP_accuracy_percent": 1.0,
        "application": "treatment planning verification",
    },
    "muon_tomo": {
        "setup_name": "Cosmic-Ray Muon Scattering Tomography",
        "reference": "IAEA; Los Alamos muon radiography",
        "mean_energy_GeV": 4.0,
        "flux_per_cm2_per_min": 1.0,
        "detector": "drift tube / RPC panels",
        "position_resolution_mm": 1.0,
        "exposure_min": 60,
        "technique": "multiple scattering tomography (MST)",
        "application": "nuclear material detection / volcano imaging",
    },
}


def get_default_experiment_spec(modality: str) -> dict:
    """Return the default experimental setup for a given modality.

    Returns a copy so callers can safely mutate it.
    Returns an empty dict if the modality is unknown.
    """
    spec = DEFAULT_EXPERIMENT_SPEC.get(modality, {})
    return dict(spec)  # shallow copy
