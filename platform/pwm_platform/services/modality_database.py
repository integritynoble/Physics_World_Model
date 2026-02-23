"""
Physics World Model — Comprehensive Modality Knowledge Base.

Authoritative database of all 64 imaging modalities with full physics
descriptions, experimental setups, forward model metadata, and canonical
references.  Used to populate the ModalityBasics database table and
serve modality information through the API and web UI.
"""

from __future__ import annotations

MODALITY_DATABASE: dict[str, dict] = {

    # ══════════════════════════════════════════════════════════════════════════
    # MICROSCOPY  (14 modalities)
    # ══════════════════════════════════════════════════════════════════════════

    "widefield": {
        "display_name": "Widefield Fluorescence Microscopy",
        "category": "microscopy",
        "description": (
            "Standard widefield epi-fluorescence microscopy where the entire field "
            "of view is illuminated simultaneously and the image is formed by "
            "convolution of the specimen fluorescence distribution with the system "
            "point spread function (PSF). Out-of-focus blur from planes above and "
            "below the focal plane is the primary degradation. The forward model is "
            "y = PSF ** x + n, where ** denotes convolution and n is mixed "
            "Poisson-Gaussian noise. Deconvolution via Richardson-Lucy or learned "
            "priors (CARE) restores resolution toward the diffraction limit."
        ),
        "experimental_setup": {
            "instrument": "Nikon Eclipse Ti2-E / Zeiss Axio Observer 7",
            "objective": "Plan Apo 60x / 1.40 NA oil immersion",
            "pixel_size_nm": 65,
            "excitation_source": "Lumencor SPECTRA X LED engine (488 nm band)",
            "excitation_nm": 488,
            "emission_nm": 520,
            "exposure_ms": 100,
            "detector": "Hamamatsu ORCA-Flash4.0 V3 sCMOS (2048x2048)",
            "dichroic": "Semrock Di03-R488-t1",
            "emission_filter": "ET525/50m",
            "reconstruction": "Richardson-Lucy deconvolution",
        },
        "physics_class": "fluorescence",
        "forward_model_family": "psf_convolution",
        "wave_model": "incoherent",
        "sensor_type": "scmos",
        "source_type": "led",
        "geometry": "planar",
        "noise_model": "poisson_gaussian",
        "typical_x_dims": [512, 512],
        "typical_y_dims": [512, 512],
        "typical_snr_range": [15.0, 40.0],
        "calibration_params": [
            "psf_measurement", "emission_wavelength", "numerical_aperture",
            "pixel_size", "flatfield_correction",
        ],
        "mismatch_modes": [
            "defocus", "spherical_aberration", "refractive_index_mismatch",
            "photobleaching", "sample_drift",
        ],
        "reconstruction_task_types": ["deconvolution", "denoising"],
        "default_solver": "richardson_lucy",
        "evaluation_metrics": ["psnr", "ssim", "nrmse", "resolution_fwhm"],
        "canonical_references": [
            "Richardson, 'Bayesian-based iterative method of image restoration', "
            "J. Opt. Soc. Am. 62, 55-59 (1972)",
            "Weigert et al., 'Content-aware image restoration (CARE)', "
            "Nature Methods 15, 1090-1097 (2018)",
        ],
        "canonical_datasets": [
            "BioSR (Zhang et al., Nature Methods 2023)",
            "Hagen et al. widefield deconvolution benchmark",
        ],
        "tags": ["microscopy", "fluorescence", "deconvolution", "psf"],
    },

    "widefield_lowdose": {
        "display_name": "Low-Dose Widefield Microscopy",
        "category": "microscopy",
        "description": (
            "Widefield fluorescence microscopy operated at very low illumination "
            "power or short exposure time to reduce phototoxicity and photobleaching "
            "in live specimens. Images are dominated by shot noise (Poisson) and "
            "read noise (Gaussian) with typical photon counts of 20-200 per pixel. "
            "The forward model is y = Poisson(alpha * PSF ** x)/alpha + N(0, sigma^2) "
            "where alpha is the photon conversion factor. Reconstruction requires "
            "joint denoising and deconvolution using PnP-HQS, Noise2Void, or CARE."
        ),
        "experimental_setup": {
            "instrument": "Nikon Eclipse Ti2-E / Zeiss Axio Observer 7",
            "objective": "Plan Apo 60x / 1.40 NA oil immersion",
            "pixel_size_nm": 65,
            "excitation_source": "LED (attenuated to 2 mW, 4% power)",
            "excitation_nm": 488,
            "emission_nm": 520,
            "exposure_ms": 5,
            "photon_budget": "50-200 photons/pixel",
            "detector": "Hamamatsu ORCA-Flash4.0 V3 sCMOS",
            "reconstruction": "PnP-HQS / Noise2Void / CARE",
        },
        "physics_class": "fluorescence",
        "forward_model_family": "psf_convolution",
        "wave_model": "incoherent",
        "sensor_type": "scmos",
        "source_type": "led",
        "geometry": "planar",
        "noise_model": "poisson_gaussian",
        "typical_x_dims": [512, 512],
        "typical_y_dims": [512, 512],
        "typical_snr_range": [3.0, 15.0],
        "calibration_params": [
            "psf_measurement", "read_noise_sigma", "photon_gain_alpha",
            "dark_frame", "pixel_size",
        ],
        "mismatch_modes": [
            "noise_model_mismatch", "gain_miscalibration",
            "hot_pixels", "background_fluorescence",
        ],
        "reconstruction_task_types": ["denoising", "deconvolution"],
        "default_solver": "pnp_hqs",
        "evaluation_metrics": ["psnr", "ssim", "nrmse"],
        "canonical_references": [
            "Krull et al., 'Noise2Void - Learning Denoising from Single Noisy Images', "
            "CVPR 2019",
            "Weigert et al., 'Content-aware image restoration (CARE)', "
            "Nature Methods 15, 1090-1097 (2018)",
        ],
        "canonical_datasets": [
            "BioSR low-SNR subset",
            "Planaria / Tribolium datasets (Weigert et al.)",
        ],
        "tags": ["microscopy", "fluorescence", "low_dose", "denoising", "photon_limited"],
    },

    "confocal_livecell": {
        "display_name": "Confocal Live-Cell Microscopy",
        "category": "microscopy",
        "description": (
            "Laser scanning confocal microscopy for live-cell imaging. A focused "
            "laser scans the specimen point by point, and a pinhole rejects "
            "out-of-focus light. The image formation is modelled as convolution with "
            "the confocal PSF (product of excitation and detection PSFs). Fast "
            "acquisition rates for live cells often sacrifice SNR due to short pixel "
            "dwell times. Reconstruction involves deconvolution with the confocal PSF "
            "and temporal denoising across frames."
        ),
        "experimental_setup": {
            "instrument": "Zeiss LSM 880 / Nikon A1R HD25",
            "objective": "Plan Apo 63x / 1.40 NA oil",
            "pixel_size_nm": 80,
            "excitation_source": "488 nm Argon laser (5 mW)",
            "pinhole_AU": 1.0,
            "dwell_time_us": 2,
            "frame_interval_s": 5,
            "time_points": 200,
            "image_size": "512x512",
            "detector": "GaAsP PMT (Zeiss Airyscan or Nikon spectral detector)",
        },
        "physics_class": "fluorescence",
        "forward_model_family": "confocal_psf_convolution",
        "wave_model": "incoherent",
        "sensor_type": "pmt",
        "source_type": "laser",
        "geometry": "raster_scan",
        "noise_model": "poisson_gaussian",
        "typical_x_dims": [512, 512],
        "typical_y_dims": [512, 512],
        "typical_snr_range": [10.0, 30.0],
        "calibration_params": [
            "confocal_psf", "pinhole_diameter", "excitation_wavelength",
            "pixel_dwell_time", "laser_power",
        ],
        "mismatch_modes": [
            "pinhole_misalignment", "laser_fluctuation",
            "photobleaching", "focal_drift",
        ],
        "reconstruction_task_types": ["deconvolution", "denoising", "temporal_denoising"],
        "default_solver": "richardson_lucy",
        "evaluation_metrics": ["psnr", "ssim", "nrmse"],
        "canonical_references": [
            "Minsky, 'Memoir on inventing the confocal microscope', Scanning 10, 128-138 (1988)",
            "McNally et al., 'Three-dimensional imaging by deconvolution microscopy', "
            "Methods 23, 210-217 (1999)",
        ],
        "canonical_datasets": [
            "Cell Tracking Challenge confocal sequences",
            "BioSR confocal subset",
        ],
        "tags": ["microscopy", "confocal", "live_cell", "scanning"],
    },

    "confocal_3d": {
        "display_name": "Confocal 3D Z-Stack",
        "category": "microscopy",
        "description": (
            "Three-dimensional confocal imaging by acquiring a z-stack of optical "
            "sections. Each slice is convolved with the 3D confocal PSF. The "
            "anisotropic PSF (axial resolution ~3x worse than lateral) is a key "
            "challenge. 3D Richardson-Lucy or CARE-3D are used for volumetric "
            "deconvolution. The forward model is y(x,y,z) = PSF_3d *** x(x,y,z) + n "
            "where *** denotes 3D convolution."
        ),
        "experimental_setup": {
            "instrument": "Zeiss LSM 880 / Leica TCS SP8",
            "objective": "Plan Apo 63x / 1.40 NA oil",
            "pixel_size_nm": 80,
            "excitation_source": "561 nm DPSS laser",
            "pinhole_AU": 1.0,
            "dwell_time_us": 8,
            "z_step_nm": 300,
            "z_slices": 64,
            "lateral_resolution_nm": 180,
            "image_size": "512x512",
            "reconstruction": "Richardson-Lucy 3D deconvolution",
        },
        "physics_class": "fluorescence",
        "forward_model_family": "confocal_3d_psf_convolution",
        "wave_model": "incoherent",
        "sensor_type": "pmt",
        "source_type": "laser",
        "geometry": "volumetric_scan",
        "noise_model": "poisson_gaussian",
        "typical_x_dims": [256, 256, 64],
        "typical_y_dims": [256, 256, 64],
        "typical_snr_range": [10.0, 30.0],
        "calibration_params": [
            "3d_psf", "voxel_size", "refractive_index",
            "coverslip_thickness", "z_calibration",
        ],
        "mismatch_modes": [
            "depth_dependent_aberration", "refractive_index_mismatch",
            "z_drift", "photobleaching_with_depth",
        ],
        "reconstruction_task_types": ["3d_deconvolution", "isotropic_resolution_recovery"],
        "default_solver": "richardson_lucy_3d",
        "evaluation_metrics": ["psnr", "ssim", "nrmse", "axial_resolution"],
        "canonical_references": [
            "McNally et al., 'Three-dimensional imaging by deconvolution microscopy', "
            "Methods 23, 210-217 (1999)",
            "Weigert et al., 'Isotropic reconstruction of 3D fluorescence microscopy "
            "images using convolutional neural networks', MICCAI 2017",
        ],
        "canonical_datasets": [
            "Planaria 3D confocal dataset (Weigert et al.)",
            "BioSR confocal 3D subset",
        ],
        "tags": ["microscopy", "confocal", "3d", "z_stack", "volumetric"],
    },

    "sim": {
        "display_name": "Structured Illumination Microscopy",
        "category": "microscopy",
        "description": (
            "Structured illumination microscopy (SIM) achieves ~2x lateral resolution "
            "improvement by illuminating the sample with sinusoidal patterns at "
            "multiple orientations and phases. Frequency mixing between the "
            "illumination pattern and sample structure shifts high-frequency "
            "information into the microscope passband. Reconstruction separates and "
            "reassembles frequency components via Wiener-SIM or deep-learning SIM. "
            "The forward model is y_k = PSF ** (I_k * x) + n for each pattern k."
        ),
        "experimental_setup": {
            "instrument": "Zeiss Elyra 7 / Nikon N-SIM S",
            "objective": "Apo TIRF 100x / 1.49 NA oil",
            "pixel_size_nm": 32,
            "excitation_source": "488 nm laser (20 mW)",
            "orientations": 3,
            "phases_per_orientation": 5,
            "raw_images": 15,
            "achieved_resolution_nm": 110,
            "detector": "Hamamatsu ORCA-Flash4.0 sCMOS",
            "pattern_generator": "SLM / diffraction grating",
            "reconstruction": "Wiener-SIM / fairSIM",
        },
        "physics_class": "structured_illumination",
        "forward_model_family": "patterned_illumination_convolution",
        "wave_model": "coherent_illumination",
        "sensor_type": "scmos",
        "source_type": "laser",
        "geometry": "planar",
        "noise_model": "poisson_gaussian",
        "typical_x_dims": [512, 512],
        "typical_y_dims": [512, 512, 9],
        "typical_snr_range": [15.0, 40.0],
        "calibration_params": [
            "pattern_frequency_vectors", "otf_measurement",
            "pattern_phase_calibration", "wiener_parameter",
        ],
        "mismatch_modes": [
            "pattern_phase_error", "illumination_nonuniformity",
            "otf_mismatch", "sample_motion_between_frames",
        ],
        "reconstruction_task_types": ["super_resolution", "sim_reconstruction"],
        "default_solver": "wiener_sim",
        "evaluation_metrics": ["psnr", "ssim", "resolution_fwhm", "frc"],
        "canonical_references": [
            "Gustafsson, 'Surpassing the lateral resolution limit by a factor of two "
            "using structured illumination microscopy', J. Microsc. 198, 82-87 (2000)",
            "Muller & Bhatt, 'Open-source image reconstruction of super-resolution "
            "structured illumination microscopy data (fairSIM)', Nature Comms 7, 10980 (2016)",
        ],
        "canonical_datasets": [
            "BioSR SIM paired dataset (Zhang et al., Nature Methods 2023)",
            "fairSIM test datasets (Hagen et al.)",
        ],
        "tags": ["microscopy", "super_resolution", "structured_illumination", "frequency_mixing"],
    },

    "lightsheet": {
        "display_name": "Light-Sheet Fluorescence Microscopy",
        "category": "microscopy",
        "description": (
            "Light-sheet microscopy (LSFM / SPIM) illuminates the sample with a thin "
            "sheet of light perpendicular to the detection axis, providing intrinsic "
            "optical sectioning. Primary artifacts are stripe patterns caused by "
            "absorption and scattering in the illumination path, plus anisotropic "
            "PSF blur. The forward model is y = S(z) * (PSF_3d *** x) + n where "
            "S(z) models the stripe attenuation. Reconstruction involves destriping "
            "followed by optional deconvolution."
        ),
        "experimental_setup": {
            "instrument": "Zeiss Lightsheet 7 / LaVision BioTec UltraMicroscope II",
            "detection_objective": "Plan Apo 20x / 1.0 NA water dipping",
            "illumination_NA": 0.1,
            "pixel_size_nm": 406,
            "sheet_thickness_um": 5,
            "excitation_source": "488 nm laser (10 mW)",
            "frame_rate_fps": 10,
            "sample": "zebrafish embryo / cleared tissue",
            "detector": "Hamamatsu ORCA-Flash4.0 sCMOS",
            "reconstruction": "deconvolution + destriping",
        },
        "physics_class": "fluorescence",
        "forward_model_family": "lightsheet_psf_convolution",
        "wave_model": "incoherent",
        "sensor_type": "scmos",
        "source_type": "laser",
        "geometry": "orthogonal_illumination",
        "noise_model": "poisson_gaussian",
        "typical_x_dims": [512, 512, 128],
        "typical_y_dims": [512, 512, 128],
        "typical_snr_range": [12.0, 35.0],
        "calibration_params": [
            "sheet_thickness", "sheet_alignment", "detection_psf",
            "refractive_index_medium", "stripe_characterization",
        ],
        "mismatch_modes": [
            "sheet_misalignment", "scattering_stripes",
            "photobleaching_gradient", "clearing_artifact",
        ],
        "reconstruction_task_types": ["destriping", "deconvolution", "3d_fusion"],
        "default_solver": "fourier_notch_destripe",
        "evaluation_metrics": ["psnr", "ssim", "stripe_residual", "nrmse"],
        "canonical_references": [
            "Huisken et al., 'Optical sectioning deep inside live embryos by SPIM', "
            "Science 305, 1007-1009 (2004)",
            "Power & Bhatt, 'A guide to light-sheet fluorescence microscopy for "
            "multiscale imaging', Nature Methods 14, 360-373 (2017)",
        ],
        "canonical_datasets": [
            "OpenSPIM sample datasets",
            "Zebrafish developmental lightsheet atlas",
        ],
        "tags": ["microscopy", "lightsheet", "spim", "3d", "optical_sectioning"],
    },

    "flim": {
        "display_name": "Fluorescence Lifetime Imaging",
        "category": "microscopy",
        "description": (
            "Fluorescence lifetime imaging microscopy (FLIM) measures the exponential "
            "decay time of fluorescence emission at each pixel, providing contrast "
            "based on the molecular environment rather than intensity alone. In "
            "time-correlated single-photon counting (TCSPC), each detected photon is "
            "time-tagged relative to the excitation pulse, building a histogram of "
            "arrival times that is fitted to single- or multi-exponential decay models. "
            "The phasor approach provides a fit-free analysis in Fourier space. Primary "
            "challenges include low photon counts and instrument response function (IRF) "
            "deconvolution."
        ),
        "experimental_setup": {
            "instrument": "Becker & Hickl SPC-150N with Zeiss LSM 880",
            "objective": "Plan Apo 63x / 1.30 NA oil",
            "pixel_size_nm": 100,
            "image_size": "256x256",
            "TCSPC_channels": 256,
            "time_resolution_ps": 50,
            "IRF_FWHM_ps": 25,
            "excitation_source": "pulsed diode laser (405 nm, 40 MHz repetition)",
            "repetition_rate_MHz": 40,
            "lifetime_range_ns": "0.5-10",
            "detector": "Hybrid PMT (Becker & Hickl HPM-100-40)",
            "analysis": "phasor / bi-exponential fit",
        },
        "physics_class": "fluorescence_lifetime",
        "forward_model_family": "temporal_decay_convolution",
        "wave_model": "incoherent",
        "sensor_type": "spad_or_pmt",
        "source_type": "pulsed_laser",
        "geometry": "raster_scan",
        "noise_model": "poisson",
        "typical_x_dims": [256, 256],
        "typical_y_dims": [256, 256, 256],
        "typical_snr_range": [5.0, 25.0],
        "calibration_params": [
            "instrument_response_function", "time_channel_width",
            "repetition_rate", "detector_afterpulsing",
        ],
        "mismatch_modes": [
            "irf_drift", "pile_up_effect", "afterpulsing",
            "incomplete_decay", "autofluorescence_background",
        ],
        "reconstruction_task_types": ["lifetime_estimation", "phasor_analysis", "denoising"],
        "default_solver": "phasor",
        "evaluation_metrics": ["lifetime_accuracy_ns", "chi_squared", "phasor_g_s_error"],
        "canonical_references": [
            "Becker, 'Advanced Time-Correlated Single Photon Counting Techniques', "
            "Springer (2005)",
            "Digman et al., 'The phasor approach to fluorescence lifetime imaging', "
            "Biophysical Journal 94, L14-L16 (2008)",
        ],
        "canonical_datasets": [
            "FLIM-FRET standard sample datasets (Becker & Hickl)",
            "FLIM phasor benchmark (Digman lab)",
        ],
        "tags": ["microscopy", "flim", "lifetime", "tcspc", "phasor", "fret"],
    },

    "fpm": {
        "display_name": "Fourier Ptychographic Microscopy",
        "category": "microscopy",
        "description": (
            "Fourier ptychographic microscopy (FPM) achieves a high space-bandwidth "
            "product by illuminating the sample from multiple angles using an LED "
            "array, capturing a set of low-resolution images, and computationally "
            "stitching them in Fourier space to synthesize a high-NA image with both "
            "amplitude and phase. Each LED angle shifts the sample's spatial frequency "
            "spectrum in Fourier space, and overlapping spectral regions provide "
            "redundancy for phase retrieval. The synthetic NA equals the objective NA "
            "plus the illumination NA. Reconstruction uses iterative phase retrieval "
            "algorithms (sequential or gradient-based)."
        ),
        "experimental_setup": {
            "instrument": "Custom FPM setup / 4f relay with LED array",
            "objective": "Plan 4x / 0.13 NA (low-power, large FOV)",
            "synthetic_NA": 0.50,
            "LED_array": "15x15 (225 LEDs) programmable matrix",
            "num_images": 225,
            "pixel_size_um": 1.56,
            "wavelength_nm": 530,
            "illumination_NA_max": 0.36,
            "detector": "Thorlabs CS895MU monochrome CMOS",
            "reconstruction": "sequential phase retrieval / DPC",
        },
        "physics_class": "fourier_ptychography",
        "forward_model_family": "fourier_spectrum_stitching",
        "wave_model": "coherent",
        "sensor_type": "cmos",
        "source_type": "led_array",
        "geometry": "planar",
        "noise_model": "poisson_gaussian",
        "typical_x_dims": [2048, 2048],
        "typical_y_dims": [225, 256, 256],
        "typical_snr_range": [15.0, 35.0],
        "calibration_params": [
            "led_positions", "led_brightness_calibration",
            "aberration_recovery", "defocus_distance",
        ],
        "mismatch_modes": [
            "led_position_error", "aberration_model_error",
            "intensity_fluctuation", "sample_motion",
        ],
        "reconstruction_task_types": ["phase_retrieval", "super_resolution", "aberration_recovery"],
        "default_solver": "sequential_phase_retrieval",
        "evaluation_metrics": ["psnr", "ssim", "phase_error", "resolution_fwhm"],
        "canonical_references": [
            "Zheng et al., 'Wide-field, high-resolution Fourier ptychographic microscopy', "
            "Nature Photonics 7, 739-745 (2013)",
            "Tian & Waller, 'Quantitative differential phase contrast imaging in an "
            "LED array microscope', Optics Express 23, 11394-11403 (2015)",
        ],
        "canonical_datasets": [
            "Zheng lab FPM datasets (UCONN)",
            "Waller lab FPM benchmark data (Berkeley)",
        ],
        "tags": ["microscopy", "ptychography", "phase_retrieval", "led_array", "synthetic_aperture"],
    },

    "two_photon": {
        "display_name": "Two-Photon / Multiphoton Microscopy",
        "category": "microscopy",
        "description": (
            "Two-photon microscopy uses ultrashort pulsed near-infrared laser light "
            "(typically 700-1000 nm) to excite fluorophores via simultaneous absorption "
            "of two photons, providing intrinsic optical sectioning because excitation "
            "only occurs at the focal volume where photon density is sufficiently high. "
            "The longer excitation wavelength enables imaging depths of 500-1000 um in "
            "scattering tissue (e.g., brain), making it the standard for in vivo "
            "neuroscience. The point-spread function is effectively the square of the "
            "excitation PSF. Primary degradations include scattering-induced signal "
            "loss with depth and wavefront aberrations from tissue inhomogeneity."
        ),
        "experimental_setup": {
            "instrument": "Thorlabs Bergamo II / Bruker Ultima Investigator",
            "objective": "XLUMPLFLN 20x / 0.95 NA water immersion (Olympus)",
            "pixel_size_nm": 330,
            "excitation_source": "Ti:Sapphire laser (Coherent Chameleon, 920 nm)",
            "pulse_width_fs": 100,
            "repetition_rate_MHz": 80,
            "average_power_mW": 30,
            "dwell_time_us": 2,
            "imaging_depth_um": 500,
            "detector": "GaAsP PMT (non-descanned)",
        },
        "physics_class": "multiphoton_fluorescence",
        "forward_model_family": "two_photon_psf_squared",
        "wave_model": "nonlinear_optical",
        "sensor_type": "pmt",
        "source_type": "ultrafast_laser",
        "geometry": "raster_scan",
        "noise_model": "poisson",
        "typical_x_dims": [512, 512],
        "typical_y_dims": [512, 512],
        "typical_snr_range": [8.0, 30.0],
        "calibration_params": [
            "excitation_psf", "laser_power", "pulse_compression",
            "objective_correction_collar", "depth_attenuation_profile",
        ],
        "mismatch_modes": [
            "scattering_with_depth", "aberration_from_tissue",
            "photobleaching", "thermal_damage",
        ],
        "reconstruction_task_types": ["denoising", "deconvolution", "depth_correction"],
        "default_solver": "richardson_lucy",
        "evaluation_metrics": ["psnr", "ssim", "imaging_depth", "snr_vs_depth"],
        "canonical_references": [
            "Denk et al., 'Two-photon laser scanning fluorescence microscopy', "
            "Science 248, 73-76 (1990)",
            "Helmchen & Denk, 'Deep tissue two-photon microscopy', "
            "Nature Methods 2, 932-940 (2005)",
        ],
        "canonical_datasets": [
            "Allen Brain Observatory two-photon calcium imaging",
            "Stringer et al. (2019) mouse V1 two-photon dataset",
        ],
        "tags": ["microscopy", "two_photon", "multiphoton", "deep_tissue", "neuroscience"],
    },

    "sted": {
        "display_name": "STED Microscopy",
        "category": "microscopy",
        "description": (
            "Stimulated emission depletion (STED) microscopy breaks the diffraction "
            "limit by overlaying the excitation focus with a doughnut-shaped depletion "
            "beam that forces fluorophores at the periphery back to the ground state "
            "via stimulated emission, effectively shrinking the fluorescent spot to "
            "50 nm or below. The effective PSF width scales as d ~ lambda/(2*NA*sqrt(1 + I/I_s)) "
            "where I is the depletion intensity and I_s is the saturation intensity. "
            "Primary challenges include high depletion laser power causing "
            "photobleaching, and the photon-limited signal from the confined volume."
        ),
        "experimental_setup": {
            "instrument": "Abberior STEDYCON / Leica TCS SP8 STED 3X",
            "objective": "HC PL APO 100x / 1.40 NA oil STED WHITE",
            "pixel_size_nm": 20,
            "excitation_source": "pulsed white-light laser (640 nm line)",
            "STED_depletion_nm": 775,
            "STED_laser": "Onefive Katana HP (775 nm, 1.2 ns pulses)",
            "STED_power_mW": 200,
            "achieved_resolution_nm": 50,
            "dwell_time_us": 20,
            "detector": "HyD hybrid detector (Leica) / APD",
            "dye": "Abberior STAR RED / ATTO 647N",
        },
        "physics_class": "stimulated_emission_depletion",
        "forward_model_family": "sted_effective_psf_convolution",
        "wave_model": "incoherent",
        "sensor_type": "hybrid_detector",
        "source_type": "pulsed_laser_pair",
        "geometry": "raster_scan",
        "noise_model": "poisson",
        "typical_x_dims": [1024, 1024],
        "typical_y_dims": [1024, 1024],
        "typical_snr_range": [5.0, 20.0],
        "calibration_params": [
            "depletion_beam_alignment", "depletion_power",
            "sted_psf_measurement", "saturation_intensity",
        ],
        "mismatch_modes": [
            "doughnut_asymmetry", "depletion_beam_misalignment",
            "photobleaching", "anti_stokes_excitation",
        ],
        "reconstruction_task_types": ["super_resolution", "deconvolution", "denoising"],
        "default_solver": "richardson_lucy",
        "evaluation_metrics": ["psnr", "ssim", "resolution_fwhm", "frc"],
        "canonical_references": [
            "Hell & Wichmann, 'Breaking the diffraction resolution limit by stimulated "
            "emission', Optics Letters 19, 780-782 (1994)",
            "Vicidomini et al., 'STED nanoscopy', Annual Review of Biophysics 47, 377-404 (2018)",
        ],
        "canonical_datasets": [
            "BioSR STED paired dataset (Zhang et al., Nature Methods 2023)",
            "Abberior STED application note sample images",
        ],
        "tags": ["microscopy", "super_resolution", "sted", "nanoscopy", "diffraction_unlimited"],
    },

    "palm_storm": {
        "display_name": "PALM/STORM Single-Molecule Localization",
        "category": "microscopy",
        "description": (
            "Photoactivated localization microscopy (PALM) and stochastic optical "
            "reconstruction microscopy (STORM) achieve nanoscale resolution by "
            "stochastically activating sparse subsets of fluorescent molecules per "
            "frame, localizing each with sub-diffraction precision (proportional to "
            "sigma/sqrt(N) where N is detected photons), and accumulating "
            "localizations over thousands of frames. Typical localization precision "
            "is 10-30 nm. Primary challenges include overlapping emitters at high "
            "density, sample drift, and blinking statistics. Reconstruction uses "
            "Gaussian fitting (ThunderSTORM) or deep learning (DECODE)."
        ),
        "experimental_setup": {
            "instrument": "Nikon N-STORM / Zeiss ELYRA 7 SMLM",
            "objective": "Apo TIRF 100x / 1.49 NA oil",
            "camera_pixel_nm": 100,
            "reconstruction_pixel_nm": 25,
            "excitation_source": "640 nm laser (200 mW at fiber tip)",
            "activation_laser_nm": 405,
            "exposure_ms": 20,
            "total_frames": 10000,
            "frame_rate_fps": 50,
            "achieved_resolution_nm": 20,
            "detector": "Andor iXon Ultra 897 EMCCD",
            "imaging_buffer": "MEA + GLOX for Alexa Fluor 647",
        },
        "physics_class": "single_molecule_localization",
        "forward_model_family": "point_emitter_psf_model",
        "wave_model": "incoherent",
        "sensor_type": "emccd_or_scmos",
        "source_type": "laser_activation_excitation",
        "geometry": "widefield_or_tirf",
        "noise_model": "poisson_gaussian",
        "typical_x_dims": [256, 256],
        "typical_y_dims": [10000, 256, 256],
        "typical_snr_range": [3.0, 15.0],
        "calibration_params": [
            "psf_model_3d", "pixel_size", "camera_gain",
            "drift_correction_fiducials", "activation_density",
        ],
        "mismatch_modes": [
            "emitter_overlap", "sample_drift", "psf_model_error",
            "background_nonuniformity", "blinking_statistics",
        ],
        "reconstruction_task_types": [
            "single_molecule_localization", "drift_correction", "density_estimation",
        ],
        "default_solver": "thunderstorm",
        "evaluation_metrics": [
            "localization_precision_nm", "recall", "jaccard_index", "frc_resolution",
        ],
        "canonical_references": [
            "Betzig et al., 'Imaging intracellular fluorescent proteins at nanometer "
            "resolution', Science 313, 1642-1645 (2006)",
            "Rust et al., 'Sub-diffraction-limit imaging by stochastic optical "
            "reconstruction microscopy (STORM)', Nature Methods 3, 793-796 (2006)",
            "Speiser et al., 'Deep learning enables fast and dense single-molecule "
            "localization (DECODE)', Nature Methods 18, 1082-1090 (2021)",
        ],
        "canonical_datasets": [
            "SMLM Challenge 2016 (Sage et al., Nature Methods 2019)",
            "ThunderSTORM tutorial datasets",
        ],
        "tags": ["microscopy", "super_resolution", "localization", "palm", "storm", "smlm"],
    },

    "tirf": {
        "display_name": "TIRF Microscopy",
        "category": "microscopy",
        "description": (
            "Total internal reflection fluorescence (TIRF) microscopy selectively "
            "excites fluorophores within ~100-200 nm of the coverslip surface using "
            "the evanescent field generated when excitation light undergoes total "
            "internal reflection at the glass-sample interface. This provides "
            "exceptional axial selectivity for imaging membrane-associated events "
            "such as vesicle fusion and focal adhesions. The lateral image follows "
            "standard widefield PSF convolution but with near-zero out-of-focus "
            "background. Primary degradations include non-uniform evanescent field "
            "and interference fringes from coherent illumination."
        ),
        "experimental_setup": {
            "instrument": "Nikon Eclipse Ti2-E TIRF / Olympus cellTIRF-4Line",
            "objective": "Apo TIRF 100x / 1.49 NA oil",
            "pixel_size_nm": 65,
            "excitation_source": "488 nm laser (Coherent OBIS, 100 mW)",
            "evanescent_depth_nm": 100,
            "exposure_ms": 30,
            "frame_rate_fps": 33,
            "detector": "Hamamatsu ORCA-Fusion BT sCMOS",
        },
        "physics_class": "evanescent_wave_fluorescence",
        "forward_model_family": "tirf_psf_convolution",
        "wave_model": "incoherent",
        "sensor_type": "scmos",
        "source_type": "laser_tirf",
        "geometry": "planar_evanescent",
        "noise_model": "poisson_gaussian",
        "typical_x_dims": [512, 512],
        "typical_y_dims": [512, 512],
        "typical_snr_range": [15.0, 40.0],
        "calibration_params": [
            "incidence_angle", "evanescent_depth", "laser_alignment",
            "flatfield_correction", "refractive_index_medium",
        ],
        "mismatch_modes": [
            "angle_calibration_error", "scattering_excitation",
            "interference_fringes", "evanescent_depth_variation",
        ],
        "reconstruction_task_types": ["deconvolution", "denoising"],
        "default_solver": "richardson_lucy",
        "evaluation_metrics": ["psnr", "ssim", "nrmse", "axial_confinement"],
        "canonical_references": [
            "Axelrod, 'Total internal reflection fluorescence microscopy in cell "
            "biology', Traffic 2, 764-774 (2001)",
        ],
        "canonical_datasets": [
            "Cell Tracking Challenge TIRF sequences",
            "FPbase TIRF imaging examples",
        ],
        "tags": ["microscopy", "tirf", "evanescent_wave", "membrane_imaging", "surface_selective"],
    },

    "polarization": {
        "display_name": "Polarization Microscopy",
        "category": "microscopy",
        "description": (
            "Polarization microscopy measures anisotropic optical properties by "
            "analysing the polarisation state of light through the sample. In Mueller "
            "matrix imaging, the sample is illuminated with known polarisation states "
            "and the output is analysed, yielding a 4x4 Mueller matrix at each pixel "
            "encoding birefringence, optical activity, and depolarisation. The "
            "LC-PolScope uses liquid crystal retarders for rapid modulation. "
            "Reconstruction involves solving for Mueller elements and Lu-Chipman "
            "decomposition into physically meaningful parameters."
        ),
        "experimental_setup": {
            "instrument": "CRi Abrio / OpenPolScope",
            "objective": "Plan Fluor 60x / 1.30 NA oil",
            "pixel_size_nm": 110,
            "wavelength_nm": 546,
            "polarisation_states": 4,
            "retarder": "liquid crystal variable retarder (Meadowlark)",
            "detector": "sCMOS 2048x2048",
            "application": "birefringence / collagen fibre mapping",
        },
        "physics_class": "polarimetric",
        "forward_model_family": "mueller_matrix",
        "wave_model": "polarised",
        "sensor_type": "ccd_or_scmos",
        "source_type": "filtered_lamp",
        "geometry": "planar",
        "noise_model": "poisson_gaussian",
        "typical_x_dims": [512, 512, 16],
        "typical_y_dims": [4, 512, 512],
        "typical_snr_range": [15.0, 35.0],
        "calibration_params": [
            "polariser_orientation", "retarder_calibration",
            "background_birefringence", "system_mueller_matrix",
        ],
        "mismatch_modes": [
            "retarder_calibration_drift", "polariser_extinction_ratio",
            "stress_birefringence_optics", "depolarisation_artefacts",
        ],
        "reconstruction_task_types": [
            "mueller_matrix_recovery", "birefringence_mapping", "lu_chipman_decomposition",
        ],
        "default_solver": "pnp_hqs",
        "evaluation_metrics": ["retardance_accuracy", "orientation_mae", "psnr"],
        "canonical_references": [
            "Mehta et al., 'Quantitative polarized light microscopy using the LC-PolScope', "
            "Live Cell Imaging: A Laboratory Manual, CSHL Press (2010)",
            "Lu & Chipman, 'Interpretation of Mueller matrices based on polar "
            "decomposition', J. Opt. Soc. Am. A 13, 1106-1113 (1996)",
        ],
        "canonical_datasets": [
            "OpenPolScope calibration data",
            "Collagen SHG/polarisation histopathology datasets",
        ],
        "tags": ["microscopy", "polarization", "birefringence", "mueller_matrix"],
    },

    "lensless": {
        "display_name": "Lensless (Diffuser Camera) Imaging",
        "category": "microscopy",
        "description": (
            "Lensless imaging replaces the objective lens with a thin optical "
            "element (phase diffuser or coded mask) placed directly near the sensor. "
            "Scene light produces a multiplexed caustic pattern encoding the entire "
            "scene. The forward model is y = H * x + n where H is determined by the "
            "mask's phase profile and mask-to-sensor distance. Each scene point "
            "contributes across many sensor pixels, yielding a multiplexing advantage. "
            "Reconstruction solves a large-scale inverse problem via ADMM or FISTA "
            "with total-variation or learned priors."
        ),
        "experimental_setup": {
            "instrument": "DiffuserCam / FlatCam prototype",
            "sensor": "Raspberry Pi HQ Camera (Sony IMX477, 4056x3040)",
            "pixel_pitch_um": 1.55,
            "diffuser_type": "optical diffuser (Luminit 0.5-deg) / coded mask",
            "diffuser_to_sensor_mm": 2.5,
            "field_of_view_deg": 40,
            "image_size": "2592x1944",
        },
        "physics_class": "lensless_computational",
        "forward_model_family": "psf_convolution_or_linear_operator",
        "wave_model": "incoherent",
        "sensor_type": "cmos",
        "source_type": "ambient_or_led",
        "geometry": "planar",
        "noise_model": "poisson_gaussian",
        "typical_x_dims": [256, 256],
        "typical_y_dims": [2592, 1944],
        "typical_snr_range": [10.0, 30.0],
        "calibration_params": [
            "psf_measurement", "mask_to_sensor_distance",
            "flatfield", "background_subtraction",
        ],
        "mismatch_modes": [
            "psf_calibration_error", "mask_sensor_misalignment",
            "depth_dependent_psf_variation", "stray_light",
        ],
        "reconstruction_task_types": ["image_reconstruction", "3d_refocusing"],
        "default_solver": "admm_tv",
        "evaluation_metrics": ["psnr", "ssim", "nrmse", "lpips"],
        "canonical_references": [
            "Antipa et al., 'DiffuserCam: lensless single-exposure 3D imaging', "
            "Optica 5, 1-9 (2018)",
            "Asif et al., 'FlatCam: Thin, Lensless Cameras Using Coded Aperture', "
            "IEEE TCI 3, 384-397 (2017)",
        ],
        "canonical_datasets": [
            "DiffuserCam lensless mirflickr dataset (Monakhova et al.)",
            "PhlatCam benchmark (Boominathan et al., IEEE TPAMI 2022)",
        ],
        "tags": ["microscopy", "lensless", "computational", "diffuser_camera", "coded_aperture"],
    },

    # ══════════════════════════════════════════════════════════════════════════
    # COMPRESSIVE  (4 modalities)
    # ══════════════════════════════════════════════════════════════════════════

    "cassi": {
        "display_name": "Coded Aperture Snapshot Spectral Imaging (CASSI)",
        "category": "compressive",
        "description": (
            "CASSI captures a 3D hyperspectral data cube (2 spatial + 1 spectral "
            "dimension) in a single 2D camera exposure. The scene is modulated by a "
            "binary coded aperture mask, spectrally dispersed by a prism, and "
            "integrated onto a 2D detector. The forward model is y = H*x + n where "
            "H encodes both coded-aperture modulation and spectral-dispersion shift. "
            "Compression ratios equal the number of spectral bands (e.g. 28:1). "
            "Reconstruction exploits spectral correlation via GAP-TV, MST, or CST."
        ),
        "experimental_setup": {
            "instrument": "Custom SD-CASSI / KAIST CASSI prototype",
            "coded_aperture": "binary random mask on photolithography substrate",
            "disperser": "Amici prism (SD-CASSI)",
            "spectral_bands": 28,
            "wavelength_range_nm": "450-650",
            "spatial_resolution": "256x256",
            "compression_ratio": 28,
            "detector": "FLIR Grasshopper3 monochrome CMOS (2048x2048)",
            "relay_lens": "4f relay system with 1:1 magnification",
        },
        "physics_class": "spectral_coding",
        "forward_model_family": "coded_aperture_dispersion",
        "wave_model": "ray",
        "sensor_type": "cmos",
        "source_type": "broadband",
        "geometry": "planar",
        "noise_model": "gaussian",
        "typical_x_dims": [256, 256, 28],
        "typical_y_dims": [256, 310],
        "typical_snr_range": [20.0, 40.0],
        "calibration_params": [
            "coded_aperture_mask", "dispersion_curve",
            "dark_frame", "spectral_response",
        ],
        "mismatch_modes": [
            "mask_misalignment", "dispersion_curve_error",
            "spectral_response_drift", "defocus",
        ],
        "reconstruction_task_types": [
            "hyperspectral_reconstruction", "spectral_demixing",
        ],
        "default_solver": "mst",
        "evaluation_metrics": ["psnr", "ssim", "sam", "ergas"],
        "canonical_references": [
            "Wagadarikar et al., 'Single disperser design for coded aperture snapshot "
            "spectral imaging', Applied Optics 47, B44-B51 (2008)",
            "Cai et al., 'Mask-guided Spectral-wise Transformer (MST++)', CVPRW 2022",
        ],
        "canonical_datasets": [
            "CAVE (Columbia, 32 scenes, 512x512x31)",
            "KAIST (30 scenes, 2704x3376x28)",
            "ARAD_1K (1000 hyperspectral images)",
        ],
        "tags": ["compressive", "spectral", "coded_aperture", "snapshot", "hyperspectral"],
    },

    "spc": {
        "display_name": "Single-Pixel Camera",
        "category": "compressive",
        "description": (
            "The single-pixel camera reconstructs a 2D image from scalar intensity "
            "measurements acquired by a photodiode after spatially modulating the "
            "scene with known patterns on a DMD. Each measurement y_i is the inner "
            "product of the scene with a pattern, giving y = Phi*x + n. Compressed "
            "sensing theory guarantees recovery from M << N measurements if the scene "
            "is sparse. The single detector can operate at wavelengths where array "
            "detectors are unavailable (SWIR, THz). Reconstruction uses FISTA with "
            "L1/TV penalties or Plug-and-Play methods."
        ),
        "experimental_setup": {
            "instrument": "Rice SPC prototype / custom DMD system",
            "spatial_modulator": "TI DLP7000 DMD (1024x768 micromirrors)",
            "detector": "Thorlabs PDA100A2 Si photodiode",
            "effective_resolution": "64x64",
            "sampling_ratio": 0.25,
            "sensing_matrix": "Walsh-Hadamard (partial)",
            "pattern_rate_Hz": 22000,
            "collection_optics": "50 mm f/1.4 lens",
        },
        "physics_class": "compressive_sensing",
        "forward_model_family": "structured_illumination_sensing",
        "wave_model": "ray",
        "sensor_type": "single_pixel_detector",
        "source_type": "broadband",
        "geometry": "planar",
        "noise_model": "gaussian",
        "typical_x_dims": [64, 64],
        "typical_y_dims": [1024],
        "typical_snr_range": [15.0, 40.0],
        "calibration_params": [
            "pattern_matrix", "detector_response",
            "pattern_alignment", "dark_current",
        ],
        "mismatch_modes": [
            "pattern_misalignment", "detector_nonlinearity",
            "diffraction_at_dmd", "ambient_light_leakage",
        ],
        "reconstruction_task_types": ["compressed_sensing_recovery", "image_reconstruction"],
        "default_solver": "pnp_fista",
        "evaluation_metrics": ["psnr", "ssim", "nrmse"],
        "canonical_references": [
            "Duarte et al., 'Single-pixel imaging via compressive sampling', "
            "IEEE Signal Processing Magazine 25, 83-91 (2008)",
            "Edgar et al., 'Principles and prospects for single-pixel imaging', "
            "Nature Photonics 13, 13-20 (2019)",
        ],
        "canonical_datasets": [
            "Set11 (11 standard test images)",
            "BSD68 (Martin et al., ICCV 2001)",
        ],
        "tags": ["compressive", "single_pixel", "compressed_sensing", "dmd", "sub_nyquist"],
    },

    "cacti": {
        "display_name": "Coded Aperture Compressive Temporal Imaging (CACTI)",
        "category": "compressive",
        "description": (
            "CACTI captures multiple video frames in a single camera exposure by "
            "modulating the scene with a shifting binary mask during the integration "
            "period. Each temporal frame sees a different mask pattern, and the "
            "detector integrates all modulated frames into a single 2D measurement. "
            "The forward model is y = sum_t M_t * x_t + n where M_t is the mask at "
            "time t. Typical compression ratios are 8-48 frames per snapshot. "
            "Reconstruction exploits temporal correlation via GAP-TV, PnP-FFDNet, "
            "or deep unfolding networks (STFormer, EfficientSCI)."
        ),
        "experimental_setup": {
            "instrument": "Custom CACTI system (Duke / USTC prototype)",
            "coded_aperture": "shifting binary mask on lithographic substrate",
            "frames_per_snapshot": 8,
            "spatial_resolution": "256x256",
            "compression_ratio": 8,
            "equivalent_fps": 1200,
            "detector": "FLIR Point Grey Grasshopper3 CMOS",
            "reconstruction": "GAP-TV / PnP-FFDNet / STFormer",
        },
        "physics_class": "temporal_coding",
        "forward_model_family": "coded_aperture_temporal",
        "wave_model": "ray",
        "sensor_type": "cmos",
        "source_type": "broadband",
        "geometry": "planar",
        "noise_model": "gaussian",
        "typical_x_dims": [256, 256, 8],
        "typical_y_dims": [256, 256],
        "typical_snr_range": [20.0, 40.0],
        "calibration_params": [
            "mask_patterns", "mask_shift_calibration",
            "dark_frame", "temporal_alignment",
        ],
        "mismatch_modes": [
            "mask_shift_error", "motion_blur_within_frame",
            "mask_diffraction", "nonuniform_illumination",
        ],
        "reconstruction_task_types": ["video_reconstruction", "compressed_sensing_recovery"],
        "default_solver": "gap_tv",
        "evaluation_metrics": ["psnr", "ssim", "nrmse"],
        "canonical_references": [
            "Llull et al., 'Coded aperture compressive temporal imaging', "
            "Optics Express 19, 10526 (2011)",
            "Yuan et al., 'Generalized alternating projection based total variation "
            "minimization (GAP-TV)', IEEE ICIP 2016",
            "Wang et al., 'Spatial-Temporal Transformer for Video Snapshot "
            "Compressive Imaging (STFormer)', ECCV 2022",
        ],
        "canonical_datasets": [
            "Kobe, Runner, Drop, Traffic (grayscale SCI benchmarks)",
            "DAVIS 2017 (adapted for SCI simulation)",
        ],
        "tags": ["compressive", "video", "temporal", "snapshot", "high_speed"],
    },

    "matrix": {
        "display_name": "Generic Compressive Matrix Sensing",
        "category": "compressive",
        "description": (
            "Generic compressive sensing framework where the measurement process is "
            "modelled as y = A*x + n with A being an explicit M x N sensing matrix "
            "(M < N). This covers any linear inverse problem including random "
            "Gaussian, Bernoulli, or structured sensing matrices. The compressed "
            "sensing theory of Candes, Romberg, and Tao guarantees exact recovery "
            "when x is sparse and A satisfies the restricted isometry property (RIP). "
            "Reconstruction uses standard proximal algorithms (FISTA, ADMM) with "
            "sparsity-promoting regularizers (L1, TV, wavelet)."
        ),
        "experimental_setup": {
            "matrix_size": "256x256",
            "sampling_ratio": 0.25,
            "sensing_matrix": "Gaussian random / partial Fourier",
            "rank_assumption": "low-rank or sparse",
            "reconstruction": "FISTA-L2 / ADMM / ISTA-Net",
        },
        "physics_class": "compressive_sensing",
        "forward_model_family": "explicit_matrix",
        "wave_model": "none",
        "sensor_type": "generic",
        "source_type": "generic",
        "geometry": "abstract",
        "noise_model": "gaussian",
        "typical_x_dims": [256, 256],
        "typical_y_dims": [16384],
        "typical_snr_range": [15.0, 50.0],
        "calibration_params": ["sensing_matrix", "noise_variance"],
        "mismatch_modes": [
            "matrix_perturbation", "quantization_error",
            "model_mismatch",
        ],
        "reconstruction_task_types": ["compressed_sensing_recovery"],
        "default_solver": "fista_l2",
        "evaluation_metrics": ["psnr", "ssim", "nrmse"],
        "canonical_references": [
            "Candes et al., 'Robust uncertainty principles: exact signal reconstruction "
            "from highly incomplete frequency information', IEEE TIT 52, 489-509 (2006)",
            "Donoho, 'Compressed sensing', IEEE TIT 52, 1289-1306 (2006)",
        ],
        "canonical_datasets": [
            "Set11 / BSD68 (simulation benchmarks)",
        ],
        "tags": ["compressive", "generic", "matrix", "compressed_sensing", "inverse_problem"],
    },

    # ══════════════════════════════════════════════════════════════════════════
    # MEDICAL IMAGING — X-ray  (9 modalities)
    # ══════════════════════════════════════════════════════════════════════════

    "ct": {
        "display_name": "X-ray Computed Tomography",
        "category": "medical",
        "description": (
            "X-ray CT reconstructs cross-sectional images from a set of line-integral "
            "projections (sinogram) acquired as an X-ray source and detector array "
            "rotate around the patient. The forward model is the Radon transform: "
            "y = R*x + n where R computes line integrals along each ray. Sparse-view "
            "and low-dose protocols reduce radiation but introduce streak artifacts "
            "and noise. Reconstruction uses filtered back-projection (FBP) or "
            "iterative methods (MBIR, DL post-processing)."
        ),
        "experimental_setup": {
            "instrument": "Siemens SOMATOM Force / GE Revolution CT",
            "image_size": "512x512",
            "num_views": 60,
            "full_dose_views": 1000,
            "detector_pixels": 736,
            "kVp": 120,
            "dose_level": "25% of full dose (quarter-dose)",
            "reconstruction": "FBP + DL denoising / end-to-end",
            "dataset": "LoDoPaB-CT, DeepLesion",
        },
        "physics_class": "tomographic",
        "forward_model_family": "radon_transform",
        "wave_model": "ray",
        "sensor_type": "scintillator_detector",
        "source_type": "xray_tube",
        "geometry": "rotational",
        "noise_model": "poisson",
        "typical_x_dims": [256, 256],
        "typical_y_dims": [180, 362],
        "typical_snr_range": [25.0, 45.0],
        "calibration_params": [
            "flat_field", "center_of_rotation", "beam_hardening_correction",
            "detector_response", "geometric_calibration",
        ],
        "mismatch_modes": [
            "center_offset", "beam_hardening", "scatter",
            "motion_artifact", "metal_artifact",
        ],
        "reconstruction_task_types": ["tomographic_reconstruction", "denoising", "artifact_removal"],
        "default_solver": "fbp",
        "evaluation_metrics": ["psnr", "ssim", "hu_accuracy", "nrmse"],
        "canonical_references": [
            "Feldkamp et al., 'Practical cone-beam algorithm', J. Opt. Soc. Am. A 1, "
            "612-619 (1984)",
            "Leuschner et al., 'LoDoPaB-CT, a benchmark dataset for low-dose CT "
            "reconstruction', Scientific Data 8, 109 (2021)",
        ],
        "canonical_datasets": [
            "LoDoPaB-CT (Scientific Data 2021)",
            "DeepLesion (NIH Clinical Center)",
            "AAPM Low-Dose CT Grand Challenge",
        ],
        "tags": ["medical", "tomography", "xray", "radon", "low_dose"],
    },

    "xray_radiography": {
        "display_name": "X-ray Radiography",
        "category": "medical",
        "description": (
            "Digital X-ray radiography produces a 2D projection image by transmitting "
            "X-rays through the body onto a flat-panel detector. The forward model "
            "follows Beer-Lambert attenuation: y = I_0 * exp(-integral(mu(s) ds)) + n "
            "where mu is the linear attenuation coefficient along each ray. The image "
            "is a superposition of all structures along the beam path. Primary "
            "degradations include quantum noise (Poisson), scatter, and geometric "
            "magnification artifacts."
        ),
        "experimental_setup": {
            "instrument": "Carestream DRX-Evolution / Siemens Ysio Max",
            "image_size": "2048x2048",
            "pixel_pitch_mm": 0.1,
            "kVp": 120,
            "mAs": 4.0,
            "SID_cm": 180,
            "detector": "flat-panel (CsI + aSi TFT)",
            "dataset": "CheXpert, MIMIC-CXR, NIH ChestX-ray14",
        },
        "physics_class": "radiographic",
        "forward_model_family": "beer_lambert_projection",
        "wave_model": "ray",
        "sensor_type": "flat_panel_detector",
        "source_type": "xray_tube",
        "geometry": "projection",
        "noise_model": "poisson",
        "typical_x_dims": [2048, 2048],
        "typical_y_dims": [2048, 2048],
        "typical_snr_range": [25.0, 45.0],
        "calibration_params": [
            "flat_field", "dark_frame", "gain_map",
            "scatter_correction", "geometric_calibration",
        ],
        "mismatch_modes": [
            "scatter", "beam_hardening", "patient_motion",
            "grid_artifact", "detector_lag",
        ],
        "reconstruction_task_types": ["denoising", "enhancement", "disease_classification"],
        "default_solver": "tv_fista",
        "evaluation_metrics": ["psnr", "ssim", "auc_roc"],
        "canonical_references": [
            "Irvin et al., 'CheXpert: A large chest radiograph dataset', AAAI 2019",
            "Wang et al., 'ChestX-ray8: Hospital-scale chest X-ray database', CVPR 2017",
        ],
        "canonical_datasets": [
            "CheXpert (Stanford, 224K studies)",
            "MIMIC-CXR (MIT/BIDMC, 377K images)",
            "NIH ChestX-ray14 (112K images)",
        ],
        "tags": ["medical", "xray", "projection", "chest", "radiography"],
    },

    "fluoroscopy": {
        "display_name": "Fluoroscopy",
        "category": "medical",
        "description": (
            "Fluoroscopy provides real-time continuous X-ray imaging for guiding "
            "interventional procedures. The forward model is the same Beer-Lambert "
            "projection as radiography but at much lower dose per frame (typically "
            "1 uGy/frame at 15-30 fps) resulting in severely photon-limited images. "
            "Temporal redundancy from the video stream enables frame-to-frame "
            "denoising and recursive filtering. Primary challenges include low SNR, "
            "motion blur from patient/organ movement, and veiling glare from scatter."
        ),
        "experimental_setup": {
            "instrument": "Siemens Artis Pheno / GE Innova IGS 630",
            "image_size": "1024x1024",
            "kVp": 70,
            "frame_rate_fps": 15,
            "dose_per_frame_uGy": 1.0,
            "detector_size_cm": "30x30",
            "detector_type": "flat-panel (CsI + aSi)",
        },
        "physics_class": "radiographic",
        "forward_model_family": "beer_lambert_projection",
        "wave_model": "ray",
        "sensor_type": "flat_panel_detector",
        "source_type": "xray_tube",
        "geometry": "projection",
        "noise_model": "poisson",
        "typical_x_dims": [1024, 1024],
        "typical_y_dims": [1024, 1024],
        "typical_snr_range": [10.0, 25.0],
        "calibration_params": [
            "flat_field", "geometric_distortion", "scatter_kernel",
            "temporal_filter_weight",
        ],
        "mismatch_modes": [
            "patient_motion", "scatter", "veiling_glare",
            "detector_lag", "pulsation_artifact",
        ],
        "reconstruction_task_types": ["temporal_denoising", "scatter_correction"],
        "default_solver": "tv_fista",
        "evaluation_metrics": ["psnr", "ssim", "cnr"],
        "canonical_references": [
            "Defined by IEC 62220-1 standard for fluoroscopy detector characterization",
        ],
        "canonical_datasets": ["Clinical fluoroscopy sequences (institution-specific)"],
        "tags": ["medical", "xray", "real_time", "interventional", "fluoroscopy"],
    },

    "mammography": {
        "display_name": "Mammography",
        "category": "medical",
        "description": (
            "Full-field digital mammography (FFDM) produces high-resolution X-ray "
            "projection images of compressed breast tissue for cancer screening. The "
            "low-energy X-ray beam (25-32 kVp with W/Rh or Mo/Mo target-filter) "
            "maximizes soft tissue contrast. Amorphous selenium flat-panel detectors "
            "provide direct conversion with ~50 um pixel pitch. The forward model "
            "follows Beer-Lambert with energy-dependent attenuation. Primary "
            "challenges include overlapping tissue structures, microcalcification "
            "detection, and dense breast tissue masking lesions."
        ),
        "experimental_setup": {
            "instrument": "Hologic Selenia Dimensions / Siemens MAMMOMAT Revelation",
            "image_size": "2294x1914",
            "kVp": 28,
            "target_filter": "W/Rh",
            "mAs": 60,
            "detector": "flat-panel amorphous selenium (direct conversion)",
            "pixel_pitch_um": 70,
            "dataset": "VinDr-Mammo, CBIS-DDSM, INbreast",
        },
        "physics_class": "radiographic",
        "forward_model_family": "beer_lambert_projection",
        "wave_model": "ray",
        "sensor_type": "flat_panel_ase",
        "source_type": "xray_tube",
        "geometry": "projection",
        "noise_model": "poisson",
        "typical_x_dims": [2294, 1914],
        "typical_y_dims": [2294, 1914],
        "typical_snr_range": [20.0, 40.0],
        "calibration_params": [
            "flat_field", "detector_calibration", "compression_thickness",
            "target_filter_combination", "AEC_calibration",
        ],
        "mismatch_modes": [
            "motion_blur", "scatter", "compression_artifact",
            "grid_artifact", "skin_fold",
        ],
        "reconstruction_task_types": ["denoising", "mass_detection", "calcification_detection"],
        "default_solver": "tv_fista",
        "evaluation_metrics": ["auc_roc", "sensitivity", "specificity", "psnr"],
        "canonical_references": [
            "VinDr-Mammo, Scientific Data 2023",
            "Lee et al., 'A curated mammography dataset (CBIS-DDSM)', Scientific Data 4, 170177 (2017)",
        ],
        "canonical_datasets": [
            "VinDr-Mammo (5000 4-view exams)",
            "CBIS-DDSM (curated DDSM subset)",
            "INbreast (410 images, Moreira et al.)",
        ],
        "tags": ["medical", "xray", "mammography", "breast", "screening"],
    },

    "dexa": {
        "display_name": "Dual-Energy X-ray Absorptiometry",
        "category": "medical",
        "description": (
            "DEXA measures bone mineral density (BMD) by acquiring two X-ray "
            "projections at different energies (typically 70 and 140 kVp) and "
            "decomposing the attenuation into bone and soft-tissue components using "
            "their known energy-dependent mass attenuation coefficients. The dual-energy "
            "forward model is y_E = I_0(E) * exp(-(mu_b(E)*t_b + mu_s(E)*t_s)) + n "
            "for each energy E. Output is areal BMD (g/cm2) and T-score for "
            "osteoporosis diagnosis. Precision errors of ~1% are achievable."
        ),
        "experimental_setup": {
            "instrument": "Hologic Discovery A / GE Lunar iDXA",
            "energies_kVp": [70, 140],
            "pixel_size_mm": 0.5,
            "scan_time_s": 30,
            "dose_uSv": 1,
            "output": "BMD (g/cm2), T-score",
            "sites": "lumbar spine, proximal femur",
        },
        "physics_class": "dual_energy_radiographic",
        "forward_model_family": "dual_energy_decomposition",
        "wave_model": "ray",
        "sensor_type": "multi_element_detector",
        "source_type": "xray_tube_dual_energy",
        "geometry": "fan_beam_scan",
        "noise_model": "poisson",
        "typical_x_dims": [512, 256],
        "typical_y_dims": [2, 512, 256],
        "typical_snr_range": [20.0, 40.0],
        "calibration_params": [
            "phantom_calibration", "beam_quality", "detector_linearity",
            "soft_tissue_baseline",
        ],
        "mismatch_modes": [
            "beam_hardening", "fat_composition_error",
            "positioning_error", "degenerative_changes",
        ],
        "reconstruction_task_types": ["material_decomposition", "bmd_estimation"],
        "default_solver": "dual_energy_decomposition",
        "evaluation_metrics": ["bmd_precision_cv", "t_score_accuracy"],
        "canonical_references": [
            "Blake & Fogelman, 'The role of DXA bone density scans in the diagnosis "
            "and treatment of osteoporosis', Postgrad. Med. J. 83, 509-517 (2007)",
        ],
        "canonical_datasets": ["NHANES DXA reference data (CDC)"],
        "tags": ["medical", "xray", "bone_density", "dexa", "osteoporosis"],
    },

    "cbct": {
        "display_name": "Cone-Beam Computed Tomography",
        "category": "medical",
        "description": (
            "Cone-beam CT (CBCT) uses a divergent cone-shaped X-ray beam and a "
            "flat-panel 2D detector to acquire volumetric data in a single rotation, "
            "unlike fan-beam CT which acquires slice-by-slice. The 3D Feldkamp-Davis-Kress "
            "(FDK) algorithm performs approximate filtered back-projection for cone "
            "geometry. CBCT is widely used in dental, ENT, and image-guided radiation "
            "therapy. Primary artifacts include cone-beam artifacts at large cone "
            "angles, scatter, and truncation. Sparse-view CBCT reduces scan time and "
            "dose but introduces streak artifacts."
        ),
        "experimental_setup": {
            "instrument": "Varian TrueBeam / Elekta XVI / iCAT dental CBCT",
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
        "physics_class": "tomographic",
        "forward_model_family": "cone_beam_projection",
        "wave_model": "ray",
        "sensor_type": "flat_panel_detector",
        "source_type": "xray_tube",
        "geometry": "cone_beam_rotational",
        "noise_model": "poisson",
        "typical_x_dims": [512, 512, 512],
        "typical_y_dims": [360, 150, 150],
        "typical_snr_range": [20.0, 40.0],
        "calibration_params": [
            "geometric_calibration", "flat_field", "scatter_correction",
            "center_of_rotation", "detector_offset",
        ],
        "mismatch_modes": [
            "cone_beam_artifact", "scatter", "truncation",
            "patient_motion", "ring_artifact",
        ],
        "reconstruction_task_types": ["volumetric_reconstruction", "scatter_correction"],
        "default_solver": "fdk",
        "evaluation_metrics": ["psnr", "ssim", "hu_accuracy", "nrmse"],
        "canonical_references": [
            "Feldkamp et al., 'Practical cone-beam algorithm', JOSA A 1, 612-619 (1984)",
        ],
        "canonical_datasets": ["ICASSP 2024 CBCT Challenge"],
        "tags": ["medical", "tomography", "cone_beam", "cbct", "dental"],
    },

    "angiography": {
        "display_name": "X-ray Angiography",
        "category": "medical",
        "description": (
            "Digital subtraction angiography (DSA) visualizes blood vessels by "
            "subtracting a pre-contrast mask image from post-contrast images acquired "
            "after injecting iodinated contrast agent. The subtraction eliminates "
            "static anatomy, isolating vascular structures. The forward model is "
            "y_post - y_pre = Delta_mu * t_vessel + n where Delta_mu is the "
            "attenuation increase from iodine. Primary challenges include patient "
            "motion between mask and contrast frames, breathing artifacts, and "
            "superposition of overlapping vessels."
        ),
        "experimental_setup": {
            "instrument": "Siemens Artis Q / Philips Allura Xper FD20",
            "image_size": "1024x1024",
            "kVp": 80,
            "frame_rate_fps": 30,
            "contrast_agent": "iodinated (Iopamidol 370 mg I/mL)",
            "injection_rate_mL_s": 4.0,
            "detector": "flat-panel (CsI)",
            "application": "cerebral / coronary angiography",
        },
        "physics_class": "radiographic",
        "forward_model_family": "subtraction_projection",
        "wave_model": "ray",
        "sensor_type": "flat_panel_detector",
        "source_type": "xray_tube",
        "geometry": "projection",
        "noise_model": "poisson",
        "typical_x_dims": [1024, 1024],
        "typical_y_dims": [1024, 1024],
        "typical_snr_range": [15.0, 35.0],
        "calibration_params": [
            "mask_registration", "pixel_shift_correction",
            "contrast_timing", "flat_field",
        ],
        "mismatch_modes": [
            "motion_misregistration", "breathing_artifact",
            "bowel_gas_motion", "contrast_timing_mismatch",
        ],
        "reconstruction_task_types": ["subtraction", "vessel_segmentation", "motion_correction"],
        "default_solver": "dsa_subtraction",
        "evaluation_metrics": ["cnr", "vessel_visibility", "psnr"],
        "canonical_references": [
            "Defined by clinical DSA standards (ACC/AHA guidelines)",
        ],
        "canonical_datasets": ["IntrA (intracranial aneurysm 3DRA dataset)"],
        "tags": ["medical", "xray", "angiography", "vascular", "interventional"],
    },

    "photoacoustic": {
        "display_name": "Photoacoustic Imaging",
        "category": "medical",
        "description": (
            "Photoacoustic imaging (PAI) is a hybrid modality that combines optical "
            "absorption contrast with ultrasonic detection. Short laser pulses "
            "(nanoseconds) are absorbed by tissue chromophores (hemoglobin, melanin), "
            "causing thermoelastic expansion that generates broadband ultrasound waves "
            "detected by transducer arrays. The forward model involves the photoacoustic "
            "wave equation: the initial pressure p_0(r) is proportional to the "
            "absorbed optical energy. Reconstruction inverts the acoustic propagation "
            "using delay-and-sum (DAS) or model-based algorithms."
        ),
        "experimental_setup": {
            "instrument": "iThera Medical MSOT inVision / Vevo LAZR-X",
            "laser_wavelengths_nm": [700, 800, 900],
            "pulse_duration_ns": 6,
            "pulse_repetition_Hz": 10,
            "transducer": "128-element linear array",
            "center_frequency_MHz": 7.5,
            "bandwidth_percent": 80,
            "sampling_rate_MHz": 31.2,
            "reconstruction": "DAS / model-based",
            "dataset": "OADAT",
        },
        "physics_class": "photoacoustic",
        "forward_model_family": "photoacoustic_wave_equation",
        "wave_model": "acoustic",
        "sensor_type": "ultrasound_transducer",
        "source_type": "pulsed_laser",
        "geometry": "tomographic_acoustic",
        "noise_model": "gaussian",
        "typical_x_dims": [256, 256],
        "typical_y_dims": [128, 2048],
        "typical_snr_range": [10.0, 30.0],
        "calibration_params": [
            "speed_of_sound", "transducer_positions",
            "laser_fluence_map", "detector_bandwidth",
        ],
        "mismatch_modes": [
            "speed_of_sound_heterogeneity", "limited_view",
            "acoustic_attenuation", "laser_fluence_variation",
        ],
        "reconstruction_task_types": ["image_reconstruction", "spectral_unmixing"],
        "default_solver": "back_projection",
        "evaluation_metrics": ["psnr", "ssim", "cnr"],
        "canonical_references": [
            "Wang & Yao, 'Photoacoustic microscopy and computed tomography', "
            "Nature Methods 13, 627-638 (2016)",
            "Manwar et al., 'OADAT: Optoacoustic dataset', J. Biophotonics 2024",
        ],
        "canonical_datasets": [
            "OADAT (optoacoustic benchmark)",
            "IPASC consensus datasets",
        ],
        "tags": ["medical", "photoacoustic", "hybrid", "optical_absorption", "ultrasound"],
    },

    "dot": {
        "display_name": "Diffuse Optical Tomography",
        "category": "medical",
        "description": (
            "Diffuse optical tomography (DOT) reconstructs 3D maps of tissue optical "
            "properties (absorption mu_a and reduced scattering mu_s') by measuring "
            "near-infrared light transport through highly scattering tissue. Multiple "
            "source-detector pairs on the tissue surface sample the diffuse photon "
            "field. The forward model is the diffusion equation: light propagation is "
            "modelled as a diffusive process with the photon fluence depending on the "
            "spatial distribution of mu_a and mu_s'. Reconstruction linearizes around "
            "a homogeneous background (Born/Rytov approximation) or uses nonlinear "
            "iterative methods. Applications include breast imaging and functional "
            "brain imaging (fNIRS-DOT)."
        ),
        "experimental_setup": {
            "instrument": "ISS Imagent / NIRx NIRScout",
            "wavelengths_nm": [685, 785, 830],
            "source_positions": 20,
            "detector_positions": 32,
            "total_SD_pairs": 640,
            "geometry": "circular array (breast) / cap (brain)",
            "reconstruction_grid_mm": "1x1x1 voxels, 50x50x30 mm volume",
            "modulation": "continuous-wave / frequency-domain",
            "reconstruction": "Born approximation / diffusion model",
        },
        "physics_class": "diffuse_optical",
        "forward_model_family": "diffusion_equation",
        "wave_model": "diffusive",
        "sensor_type": "apd_or_spad",
        "source_type": "nir_laser_diode",
        "geometry": "multi_source_detector",
        "noise_model": "poisson_gaussian",
        "typical_x_dims": [50, 50, 30],
        "typical_y_dims": [640],
        "typical_snr_range": [10.0, 30.0],
        "calibration_params": [
            "source_detector_positions", "coupling_coefficients",
            "background_optical_properties", "head_model",
        ],
        "mismatch_modes": [
            "coupling_variation", "position_uncertainty",
            "boundary_model_error", "physiological_noise",
        ],
        "reconstruction_task_types": ["tomographic_reconstruction", "absorption_mapping"],
        "default_solver": "born_approx",
        "evaluation_metrics": ["cnr", "spatial_resolution_mm", "localization_error"],
        "canonical_references": [
            "Arridge, 'Optical tomography in medical imaging', Inverse Problems 15, "
            "R41-R93 (1999)",
            "Boas et al., 'Imaging the body with diffuse optical tomography', "
            "IEEE Signal Processing Magazine 18, 57-75 (2001)",
        ],
        "canonical_datasets": [
            "UCL DOT phantom datasets",
            "BU fNIRS-DOT brain imaging benchmarks",
        ],
        "tags": ["medical", "optical", "diffuse", "tomography", "nir", "brain"],
    },

    # ══════════════════════════════════════════════════════════════════════════
    # MEDICAL IMAGING — Nuclear  (2 modalities)
    # ══════════════════════════════════════════════════════════════════════════

    "pet": {
        "display_name": "Positron Emission Tomography",
        "category": "medical",
        "description": (
            "PET images the 3D distribution of a positron-emitting radiotracer "
            "(e.g. 18F-FDG) by detecting coincident 511 keV annihilation photon "
            "pairs along lines of response (LORs). The forward model is a system "
            "matrix encoding the detection probability for each voxel-LOR pair, "
            "incorporating attenuation, scatter, randoms, and detector response. "
            "Reconstruction uses iterative ML-EM/OSEM algorithms with attenuation "
            "correction from co-registered CT. Low count rates yield Poisson noise; "
            "time-of-flight (TOF) information improves SNR."
        ),
        "experimental_setup": {
            "instrument": "Siemens Biograph Vision / GE Discovery MI",
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
        "physics_class": "emission_tomographic",
        "forward_model_family": "system_matrix_emission",
        "wave_model": "particle",
        "sensor_type": "scintillation_detector",
        "source_type": "radiotracer_positron",
        "geometry": "ring_tomographic",
        "noise_model": "poisson",
        "typical_x_dims": [256, 256, 256],
        "typical_y_dims": [400, 400, 47],
        "typical_snr_range": [8.0, 25.0],
        "calibration_params": [
            "normalization_sinogram", "attenuation_map",
            "scatter_correction", "dead_time_correction", "decay_correction",
        ],
        "mismatch_modes": [
            "attenuation_correction_error", "scatter_residual",
            "patient_motion", "randoms_subtraction_error",
        ],
        "reconstruction_task_types": ["emission_tomography", "denoising", "attenuation_correction"],
        "default_solver": "mlem",
        "evaluation_metrics": ["psnr", "ssim", "suv_accuracy", "cnr"],
        "canonical_references": [
            "Shepp & Vardi, 'Maximum likelihood reconstruction for emission "
            "tomography', IEEE TMI 1, 113-122 (1982)",
            "Gatidis et al., 'AutoPET Challenge 2022', MICCAI 2022",
        ],
        "canonical_datasets": [
            "AutoPET Challenge (whole-body FDG-PET/CT)",
            "TCIA PET/CT collections",
        ],
        "tags": ["medical", "nuclear", "pet", "emission", "fdg", "oncology"],
    },

    "spect": {
        "display_name": "Single Photon Emission Computed Tomography",
        "category": "medical",
        "description": (
            "SPECT images the 3D distribution of a gamma-emitting radiotracer "
            "(e.g. 99mTc-sestamibi) by detecting single photons with rotating "
            "gamma cameras equipped with parallel-hole collimators. The collimator "
            "creates a projection of the activity distribution, and multiple angles "
            "enable tomographic reconstruction. The forward model includes collimator "
            "response (depth-dependent blurring), photon attenuation, and scatter. "
            "Reconstruction uses OSEM with corrections for attenuation (AC), scatter "
            "(SC), and resolution recovery (RR)."
        ),
        "experimental_setup": {
            "instrument": "Siemens Symbia Intevo / GE NM/CT 870 CZT",
            "matrix_size": "64x64",
            "projections": 64,
            "reconstruction": "OSEM (AC+SC+RR)",
            "iterations": 8,
            "subsets": 8,
            "post_filter_fwhm_mm": 8.0,
            "isotope": "99mTc-sestamibi",
            "application": "myocardial perfusion imaging",
            "acquisition_time_per_view_s": 20,
        },
        "physics_class": "emission_tomographic",
        "forward_model_family": "collimator_projection",
        "wave_model": "particle",
        "sensor_type": "gamma_camera",
        "source_type": "radiotracer_gamma",
        "geometry": "rotational",
        "noise_model": "poisson",
        "typical_x_dims": [64, 64, 64],
        "typical_y_dims": [64, 64, 64],
        "typical_snr_range": [5.0, 20.0],
        "calibration_params": [
            "collimator_response_function", "attenuation_map",
            "scatter_window", "center_of_rotation",
        ],
        "mismatch_modes": [
            "attenuation_correction_error", "scatter_residual",
            "collimator_response_model_error", "patient_motion",
        ],
        "reconstruction_task_types": ["emission_tomography", "denoising"],
        "default_solver": "mlem",
        "evaluation_metrics": ["cnr", "uniformity", "resolution_fwhm"],
        "canonical_references": [
            "Hudson & Larkin, 'Accelerated image reconstruction using ordered subsets "
            "of projection data (OSEM)', IEEE TMI 13, 601-609 (1994)",
        ],
        "canonical_datasets": ["Clinical SPECT benchmark collections"],
        "tags": ["medical", "nuclear", "spect", "emission", "perfusion"],
    },

    # ══════════════════════════════════════════════════════════════════════════
    # MEDICAL IMAGING — MRI family  (4 modalities)
    # ══════════════════════════════════════════════════════════════════════════

    "mri": {
        "display_name": "Magnetic Resonance Imaging",
        "category": "medical",
        "description": (
            "MRI forms images by exciting hydrogen nuclei with RF pulses in a strong "
            "magnetic field (1.5-7T) and measuring the emitted RF signal with receive "
            "coils. Spatial encoding uses gradient fields to map signal frequency and "
            "phase to spatial position, acquiring data in k-space (spatial frequency "
            "domain). The forward model for parallel imaging is y_c = F_u * S_c * x + n_c "
            "where F_u is the undersampled Fourier transform, S_c are coil sensitivity "
            "maps, and n_c is complex Gaussian noise. Accelerated MRI undersamples "
            "k-space (4-8x) and uses SENSE, GRAPPA, or deep-learning (E2E-VarNet) "
            "for reconstruction."
        ),
        "experimental_setup": {
            "instrument": "Siemens MAGNETOM Prisma / GE SIGNA Premier 3T",
            "anatomy": "knee / brain",
            "matrix_size": "320x320",
            "field_strength_T": 3.0,
            "receive_coils": 15,
            "acceleration_factor": 4,
            "k_space_sampling": "variable-density random Cartesian",
            "center_fraction": 0.08,
            "sequence": "TSE (turbo spin echo)",
            "reconstruction": "SENSE / E2E-VarNet",
            "dataset": "fastMRI (knee: 1594, brain: 6970 volumes)",
        },
        "physics_class": "fourier_sampling",
        "forward_model_family": "fourier_undersampling",
        "wave_model": "em_precession",
        "sensor_type": "rf_coil",
        "source_type": "rf_pulse",
        "geometry": "k_space",
        "noise_model": "gaussian",
        "typical_x_dims": [320, 320],
        "typical_y_dims": [15, 320, 80],
        "typical_snr_range": [15.0, 35.0],
        "calibration_params": [
            "coil_sensitivity_maps", "field_inhomogeneity",
            "k_space_trajectory", "noise_covariance",
        ],
        "mismatch_modes": [
            "off_resonance", "motion", "eddy_current",
            "coil_sensitivity_error", "trajectory_error",
        ],
        "reconstruction_task_types": ["parallel_imaging", "compressed_sensing_mri", "denoising"],
        "default_solver": "sense",
        "evaluation_metrics": ["psnr", "ssim", "nrmse"],
        "canonical_references": [
            "Pruessmann et al., 'SENSE: Sensitivity encoding for fast MRI', "
            "Magnetic Resonance in Medicine 42, 952-962 (1999)",
            "Zbontar et al., 'fastMRI: An open dataset and benchmarks for "
            "accelerated MRI', arXiv:1811.08839 (2018)",
            "Sriram et al., 'End-to-End Variational Networks for Accelerated MRI "
            "Reconstruction (E2E-VarNet)', MICCAI 2020",
        ],
        "canonical_datasets": [
            "fastMRI (knee: 1594 volumes, brain: 6970 volumes)",
            "Calgary-Campinas (brain, multi-coil)",
            "SKM-TEA (Stanford knee MRI)",
        ],
        "tags": ["medical", "mri", "fourier", "k_space", "parallel_imaging"],
    },

    "fmri": {
        "display_name": "Functional MRI (BOLD)",
        "category": "medical",
        "description": (
            "Functional MRI detects neural activity indirectly via the blood-oxygen-level "
            "dependent (BOLD) contrast mechanism. Active brain regions increase local "
            "blood flow and oxygenation, altering the ratio of diamagnetic oxyhemoglobin "
            "to paramagnetic deoxyhemoglobin, causing T2* signal changes of 1-5%. "
            "Data is acquired with fast gradient-echo EPI sequences at high temporal "
            "resolution (TR 0.5-2s). The forward model includes the hemodynamic "
            "response function (HRF) convolved with neural activity. Primary "
            "challenges include physiological noise, head motion, and the low CNR "
            "of the BOLD signal."
        ),
        "experimental_setup": {
            "instrument": "Siemens MAGNETOM Prisma (HCP protocol)",
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
        "physics_class": "fourier_sampling",
        "forward_model_family": "bold_hemodynamic",
        "wave_model": "em_precession",
        "sensor_type": "rf_coil",
        "source_type": "rf_pulse",
        "geometry": "k_space",
        "noise_model": "gaussian",
        "typical_x_dims": [104, 90, 72],
        "typical_y_dims": [104, 90, 72, 1200],
        "typical_snr_range": [5.0, 20.0],
        "calibration_params": [
            "field_map", "distortion_correction",
            "motion_parameters", "physiological_regressors",
        ],
        "mismatch_modes": [
            "head_motion", "susceptibility_distortion",
            "physiological_noise", "signal_dropout",
        ],
        "reconstruction_task_types": ["epi_reconstruction", "distortion_correction", "denoising"],
        "default_solver": "sense",
        "evaluation_metrics": ["tsnr", "motion_mm", "activation_t_stat"],
        "canonical_references": [
            "Ogawa et al., 'Brain magnetic resonance imaging with contrast dependent "
            "on blood oxygenation', PNAS 87, 9868-9872 (1990)",
            "Glasser et al., 'The minimal preprocessing pipelines for the Human "
            "Connectome Project', NeuroImage 80, 105-124 (2013)",
        ],
        "canonical_datasets": [
            "Human Connectome Project (HCP) 3T (1200 subjects)",
            "UK Biobank brain imaging",
        ],
        "tags": ["medical", "fmri", "bold", "functional", "neuroscience", "brain"],
    },

    "mrs": {
        "display_name": "MR Spectroscopy",
        "category": "medical",
        "description": (
            "Magnetic resonance spectroscopy (MRS) measures the concentration of "
            "metabolites in a localized tissue volume by exploiting the chemical shift "
            "— the slight difference in Larmor frequency caused by the electronic "
            "environment of different molecular groups. The free induction decay (FID) "
            "or spin echo signal is Fourier-transformed to a spectrum where each "
            "metabolite produces characteristic peaks (e.g. NAA at 2.01 ppm, Cr at "
            "3.03 ppm). Quantification involves fitting the spectrum to a linear "
            "combination of basis spectra (LCModel, OSPREY). Challenges include low "
            "SNR, spectral overlap, water/lipid suppression, and B0 inhomogeneity "
            "causing linewidth broadening."
        ),
        "experimental_setup": {
            "instrument": "Siemens MAGNETOM Prisma 3T",
            "sequence": "PRESS (Point RESolved Spectroscopy)",
            "TE_ms": 30,
            "TR_ms": 2000,
            "voxel_size_cm3": "2x2x2 (8 mL)",
            "transients": 64,
            "metabolites": ["NAA", "Cr", "Cho", "Glx", "mI"],
            "fitting": "LCModel / OSPREY",
        },
        "physics_class": "fourier_sampling",
        "forward_model_family": "spectral_fitting",
        "wave_model": "em_precession",
        "sensor_type": "rf_coil",
        "source_type": "rf_pulse",
        "geometry": "single_voxel",
        "noise_model": "gaussian",
        "typical_x_dims": [1, 1, 1, 2048],
        "typical_y_dims": [1, 1, 1, 2048],
        "typical_snr_range": [5.0, 25.0],
        "calibration_params": [
            "water_reference", "b0_shimming",
            "eddy_current_correction", "basis_set",
        ],
        "mismatch_modes": [
            "b0_inhomogeneity", "voxel_contamination",
            "lipid_contamination", "eddy_current",
        ],
        "reconstruction_task_types": ["spectral_fitting", "metabolite_quantification"],
        "default_solver": "lcmodel",
        "evaluation_metrics": ["crlb_percent", "snr_naa", "linewidth_hz"],
        "canonical_references": [
            "Provencher, 'Estimation of metabolite concentrations from localized in "
            "vivo proton NMR spectra (LCModel)', MRM 30, 672-679 (1993)",
            "Wilson et al., 'Methodological consensus on clinical proton MRS of the "
            "brain (MRSinMRS)', NMR in Biomedicine 34, e4484 (2021)",
        ],
        "canonical_datasets": [
            "ISMRM MRS fitting challenge datasets",
            "Big GABA multi-site MRS data",
        ],
        "tags": ["medical", "spectroscopy", "metabolites", "mrs", "brain"],
    },

    "diffusion_mri": {
        "display_name": "Diffusion MRI (DTI)",
        "category": "medical",
        "description": (
            "Diffusion MRI measures the random Brownian motion of water molecules in "
            "tissue by applying magnetic field gradient pulses that encode microscopic "
            "displacement. The signal attenuation follows S = S_0 * exp(-b * D_eff) "
            "where b is the diffusion weighting factor and D_eff is the effective "
            "diffusion coefficient along the gradient direction. Acquiring measurements "
            "in multiple gradient directions enables estimation of the diffusion "
            "tensor (DTI) and derived scalar maps (FA, MD, AD, RD). Advanced models "
            "(NODDI, CSD) resolve intra-voxel fiber crossings. Primary degradations "
            "include EPI distortion, eddy currents, and motion sensitivity."
        ),
        "experimental_setup": {
            "instrument": "Siemens MAGNETOM Prisma (HCP protocol)",
            "field_strength_T": 3.0,
            "b_values": [0, 1000],
            "diffusion_directions": 64,
            "matrix_size": "128x128",
            "TR_ms": 4000,
            "TE_ms": 80,
            "voxel_size_mm": "2x2x2",
            "sequence": "single-shot spin-echo EPI DWI",
            "reconstruction": "weighted least squares / RESTORE",
            "dataset": "HCP, UK Biobank",
        },
        "physics_class": "fourier_sampling",
        "forward_model_family": "diffusion_signal_model",
        "wave_model": "em_precession",
        "sensor_type": "rf_coil",
        "source_type": "rf_pulse",
        "geometry": "k_space",
        "noise_model": "rician",
        "typical_x_dims": [128, 128, 72],
        "typical_y_dims": [65, 128, 128, 72],
        "typical_snr_range": [8.0, 25.0],
        "calibration_params": [
            "b_matrix", "gradient_nonlinearity_correction",
            "eddy_current_distortion", "susceptibility_fieldmap",
        ],
        "mismatch_modes": [
            "eddy_current_distortion", "susceptibility_artifact",
            "head_motion", "signal_dropout", "gibbs_ringing",
        ],
        "reconstruction_task_types": ["tensor_fitting", "fiber_tractography", "distortion_correction"],
        "default_solver": "weighted_least_squares",
        "evaluation_metrics": ["fa_accuracy", "md_accuracy", "tract_dice"],
        "canonical_references": [
            "Basser et al., 'MR diffusion tensor spectroscopy and imaging', "
            "Biophysical Journal 66, 259-267 (1994)",
            "Sotiropoulos et al., 'Advances in diffusion MRI acquisition and "
            "processing in the HCP', NeuroImage 80, 125-143 (2013)",
        ],
        "canonical_datasets": [
            "Human Connectome Project (HCP) diffusion data",
            "UK Biobank diffusion imaging",
        ],
        "tags": ["medical", "diffusion", "dti", "tractography", "white_matter", "brain"],
    },

    # ══════════════════════════════════════════════════════════════════════════
    # MEDICAL IMAGING — Ultrasound  (3 modalities)
    # ══════════════════════════════════════════════════════════════════════════

    "ultrasound": {
        "display_name": "Ultrasound Imaging",
        "category": "medical",
        "description": (
            "Ultrasound imaging forms images by transmitting acoustic pulses into "
            "tissue and recording echoes reflected from impedance boundaries. In "
            "ultrafast plane-wave imaging, unfocused plane waves at multiple steering "
            "angles are transmitted and the received channel data are coherently "
            "compounded using delay-and-sum (DAS) beamforming. The forward model is "
            "governed by the acoustic wave equation with tissue-dependent speed of "
            "sound and attenuation. Primary degradations include speckle noise "
            "(coherent interference), limited bandwidth, and aberration from "
            "heterogeneous tissue."
        ),
        "experimental_setup": {
            "instrument": "Verasonics Vantage 256 / GE LOGIQ E10",
            "probe": "L11-5v linear array (128 elements)",
            "center_frequency_MHz": 5.21,
            "plane_wave_angles": 11,
            "compound_frame_rate_Hz": 1000,
            "imaging_depth_mm": 40,
            "speed_of_sound_m_s": 1540,
            "lateral_resolution_mm": 0.3,
            "axial_resolution_mm": 0.15,
            "dataset": "PICMUS Challenge (IEEE IUS)",
        },
        "physics_class": "acoustic",
        "forward_model_family": "acoustic_wave_equation",
        "wave_model": "acoustic",
        "sensor_type": "piezoelectric_array",
        "source_type": "piezoelectric_transducer",
        "geometry": "linear_array",
        "noise_model": "speckle",
        "typical_x_dims": [256, 256],
        "typical_y_dims": [128, 2048],
        "typical_snr_range": [15.0, 40.0],
        "calibration_params": [
            "speed_of_sound", "element_positions",
            "element_directivity", "time_delay_calibration",
        ],
        "mismatch_modes": [
            "speed_of_sound_error", "phase_aberration",
            "element_failure", "grating_lobes",
        ],
        "reconstruction_task_types": ["beamforming", "speckle_reduction", "super_resolution"],
        "default_solver": "tv_fista",
        "evaluation_metrics": ["psnr", "ssim", "cnr", "lateral_resolution_mm"],
        "canonical_references": [
            "Montaldo et al., 'Coherent plane-wave compounding for very high frame "
            "rate ultrasonography', IEEE TUFFC 56, 489-506 (2009)",
            "Liebgott et al., 'PICMUS: Plane-wave Imaging Challenge in Medical "
            "Ultrasound', IEEE IUS 2016",
        ],
        "canonical_datasets": [
            "PICMUS Challenge (plane-wave ultrasound)",
            "CUBDL (deep learning ultrasound beamforming)",
        ],
        "tags": ["medical", "ultrasound", "acoustic", "beamforming", "plane_wave"],
    },

    "doppler_ultrasound": {
        "display_name": "Doppler Ultrasound",
        "category": "medical",
        "description": (
            "Doppler ultrasound measures blood flow velocity by detecting the "
            "frequency shift of ultrasound echoes reflected from moving red blood "
            "cells. The Doppler shift f_d = 2*f_0*v*cos(theta)/c relates velocity v "
            "to the observed frequency shift. Color Doppler maps 2D velocity fields "
            "by applying autocorrelation estimators to ensembles of pulse-echo "
            "data at each spatial location. A wall filter (high-pass) separates slow "
            "tissue clutter from blood flow signals. Challenges include aliasing when "
            "velocity exceeds the Nyquist limit (PRF/2) and angle-dependence of the "
            "velocity estimate."
        ),
        "experimental_setup": {
            "instrument": "GE LOGIQ E10 / Philips EPIQ Elite",
            "probe_frequency_MHz": 5.0,
            "PRF_kHz": 10.0,
            "ensemble_length": 16,
            "wall_filter": "polynomial regression / SVD clutter filter",
            "velocity_range_cm_s": "0-200",
            "application": "carotid / renal flow imaging",
        },
        "physics_class": "acoustic",
        "forward_model_family": "doppler_frequency_shift",
        "wave_model": "acoustic",
        "sensor_type": "piezoelectric_array",
        "source_type": "piezoelectric_transducer",
        "geometry": "linear_array",
        "noise_model": "speckle",
        "typical_x_dims": [256, 256],
        "typical_y_dims": [16, 128, 2048],
        "typical_snr_range": [10.0, 30.0],
        "calibration_params": [
            "prf", "doppler_angle", "wall_filter_cutoff",
            "velocity_scale", "beam_steering_angle",
        ],
        "mismatch_modes": [
            "aliasing", "angle_dependence", "clutter_residual",
            "spectral_broadening", "wall_filter_artifact",
        ],
        "reconstruction_task_types": ["velocity_estimation", "clutter_filtering"],
        "default_solver": "autocorrelation_estimator",
        "evaluation_metrics": ["velocity_rmse", "clutter_rejection_dB", "aliasing_rate"],
        "canonical_references": [
            "Kasai et al., 'Real-time two-dimensional blood flow imaging using an "
            "autocorrelation technique', IEEE Trans. Sonics Ultrasonics 32, 458-464 (1985)",
        ],
        "canonical_datasets": ["Clinical Doppler benchmark collections"],
        "tags": ["medical", "ultrasound", "doppler", "flow", "velocity"],
    },

    "elastography": {
        "display_name": "Shear-Wave Elastography",
        "category": "medical",
        "description": (
            "Shear-wave elastography (SWE) quantifies tissue stiffness by generating "
            "shear waves using an acoustic radiation force impulse (ARFI) push and "
            "tracking their propagation with ultrafast ultrasound imaging (10,000+ "
            "fps). The shear wave speed c_s is related to the shear modulus by "
            "mu = rho * c_s^2, enabling quantitative mapping of Young's modulus "
            "E = 3*mu (assuming incompressibility). The technique is clinically "
            "validated for liver fibrosis staging (F0-F4) and breast lesion "
            "characterization. Challenges include shear wave attenuation in deep "
            "tissue and reflections from boundaries."
        ),
        "experimental_setup": {
            "instrument": "Supersonic Imagine Aixplorer MACH 30 / Siemens ACUSON Sequoia",
            "probe_frequency_MHz": 4.0,
            "push_frequency_Hz": 50,
            "shear_wave_speed_range_m_s": "1-5",
            "method": "ARFI / supersonic shear imaging (SSI)",
            "stiffness_range_kPa": "1-75",
            "ultrafast_frame_rate_fps": 10000,
            "application": "liver fibrosis staging",
        },
        "physics_class": "acoustic",
        "forward_model_family": "shear_wave_propagation",
        "wave_model": "acoustic_shear",
        "sensor_type": "piezoelectric_array",
        "source_type": "arfi_push",
        "geometry": "linear_array",
        "noise_model": "gaussian",
        "typical_x_dims": [128, 128],
        "typical_y_dims": [100, 128, 2048],
        "typical_snr_range": [10.0, 30.0],
        "calibration_params": [
            "push_sequence", "tracking_prf",
            "tissue_density", "speed_estimation_kernel",
        ],
        "mismatch_modes": [
            "shear_wave_attenuation", "boundary_reflection",
            "tissue_viscosity", "push_beam_artifact",
        ],
        "reconstruction_task_types": ["shear_wave_speed_estimation", "stiffness_mapping"],
        "default_solver": "time_of_flight_inversion",
        "evaluation_metrics": ["stiffness_kpa_accuracy", "reproducibility_cv"],
        "canonical_references": [
            "Bercoff et al., 'Supersonic shear imaging: a new technique for soft "
            "tissue elasticity mapping', IEEE TUFFC 51, 396-409 (2004)",
            "Barr et al., 'Elastography assessment of liver fibrosis', "
            "Radiology 276, 845-861 (2015)",
        ],
        "canonical_datasets": ["Clinical SWE liver fibrosis benchmark data"],
        "tags": ["medical", "ultrasound", "elastography", "stiffness", "liver_fibrosis"],
    },

    # ══════════════════════════════════════════════════════════════════════════
    # COHERENT IMAGING  (3 modalities)
    # ══════════════════════════════════════════════════════════════════════════

    "ptychography": {
        "display_name": "Ptychographic Imaging",
        "category": "coherent",
        "description": (
            "Ptychography is a scanning coherent diffractive imaging technique where "
            "a coherent beam (X-ray or electron) illuminates overlapping regions of "
            "the sample and far-field diffraction patterns are recorded at each scan "
            "position. The overlap between adjacent probe positions provides "
            "redundancy that enables simultaneous recovery of the complex-valued "
            "object transmission function and the illumination probe via iterative "
            "algorithms (ePIE, difference map). The forward model at each position "
            "is I_j = |F{P(r-r_j) * O(r)}|^2 where P is the probe and O is the "
            "object. Achievable resolution is limited by the detector NA, not the "
            "optics, reaching sub-10 nm for X-rays."
        ),
        "experimental_setup": {
            "instrument": "Diamond Light Source I13 / APS 2-ID / ESRF ID16A",
            "photon_energy_keV": 12.4,
            "wavelength_nm": 0.1,
            "detector": "Eiger 500K (512x512 px, 75 um pitch)",
            "probe_size_um": 1.0,
            "step_size_nm": 200,
            "overlap_ratio": 0.70,
            "propagation_distance_m": 2.1,
            "achieved_resolution_nm": 10,
            "reconstruction": "ePIE / difference map / Adam-based optimization",
        },
        "physics_class": "coherent_diffraction",
        "forward_model_family": "ptychographic_forward",
        "wave_model": "scalar_wave",
        "sensor_type": "photon_counter",
        "source_type": "coherent_beam",
        "geometry": "ptychographic_scan",
        "noise_model": "poisson",
        "typical_x_dims": [512, 512],
        "typical_y_dims": [64, 128, 128],
        "typical_snr_range": [10.0, 30.0],
        "calibration_params": [
            "probe_function", "scan_positions", "detector_distance",
            "wavelength", "pixel_size",
        ],
        "mismatch_modes": [
            "position_error", "partial_coherence",
            "probe_drift", "detector_saturation",
        ],
        "reconstruction_task_types": ["phase_retrieval", "probe_recovery"],
        "default_solver": "epie",
        "evaluation_metrics": ["psnr", "ssim", "phase_error", "frc"],
        "canonical_references": [
            "Rodenburg & Faulkner, 'A phase retrieval algorithm for shifting "
            "illumination (ePIE)', Appl. Phys. Lett. 85, 4795-4797 (2004)",
            "Thibault et al., 'High-resolution scanning X-ray diffraction microscopy', "
            "Science 321, 379-382 (2008)",
        ],
        "canonical_datasets": [
            "PtychoNN benchmark datasets (Cherukara et al.)",
            "Diamond I13 ptychography test data",
        ],
        "tags": ["coherent", "phase_retrieval", "scanning", "xray", "nanoscale"],
    },

    "holography": {
        "display_name": "Digital Holographic Microscopy",
        "category": "coherent",
        "description": (
            "Digital holographic microscopy (DHM) records the interference pattern "
            "between an object wave (scattered by the sample) and a reference wave "
            "on a digital sensor. The hologram encodes both amplitude and phase "
            "of the object wavefield. In off-axis configuration, the object spectrum "
            "is separated from the zero-order and twin-image terms in Fourier space. "
            "Numerical propagation (angular spectrum method) refocuses the wavefield "
            "at any desired plane, enabling quantitative phase imaging (QPI) with "
            "nanometer path-length sensitivity. Applications include label-free "
            "cell imaging and topography measurement."
        ),
        "experimental_setup": {
            "instrument": "Lyncee Tec DHM T1000 / custom Mach-Zehnder setup",
            "wavelength_nm": 532,
            "pixel_size_um": 3.45,
            "sensor": "sCMOS 2048x2048 (Hamamatsu ORCA-Flash4.0)",
            "propagation_distance_mm": 100,
            "coherence_length_mm": ">1 (laser source)",
            "reconstruction": "angular spectrum method",
            "application": "quantitative phase imaging (QPI)",
        },
        "physics_class": "interferometric",
        "forward_model_family": "holographic_forward",
        "wave_model": "scalar_wave",
        "sensor_type": "cmos",
        "source_type": "laser",
        "geometry": "planar",
        "noise_model": "gaussian",
        "typical_x_dims": [512, 512],
        "typical_y_dims": [512, 512],
        "typical_snr_range": [20.0, 45.0],
        "calibration_params": [
            "wavelength", "propagation_distance",
            "reference_beam_angle", "pixel_size",
        ],
        "mismatch_modes": [
            "twin_image", "reference_error",
            "coherence_loss", "vibration",
        ],
        "reconstruction_task_types": ["phase_retrieval", "numerical_refocusing"],
        "default_solver": "angular_spectrum",
        "evaluation_metrics": ["psnr", "ssim", "phase_error_rad"],
        "canonical_references": [
            "Cuche et al., 'Digital holography for quantitative phase-contrast imaging', "
            "Optics Letters 24, 291-293 (1999)",
            "Kim, 'Principles and techniques of digital holographic microscopy', "
            "SPIE Reviews 1, 018005 (2010)",
        ],
        "canonical_datasets": [
            "Lyncee Tec DHM application datasets",
            "HoloGAN benchmark (simulated holograms)",
        ],
        "tags": ["coherent", "interferometric", "phase", "holography", "qpi"],
    },

    "phase_retrieval": {
        "display_name": "Coherent Diffractive Imaging / Phase Retrieval",
        "category": "coherent",
        "description": (
            "Coherent diffractive imaging (CDI) recovers the complex-valued exit "
            "wave from a coherent scattering experiment where only the diffraction "
            "intensity |F{O}|^2 is measured (the phase is lost). Phase retrieval "
            "algorithms (HIO + ER, Fienup) iteratively enforce constraints in both "
            "real space (finite support, non-negativity) and reciprocal space "
            "(measured intensity). The oversampling condition (sampling at least 2x "
            "the Nyquist rate) ensures sufficient information for unique phase "
            "recovery. CDI achieves diffraction-limited resolution without imaging "
            "optics. Applications include imaging of nanocrystals, viruses, and "
            "materials at X-ray and electron wavelengths."
        ),
        "experimental_setup": {
            "instrument": "LCLS XFEL / APS coherent scattering beamline",
            "accelerating_voltage_kV": 300,
            "wavelength_pm": 1.97,
            "detector": "CSPAD / Jungfrau 4M (direct detection)",
            "oversampling_ratio": 4,
            "resolution_nm": 2,
            "reconstruction": "HIO + ER (hybrid input-output + error reduction)",
        },
        "physics_class": "coherent_diffraction",
        "forward_model_family": "fourier_magnitude",
        "wave_model": "scalar_wave",
        "sensor_type": "photon_counter",
        "source_type": "coherent_beam",
        "geometry": "far_field",
        "noise_model": "poisson",
        "typical_x_dims": [512, 512],
        "typical_y_dims": [1024, 1024],
        "typical_snr_range": [5.0, 25.0],
        "calibration_params": [
            "support_mask", "beam_stop_mask",
            "wavelength", "detector_distance",
        ],
        "mismatch_modes": [
            "support_error", "partial_coherence",
            "missing_center", "detector_gap",
        ],
        "reconstruction_task_types": ["phase_retrieval", "support_estimation"],
        "default_solver": "hio",
        "evaluation_metrics": ["phase_error", "r_factor", "frc", "prtf"],
        "canonical_references": [
            "Miao et al., 'Extending the methodology of X-ray crystallography to "
            "non-crystalline specimens', Nature 400, 342-344 (1999)",
            "Fienup, 'Phase retrieval algorithms: a comparison', "
            "Applied Optics 21, 2758-2769 (1982)",
        ],
        "canonical_datasets": [
            "CXIDB (Coherent X-ray Imaging Data Bank)",
            "Simulated CDI benchmark (Marchesini et al.)",
        ],
        "tags": ["coherent", "phase_retrieval", "lensless", "cdi", "xfel"],
    },

    # ══════════════════════════════════════════════════════════════════════════
    # NEURAL RENDERING  (2 modalities)
    # ══════════════════════════════════════════════════════════════════════════

    "nerf": {
        "display_name": "Neural Radiance Fields (NeRF)",
        "category": "neural_rendering",
        "description": (
            "Neural radiance fields (NeRF) represent a 3D scene as a continuous "
            "volumetric function F(x,y,z,theta,phi) -> (RGB, sigma) parameterized "
            "by a multi-layer perceptron that maps 5D coordinates (position + viewing "
            "direction) to color and volume density. Novel views are synthesized by "
            "marching camera rays through the volume and integrating color weighted "
            "by transmittance using quadrature. Training optimizes the MLP weights to "
            "minimize photometric loss between rendered and observed images. Primary "
            "challenges include slow training/rendering, view-dependent effects, and "
            "the need for accurate camera poses (from COLMAP)."
        ),
        "experimental_setup": {
            "training_views": 100,
            "test_views": 200,
            "image_resolution": "800x800",
            "scene_type": "object-centric 360 deg (Blender synthetic)",
            "architecture": "positional encoding + MLP (8 layers, 256 hidden)",
            "training_iterations": 200000,
            "batch_size_rays": 4096,
            "evaluation": "PSNR / SSIM / LPIPS",
            "dataset": "Blender Synthetic (8 scenes), LLFF (8 forward-facing)",
        },
        "physics_class": "neural_volume",
        "forward_model_family": "volumetric_rendering_integral",
        "wave_model": "ray",
        "sensor_type": "rgb_camera",
        "source_type": "ambient",
        "geometry": "multi_view",
        "noise_model": "gaussian",
        "typical_x_dims": [800, 800, 3],
        "typical_y_dims": [100, 800, 800, 3],
        "typical_snr_range": [25.0, 45.0],
        "calibration_params": [
            "camera_intrinsics", "camera_extrinsics",
            "scene_scale", "near_far_bounds",
        ],
        "mismatch_modes": [
            "pose_error", "exposure_variation",
            "transient_objects", "unbounded_scenes",
        ],
        "reconstruction_task_types": ["novel_view_synthesis", "3d_reconstruction"],
        "default_solver": "nerf_mlp",
        "evaluation_metrics": ["psnr", "ssim", "lpips"],
        "canonical_references": [
            "Mildenhall et al., 'NeRF: Representing scenes as neural radiance fields "
            "for view synthesis', ECCV 2020",
            "Muller et al., 'Instant Neural Graphics Primitives (Instant-NGP)', "
            "SIGGRAPH 2022",
        ],
        "canonical_datasets": [
            "NeRF Blender Synthetic (8 scenes)",
            "LLFF (8 forward-facing scenes)",
            "Mip-NeRF 360 (9 unbounded scenes)",
        ],
        "tags": ["neural_rendering", "nerf", "3d", "view_synthesis", "volumetric"],
    },

    "gaussian_splatting": {
        "display_name": "3D Gaussian Splatting",
        "category": "neural_rendering",
        "description": (
            "3D Gaussian splatting represents scenes as a collection of learnable "
            "3D Gaussian primitives, each parameterized by position, covariance "
            "(anisotropic 3D extent), opacity, and spherical harmonic color "
            "coefficients. Rendering rasterizes the Gaussians by projecting them to "
            "2D screen space, sorting by depth, and alpha-compositing with a "
            "tile-based differentiable rasterizer. Training optimizes Gaussian "
            "parameters via gradient descent with adaptive density control "
            "(splitting, cloning, pruning). This achieves real-time (30+ fps) "
            "rendering at quality comparable to NeRF, from SfM point cloud "
            "initialization (COLMAP)."
        ),
        "experimental_setup": {
            "training_views": "24-300 (scene-dependent)",
            "image_resolution": "~1600x1200",
            "initialization": "SfM point cloud (COLMAP)",
            "rendering_fps": 30,
            "scene_type": "unbounded indoor / outdoor",
            "training_iterations": 30000,
            "evaluation": "PSNR / SSIM / LPIPS",
            "dataset": "Mip-NeRF360, Tanks & Temples, Deep Blending",
        },
        "physics_class": "neural_volume",
        "forward_model_family": "gaussian_rasterization",
        "wave_model": "ray",
        "sensor_type": "rgb_camera",
        "source_type": "ambient",
        "geometry": "multi_view",
        "noise_model": "gaussian",
        "typical_x_dims": [1600, 1200, 3],
        "typical_y_dims": [300, 1600, 1200, 3],
        "typical_snr_range": [25.0, 45.0],
        "calibration_params": [
            "camera_intrinsics", "camera_extrinsics",
            "sfm_point_cloud", "scene_scale",
        ],
        "mismatch_modes": [
            "pose_error", "sparse_initialization",
            "floater_artifacts", "popping_artifacts",
        ],
        "reconstruction_task_types": ["novel_view_synthesis", "3d_reconstruction"],
        "default_solver": "gaussian_splatting_3dgs",
        "evaluation_metrics": ["psnr", "ssim", "lpips", "rendering_fps"],
        "canonical_references": [
            "Kerbl et al., '3D Gaussian Splatting for Real-Time Radiance Field "
            "Rendering', SIGGRAPH 2023",
        ],
        "canonical_datasets": [
            "Mip-NeRF 360 (9 scenes)",
            "Tanks & Temples (Knapitsch et al.)",
            "Deep Blending (Hedman et al.)",
        ],
        "tags": ["neural_rendering", "gaussian_splatting", "3d", "real_time", "point_based"],
    },

    # ══════════════════════════════════════════════════════════════════════════
    # COMPUTATIONAL PHOTOGRAPHY  (3 modalities)
    # ══════════════════════════════════════════════════════════════════════════

    "panorama": {
        "display_name": "Panorama Multi-Focus Fusion",
        "category": "computational",
        "description": (
            "Multi-focus panoramic fusion combines images captured at different focal "
            "planes and/or different spatial positions to produce an all-in-focus "
            "image with extended depth of field and wide field of view. Focus stacking "
            "selects the sharpest regions from each focal plane using local contrast "
            "measures, then blends them via Laplacian pyramid fusion or wavelet-based "
            "methods. Panoramic stitching aligns overlapping images using feature "
            "matching (SIFT/SURF) and blends seams. Primary challenges include "
            "parallax at scene edges and focus measure ambiguity in low-texture regions."
        ),
        "experimental_setup": {
            "image_size": "4096x2048 (equirectangular)",
            "focus_planes": 6,
            "overlap_percent": 30,
            "fusion": "Laplacian pyramid / wavelet fusion",
            "application": "all-in-focus / extended depth of field",
        },
        "physics_class": "multi_focus",
        "forward_model_family": "defocus_stack",
        "wave_model": "incoherent",
        "sensor_type": "cmos",
        "source_type": "ambient",
        "geometry": "planar",
        "noise_model": "gaussian",
        "typical_x_dims": [4096, 2048, 3],
        "typical_y_dims": [6, 4096, 2048, 3],
        "typical_snr_range": [25.0, 45.0],
        "calibration_params": [
            "focal_distances", "camera_intrinsics",
            "overlap_registration", "vignetting_correction",
        ],
        "mismatch_modes": [
            "parallax_error", "registration_error",
            "exposure_variation", "ghost_from_motion",
        ],
        "reconstruction_task_types": ["focus_fusion", "panoramic_stitching"],
        "default_solver": "laplacian_pyramid_fusion",
        "evaluation_metrics": ["psnr", "ssim", "edge_sharpness"],
        "canonical_references": [
            "Burt & Adelson, 'The Laplacian Pyramid as a Compact Image Code', "
            "IEEE Trans. Commun. 31, 532-540 (1983)",
        ],
        "canonical_datasets": ["Lytro multi-focus test set"],
        "tags": ["computational", "panorama", "fusion", "focus_stacking", "extended_dof"],
    },

    "light_field": {
        "display_name": "Light Field Imaging",
        "category": "computational",
        "description": (
            "Light field imaging captures the full 4D radiance function L(x,y,u,v) "
            "describing both spatial position (x,y) and angular direction (u,v) of "
            "light rays. A microlens array placed before the sensor captures multiple "
            "sub-aperture views simultaneously, enabling post-capture refocusing, "
            "depth estimation, and perspective shifts. Each microlens images the "
            "objective's exit pupil, trading spatial resolution for angular resolution. "
            "The 4D light field can be processed with shift-and-sum for refocusing, "
            "disparity estimation for depth, or epipolar-plane image (EPI) analysis. "
            "Primary challenges include the inherent spatial-angular resolution tradeoff "
            "and microlens aberrations."
        ),
        "experimental_setup": {
            "instrument": "Lytro Illum / Raytrix R42",
            "micro_lens_pitch_um": 14,
            "angular_resolution": "9x9 (HCI) / 15x15 (Lytro Illum)",
            "total_sensor_px": "7728x5368",
            "spatial_per_view": "434x625",
            "dataset": "HCI 4D LF Benchmark, Stanford Lego Gantry",
        },
        "physics_class": "light_field",
        "forward_model_family": "plenoptic_sampling",
        "wave_model": "ray",
        "sensor_type": "cmos_with_microlens",
        "source_type": "ambient",
        "geometry": "multi_view",
        "noise_model": "gaussian",
        "typical_x_dims": [512, 512, 9, 9],
        "typical_y_dims": [7728, 5368],
        "typical_snr_range": [20.0, 40.0],
        "calibration_params": [
            "microlens_calibration", "pixel_to_ray_mapping",
            "vignetting_correction", "white_balance",
        ],
        "mismatch_modes": [
            "microlens_crosstalk", "vignetting",
            "depth_range_limitation", "angular_aliasing",
        ],
        "reconstruction_task_types": ["depth_estimation", "refocusing", "view_synthesis"],
        "default_solver": "shift_and_sum",
        "evaluation_metrics": ["psnr", "ssim", "depth_mse", "badpix_007"],
        "canonical_references": [
            "Levoy & Hanrahan, 'Light field rendering', SIGGRAPH 1996",
            "Ng et al., 'Light field photography with a hand-held plenoptic camera', "
            "Stanford Tech Report CTSR 2005-02",
        ],
        "canonical_datasets": [
            "HCI 4D Light Field Benchmark",
            "Stanford Lego Gantry Archive",
            "INRIA Lytro Light Field Dataset",
        ],
        "tags": ["computational", "light_field", "plenoptic", "depth", "refocusing"],
    },

    "integral": {
        "display_name": "Integral Photography",
        "category": "computational",
        "description": (
            "Integral photography (IP), originally proposed by Lippmann in 1908, "
            "captures a light field using a fly-eye lens array (matrix of small "
            "lenses) where each lenslet records a small elemental image from a "
            "slightly different perspective. The array of elemental images encodes "
            "3D scene information, enabling computational refocusing, depth estimation, "
            "and autostereoscopic 3D display. Compared to microlens-based plenoptic "
            "cameras, IP typically uses larger lenslets with correspondingly more "
            "pixels per lens. Reconstruction includes depth-from-correspondence "
            "between elemental images and 3D focal stack computation."
        ),
        "experimental_setup": {
            "instrument": "Custom integral imaging setup / ETRI prototype",
            "micro_lens_pitch_mm": 1.0,
            "micro_lens_NA": 0.16,
            "sensor_pixel_um": 5.5,
            "pixels_per_lens": "20x20",
            "reconstruction": "3D focal-stack / depth estimation",
        },
        "physics_class": "light_field",
        "forward_model_family": "elemental_image_formation",
        "wave_model": "ray",
        "sensor_type": "cmos",
        "source_type": "ambient",
        "geometry": "multi_view",
        "noise_model": "gaussian",
        "typical_x_dims": [512, 512, 64],
        "typical_y_dims": [512, 512],
        "typical_snr_range": [22.0, 42.0],
        "calibration_params": [
            "microlens_pitch", "microlens_focal_length",
            "sensor_pixel_size", "display_gap",
        ],
        "mismatch_modes": [
            "microlens_alignment", "crosstalk", "fill_factor_loss",
            "field_curvature", "depth_reversal",
        ],
        "reconstruction_task_types": ["depth_estimation", "3d_focal_stack", "perspective_synthesis"],
        "default_solver": "depth_estimation",
        "evaluation_metrics": ["psnr", "ssim", "depth_mae"],
        "canonical_references": [
            "Lippmann, C. R. Acad. Sci. Paris 146, 446 (1908)",
            "Park et al., 'Recent progress in 3D imaging systems', "
            "J. Opt. Soc. Am. A 26, 2538 (2009)",
        ],
        "canonical_datasets": [
            "ETRI integral imaging test set",
            "Middlebury multi-view stereo (adapted)",
        ],
        "tags": ["computational", "integral", "multi_view", "3d_display", "depth"],
    },

    # ══════════════════════════════════════════════════════════════════════════
    # CLINICAL OPTICS  (4 modalities)
    # ══════════════════════════════════════════════════════════════════════════

    "oct": {
        "display_name": "Optical Coherence Tomography",
        "category": "clinical_optics",
        "description": (
            "OCT is a low-coherence interferometric imaging technique that measures "
            "depth-resolved backscattering profiles (A-scans) by interfering "
            "sample-arm reflections with a reference mirror. In spectral-domain OCT, "
            "the interference spectrum is recorded by a spectrometer and the axial "
            "profile is obtained via Fourier transform. Axial resolution is determined "
            "by the source bandwidth (typically 3-7 um in tissue) and imaging depth "
            "by spectrometer resolution. Dominant artifacts include speckle noise, "
            "motion artifacts, and sensitivity roll-off with depth."
        ),
        "experimental_setup": {
            "instrument": "Heidelberg Spectralis HRA+OCT / Zeiss Cirrus HD-OCT 5000",
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
        "physics_class": "interferometric",
        "forward_model_family": "low_coherence_interferometry",
        "wave_model": "scalar_wave",
        "sensor_type": "spectrometer",
        "source_type": "low_coherence",
        "geometry": "tomographic_axial",
        "noise_model": "speckle",
        "typical_x_dims": [512, 512, 512],
        "typical_y_dims": [512, 512, 1024],
        "typical_snr_range": [15.0, 40.0],
        "calibration_params": [
            "wavelength_axis", "dispersion_coefficients",
            "reference_delay", "sensitivity_roll_off",
        ],
        "mismatch_modes": [
            "dispersion_mismatch", "motion_artifact",
            "sensitivity_rolloff", "mirror_artifact",
        ],
        "reconstruction_task_types": ["axial_reconstruction", "speckle_reduction", "segmentation"],
        "default_solver": "fft_recon",
        "evaluation_metrics": ["psnr", "ssim", "cnr", "layer_segmentation_dice"],
        "canonical_references": [
            "Huang et al., 'Optical coherence tomography', Science 254, 1178 (1991)",
            "de Boer et al., 'Twenty-five years of OCT', "
            "Biomed. Opt. Express 8, 3248 (2017)",
        ],
        "canonical_datasets": [
            "Duke SD-OCT DME dataset (Chiu et al.)",
            "RETOUCH Challenge (retinal OCT)",
            "OCTA-500 (Li et al., Scientific Data 2024)",
        ],
        "tags": ["clinical", "oct", "retinal", "interferometric", "depth_resolved"],
    },

    "octa": {
        "display_name": "OCT Angiography",
        "category": "clinical_optics",
        "description": (
            "OCT angiography extends standard OCT by acquiring repeated B-scans at "
            "the same location and computing the decorrelation of the complex OCT "
            "signal between successive scans. Moving red blood cells cause temporal "
            "fluctuations that differ from static tissue, enabling label-free "
            "visualization of retinal vasculature. The contrast mechanism uses "
            "amplitude decorrelation (SSADA), phase variance, or complex-signal "
            "algorithms. Key limitations include motion artifacts, projection "
            "artifacts from superficial vessels, and limited field of view."
        ),
        "experimental_setup": {
            "instrument": "Zeiss PLEX Elite 9000 / Optovue AngioVue",
            "wavelength_nm": 840,
            "A_scan_rate_kHz": 68,
            "scan_pattern": "6x6 mm",
            "repeated_B_scans": 4,
            "en_face_resolution_um": 15,
            "algorithm": "SSADA / OCTA ratio",
        },
        "physics_class": "interferometric",
        "forward_model_family": "decorrelation_contrast",
        "wave_model": "scalar_wave",
        "sensor_type": "spectrometer",
        "source_type": "low_coherence",
        "geometry": "tomographic_axial",
        "noise_model": "speckle",
        "typical_x_dims": [304, 304, 640],
        "typical_y_dims": [304, 304, 4, 1024],
        "typical_snr_range": [12.0, 35.0],
        "calibration_params": [
            "interscan_time", "decorrelation_threshold",
            "layer_segmentation", "bulk_motion_correction",
        ],
        "mismatch_modes": [
            "bulk_motion", "projection_artifact",
            "shadow_artifact", "saccade_artifact",
        ],
        "reconstruction_task_types": ["vessel_segmentation", "flow_quantification"],
        "default_solver": "ssada",
        "evaluation_metrics": ["vessel_density", "faz_area", "dice_coefficient"],
        "canonical_references": [
            "Jia et al., 'Split-spectrum amplitude-decorrelation angiography (SSADA)', "
            "Opt. Express 20, 4710 (2012)",
            "Spaide et al., 'OCT Angiography', Prog. Retin. Eye Res. 64, 1 (2018)",
        ],
        "canonical_datasets": [
            "OCTA-500 (Li et al., Scientific Data 2024)",
            "ROSE retinal OCTA vessel segmentation",
        ],
        "tags": ["clinical", "oct", "angiography", "vascular", "retinal"],
    },

    "fundus": {
        "display_name": "Fundus Camera",
        "category": "clinical_optics",
        "description": (
            "A fundus camera captures a 2D color photograph of the retinal surface "
            "by illuminating the fundus through the pupil with a ring-shaped flash "
            "and imaging the reflected light through the central pupillary zone. "
            "The optical system images the curved retina onto a flat detector with "
            "30-50 degree field of view. Image quality is degraded by media opacities "
            "(cataract), small pupil, and uneven illumination. Fundus images are "
            "widely used for automated screening of diabetic retinopathy, glaucoma, "
            "and AMD via deep learning."
        ),
        "experimental_setup": {
            "instrument": "Topcon TRC-NW400 / Canon CR-2 AF",
            "image_size": "2124x2056",
            "field_of_view_deg": 45,
            "wavelength_range_nm": "500-700",
            "flash_exposure_ms": 0.04,
            "dataset": "EyePACS, DRIVE, MESSIDOR, APTOS",
        },
        "physics_class": "imaging",
        "forward_model_family": "lens_imaging",
        "wave_model": "ray",
        "sensor_type": "cmos",
        "source_type": "flash",
        "geometry": "planar",
        "noise_model": "gaussian",
        "typical_x_dims": [2124, 2056, 3],
        "typical_y_dims": [2124, 2056, 3],
        "typical_snr_range": [25.0, 45.0],
        "calibration_params": [
            "field_of_view", "pupil_diameter",
            "illumination_uniformity", "white_balance",
        ],
        "mismatch_modes": [
            "media_opacity", "uneven_illumination",
            "small_pupil", "motion_blur",
        ],
        "reconstruction_task_types": ["vessel_segmentation", "disease_classification", "enhancement"],
        "default_solver": "richardson_lucy",
        "evaluation_metrics": ["auc_roc", "sensitivity", "specificity", "dice_coefficient"],
        "canonical_references": [
            "Gulshan et al., 'Development and validation of a deep learning algorithm "
            "for detection of diabetic retinopathy', JAMA 316, 2402 (2016)",
            "Staal et al., 'Ridge-based vessel segmentation (DRIVE)', "
            "IEEE TMI 23, 501 (2004)",
        ],
        "canonical_datasets": [
            "EyePACS (diabetic retinopathy screening)",
            "DRIVE (Digital Retinal Images for Vessel Extraction)",
            "MESSIDOR-2",
            "APTOS 2019 Blindness Detection",
        ],
        "tags": ["clinical", "retinal", "fundus", "screening", "ophthalmology"],
    },

    "endoscopy": {
        "display_name": "Fiber Bundle Endoscopy",
        "category": "clinical_optics",
        "description": (
            "Fiber bundle endoscopy transmits images through a coherent fiber bundle "
            "of 10,000-50,000 individual optical fibers. Each fiber core acts as a "
            "spatial sample, producing a honeycomb pattern. Image quality is limited "
            "by inter-core spacing (pixelation), inter-core coupling (crosstalk), and "
            "core-to-core transmission variation. White-light or narrow-band "
            "illumination is delivered through the bundle or alongside it. "
            "Reconstruction involves core localization, transmission calibration, "
            "interpolation to a regular grid, and denoising."
        ),
        "experimental_setup": {
            "instrument": "Olympus GIF-H290Z / Karl Storz IMAGE1 S",
            "fiber_cores": 30000,
            "resolution": "1920x1080 (HD output)",
            "frame_rate_fps": 60,
            "field_of_view_deg": 140,
            "working_channel_mm": 3.7,
            "wavelength_range_nm": "400-700 (white light)",
            "dataset": "Kvasir, CVC-ClinicDB, HyperKvasir",
        },
        "physics_class": "fiber_bundle",
        "forward_model_family": "fiber_sampling",
        "wave_model": "ray",
        "sensor_type": "cmos",
        "source_type": "led",
        "geometry": "planar",
        "noise_model": "poisson_gaussian",
        "typical_x_dims": [512, 512, 3],
        "typical_y_dims": [512, 512, 3],
        "typical_snr_range": [20.0, 38.0],
        "calibration_params": [
            "core_map", "transmission_calibration",
            "distortion_coefficients", "flat_field",
        ],
        "mismatch_modes": [
            "core_crosstalk", "fixed_pattern_noise",
            "bending_loss", "specular_reflection",
        ],
        "reconstruction_task_types": ["honeycomb_removal", "super_resolution", "polyp_detection"],
        "default_solver": "tv_fista",
        "evaluation_metrics": ["psnr", "ssim", "polyp_detection_ap"],
        "canonical_references": [
            "Lee & Bhatt, 'Fiber bundle endoscopy advances', "
            "J. Biophotonics 12, e201900004 (2019)",
        ],
        "canonical_datasets": [
            "Kvasir-SEG (polyp segmentation)",
            "CVC-ClinicDB (colonoscopy)",
            "HyperKvasir (multi-class GI dataset)",
        ],
        "tags": ["clinical", "endoscopy", "fiber", "gastrointestinal", "minimally_invasive"],
    },

    # ══════════════════════════════════════════════════════════════════════════
    # ELECTRON MICROSCOPY  (8 modalities)
    # ══════════════════════════════════════════════════════════════════════════

    "sem": {
        "display_name": "Scanning Electron Microscopy",
        "category": "electron_microscopy",
        "description": (
            "SEM forms images by rastering a focused electron beam (1-30 keV) across "
            "the specimen surface and collecting secondary electrons (SE, topographic "
            "contrast) or backscattered electrons (BSE, compositional Z-contrast). "
            "Resolution is determined by the probe diameter (1-10 nm), accelerating "
            "voltage, and interaction volume. Key artifacts include charging in "
            "non-conductive specimens, drift, and contamination."
        ),
        "experimental_setup": {
            "instrument": "JEOL JSM-7800F / Thermo Fisher Apreo 2 / Zeiss GeminiSEM 560",
            "accelerating_voltage_kV": 10,
            "beam_current_nA": 0.54,
            "working_distance_mm": 10,
            "pixel_size_nm": 7.1,
            "magnification": "20,000x",
            "detector": "Everhart-Thornley (SE2) + in-lens (SE1)",
            "image_size": "1024x768",
        },
        "physics_class": "electron_beam",
        "forward_model_family": "raster_scan_detection",
        "wave_model": "particle",
        "sensor_type": "electron_detector",
        "source_type": "field_emission_gun",
        "geometry": "raster_scan",
        "noise_model": "poisson",
        "typical_x_dims": [1024, 768],
        "typical_y_dims": [1024, 768],
        "typical_snr_range": [15.0, 40.0],
        "calibration_params": [
            "beam_current", "accelerating_voltage", "working_distance",
            "stigmation", "aperture_alignment",
        ],
        "mismatch_modes": [
            "charging", "drift", "contamination",
            "astigmatism", "vibration",
        ],
        "reconstruction_task_types": ["denoising", "super_resolution", "segmentation"],
        "default_solver": "direct_imaging",
        "evaluation_metrics": ["psnr", "ssim", "frc", "edge_sharpness"],
        "canonical_references": [
            "Goldstein et al., 'Scanning Electron Microscopy and X-ray Microanalysis', "
            "Springer (2018)",
        ],
        "canonical_datasets": [
            "SEM Dataset for Nanomaterial Segmentation (Aversa et al.)",
            "NIST SEM calibration images",
        ],
        "tags": ["electron", "scanning", "surface", "topographic", "nanoscale"],
    },

    "tem": {
        "display_name": "Transmission Electron Microscopy",
        "category": "electron_microscopy",
        "description": (
            "TEM transmits a high-energy electron beam (80-300 keV) through an "
            "ultra-thin specimen (<100 nm), magnifying the exit wave with EM lenses. "
            "In HRTEM, the image records interference between direct and diffracted "
            "beams, convolved by the contrast transfer function (CTF). The CTF "
            "introduces oscillating contrast reversals modulated by defocus and "
            "spherical aberration. Reconstruction involves CTF correction and, for "
            "biological specimens, single-particle averaging."
        ),
        "experimental_setup": {
            "instrument": "Thermo Fisher Titan Themis 300 / JEOL JEM-ARM300F2",
            "accelerating_voltage_kV": 300,
            "Cs_corrected": True,
            "information_limit_pm": 50,
            "detector": "Gatan K3 direct electron (5760x4092)",
            "pixel_size_pm": 50,
            "dose_e_per_A2": 30,
            "magnification": "1,000,000x",
        },
        "physics_class": "electron_beam",
        "forward_model_family": "ctf_imaging",
        "wave_model": "wave_optics",
        "sensor_type": "direct_electron_detector",
        "source_type": "field_emission_gun",
        "geometry": "transmission",
        "noise_model": "poisson",
        "typical_x_dims": [4096, 4096],
        "typical_y_dims": [4096, 4096],
        "typical_snr_range": [5.0, 30.0],
        "calibration_params": [
            "defocus", "spherical_aberration", "beam_tilt",
            "astigmatism", "pixel_calibration",
        ],
        "mismatch_modes": [
            "defocus_error", "residual_aberration", "specimen_drift",
            "beam_damage", "contamination",
        ],
        "reconstruction_task_types": ["ctf_correction", "exit_wave_reconstruction", "denoising"],
        "default_solver": "ctf_correction",
        "evaluation_metrics": ["psnr", "ssim", "frc", "information_limit"],
        "canonical_references": [
            "Williams & Carter, 'Transmission Electron Microscopy', Springer (2009)",
            "Haider et al., 'Electron microscopy image enhanced', Nature 392, 768 (1998)",
        ],
        "canonical_datasets": [
            "EMPIAR (Electron Microscopy Public Image Archive)",
            "NCEM atomic-resolution HRTEM benchmarks",
        ],
        "tags": ["electron", "transmission", "high_resolution", "atomic", "ctf"],
    },

    "stem": {
        "display_name": "Scanning Transmission Electron Microscopy",
        "category": "electron_microscopy",
        "description": (
            "STEM focuses the electron beam to a sub-angstrom probe and scans it "
            "across a thin specimen. The HAADF detector collects electrons scattered "
            "to large angles (>50 mrad), producing incoherent Z-contrast images where "
            "intensity scales as ~Z^1.7, enabling direct compositional interpretation "
            "at atomic resolution. Aberration correction (C3/C5 correctors) achieves "
            "sub-50 pm probe sizes. Primary degradations include scan distortion, "
            "probe instability, and radiation damage."
        ),
        "experimental_setup": {
            "instrument": "Nion UltraSTEM 200 / JEOL JEM-ARM200F / Thermo Fisher Titan Cubed",
            "accelerating_voltage_kV": 200,
            "convergence_semiangle_mrad": 21,
            "beam_current_pA": 10,
            "probe_size_pm": 70,
            "HAADF_inner_angle_mrad": 80,
            "HAADF_outer_angle_mrad": 200,
            "image_size": "512x512",
            "dwell_time_us": 20,
        },
        "physics_class": "electron_beam",
        "forward_model_family": "incoherent_z_contrast",
        "wave_model": "wave_optics",
        "sensor_type": "annular_detector",
        "source_type": "cold_field_emission",
        "geometry": "raster_scan",
        "noise_model": "poisson",
        "typical_x_dims": [512, 512],
        "typical_y_dims": [512, 512],
        "typical_snr_range": [8.0, 35.0],
        "calibration_params": [
            "convergence_angle", "detector_angles", "aberration_coefficients",
            "probe_current", "pixel_calibration",
        ],
        "mismatch_modes": [
            "scan_distortion", "probe_instability", "specimen_drift",
            "contamination", "beam_damage",
        ],
        "reconstruction_task_types": ["denoising", "atom_column_detection", "strain_mapping"],
        "default_solver": "direct_imaging",
        "evaluation_metrics": ["psnr", "ssim", "frc", "atom_position_precision"],
        "canonical_references": [
            "Pennycook & Nellist, 'Z-Contrast STEM Imaging', Springer (2011)",
            "Krivanek et al., 'Atom-by-atom structural and chemical analysis by "
            "annular dark-field electron microscopy', Nature 464, 571 (2010)",
        ],
        "canonical_datasets": [
            "NCEM Molecular Foundry STEM benchmarks",
            "EMPIAR STEM datasets",
        ],
        "tags": ["electron", "scanning", "transmission", "z_contrast", "atomic_resolution"],
    },

    "electron_tomography": {
        "display_name": "Electron Tomography",
        "category": "electron_microscopy",
        "description": (
            "Electron tomography reconstructs 3D structure from a tilt series of "
            "2D projections acquired as the specimen is rotated (+/-60-70 deg, "
            "1-2 deg increments). The missing wedge of angular coverage causes "
            "elongation artifacts along the beam direction. Alignment of the tilt "
            "series (using fiducial gold markers or cross-correlation) is critical. "
            "Reconstruction uses WBP, SIRT, or compressed sensing methods with TV "
            "priors to mitigate missing-wedge artifacts."
        ),
        "experimental_setup": {
            "instrument": "Thermo Fisher Titan Krios G4 / JEOL JEM-2200FS",
            "accelerating_voltage_kV": 200,
            "tilt_range_deg": [-70, 70],
            "tilt_increment_deg": 2.0,
            "number_of_projections": 71,
            "detector": "HAADF-STEM / Gatan K3",
            "pixel_size_nm": 0.71,
            "total_dose_e_per_nm2": 39000,
            "reconstruction": "SIRT / WBP",
        },
        "physics_class": "tomographic",
        "forward_model_family": "projection_tilt_series",
        "wave_model": "particle",
        "sensor_type": "direct_electron_detector",
        "source_type": "field_emission_gun",
        "geometry": "limited_angle_tomographic",
        "noise_model": "poisson",
        "typical_x_dims": [512, 512, 512],
        "typical_y_dims": [71, 512, 512],
        "typical_snr_range": [3.0, 20.0],
        "calibration_params": [
            "tilt_angles", "tilt_axis_orientation",
            "fiducial_positions", "magnification_per_tilt",
        ],
        "mismatch_modes": [
            "missing_wedge", "alignment_error", "specimen_shrinkage",
            "beam_damage", "focus_gradient",
        ],
        "reconstruction_task_types": ["3d_reconstruction", "missing_wedge_filling", "segmentation"],
        "default_solver": "sirt",
        "evaluation_metrics": ["psnr", "ssim", "fsc", "isotropy_index"],
        "canonical_references": [
            "Frank, 'Electron Tomography', Springer (2006)",
            "Midgley & Dunin-Borkowski, 'Electron tomography and holography in "
            "materials science', Nature Materials 8, 271 (2009)",
        ],
        "canonical_datasets": [
            "EMPIAR cryo-ET tilt series (e.g., EMPIAR-10045)",
            "ETDB (Electron Tomography Database, Caltech)",
        ],
        "tags": ["electron", "tomography", "3d", "tilt_series", "missing_wedge"],
    },

    "electron_diffraction": {
        "display_name": "4D-STEM Electron Diffraction",
        "category": "electron_microscopy",
        "description": (
            "4D-STEM acquires a full 2D convergent-beam electron diffraction (CBED) "
            "pattern at each probe position during a 2D STEM scan, yielding a "
            "4D dataset (2 real-space + 2 reciprocal-space dimensions). This enables "
            "simultaneous mapping of strain, orientation, electric fields, and "
            "thickness with nanometer spatial resolution. Phase retrieval from the "
            "4D dataset (electron ptychography) can achieve sub-angstrom resolution. "
            "High data rates (>1 GB/s) from fast pixelated detectors create "
            "computational challenges."
        ),
        "experimental_setup": {
            "instrument": "Thermo Fisher Titan with Medipix3 / JEOL ARM with EMPAD",
            "accelerating_voltage_kV": 200,
            "convergence_angle_mrad": 1.5,
            "step_size_nm": 1.0,
            "detector": "Medipix3 / Merlin (256x256 px)",
            "exposure_ms": 1,
            "camera_length_mm": 580,
            "reconstruction": "ptychographic phase retrieval / WDD",
        },
        "physics_class": "coherent_diffraction",
        "forward_model_family": "cbed_forward",
        "wave_model": "wave_optics",
        "sensor_type": "pixelated_detector",
        "source_type": "field_emission_gun",
        "geometry": "scanning_diffraction",
        "noise_model": "poisson",
        "typical_x_dims": [256, 256],
        "typical_y_dims": [128, 128, 256, 256],
        "typical_snr_range": [5.0, 25.0],
        "calibration_params": [
            "camera_length", "convergence_angle", "beam_center",
            "rotation_angle", "detector_gain",
        ],
        "mismatch_modes": [
            "scan_distortion", "detector_saturation",
            "dynamical_scattering", "specimen_tilt",
        ],
        "reconstruction_task_types": ["strain_mapping", "ptychographic_reconstruction", "orientation_mapping"],
        "default_solver": "ptychography_epie",
        "evaluation_metrics": ["strain_precision", "phase_error", "spatial_resolution"],
        "canonical_references": [
            "Ophus, 'Four-dimensional scanning transmission electron microscopy', "
            "Microscopy & Microanalysis 25, 563 (2019)",
            "Jiang et al., 'Electron ptychography of 2D materials to deep sub-angstrom "
            "resolution', Nature 559, 343 (2018)",
        ],
        "canonical_datasets": [
            "4D-STEM benchmark datasets (Ophus group, NCEM)",
        ],
        "tags": ["electron", "diffraction", "4d_stem", "strain", "ptychography"],
    },

    "ebsd": {
        "display_name": "Electron Backscatter Diffraction",
        "category": "electron_microscopy",
        "description": (
            "EBSD maps crystallographic orientation by tilting a polished specimen "
            "to ~70 degrees in an SEM and recording Kikuchi diffraction patterns "
            "on a phosphor screen. Each pattern encodes the local crystal orientation, "
            "which is determined by automated indexing (Hough transform or dictionary "
            "indexing). Scanning the beam produces orientation maps (IPF), grain "
            "boundary maps, and texture information. Challenges include pattern "
            "quality degradation from surface damage, pseudosymmetry in indexing, "
            "and angular resolution limitations (~0.5 deg)."
        ),
        "experimental_setup": {
            "instrument": "Oxford Instruments Symmetry S2 / EDAX Hikari Super",
            "accelerating_voltage_kV": 20,
            "sample_tilt_deg": 70,
            "step_size_um": 0.5,
            "camera_resolution": "622x512 (Symmetry S2)",
            "exposure_ms": 10,
            "indexing": "Hough transform / dictionary indexing",
            "output": "grain orientation map (IPF), misorientation",
        },
        "physics_class": "diffraction",
        "forward_model_family": "kikuchi_pattern_simulation",
        "wave_model": "wave_optics",
        "sensor_type": "phosphor_screen_ccd",
        "source_type": "field_emission_gun",
        "geometry": "backscatter",
        "noise_model": "poisson_gaussian",
        "typical_x_dims": [512, 512],
        "typical_y_dims": [512, 622, 512],
        "typical_snr_range": [10.0, 30.0],
        "calibration_params": [
            "pattern_center", "detector_distance",
            "sample_tilt", "crystal_structure_library",
        ],
        "mismatch_modes": [
            "surface_damage", "pseudosymmetry",
            "pattern_overlap", "drift",
        ],
        "reconstruction_task_types": ["orientation_indexing", "grain_reconstruction"],
        "default_solver": "hough_indexing",
        "evaluation_metrics": ["indexing_rate", "angular_resolution_deg", "ci_confidence"],
        "canonical_references": [
            "Schwartz et al., 'Electron Backscatter Diffraction in Materials Science', "
            "Springer (2009)",
        ],
        "canonical_datasets": ["DREAM.3D synthetic EBSD benchmarks"],
        "tags": ["electron", "crystallography", "orientation", "ebsd", "grain_mapping"],
    },

    "eels": {
        "display_name": "Electron Energy Loss Spectroscopy",
        "category": "electron_microscopy",
        "description": (
            "STEM-EELS measures the energy distribution of electrons transmitted "
            "through a thin specimen, where inelastic scattering events encode "
            "information about elemental composition, bonding, and electronic "
            "structure. The energy loss spectrum contains core-loss edges "
            "(characteristic of specific elements) and low-loss features (plasmons, "
            "band gaps). A magnetic prism spectrometer disperses the energy spectrum "
            "onto a position-sensitive detector. Spectrum imaging acquires a full "
            "spectrum at each scan position, enabling elemental mapping with "
            "atomic-scale spatial resolution."
        ),
        "experimental_setup": {
            "instrument": "Gatan Quantum GIF / Gatan Continuum / Nion HERMES",
            "accelerating_voltage_kV": 100,
            "energy_resolution_eV": 0.3,
            "dispersion_eV_per_ch": 0.1,
            "collection_angle_mrad": 30,
            "dwell_time_ms": 50,
            "spectrum_range": "core-loss + low-loss",
            "analysis": "elemental mapping, ELNES fine structure",
        },
        "physics_class": "spectroscopic",
        "forward_model_family": "energy_loss_cross_section",
        "wave_model": "wave_optics",
        "sensor_type": "scintillator_ccd",
        "source_type": "cold_field_emission",
        "geometry": "transmission_spectroscopic",
        "noise_model": "poisson",
        "typical_x_dims": [64, 64, 2048],
        "typical_y_dims": [64, 64, 2048],
        "typical_snr_range": [3.0, 20.0],
        "calibration_params": [
            "energy_dispersion", "zero_loss_alignment",
            "collection_angle", "convergence_angle",
        ],
        "mismatch_modes": [
            "plural_scattering", "channel_to_channel_gain",
            "drift_during_acquisition", "radiation_damage",
        ],
        "reconstruction_task_types": ["elemental_mapping", "fine_structure_analysis", "denoising"],
        "default_solver": "fourier_ratio",
        "evaluation_metrics": ["elemental_detection_limit", "energy_resolution_eV", "snr"],
        "canonical_references": [
            "Egerton, 'Electron Energy-Loss Spectroscopy in the Electron Microscope', "
            "Springer (2011)",
        ],
        "canonical_datasets": ["EELS Atlas (Ahn & Krivanek)"],
        "tags": ["electron", "spectroscopy", "energy_loss", "eels", "elemental_mapping"],
    },

    "electron_holography": {
        "display_name": "Electron Holography",
        "category": "electron_microscopy",
        "description": (
            "Off-axis electron holography records the interference pattern between "
            "an object wave (passed through the specimen) and a reference wave "
            "(passed through vacuum) using an electrostatic biprism. The hologram "
            "encodes the phase shift imparted by electric and magnetic fields within "
            "the specimen. Fourier filtering isolates the sideband carrying the "
            "complex wave information, from which amplitude and phase are extracted. "
            "Phase sensitivity of ~2*pi/1000 enables mapping of nanoscale electric "
            "and magnetic fields in materials."
        ),
        "experimental_setup": {
            "instrument": "Thermo Fisher Titan Holography / JEOL JEM-3000F",
            "accelerating_voltage_kV": 300,
            "wavelength_pm": 1.97,
            "detector": "Gatan Orius CCD (2k x 2k)",
            "biprism_voltage_V": 150,
            "exposure_s": 2,
            "fringe_spacing_nm": 0.3,
            "reconstruction": "Fourier filtering + phase unwrapping",
            "application": "magnetic / electric field mapping",
        },
        "physics_class": "interferometric",
        "forward_model_family": "electron_interference",
        "wave_model": "wave_optics",
        "sensor_type": "ccd",
        "source_type": "field_emission_gun",
        "geometry": "off_axis_interference",
        "noise_model": "poisson",
        "typical_x_dims": [2048, 2048],
        "typical_y_dims": [2048, 2048],
        "typical_snr_range": [10.0, 30.0],
        "calibration_params": [
            "biprism_voltage", "fringe_spacing",
            "reference_hologram", "magnification",
        ],
        "mismatch_modes": [
            "fringe_distortion", "biprism_charging",
            "specimen_drift", "inelastic_scattering",
        ],
        "reconstruction_task_types": ["phase_reconstruction", "field_mapping"],
        "default_solver": "fourier_sideband",
        "evaluation_metrics": ["phase_sensitivity_rad", "spatial_resolution_nm"],
        "canonical_references": [
            "Dunin-Borkowski et al., 'Electron holography of nanostructured materials', "
            "Encyclopedia of Nanoscience and Nanotechnology (2004)",
            "Lichte & Lehmann, 'Electron holography — basics and applications', "
            "Rep. Prog. Phys. 71, 016102 (2008)",
        ],
        "canonical_datasets": [
            "Holography benchmark datasets (Forschungszentrum Julich)",
        ],
        "tags": ["electron", "holography", "phase", "magnetic_field", "electric_field"],
    },

    # ══════════════════════════════════════════════════════════════════════════
    # DEPTH / TOF IMAGING  (3 modalities)
    # ══════════════════════════════════════════════════════════════════════════

    "tof_camera": {
        "display_name": "Time-of-Flight Depth Camera",
        "category": "depth_imaging",
        "description": (
            "ToF cameras measure per-pixel depth by emitting modulated near-infrared "
            "light and measuring the phase delay of the reflected signal relative to "
            "the emitted signal. In amplitude-modulated continuous-wave (AMCW) ToF, "
            "the phase offset phi = 2*pi*f*2d/c encodes the round-trip distance 2d. "
            "Multiple modulation frequencies resolve depth ambiguity. Primary "
            "degradations include multi-path interference (MPI), motion blur, and "
            "systematic errors at depth discontinuities (flying pixels)."
        ),
        "experimental_setup": {
            "instrument": "Intel RealSense L515 / Microsoft Azure Kinect DK",
            "depth_resolution": "640x480",
            "range_m": "0.1-6.0",
            "frame_rate_fps": 30,
            "wavelength_nm": 850,
            "depth_accuracy_mm": 2.0,
            "modulation": "AMCW (amplitude-modulated continuous wave)",
        },
        "physics_class": "time_of_flight",
        "forward_model_family": "phase_delay_depth",
        "wave_model": "ray",
        "sensor_type": "tof_sensor",
        "source_type": "nir_led_vcsel",
        "geometry": "planar",
        "noise_model": "gaussian",
        "typical_x_dims": [640, 480],
        "typical_y_dims": [640, 480],
        "typical_snr_range": [15.0, 35.0],
        "calibration_params": [
            "modulation_frequency", "phase_offset",
            "lens_distortion", "depth_nonlinearity",
        ],
        "mismatch_modes": [
            "multipath_interference", "flying_pixels",
            "motion_blur", "ambient_light_saturation",
        ],
        "reconstruction_task_types": ["depth_denoising", "multipath_correction"],
        "default_solver": "tv_fista",
        "evaluation_metrics": ["depth_mae_mm", "depth_rmse", "bad_pixel_percent"],
        "canonical_references": [
            "Hansard et al., 'Time-of-Flight Cameras: Principles, Methods and "
            "Applications', Springer (2013)",
        ],
        "canonical_datasets": [
            "NYU Depth V2 (Silberman et al.)",
            "KITTI depth benchmark (adapted)",
        ],
        "tags": ["depth", "tof", "3d", "nir", "range_imaging"],
    },

    "structured_light": {
        "display_name": "Structured-Light Depth Camera",
        "category": "depth_imaging",
        "description": (
            "Structured-light depth cameras project a known pattern (IR dot pattern, "
            "fringe, or binary code) onto the scene and infer depth from the pattern "
            "deformation observed by a camera offset from the projector. For coded "
            "structured light (e.g., Kinect v1), depth is computed via triangulation "
            "from the correspondence between projected and observed pattern features. "
            "For phase-shifting methods, multiple fringe patterns encode depth as "
            "the local phase. Primary challenges include occlusion in the projector-camera "
            "baseline, ambient light interference, and depth discontinuity errors."
        ),
        "experimental_setup": {
            "instrument": "Intel RealSense D435i / Apple TrueDepth / Kinect v1",
            "pattern": "pseudorandom IR dot pattern / fringe projection",
            "wavelength_nm": 850,
            "range_m": "0.2-10.0",
            "depth_resolution": "1280x720",
            "accuracy_mm": 1.0,
            "frame_rate_fps": 30,
            "baseline_mm": 55,
        },
        "physics_class": "structured_light",
        "forward_model_family": "triangulation",
        "wave_model": "ray",
        "sensor_type": "cmos_ir",
        "source_type": "ir_projector",
        "geometry": "stereo_baseline",
        "noise_model": "gaussian",
        "typical_x_dims": [1280, 720],
        "typical_y_dims": [1280, 720],
        "typical_snr_range": [15.0, 35.0],
        "calibration_params": [
            "projector_camera_extrinsics", "intrinsics",
            "pattern_calibration", "lens_distortion",
        ],
        "mismatch_modes": [
            "occlusion", "ambient_light", "specular_reflection",
            "pattern_interference", "depth_shadow",
        ],
        "reconstruction_task_types": ["depth_estimation", "hole_filling", "denoising"],
        "default_solver": "phase_unwrap",
        "evaluation_metrics": ["depth_mae_mm", "completeness_percent", "bad_pixel_percent"],
        "canonical_references": [
            "Geng, 'Structured-light 3D surface imaging: a tutorial', "
            "Advances in Optics and Photonics 3, 128-160 (2011)",
        ],
        "canonical_datasets": [
            "Middlebury stereo benchmark",
            "ETH3D multi-view stereo benchmark",
        ],
        "tags": ["depth", "structured_light", "3d", "triangulation", "ir_projection"],
    },

    "lidar": {
        "display_name": "LiDAR Scanner",
        "category": "depth_imaging",
        "description": (
            "LiDAR (Light Detection and Ranging) measures distances by emitting "
            "laser pulses and timing the round-trip to the reflecting surface. "
            "Automotive LiDAR systems use rotating multi-beam scanners (e.g., "
            "Velodyne HDL-64E) or solid-state flash LiDAR to acquire 3D point "
            "clouds at 10-20 Hz. The forward model is simple time-of-flight: "
            "d = c*t/2. The resulting sparse point cloud requires densification, "
            "ground segmentation, and object detection. Primary challenges include "
            "sparse sampling, intensity variation with surface reflectivity, and "
            "rain/fog attenuation."
        ),
        "experimental_setup": {
            "instrument": "Velodyne HDL-64E / Ouster OS1-128 / Livox Avia",
            "channels": 64,
            "range_m": 120,
            "horizontal_FOV_deg": 360,
            "vertical_FOV_deg": 27,
            "horizontal_resolution_deg": 0.08,
            "rotation_rate_Hz": 10,
            "wavelength_nm": 905,
            "points_per_second": 2200000,
            "dataset": "KITTI, nuScenes, Waymo Open",
        },
        "physics_class": "time_of_flight",
        "forward_model_family": "pulse_tof",
        "wave_model": "ray",
        "sensor_type": "spad_or_apd",
        "source_type": "pulsed_laser",
        "geometry": "rotating_scan",
        "noise_model": "gaussian",
        "typical_x_dims": [64, 2048],
        "typical_y_dims": [64, 2048],
        "typical_snr_range": [15.0, 40.0],
        "calibration_params": [
            "extrinsic_to_camera", "beam_angles",
            "range_calibration", "intensity_calibration",
        ],
        "mismatch_modes": [
            "rain_fog_attenuation", "crosstalk",
            "motion_distortion", "low_reflectivity_dropout",
        ],
        "reconstruction_task_types": ["point_cloud_densification", "object_detection", "segmentation"],
        "default_solver": "tv_fista",
        "evaluation_metrics": ["depth_mae_m", "chamfer_distance", "iou_3d"],
        "canonical_references": [
            "Geiger et al., 'Are we ready for autonomous driving? The KITTI vision "
            "benchmark suite', CVPR 2012",
        ],
        "canonical_datasets": [
            "KITTI 3D object detection",
            "nuScenes (1000 driving scenes)",
            "Waymo Open Dataset",
        ],
        "tags": ["depth", "lidar", "point_cloud", "autonomous_driving", "3d"],
    },

    # ══════════════════════════════════════════════════════════════════════════
    # REMOTE SENSING  (2 modalities)
    # ══════════════════════════════════════════════════════════════════════════

    "sar": {
        "display_name": "Synthetic Aperture Radar",
        "category": "remote_sensing",
        "description": (
            "SAR synthesizes a large antenna aperture by combining coherent radar "
            "returns collected as the platform (satellite/aircraft) moves along its "
            "flight path. The azimuth resolution is achieved by coherent integration "
            "of the Doppler history, while range resolution comes from pulse "
            "compression (chirp). The forward model is a 2D convolution with the "
            "SAR impulse response in range and azimuth. SAR images exhibit speckle "
            "noise (multiplicative, fully developed) from coherent interference of "
            "distributed scatterers. Applications include Earth observation, terrain "
            "mapping, and interferometric displacement measurement."
        ),
        "experimental_setup": {
            "instrument": "Sentinel-1 (ESA Copernicus) / TerraSAR-X",
            "frequency_band": "C-band (5.405 GHz)",
            "wavelength_cm": 5.6,
            "mode": "IW (Interferometric Wide Swath)",
            "spatial_resolution_m": "5 (range) x 20 (azimuth)",
            "swath_km": 250,
            "polarization": "VV + VH (dual-pol)",
            "incidence_angle_deg": "29.1-46.0",
            "revisit_days": 6,
        },
        "physics_class": "radar",
        "forward_model_family": "sar_focusing",
        "wave_model": "coherent_em",
        "sensor_type": "radar_receiver",
        "source_type": "radar_transmitter",
        "geometry": "side_looking",
        "noise_model": "speckle",
        "typical_x_dims": [1024, 1024],
        "typical_y_dims": [1024, 1024],
        "typical_snr_range": [10.0, 30.0],
        "calibration_params": [
            "orbit_state_vectors", "antenna_pattern",
            "radiometric_calibration", "terrain_correction",
        ],
        "mismatch_modes": [
            "speckle", "layover", "foreshortening",
            "shadow", "atmospheric_delay",
        ],
        "reconstruction_task_types": ["sar_focusing", "speckle_filtering", "insar"],
        "default_solver": "backprojection",
        "evaluation_metrics": ["enl", "psnr", "ssim", "spatial_resolution_m"],
        "canonical_references": [
            "Cumming & Wong, 'Digital Processing of Synthetic Aperture Radar Data', "
            "Artech House (2005)",
            "Torres et al., 'GMES Sentinel-1 mission', Remote Sensing of Environment "
            "120, 9-24 (2012)",
        ],
        "canonical_datasets": [
            "SEN12MS (Schmitt et al., multi-modal Sentinel-1/2)",
            "SpaceNet 6 (SAR building footprints)",
        ],
        "tags": ["remote_sensing", "radar", "sar", "microwave", "earth_observation"],
    },

    "sonar": {
        "display_name": "Sonar Imaging",
        "category": "remote_sensing",
        "description": (
            "Side-scan sonar maps the seabed by transmitting acoustic pulses "
            "perpendicular to the survey vessel's track and recording the "
            "backscattered energy as a function of time (range). The along-track "
            "resolution is determined by the beam width, while the across-track "
            "resolution comes from the pulse length. The sonar image is a 2D "
            "acoustic backscatter map where intensity encodes seabed roughness, "
            "composition, and the presence of objects. Acoustic shadows behind "
            "elevated objects provide height information. Challenges include "
            "multipath reflections, variable sound speed profile, and non-uniform "
            "ensonification."
        ),
        "experimental_setup": {
            "instrument": "EdgeTech 4125 / Klein 3000 / Kongsberg EM 2040",
            "frequency_kHz": 900,
            "range_m": 100,
            "resolution_m": 0.1,
            "swath_m": 200,
            "platform": "AUV / towed body",
            "application": "seabed mapping / mine detection",
        },
        "physics_class": "acoustic",
        "forward_model_family": "acoustic_backscatter",
        "wave_model": "acoustic",
        "sensor_type": "hydrophone_array",
        "source_type": "acoustic_transducer",
        "geometry": "side_looking",
        "noise_model": "speckle",
        "typical_x_dims": [1024, 512],
        "typical_y_dims": [1024, 512],
        "typical_snr_range": [10.0, 30.0],
        "calibration_params": [
            "sound_speed_profile", "tow_body_attitude",
            "beam_pattern", "tvg_correction",
        ],
        "mismatch_modes": [
            "multipath", "sound_speed_variation",
            "tow_body_instability", "bottom_type_ambiguity",
        ],
        "reconstruction_task_types": ["image_formation", "object_detection", "classification"],
        "default_solver": "beamform_das",
        "evaluation_metrics": ["detection_pd", "false_alarm_rate", "resolution_m"],
        "canonical_references": [
            "Blondel, 'The Handbook of Sidescan Sonar', Springer (2009)",
        ],
        "canonical_datasets": [
            "UATD underwater acoustic target detection dataset",
            "S3Simulator synthetic sonar (2024)",
        ],
        "tags": ["remote_sensing", "sonar", "underwater", "acoustic", "seabed"],
    },

    # ══════════════════════════════════════════════════════════════════════════
    # PARTICLE IMAGING  (3 modalities)
    # ══════════════════════════════════════════════════════════════════════════

    "neutron_tomo": {
        "display_name": "Neutron Radiography / Tomography",
        "category": "particle_imaging",
        "description": (
            "Neutron imaging exploits the unique interaction of thermal neutrons "
            "with matter — neutrons are attenuated strongly by light elements "
            "(hydrogen, lithium, boron) while penetrating heavy elements (lead, iron) "
            "that are opaque to X-rays. The forward model follows Beer-Lambert: "
            "I = I_0 * exp(-integral(Sigma(s) ds)) where Sigma is the macroscopic "
            "cross-section. Tomographic reconstruction from multiple projection "
            "angles uses FBP or iterative methods. Neutron sources include research "
            "reactors and spallation sources. The lower flux compared to X-rays "
            "requires longer exposures (seconds) and results in lower spatial "
            "resolution (50-100 um)."
        ),
        "experimental_setup": {
            "instrument": "PSI ICON beamline / NIST BT-2 / ORNL CG-1D",
            "neutron_energy_eV": 0.025,
            "energy_type": "thermal",
            "detector": "LiF/ZnS scintillator + CCD",
            "pixel_size_um": 100,
            "image_size": "2048x2048",
            "exposure_s": 10,
            "flux_n_per_cm2_s": 1e8,
            "facility": "research reactor / spallation source",
        },
        "physics_class": "particle_transmission",
        "forward_model_family": "beer_lambert_neutron",
        "wave_model": "particle",
        "sensor_type": "scintillator_ccd",
        "source_type": "neutron_source",
        "geometry": "rotational",
        "noise_model": "poisson",
        "typical_x_dims": [512, 512],
        "typical_y_dims": [180, 2048],
        "typical_snr_range": [10.0, 30.0],
        "calibration_params": [
            "open_beam_normalization", "dark_current",
            "center_of_rotation", "scattering_correction",
        ],
        "mismatch_modes": [
            "neutron_scatter", "beam_hardening",
            "sample_activation", "gamma_background",
        ],
        "reconstruction_task_types": ["tomographic_reconstruction", "radiography"],
        "default_solver": "filtered_back_projection",
        "evaluation_metrics": ["psnr", "ssim", "cnr", "spatial_resolution_um"],
        "canonical_references": [
            "Kardjilov et al., 'Advances in neutron imaging', "
            "Materials Today 21, 652-672 (2018)",
            "IAEA, 'Neutron Imaging: A Non-Destructive Tool for Materials Testing', "
            "IAEA-TECDOC-1604 (2008)",
        ],
        "canonical_datasets": [
            "PSI ICON neutron imaging benchmark data",
            "NIST neutron radiography reference images",
        ],
        "tags": ["particle", "neutron", "tomography", "hydrogen_sensitive", "ndt"],
    },

    "proton_radiography": {
        "display_name": "Proton Radiography",
        "category": "particle_imaging",
        "description": (
            "Proton radiography/CT uses high-energy proton beams (100-250 MeV) to "
            "image the relative stopping power (RSP) of tissue, which is the quantity "
            "directly needed for proton therapy treatment planning. Unlike X-rays "
            "which measure attenuation, proton imaging measures the energy loss and "
            "scattering of individual protons as they traverse the object. Each proton's "
            "entry/exit position and angle are tracked, and the residual energy is "
            "measured. The RSP is reconstructed from many proton histories using "
            "iterative algorithms. Challenges include multiple Coulomb scattering "
            "(which blurs the spatial resolution to ~1 mm) and the need for "
            "single-proton tracking at high rates."
        ),
        "experimental_setup": {
            "instrument": "Phase-II proton CT prototype (Loma Linda / NIU)",
            "proton_energy_MeV": 200,
            "detector": "scintillating fiber tracker + residual energy calorimeter",
            "tracker_planes": 4,
            "image_matrix": "256x256",
            "projections": 360,
            "RSP_accuracy_percent": 1.0,
            "application": "proton therapy treatment planning verification",
        },
        "physics_class": "particle_transmission",
        "forward_model_family": "energy_loss_scattering",
        "wave_model": "particle",
        "sensor_type": "particle_tracker",
        "source_type": "proton_accelerator",
        "geometry": "rotational",
        "noise_model": "gaussian",
        "typical_x_dims": [256, 256],
        "typical_y_dims": [360, 256, 256],
        "typical_snr_range": [10.0, 25.0],
        "calibration_params": [
            "beam_energy", "tracker_alignment",
            "energy_detector_calibration", "water_equivalent_calibration",
        ],
        "mismatch_modes": [
            "multiple_coulomb_scattering", "nuclear_interactions",
            "tracker_resolution", "energy_straggling",
        ],
        "reconstruction_task_types": ["rsp_reconstruction", "treatment_planning"],
        "default_solver": "filtered_back_projection",
        "evaluation_metrics": ["rsp_accuracy_percent", "spatial_resolution_mm"],
        "canonical_references": [
            "Schulte et al., 'Conceptual design of a proton computed tomography "
            "system for applications in proton radiation therapy', "
            "IEEE Trans. Nucl. Sci. 51, 866-872 (2004)",
        ],
        "canonical_datasets": ["Simulated proton CT phantoms (Penfold et al.)"],
        "tags": ["particle", "proton", "radiography", "therapy_planning", "medical"],
    },

    "muon_tomo": {
        "display_name": "Muon Tomography",
        "category": "particle_imaging",
        "description": (
            "Muon tomography uses naturally occurring cosmic-ray muons (mean energy "
            "~4 GeV, flux ~1/cm2/min at sea level) to image the interior of large, "
            "dense objects by measuring the scattering angle of each muon as it "
            "traverses the object. High-Z materials (uranium, plutonium, lead) cause "
            "large-angle scattering that is readily distinguished from low-Z materials. "
            "Position-sensitive detectors (drift tubes, RPCs) above and below the "
            "object track each muon's trajectory. The scattering density is "
            "proportional to Z^2/A. Reconstruction uses the point-of-closest-approach "
            "(POCA) algorithm or maximum-likelihood/expectation-maximization (ML-EM). "
            "Long exposure times (minutes to hours) are needed due to the low natural "
            "muon flux. Applications include nuclear material detection and volcano "
            "interior imaging (muography)."
        ),
        "experimental_setup": {
            "instrument": "Los Alamos muon radiography / Decision Sciences MMS",
            "mean_energy_GeV": 4.0,
            "flux_per_cm2_per_min": 1.0,
            "detector": "drift tube / RPC panels (2 tracking planes above + 2 below)",
            "position_resolution_mm": 1.0,
            "angular_resolution_mrad": 3.0,
            "exposure_min": 60,
            "technique": "multiple scattering tomography (MST)",
            "application": "nuclear material detection / volcano imaging",
        },
        "physics_class": "particle_scattering",
        "forward_model_family": "coulomb_scattering",
        "wave_model": "particle",
        "sensor_type": "gas_detector",
        "source_type": "cosmic_ray",
        "geometry": "transmission_scattering",
        "noise_model": "gaussian",
        "typical_x_dims": [64, 64, 64],
        "typical_y_dims": [100000, 6],
        "typical_snr_range": [3.0, 15.0],
        "calibration_params": [
            "detector_alignment", "detector_efficiency",
            "momentum_estimation", "acceptance_correction",
        ],
        "mismatch_modes": [
            "low_statistics", "momentum_uncertainty",
            "multiple_scattering_model", "detector_misalignment",
        ],
        "reconstruction_task_types": ["scattering_tomography", "material_discrimination"],
        "default_solver": "poca_reconstruction",
        "evaluation_metrics": ["detection_sensitivity", "spatial_resolution_cm", "z_discrimination"],
        "canonical_references": [
            "Borozdin et al., 'Radiographic imaging with cosmic-ray muons', "
            "Nature 422, 277 (2003)",
            "Tanaka et al., 'Imaging the conduit size of the dome with cosmic-ray "
            "muons: The structure beneath Showa-Shinzan Lava Dome', "
            "Geophysical Research Letters 34, L22311 (2007)",
        ],
        "canonical_datasets": [
            "Los Alamos muon tomography simulation benchmarks",
            "IAEA muon imaging reference data",
        ],
        "tags": ["particle", "muon", "tomography", "cosmic_ray", "nuclear_security", "muography"],
    },
}


# ── Modality Introductions ─────────────────────────────────────────────────
# Comprehensive introduction for each modality: physical principle, how to
# build the experimental system, common reconstruction algorithms, common
# mistakes, and how to avoid them.  Merged into MODALITY_DATABASE at the
# bottom of this section.

_MODALITY_INTRODUCTIONS: dict[str, dict] = {

    # ── MICROSCOPY ─────────────────────────────────────────────────────────

    "widefield": {
        "principle": (
            "The entire specimen is illuminated uniformly and fluorescence "
            "from all planes is collected simultaneously. The image is the "
            "convolution of the 3-D fluorescence distribution with the "
            "microscope point-spread function (PSF), dominated by out-of-focus "
            "blur from planes above and below the focal plane."
        ),
        "setup_guide": (
            "Mount an infinity-corrected high-NA objective (≥1.3 NA oil) on an "
            "inverted body (Nikon Ti2 or Zeiss Observer). Install a multi-band "
            "LED engine (e.g., Lumencor SPECTRA X) coupled through a liquid "
            "light guide. Select matched excitation/dichroic/emission filter "
            "sets. Focus Köhler illumination for flat-field. Attach an sCMOS "
            "camera (Hamamatsu Flash4 or Photometrics Prime BSI) at the side "
            "port. Calibrate pixel size with a stage micrometer."
        ),
        "common_algorithms": [
            "Richardson-Lucy deconvolution",
            "Wiener filtering",
            "CARE (Content-Aware image REstoration) deep-learning deconvolution",
            "Total-variation regularized deconvolution",
            "Blind deconvolution (PSF estimation + image update)",
        ],
        "common_mistakes": [
            "Using an incorrect or measured PSF with wrong refractive-index setting",
            "Ignoring flatfield non-uniformity, leading to intensity shading",
            "Over-iterating Richardson-Lucy causing noise amplification",
            "Mismatched immersion medium vs. coverslip thickness causing spherical aberration",
            "Not correcting for photobleaching across a time-lapse series",
        ],
        "how_to_avoid_mistakes": [
            "Measure the PSF with sub-diffraction beads at the same coverslip/medium as the sample",
            "Acquire and apply a flatfield correction image before deconvolution",
            "Use regularization or early stopping (monitor residual) in iterative deconvolution",
            "Match immersion oil RI to the coverslip and mounting medium specifications",
            "Normalize intensity per frame or use photobleaching-corrected models",
        ],
    },

    "widefield_lowdose": {
        "principle": (
            "Identical optical path to standard widefield but operated at very "
            "low photon budgets (short exposure or attenuated excitation) to "
            "minimize phototoxicity in live cells. The acquired images are "
            "severely photon-starved, making Poisson noise the dominant "
            "degradation rather than out-of-focus blur."
        ),
        "setup_guide": (
            "Use the same widefield microscope but reduce LED power to 1-5 % "
            "and/or shorten exposure to 5-20 ms. A high-QE back-illuminated "
            "sCMOS sensor (>80 % QE) is essential for capturing the limited "
            "photon signal. Install an environmental chamber for live-cell "
            "stability (37 °C, 5 % CO₂). Validate that the camera read noise "
            "floor is well below the expected signal."
        ),
        "common_algorithms": [
            "CARE (Content-Aware image REstoration)",
            "Noise2Void / Noise2Self (self-supervised denoising)",
            "BM3D / VST + BM3D for Poisson-Gaussian denoising",
            "PURE-LET (Poisson Unbiased Risk Estimator)",
            "Noise2Noise paired denoising networks",
        ],
        "common_mistakes": [
            "Setting read-noise-dominated regime by using too-low gain or old CCD",
            "Training denoising networks on data with different noise statistics than test data",
            "Clipping near-zero intensities by incorrect camera offset subtraction",
            "Ignoring sCMOS pixel-dependent noise (fixed-pattern noise)",
            "Exceeding live-cell phototoxicity budget despite intending low-dose imaging",
        ],
        "how_to_avoid_mistakes": [
            "Characterize camera noise model (gain, offset, variance map) before acquisition",
            "Train and evaluate denoising models at the same SNR and microscope settings",
            "Keep camera offset (dark current) calibration current and subtract properly",
            "Apply per-pixel gain and offset maps for sCMOS cameras",
            "Monitor cell health markers (morphology, division rate) to confirm non-toxic dose",
        ],
    },

    "confocal_livecell": {
        "principle": (
            "A focused laser spot is scanned across the specimen and a pinhole "
            "in front of the detector rejects out-of-focus fluorescence, "
            "providing optical sectioning. The image formation is modeled as a "
            "point-by-point convolution with the confocal PSF (product of "
            "excitation and detection PSFs). For live-cell work, speed and "
            "gentleness are prioritized."
        ),
        "setup_guide": (
            "Equip a laser-scanning confocal head (e.g., Nikon A1R, Zeiss LSM "
            "980 Airyscan) on an inverted microscope with an environmental "
            "enclosure. Use a resonant scanner for fast (30 fps) imaging. "
            "Set pinhole to 1 Airy unit for best sectioning or open slightly "
            "(1.2 AU) for more signal. Use 40-60x water-immersion objectives "
            "for live cells to match RI of aqueous media."
        ),
        "common_algorithms": [
            "Airyscan joint deconvolution (Zeiss)",
            "Richardson-Lucy with measured confocal PSF",
            "Sparse deconvolution (Hessian regularization)",
            "Deep-learning denoising (Noise2Fast, DnCNN)",
            "Pixel reassignment (ISM) for resolution doubling",
        ],
        "common_mistakes": [
            "Setting pinhole too small, drastically reducing signal in live cells",
            "Scanning too slowly, causing phototoxicity and photobleaching",
            "Using oil-immersion objectives for aqueous samples, introducing spherical aberration",
            "Ignoring chromatic aberration when imaging multiple channels simultaneously",
            "Oversampling (too many pixels) leading to excessive total dose with no resolution gain",
        ],
        "how_to_avoid_mistakes": [
            "Match pinhole to 1 AU and use resonant scanning + frame averaging for speed",
            "Minimize pixel dwell time and total exposure; use sensitive GaAsP detectors",
            "Select water-immersion objectives for live aqueous samples",
            "Calibrate chromatic offsets with multi-color beads and apply corrections",
            "Follow Nyquist sampling (pixel size ~ 0.4× resolution limit); avoid oversampling",
        ],
    },

    "confocal_3d": {
        "principle": (
            "Same confocal principle as live-cell mode but acquiring a full "
            "z-stack by stepping the objective or sample through the focal "
            "plane. Each optical section is convolved with the 3-D confocal "
            "PSF, and the full volume is reconstructed by 3-D deconvolution "
            "to recover isotropic resolution."
        ),
        "setup_guide": (
            "Use a high-NA objective (60-100x, 1.4 NA oil or 1.2 NA water) "
            "with a piezo z-stage for precise, repeatable z-steps (typ. 200-"
            "300 nm). Acquire z-stacks covering the specimen thickness with "
            "Nyquist z-sampling. For fixed samples, oil immersion is preferred; "
            "for thick tissue, use silicone oil or glycerol objectives to "
            "minimize RI mismatch deep in the sample."
        ),
        "common_algorithms": [
            "3-D Richardson-Lucy deconvolution",
            "3-D Wiener / Tikhonov deconvolution",
            "Huygens Professional iterative deconvolution",
            "DeconvolutionLab2 (GPU-accelerated 3-D)",
            "Deep-learning volumetric restoration (3-D U-Net, RCAN3D)",
        ],
        "common_mistakes": [
            "Using z-step larger than Nyquist, causing axial aliasing",
            "Depth-dependent spherical aberration from RI mismatch not corrected",
            "Not accounting for signal attenuation deeper in the sample",
            "Applying 2-D deconvolution slice-by-slice instead of full 3-D",
            "Incorrect PSF model (2-D Gaussian instead of 3-D Born & Wolf model)",
        ],
        "how_to_avoid_mistakes": [
            "Calculate Nyquist z-step (λ / (4·n·(1-cos α))) and sample accordingly",
            "Use depth-dependent PSF models or adaptive optics for thick specimens",
            "Apply intensity normalization per z-slice before deconvolution",
            "Always perform true 3-D deconvolution to preserve axial information",
            "Use measured 3-D PSF from sub-diffraction beads embedded at the correct depth",
        ],
    },

    "sim": {
        "principle": (
            "Structured Illumination Microscopy projects a known sinusoidal "
            "pattern onto the specimen, shifting high-frequency spatial "
            "information into the observable passband via Moiré interference. "
            "Multiple images (typically 9-15) are acquired at different pattern "
            "orientations and phases, then computationally recombined in "
            "Fourier space to achieve ~2× lateral resolution improvement beyond "
            "the diffraction limit."
        ),
        "setup_guide": (
            "Install a SIM-capable microscope (Nikon N-SIM, Zeiss Elyra 7, "
            "or custom with SLM/DMD). Use a high-NA objective (100x 1.49 NA "
            "TIRF) for maximum frequency extension. The illumination grating "
            "(SLM or fiber interference) generates the sinusoidal pattern. "
            "Acquire 3 orientations × 3-5 phases. A fast sCMOS camera captures "
            "all raw frames in ~100-500 ms for 2D-SIM. Careful alignment of "
            "the pattern contrast is critical."
        ),
        "common_algorithms": [
            "Gustafsson/Heintzmann frequency-domain SIM reconstruction",
            "Open-source fairSIM (ImageJ plugin)",
            "Wiener-filtered order separation and recombination",
            "Deep-learning SIM (ML-SIM, reconstruction from fewer frames)",
            "Hessian-SIM for live-cell with reduced artifacts",
        ],
        "common_mistakes": [
            "Insufficient pattern contrast causing weak Moiré fringes and honeycomb artifacts",
            "Misaligned illumination orders producing stripe artifacts in the reconstruction",
            "Over-processing (too aggressive Wiener parameter) creating ringing artifacts",
            "Using objectives with insufficient NA for the desired resolution gain",
            "Photobleaching between pattern acquisitions causing intensity inconsistency",
        ],
        "how_to_avoid_mistakes": [
            "Verify pattern contrast >0.5 on a thin uniform fluorescent layer before experiments",
            "Calibrate illumination pattern positions/angles using SIMcheck (ImageJ plugin)",
            "Tune the Wiener parameter conservatively; use SIMcheck to assess reconstruction quality",
            "Use 1.49 NA objectives for maximum resolution; 1.40 NA limits SIM performance",
            "Minimize total acquisition time; use fast cameras and short exposures",
        ],
    },

    "lightsheet": {
        "principle": (
            "A thin sheet of laser light illuminates only the focal plane of "
            "the detection objective, providing intrinsic optical sectioning "
            "with minimal out-of-plane photobleaching. The orthogonal geometry "
            "between illumination and detection decouples sectioning from "
            "resolution. Detection is widefield, enabling fast volumetric "
            "imaging of large specimens."
        ),
        "setup_guide": (
            "Arrange two orthogonal objective arms: one for the excitation "
            "sheet (cylindrical lens or digitally scanned Gaussian/Bessel beam) "
            "and one for detection (high-NA water-dipping). Mount the sample "
            "in agarose or hold in a chamber compatible with the dual-objective "
            "geometry. Use a fast sCMOS camera for detection. Stage scanning or "
            "sheet scanning acquires z-stacks. Consider diSPIM (dual-view) for "
            "isotropic resolution."
        ),
        "common_algorithms": [
            "Multi-view fusion (weighted averaging of complementary views)",
            "Multi-view deconvolution (Bayesian, joint Richardson-Lucy)",
            "Content-based image fusion",
            "Deep-learning denoising for high-speed acquisitions (CARE)",
            "Stripe artifact removal (wavelet-FFT filtering)",
        ],
        "common_mistakes": [
            "Light sheet too thick, degrading axial resolution and sectioning",
            "Absorption and scattering in thick tissue causing shadow artifacts (stripes)",
            "Misalignment between sheet focal plane and detection focal plane",
            "Improper sample mounting causing drift or deformation during long acquisitions",
            "Ignoring refractive-index variations causing sheet deflection inside tissue",
        ],
        "how_to_avoid_mistakes": [
            "Use Bessel or lattice light sheet for thin, uniform illumination profiles",
            "Pivot the light sheet or use dual-side illumination to reduce shadow artifacts",
            "Carefully co-align illumination and detection planes using fluorescent beads",
            "Use stable, low-melting-point agarose embedding and vibration-isolated stages",
            "Clear or match refractive index of tissue where possible; use adaptive optics",
        ],
    },

    "flim": {
        "principle": (
            "Fluorescence Lifetime Imaging measures the exponential decay time "
            "of fluorophore emission (typically 1-10 ns) rather than intensity. "
            "Lifetime is sensitive to the fluorophore's local chemical "
            "environment (pH, ion concentration, FRET) but independent of "
            "concentration and photobleaching. Detection uses either "
            "time-correlated single-photon counting (TCSPC) or frequency-domain "
            "phase/modulation methods."
        ),
        "setup_guide": (
            "Add a pulsed laser source (ps diode laser or Ti:Sapphire, 40-80 "
            "MHz repetition rate) to a confocal or widefield microscope. For "
            "TCSPC, install single-photon counting detectors (hybrid PMTs or "
            "SPADs) with timing electronics (Becker & Hickl SPC-150/830 or "
            "PicoQuant TimeHarp). For widefield FLIM, use a gated or modulated "
            "camera (Lambert Instruments). Synchronize laser pulses with "
            "detector timing."
        ),
        "common_algorithms": [
            "Mono-exponential / bi-exponential tail fitting (least-squares or MLE)",
            "Phasor analysis (model-free lifetime decomposition)",
            "Global analysis (linked lifetime fitting across pixels)",
            "Bayesian lifetime estimation",
            "Deep-learning FLIM (FLIMnet, rapid lifetime prediction from few photons)",
        ],
        "common_mistakes": [
            "Insufficient photon counts for reliable lifetime fitting (need ≥1000 photons/pixel)",
            "Ignoring instrument response function (IRF) convolution in the fit",
            "Using mono-exponential fit for multi-component decays, obtaining incorrect average lifetimes",
            "Pile-up effect at high count rates distorting the decay histogram",
            "Background autofluorescence contributing a long-lifetime component",
        ],
        "how_to_avoid_mistakes": [
            "Collect sufficient photons; use longer acquisition or binning if needed",
            "Measure IRF with a scattering sample and convolve with the model in fitting",
            "Evaluate fit residuals; use bi-exponential or phasor if mono-exponential is poor",
            "Keep count rate below 1-5 % of the laser repetition rate to avoid pile-up",
            "Measure autofluorescence lifetime separately and include in the fit model",
        ],
    },

    "fpm": {
        "principle": (
            "Fourier Ptychographic Microscopy synthetically increases the NA of "
            "a low-magnification objective by illuminating the sample from "
            "multiple angles (LED array) and computationally stitching "
            "together the resulting images in Fourier space. Each LED angle "
            "shifts the sample spectrum so different spatial-frequency bands "
            "enter the objective pupil, allowing recovery of both amplitude "
            "and phase at high resolution over a large field of view."
        ),
        "setup_guide": (
            "Replace the microscope condenser with a programmable LED matrix "
            "(e.g., 32×32 RGB LED array, ~4 mm pitch, placed ~80 mm above the "
            "sample). Use a low-magnification objective (4-10×, 0.1-0.3 NA) "
            "for large FOV. Acquire one image per LED (typically 100-300 images "
            "for the full matrix). Precise knowledge of LED positions is "
            "required for Fourier-space stitching."
        ),
        "common_algorithms": [
            "Alternating projection (Gerchberg-Saxton style in Fourier space)",
            "Embedded pupil function recovery (joint sample + aberration estimation)",
            "Wirtinger gradient descent with total-variation regularization",
            "Neural network-accelerated FPM (learned initialization + refinement)",
            "Multiplexed FPM (multiple LEDs simultaneously for faster acquisition)",
        ],
        "common_mistakes": [
            "Inaccurate LED position calibration causing ghosting and resolution loss",
            "Insufficient overlap between Fourier-space patches (need ≥60 % overlap)",
            "Ignoring pupil aberrations of the low-NA objective",
            "LED intensity non-uniformity not corrected across the array",
            "Vibration or sample drift between sequential LED acquisitions",
        ],
        "how_to_avoid_mistakes": [
            "Calibrate LED positions using a self-calibration algorithm or known test target",
            "Ensure adequate angular spacing to maintain >60% Fourier overlap between adjacent LEDs",
            "Use embedded pupil recovery to jointly estimate and correct aberrations",
            "Normalize LED intensities with a blank-sample calibration acquisition",
            "Stabilize the setup mechanically; use fast cameras to minimize inter-frame drift",
        ],
    },

    "two_photon": {
        "principle": (
            "Two-photon excitation uses a pulsed near-infrared laser so that "
            "two photons are absorbed simultaneously by a fluorophore, producing "
            "fluorescence equivalent to a single photon of half the wavelength. "
            "Because absorption depends on the square of intensity, fluorescence "
            "is generated only at the tight focus, providing intrinsic optical "
            "sectioning without a pinhole. Deep tissue penetration (up to ~1 mm) "
            "is achieved due to reduced scattering at NIR wavelengths."
        ),
        "setup_guide": (
            "Install a mode-locked Ti:Sapphire laser (680-1080 nm, ~100 fs "
            "pulses, 80 MHz, Coherent Chameleon or Spectra-Physics InSight) "
            "on a laser-scanning microscope. Use a high-NA water-dipping "
            "objective (25x 1.05 NA or 20x 1.0 NA) for deep imaging. Non-"
            "descanned detectors (GaAsP PMTs) collect scattered fluorescence "
            "close to the objective for maximum efficiency. Add a Pockels cell "
            "for fast power modulation."
        ),
        "common_algorithms": [
            "Adaptive background subtraction for in-depth imaging",
            "Motion correction and image registration for in-vivo data",
            "Suite2p / CaImAn (calcium imaging segmentation and trace extraction)",
            "Deep-learning denoising (DeepInterpolation, Noise2Void)",
            "Attenuation compensation (exponential depth correction)",
        ],
        "common_mistakes": [
            "Excessive laser power causing photodamage and heating deep in tissue",
            "Pre-chirp not compensated, broadening pulses and reducing two-photon efficiency",
            "Crosstalk between emission channels when using multiple fluorophores",
            "Brain motion artifacts in in-vivo imaging not corrected",
            "Imaging too deep without correcting for signal attenuation with depth",
        ],
        "how_to_avoid_mistakes": [
            "Titrate laser power to minimum effective level; monitor for tissue damage signs",
            "Use a prism-pair or grating pre-chirp compressor to maintain short pulses at the focus",
            "Select well-separated emission spectra and use appropriate dichroics and filters",
            "Apply real-time or post-hoc motion correction algorithms (rigid or non-rigid)",
            "Use adaptive optics or longer-wavelength excitation (three-photon) for deep tissue",
        ],
    },

    "sted": {
        "principle": (
            "Stimulated Emission Depletion microscopy breaks the diffraction "
            "limit by using a donut-shaped depletion beam to force fluorophores "
            "at the periphery of the excitation spot back to the ground state "
            "via stimulated emission. Only fluorophores at the very center of "
            "the donut emit spontaneously, shrinking the effective PSF to "
            "30-70 nm lateral resolution depending on depletion power."
        ),
        "setup_guide": (
            "Combine an excitation laser (e.g., 640 nm pulsed) with a "
            "co-aligned depletion laser (775 nm pulsed, ~1 ns) that passes "
            "through a vortex phase plate to create the donut. Use a high-NA "
            "objective (100x 1.4 NA oil). Time-gate detection (1-6 ns after "
            "excitation pulse) to reject depletion photon leakage. Single-"
            "photon counting detectors (APDs or hybrid PMTs) are essential. "
            "Align the donut null precisely at the excitation center."
        ),
        "common_algorithms": [
            "Richardson-Lucy deconvolution with STED PSF",
            "Wiener deconvolution with known STED PSF",
            "Deep-learning restoration (content-aware STED denoising)",
            "Linear unmixing for multi-color STED",
            "Time-gated STED (g-STED) background subtraction",
        ],
        "common_mistakes": [
            "Misaligned donut null causing asymmetric PSF and resolution loss",
            "Excessive depletion power causing photobleaching of organic dyes",
            "Depletion laser leaking into fluorescence detection channel",
            "Insufficient time-gating, recording stimulated emission as signal",
            "Using fluorophores with poor STED compatibility (low stimulated-emission cross-section)",
        ],
        "how_to_avoid_mistakes": [
            "Regularly check and optimize donut alignment using gold nanoparticle scattering",
            "Use STED-optimized dyes (ATTO647N, SiR, Abberior STAR) and minimize power",
            "Install proper spectral filters and use time-gating to reject depletion photons",
            "Apply 1-6 ns detection gate synchronized with the pulsed excitation",
            "Choose fluorophores specifically designed for STED with high photostability",
        ],
    },

    "palm_storm": {
        "principle": (
            "Single-Molecule Localization Microscopy (PALM/STORM) achieves "
            "~20 nm resolution by stochastically switching individual "
            "fluorophores between bright and dark states. In each frame, only "
            "a sparse subset of molecules emit, allowing their positions to be "
            "localized with sub-pixel precision by fitting 2-D Gaussians. "
            "Thousands of frames are accumulated and all localizations are "
            "plotted to form a super-resolution image."
        ),
        "setup_guide": (
            "Use a TIRF microscope (100x 1.49 NA oil objective) with powerful "
            "laser excitation (200-500 mW at the sample, 647 nm for Alexa647 "
            "STORM or 561 nm for mEos PALM). TIRF geometry reduces background. "
            "An oxygen-scavenging buffer with thiol (MEA/BME) is critical for "
            "Alexa647 blinking. Use an EMCCD (Andor iXon 897) or fast sCMOS "
            "camera at 30-100 Hz frame rate. Acquire 10,000-50,000 frames."
        ),
        "common_algorithms": [
            "ThunderSTORM (ImageJ plugin, MLE/LSQ Gaussian fitting)",
            "SMLM ZOLA-3D (deep-learning 3D localization)",
            "DAOSTORM (multi-emitter fitting for high density)",
            "Drift correction (fiducial-based or cross-correlation)",
            "HAWK / ANNA-PALM (deep-learning for accelerated SMLM)",
        ],
        "common_mistakes": [
            "Density of active emitters too high, causing overlapping PSFs and localization errors",
            "Insufficient photon count per localization, yielding poor precision (>30 nm)",
            "Sample drift during long acquisitions not corrected",
            "Poor blinking statistics (incomplete on-off switching) from wrong buffer conditions",
            "Mistaking fixed-pattern noise or autofluorescence for single molecules",
        ],
        "how_to_avoid_mistakes": [
            "Tune activation laser to achieve sparse single-molecule density per frame",
            "Optimize buffer (pH, thiol concentration, oxygen scavenger) for bright blinks (>1000 photons)",
            "Include fiducial markers (gold beads or TetraSpeck) and apply drift correction",
            "Prepare fresh imaging buffer immediately before acquisition; degas thoroughly",
            "Apply quality filters (photon threshold, localization precision, PSF shape) in analysis",
        ],
    },

    "tirf": {
        "principle": (
            "Total Internal Reflection Fluorescence microscopy creates an "
            "evanescent wave that penetrates only ~100-200 nm into the sample "
            "when the excitation beam is totally internally reflected at the "
            "glass-sample interface. This provides excellent optical sectioning "
            "of membrane-proximal events (vesicle fusion, protein dynamics at "
            "the plasma membrane) with very low background."
        ),
        "setup_guide": (
            "Use a TIRF-capable objective (60-100x, 1.49 NA oil) on an "
            "inverted microscope. Launch the laser at the critical angle "
            "through the objective periphery (objective-type TIRF) or through "
            "a prism (prism-type TIRF). Verify total internal reflection by "
            "observing the evanescent field depth with a calibration sample. "
            "Cells must be plated on clean, high-RI coverslips (#1.5H, 170 μm)."
        ),
        "common_algorithms": [
            "Single-particle tracking (SPT) algorithms",
            "Multi-angle TIRF for axial sectioning (variable penetration depth)",
            "Denoising (Gaussian filtering, wavelet, or deep-learning)",
            "Photobleaching step analysis for molecular counting",
            "Temporal median filtering for background subtraction",
        ],
        "common_mistakes": [
            "Laser angle not precisely at TIR, partially exciting bulk fluorescence",
            "Dirty coverslips causing scattering and destroying evanescent field uniformity",
            "Cells not well-adhered to the coverslip surface, out of evanescent field range",
            "Using objectives with NA < 1.45, insufficient for TIR at aqueous interfaces",
            "Evanescent field depth not calibrated, making quantitative axial analysis unreliable",
        ],
        "how_to_avoid_mistakes": [
            "Fine-tune the TIR angle while observing a known sample; verify exponential depth decay",
            "Clean coverslips rigorously (plasma cleaning or acid wash) before plating cells",
            "Use poly-L-lysine or fibronectin coating to ensure cells adhere to the coverslip",
            "Use 1.49 NA objectives; 1.45 NA is the minimum for aqueous TIR",
            "Calibrate evanescent field depth using fluorescent beads at known axial positions",
        ],
    },

    "polarization": {
        "principle": (
            "Polarization microscopy exploits the birefringence (orientation-"
            "dependent refractive index) of ordered biological structures such "
            "as collagen fibers, spindle microtubules, and crystalline "
            "inclusions. By analyzing the polarization state of transmitted or "
            "reflected light, structural anisotropy can be measured without "
            "fluorescent labeling. Quantitative techniques (LC-PolScope) "
            "measure both retardance magnitude and slow-axis orientation."
        ),
        "setup_guide": (
            "Mount a liquid-crystal universal compensator (LC-PolScope by "
            "OpenPolScope, or Abrio system) on a standard brightfield "
            "microscope. Use strain-free optics and rotate the analyzer while "
            "keeping the polarizer fixed (or use a rotating stage). For "
            "quantitative imaging, acquire 4-5 images at different compensator "
            "settings. A monochromatic light source (546 nm green filter) "
            "minimizes chromatic effects."
        ),
        "common_algorithms": [
            "Mueller matrix decomposition (full polarimetric imaging)",
            "Jones calculus for coherent polarization analysis",
            "Background retardance subtraction",
            "Stokes parameter reconstruction from intensity measurements",
            "Deep-learning retardance estimation from fewer raw frames",
        ],
        "common_mistakes": [
            "Strain birefringence in optical components contaminating the measurement",
            "Incorrect compensator calibration producing quantitative retardance errors",
            "Not accounting for sample tilt introducing apparent birefringence artifacts",
            "Using polychromatic light causing wavelength-dependent retardance errors",
            "Ignoring depolarization effects in thick or scattering samples",
        ],
        "how_to_avoid_mistakes": [
            "Use strain-free objectives and verify zero retardance on a blank field",
            "Calibrate the liquid-crystal compensator at each session using a known retarder",
            "Ensure sample is flat and perpendicular to the optical axis",
            "Use narrow-band illumination or measure dispersion for wavelength correction",
            "For thick samples, consider Mueller matrix imaging to capture depolarization",
        ],
    },

    "lensless": {
        "principle": (
            "Lensless (diffuser-cam) imaging replaces the imaging lens with "
            "a thin diffuser or coded mask placed directly before the sensor. "
            "The sensor records a multiplexed pattern (caustic or speckle) that "
            "encodes the 3-D scene. Computational reconstruction inverts the "
            "known point-spread function of the diffuser to recover the image, "
            "enabling an extremely compact, lightweight camera suitable for "
            "miniaturized or in-vivo applications."
        ),
        "setup_guide": (
            "Place a thin diffuser (ground glass, engineered phase mask, or "
            "Scotch tape) at a fixed, small distance (~1-5 mm) from a bare "
            "sensor (CMOS, e.g., Sony IMX sensor). Precisely characterize the "
            "diffuser PSF by scanning a point source across the field of view. "
            "Mount rigidly to prevent any relative motion between diffuser and "
            "sensor. For 3-D reconstruction, the depth-dependent PSF must be "
            "calibrated at multiple axial planes."
        ),
        "common_algorithms": [
            "ADMM (alternating direction method of multipliers) with TV regularization",
            "Wiener deconvolution (fast, single-step but lower quality)",
            "Gradient descent with learned priors (DiffuserCam, neural network prior)",
            "Tikhonov-regularized least squares",
            "Unrolled optimization networks (physics-informed deep learning)",
        ],
        "common_mistakes": [
            "Inaccurate PSF calibration causing reconstruction artifacts",
            "Insufficient sensor dynamic range for the caustic intensity peaks",
            "Motion between diffuser and sensor during capture invalidating the PSF model",
            "Regularization too strong, over-smoothing fine details in the reconstruction",
            "Ignoring the depth-dependence of the PSF when imaging 3-D scenes",
        ],
        "how_to_avoid_mistakes": [
            "Calibrate PSF carefully with a point source at the exact sample distance",
            "Use HDR acquisition or high-bit-depth sensors to capture full caustic range",
            "Rigidly bond the diffuser to the sensor; verify alignment stability",
            "Tune regularization weight (e.g., via L-curve or cross-validation)",
            "Calibrate PSF at multiple depths for 3-D scenes; use depth-varying reconstruction",
        ],
    },

    # ── COMPRESSIVE ────────────────────────────────────────────────────────

    "cassi": {
        "principle": (
            "Coded Aperture Snapshot Spectral Imaging (CASSI) captures a full "
            "3-D spectral datacube (x, y, λ) in a single 2-D snapshot by "
            "encoding the scene with a binary coded aperture and spectrally "
            "dispersing it with a prism onto the detector. Different spectral "
            "channels are shifted and superimposed on the sensor, creating a "
            "compressed measurement. Computational algorithms recover the full "
            "datacube from this single measurement using sparsity priors."
        ),
        "setup_guide": (
            "Build an optical relay with an objective lens, place a binary "
            "coded aperture (lithographic chrome-on-glass mask or DMD) at an "
            "intermediate image plane, then disperse with an Amici or double-"
            "Amici prism, and re-image onto a high-resolution detector (2048× "
            "2048+ pixels). Precisely calibrate the spectral dispersion curve "
            "(nm/pixel). The coded aperture pattern should have ~50 % transmittance "
            "and good conditioning."
        ),
        "common_algorithms": [
            "TwIST (Two-step Iterative Shrinkage/Thresholding)",
            "GAP-TV (Generalized Alternating Projection with Total Variation)",
            "ADMM with sparsity in DCT or wavelet domain",
            "Deep unfolding networks (DGSMP, TSA-Net, BIRNAT)",
            "Plug-and-Play ADMM with learned denoisers",
        ],
        "common_mistakes": [
            "Poor spectral calibration causing wavelength assignment errors across the datacube",
            "Coded aperture not precisely at the image plane, blurring the code modulation",
            "Insufficient detector resolution relative to the number of spectral bands",
            "Ignoring optical aberrations in the dispersive relay that vary with wavelength",
            "Using a random mask without checking its sensing matrix condition number",
        ],
        "how_to_avoid_mistakes": [
            "Calibrate spectral mapping with monochromatic sources at known wavelengths",
            "Mount coded aperture on a precision z-stage and focus to maximize modulation contrast",
            "Ensure detector pixel count > (spatial pixels × spectral bands) for adequate compression ratio",
            "Design the relay optics for uniform imaging quality across the spectral range",
            "Optimize or simulate the mask pattern for low coherence (good RIP) before fabrication",
        ],
    },

    "spc": {
        "principle": (
            "A single-pixel camera uses a spatial light modulator (DMD) to "
            "project a sequence of binary or grayscale patterns onto the scene. "
            "Each pattern multiplies the scene, and a single bucket detector "
            "(photodiode or PMT) measures the total light for each pattern, "
            "producing one scalar measurement per pattern. Compressive sensing "
            "recovers the image from far fewer measurements than Nyquist by "
            "exploiting sparsity in a transform domain."
        ),
        "setup_guide": (
            "Place a DMD (e.g., Texas Instruments DLP LightCrafter) at the "
            "image plane of a relay lens. Focus the scene onto the DMD. After "
            "the DMD, collect all reflected light onto a single photodetector "
            "(avalanche photodiode for low light, or silicon photodiode for "
            "visible). Display Hadamard, random, or optimized patterns at "
            "10-22 kHz DMD rate. Synchronize pattern display with detector "
            "readout."
        ),
        "common_algorithms": [
            "Basis pursuit / L1 minimization (LASSO)",
            "Orthogonal matching pursuit (OMP)",
            "Total-variation minimization (TV-CS)",
            "TVAL3 (TV with augmented Lagrangian and alternating direction)",
            "Deep compressive sensing networks (ReconNet, CSNet)",
        ],
        "common_mistakes": [
            "Pattern-detector timing mismatch causing wrong measurement-to-pattern association",
            "DMD diffraction effects not accounted for at oblique illumination angles",
            "Insufficient measurements for the scene complexity (under-sampling ratio too aggressive)",
            "Analog-to-digital converter resolution too low for the dynamic range of measurements",
            "Not calibrating detector linearity and dark current drift during long acquisitions",
        ],
        "how_to_avoid_mistakes": [
            "Hardware-trigger the detector acquisition from the DMD synchronization signal",
            "Calibrate the effective pattern at the sample plane (not just the DMD command pattern)",
            "Start with 25-50 % measurement ratio for natural scenes; reduce only if sparsity allows",
            "Use 16-bit or higher ADC; verify linearity with a calibrated light source",
            "Measure dark frames periodically and subtract; maintain stable detector temperature",
        ],
    },

    "cacti": {
        "principle": (
            "Coded Aperture Compressive Temporal Imaging (CACTI) compresses "
            "multiple high-speed video frames into a single sensor exposure by "
            "modulating the scene with a dynamic coded aperture (shifting mask) "
            "during the integration time. The sensor accumulates a coded sum "
            "of B consecutive frames, and computational algorithms recover all "
            "B frames from the single compressed measurement using video "
            "sparsity priors."
        ),
        "setup_guide": (
            "Build a relay optical system with a physical translating mask or "
            "use a DMD as the coded aperture at an intermediate image plane. "
            "The mask shifts by one pixel per sub-frame interval during the "
            "camera integration time, effectively encoding B temporal frames. "
            "Use a standard camera at normal frame rate (e.g., 30 fps) to "
            "capture the compressed measurement. Calibrate the mask pattern "
            "and its motion precisely."
        ),
        "common_algorithms": [
            "GAP-TV (Generalized Alternating Projection with Total Variation)",
            "DeSCI (Decompress Snapshot Compressive Imaging, GMM prior)",
            "PnP-FFDNet (Plug-and-Play with FFDNet denoiser)",
            "Deep unfolding: BIRNAT, RevSCI, EfficientSCI",
            "E2E-trained networks: STFormer, CST (transformer-based)",
        ],
        "common_mistakes": [
            "Mask calibration error causing temporal frame misalignment in reconstruction",
            "Compression ratio too high (too many sub-frames per snapshot) for the scene motion",
            "Motion blur within individual sub-frame intervals when scene moves fast",
            "Non-uniform mask illumination creating brightness gradients in recovered frames",
            "Choosing masks with poor conditioning (high mutual coherence between rows)",
        ],
        "how_to_avoid_mistakes": [
            "Calibrate mask position precisely using a static known pattern before experiments",
            "Limit compression ratio (B ≤ 8-10 for complex natural scenes; B ≤ 24-48 for simpler scenes)",
            "Ensure sub-frame exposure is short enough that intra-frame motion is negligible",
            "Flatfield-correct the mask modulation using a uniform target calibration",
            "Simulate reconstruction quality with candidate mask patterns before hardware fabrication",
        ],
    },

    "matrix": {
        "principle": (
            "Generic matrix sensing models the forward process as y = Ax + n, "
            "where A is an arbitrary measurement matrix (not necessarily "
            "structured like a convolution or Radon transform). This is the "
            "most general compressive sensing framework, applicable to random "
            "projections, coded apertures, and any linear dimensionality "
            "reduction scheme. The key requirement is that A satisfies the "
            "Restricted Isometry Property (RIP) for successful sparse recovery."
        ),
        "setup_guide": (
            "Implementation depends on the physical sensing modality. For "
            "optical random projections, use a DMD or scattering medium to "
            "implement pseudo-random measurement vectors. Calibrate the "
            "measurement matrix A by measuring the system response to a "
            "complete basis set (e.g., Hadamard patterns). Store A as a "
            "dense or structured matrix. Ensure the measurement SNR is "
            "adequate for the desired reconstruction quality."
        ),
        "common_algorithms": [
            "ISTA / FISTA (Iterative Shrinkage-Thresholding Algorithm)",
            "Basis pursuit (L1 minimization via linear programming)",
            "AMP (Approximate Message Passing)",
            "ADMM with various regularizers (TV, wavelet sparsity, low-rank)",
            "Learned ISTA (LISTA) and other deep unfolding networks",
        ],
        "common_mistakes": [
            "Measurement matrix does not satisfy RIP (too coherent or poorly conditioned)",
            "Mismatch between calibrated A and actual system behavior (model error)",
            "Not accounting for measurement noise level when setting regularization strength",
            "Using an insufficiently sparse signal model for the reconstruction",
            "Ignoring quantization effects of the detector in the measurement model",
        ],
        "how_to_avoid_mistakes": [
            "Verify the condition number and coherence of A; use random or optimized designs",
            "Re-calibrate A periodically to account for system drift",
            "Set regularization parameter proportional to noise level (e.g., via cross-validation)",
            "Validate sparsity assumption on representative signals before deploying CS",
            "Include quantization noise in the forward model or use dithering techniques",
        ],
    },

    # ── MEDICAL ────────────────────────────────────────────────────────────

    "ct": {
        "principle": (
            "X-ray Computed Tomography reconstructs cross-sectional images "
            "from multiple X-ray projection measurements acquired at different "
            "angles around the patient. The Beer-Lambert law governs X-ray "
            "attenuation: I = I₀ exp(-∫μ(x,y) dl), and the Radon transform "
            "relates projections to the attenuation map. Filtered back-"
            "projection or iterative algorithms invert the Radon transform "
            "to produce volumetric images."
        ),
        "setup_guide": (
            "A clinical CT scanner consists of a rotating gantry with an "
            "X-ray tube (80-140 kVp, 50-800 mA) and a curved detector array "
            "(64-320 rows of scintillator-photodiode elements) on opposing "
            "sides. The gantry rotates at 0.25-0.5 s per revolution. Helical "
            "scanning moves the patient table continuously through the gantry. "
            "Key calibrations: air scans, detector gain normalization, "
            "beam-hardening correction LUTs, and geometric calibration."
        ),
        "common_algorithms": [
            "Filtered back-projection (FBP) with Ram-Lak or Shepp-Logan filter",
            "FDK (Feldkamp-Davis-Kress) for cone-beam geometry",
            "Iterative reconstruction: SART, OS-SIRT",
            "Model-based iterative reconstruction (MBIR) with statistical noise model",
            "Deep-learning reconstruction (FBPConvNet, LEARN, WGAN-VGG for low-dose CT)",
        ],
        "common_mistakes": [
            "Ring artifacts from uncorrected detector gain variations",
            "Beam-hardening artifacts (cupping, streaks near bone/metal) not corrected",
            "Patient motion during scan causing blurring and streaks",
            "Insufficient angular sampling producing streak or aliasing artifacts",
            "Metal artifacts from implants overwhelming reconstruction algorithms",
        ],
        "how_to_avoid_mistakes": [
            "Perform regular air calibrations and detector flatfield corrections",
            "Apply polynomial beam-hardening correction or dual-energy decomposition",
            "Use gating (cardiac/respiratory) or fast rotation to reduce motion artifacts",
            "Ensure adequate number of projections (≥ π × detector columns for FBP)",
            "Use metal artifact reduction algorithms (MAR, iterative forward-projection inpainting)",
        ],
    },

    "xray_radiography": {
        "principle": (
            "X-ray radiography produces a 2-D projection image of the patient's "
            "internal structures by measuring the transmitted X-ray intensity "
            "after passing through the body. Dense structures (bone, metal) "
            "attenuate more X-rays and appear bright on the detector. The "
            "image represents the line-integral of the attenuation coefficient "
            "along each ray path."
        ),
        "setup_guide": (
            "An X-ray tube (stationary or rotating anode, 40-150 kVp) "
            "produces a divergent beam. The patient stands or lies between "
            "the tube and a flat-panel detector (amorphous silicon with CsI "
            "scintillator, or amorphous selenium for direct conversion). "
            "Anti-scatter grid (Bucky grid) is placed before the detector. "
            "Automatic exposure control (AEC) sets mAs based on patient "
            "thickness. Calibration includes dark field, flatfield, and "
            "defective pixel mapping."
        ),
        "common_algorithms": [
            "Flat-field correction (gain/offset normalization)",
            "Logarithmic transform for linear attenuation mapping",
            "Anti-scatter grid artifact removal",
            "Dual-energy subtraction (bone/soft-tissue separation)",
            "Deep-learning denoising for low-dose radiography",
        ],
        "common_mistakes": [
            "Under-exposure causing excessive quantum noise, especially in obese patients",
            "Grid artifacts from misaligned anti-scatter grid",
            "Patient motion blur in long-exposure radiographs",
            "Incorrect windowing (display LUT) obscuring diagnostic information",
            "Scatter radiation degrading image contrast in thick body parts",
        ],
        "how_to_avoid_mistakes": [
            "Use AEC and verify exposure indicator falls within acceptable range",
            "Ensure grid is properly aligned with the X-ray focal spot distance",
            "Use shortest possible exposure time; instruct patient to hold breath",
            "Apply appropriate DICOM windowing presets for the anatomical region",
            "Use an appropriate anti-scatter grid ratio (8:1 to 12:1) for thick body parts",
        ],
    },

    "fluoroscopy": {
        "principle": (
            "Fluoroscopy provides real-time continuous X-ray imaging for "
            "guiding interventional procedures. A pulsed or continuous X-ray "
            "beam produces live projection images at 7.5-30 fps on a flat-panel "
            "detector. The trade-off is between frame rate, radiation dose, and "
            "image quality. Temporal filtering and dose-saving modes reduce "
            "patient exposure while maintaining diagnostic quality."
        ),
        "setup_guide": (
            "A C-arm fluoroscopy unit has an X-ray tube and flat-panel "
            "detector on a C-shaped gantry that can rotate around the patient. "
            "Modern systems use pulsed fluoroscopy (variable pulse rate 3.75-"
            "30 fps) with automatic brightness control. Install last-image-hold "
            "and virtual collimation features. Calibrate geometric distortion "
            "for 3-D cone-beam reconstruction capability. Regular dosimetry "
            "checks (DAP meter calibration) are mandatory."
        ),
        "common_algorithms": [
            "Recursive temporal averaging (IIR filtering for noise reduction)",
            "Contrast-enhanced subtraction (road-mapping for angiography)",
            "Motion-compensated temporal filtering",
            "Cone-beam CT reconstruction from rotational fluoroscopy runs",
            "Deep-learning frame interpolation for reduced pulse-rate operation",
        ],
        "common_mistakes": [
            "Excessive radiation dose from unnecessarily high frame rate or continuous mode",
            "Image lag / ghosting from slow detector response at low dose",
            "Geometric distortion from C-arm flex not calibrated",
            "Scatter degrading contrast in lateral or oblique views of thick anatomy",
            "Patient skin dose exceeding threshold (2 Gy) during long procedures",
        ],
        "how_to_avoid_mistakes": [
            "Use lowest acceptable pulse rate; employ last-image-hold instead of continuous fluoro",
            "Use fast flat-panel detectors (GOS or CsI with fast readout) to minimize lag",
            "Perform regular geometric calibration with a phantom for accurate 3D reconstruction",
            "Collimate tightly and use appropriate anti-scatter grids",
            "Monitor cumulative dose (DAP) and skin dose during procedures; rotate beam angles",
        ],
    },

    "mammography": {
        "principle": (
            "Mammography uses low-energy X-rays (25-35 kVp) with specialized "
            "anode/filter combinations (Mo/Mo, Mo/Rh, W/Rh) to optimize "
            "contrast between breast tissue types (adipose, glandular, "
            "calcifications). Breast compression reduces thickness and scatter, "
            "improving contrast and reducing dose. Digital mammography uses "
            "flat-panel detectors for direct or indirect X-ray detection."
        ),
        "setup_guide": (
            "A dedicated mammography unit with a compression paddle, specialized "
            "X-ray tube (Mo, Rh, or W anode), and high-resolution flat-panel "
            "detector (50-100 μm pixel size, amorphous selenium for direct "
            "conversion). Automatic optimization of target/filter and kVp "
            "based on compressed breast thickness. Regular quality assurance "
            "per ACR/MQSA requirements: phantom images, SNR measurements, "
            "artifact checks, and AEC calibration."
        ),
        "common_algorithms": [
            "Contrast-limited adaptive histogram equalization (CLAHE) for display",
            "Computer-aided detection (CAD) for microcalcification and mass detection",
            "Digital breast tomosynthesis (DBT) reconstruction (FBP or iterative)",
            "Deep-learning breast density classification (BI-RADS categories)",
            "Synthetic 2D mammography from DBT volumes",
        ],
        "common_mistakes": [
            "Insufficient breast compression, increasing dose and reducing contrast",
            "Positioning errors cutting off breast tissue (especially axillary tail)",
            "Grid artifacts or grid cutoff from misaligned Bucky grid",
            "Exposure errors from AEC sensor placed over dense tissue vs. adipose",
            "Motion blur from long exposure times in thick or dense breasts",
        ],
        "how_to_avoid_mistakes": [
            "Apply firm, consistent compression; verify thickness readout is reasonable",
            "Follow standardized positioning protocols (CC, MLO) with technologist training",
            "Verify grid alignment and use reciprocating grid to eliminate grid lines",
            "Position AEC sensor appropriately for breast density; adjust manually if needed",
            "Use shortest possible exposure with adequate mAs; consider large-angle tomosynthesis",
        ],
    },

    "dexa": {
        "principle": (
            "Dual-Energy X-ray Absorptiometry uses two X-ray beam energies to "
            "decompose the body into bone mineral and soft tissue compartments. "
            "The differential attenuation of the two energies allows separation "
            "of bone from soft tissue. Bone mineral density (BMD, g/cm²) is "
            "computed by comparing attenuation to calibration phantoms."
        ),
        "setup_guide": (
            "A DEXA scanner (Hologic Discovery/Horizon or GE Lunar) uses a "
            "fan-beam or pencil-beam X-ray source with two energies (typically "
            "70 and 140 kVp, or k-edge filtration). The detector is directly "
            "opposite the source below the patient table. Daily quality "
            "assurance with a calibration phantom (anthropomorphic spine) is "
            "mandatory. Cross-calibration is needed when changing scanners. "
            "Scan modes include AP spine, dual femur, whole body, and lateral "
            "vertebral assessment."
        ),
        "common_algorithms": [
            "Dual-energy decomposition (two-material model: bone + soft tissue)",
            "Edge detection for region-of-interest (ROI) identification",
            "BMD calculation relative to calibration phantom",
            "T-score / Z-score computation against normative databases",
            "Body composition analysis (lean mass, fat mass from whole-body scans)",
        ],
        "common_mistakes": [
            "Patient positioning errors (rotation, wrong vertebral level) affecting BMD",
            "Not removing metal objects (belts, jewelry) that artifactually increase BMD",
            "Comparing BMD values from different scanner manufacturers without cross-calibration",
            "Degenerative changes (osteophytes) falsely elevating spine BMD",
            "Analyzing the wrong vertebral levels or including fractured vertebrae",
        ],
        "how_to_avoid_mistakes": [
            "Standardize patient positioning with positioning aids; verify on scout image",
            "Remove all metal from scan field; use lateral spine view to avoid artifacts",
            "Use same scanner for serial monitoring; cross-calibrate if changing equipment",
            "Evaluate AP spine image for degenerative changes; consider lateral spine or femur",
            "Follow ISCD guidelines for vertebral inclusion/exclusion criteria in analysis",
        ],
    },

    "cbct": {
        "principle": (
            "Cone-Beam CT uses a divergent cone-shaped X-ray beam and a 2-D "
            "flat-panel detector to acquire a volumetric CT dataset in a single "
            "rotation. Unlike multi-slice CT with a narrow fan beam, CBCT "
            "covers the full volume simultaneously, enabling faster acquisition "
            "but with increased scatter and cone-beam artifacts compared to "
            "conventional CT."
        ),
        "setup_guide": (
            "Mount a flat-panel detector (typically 30×40 cm, CsI scintillator) "
            "opposite an X-ray tube on a rotating gantry or C-arm. Common "
            "implementations: dental CBCT (small FOV, 90 kVp), image-guided "
            "radiation therapy CBCT (kV source on linac gantry), and C-arm "
            "CBCT (interventional). Calibrate: geometric parameters (source-"
            "detector distances, isocenter), detector offset corrections, "
            "and scatter correction LUTs."
        ),
        "common_algorithms": [
            "FDK (Feldkamp-Davis-Kress) cone-beam filtered back-projection",
            "Iterative CBCT (SART, SIRT with cone-beam projector)",
            "Scatter correction (measurement-based or Monte Carlo simulation)",
            "Motion-compensated CBCT (4D-CBCT for respiratory motion)",
            "Deep-learning CBCT-to-CT synthesis for radiation therapy planning",
        ],
        "common_mistakes": [
            "Severe scatter artifacts (cupping, shading) in large FOV acquisitions",
            "Cone-beam artifacts near the edges of the FOV (Feldkamp approximation breaks down)",
            "Truncation artifacts when anatomy extends outside the FOV",
            "Motion artifacts in thorax/abdomen from respiratory and cardiac motion",
            "Insufficient angular sampling causing streak artifacts",
        ],
        "how_to_avoid_mistakes": [
            "Apply scatter correction (anti-scatter grid, software correction, or beam-blocker method)",
            "Limit cone angle or use exact reconstruction algorithms for large cone angles",
            "Use extended FOV techniques (shifted detector, multiple scans) for large anatomy",
            "Apply 4D-CBCT or gated acquisition for moving anatomy",
            "Acquire sufficient projections (≥600 for a full rotation) with uniform angular spacing",
        ],
    },

    "angiography": {
        "principle": (
            "X-ray angiography visualizes blood vessels by injecting iodinated "
            "contrast agent and acquiring rapid-sequence fluoroscopic images. "
            "Digital Subtraction Angiography (DSA) subtracts a pre-contrast "
            "mask image from post-contrast frames, removing bone and soft "
            "tissue to show only the contrast-filled vasculature with high "
            "contrast and spatial resolution."
        ),
        "setup_guide": (
            "Use a biplane or single-plane angiography suite with high-speed "
            "flat-panel detectors (30-60 fps capability). The C-arm provides "
            "multi-angle positioning. Power injector delivers iodinated "
            "contrast (350-370 mgI/mL) at controlled rates. Road-mapping "
            "mode overlays vessel map on live fluoro for catheter guidance. "
            "3-D rotational angiography acquires a spin to reconstruct a "
            "volume of the vasculature."
        ),
        "common_algorithms": [
            "Digital subtraction (mask-live image subtraction)",
            "Pixel shifting for motion compensation in DSA",
            "3-D rotational angiography reconstruction (FDK or iterative)",
            "Time-density curve analysis for perfusion assessment",
            "Deep-learning vessel segmentation and stenosis quantification",
        ],
        "common_mistakes": [
            "Patient motion between mask and contrast frames causing misregistration artifacts",
            "Inadequate contrast bolus timing causing suboptimal vessel opacification",
            "Overexposure or underexposure of the detector outside the linear range",
            "Bowel gas or cardiac motion causing subtraction artifacts",
            "Injecting contrast too fast, creating reflux or missing distal vessels",
        ],
        "how_to_avoid_mistakes": [
            "Instruct patients to remain still; use pixel shifting or elastic registration",
            "Use test bolus or timing run to determine optimal injection-to-imaging delay",
            "Use automatic dose rate control; verify detector within calibrated dynamic range",
            "Use cardiac gating for coronary or thoracic angiography",
            "Adjust injection rate and volume to vessel size and flow characteristics",
        ],
    },

    "photoacoustic": {
        "principle": (
            "Photoacoustic imaging converts absorbed pulsed laser light into "
            "ultrasound via thermoelastic expansion. Short laser pulses (<10 ns) "
            "are absorbed by tissue chromophores (hemoglobin, melanin), causing "
            "rapid thermal expansion that generates broadband acoustic waves. "
            "These waves are detected by ultrasound transducers and "
            "reconstructed to form images reflecting optical absorption "
            "contrast at ultrasonic spatial resolution."
        ),
        "setup_guide": (
            "Combine a tunable pulsed laser (Nd:YAG pumped OPO, 680-1100 nm, "
            "5-20 ns pulses, 10-20 Hz) with an ultrasound transducer array "
            "(linear or curved, 5-40 MHz). Deliver light via fiber bundle to "
            "the tissue surface adjacent to the transducer. Use a multi-channel "
            "DAQ (12-14 bit, 40-100 MS/s) to record acoustic signals. For "
            "tomographic PAT, surround the sample with a ring or spherical "
            "array of transducers."
        ),
        "common_algorithms": [
            "Universal back-projection for photoacoustic tomography",
            "Time-reversal reconstruction",
            "Model-based iterative reconstruction with acoustic heterogeneity",
            "Spectral unmixing for multi-wavelength functional PA imaging",
            "Deep-learning PA image reconstruction (U-Net, pixel-wise inversion)",
        ],
        "common_mistakes": [
            "Insufficient laser fluence reaching target depth due to tissue scattering",
            "Acoustic heterogeneity (speed-of-sound variations) causing image distortion",
            "Limited-view artifacts from incomplete transducer coverage around the sample",
            "Coupling medium mismatch between transducer and tissue",
            "Laser safety violations from excessive skin surface fluence (>20 mJ/cm²)",
        ],
        "how_to_avoid_mistakes": [
            "Use NIR wavelengths (700-900 nm optical window) for deeper penetration",
            "Use speed-of-sound correction maps or joint reconstruction for heterogeneous media",
            "Maximize angular coverage of transducer array; use virtual-detector techniques",
            "Use appropriate acoustic coupling gel or water bath between transducer and tissue",
            "Monitor laser fluence at the tissue surface; comply with ANSI Z136.1 MPE limits",
        ],
    },

    "dot": {
        "principle": (
            "Diffuse Optical Tomography reconstructs 3-D maps of tissue optical "
            "properties (absorption μₐ and reduced scattering μ'ₛ) from "
            "measurements of multiply scattered near-infrared light transmitted "
            "through tissue. Multiple source-detector pairs on the tissue "
            "surface provide overlapping sensitivity profiles. The diffusion "
            "equation models light propagation in the multiple-scattering "
            "regime."
        ),
        "setup_guide": (
            "Place fiber-coupled NIR sources (670-850 nm laser diodes, CW or "
            "frequency-domain modulated at 100-300 MHz, or time-domain pulsed) "
            "and detector fibers (avalanche photodiodes or PMTs) on the tissue "
            "surface in an array. A multiplexer switches between source "
            "positions. For breast DOT, 32-128 optode positions on a cup or "
            "ring geometry. Calibrate with known optical phantoms (Intralipid "
            "+ ink solutions)."
        ),
        "common_algorithms": [
            "Normalized Born approximation (linearized diffuse optical tomography)",
            "Nonlinear Newton-type iterative reconstruction (Gauss-Newton, Levenberg-Marquardt)",
            "Finite-element method (FEM) based forward solver + Tikhonov regularization",
            "TOAST++ (Time-resolved Optical Absorption and Scattering Tomography)",
            "Deep-learning DOT (learned regularization, direct inversion networks)",
        ],
        "common_mistakes": [
            "Poor optode-tissue coupling due to hair, uneven surfaces, or insufficient pressure",
            "Inadequate source-detector pair coverage causing reconstruction blind spots",
            "Cross-talk between source channels if multiplexing is not properly timed",
            "Using the diffusion approximation too close to sources or in low-scattering regions",
            "Ignoring tissue heterogeneity in the background optical property estimate",
        ],
        "how_to_avoid_mistakes": [
            "Use spring-loaded optodes with coupling checks; shave hair in the measurement area",
            "Design source-detector geometry with overlapping sensitivity to cover the volume of interest",
            "Ensure clean channel switching with adequate settling time between multiplexed measurements",
            "Use higher-order transport models (radiative transfer) near sources if needed",
            "Initialize reconstruction with patient-specific anatomical prior (from MRI or CT)",
        ],
    },

    "pet": {
        "principle": (
            "Positron Emission Tomography detects pairs of 511 keV gamma rays "
            "emitted in opposite directions when a positron from a radiotracer "
            "annihilates with an electron. Coincidence detection of the two "
            "photons defines a line of response (LOR). Many LORs from different "
            "angles are reconstructed into a 3-D activity distribution map, "
            "providing functional and metabolic information."
        ),
        "setup_guide": (
            "A PET scanner consists of a ring of scintillation detector blocks "
            "(LYSO or LSO crystals coupled to SiPMs) surrounding the patient. "
            "Each detector block has a matrix of small crystals (3-4 mm pitch). "
            "Coincidence electronics pair detected events within a timing "
            "window (4-6 ns for TOF-PET). Modern digital PET systems achieve "
            "200-300 ps timing resolution for time-of-flight. Daily quality "
            "checks include detector normalization, timing calibration, and "
            "sensitivity phantom scans."
        ),
        "common_algorithms": [
            "OSEM (Ordered Subset Expectation Maximization)",
            "3D OSEM with resolution modeling (PSF reconstruction)",
            "TOF-OSEM (time-of-flight enhanced OSEM)",
            "Attenuation correction from CT (PET/CT) or Dixon MR (PET/MR)",
            "Deep-learning PET denoising (low-count to full-count prediction)",
        ],
        "common_mistakes": [
            "Incorrect attenuation correction map (misregistration between PET and CT)",
            "Patient motion between PET and CT causing attenuation-emission mismatch",
            "Metal artifacts in CT propagating into PET attenuation correction",
            "Scatter correction errors in patients with large body habitus",
            "SUV calculation errors from incorrect weight, dose, or timing entries",
        ],
        "how_to_avoid_mistakes": [
            "Verify PET-CT registration quality; use respiratory gating for thorax/abdomen",
            "Minimize time between CT and PET acquisitions; co-register if needed",
            "Use MAR-corrected CT or MR-based attenuation correction to avoid metal artifacts",
            "Use Monte Carlo scatter correction models validated for the patient population",
            "Double-check injected dose, patient weight, injection time, and decay correction",
        ],
    },

    "spect": {
        "principle": (
            "Single Photon Emission Computed Tomography detects single gamma-ray "
            "photons emitted by a radiotracer (⁹⁹ᵐTc, ¹²³I, ²⁰¹Tl) using a "
            "rotating gamma camera with a parallel-hole or pinhole collimator. "
            "The collimator provides directional sensitivity at the cost of "
            "low geometric efficiency (~0.01 %). Projections from multiple "
            "angles are reconstructed into 3-D activity maps."
        ),
        "setup_guide": (
            "A dual-head gamma camera (e.g., Siemens Symbia, GE Discovery) with "
            "NaI(Tl) scintillator crystals (9.5 mm thick) and parallel-hole "
            "collimators rotates around the patient (typically 60-128 angular "
            "stops over 360°). For cardiac SPECT, use dedicated CZT-based "
            "cameras with pinhole or multi-pinhole collimators. Acquire "
            "in step-and-shoot or continuous rotation mode. Energy windows are "
            "set around the photopeak (e.g., 140 keV ± 10 % for ⁹⁹ᵐTc)."
        ),
        "common_algorithms": [
            "FBP with ramp-Butterworth filter",
            "OSEM with attenuation and scatter correction",
            "Resolution recovery (collimator-detector response modeling in OSEM)",
            "CT-based attenuation correction (SPECT/CT)",
            "Deep-learning SPECT reconstruction (dose reduction, resolution enhancement)",
        ],
        "common_mistakes": [
            "Insufficient count statistics causing noisy, unreliable reconstructions",
            "Not correcting for depth-dependent collimator blur (resolution degrades with distance)",
            "Attenuation artifacts in uncorrected SPECT (false defects in myocardial perfusion)",
            "Patient motion during the long SPECT acquisition (15-30 minutes)",
            "Incorrect energy window or scatter window setup leading to poor image quality",
        ],
        "how_to_avoid_mistakes": [
            "Ensure adequate injected dose and acquisition time for sufficient count statistics",
            "Use resolution recovery (distance-dependent PSF modeling) in iterative reconstruction",
            "Apply CT-based attenuation correction; verify CT-SPECT registration",
            "Use motion detection and correction algorithms; shorter acquisitions with CZT cameras",
            "Verify energy window settings match the radionuclide photopeak and scatter windows",
        ],
    },

    "mri": {
        "principle": (
            "Magnetic Resonance Imaging measures the precession of hydrogen "
            "nuclear spins in a strong magnetic field (1.5-7 T). Radiofrequency "
            "pulses tip spins away from equilibrium, and gradient fields "
            "spatially encode the MR signal into k-space (spatial frequency "
            "domain). The image is obtained by inverse Fourier transform of "
            "k-space data. Contrast depends on tissue T1, T2, and proton "
            "density via the pulse sequence timing parameters."
        ),
        "setup_guide": (
            "A clinical MRI scanner has a superconducting magnet (1.5 T or 3 T), "
            "gradient coils (40-80 mT/m, 200 T/m/s slew rate), RF transmit "
            "body coil, and local receive coil arrays (8-128 channels). The "
            "patient lies inside the bore on a table. Key calibrations: "
            "center frequency, RF transmit calibration (B₁ mapping), shimming "
            "(B₀ homogeneity), and gradient eddy current compensation. Use "
            "pulse sequences optimized for the clinical question (T1w, T2w, "
            "FLAIR, DWI, etc.)."
        ),
        "common_algorithms": [
            "Inverse FFT (standard Cartesian k-space reconstruction)",
            "GRAPPA (GeneRalized Autocalibrating Partially Parallel Acquisitions)",
            "SENSE (SENSitivity Encoding) parallel imaging",
            "Compressed sensing MRI (L1-wavelet + TV regularization)",
            "Deep-learning MRI reconstruction (fastMRI, variational networks, E2E-VarNet)",
        ],
        "common_mistakes": [
            "Aliasing artifacts from insufficient FOV or acceleration too aggressive",
            "Motion artifacts (ghosting in phase-encode direction) from patient or physiological motion",
            "B₀ inhomogeneity causing geometric distortion and signal dropout (especially at 3T+)",
            "Fat-water chemical shift artifacts at fat-tissue interfaces",
            "Incorrect coil sensitivity maps causing SENSE/GRAPPA reconstruction artifacts",
        ],
        "how_to_avoid_mistakes": [
            "Set FOV to cover the anatomy with margin; use saturation bands to suppress aliasing",
            "Apply motion correction (navigator, PROPELLER, prospective correction) for moving anatomy",
            "Perform careful shimming; use distortion correction maps for EPI sequences",
            "Use fat suppression or water-fat separation (Dixon) sequences",
            "Acquire adequate auto-calibration data for parallel imaging; use robust coil maps",
        ],
    },

    "fmri": {
        "principle": (
            "Functional MRI detects brain activity indirectly through the Blood "
            "Oxygen Level Dependent (BOLD) contrast mechanism. Neural activity "
            "increases local blood flow and oxygenation, changing the ratio of "
            "diamagnetic oxyhemoglobin to paramagnetic deoxyhemoglobin. This "
            "alters the local T2* relaxation time, producing a small (~1-5 %) "
            "signal change detectable by gradient-echo EPI sequences acquired "
            "rapidly at whole-brain coverage."
        ),
        "setup_guide": (
            "Use a 3T MRI scanner with a 32-64 channel head coil. Acquire "
            "multi-band (simultaneous multi-slice) gradient-echo EPI sequences "
            "(TR 0.5-1.5 s, TE ~30 ms, 2 mm isotropic voxels, multiband "
            "factor 4-8). Include a high-resolution T1w structural scan for "
            "registration. Physiological monitoring (pulse oximetry, "
            "respiratory bellows) enables noise regression. Use foam padding "
            "to minimize head motion."
        ),
        "common_algorithms": [
            "General Linear Model (GLM) for task-based fMRI (FSL FEAT, SPM)",
            "ICA (Independent Component Analysis) for resting-state networks",
            "Seed-based functional connectivity analysis",
            "Motion correction and nuisance regression (6-parameter rigid body + CompCor)",
            "Deep-learning denoising and parcellation (BrainNetCNN, fMRIPrep pipeline)",
        ],
        "common_mistakes": [
            "Excessive head motion causing false activations or connectivity artifacts",
            "Not correcting for physiological noise (cardiac, respiratory) in the signal",
            "Insufficient statistical correction for multiple comparisons (inflated false positives)",
            "Using too long a TR, missing the hemodynamic response in fast event-related designs",
            "Geometric distortion in EPI not corrected before registration to structural scan",
        ],
        "how_to_avoid_mistakes": [
            "Use prospective motion correction and strict motion exclusion criteria (<0.5 mm FD)",
            "Acquire and regress physiological signals; use ICA-based denoising (ICA-AROMA)",
            "Apply proper multiple-comparison correction (FWE, FDR, cluster-based thresholding)",
            "Use multiband EPI for sub-second TR to adequately sample the HRF",
            "Acquire field maps (B₀) and apply distortion correction (topup, fieldmap-based)",
        ],
    },

    "mrs": {
        "principle": (
            "MR Spectroscopy measures the chemical shift spectrum of nuclear "
            "spins (usually ¹H) from a localized volume in the body, providing "
            "concentrations of metabolites such as NAA, creatine, choline, "
            "lactate, myo-inositol, and glutamate/glutamine. Chemical shift "
            "differences (in ppm) arise from the varying electronic shielding "
            "of nuclei in different molecular environments."
        ),
        "setup_guide": (
            "Use PRESS or STEAM single-voxel localization on a 1.5T or 3T "
            "scanner. Voxel sizes are typically 2×2×2 cm³ for brain. Suppress "
            "the dominant water signal (CHESS or VAPOR water suppression). "
            "Acquire 64-256 averages (NEX) for adequate SNR. Shimming is "
            "critical: water linewidth should be <12 Hz (3T) for the voxel. "
            "Multi-voxel CSI (Chemical Shift Imaging) maps metabolite "
            "distributions but requires longer acquisition and careful "
            "lipid suppression."
        ),
        "common_algorithms": [
            "LCModel (frequency-domain linear combination fitting)",
            "TARQUIN (open-source time-domain fitting)",
            "jMRUI (time-domain quantification with AMARES/QUEST)",
            "HSVD (Hankel SVD) for water removal and baseline correction",
            "Deep-learning spectral quantification (DeepSpectra, convolutional fitting)",
        ],
        "common_mistakes": [
            "Poor shimming producing broad linewidths that overlap metabolite peaks",
            "Voxel placed partly outside the brain, contaminating spectrum with lipid signal",
            "Insufficient water suppression saturating the spectrum baseline",
            "Too few averages, producing noisy spectra with unreliable metabolite estimates",
            "Ignoring macromolecular baseline contributions in fitting",
        ],
        "how_to_avoid_mistakes": [
            "Iteratively shim the voxel to achieve <12 Hz water linewidth (3T) before acquisition",
            "Place the voxel with margin from skull and subcutaneous fat; use outer-volume suppression",
            "Optimize water suppression parameters; acquire separate water reference for quantification",
            "Acquire sufficient averages: 128-256 for metabolites at low concentration (e.g., GABA)",
            "Include macromolecular basis set or measured baseline in the fitting model",
        ],
    },

    "diffusion_mri": {
        "principle": (
            "Diffusion MRI sensitizes the MR signal to the Brownian motion of "
            "water molecules by applying strong magnetic field gradient pulses "
            "(Stejskal-Tanner scheme). In fibrous tissue (e.g., white matter), "
            "water diffuses preferentially along fibers, creating directional "
            "diffusion anisotropy. Diffusion Tensor Imaging (DTI) models this "
            "as a 3×3 tensor; higher-order models (HARDI, CSD) resolve "
            "crossing fibers."
        ),
        "setup_guide": (
            "Acquire on a 3T scanner with high-performance gradients (80 mT/m, "
            "200 T/m/s). Use spin-echo EPI with multiple b-values (e.g., b=0, "
            "1000, 2000 s/mm²) and 30-300 diffusion directions uniformly "
            "distributed on the sphere. Include reverse-phase-encode b=0 "
            "images for EPI distortion correction. Multi-band (SMS) acceleration "
            "reduces scan time. Typical parameters: 2 mm isotropic, TE 60-90 ms, "
            "TR 3-5 s."
        ),
        "common_algorithms": [
            "DTI tensor fitting (least-squares or weighted least-squares)",
            "CSD (Constrained Spherical Deconvolution) for fiber orientation distribution",
            "NODDI (Neurite Orientation Dispersion and Density Imaging)",
            "Probabilistic tractography (FSL probtrackx, MRtrix3 iFOD2)",
            "Deep-learning tract segmentation (TractSeg, DeepBundle)",
        ],
        "common_mistakes": [
            "Eddy current and EPI geometric distortions not corrected, causing tract errors",
            "Insufficient number of diffusion directions for the chosen model complexity",
            "Using DTI in regions with crossing fibers, producing incorrect FA and tract directions",
            "Susceptibility-induced signal dropout near air-tissue interfaces (sinuses, temporal lobes)",
            "Head motion between diffusion volumes causing inter-volume misalignment",
        ],
        "how_to_avoid_mistakes": [
            "Apply FSL eddy or equivalent for eddy current, motion, and susceptibility correction",
            "Use ≥30 directions for DTI, ≥60 for CSD, and ≥90 for multi-shell models",
            "Use multi-fiber models (CSD, NODDI) in regions known to have crossing fibers",
            "Use reduced FOV or multi-shot EPI near susceptibility-prone regions",
            "Include interspersed b=0 volumes for robust motion and drift correction",
        ],
    },

    "ultrasound": {
        "principle": (
            "Medical ultrasound imaging transmits short pulses of high-frequency "
            "sound waves (1-20 MHz) into tissue and detects the echoes reflected "
            "from acoustic impedance boundaries. The time delay of each echo "
            "determines the reflector depth, and beamforming focuses the "
            "transmitted and received beams to form a 2-D cross-sectional "
            "image. Spatial resolution improves with frequency but penetration "
            "depth decreases."
        ),
        "setup_guide": (
            "A clinical ultrasound system consists of a multi-element "
            "transducer array (linear 7-15 MHz for superficial, curvilinear "
            "2-5 MHz for abdominal, phased array 1-5 MHz for cardiac) "
            "connected to a beamformer and image processor. Modern systems "
            "use 128-192 element arrays with digital beamforming. Apply "
            "acoustic coupling gel between transducer and skin. Adjust gain, "
            "depth, focus, and frequency for the specific examination."
        ),
        "common_algorithms": [
            "Delay-and-sum (DAS) beamforming",
            "Adaptive beamforming (Capon, MVDR) for improved resolution",
            "Synthetic aperture focusing (SAFT)",
            "Plane-wave compounding for ultrafast imaging",
            "Deep-learning beamforming and speckle reduction",
        ],
        "common_mistakes": [
            "Incorrect transducer selection (frequency too high for deep structures or too low for superficial)",
            "Poor acoustic coupling (air gaps) causing signal dropout",
            "Gain set too high, saturating the image and masking pathology",
            "Acoustic shadowing behind highly reflective structures misinterpreted as pathology",
            "Not adjusting focus zone depth to the region of interest",
        ],
        "how_to_avoid_mistakes": [
            "Select transducer frequency appropriate for the imaging depth required",
            "Apply generous coupling gel and maintain constant contact pressure",
            "Adjust TGC (time-gain compensation) curve for uniform brightness with depth",
            "Recognize and account for acoustic artifacts (shadowing, enhancement, reverberation)",
            "Set the transmit focal zone at the depth of the target structure",
        ],
    },

    "doppler_ultrasound": {
        "principle": (
            "Doppler ultrasound measures blood flow velocity by detecting the "
            "frequency shift of echoes reflected from moving red blood cells. "
            "The Doppler equation relates the frequency shift to velocity: "
            "Δf = 2f₀·v·cos(θ)/c, where θ is the beam-flow angle. Color "
            "Doppler maps velocity spatially, spectral Doppler provides "
            "velocity-time waveforms at a sample volume, and power Doppler "
            "shows flow amplitude regardless of direction."
        ),
        "setup_guide": (
            "Use a clinical ultrasound system with Doppler capability. For "
            "vascular studies, use a linear array transducer (5-12 MHz). "
            "Steer the beam to achieve a Doppler angle <60° to the vessel "
            "axis. Set the velocity scale (PRF) to match expected flow speeds "
            "(avoid aliasing). For spectral Doppler, place the sample volume "
            "within the vessel lumen and adjust the gate size. Angle correction "
            "must be applied for accurate velocity measurements."
        ),
        "common_algorithms": [
            "Autocorrelation-based color flow estimation (Kasai algorithm)",
            "FFT spectral analysis for pulsed-wave Doppler",
            "Clutter filtering (wall filtering) to remove tissue motion",
            "Power Doppler (amplitude mode) for slow flow detection",
            "Ultrafast Doppler (plane-wave compounding) for functional ultrasound",
        ],
        "common_mistakes": [
            "Doppler angle >60° causing large velocity measurement errors",
            "Aliasing in color or spectral Doppler from PRF set too low for flow velocity",
            "Wall filter too aggressive, eliminating slow venous flow signals",
            "Blooming artifact in color Doppler from excessive gain",
            "Not correcting for angle in spectral Doppler velocity measurements",
        ],
        "how_to_avoid_mistakes": [
            "Maintain Doppler angle <60°; ideally 30-60° for best accuracy",
            "Increase PRF (velocity scale) until aliasing resolves; or use CW Doppler",
            "Reduce wall filter setting when looking for slow flow (venous, microvascular)",
            "Reduce color Doppler gain until color just fills the vessel without overflow",
            "Always apply angle correction cursor parallel to the vessel wall for spectral Doppler",
        ],
    },

    "elastography": {
        "principle": (
            "Shear-wave elastography measures tissue stiffness by tracking "
            "the propagation speed of shear waves generated by an acoustic "
            "radiation force impulse (ARFI) or external vibration. Shear-wave "
            "speed is proportional to the square root of the shear modulus: "
            "cₛ = √(μ/ρ). Stiffer tissues (fibrosis, tumors) have faster "
            "shear-wave propagation. Results are displayed as quantitative "
            "elasticity maps (in kPa or m/s)."
        ),
        "setup_guide": (
            "Use a clinical ultrasound system with shear-wave elastography "
            "mode (Supersonic Imagine Aixplorer, Siemens ARFI/VTQ, or GE "
            "2D-SWE). The transducer generates a focused push pulse to create "
            "shear waves, then tracks their propagation with ultrafast plane-"
            "wave imaging (up to 10,000 fps). Place the ROI in a region free "
            "of large vessels and interfaces. Patient should hold breath for "
            "liver measurements. Calibrate with an elasticity phantom."
        ),
        "common_algorithms": [
            "Time-to-peak shear-wave arrival estimation",
            "Phase-gradient shear-wave speed inversion",
            "2-D shear-wave elastography mapping (real-time SWE)",
            "Transient elastography (FibroScan 1-D measurement)",
            "Deep-learning elasticity estimation from B-mode + SWE data",
        ],
        "common_mistakes": [
            "Pre-compression by pressing transducer too hard, artifactually increasing stiffness",
            "Measuring in the near-field where push pulse is unreliable",
            "Not having patient hold breath for liver measurements (respiratory motion invalidates SWE)",
            "Placing ROI near large vessels or liver capsule causing boundary artifacts",
            "Not waiting for the measurement to stabilize (IQR/median >30 % indicates unreliable data)",
        ],
        "how_to_avoid_mistakes": [
            "Apply light transducer pressure with coupling gel; avoid compressing tissue",
            "Place measurement ROI at 1.5-2 cm depth in liver; avoid the near-field zone",
            "Instruct patient to suspend breathing calmly during each SWE measurement",
            "Avoid ROI placement near vessels, liver edges, or ribs",
            "Acquire ≥10 valid measurements and check IQR/median <30 % per EFSUMB guidelines",
        ],
    },

    # ── COHERENT ───────────────────────────────────────────────────────────

    "ptychography": {
        "principle": (
            "Ptychography is a scanning coherent diffractive imaging technique "
            "where a coherent beam (visible, X-ray, or electron) illuminates "
            "overlapping regions of the sample. At each scan position, a "
            "far-field diffraction pattern is recorded. The redundancy from "
            "overlapping illumination positions constrains the phase-retrieval "
            "problem, enabling simultaneous recovery of both the complex "
            "sample transmittance and the illumination probe function."
        ),
        "setup_guide": (
            "For X-ray ptychography at a synchrotron: focus the beam to a "
            "defined spot (0.1-1 μm) using a Fresnel zone plate or KB mirrors. "
            "Mount the sample on a precision piezo scanning stage. Place a "
            "photon-counting area detector (Eiger, Pilatus) in the far field "
            "(1-5 m downstream). Scan positions should overlap by 60-70 %. "
            "For visible-light or electron ptychography, adapt the geometry "
            "but maintain the overlap requirement."
        ),
        "common_algorithms": [
            "ePIE (extended Ptychographic Iterative Engine)",
            "Difference Map algorithm",
            "Maximum Likelihood refinement (MLR)",
            "PtychoShelves (modular framework for ptychographic reconstruction)",
            "Deep-learning ptychography (PtychoNN, learned phase retrieval)",
        ],
        "common_mistakes": [
            "Insufficient overlap between adjacent scan positions (need ≥60 %)",
            "Position errors in the scanning stage causing reconstruction artifacts",
            "Partial coherence effects not modeled, degrading recovered phase",
            "Vibration or drift during the scan corrupting the diffraction data",
            "Detector saturation at the central beam stop region",
        ],
        "how_to_avoid_mistakes": [
            "Maintain ≥65 % overlap; include position correction in the reconstruction algorithm",
            "Use position refinement (annealing) as part of the ptychographic reconstruction",
            "Include mixed-state (multi-mode) probe to model partial coherence",
            "Use interferometric position feedback and short dwell times per point",
            "Use a semi-transparent beam stop or high-dynamic-range detector modes",
        ],
    },

    "holography": {
        "principle": (
            "Digital holographic microscopy records the interference pattern "
            "(hologram) between a reference wave and the wave scattered by "
            "the sample. The complex field (amplitude and phase) is recovered "
            "by numerical propagation of the recorded hologram to the object "
            "plane. Phase imaging reveals optical path length changes caused "
            "by refractive index or thickness variations, providing "
            "quantitative phase contrast without staining."
        ),
        "setup_guide": (
            "Build an off-axis Mach-Zehnder interferometer: split a coherent "
            "source (He-Ne laser, 633 nm, or laser diode) into object and "
            "reference beams. The object beam passes through the sample via "
            "a microscope objective. The reference beam tilts at a small angle "
            "(off-axis) to create carrier fringes. Both beams interfere on a "
            "CMOS camera. The carrier frequency must be high enough to "
            "separate the twin image in Fourier space. Vibration isolation is "
            "essential."
        ),
        "common_algorithms": [
            "Fourier filtering (off-axis hologram: spatial filtering of +1 order)",
            "Angular spectrum propagation method",
            "Phase unwrapping (Goldstein, quality-guided, or least-squares)",
            "Numerical autofocusing (Tamura coefficient, Brenner gradient)",
            "Deep-learning phase retrieval (PhaseNet, holographic reconstruction CNN)",
        ],
        "common_mistakes": [
            "Vibration causing fringe instability and phase noise",
            "Twin image and DC term not properly separated in on-axis holography",
            "Phase wrapping artifacts not resolved in thick or rapidly varying samples",
            "Coherence noise (speckle) from high temporal coherence of the laser source",
            "Incorrect propagation distance causing defocused reconstruction",
        ],
        "how_to_avoid_mistakes": [
            "Use an optical table with active vibration isolation; enclose the setup",
            "Use off-axis geometry with sufficient carrier frequency for clean Fourier separation",
            "Apply robust phase unwrapping algorithms; use multi-wavelength for large OPD",
            "Use a low-coherence source (LED or SLD) for speckle reduction in off-axis DHM",
            "Implement numerical autofocusing or calibrate propagation distance precisely",
        ],
    },

    "phase_retrieval": {
        "principle": (
            "Coherent Diffractive Imaging (CDI) records the far-field "
            "diffraction pattern of an isolated object illuminated by a "
            "coherent beam. Only intensity (not phase) is measured on the "
            "detector. Phase retrieval algorithms iteratively recover the "
            "lost phase by enforcing known constraints: the measured Fourier "
            "modulus and the finite support of the object in real space. "
            "CDI achieves diffraction-limited resolution without any imaging "
            "lens."
        ),
        "setup_guide": (
            "Illuminate an isolated object (nanocrystal, cell, virus particle) "
            "with a coherent, quasi-plane-wave beam (X-ray from synchrotron or "
            "XFEL, or visible laser). Record the continuous diffraction pattern "
            "on a pixel detector (Eiger, Jungfrau for X-ray; CMOS for visible) "
            "placed far enough for adequate oversampling (oversampling ratio ≥ 2 "
            "in each dimension). Remove the direct beam with a beam stop. "
            "Ensure the object is isolated (no other scatterers in the beam)."
        ),
        "common_algorithms": [
            "Hybrid Input-Output (HIO) algorithm",
            "Error Reduction (ER) algorithm",
            "Shrink-Wrap (adaptive support HIO)",
            "Relaxed Averaged Alternating Reflections (RAAR)",
            "Deep-learning phase retrieval (PhaseDNN, learned proximal operator)",
        ],
        "common_mistakes": [
            "Insufficient oversampling (detector pixels too coarse or too close to sample)",
            "Object not truly isolated, violating the support constraint",
            "Missing low-frequency data due to beam stop causing artifacts",
            "Stagnation in reconstruction (trapped in local minimum) without proper initialization",
            "Ignoring partial coherence effects from finite source size or bandwidth",
        ],
        "how_to_avoid_mistakes": [
            "Ensure oversampling ratio ≥ 2× (linear) in each dimension; use a large detector",
            "Isolate the object on a thin membrane or in free space; verify no neighbor scattering",
            "Use low-frequency intensity constraints or a semi-transparent beam stop",
            "Run multiple random starts and use HIO-ER hybrid strategies to escape local minima",
            "Model partial coherence in the forward model or select sufficiently coherent beams",
        ],
    },

    # ── NEURAL RENDERING ───────────────────────────────────────────────────

    "nerf": {
        "principle": (
            "Neural Radiance Fields (NeRF) represent a 3-D scene as a "
            "continuous volumetric function F(x,y,z,θ,φ) → (RGB, σ) "
            "parameterized by a multi-layer perceptron (MLP). The network "
            "maps 3-D position and viewing direction to color and volume "
            "density. Novel views are synthesized by differentiable volume "
            "rendering along camera rays, and the network is trained by "
            "minimizing photometric loss against a set of posed 2-D images."
        ),
        "setup_guide": (
            "Capture 50-200 images of a scene from diverse viewpoints using "
            "a calibrated camera (known intrinsics) or estimate camera poses "
            "with COLMAP structure-from-motion. Images should cover the scene "
            "uniformly. Train a NeRF MLP (typically 8 layers, 256 units, with "
            "positional encoding of input coordinates) on a GPU (≥12 GB VRAM). "
            "Training takes 12-48 hours on a single V100. Use mip-NeRF, "
            "Instant-NGP, or TensoRF for faster convergence."
        ),
        "common_algorithms": [
            "Vanilla NeRF (MLP + positional encoding)",
            "Instant-NGP (multi-resolution hash encoding, minutes training)",
            "mip-NeRF (anti-aliased cone tracing)",
            "Nerfacto (nerfstudio default combining multiple improvements)",
            "TensoRF (tensor factorization for compact radiance fields)",
        ],
        "common_mistakes": [
            "Insufficient camera pose accuracy (SfM failure) causing blurry results",
            "Too few input views or views clustered in a narrow angular range",
            "Training only at one scale without mip-NeRF, causing aliasing at novel distances",
            "Floater artifacts in empty space from insufficient regularization",
            "Very slow training and rendering with vanilla NeRF (hours to train, seconds per frame)",
        ],
        "how_to_avoid_mistakes": [
            "Verify COLMAP pose estimation quality; add more images if registration fails",
            "Capture views uniformly around the scene; include close-up and distant views",
            "Use mip-NeRF or multi-scale training for scale consistency",
            "Add distortion loss or density regularization to eliminate floater artifacts",
            "Use Instant-NGP or 3D Gaussian Splatting for real-time rendering requirements",
        ],
    },

    "gaussian_splatting": {
        "principle": (
            "3-D Gaussian Splatting represents a scene as a set of anisotropic "
            "3-D Gaussians, each with position, covariance, opacity, and "
            "spherical harmonics color coefficients. Novel views are rendered "
            "by projecting (splatting) these Gaussians onto the image plane "
            "and alpha-compositing them in depth order. Unlike NeRF, rendering "
            "is rasterization-based and achieves real-time frame rates (≥100 fps) "
            "with high visual quality."
        ),
        "setup_guide": (
            "Start with the same multi-view image dataset as NeRF (50-200 posed "
            "images via COLMAP). Initialize 3-D Gaussians from the SfM point "
            "cloud. Train by differentiable rasterization: project Gaussians to "
            "each training view, compute photometric loss (L1 + SSIM), and "
            "optimize positions, covariances, colors, and opacities via Adam. "
            "Adaptive densification (splitting/cloning Gaussians) and pruning "
            "runs periodically during training. Training takes ~15-30 minutes "
            "on a modern GPU."
        ),
        "common_algorithms": [
            "3D Gaussian Splatting (original, Kerbl et al. 2023)",
            "Mip-Splatting (anti-aliased multi-scale Gaussian splatting)",
            "SuGaR (Surface-Aligned Gaussian Splatting for mesh extraction)",
            "Dynamic 3D Gaussians (for dynamic scenes / video)",
            "Compact-3DGS (compressed Gaussian representations)",
        ],
        "common_mistakes": [
            "Insufficient initial SfM points causing sparse reconstruction",
            "Too few training views creating holes or floater artifacts in novel views",
            "Excessive Gaussian count (millions) consuming too much GPU memory",
            "Not using adaptive densification, leaving under-reconstructed regions",
            "Ignoring exposure variation between training images",
        ],
        "how_to_avoid_mistakes": [
            "Use dense SfM initialization; increase COLMAP matching thoroughness if sparse",
            "Capture more views, especially in regions that are under-represented",
            "Apply periodic pruning of low-opacity Gaussians to control memory",
            "Enable adaptive densification and set proper gradient thresholds for splitting",
            "Apply per-image exposure compensation or normalize images before training",
        ],
    },

    # ── COMPUTATIONAL ──────────────────────────────────────────────────────

    "panorama": {
        "principle": (
            "Panoramic multi-focus fusion captures multiple images of the same "
            "wide scene at different focal distances and combines them to "
            "produce a single all-in-focus panorama with extended depth of "
            "field. Image stitching aligns overlapping frames using feature "
            "matching and homography estimation, while focus fusion selects "
            "the sharpest pixels from each focal plane."
        ),
        "setup_guide": (
            "Mount a camera on a motorized panoramic head (nodal point rotation). "
            "For each pan/tilt position, capture a focus stack (3-10 images at "
            "different focus distances). Use a medium-aperture setting (f/5.6-"
            "f/8) for each frame. Stitch overlapping views (30 % horizontal "
            "overlap) and fuse focus stacks per view tile. Calibrate the "
            "panoramic head to rotate around the lens entrance pupil to "
            "minimize parallax."
        ),
        "common_algorithms": [
            "Laplacian pyramid focus fusion (weighted blending by local contrast)",
            "SIFT/SURF feature matching + RANSAC homography estimation",
            "Multi-band blending (Burt-Adelson) for seamless stitching",
            "Exposure fusion (Mertens et al.) for HDR panoramas",
            "Deep-learning focus stacking (DFDF, DeepFocus)",
        ],
        "common_mistakes": [
            "Parallax errors from rotation not centered on the lens entrance pupil",
            "Ghosting from moving objects between sequential captures",
            "Color inconsistency between overlapping tiles due to auto-exposure variation",
            "Incomplete focus coverage leaving blurry regions in the final panorama",
            "Stitching artifacts at seam lines visible in the final output",
        ],
        "how_to_avoid_mistakes": [
            "Use a calibrated panoramic head; verify no-parallax point for the specific lens",
            "Mask out or blend moving objects; capture quickly or use simultaneous multi-camera rigs",
            "Lock exposure, white balance, and focus (manual mode) across all tiles",
            "Plan focus distances to cover the entire depth range of the scene",
            "Use multi-band blending and choose seam lines in textureless regions",
        ],
    },

    "light_field": {
        "principle": (
            "Light-field imaging captures both the spatial position and "
            "direction of light rays in a scene, recording a 4-D light field "
            "L(u,v,s,t) where (u,v) parameterize the aperture and (s,t) "
            "parameterize the spatial position. This enables computational "
            "refocusing, depth estimation, and novel viewpoint synthesis from "
            "a single capture. A microlens array placed before the sensor "
            "trades spatial resolution for angular resolution."
        ),
        "setup_guide": (
            "Place a microlens array (MLA) at the sensor plane of a camera, "
            "one focal length in front of the image sensor. Each microlens "
            "captures the angular distribution of light from a corresponding "
            "spatial position (Lytro-style plenoptic camera). Alternative: "
            "use a camera array (e.g., 4×4 or 8×8 synchronized cameras) for "
            "higher angular and spatial resolution. Calibrate MLA alignment, "
            "microlens pitch, and main lens parameters."
        ),
        "common_algorithms": [
            "Shift-and-sum refocusing (synthetic aperture)",
            "Depth estimation from disparity between sub-aperture images",
            "Fourier slice theorem for light-field refocusing",
            "Light-field super-resolution (recovering spatial resolution lost to MLA)",
            "Deep-learning view synthesis (light field reconstruction from sparse views)",
        ],
        "common_mistakes": [
            "Microlens array misaligned with sensor pixels, causing vignetting and crosstalk",
            "Insufficient angular samples for accurate depth estimation in textureless regions",
            "Not calibrating MLA-to-sensor alignment, producing decoding artifacts",
            "Confusing spatial and angular resolution trade-off limits of the plenoptic design",
            "Ignoring diffraction effects at the microlens apertures",
        ],
        "how_to_avoid_mistakes": [
            "Precisely align MLA to sensor with sub-pixel accuracy; use calibration targets",
            "Increase camera array density or use coded-aperture techniques for more angular samples",
            "Calibrate using a white image and point-source images for precise microlens grid mapping",
            "Design the system with the desired spatial-angular trade-off explicitly computed",
            "Use microlens diameters larger than the diffraction limit (> 10× wavelength)",
        ],
    },

    "integral": {
        "principle": (
            "Integral photography (also known as integral imaging) uses a 2-D "
            "array of elemental lenses to capture multi-perspective views of a "
            "3-D scene simultaneously. Each elemental lens records a small "
            "perspective image, and the full set encodes the 4-D light field. "
            "Computational reconstruction produces 3-D images that can be "
            "viewed from different angles or refocused without glasses."
        ),
        "setup_guide": (
            "Place a 2-D microlens or lenslet array (pitch 0.5-1 mm, ~50-200 "
            "elements per side) at one focal length from a high-resolution "
            "sensor. Each lenslet forms a separate elemental image. For "
            "display: show the integral image on a high-resolution display "
            "with a matched output lenslet array. Calibrate lenslet grid "
            "alignment, individual lens focal lengths, and vignetting "
            "correction. Use telecentric imaging for uniform magnification."
        ),
        "common_algorithms": [
            "Computational refocusing via pixel rearrangement and summation",
            "Depth estimation from elemental image disparity analysis",
            "3-D scene reconstruction from integral images",
            "Super-resolution integral imaging (combining multiple shifted captures)",
            "Deep-learning integral image reconstruction and view synthesis",
        ],
        "common_mistakes": [
            "Lenslet array not properly aligned with the sensor pixel grid",
            "Insufficient number of elemental lenses for the desired depth range",
            "Crosstalk between adjacent elemental images due to lens aberrations",
            "Not correcting for vignetting variations across the lenslet array",
            "Pseudoscopic (depth-reversed) images if reconstruction is not properly handled",
        ],
        "how_to_avoid_mistakes": [
            "Align lenslet array to sensor with precision jigs and verify with calibration patterns",
            "Design lenslet pitch and focal length for the required depth-of-field",
            "Use high-quality molded lenslets and baffles to minimize crosstalk",
            "Apply per-lenslet calibration including vignetting and distortion correction",
            "Use computational depth inversion to correct pseudoscopic effects",
        ],
    },

    # ── CLINICAL OPTICS ────────────────────────────────────────────────────

    "oct": {
        "principle": (
            "Optical Coherence Tomography uses low-coherence interferometry "
            "to produce cross-sectional images of tissue microstructure. A "
            "broadband light source (superluminescent diode, ~840 nm or ~1310 nm) "
            "is split between sample and reference arms. Interference occurs "
            "only when the path lengths match within the coherence length "
            "(~5-10 μm), providing axial resolution. Spectral-domain OCT "
            "records the spectral interferogram and uses FFT for fast "
            "depth-resolved imaging."
        ),
        "setup_guide": (
            "Build or acquire a spectral-domain OCT system: broadband SLD "
            "source (center 840 nm, 50 nm bandwidth for retinal; 1310 nm for "
            "dermal/cardiac), fiber-based Michelson interferometer, galvo "
            "scanner for lateral scanning, and a spectrometer with line camera "
            "(2048-4096 pixels) for spectral detection. Calibrate wavelength-"
            "to-wavenumber mapping, dispersion compensation, and reference "
            "arm delay. For swept-source OCT, use a frequency-swept laser "
            "(100-400 kHz sweep rate) and balanced detector."
        ),
        "common_algorithms": [
            "FFT-based spectral-domain OCT reconstruction (spectral interferogram → A-scan)",
            "Dispersion compensation (numerical or hardware)",
            "Speckle reduction (spatial/angular compounding, or deep-learning)",
            "Segmentation of retinal layers (graph-based, U-Net, or transformer models)",
            "OCT Angiography (OCTA) via decorrelation or phase-variance of repeated B-scans",
        ],
        "common_mistakes": [
            "Dispersion mismatch between sample and reference arms degrading axial resolution",
            "Mirror image artifact from complex conjugate ambiguity in SD-OCT",
            "Sensitivity roll-off at deeper imaging depths not compensated",
            "Motion artifacts in 3-D OCT volumes (eye motion for ophthalmic OCT)",
            "Incorrect refractive index assumption for depth scale calibration",
        ],
        "how_to_avoid_mistakes": [
            "Match fiber lengths and add numerical dispersion compensation in reconstruction",
            "Place the zero-delay near the sample surface; use full-range OCT if needed",
            "Use swept-source OCT for reduced roll-off; optimize spectrometer for uniform sensitivity",
            "Apply eye-tracking or motion-correction algorithms; average repeated B-scans",
            "Calibrate depth scale with a known-thickness reference standard",
        ],
    },

    "octa": {
        "principle": (
            "OCT Angiography detects blood flow non-invasively by comparing "
            "repeated OCT B-scans at the same location. Moving red blood cells "
            "cause temporal fluctuations in the OCT signal (amplitude and/or "
            "phase), while static tissue remains constant. Decorrelation, "
            "variance, or differential analysis between repeated scans produces "
            "a motion-contrast image revealing the vasculature without the "
            "need for injectable contrast agents."
        ),
        "setup_guide": (
            "Use a high-speed OCT system (≥70 kHz A-scan rate, swept-source "
            "preferred) capable of repeated B-scans at the same location. "
            "Acquire 2-4 repeated B-scans at each position with inter-scan "
            "time of 3-10 ms. An eye-tracking system is essential for "
            "ophthalmic OCTA to correct microsaccades. Process with split-"
            "spectrum amplitude-decorrelation (SSADA), optical microangiography "
            "(OMAG), or phase-variance algorithms."
        ),
        "common_algorithms": [
            "SSADA (Split-Spectrum Amplitude-Decorrelation Angiography)",
            "OMAG (Optical Micro-Angiography, complex signal differential)",
            "Phase-variance OCTA",
            "Deep-learning OCTA denoising and vessel segmentation",
            "Projection artifact removal algorithms",
        ],
        "common_mistakes": [
            "Bulk tissue motion producing decorrelation artifacts (false flow signals)",
            "Projection artifacts where superficial vessel shadows appear in deeper layers",
            "Shadow artifacts beneath large vessels causing false flow voids",
            "Insufficient inter-scan interval for detecting slow capillary flow",
            "Motion artifacts from blinks or microsaccades corrupting OCTA volumes",
        ],
        "how_to_avoid_mistakes": [
            "Apply bulk motion correction (axial and lateral registration) before decorrelation analysis",
            "Use projection artifact removal algorithms (slab subtraction or OMAG-based)",
            "Increase number of repeated B-scans to improve SNR and reduce shadow impact",
            "Optimize inter-scan time: shorter for fast flow, longer for slow capillary flow",
            "Use active eye tracking and discard frames with large motion; average multiple volumes",
        ],
    },

    "fundus": {
        "principle": (
            "A fundus camera images the posterior segment of the eye (retina, "
            "optic disc, macula, vasculature) by illuminating the retina "
            "through the pupil and capturing the reflected/backscattered light. "
            "The optical path is designed to separate illumination and "
            "observation through different portions of the pupil to avoid "
            "corneal reflections. Standard fundus imaging provides 30-50° "
            "field-of-view color photographs of the retina."
        ),
        "setup_guide": (
            "Use a dedicated fundus camera (e.g., Topcon TRC-NW400, Canon "
            "CR-2 AF) or a scanning laser ophthalmoscope (Optos for widefield). "
            "Dilate the patient's pupil (tropicamide 1%) for standard fundus "
            "photography. Align the camera to center on the macula or optic "
            "disc. Set appropriate flash intensity and focus. Capture color "
            "and red-free (green channel) images. For fluorescein angiography, "
            "inject sodium fluorescein IV and capture timed image series "
            "with excitation/barrier filters."
        ),
        "common_algorithms": [
            "Image quality assessment and auto-focus/auto-exposure",
            "Vessel segmentation (U-Net, DeepVessel)",
            "Optic disc and cup segmentation for glaucoma screening",
            "Diabetic retinopathy grading (deep-learning classifiers)",
            "Multi-frame averaging and super-resolution for fundus images",
        ],
        "common_mistakes": [
            "Insufficient pupil dilation causing vignetting at the field edges",
            "Corneal reflections (flare) obscuring the central retinal image",
            "Image out of focus due to refractive error not compensated",
            "Eyelash or eyelid obstruction in the image",
            "Uneven illumination across the retinal image",
        ],
        "how_to_avoid_mistakes": [
            "Ensure adequate mydriasis (>5 mm pupil diameter) before imaging",
            "Align the camera carefully to separate illumination and observation through different pupil zones",
            "Use auto-focus and compensate for patient refractive error in the camera optics",
            "Ask patients to open eyes wide; use a fixation target for gaze direction",
            "Verify uniform illumination before capture; adjust camera alignment if uneven",
        ],
    },

    "endoscopy": {
        "principle": (
            "Fiber-bundle endoscopy transmits an image through a flexible "
            "coherent fiber bundle (10,000-100,000 individual fiber cores) to "
            "visualize internal body cavities. Each fiber core acts as a single "
            "pixel, transmitting light from the distal end to the proximal end "
            "where a camera captures the image. The hexagonal fiber packing "
            "imposes a fixed pixelation pattern (comb/honeycomb structure) "
            "on the image."
        ),
        "setup_guide": (
            "A medical endoscope has a flexible insertion tube containing the "
            "coherent fiber bundle (or a distal CMOS chip for video endoscopes), "
            "illumination fibers, working channels, and air/water channels. "
            "Light source: LED or Xenon lamp transmitted through illumination "
            "fibers. For fiber-bundle type: attach a high-resolution camera "
            "and relay lens at the proximal end. Calibrate fiber core positions "
            "and individual fiber transmission for computational image "
            "improvement."
        ),
        "common_algorithms": [
            "Fiber core mapping and interpolation (honeycomb artifact removal)",
            "Deep-learning super-resolution for fiber-bundle images",
            "Structure-from-motion for endoscopic 3-D reconstruction",
            "Defogging / dehazing for underwater or smoke-obscured endoscopy",
            "Real-time mosaicking for extended field-of-view endoscopy",
        ],
        "common_mistakes": [
            "Honeycomb pattern artifact from fiber core spacing not removed",
            "Broken fibers (dark spots) accumulating over time and degrading image quality",
            "Specular reflections (glare) from wet tissue surfaces saturating the image",
            "Insufficient illumination causing noisy images in deep body cavities",
            "Image distortion from fiber bundle bending not corrected",
        ],
        "how_to_avoid_mistakes": [
            "Apply fiber core interpolation or deep-learning super-resolution in post-processing",
            "Replace fiber bundles when broken fiber percentage exceeds acceptable threshold",
            "Use polarization filtering or computational specular removal algorithms",
            "Use bright LED sources and adjust exposure/gain for adequate signal",
            "Calibrate and correct for bending-dependent distortion using test patterns",
        ],
    },

    # ── ELECTRON MICROSCOPY ────────────────────────────────────────────────

    "sem": {
        "principle": (
            "Scanning Electron Microscopy rasters a focused electron beam "
            "(0.1-30 keV) across the sample surface. Secondary electrons (SE) "
            "emitted from the top few nanometers provide topographic contrast, "
            "while backscattered electrons (BSE) from deeper interactions "
            "reveal compositional contrast (higher Z → more BSE). The image "
            "is formed point-by-point, with resolution down to 1-5 nm "
            "determined by the probe size."
        ),
        "setup_guide": (
            "Operate a field-emission SEM (FEG-SEM, e.g., Zeiss GeminiSEM, "
            "JEOL JSM-7800F) under high vacuum (< 10⁻⁴ Pa). Mount samples on "
            "conductive stubs with carbon tape or silver paint. Non-conductive "
            "samples must be sputter-coated (5-10 nm Au/Pd or C) to prevent "
            "charging. Set accelerating voltage (1-5 kV for surface detail, "
            "10-20 kV for BSE compositional contrast). Select appropriate "
            "detectors (Everhart-Thornley for SE, solid-state for BSE). "
            "Align the column and perform astigmatism correction."
        ),
        "common_algorithms": [
            "Noise reduction by frame averaging or Kalman filtering",
            "Charging artifact compensation (dynamic focus, low-kV imaging)",
            "3-D surface reconstruction from stereo-pair SEM images",
            "Deep-learning SEM denoising (for low-dose or fast-scan images)",
            "Automated particle analysis and morphometry",
        ],
        "common_mistakes": [
            "Sample charging causing bright streaks and image distortion",
            "Astigmatism not corrected, producing elongated features",
            "Excessive beam current damaging or contaminating delicate samples",
            "Carbon contamination from residual hydrocarbons in the chamber",
            "Wrong working distance causing suboptimal resolution or depth of field",
        ],
        "how_to_avoid_mistakes": [
            "Coat non-conductive samples or use low-vacuum/variable-pressure mode",
            "Correct astigmatism carefully using the wobbler on a recognizable feature",
            "Use the minimum beam current needed; work at low kV for beam-sensitive samples",
            "Plasma-clean the chamber and samples; use a cold trap to reduce contamination",
            "Optimize working distance for the specific detector and resolution requirement",
        ],
    },

    "tem": {
        "principle": (
            "Transmission Electron Microscopy transmits a high-energy electron "
            "beam (80-300 keV) through an ultra-thin specimen (<100 nm). "
            "Electrons interact with the sample via elastic scattering "
            "(diffraction contrast, phase contrast) and inelastic scattering "
            "(energy loss). The transmitted beam is magnified by electromagnetic "
            "lenses to form an image with atomic-level resolution "
            "(0.05-0.2 nm in aberration-corrected TEMs)."
        ),
        "setup_guide": (
            "Operate a TEM (e.g., JEOL JEM-2100, Thermo Fisher Talos/Titan) "
            "under high vacuum (< 10⁻⁵ Pa). Prepare ultra-thin specimens using "
            "ultramicrotomy (biological), focused ion beam (FIB) milling "
            "(materials), or electropolishing (metals). Load samples on 3 mm "
            "TEM grids (Cu or Mo). Align the beam, correct condenser and "
            "objective astigmatism, and set appropriate defocus for phase "
            "contrast imaging. Use direct-electron detectors for highest DQE."
        ),
        "common_algorithms": [
            "CTF correction (Contrast Transfer Function for phase contrast imaging)",
            "Single-particle analysis (cryo-EM: classification, 3-D reconstruction)",
            "Selected-area electron diffraction (SAED) pattern analysis",
            "HRTEM image simulation (multislice or Bloch wave)",
            "Deep-learning denoising for low-dose cryo-EM (Topaz, Warp, cryoSPARC)",
        ],
        "common_mistakes": [
            "Specimen too thick, causing multiple scattering and loss of interpretable contrast",
            "Beam damage to organic or beam-sensitive materials from excessive electron dose",
            "Astigmatism and coma not corrected, degrading high-resolution images",
            "Not accounting for CTF effects when interpreting HRTEM images",
            "Contamination building up on the specimen under the beam (hydrocarbon deposition)",
        ],
        "how_to_avoid_mistakes": [
            "Prepare specimens to <50 nm thickness; verify with EELS log-ratio thickness mapping",
            "Use low-dose protocols and cryo-cooling for beam-sensitive specimens",
            "Perform careful alignment including Zemlin tableau for Cs-corrected instruments",
            "Simulate TEM images with known structure and compare; always correct CTF in analysis",
            "Plasma-clean grids and specimens before loading; use a cryo-shield during imaging",
        ],
    },

    "stem": {
        "principle": (
            "Scanning TEM focuses the electron beam to a fine probe (0.05-1 nm) "
            "and scans it across the specimen. Multiple detectors collect "
            "signals simultaneously: bright-field (BF), annular dark-field "
            "(ADF), and high-angle annular dark-field (HAADF). HAADF-STEM "
            "provides Z-contrast imaging where intensity scales approximately "
            "as Z^1.7, enabling direct interpretation of atomic columns by "
            "atomic number."
        ),
        "setup_guide": (
            "Use an aberration-corrected STEM (probe-corrected, e.g., Thermo "
            "Fisher Titan Themis or JEOL ARM300F). Align the probe-corrector "
            "to minimize C₃ and C₅ aberrations, achieving sub-Ångström probe "
            "size. Adjust camera length for HAADF inner angle (typically "
            "50-80 mrad for Z-contrast). Prepare atomically thin specimens "
            "by FIB or mechanical exfoliation. Use drift-corrected frame "
            "integration for high-quality atomic-resolution images."
        ),
        "common_algorithms": [
            "Atom column detection and quantification (peak finding, Gaussian fitting)",
            "Strain mapping via geometric phase analysis (GPA) or peak-pair analysis",
            "Multi-frame averaging with rigid/non-rigid registration for noise reduction",
            "HAADF simulation (frozen-phonon multislice) for quantitative comparison",
            "Deep-learning STEM image denoising and super-resolution",
        ],
        "common_mistakes": [
            "Probe aberrations not fully corrected, producing probe tails and delocalization",
            "Scan distortion (flyback, drift) causing apparent lattice strain artifacts",
            "Sample mistilt from zone axis, reducing contrast of atomic columns",
            "Amorphous surface layers (from FIB damage) obscuring atomic contrast",
            "Electron channeling effects complicating quantitative HAADF interpretation",
        ],
        "how_to_avoid_mistakes": [
            "Tune corrector regularly using Zemlin tableau or Ronchigram analysis",
            "Apply scan distortion correction using known lattice spacings as reference",
            "Tilt to exact zone axis using CBED pattern or Ronchigram fine alignment",
            "Use low-kV FIB final polishing or Ar-ion milling to minimize surface damage",
            "Simulate HAADF images with the exact specimen thickness for quantitative analysis",
        ],
    },

    "electron_tomography": {
        "principle": (
            "Electron tomography reconstructs a 3-D volume from a tilt series "
            "of 2-D TEM or STEM projections acquired at different specimen "
            "tilts (typically ±60-70°). The Radon transform (or its "
            "generalization) relates the projections to the 3-D structure. "
            "The limited tilt range causes a 'missing wedge' artifact — "
            "elongation in the beam direction — which must be addressed by "
            "regularization or dual-axis acquisition."
        ),
        "setup_guide": (
            "Use a TEM/STEM with a high-tilt specimen holder (±70-80°). "
            "Acquire images at tilt increments of 1-2° across the full range. "
            "For STEM tomography, HAADF signal provides monotonic contrast "
            "(no CTF complications). Include gold nanoparticles as fiducial "
            "markers for alignment. Automated acquisition software (SerialEM, "
            "Tomography by Thermo Fisher) controls stage tilt, focus tracking, "
            "and image acquisition."
        ),
        "common_algorithms": [
            "Weighted back-projection (WBP)",
            "SIRT / SART (Simultaneous Iterative Reconstruction Techniques)",
            "GENFIRE (GENeralized Fourier Iterative REconstruction)",
            "Compressed sensing tomography for missing-wedge artifact reduction",
            "Deep-learning tomographic reconstruction (TomoGAN, DeepRecon)",
        ],
        "common_mistakes": [
            "Poor tilt-series alignment causing blurring in the reconstruction",
            "Missing wedge artifacts not addressed, distorting features along the beam axis",
            "Specimen drift or deformation during the tilt series (especially for biological specimens)",
            "Dose damage accumulating through the tilt series degrading later images",
            "Inaccurate tilt angles due to stage mechanical backlash",
        ],
        "how_to_avoid_mistakes": [
            "Align tilt series carefully using fiducial markers; refine with cross-correlation",
            "Use dual-axis tomography or compressed-sensing reconstruction to fill the missing wedge",
            "Apply autofocus and drift tracking at each tilt; use cryo-conditions for biology",
            "Distribute dose evenly; start at high tilts where damage impact is greatest",
            "Calibrate stage tilt angle accuracy; use Saxton scheme (non-linear tilt increments)",
        ],
    },

    "electron_diffraction": {
        "principle": (
            "4D-STEM electron diffraction scans a convergent electron beam "
            "across the specimen and records a full 2-D diffraction pattern "
            "(convergent beam electron diffraction, CBED) at each scan "
            "position. The resulting 4-D dataset (2-D scan × 2-D diffraction) "
            "enables mapping of crystal structure, orientation, strain, "
            "electric fields, and charge density with nanometer spatial "
            "resolution."
        ),
        "setup_guide": (
            "Use a STEM equipped with a fast pixelated detector (Medipix3, "
            "EMPAD, or Dectris ARINA) capable of recording diffraction patterns "
            "at >1000 fps. Set a small convergence semi-angle (1-5 mrad) for "
            "nanobeam diffraction or large (20-30 mrad) for CBED. The scan "
            "step should be comparable to the probe size. Data volumes are "
            "large (tens of GB per scan), requiring efficient data pipeline "
            "and storage."
        ),
        "common_algorithms": [
            "Virtual detector imaging (synthesized BF, DF, iDPC from 4D data)",
            "Center-of-mass (COM) analysis for electric field mapping",
            "Ptychographic reconstruction from 4D-STEM data",
            "Orientation mapping (template matching against simulated patterns)",
            "Strain mapping via disk position analysis",
        ],
        "common_mistakes": [
            "Detector dynamic range insufficient for simultaneous central beam and weak diffraction",
            "Scan step too large relative to probe size, under-sampling the specimen",
            "Not accounting for specimen thickness variation in diffraction pattern interpretation",
            "Excessive electron dose for beam-sensitive materials (organics, 2D materials)",
            "Misindexing diffraction patterns due to double diffraction or overlapping grains",
        ],
        "how_to_avoid_mistakes": [
            "Use counting-mode detectors (Medipix) with high dynamic range or electron counting",
            "Match scan step to probe size for complete spatial sampling",
            "Simulate diffraction patterns at the measured thickness for accurate interpretation",
            "Use low-dose 4D-STEM protocols with fast detectors to minimize beam damage",
            "Carefully index patterns considering multiple scattering; compare with simulations",
        ],
    },

    "ebsd": {
        "principle": (
            "Electron Backscatter Diffraction (EBSD) maps the crystallographic "
            "orientation of polycrystalline materials at each surface point. "
            "A focused electron beam (15-30 keV) strikes a tilted (70°) "
            "polished specimen, generating backscattered electrons that form "
            "Kikuchi diffraction patterns on a phosphor screen/CMOS camera. "
            "Automated pattern indexing determines the crystal orientation at "
            "each point with ~0.5° angular resolution."
        ),
        "setup_guide": (
            "Install an EBSD detector (phosphor screen + CCD/CMOS camera, "
            "e.g., Oxford Instruments Symmetry, EDAX Velocity) in an SEM "
            "chamber. Tilt the specimen to 70° toward the detector. Polish "
            "the sample surface to remove any deformation layer (final step: "
            "colloidal silica or ion milling). Set accelerating voltage "
            "15-30 kV, high probe current (1-20 nA). Map with step sizes "
            "of 50 nm to 5 μm depending on grain size."
        ),
        "common_algorithms": [
            "Hough transform band detection for Kikuchi pattern indexing",
            "Dictionary indexing (template matching against simulated patterns)",
            "Spherical indexing (GPU-accelerated orientation determination)",
            "Neighbor pattern averaging and reindexing (NPAR) for noisy patterns",
            "Deep-learning EBSD pattern indexing (faster and more robust than Hough)",
        ],
        "common_mistakes": [
            "Poor surface preparation leaving a deformed layer that degrades pattern quality",
            "Camera settings (gain, exposure) not optimized, producing noisy or saturated patterns",
            "Step size too large relative to the grain size, missing small grains or twin boundaries",
            "Incorrect crystal structure or phase files used for indexing",
            "Drift during long-duration EBSD maps distorting the scanned area",
        ],
        "how_to_avoid_mistakes": [
            "Use final polishing with colloidal silica (OPS) or broad Ar-ion milling",
            "Optimize camera parameters with a reference crystal before mapping",
            "Set step size ≤ 1/10 of the smallest grain dimension of interest",
            "Verify crystal structure and lattice parameters in the phase file before indexing",
            "Use beam shift or stage drift correction for maps longer than ~30 minutes",
        ],
    },

    "eels": {
        "principle": (
            "Electron Energy Loss Spectroscopy measures the energy lost by "
            "transmitted electrons due to inelastic interactions with the "
            "specimen. The energy-loss spectrum contains characteristic edges "
            "corresponding to inner-shell ionization of specific elements, "
            "enabling elemental mapping with atomic spatial resolution. "
            "Near-edge fine structure (ELNES) reveals chemical bonding, and "
            "low-loss features probe band structure and optical properties."
        ),
        "setup_guide": (
            "Attach a post-column energy filter (Gatan GIF Quantum/Continuum) "
            "to a TEM/STEM. For STEM-EELS spectrum imaging: scan the probe "
            "and record a full energy-loss spectrum (0-2000 eV range) at each "
            "pixel. Use a monochromated source (ΔE < 0.3 eV) for near-edge "
            "fine structure studies. Energy dispersion is typically 0.1-0.5 "
            "eV/channel. Acquire both core-loss edges (elemental maps) and "
            "low-loss region (thickness mapping, optical properties)."
        ),
        "common_algorithms": [
            "Background subtraction (power-law fitting before edge onset)",
            "Multiple linear least-squares (MLLS) fitting for overlapping edges",
            "Principal component analysis (PCA) for denoising spectrum images",
            "Kramers-Kronig analysis for optical constants from low-loss EELS",
            "Deep-learning EELS denoising and quantification",
        ],
        "common_mistakes": [
            "Specimen too thick causing plural scattering that distorts edge shapes",
            "Incorrect background model for edge extraction (wrong fitting window)",
            "Energy drift during long spectrum-image acquisitions",
            "Not accounting for plural scattering when quantifying elemental ratios",
            "Beam damage altering the specimen chemistry during EELS acquisition",
        ],
        "how_to_avoid_mistakes": [
            "Keep specimen thickness < 0.5 inelastic mean free path (t/λ < 0.5)",
            "Fit background in a window just before the edge; use multiple-window methods if needed",
            "Apply energy drift correction using the zero-loss peak or a known edge",
            "Deconvolve plural scattering using Fourier-log method before quantification",
            "Use low-dose protocols and fast spectrum imaging to minimize beam damage",
        ],
    },

    "electron_holography": {
        "principle": (
            "Electron holography uses the interference between an object wave "
            "(transmitted through the specimen) and a reference wave (passing "
            "through vacuum) to record both amplitude and phase of the electron "
            "wave. An electrostatic biprism (charged wire) deflects the two "
            "waves to overlap and form interference fringes. Numerical "
            "reconstruction recovers the phase shift, which is sensitive to "
            "electrostatic potentials and magnetic fields in the specimen."
        ),
        "setup_guide": (
            "Use a TEM (≥200 kV, FEG source for high coherence) equipped "
            "with an electron biprism (a thin metallized quartz fiber at "
            "adjustable voltage 50-300 V). Position the specimen so one half "
            "of the biprism overlaps the specimen edge and the other half is "
            "in vacuum. Record the hologram on a direct-electron detector. "
            "Fringe spacing should be 3-4× the desired resolution. Acquire "
            "reference holograms (empty) for normalization."
        ),
        "common_algorithms": [
            "Fourier filtering (sideband extraction and inverse FFT for phase/amplitude)",
            "Phase unwrapping for large phase shifts (>2π)",
            "Mean inner potential measurement from phase maps",
            "Magnetic induction mapping (from phase gradient of Lorentz holography)",
            "In-line holography (through-focus series) with transport-of-intensity equation",
        ],
        "common_mistakes": [
            "Biprism voltage too low, giving insufficient overlap and poor fringe contrast",
            "Fresnel fringes from specimen edge contaminating the holographic fringes",
            "Not acquiring and dividing by a reference hologram, leaving biprism distortions",
            "Specimen too thick, reducing fringe visibility from inelastic scattering",
            "Stray magnetic fields causing unwanted phase shifts in the reference wave",
        ],
        "how_to_avoid_mistakes": [
            "Optimize biprism voltage for 3-4× oversampling of desired resolution with good contrast",
            "Extend vacuum reference beyond the specimen edge; mask Fresnel fringe regions",
            "Always acquire reference holograms and compute the normalized phase",
            "Use thin specimens (< 50-80 nm) to maintain fringe contrast above 10%",
            "Enclose the TEM column in mu-metal shielding; degauss the objective lens for Lorentz mode",
        ],
    },

    # ── DEPTH IMAGING ──────────────────────────────────────────────────────

    "tof_camera": {
        "principle": (
            "A Time-of-Flight depth camera measures the round-trip time of "
            "modulated light (typically near-infrared LEDs at 850 nm) "
            "reflected from the scene. The sensor measures the phase shift "
            "between emitted and received modulated signals at each pixel, "
            "which is proportional to the target distance: d = c·Δφ/(4π·f_mod). "
            "Typical modulation frequencies are 20-100 MHz, providing depth "
            "ranges of 0.5-10 meters with mm-cm precision."
        ),
        "setup_guide": (
            "Use an integrated ToF camera module (e.g., Microsoft Azure Kinect "
            "DK, PMD CamBoard pico, Texas Instruments OPT8241). The module "
            "contains the NIR light source, modulation driver, and ToF sensor "
            "with per-pixel demodulation circuits. Mount rigidly and calibrate "
            "intrinsic parameters (lens distortion, depth offset) and "
            "phase-to-depth nonlinearities. For multi-camera setups, "
            "synchronize or frequency-multiplex to avoid interference."
        ),
        "common_algorithms": [
            "Four-phase demodulation for distance extraction",
            "Multi-frequency unwrapping for extended unambiguous range",
            "Flying-pixel filtering (mixed pixels at depth discontinuities)",
            "Multi-path interference correction",
            "Deep-learning depth denoising and completion",
        ],
        "common_mistakes": [
            "Multi-path interference causing systematic depth errors in concave scenes",
            "Flying pixels at depth edges producing incorrect intermediate depth values",
            "Phase wrapping ambiguity when objects exceed the unambiguous range",
            "Interference from ambient NIR light (sunlight) degrading outdoor performance",
            "Systematic depth errors from non-ideal sensor response not calibrated out",
        ],
        "how_to_avoid_mistakes": [
            "Use multi-path correction algorithms or multi-frequency modulation",
            "Apply flying-pixel detection and removal based on amplitude and neighbor consistency",
            "Use dual-frequency operation to extend the unambiguous range",
            "Use narrow-band optical filter and higher modulation power for outdoor use",
            "Perform per-pixel depth calibration with a known flat reference at multiple distances",
        ],
    },

    "structured_light": {
        "principle": (
            "Structured-light depth sensing projects a known pattern (stripes, "
            "dots, coded binary patterns) onto the scene and observes the "
            "pattern deformation with a camera from a different viewpoint. "
            "The displacement (disparity) of each pattern element between "
            "projected and observed positions encodes the surface depth via "
            "triangulation. Dense depth maps are obtained by identifying "
            "pattern correspondences across the scene."
        ),
        "setup_guide": (
            "Arrange a projector (DLP or laser dot projector) and camera "
            "with a known baseline separation (5-25 cm) and convergent "
            "geometry. Calibrate the projector-camera system (intrinsics and "
            "extrinsics) using a planar calibration target. For temporal "
            "coding (Gray code), project multiple patterns sequentially. "
            "For spatial coding (single-shot, e.g., Apple FaceID dot projector), "
            "use a diffractive optical element to generate a unique dot pattern."
        ),
        "common_algorithms": [
            "Gray code + phase shifting (sequential multi-pattern decoding)",
            "Single-shot coded pattern matching (speckle or pseudo-random dot decoding)",
            "Phase unwrapping for sinusoidal fringe projection",
            "Stereo matching applied to textured scenes (active stereo)",
            "Deep-learning depth estimation from structured-light patterns",
        ],
        "common_mistakes": [
            "Ambient light washing out the projected pattern, losing depth information",
            "Specular (shiny) surfaces reflecting the projector into the camera, causing erroneous depth",
            "Occlusion zones where the projector illuminates but the camera cannot see (shadowed regions)",
            "Insufficient projector resolution limiting the achievable depth precision",
            "Color/reflectance variations in the scene altering perceived pattern intensity",
        ],
        "how_to_avoid_mistakes": [
            "Use NIR projector + camera with ambient-light rejection filter",
            "Apply polarization filtering or spray surfaces with matte coating for calibration",
            "Add a second camera or projector to reduce occlusion zones",
            "Use high-resolution projectors (1080p+) and fine patterns for sub-mm precision",
            "Use binary or phase-shifting patterns that are robust to reflectance variations",
        ],
    },

    "lidar": {
        "principle": (
            "Light Detection and Ranging (LiDAR) measures distances by emitting "
            "laser pulses (905 nm or 1550 nm) and timing their return after "
            "reflection from the scene (time-of-flight: d = c·t/2). "
            "A scanning mechanism (rotating mirror, MEMS, or optical phased "
            "array) sweeps the beam to build a 3-D point cloud of the "
            "environment. Resolution depends on the beam divergence, scanning "
            "density, and pulse timing precision."
        ),
        "setup_guide": (
            "Select a LiDAR sensor appropriate for the application: mechanical "
            "spinning (Velodyne VLP-16/128 for autonomous vehicles), solid-"
            "state (Livox, Ouster), or airborne (Leica ALS80 for terrain "
            "mapping). Mount rigidly and combine with an IMU and GNSS for "
            "georeferencing. Calibrate intrinsic parameters (beam angles, "
            "timing offsets, intensity response) and extrinsics (relative to "
            "vehicle coordinate frame). Process returns: first/last/full "
            "waveform for different applications."
        ),
        "common_algorithms": [
            "Point cloud registration (ICP, NDT for multi-scan alignment)",
            "Ground filtering and classification (progressive morphological filter)",
            "SLAM (Simultaneous Localization and Mapping) with LiDAR",
            "Object detection and segmentation (PointNet, PointPillars)",
            "Surface reconstruction from point clouds (Poisson, ball-pivoting)",
        ],
        "common_mistakes": [
            "Multi-echo / multi-path reflections causing ghost points",
            "Motion distortion in the point cloud from vehicle movement during one scan rotation",
            "Incorrect calibration causing misalignment between LiDAR and camera data",
            "Rain, fog, or dust causing false returns and reduced range",
            "Near-range blind zone where the receiver is not sensitive to returns",
        ],
        "how_to_avoid_mistakes": [
            "Filter ghost points using intensity thresholds and multi-return analysis",
            "Apply ego-motion compensation using IMU data to deskew each scan",
            "Perform target-based or targetless calibration between LiDAR and other sensors",
            "Use 1550 nm wavelength (eye-safe and less affected by rain) for outdoor applications",
            "Account for minimum range specification; fuse with short-range sensors if needed",
        ],
    },

    # ── REMOTE SENSING ─────────────────────────────────────────────────────

    "sar": {
        "principle": (
            "Synthetic Aperture Radar achieves fine azimuth resolution by "
            "coherently processing radar echoes collected as the antenna moves "
            "along its flight path, synthesizing an aperture much larger than "
            "the physical antenna. The SAR signal processor applies matched "
            "filtering (pulse compression) in both range and azimuth to form "
            "a high-resolution complex image. SAR operates through clouds, "
            "at night, and in all weather conditions."
        ),
        "setup_guide": (
            "Mount a microwave transmitter/receiver (C-band 5.4 GHz, L-band "
            "1.3 GHz, or X-band 9.6 GHz) on a satellite (Sentinel-1, "
            "RADARSAT) or aircraft. The antenna illuminates a strip on the "
            "ground as the platform moves. Record the complex (I/Q) echo "
            "data with precise pulse timing and platform position/velocity "
            "from GNSS/INS. Range resolution is set by pulse bandwidth "
            "(1-200 MHz); azimuth resolution equals L_ant/2 (half the "
            "antenna length)."
        ),
        "common_algorithms": [
            "Range-Doppler algorithm (range compression + azimuth compression)",
            "Chirp scaling algorithm for wide-swath SAR",
            "Omega-K (wavenumber domain) algorithm for high-resolution spotlight SAR",
            "InSAR (Interferometric SAR) for DEM generation and deformation mapping",
            "PolSAR decomposition (Cloude-Pottier, Freeman-Durden) for land classification",
        ],
        "common_mistakes": [
            "Incorrect motion compensation causing azimuth defocusing",
            "Range cell migration not properly corrected for squinted geometries",
            "Phase errors from atmospheric delay (troposphere, ionosphere) in InSAR",
            "Ambiguities (range or azimuth) from incorrect PRF selection",
            "Speckle noise mistaken for real features in SAR imagery",
        ],
        "how_to_avoid_mistakes": [
            "Use precise INS/GNSS data for autofocus and motion compensation",
            "Apply appropriate RCMC (Range Cell Migration Correction) for the imaging geometry",
            "Use atmospheric phase screens (from weather models or GNSS delays) for InSAR correction",
            "Design PRF to avoid range and azimuth ambiguity constraints for the swath geometry",
            "Apply multi-look or speckle filtering (Lee, refined-Lee) before interpretation",
        ],
    },

    "sonar": {
        "principle": (
            "Sonar imaging uses acoustic waves (typically 50 kHz to 1 MHz) "
            "to image underwater scenes. Active sonar transmits a sound pulse "
            "and records the echoes from the seabed, objects, or water column. "
            "The propagation speed in water (~1500 m/s, varying with "
            "temperature, salinity, and pressure) determines the time-to-"
            "distance relationship. Side-scan sonar and multibeam bathymetry "
            "produce 2-D and 3-D maps of the underwater environment."
        ),
        "setup_guide": (
            "For side-scan sonar: mount a towfish with two transducer arrays "
            "(port and starboard) that ensonify a swath perpendicular to the "
            "survey track. For multibeam: mount a hull-mounted array (e.g., "
            "Kongsberg EM2040, 200-400 kHz). Sound velocity profiler (SVP) "
            "measurements are essential for ray-tracing corrections. Integrate "
            "with GNSS positioning and motion reference unit (MRU) for heave, "
            "pitch, and roll compensation."
        ),
        "common_algorithms": [
            "Beamforming (delay-and-sum for multibeam sonar)",
            "Synthetic aperture sonar (SAS) processing for enhanced azimuth resolution",
            "Bottom detection and bathymetric surface extraction",
            "Acoustic backscatter classification for seabed characterization",
            "Deep-learning object detection for mine countermeasures or marine archaeology",
        ],
        "common_mistakes": [
            "Incorrect sound velocity profile causing depth and position errors",
            "Multipath reflections (surface bounce, bottom bounce) creating ghost targets",
            "Nadir gap (directly beneath the sonar) with no acoustic coverage",
            "Motion artifacts from ship heave/pitch/roll not compensated",
            "Side-lobe artifacts creating false targets near strong reflectors",
        ],
        "how_to_avoid_mistakes": [
            "Measure SVP at the survey site; update periodically during long surveys",
            "Use multiple-return filtering and angle-based discrimination to remove multipath",
            "Overlap adjacent swaths to fill the nadir gap; use a vertical beam sounder",
            "Apply real-time MRU data for heave, pitch, and roll correction of depth measurements",
            "Use advanced beamforming (CAPON, MVDR) to suppress side-lobe responses",
        ],
    },

    # ── PARTICLE IMAGING ───────────────────────────────────────────────────

    "neutron_tomo": {
        "principle": (
            "Neutron radiography and tomography image the transmission of a "
            "thermal or cold neutron beam through a sample. Neutrons interact "
            "with nuclei (not electrons), providing complementary contrast to "
            "X-rays: hydrogen-rich materials (water, polymers, organics) "
            "attenuate neutrons strongly, while metals like aluminum and lead "
            "are relatively transparent. Tomographic reconstruction from "
            "multiple projection angles yields 3-D maps of neutron "
            "attenuation."
        ),
        "setup_guide": (
            "Access a research reactor or spallation neutron source with an "
            "imaging beamline (e.g., ICON at PSI, IMAT at ISIS, NIST BT-2). "
            "A collimated neutron beam (thermal or cold, 1-10 Å) passes "
            "through the sample, and a scintillator-camera system (⁶LiF/ZnS "
            "screen + sCMOS camera) records the transmitted intensity. "
            "Rotate the sample through 180° or 360° for tomography. Spatial "
            "resolution is typically 20-100 μm, limited by beam divergence "
            "and scintillator thickness."
        ),
        "common_algorithms": [
            "Filtered back-projection (FBP) adapted for neutron tomography",
            "Iterative reconstruction (SIRT, CGLS) for limited-angle or noisy data",
            "Beam hardening correction for polychromatic neutron spectra",
            "Scattering correction (point-scattered function approach)",
            "Neutron phase-contrast tomography (grating interferometry)",
        ],
        "common_mistakes": [
            "Scattering from hydrogen-rich samples producing artifacts (halo around sample)",
            "Beam hardening (spectral hardening) not corrected for polychromatic beams",
            "Activation of sample materials, creating radiation safety issues post-experiment",
            "Gamma contamination in the beam degrading image quality",
            "Insufficient exposure time per projection, yielding noisy tomograms",
        ],
        "how_to_avoid_mistakes": [
            "Apply scattering correction algorithms; use thin or diluted hydrogen-rich samples",
            "Correct beam hardening with polynomial methods or by using a velocity selector (monochromatic)",
            "Check sample activation potential before irradiation; use short-lived isotope-free materials",
            "Use gamma-blind detectors (⁶Li glass) or filters to reject gamma contamination",
            "Optimize exposure per projection for adequate SNR; total scan time often 2-8 hours",
        ],
    },

    "proton_radiography": {
        "principle": (
            "Proton radiography images the transmission and scattering of "
            "high-energy protons (50-800 MeV) through dense objects. Unlike "
            "X-rays, protons undergo significant multiple Coulomb scattering "
            "(MCS) in matter, which provides density and compositional contrast. "
            "Both transmission (energy loss) and scattering angle measurements "
            "contribute to image formation. Proton radiography can penetrate "
            "very dense materials (steel, depleted uranium) that are opaque "
            "to X-rays."
        ),
        "setup_guide": (
            "Requires a high-energy proton accelerator facility (synchrotron "
            "or cyclotron delivering 200-800 MeV protons). The object is "
            "placed in the beam path between tracking detectors (silicon strip "
            "or GEM detectors) that measure each proton's position and angle "
            "before and after the object. A magnetic spectrometer (quadrupole "
            "lens system, e.g., at LANL pRad facility) focuses transmitted "
            "protons onto a scintillator + camera detector."
        ),
        "common_algorithms": [
            "Most Likely Path (MLP) estimation for proton CT reconstruction",
            "Filtered back-projection with scattering-angle weighting",
            "Algebraic reconstruction (ART) with MCS forward model",
            "Material discrimination from dual-parameter (transmission + scattering) analysis",
            "Deep-learning proton CT reconstruction for reduced view angles",
        ],
        "common_mistakes": [
            "Ignoring multiple Coulomb scattering in the reconstruction model, causing blur",
            "Nuclear interaction losses (protons stopped or scattered out of detector acceptance)",
            "Insufficient proton statistics leading to noisy images",
            "Energy straggling not modeled, causing depth-of-field blur in radiography",
            "Detector alignment errors between upstream and downstream tracking systems",
        ],
        "how_to_avoid_mistakes": [
            "Use MLP or cubic spline path estimation in iterative reconstruction algorithms",
            "Account for nuclear interaction losses in the forward model; filter outlier tracks",
            "Accumulate sufficient proton histories (>10⁶ for radiography, >10⁸ for proton CT)",
            "Include energy straggling in the forward model or use higher energy protons to reduce it",
            "Carefully align tracking detectors with survey or use track-based alignment algorithms",
        ],
    },

    "muon_tomo": {
        "principle": (
            "Muon tomography uses naturally occurring cosmic-ray muons to "
            "image the internal density structure of large objects (buildings, "
            "volcanoes, cargo containers). Muons undergo multiple Coulomb "
            "scattering, with the scattering angle proportional to the areal "
            "density and atomic number of the traversed material. By measuring "
            "the incoming and outgoing muon trajectories, the density "
            "distribution inside the object can be tomographically "
            "reconstructed."
        ),
        "setup_guide": (
            "Place tracking detectors (drift tubes, scintillator strips, "
            "resistive plate chambers, or GEM detectors) above and below "
            "(or around) the object to be imaged. Each detector station "
            "measures the position and angle of each cosmic-ray muon before "
            "and after it traverses the object. Typical cosmic-ray muon flux "
            "is ~10,000 muons/m²/min at sea level. Exposure times range from "
            "minutes (for dense nuclear materials) to months (for geological "
            "structures like volcanoes)."
        ),
        "common_algorithms": [
            "Point of Closest Approach (POCA) voxel reconstruction",
            "Maximum Likelihood / Expectation Maximization (ML/EM) scattering tomography",
            "Angle Statistics Reconstruction (ASR) for material discrimination",
            "Binned scattering density reconstruction",
            "Deep-learning muon tomography for faster convergence with fewer muons",
        ],
        "common_mistakes": [
            "Insufficient muon statistics for the desired spatial resolution (need long exposure)",
            "Detector alignment errors causing incorrect scattering angle measurements",
            "Not accounting for muon momentum spectrum (affects scattering angle distribution)",
            "Background tracks (electrons, low-momentum muons) contaminating the data",
            "POCA algorithm limitations in complex, non-point-like geometries",
        ],
        "how_to_avoid_mistakes": [
            "Calculate required exposure time based on object size, density, and desired resolution",
            "Align detectors carefully using straight-through cosmic ray tracks as calibration",
            "Use momentum measurement (from curvature in a magnetic field) or momentum-dependent MCS model",
            "Apply track quality cuts (chi-squared, minimum number of detector hits) to reject background",
            "Use iterative reconstruction (ML/EM) rather than POCA for quantitative density imaging",
        ],
    },
}

# Merge introductions into MODALITY_DATABASE
for _key, _intro in _MODALITY_INTRODUCTIONS.items():
    if _key in MODALITY_DATABASE:
        MODALITY_DATABASE[_key]["introduction"] = _intro

# Inject setup_diagram_url for each modality (served from /static)
for _key in MODALITY_DATABASE:
    MODALITY_DATABASE[_key]["setup_diagram_url"] = f"/static/img/setups/{_key}.png"


# ── Public API ──────────────────────────────────────────────────────────────


def get_modality_info(modality_key: str) -> dict:
    """Return full modality record.  Raises KeyError if not found."""
    if modality_key not in MODALITY_DATABASE:
        raise KeyError(
            f"Unknown modality '{modality_key}'. "
            f"Available: {sorted(MODALITY_DATABASE.keys())}"
        )
    return dict(MODALITY_DATABASE[modality_key])


def get_modality_description(modality_key: str) -> str:
    """Return the physics description for a modality."""
    return get_modality_info(modality_key)["description"]


def get_experimental_setup(modality_key: str) -> dict:
    """Return the default experimental setup for a modality."""
    return dict(get_modality_info(modality_key)["experimental_setup"])


def list_modalities_by_category(category: str) -> list[str]:
    """Return modality keys belonging to a category."""
    return [k for k, v in MODALITY_DATABASE.items() if v["category"] == category]


def list_all_categories() -> list[str]:
    """Return sorted list of unique categories."""
    return sorted({v["category"] for v in MODALITY_DATABASE.values()})


def list_all_modality_keys() -> list[str]:
    """Return all 64 modality keys in insertion order."""
    return list(MODALITY_DATABASE.keys())
