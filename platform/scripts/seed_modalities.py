"""
Seed the modality_basics knowledge base with all 64 imaging modalities.
Run: python scripts/seed_modalities.py
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Full-detail entries for the 6 core benchmarked modalities ─────────────

_CORE_MODALITIES = [
    {
        "modality_key": "cassi",
        "display_name": "CASSI (Coded Aperture Snapshot Spectral Imaging)",
        "category": "compressive",
        "physics_class": "spectral_coding",
        "forward_model_family": "coded_aperture_dispersion",
        "primitive_gates": ["dispersion", "coded_aperture", "integration"],
        "wave_model": "ray",
        "sensor_type": "cmos",
        "source_type": "broadband",
        "geometry": "planar",
        "typical_x_dims": [256, 256, 28],
        "typical_y_dims": [256, 310],
        "typical_snr_range": [20.0, 40.0],
        "calibration_params": ["mask", "dispersion_curve", "dark_frame"],
        "mismatch_modes": ["mask_shift", "spectral_response_drift"],
        "noise_model": "gaussian",
        "default_solver": "mst",
        "evaluation_metrics": ["psnr", "ssim", "sam"],
        "tags": ["spectral", "coded_aperture", "snapshot", "benchmark"],
    },
    {
        "modality_key": "ct",
        "display_name": "CT (X-ray Computed Tomography)",
        "category": "medical",
        "physics_class": "tomographic",
        "forward_model_family": "radon_transform",
        "primitive_gates": ["xray_source", "rotation", "line_integral", "detection"],
        "wave_model": "ray",
        "sensor_type": "scintillator_detector",
        "source_type": "xray_tube",
        "geometry": "rotational",
        "typical_x_dims": [256, 256],
        "typical_y_dims": [180, 362],
        "typical_snr_range": [25.0, 45.0],
        "calibration_params": ["flat_field", "center_of_rotation", "beam_hardening_correction"],
        "mismatch_modes": ["center_offset", "beam_hardening", "scatter"],
        "noise_model": "poisson",
        "default_solver": "fbp",
        "evaluation_metrics": ["psnr", "ssim", "hu_accuracy"],
        "tags": ["medical", "tomography", "xray"],
    },
    {
        "modality_key": "mri",
        "display_name": "MRI (Magnetic Resonance Imaging)",
        "category": "medical",
        "physics_class": "fourier_sampling",
        "forward_model_family": "fourier_undersampling",
        "primitive_gates": ["rf_excitation", "gradient_encoding", "k_space_sampling", "coil_sensitivity"],
        "wave_model": "em_precession",
        "sensor_type": "rf_coil",
        "source_type": "rf_pulse",
        "geometry": "k_space",
        "typical_x_dims": [256, 256],
        "typical_y_dims": [256, 128],
        "typical_snr_range": [15.0, 35.0],
        "calibration_params": ["coil_sensitivity_maps", "field_inhomogeneity", "trajectory"],
        "mismatch_modes": ["off_resonance", "motion", "eddy_current"],
        "noise_model": "gaussian",
        "default_solver": "sense",
        "evaluation_metrics": ["psnr", "ssim", "nrmse"],
        "tags": ["medical", "fourier", "k_space"],
    },
    {
        "modality_key": "ptychography",
        "display_name": "Ptychography",
        "category": "coherent",
        "physics_class": "coherent_diffraction",
        "forward_model_family": "ptychographic_forward",
        "primitive_gates": ["coherent_illumination", "probe_scan", "far_field_diffraction"],
        "wave_model": "scalar_wave",
        "sensor_type": "photon_counter",
        "source_type": "coherent_beam",
        "geometry": "ptychographic_scan",
        "typical_x_dims": [512, 512],
        "typical_y_dims": [64, 128, 128],
        "typical_snr_range": [10.0, 30.0],
        "calibration_params": ["probe_function", "scan_positions", "detector_distance"],
        "mismatch_modes": ["position_error", "partial_coherence", "probe_drift"],
        "noise_model": "poisson",
        "default_solver": "epie",
        "evaluation_metrics": ["psnr", "ssim", "phase_error"],
        "tags": ["coherent", "phase_retrieval", "scanning"],
    },
    {
        "modality_key": "holography",
        "display_name": "Digital Holographic Microscopy",
        "category": "coherent",
        "physics_class": "interferometric",
        "forward_model_family": "holographic_forward",
        "primitive_gates": ["coherent_illumination", "interference", "propagation"],
        "wave_model": "scalar_wave",
        "sensor_type": "cmos",
        "source_type": "laser",
        "geometry": "planar",
        "typical_x_dims": [512, 512],
        "typical_y_dims": [512, 512],
        "calibration_params": ["wavelength", "propagation_distance", "reference_beam"],
        "mismatch_modes": ["twin_image", "reference_error", "coherence_loss"],
        "noise_model": "gaussian",
        "default_solver": "angular_spectrum",
        "evaluation_metrics": ["psnr", "ssim", "phase_error"],
        "tags": ["coherent", "interferometric", "phase"],
    },
    {
        "modality_key": "spc",
        "display_name": "Single-Pixel Camera",
        "category": "compressive",
        "physics_class": "compressive_sensing",
        "forward_model_family": "structured_illumination_sensing",
        "primitive_gates": ["spatial_modulation", "bucket_detection"],
        "wave_model": "ray",
        "sensor_type": "single_pixel_detector",
        "source_type": "broadband",
        "geometry": "planar",
        "typical_x_dims": [64, 64],
        "typical_y_dims": [1024],
        "calibration_params": ["pattern_matrix", "detector_response"],
        "mismatch_modes": ["pattern_misalignment", "detector_nonlinearity"],
        "noise_model": "gaussian",
        "default_solver": "pnp_fista",
        "evaluation_metrics": ["psnr", "ssim"],
        "tags": ["compressive_sensing", "single_pixel", "computational"],
    },
]

# ── Compact entries for the remaining 58 modalities ───────────────────────
# Format: (modality_key, display_name, category, physics_class, default_solver, tags)

_EXTRA_MODALITIES_COMPACT = [
    # ── Microscopy ────────────────────────────────────────────────────────
    ("widefield", "Widefield Fluorescence Microscopy", "microscopy",
     "fluorescence", "richardson_lucy",
     ["microscopy", "fluorescence", "deconvolution"]),
    ("widefield_lowdose", "Low-Dose Widefield Microscopy", "microscopy",
     "fluorescence", "pnp_hqs",
     ["microscopy", "fluorescence", "low_dose", "denoising"]),
    ("confocal_livecell", "Confocal Live-Cell Microscopy", "microscopy",
     "fluorescence", "richardson_lucy",
     ["microscopy", "confocal", "live_cell"]),
    ("confocal_3d", "Confocal 3D Z-Stack", "microscopy",
     "fluorescence", "richardson_lucy_3d",
     ["microscopy", "confocal", "3d", "z_stack"]),
    ("sim", "Structured Illumination Microscopy (SIM)", "microscopy",
     "structured_illumination", "wiener_sim",
     ["microscopy", "super_resolution", "structured_illumination"]),
    ("lightsheet", "Light-Sheet Fluorescence Microscopy", "microscopy",
     "fluorescence", "fourier_notch_destripe",
     ["microscopy", "lightsheet", "3d"]),
    ("two_photon", "Two-Photon / Multiphoton Microscopy", "microscopy",
     "fluorescence", "richardson_lucy",
     ["microscopy", "two_photon", "deep_tissue"]),
    ("sted", "STED Microscopy", "microscopy",
     "fluorescence", "richardson_lucy",
     ["microscopy", "super_resolution", "sted"]),
    ("tirf", "TIRF Microscopy", "microscopy",
     "fluorescence", "richardson_lucy",
     ["microscopy", "tirf", "evanescent_wave"]),
    ("flim", "Fluorescence Lifetime Imaging (FLIM)", "microscopy",
     "fluorescence_lifetime", "phasor",
     ["microscopy", "flim", "lifetime"]),
    ("fpm", "Fourier Ptychographic Microscopy", "microscopy",
     "fourier_ptychography", "sequential_phase_retrieval",
     ["microscopy", "ptychography", "phase_retrieval"]),
    ("palm_storm", "PALM/STORM Single-Molecule Localization", "microscopy",
     "single_molecule", "thunderstorm",
     ["microscopy", "super_resolution", "localization"]),
    ("polarization", "Polarization Microscopy", "microscopy",
     "polarimetric", "pnp_hqs",
     ["microscopy", "polarization", "birefringence"]),

    # ── Medical (non-core) ────────────────────────────────────────────────
    ("cbct", "Cone-Beam CT (CBCT)", "medical",
     "tomographic", "fdk",
     ["medical", "tomography", "cone_beam"]),
    ("fmri", "Functional MRI (BOLD)", "medical",
     "fourier_sampling", "sense",
     ["medical", "fmri", "bold", "functional"]),
    ("diffusion_mri", "Diffusion MRI (DTI)", "medical",
     "fourier_sampling", "weighted_least_squares",
     ["medical", "diffusion", "dti", "tractography"]),
    ("mrs", "MR Spectroscopy", "medical",
     "fourier_sampling", "lcmodel",
     ["medical", "spectroscopy", "metabolites"]),
    ("pet", "Positron Emission Tomography (PET)", "medical",
     "emission_tomographic", "mlem",
     ["medical", "nuclear", "pet", "emission"]),
    ("spect", "Single Photon Emission CT (SPECT)", "medical",
     "emission_tomographic", "mlem",
     ["medical", "nuclear", "spect", "emission"]),
    ("ultrasound", "Ultrasound Imaging", "medical",
     "acoustic", "tv_fista",
     ["medical", "ultrasound", "acoustic"]),
    ("doppler_ultrasound", "Doppler Ultrasound", "medical_ultrasound",
     "acoustic", "autocorrelation_estimator",
     ["medical", "ultrasound", "doppler", "flow"]),
    ("elastography", "Shear-Wave Elastography", "medical_ultrasound",
     "acoustic", "time_of_flight_inversion",
     ["medical", "ultrasound", "elastography", "stiffness"]),
    ("fluoroscopy", "Fluoroscopy", "medical",
     "radiographic", "tv_fista",
     ["medical", "xray", "real_time"]),
    ("angiography", "X-ray Angiography", "medical",
     "radiographic", "dsa_subtraction",
     ["medical", "xray", "angiography", "vascular"]),
    ("xray_radiography", "X-ray Radiography", "medical",
     "radiographic", "tv_fista",
     ["medical", "xray", "projection"]),
    ("mammography", "Mammography", "medical",
     "radiographic", "tv_fista",
     ["medical", "xray", "mammography", "breast"]),
    ("dexa", "Dual-Energy X-ray Absorptiometry (DEXA)", "medical",
     "radiographic", "dual_energy_decomposition",
     ["medical", "xray", "bone_density"]),
    ("dot", "Diffuse Optical Tomography", "medical",
     "diffuse_optical", "born_approx",
     ["medical", "optical", "diffuse", "tomography"]),
    ("photoacoustic", "Photoacoustic Imaging", "medical",
     "photoacoustic", "back_projection",
     ["medical", "photoacoustic", "hybrid"]),

    # ── Electron Microscopy ───────────────────────────────────────────────
    ("sem", "Scanning Electron Microscopy (SEM)", "electron_microscopy",
     "electron_beam", "direct_imaging",
     ["electron", "scanning", "surface"]),
    ("tem", "Transmission Electron Microscopy (TEM)", "electron_microscopy",
     "electron_beam", "ctf_correction",
     ["electron", "transmission", "high_resolution"]),
    ("stem", "Scanning TEM (STEM)", "electron_microscopy",
     "electron_beam", "direct_imaging",
     ["electron", "scanning", "transmission"]),
    ("electron_tomography", "Electron Tomography", "electron_microscopy",
     "tomographic", "sirt",
     ["electron", "tomography", "3d"]),
    ("electron_diffraction", "4D-STEM Electron Diffraction", "electron_microscopy",
     "coherent_diffraction", "ptychography_epie",
     ["electron", "diffraction", "4d_stem"]),
    ("electron_holography", "Electron Holography", "electron_microscopy",
     "interferometric", "fourier_sideband",
     ["electron", "holography", "phase"]),
    ("ebsd", "Electron Backscatter Diffraction (EBSD)", "electron_microscopy",
     "diffraction", "hough_indexing",
     ["electron", "crystallography", "orientation"]),
    ("eels", "Electron Energy Loss Spectroscopy (EELS)", "electron_microscopy",
     "spectroscopic", "fourier_ratio",
     ["electron", "spectroscopy", "energy_loss"]),

    # ── Coherent (non-core) ───────────────────────────────────────────────
    ("phase_retrieval", "Coherent Diffractive Imaging (CDI)", "coherent",
     "coherent_diffraction", "hio",
     ["coherent", "phase_retrieval", "lensless"]),

    # ── Compressive (non-core) ────────────────────────────────────────────
    ("cacti", "CACTI (Coded Aperture Compressive Temporal Imaging)", "compressive",
     "temporal_coding", "gap_tv",
     ["compressive", "video", "temporal", "snapshot"]),
    ("matrix", "Generic Matrix Sensing", "compressive",
     "compressive_sensing", "fista_l2",
     ["compressive", "generic", "matrix"]),

    # ── Computational ─────────────────────────────────────────────────────
    ("light_field", "Light Field Imaging", "computational",
     "light_field", "shift_and_sum",
     ["computational", "light_field", "plenoptic"]),
    ("integral", "Integral Photography", "computational",
     "light_field", "depth_estimation",
     ["computational", "integral", "multi_view"]),
    ("lensless", "Lensless (Diffuser Camera) Imaging", "computational_photography",
     "lensless", "admm_tv",
     ["computational", "lensless", "diffuser"]),
    ("panorama", "Panorama Multi-Focus Fusion", "computational_photography",
     "multi_focus", "laplacian_pyramid_fusion",
     ["computational", "panorama", "fusion"]),

    # ── Depth Imaging ─────────────────────────────────────────────────────
    ("tof_camera", "Time-of-Flight Depth Camera", "depth_imaging",
     "time_of_flight", "tv_fista",
     ["depth", "tof", "3d"]),
    ("structured_light", "Structured-Light Depth Camera", "depth_imaging",
     "structured_light", "phase_unwrap",
     ["depth", "structured_light", "3d"]),
    ("lidar", "LiDAR Scanner", "depth_imaging",
     "time_of_flight", "tv_fista",
     ["depth", "lidar", "point_cloud"]),

    # ── Clinical Optics ───────────────────────────────────────────────────
    ("oct", "Optical Coherence Tomography (OCT)", "clinical_optics",
     "interferometric", "fft_recon",
     ["clinical", "oct", "retinal"]),
    ("octa", "OCT Angiography", "clinical_optics",
     "interferometric", "tv_fista",
     ["clinical", "oct", "angiography", "vascular"]),
    ("fundus", "Fundus Camera", "clinical_optics",
     "imaging", "richardson_lucy",
     ["clinical", "retinal", "fundus"]),
    ("endoscopy", "Fiber Bundle Endoscopy", "clinical_optics",
     "fiber_bundle", "tv_fista",
     ["clinical", "endoscopy", "fiber"]),

    # ── Neural Rendering ──────────────────────────────────────────────────
    ("nerf", "Neural Radiance Fields (NeRF)", "neural_rendering",
     "neural_volume", "nerf_mlp",
     ["neural", "3d", "view_synthesis"]),
    ("gaussian_splatting", "3D Gaussian Splatting", "neural_rendering",
     "neural_volume", "gaussian_splatting_3dgs",
     ["neural", "3d", "gaussian", "real_time"]),

    # ── Remote Sensing ────────────────────────────────────────────────────
    ("sar", "Synthetic Aperture Radar (SAR)", "remote_sensing",
     "radar", "backprojection",
     ["remote_sensing", "radar", "microwave"]),
    ("sonar", "Sonar Imaging", "remote_sensing",
     "acoustic", "beamform_das",
     ["remote_sensing", "sonar", "underwater"]),

    # ── Particle Imaging ──────────────────────────────────────────────────
    ("neutron_tomo", "Neutron Radiography / Tomography", "particle_imaging",
     "tomographic", "filtered_back_projection",
     ["particle", "neutron", "tomography"]),
    ("proton_radiography", "Proton Radiography", "particle_imaging",
     "radiographic", "filtered_back_projection",
     ["particle", "proton", "radiography"]),
    ("muon_tomo", "Muon Tomography", "particle_imaging",
     "tomographic", "poca_reconstruction",
     ["particle", "muon", "tomography"]),
]


def _expand_compact(tuples):
    """Expand compact tuples into full dicts."""
    out = []
    for key, display, cat, phys, solver, tags in tuples:
        out.append({
            "modality_key": key,
            "display_name": display,
            "category": cat,
            "physics_class": phys,
            "default_solver": solver,
            "tags": tags,
        })
    return out


SEED_MODALITIES = _CORE_MODALITIES + _expand_compact(_EXTRA_MODALITIES_COMPACT)


async def main():
    from sqlalchemy import select
    from pwm_platform.db.database import async_session_factory, init_db
    from pwm_platform.db.models import ModalityBasics

    await init_db()

    async with async_session_factory() as db:
        for data in SEED_MODALITIES:
            key = data["modality_key"]
            result = await db.execute(
                select(ModalityBasics).where(ModalityBasics.modality_key == key)
            )
            if result.scalar_one_or_none():
                print(f"  Skip (exists): {key}")
                continue

            m = ModalityBasics(**data)
            db.add(m)
            print(f"  Seed: {key} — {data['display_name']}")

        await db.commit()
        count = len(SEED_MODALITIES)
        print(f"Seeding complete. {count} modalities in registry.")


if __name__ == "__main__":
    asyncio.run(main())
