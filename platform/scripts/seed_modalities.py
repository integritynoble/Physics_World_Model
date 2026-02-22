"""
Seed the modality_basics knowledge base with initial modalities.
Run: python scripts/seed_modalities.py
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEED_MODALITIES = [
    {
        "modality_key": "cassi",
        "display_name": "CASSI (Coded Aperture Snapshot Spectral Imaging)",
        "category": "spectral",
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
        "default_solver": "gap_tv",
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
        "default_solver": "compressed_sensing",
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
        "default_solver": "pie",
        "evaluation_metrics": ["psnr", "ssim", "phase_error"],
        "tags": ["coherent", "phase_retrieval", "scanning"],
    },
    {
        "modality_key": "holography",
        "display_name": "Digital Holography",
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
        "category": "computational",
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
        "default_solver": "tv_fista",
        "evaluation_metrics": ["psnr", "ssim"],
        "tags": ["compressive_sensing", "single_pixel", "computational"],
    },
]


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
        print("Seeding complete.")


if __name__ == "__main__":
    asyncio.run(main())
