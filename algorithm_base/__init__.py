"""PWM Algorithm Base — 168 modalities, 541 solvers.

Usage:
    # Import by modality
    from algorithm_base.cassi import run_solver
    x_hat = run_solver("traditional_cpu", y, operator)

    # Via registry
    from algorithm_base import get_solver, list_solvers, list_modalities
    solver_fn = get_solver("cassi", "traditional_cpu")
    x_hat = solver_fn(y, operator, {})
"""
from ._registry import get_solver, list_solvers, list_modalities, run_solver

MODALITIES = ['acoustic_emission', 'acoustic_microscopy', 'active_thermography', 'adaptive_optics', 'afm', 'angiography', 'asl_mri', 'atom_probe', 'bioluminescence_tomo', 'brachytherapy_img', 'brillouin', 'cacti', 'cars', 'cassi', 'cathodoluminescence', 'cbct', 'cest_mri', 'ceus', 'clem', 'coded_exposure', 'confocal_3d', 'confocal_endomicroscopy', 'confocal_livecell', 'coronagraphy', 'cryo_em', 'cryo_et', 'ct', 'ct_fluorescence', 'cup', 'dark_field', 'desi', 'dexa', 'dic', 'diffusion_mri', 'digital_breast_tomo', 'dna_paint', 'doppler_ultrasound', 'dot', 'ebsd', 'eddy_current', 'edx_mapping', 'eels', 'eht_imaging', 'elastography', 'electron_diffraction', 'electron_holography', 'electron_tomography', 'endoscopy', 'entangled_photon', 'event_camera', 'expansion', 'fib_sem', 'flash_lidar', 'flim', 'fluoroscopy', 'fmri', 'fpm', 'ftir_imaging', 'fundus', 'fwi', 'gaussian_splatting', 'ghost_imaging', 'gpr', 'gravitational_wave', 'hdr_imaging', 'holography', 'hyperspectral_remote', 'impedance_tomo', 'industrial_ct', 'insar', 'integral', 'ism', 'ivus', 'lattice_lightsheet', 'lensless', 'libs', 'lidar', 'light_field', 'lightsheet', 'lucky_imaging', 'machine_vision', 'magnetic_particle', 'maldi_msi', 'mammography', 'matrix', 'mfm', 'minflux', 'mr_elastography', 'mr_fingerprinting', 'mra', 'mri', 'mrs', 'multispectral_sat', 'muon_tomo', 'nerf', 'neutron_diffraction', 'neutron_tomo', 'nirs_brain', 'nsom', 'ocean_acoustic_tomo', 'ocean_color', 'oct', 'octa', 'odt', 'palm_storm', 'panorama', 'particle_calorimetry', 'passive_microwave', 'pet', 'pet_ct', 'pet_mr', 'phase_contrast', 'phase_retrieval', 'photoacoustic', 'photometric_stereo', 'polarization', 'polsar', 'portal_imaging', 'proton_radiography', 'proton_therapy_img', 'ptychography', 'pump_probe', 'quantum_illumination', 'radio_astronomy', 'radio_interferometry', 'raman_imaging', 'sar', 'saxs', 'seismic_tomo', 'sem', 'shearography', 'shg', 'sim', 'sims', 'solar_imaging', 'sonar', 'spc', 'spect', 'spect_ct', 'spectral_ct', 'spinning_disk', 'srs', 'sted', 'stem', 'stm', 'streak_camera', 'structured_light', 'swi', 'talbot_lau', 'tem', 'terahertz', 'three_photon', 'tirf', 'tof_camera', 'two_photon', 'ultrasonic_phased_array', 'ultrasound', 'us_mri', 'waxs', 'weather_radar', 'widefield', 'widefield_lowdose', 'xfel_sfx', 'xray_crystallography', 'xray_ndt', 'xray_radiography', 'xrf_imaging', 'xrf_tomo']

__all__ = ["get_solver", "list_solvers", "list_modalities", "run_solver", "MODALITIES"]
