# Benchmark Dataset State Tracker

Last updated: 2026-03-11T06:30Z

## Status Legend
- **Dataset**: creating public, dev and hidden dataset
- **Benchmark**: update the benchmark of https://pwm.platformai.org/benchmark
- done / in-progress / pending

## Priority 1 — Core Medical & Imaging

| Modality | Dataset | Benchmark | Notes |
|----------|---------|-----------|-------|
| ct | done | done | LoDoPaB-CT, fan-beam Radon, 11/20/20 |
| mri | done | done | M4Raw k-space Fourier, 12/20/20, ZF-IFFT baseline |
| pet | done | done | Radon+Poisson+attenuation, 12/20/20, FBP baseline ~35 dB |
| ultrasound | done | done | PSF+speckle+attenuation, 12/20/20, Wiener baseline |
| oct | done | done | Retinal B-scan, PSF+speckle+rolloff, 12/20/20, median baseline ~22 dB |
| mammography | done | done | Beer-Lambert+Poisson+scatter, 12/20/20, Wiener+TV ~22 dB |
| cbct | done | done | Cone-beam CT, FDK recon, 10/20/20 |
| spect | done | done | Radon+attenuation+Poisson+scatter, 12/20/20, FBP baseline |
| fundus | done | done | Defocus PSF+illumination+Poisson-Gaussian, 12/20/20 |
| endoscopy | done | done | Barrel distortion+vignetting, 12/20/20, Wiener ~15 dB |
| fmri | done | done | BOLD+k-space undersampling, 12/20/20, ZF-IFFT baseline |
| diffusion_mri | done | done | ADC contrast, k-space undersampling, 12/20/20 |

## Priority 2 — Microscopy & Optical

| Modality | Dataset | Benchmark | Notes |
|----------|---------|-----------|-------|
| palm_storm | done | done | SMLM Poisson+PSF+readout, 12/20/20, Gaussian fitting ~31 dB |
| sted | done | done | STED depletion PSF+Poisson, 12/20/20, RL deconv baseline |
| sim | done | done | Patterned illumination 9-frame, 12/20/20, Wiener SIM recon |
| confocal_3d | done | done | Confocal PSF+pinhole+out-of-focus, 12/20/20, RL deconv |
| lightsheet | done | done | Sheet profile+scatter+stripe, 12/20/20, RL deconv |
| two_photon | done | done | 2P PSF+depth attenuation, 12/20/20, depth-corrected RL |
| cryo_em | done | done | CTF+extreme noise, 12/20/20, Wiener CTF ~17 dB |
| sem | done | done | BSE/SE yield+charging, 12/20/20, NLM denoising ~26 dB |
| tem | done | done | CTF+Beer-Lambert, 12/20/20, Wiener CTF ~19 dB |
| widefield | done | done | Widefield PSF+haze+autofluor, 12/20/20, RL ~27 dB |
| photoacoustic | done | done | Limited-angle Radon+UBP, 12/20/20, UBP baseline |

## Priority 3 — Computational & Advanced

| Modality | Dataset | Benchmark | Notes |
|----------|---------|-----------|-------|
| holography | done | done | Angular spectrum propagation+interference, 12/20/20 |
| ptychography | done | done | ePIE phase retrieval, 12/20/20 |
| lensless | done | done | Coded aperture Wiener deconv, 12/20/20 |
| nerf | pending | pending | Neural radiance fields |
| gaussian_splatting | done | done | 3DGS alpha-blending, 12/20/20 |
| phase_retrieval | in-progress | pending | GS alternating projection |
| fpm | in-progress | pending | Fourier ptychographic microscopy |
| odt | in-progress | pending | Optical diffraction tomography |
| ghost_imaging | in-progress | pending | Computational ghost imaging |

## Priority 4 — Spectroscopy & Remote Sensing

| Modality | Dataset | Benchmark | Notes |
|----------|---------|-----------|-------|
| raman_imaging | pending | pending | Raman spectroscopy |
| ftir_imaging | pending | pending | FTIR imaging |
| sar | pending | pending | Synthetic aperture radar |
| lidar | pending | pending | LiDAR point cloud |
| hyperspectral_remote | pending | pending | Hyperspectral satellite |
| insar | pending | pending | Interferometric SAR |
| multispectral_sat | pending | pending | Multispectral satellite |

## Priority 5 — Nuclear & Particle

| Modality | Dataset | Benchmark | Notes |
|----------|---------|-----------|-------|
| pet_ct | pending | pending | Combined PET/CT |
| pet_mr | pending | pending | Combined PET/MRI |
| spect_ct | pending | pending | Combined SPECT/CT |
| spectral_ct | pending | pending | Spectral/dual-energy CT |
| industrial_ct | pending | pending | Industrial CT |

## Priority 6 — Remaining Modalities

| Modality | Dataset | Benchmark | Notes |
|----------|---------|-----------|-------|
| acoustic_emission | pending | pending | |
| acoustic_microscopy | pending | pending | |
| active_thermography | pending | pending | |
| adaptive_optics | pending | pending | |
| afm | pending | pending | |
| angiography | pending | pending | |
| asl_mri | pending | pending | |
| atom_probe | pending | pending | |
| bioluminescence_tomo | pending | pending | |
| brachytherapy_img | pending | pending | |
| brillouin | pending | pending | |
| cacti | pending | pending | |
| cars | pending | pending | |
| cassi | pending | pending | |
| cathodoluminescence | pending | pending | |
| cest_mri | pending | pending | |
| ceus | pending | pending | |
| clem | pending | pending | |
| coded_exposure | pending | pending | |
| confocal_endomicroscopy | pending | pending | |
| confocal_livecell | pending | pending | |
| coronagraphy | pending | pending | |
| cryo_et | pending | pending | |
| ct_fluorescence | pending | pending | |
| cup | pending | pending | |
| dark_field | pending | pending | |
| desi | pending | pending | |
| dexa | pending | pending | |
| dic | pending | pending | |
| digital_breast_tomo | pending | pending | |
| dna_paint | pending | pending | |
| doppler_ultrasound | pending | pending | |
| dot | pending | pending | |
| ebsd | pending | pending | |
| eddy_current | pending | pending | |
| edx_mapping | pending | pending | |
| eels | pending | pending | |
| eht_imaging | pending | pending | |
| elastography | pending | pending | |
| electron_diffraction | pending | pending | |
| electron_holography | pending | pending | |
| electron_tomography | pending | pending | |
| entangled_photon | pending | pending | |
| event_camera | pending | pending | |
| expansion | pending | pending | |
| fib_sem | pending | pending | |
| flash_lidar | pending | pending | |
| flim | pending | pending | |
| fluoroscopy | pending | pending | |
| fwi | pending | pending | |
| gpr | pending | pending | |
| gravitational_wave | pending | pending | |
| hdr_imaging | pending | pending | |
| impedance_tomo | pending | pending | |
| integral | pending | pending | |
| ism | pending | pending | |
| ivus | pending | pending | |
| lattice_lightsheet | pending | pending | |
| libs | pending | pending | |
| light_field | pending | pending | |
| lucky_imaging | pending | pending | |
| machine_vision | pending | pending | |
| magnetic_particle | pending | pending | |
| maldi_msi | pending | pending | |
| matrix | pending | pending | |
| mfm | pending | pending | |
| minflux | pending | pending | |
| mr_elastography | pending | pending | |
| mr_fingerprinting | pending | pending | |
| mra | pending | pending | |
| mrs | pending | pending | |
| muon_tomo | pending | pending | |
| neutron_diffraction | pending | pending | |
| neutron_tomo | pending | pending | |
| nirs_brain | pending | pending | |
| nsom | pending | pending | |
| ocean_acoustic_tomo | pending | pending | |
| ocean_color | pending | pending | |
| octa | pending | pending | |
| panorama | pending | pending | |
| particle_calorimetry | pending | pending | |
| passive_microwave | pending | pending | |
| phase_contrast | pending | pending | |
| photometric_stereo | pending | pending | |
| polarization | pending | pending | |
| polsar | pending | pending | |
| portal_imaging | pending | pending | |
| proton_radiography | pending | pending | |
| proton_therapy_img | pending | pending | |
| pump_probe | pending | pending | |
| quantum_illumination | pending | pending | |
| radio_astronomy | pending | pending | |
| radio_interferometry | pending | pending | |
| saxs | pending | pending | |
| seismic_tomo | pending | pending | |
| shearography | pending | pending | |
| shg | pending | pending | |
| sims | pending | pending | |
| solar_imaging | pending | pending | |
| sonar | pending | pending | |
| spc | pending | pending | |
| spinning_disk | pending | pending | |
| srs | pending | pending | |
| stem | pending | pending | |
| stm | pending | pending | |
| streak_camera | pending | pending | |
| structured_light | pending | pending | |
| swi | pending | pending | |
| talbot_lau | pending | pending | |
| terahertz | pending | pending | |
| three_photon | pending | pending | |
| tirf | pending | pending | |
| tof_camera | pending | pending | |
| ultrasonic_phased_array | pending | pending | |
| us_mri | pending | pending | |
| waxs | pending | pending | |
| weather_radar | pending | pending | |
| widefield_lowdose | pending | pending | |
| xfel_sfx | pending | pending | |
| xray_crystallography | pending | pending | |
| xray_ndt | pending | pending | |
| xray_radiography | pending | pending | |
| xrf_imaging | pending | pending | |
| xrf_tomo | pending | pending | |
