# Benchmark Dataset State Tracker

Last updated: 2026-03-11T16:00Z

## Status Legend

| State | Description |
|-------|-------------|
| **1) Dataset** | Creating public (≥10), dev (20), hidden (20) + spec.json + true_spec.json + per-sample images |
| **2) Benchmark** | Update https://pwm.platformai.org/benchmark — gallery, data preview, baseline CPU recon |
| **3) Algorithms** | All algorithms tested (1 public sample each): classical CPU + GPU via Modal/server |
| **4) SpecLab** | All algorithm tests integrated into https://pwm.platformai.org/speclab |

Values: `done` / `in-progress` / `pending`

---

## Priority 1 — Core Medical & Imaging

| Modality | 1) Dataset | 2) Benchmark | 3) Algorithms | 4) SpecLab | Notes |
|----------|-----------|-------------|--------------|-----------|-------|
| ct | done | done | in-progress | pending | LoDoPaB-CT, fan-beam Radon, 11/20/20, FBP+TV baseline |
| mri | done | done | in-progress | pending | M4Raw k-space Fourier, 12/20/20, ZF-IFFT baseline |
| pet | done | done | in-progress | pending | Radon+Poisson+attenuation, 12/20/20, FBP ~35 dB |
| ultrasound | done | done | in-progress | pending | PSF+speckle+attenuation, 12/20/20, Wiener baseline |
| oct | done | done | in-progress | pending | Retinal B-scan, PSF+speckle+rolloff, 12/20/20, median ~22 dB |
| mammography | done | done | in-progress | pending | Beer-Lambert+Poisson+scatter, 12/20/20, Wiener+TV ~22 dB |
| cbct | done | done | in-progress | pending | Cone-beam CT, FDK recon, 10/20/20 |
| spect | done | done | in-progress | pending | Radon+attenuation+Poisson+scatter, 12/20/20, FBP baseline |
| fundus | done | done | pending | pending | Defocus PSF+illumination+Poisson-Gaussian, 12/20/20 |
| endoscopy | done | done | pending | pending | Barrel distortion+vignetting, 12/20/20, Wiener ~15 dB |
| fmri | done | done | pending | pending | BOLD+k-space undersampling, 12/20/20, ZF-IFFT baseline |
| diffusion_mri | done | done | pending | pending | ADC contrast, k-space undersampling, 12/20/20 |

## Priority 2 — Microscopy & Optical

| Modality | 1) Dataset | 2) Benchmark | 3) Algorithms | 4) SpecLab | Notes |
|----------|-----------|-------------|--------------|-----------|-------|
| palm_storm | done | done | pending | pending | SMLM Poisson+PSF+readout, 12/20/20, Gaussian fitting ~31 dB |
| sted | done | done | pending | pending | STED depletion PSF+Poisson, 12/20/20, RL deconv baseline |
| sim | done | done | pending | pending | Patterned illumination 9-frame, 12/20/20, Wiener SIM recon |
| confocal_3d | done | done | pending | pending | Confocal PSF+pinhole+out-of-focus, 12/20/20, RL deconv |
| lightsheet | done | done | pending | pending | Sheet profile+scatter+stripe, 12/20/20, RL deconv |
| two_photon | done | done | pending | pending | 2P PSF+depth attenuation, 12/20/20, depth-corrected RL |
| cryo_em | done | done | pending | pending | CTF+extreme noise, 12/20/20, Wiener CTF ~17 dB |
| sem | done | done | pending | pending | BSE/SE yield+charging, 12/20/20, NLM denoising ~26 dB |
| tem | done | done | pending | pending | CTF+Beer-Lambert, 12/20/20, Wiener CTF ~19 dB |
| widefield | done | done | pending | pending | Widefield PSF+haze+autofluor, 12/20/20, RL ~27 dB |
| photoacoustic | done | done | pending | pending | Limited-angle Radon+UBP, 12/20/20, UBP baseline |

## Priority 3 — Computational & Advanced

| Modality | 1) Dataset | 2) Benchmark | 3) Algorithms | 4) SpecLab | Notes |
|----------|-----------|-------------|--------------|-----------|-------|
| holography | done | done | pending | pending | Angular spectrum propagation+interference, 12/20/20 |
| ptychography | done | done | pending | pending | ePIE phase retrieval, 12/20/20 |
| lensless | done | done | pending | pending | Coded aperture Wiener deconv, 12/20/20 |
| gaussian_splatting | done | done | pending | pending | 3DGS alpha-blending, 12/20/20 |
| phase_retrieval | done | done | pending | pending | GS alternating projection, 12/20/20 |
| fpm | done | done | pending | pending | Fourier ptychographic microscopy, 12/20/20 |
| odt | done | done | pending | pending | Optical diffraction tomography, 12/20/20 |
| ghost_imaging | done | done | pending | pending | Computational ghost imaging, 12/20/20 |
| nerf | pending | pending | pending | pending | Neural radiance fields — complex scene generation |

## Priority 4 — Spectroscopy & Remote Sensing

| Modality | 1) Dataset | 2) Benchmark | 3) Algorithms | 4) SpecLab | Notes |
|----------|-----------|-------------|--------------|-----------|-------|
| raman_imaging | done | done | pending | pending | Raman spectroscopy, 12/20/20 |
| ftir_imaging | done | done | pending | pending | FTIR imaging, 12/20/20 |
| sar | done | done | pending | pending | SAR, 12/20/20, Lee+MF baseline ~15 dB |
| lidar | done | done | pending | pending | LiDAR, 12/20/20, bilateral+range ~12 dB |
| hyperspectral_remote | done | done | pending | pending | Hyperspectral, 12/20/20, ATCOR+Wiener ~21 dB |
| insar | done | done | pending | pending | InSAR, 12/20/20, Goldstein unwrap ~18 dB |
| multispectral_sat | pending | pending | pending | pending | Multispectral satellite imaging |

## Priority 5 — Nuclear & Particle / Multimodality

| Modality | 1) Dataset | 2) Benchmark | 3) Algorithms | 4) SpecLab | Notes |
|----------|-----------|-------------|--------------|-----------|-------|
| pet_ct | done | done | pending | pending | Combined PET/CT, 12/20/20, joint Radon+attenuation |
| pet_mr | pending | pending | pending | pending | Combined PET/MRI |
| spect_ct | done | pending | pending | pending | Combined SPECT/CT, 6/20/20, SPECT+CT attenuation |
| spectral_ct | pending | pending | pending | pending | Spectral/dual-energy CT |
| industrial_ct | pending | pending | pending | pending | Industrial CT |

## Priority 6 — Remaining Modalities

| Modality | 1) Dataset | 2) Benchmark | 3) Algorithms | 4) SpecLab | Notes |
|----------|-----------|-------------|--------------|-----------|-------|
| acoustic_emission | pending | pending | pending | pending | |
| acoustic_microscopy | pending | pending | pending | pending | |
| active_thermography | pending | pending | pending | pending | |
| adaptive_optics | pending | pending | pending | pending | |
| afm | pending | pending | pending | pending | |
| angiography | pending | pending | pending | pending | |
| asl_mri | pending | pending | pending | pending | |
| atom_probe | pending | pending | pending | pending | |
| bioluminescence_tomo | pending | pending | pending | pending | |
| brachytherapy_img | pending | pending | pending | pending | |
| brillouin | pending | pending | pending | pending | |
| cacti | pending | pending | pending | pending | |
| cars | pending | pending | pending | pending | |
| cassi | pending | pending | pending | pending | |
| cathodoluminescence | pending | pending | pending | pending | |
| cest_mri | pending | pending | pending | pending | |
| ceus | pending | pending | pending | pending | |
| clem | pending | pending | pending | pending | |
| coded_exposure | pending | pending | pending | pending | |
| confocal_endomicroscopy | pending | pending | pending | pending | |
| confocal_livecell | pending | pending | pending | pending | |
| coronagraphy | pending | pending | pending | pending | |
| cryo_et | pending | pending | pending | pending | |
| ct_fluorescence | pending | pending | pending | pending | |
| cup | pending | pending | pending | pending | |
| dark_field | pending | pending | pending | pending | |
| desi | pending | pending | pending | pending | |
| dexa | pending | pending | pending | pending | |
| dic | pending | pending | pending | pending | |
| digital_breast_tomo | pending | pending | pending | pending | |
| dna_paint | pending | pending | pending | pending | |
| doppler_ultrasound | pending | pending | pending | pending | |
| dot | pending | pending | pending | pending | |
| ebsd | pending | pending | pending | pending | |
| eddy_current | pending | pending | pending | pending | |
| edx_mapping | pending | pending | pending | pending | |
| eels | pending | pending | pending | pending | |
| eht_imaging | pending | pending | pending | pending | |
| elastography | pending | pending | pending | pending | |
| electron_diffraction | pending | pending | pending | pending | |
| electron_holography | pending | pending | pending | pending | |
| electron_tomography | pending | pending | pending | pending | |
| entangled_photon | pending | pending | pending | pending | |
| event_camera | pending | pending | pending | pending | |
| expansion | pending | pending | pending | pending | |
| fib_sem | pending | pending | pending | pending | |
| flash_lidar | pending | pending | pending | pending | |
| flim | pending | pending | pending | pending | |
| fluoroscopy | pending | pending | pending | pending | |
| fwi | pending | pending | pending | pending | |
| gpr | pending | pending | pending | pending | |
| gravitational_wave | pending | pending | pending | pending | |
| hdr_imaging | pending | pending | pending | pending | |
| impedance_tomo | pending | pending | pending | pending | |
| integral | pending | pending | pending | pending | |
| ism | pending | pending | pending | pending | |
| ivus | pending | pending | pending | pending | |
| lattice_lightsheet | pending | pending | pending | pending | |
| libs | pending | pending | pending | pending | |
| light_field | pending | pending | pending | pending | |
| lucky_imaging | pending | pending | pending | pending | |
| machine_vision | pending | pending | pending | pending | |
| magnetic_particle | pending | pending | pending | pending | |
| maldi_msi | pending | pending | pending | pending | |
| matrix | pending | pending | pending | pending | |
| mfm | pending | pending | pending | pending | |
| minflux | pending | pending | pending | pending | |
| mr_elastography | pending | pending | pending | pending | |
| mr_fingerprinting | pending | pending | pending | pending | |
| mra | pending | pending | pending | pending | |
| mrs | pending | pending | pending | pending | |
| muon_tomo | pending | pending | pending | pending | |
| neutron_diffraction | pending | pending | pending | pending | |
| neutron_tomo | pending | pending | pending | pending | |
| nirs_brain | pending | pending | pending | pending | |
| nsom | pending | pending | pending | pending | |
| ocean_acoustic_tomo | pending | pending | pending | pending | |
| ocean_color | pending | pending | pending | pending | |
| octa | pending | pending | pending | pending | |
| panorama | pending | pending | pending | pending | |
| particle_calorimetry | pending | pending | pending | pending | |
| passive_microwave | pending | pending | pending | pending | |
| phase_contrast | pending | pending | pending | pending | |
| photometric_stereo | pending | pending | pending | pending | |
| polarization | pending | pending | pending | pending | |
| polsar | pending | pending | pending | pending | |
| portal_imaging | pending | pending | pending | pending | |
| proton_radiography | pending | pending | pending | pending | |
| proton_therapy_img | pending | pending | pending | pending | |
| pump_probe | pending | pending | pending | pending | |
| quantum_illumination | pending | pending | pending | pending | |
| radio_astronomy | pending | pending | pending | pending | |
| radio_interferometry | pending | pending | pending | pending | |
| saxs | pending | pending | pending | pending | |
| seismic_tomo | pending | pending | pending | pending | |
| shearography | pending | pending | pending | pending | |
| shg | pending | pending | pending | pending | |
| sims | pending | pending | pending | pending | |
| solar_imaging | pending | pending | pending | pending | |
| sonar | pending | pending | pending | pending | |
| spc | pending | pending | pending | pending | |
| spinning_disk | pending | pending | pending | pending | |
| srs | pending | pending | pending | pending | |
| stem | pending | pending | pending | pending | |
| stm | pending | pending | pending | pending | |
| streak_camera | pending | pending | pending | pending | |
| structured_light | pending | pending | pending | pending | |
| swi | pending | pending | pending | pending | |
| talbot_lau | pending | pending | pending | pending | |
| terahertz | pending | pending | pending | pending | |
| three_photon | pending | pending | pending | pending | |
| tirf | pending | pending | pending | pending | |
| tof_camera | pending | pending | pending | pending | |
| ultrasonic_phased_array | pending | pending | pending | pending | |
| us_mri | pending | pending | pending | pending | |
| waxs | pending | pending | pending | pending | |
| weather_radar | pending | pending | pending | pending | |
| widefield_lowdose | pending | pending | pending | pending | |
| xfel_sfx | pending | pending | pending | pending | |
| xray_crystallography | pending | pending | pending | pending | |
| xray_ndt | pending | pending | pending | pending | |
| xray_radiography | pending | pending | pending | pending | |
| xrf_imaging | pending | pending | pending | pending | |
| xrf_tomo | pending | pending | pending | pending | |
