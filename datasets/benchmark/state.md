# Benchmark Algorithm Test State

Last updated: 2026-03-11 — 166 modalities with datasets, 168 with YAML configs

## Legend
- `done (X.XX dB)`: tested, PSNR/SSIM recorded in benchmark run
- `importable`: module implemented in pwm_core, ready to run
- `pending`: module not yet implemented
- `reference`: algorithm from https://pwm.platformai.org/benchmark leaderboard

Reference leaderboard: https://pwm.platformai.org/benchmark
Total algorithms on leaderboard: 1,367 across 168 modalities

## Quick Status Table

| Modality | Dataset | Tested | Best PSNR | YAML Algos | Leaderboard Ref | Speclab |
|----------|---------|--------|-----------|------------|-----------------|---------|
| acoustic_emission | done | 1 | 20.2 dB | 3 | — | pending |
| acoustic_microscopy | done | 1 | 10.0 dB | 3 | — | pending |
| active_thermography | done | 1 | 6.5 dB | 3 | — | pending |
| adaptive_optics | done | 1 | 100.0 dB | 3 | — | pending |
| afm | done | 3 | 31.3 dB | 3 | 7 | pending |
| angiography | done | 2 | 12.9 dB | 3 | — | pending |
| asl_mri | done | 1 | 2.7 dB | 3 | 5 | pending |
| atom_probe | done | 1 | 40.2 dB | 3 | — | pending |
| bioluminescence_tomo | done | 1 | 13.3 dB | 3 | — | pending |
| brachytherapy_img | done | 1 | 20.5 dB | 3 | — | pending |
| brillouin | done | 1 | 35.8 dB | 3 | — | pending |
| cacti | done | 5 | 11.5 dB | 4 | 8 | pending |
| cars | done | 1 | 14.2 dB | 3 | — | pending |
| cassi | pending | 1 | 2.6 dB | 4 | 12 | pending |
| cathodoluminescence | done | 1 | 28.9 dB | 3 | — | pending |
| cbct | done | 3 | 15.2 dB | 3 | 9 | pending |
| cest_mri | done | 1 | 31.0 dB | 3 | 6 | pending |
| ceus | done | 1 | 24.5 dB | 3 | — | pending |
| clem | done | 1 | 17.0 dB | 3 | — | pending |
| coded_exposure | done | 1 | 19.9 dB | 3 | — | pending |
| confocal_3d | done | 6 | 27.3 dB | 4 | 7 | pending |
| confocal_endomicroscopy | done | 1 | 34.0 dB | 3 | — | pending |
| confocal_livecell | done | 5 | 32.3 dB | 4 | — | pending |
| coronagraphy | done | 1 | 25.2 dB | 3 | — | pending |
| cryo_em | done | 2 | 19.2 dB | 3 | 10 | pending |
| cryo_et | done | 3 | 13.2 dB | 3 | — | pending |
| ct | done | 6 | 13.8 dB | 4 | 24 | pending |
| ct_fluorescence | done | 1 | -37.6 dB | 3 | — | pending |
| cup | done | 1 | -2.3 dB | 3 | — | pending |
| dark_field | done | 3 | 25.1 dB | 3 | — | pending |
| desi | done | 1 | 15.1 dB | 3 | — | pending |
| dexa | done | 1 | 9.5 dB | 3 | — | pending |
| dic | done | 3 | 15.6 dB | 3 | — | pending |
| diffusion_mri | done | 1 | 11.3 dB | 3 | 7 | pending |
| digital_breast_tomo | done | 1 | -36.0 dB | 3 | — | pending |
| dna_paint | done | 3 | 28.5 dB | 3 | — | pending |
| doppler_ultrasound | pending | 0 | — | 3 | — | pending |
| dot | pending | 0 | — | 3 | — | pending |
| ebsd | done | 1 | 21.8 dB | 3 | — | pending |
| eddy_current | done | 1 | 4.8 dB | 3 | — | pending |
| edx_mapping | done | 2 | 22.0 dB | 3 | — | pending |
| eels | done | 1 | 24.6 dB | 4 | 6 | pending |
| eht_imaging | done | 1 | 11.3 dB | 3 | 5 | pending |
| elastography | done | 1 | 5.7 dB | 3 | — | pending |
| electron_diffraction | done | 2 | 42.0 dB | 3 | — | pending |
| electron_holography | done | 2 | 9.5 dB | 3 | — | pending |
| electron_tomography | done | 2 | 25.1 dB | 3 | — | pending |
| endoscopy | done | 3 | 11.8 dB | 3 | — | pending |
| entangled_photon | done | 1 | 31.8 dB | 3 | — | pending |
| event_camera | done | 1 | 7.3 dB | 3 | — | pending |
| expansion | done | 3 | 33.9 dB | 3 | — | pending |
| fib_sem | done | 3 | 28.1 dB | 3 | — | pending |
| flash_lidar | done | 1 | 4.3 dB | 3 | — | pending |
| flim | done | 2 | 30.7 dB | 4 | — | pending |
| fluoroscopy | done | 2 | 43.5 dB | 3 | — | pending |
| fmri | done | 1 | 4.9 dB | 3 | 7 | pending |
| fpm | done | 2 | 16.9 dB | 4 | — | pending |
| ftir_imaging | done | 1 | 14.8 dB | 3 | — | pending |
| fundus | done | 4 | 35.9 dB | 3 | — | pending |
| fwi | done | 1 | 8.7 dB | 3 | — | pending |
| gaussian_splatting | done | 5 | inf dB | 4 | — | pending |
| ghost_imaging | done | 1 | 6.6 dB | 3 | 6 | pending |
| gpr | done | 1 | 10.6 dB | 3 | — | pending |
| gravitational_wave | done | 1 | 100.0 dB | 3 | — | pending |
| hdr_imaging | done | 1 | 36.8 dB | 3 | — | pending |
| holography | done | 5 | 14.9 dB | 4 | 6 | pending |
| hyperspectral_remote | done | 1 | 29.1 dB | 3 | — | pending |
| impedance_tomo | done | 1 | 11.2 dB | 3 | — | pending |
| industrial_ct | done | 1 | 20.3 dB | 3 | — | pending |
| insar | done | 1 | 31.8 dB | 3 | 4 | pending |
| integral | done | 2 | 40.0 dB | 4 | — | pending |
| ism | done | 3 | 3.1 dB | 3 | — | pending |
| ivus | done | 1 | 19.8 dB | 3 | — | pending |
| lattice_lightsheet | done | 3 | 25.1 dB | 3 | — | pending |
| lensless | done | 5 | 11.9 dB | 4 | — | pending |
| libs | done | 1 | 18.0 dB | 3 | — | pending |
| lidar | done | 1 | 32.6 dB | 3 | — | pending |
| light_field | done | 5 | 27.3 dB | 4 | — | pending |
| lightsheet | done | 7 | 20.0 dB | 4 | — | pending |
| lucky_imaging | done | 1 | 29.2 dB | 3 | — | pending |
| machine_vision | done | 1 | 26.5 dB | 3 | — | pending |
| magnetic_particle | done | 1 | 26.5 dB | 3 | — | pending |
| maldi_msi | done | 1 | 26.3 dB | 3 | — | pending |
| mammography | done | 2 | 20.9 dB | 3 | — | pending |
| matrix | done | 1 | 22.0 dB | 4 | — | pending |
| mfm | done | 3 | 34.3 dB | 3 | — | pending |
| minflux | done | 3 | 29.5 dB | 3 | — | pending |
| mr_elastography | done | 1 | 6.0 dB | 3 | 5 | pending |
| mr_fingerprinting | done | 1 | 1.8 dB | 3 | 6 | pending |
| mra | done | 1 | 12.1 dB | 3 | 6 | pending |
| mri | done | 3 | 13.0 dB | 4 | 27 | pending |
| mrs | done | 1 | 1.9 dB | 3 | — | pending |
| multispectral_sat | done | 1 | 10.8 dB | 3 | — | pending |
| muon_tomo | done | 2 | 5.2 dB | 3 | — | pending |
| nerf | done | 2 | 29.0 dB | 4 | 6 | pending |
| neutron_diffraction | done | 1 | 8.5 dB | 3 | — | pending |
| neutron_tomo | done | 2 | 4.3 dB | 3 | — | pending |
| nirs_brain | done | 1 | 14.5 dB | 3 | — | pending |
| nsom | done | 3 | 22.3 dB | 3 | — | pending |
| ocean_acoustic_tomo | done | 1 | 5.6 dB | 3 | — | pending |
| ocean_color | done | 1 | 44.1 dB | 3 | — | pending |
| oct | done | 6 | 23.5 dB | 4 | 8 | pending |
| octa | done | 2 | 16.8 dB | 3 | — | pending |
| odt | done | 1 | 25.5 dB | 3 | — | pending |
| palm_storm | done | 3 | 32.4 dB | 3 | — | pending |
| panorama | done | 2 | 15.1 dB | 4 | — | pending |
| particle_calorimetry | done | 1 | 36.2 dB | 3 | — | pending |
| passive_microwave | done | 1 | 9.2 dB | 3 | — | pending |
| pet | done | 4 | 33.1 dB | 3 | 10 | pending |
| pet_ct | done | 1 | 13.0 dB | 3 | — | pending |
| pet_mr | done | 1 | 11.0 dB | 3 | — | pending |
| phase_contrast | done | 3 | 45.6 dB | 3 | — | pending |
| phase_retrieval | done | 2 | 12.6 dB | 4 | — | pending |
| photoacoustic | done | 2 | 19.1 dB | 4 | 6 | pending |
| photometric_stereo | done | 1 | 29.0 dB | 3 | — | pending |
| polarization | done | 2 | 15.8 dB | 3 | — | pending |
| polsar | done | 1 | 3.5 dB | 3 | 13 | pending |
| portal_imaging | done | 1 | 10.5 dB | 3 | — | pending |
| proton_radiography | done | 2 | 10.9 dB | 3 | — | pending |
| proton_therapy_img | done | 1 | 17.8 dB | 3 | — | pending |
| ptychography | done | 3 | 21.0 dB | 4 | 6 | pending |
| pump_probe | done | 1 | 18.2 dB | 3 | — | pending |
| quantum_illumination | done | 1 | 20.2 dB | 3 | — | pending |
| radio_astronomy | done | 1 | 16.1 dB | 3 | 7 | pending |
| radio_interferometry | done | 1 | 23.2 dB | 3 | 6 | pending |
| raman_imaging | done | 1 | 14.1 dB | 3 | — | pending |
| sar | done | 2 | 17.3 dB | 3 | 13 | pending |
| saxs | done | 1 | 8.4 dB | 3 | — | pending |
| seismic_tomo | done | 1 | 9.0 dB | 3 | — | pending |
| sem | done | 2 | 23.2 dB | 3 | 8 | pending |
| shearography | done | 1 | 8.0 dB | 3 | — | pending |
| shg | done | 3 | 23.0 dB | 3 | — | pending |
| sim | done | 3 | 21.6 dB | 4 | — | pending |
| sims | done | 1 | 20.5 dB | 3 | — | pending |
| solar_imaging | done | 1 | 28.4 dB | 3 | — | pending |
| sonar | done | 1 | 10.3 dB | 3 | — | pending |
| spc | pending | 1 | 6.8 dB | 4 | — | pending |
| spect | done | 3 | 30.0 dB | 3 | 10 | pending |
| spect_ct | done | 1 | 11.4 dB | 3 | — | pending |
| spectral_ct | done | 1 | 12.3 dB | 3 | — | pending |
| spinning_disk | done | 3 | 30.6 dB | 3 | — | pending |
| srs | done | 1 | 29.1 dB | 3 | — | pending |
| sted | done | 3 | 25.0 dB | 3 | — | pending |
| stem | done | 2 | 31.0 dB | 3 | 6 | pending |
| stm | done | 3 | 23.3 dB | 3 | 6 | pending |
| streak_camera | done | 1 | 14.3 dB | 3 | — | pending |
| structured_light | done | 1 | 8.0 dB | 3 | — | pending |
| swi | done | 1 | 1.9 dB | 3 | 5 | pending |
| talbot_lau | done | 1 | 6.6 dB | 3 | — | pending |
| tem | done | 1 | 25.3 dB | 3 | 7 | pending |
| terahertz | done | 1 | 37.1 dB | 3 | — | pending |
| three_photon | done | 3 | 20.8 dB | 3 | — | pending |
| tirf | done | 2 | 31.2 dB | 3 | — | pending |
| tof_camera | done | 1 | 42.0 dB | 3 | — | pending |
| two_photon | done | 3 | 33.8 dB | 3 | — | pending |
| ultrasonic_phased_array | done | 1 | 29.6 dB | 3 | — | pending |
| ultrasound | done | 2 | 14.6 dB | 3 | 8 | pending |
| us_mri | done | 1 | 7.6 dB | 3 | — | pending |
| waxs | done | 1 | 20.6 dB | 3 | — | pending |
| weather_radar | done | 1 | 26.9 dB | 3 | — | pending |
| widefield | done | 5 | 25.0 dB | 4 | — | pending |
| widefield_lowdose | done | 3 | 29.0 dB | 4 | — | pending |
| xfel_sfx | done | 1 | 24.1 dB | 3 | — | pending |
| xray_crystallography | done | 1 | 22.4 dB | 3 | 6 | pending |
| xray_ndt | done | 1 | 16.7 dB | 3 | — | pending |
| xray_radiography | done | 2 | 26.3 dB | 3 | — | pending |
| xrf_imaging | done | 1 | 22.1 dB | 3 | — | pending |
| xrf_tomo | done | 1 | 15.6 dB | 3 | — | pending |

**Summary:** 164/168 datasets done, 301 solver tests recorded across 166/166 modalities with datasets

---

## Detailed Algorithm Test Results by Modality

Each modality shows three sections:
1. **Tested** — results from our benchmark runs (PSNR/SSIM recorded)
2. **YAML Solvers** — algorithms defined in config (implementation status)
3. **Leaderboard Reference** — top algorithms from pwm.platformai.org/benchmark

### Astronomy & Space Imaging

#### coronagraphy — Stellar Coronagraphy

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 25.17 | 0.2028 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | DL-SpeckleNull | deep_learning | pending | — | — | — | YAML config |

#### eht_imaging — Event Horizon Telescope (EHT) Imaging

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 11.29 | 0.0394 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | EHT-PRIMO | deep_learning | pending | — | — | — | YAML config |
| #1 | EHT-PRIMO | deep_learning | reference | 36.2 | 0.941 | — | pwm.platformai.org |
| #2 | ngEHT-Net | deep_learning | reference | 34.5 | 0.925 | — | pwm.platformai.org |
| #3 | SMILI | traditional | reference | 32.8 | 0.907 | — | pwm.platformai.org |
| #4 | DIFMAP-CLEAN | traditional | reference | 31.1 | 0.887 | — | pwm.platformai.org |
| #5 | eht-imaging MEM | traditional | reference | 29.4 | 0.864 | — | pwm.platformai.org |

#### lucky_imaging — Lucky Imaging

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 29.22 | 0.9746 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | Lucky-DL | deep_learning | pending | — | — | — | YAML config |

#### solar_imaging — Solar EUV/X-ray Imaging

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 28.37 | 0.9958 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | SolarNet | deep_learning | pending | — | — | — | YAML config |

### Broader Experimental Science

#### acoustic_emission — Acoustic Emission Testing (AE)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 20.21 | 0.0741 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | DeepAE-Net | deep_learning | pending | — | — | — | YAML config |

#### adaptive_optics — Adaptive Optics (AO) Imaging

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 100.00 | 1.0000 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | Deep-AO | traditional | pending | — | — | — | YAML config |

#### bioluminescence_tomo — Bioluminescence Tomography (BLT)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 13.30 | 0.3431 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | BLT-Net | deep_learning | pending | — | — | — | YAML config |

#### fwi — Full-Waveform Inversion (FWI)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 8.73 | 0.0125 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | InversionNet | deep_learning | pending | — | — | — | YAML config |

#### gravitational_wave — Gravitational Wave Detection

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 100.00 | 0.8666 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | GW-DL (PyCBC-ML) | deep_learning | pending | — | — | — | YAML config |

#### impedance_tomo — Electrical Impedance Tomography (EIT)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 11.20 | 0.3124 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | EIT-Net | deep_learning | pending | — | — | — | YAML config |

#### magnetic_particle — Magnetic Particle Imaging (MPI)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 26.49 | 0.9576 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | MPI-Net | deep_learning | pending | — | — | — | YAML config |

#### ocean_acoustic_tomo — Ocean Acoustic Tomography

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 5.62 | 0.6714 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | OAT-Net | deep_learning | pending | — | — | — | YAML config |

#### particle_calorimetry — Particle Calorimetry

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 36.19 | 0.7914 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | CaloDiffusion | deep_learning | pending | — | — | — | YAML config |

#### radio_astronomy — Radio Aperture Synthesis

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 16.05 | 0.2876 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | RadioAST-DL | deep_learning | pending | — | — | — | YAML config |
| #1 | Radio-FM | deep_learning | reference | 34.8 | 0.930 | — | pwm.platformai.org |
| #2 | CLEAN-Net | deep_learning | reference | 33.1 | 0.914 | — | pwm.platformai.org |
| #3 | MS-CLEAN | traditional | reference | 31.5 | 0.896 | — | pwm.platformai.org |
| #4 | WSCLEAN | traditional | reference | 29.8 | 0.875 | — | pwm.platformai.org |
| #5 | CLEAN (Hogbom) | traditional | reference | 28.2 | 0.852 | — | pwm.platformai.org |
| #6 | CLEAN (Clark) | traditional | reference | 26.5 | 0.828 | — | pwm.platformai.org |
| #7 | MEM (Maximum Entropy) | traditional | reference | 24.9 | 0.803 | — | pwm.platformai.org |

#### seismic_tomo — Seismic Tomography

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 9.05 | 0.5099 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | SeisInversion-Net | deep_learning | pending | — | — | — | YAML config |

### Coherent Imaging

#### holography — Digital Holographic Microscopy

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | PhaseNet | deep_learning | **done** | 14.86 | 0.9212 | 0.36 | benchmark run |
| 2 | PhaseNet | deep_learning | **done** | 14.86 | 0.9212 | 0.13 | benchmark run |
| 3 | PhaseNet | deep_learning | **done** | 14.86 | 0.9212 | 0.15 | benchmark run |
| 4 | sqrt(Intensity) Amplitude | traditional | **done** | -20.07 | 0.0003 | 0.00 | benchmark run |
| 5 | Angular Spectrum | traditional | **done** | 4.90 | 0.0036 | 0.48 | benchmark run |
| #1 | Holo-FM | deep_learning | reference | 36.8 | 0.944 | — | pwm.platformai.org |
| #2 | HoloNet | deep_learning | reference | 35.1 | 0.929 | — | pwm.platformai.org |
| #3 | U-Net-Holo | deep_learning | reference | 33.5 | 0.913 | — | pwm.platformai.org |
| #4 | Phase Retrieval (HIO) | traditional | reference | 31.8 | 0.893 | — | pwm.platformai.org |
| #5 | Angular Spectrum | traditional | reference | 30.0 | 0.870 | — | pwm.platformai.org |
| #6 | Fresnel Propagation | traditional | reference | 28.2 | 0.845 | — | pwm.platformai.org |

#### odt — Optical Diffraction Tomography (ODT)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 25.46 | 0.9509 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | ODT-Net (PhaseNet) | deep_learning | pending | — | — | — | YAML config |

#### phase_retrieval — Coherent Diffractive Imaging / Phase Retrieval

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 12.51 | -0.1670 | 0.00 | benchmark run |
| 2 | HIO | traditional | **done** | 12.55 | 0.3297 | 9.29 | benchmark run |
| 3 | RAAR | deep_learning | pending | — | — | — | YAML config |
| 4 | prDeep | deep_learning | pending | — | — | — | YAML config |
| 5 | prDeep | deep_learning | pending | — | — | — | YAML config |

#### ptychography — Ptychographic Imaging

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 20.97 | 0.2841 | 0.00 | benchmark run |
| 2 | Phase Baseline (precomputed) | traditional | **done** | 10.46 | -0.0059 | 0.00 | benchmark run |
| 3 | ePIE | traditional | **done** | 11.66 | 0.5129 | 0.94 | benchmark run |
| #1 | PtychoNN | deep_learning | reference | 39.5 | 0.961 | — | pwm.platformai.org |
| #2 | PtychoFormer | deep_learning | reference | 37.8 | 0.946 | — | pwm.platformai.org |
| #3 | PtychoDL | deep_learning | reference | 36.2 | 0.930 | — | pwm.platformai.org |
| #4 | ePIE (electron) | traditional | reference | 34.5 | 0.912 | — | pwm.platformai.org |
| #5 | PIE (sequential) | traditional | reference | 32.8 | 0.892 | — | pwm.platformai.org |
| #6 | DM (Difference Map) | traditional | reference | 31.0 | 0.869 | — | pwm.platformai.org |

#### talbot_lau — Talbot-Lau X-ray Grating Interferometry

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 6.58 | 0.1206 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | Talbot-Net | deep_learning | pending | — | — | — | YAML config |

### Compressive Imaging

#### cacti — Coded Aperture Compressive Temporal Imaging (CACTI)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | EfficientSCI | deep_learning | **done** | 11.54 | 0.1521 | 0.36 | benchmark run |
| 2 | GAP-TV | traditional | **done** | 4.01 | 0.1216 | 0.43 | benchmark run |
| 3 | Mask Division Baseline | traditional | **done** | 7.01 | 0.3554 | 0.01 | benchmark run |
| 4 | EfficientSCI-T | deep_learning | **done** | 3.65 | -0.0040 | 0.05 | benchmark run |
| 5 | GAP-TV | traditional | **done** | 2.15 | 0.0242 | 0.00 | benchmark run |
| #1 | HiSViT-9 | deep_learning | reference | 33.5 | 0.188 | — | pwm.platformai.org |
| #2 | EfficientSCI | deep_learning | reference | 32.8 | 0.189 | — | pwm.platformai.org |
| #3 | EfficientSCI-T | deep_learning | reference | 32.2 | 0.185 | — | pwm.platformai.org |
| #4 | ELP-Unfolding | deep_learning | reference | 31.6 | 0.191 | — | pwm.platformai.org |
| #5 | STFormer | deep_learning | reference | 31.0 | 0.180 | — | pwm.platformai.org |
| #6 | GAP-TV | traditional | reference | 29.8 | 0.165 | — | pwm.platformai.org |
| #7 | ADMM-TV | traditional | reference | 28.5 | 0.151 | — | pwm.platformai.org |
| #8 | TwIST-TV | traditional | reference | 27.2 | 0.138 | — | pwm.platformai.org |

#### cassi — Coded Aperture Snapshot Spectral Imaging (CASSI)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | GAP-TV | traditional | **done** | 2.58 | -0.0246 | 0.00 | benchmark run |
| #1 | MST++ | deep_learning | reference | — | 0.652 | — | pwm.platformai.org |
| #2 | SSR-L | deep_learning | reference | — | 0.626 | — | pwm.platformai.org |
| #3 | HDNet | deep_learning | reference | — | 0.618 | — | pwm.platformai.org |
| #4 | DGSMP | deep_learning | reference | — | 0.610 | — | pwm.platformai.org |
| #5 | GAP-Net | deep_learning | reference | — | 0.604 | — | pwm.platformai.org |
| #6 | MST-L | deep_learning | reference | — | 0.550 | — | pwm.platformai.org |
| #7 | GAP-TV | traditional | reference | — | 0.593 | — | pwm.platformai.org |
| #8 | ADMM-Net | traditional | reference | — | 0.578 | — | pwm.platformai.org |
| #9 | TwIST | traditional | reference | — | 0.561 | — | pwm.platformai.org |
| #10 | ISTA-Net | traditional | reference | — | 0.543 | — | pwm.platformai.org |
| #11 | FISTA-TV | traditional | reference | — | 0.525 | — | pwm.platformai.org |
| #12 | ELP-Unfolding | deep_learning | reference | — | 0.515 | — | pwm.platformai.org |

#### matrix — Generic Matrix Sensing

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 22.04 | 0.6949 | 0.00 | benchmark run |
| 2 | Tikhonov / FISTA-L2 | traditional | pending | — | — | — | YAML config |

#### spc — Single-Pixel Camera (SPC)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | TVAL3 | traditional | **done** | 6.83 | 0.0161 | 0.00 | benchmark run |

### Computational Optics

#### integral — Integral Photography

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 40.02 | 0.9990 | 0.00 | benchmark run |
| 2 | Depth Estimation | traditional | **done** | 33.40 | 0.9013 | 0.00 | benchmark run |
| 3 | DIBR | deep_learning | pending | — | — | — | YAML config |
| 4 | EPINet | deep_learning | pending | — | — | — | YAML config |
| 5 | EPINet | deep_learning | pending | — | — | — | YAML config |

#### light_field — Light Field Imaging

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | LFBM5D | deep_learning | **done** | 4.27 | 0.0170 | 0.00 | benchmark run |
| 2 | LFSSR | deep_learning | **done** | 16.28 | 0.1186 | 2.09 | benchmark run |
| 3 | Precomputed Baseline | traditional | **done** | 27.26 | 0.9439 | 0.00 | benchmark run |
| 4 | LFSSR | deep_learning | **done** | 16.28 | 0.1186 | 2.03 | benchmark run |
| 5 | Shift-and-Sum | traditional | **done** | 16.28 | 0.1186 | 2.32 | benchmark run |

### Computational Photography

#### coded_exposure — Coded Exposure / Flutter Shutter

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 19.86 | 0.8073 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | FlowNet-Coded | deep_learning | pending | — | — | — | YAML config |

#### event_camera — Event Camera / Dynamic Vision Sensor (DVS)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 7.30 | 0.0574 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | E2VID+ | deep_learning | pending | — | — | — | YAML config |

#### hdr_imaging — High Dynamic Range (HDR) Imaging

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 36.82 | 0.8232 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | HDR-Net | deep_learning | pending | — | — | — | YAML config |

#### lensless — Lensless (Diffuser Camera) Imaging

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | FlatNet | deep_learning | **done** | 0.48 | 0.0001 | 0.69 | benchmark run |
| 2 | FlatNet | deep_learning | **done** | 0.48 | 0.0001 | 0.43 | benchmark run |
| 3 | FlatNet-Lite | deep_learning | **done** | 0.48 | 0.0001 | 0.41 | benchmark run |
| 4 | ADMM-TV | traditional | **done** | 11.92 | 0.5896 | 0.00 | benchmark run |
| 5 | Wiener Deconvolution | traditional | **done** | 11.81 | 0.0031 | 0.01 | benchmark run |

#### panorama — Panorama Multi-Focus Fusion

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 15.07 | 0.6418 | 0.00 | benchmark run |
| 2 | Laplacian Pyramid Fusion | traditional | **done** | 14.61 | 0.0520 | 0.05 | benchmark run |

### Depth Imaging

#### flash_lidar — Flash LiDAR

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 4.25 | -0.6337 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | FlashLiDAR-Net | deep_learning | pending | — | — | — | YAML config |

#### lidar — LiDAR Scanner

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 32.56 | 0.9955 | 0.00 | benchmark run |
| 2 | FISTA-L2 (depth) | traditional | pending | — | — | — | YAML config |
| 3 | PointNeXt | deep_learning | pending | — | — | — | YAML config |
| 4 | PointNet++ | deep_learning | pending | — | — | — | YAML config |

#### photometric_stereo — Photometric Stereo

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 29.01 | 0.9583 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | PS-FCN | deep_learning | pending | — | — | — | YAML config |

#### structured_light — Structured-Light Depth Camera

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 7.96 | -0.0287 | 0.00 | benchmark run |
| 2 | FISTA-L2 (phase unwrap) | traditional | pending | — | — | — | YAML config |
| 3 | SL-Net | deep_learning | pending | — | — | — | YAML config |
| 4 | FTPD | deep_learning | pending | — | — | — | YAML config |

#### tof_camera — Time-of-Flight Depth Camera

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 41.99 | 0.9994 | 0.00 | benchmark run |
| 2 | FISTA-L2 (depth) | traditional | pending | — | — | — | YAML config |
| 3 | ToF-Net | deep_learning | pending | — | — | — | YAML config |
| 4 | ToF-MPI Deconv | deep_learning | pending | — | — | — | YAML config |

### Electron Microscopy

#### cryo_et — Cryo-Electron Tomography (Cryo-ET)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | CARE | deep_learning | **done** | 13.24 | 0.4127 | 0.17 | benchmark run |
| 2 | Precomputed Baseline | traditional | **done** | 8.44 | 0.2037 | 0.00 | benchmark run |
| 3 | Richardson-Lucy | traditional | **done** | 2.65 | 0.0966 | 0.00 | benchmark run |
| 4 | CryoCARE | deep_learning | pending | — | — | — | YAML config |

#### ebsd — Electron Backscatter Diffraction (EBSD)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 21.82 | 0.9677 | 0.00 | benchmark run |
| 2 | FISTA-L2 (Hough baseline) | traditional | pending | — | — | — | YAML config |
| 3 | EBSD-DL (DictIndex) | deep_learning | pending | — | — | — | YAML config |
| 4 | EMsoft-EBSD | deep_learning | pending | — | — | — | YAML config |

#### edx_mapping — STEM-EDX Elemental Mapping

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 21.97 | 0.9307 | 0.00 | benchmark run |
| 2 | Richardson-Lucy | traditional | **done** | 3.23 | 0.2667 | 0.00 | benchmark run |
| 3 | EDX-Net | deep_learning | pending | — | — | — | YAML config |

#### eels — Electron Energy Loss Spectroscopy (EELS)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 24.59 | 0.9842 | 0.00 | benchmark run |
| 2 | FISTA-L2 (Fourier ratio) | traditional | pending | — | — | — | YAML config |
| 3 | EELS-Net | deep_learning | pending | — | — | — | YAML config |
| 4 | MLLS-EELS | deep_learning | pending | — | — | — | YAML config |
| 5 | EELS-Net | deep_learning | pending | — | — | — | YAML config |
| #1 | EELS-Net | deep_learning | reference | 36.5 | 0.942 | — | pwm.platformai.org |
| #2 | MLLS-EELS (DL) | deep_learning | reference | 34.8 | 0.926 | — | pwm.platformai.org |
| #3 | U-Net-EELS | deep_learning | reference | 33.2 | 0.909 | — | pwm.platformai.org |
| #4 | MLLS-EELS | traditional | reference | 31.5 | 0.889 | — | pwm.platformai.org |
| #5 | FISTA-L2 (Fourier ratio) | traditional | reference | 29.8 | 0.866 | — | pwm.platformai.org |
| #6 | Principal Component Analysis | traditional | reference | 28.1 | 0.841 | — | pwm.platformai.org |

#### electron_diffraction — 4D-STEM Electron Diffraction

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 42.01 | 0.9901 | 0.00 | benchmark run |
| 2 | ePIE (electron ptychography) | traditional | **done** | 42.01 | 0.9889 | 0.00 | benchmark run |
| 3 | ED-Net | deep_learning | pending | — | — | — | YAML config |
| 4 | CRISP-ED | deep_learning | pending | — | — | — | YAML config |

#### electron_holography — Electron Holography

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 9.51 | -0.0481 | 0.00 | benchmark run |
| 2 | Phase Retrieval (HIO) | traditional | **done** | 5.60 | 0.0115 | 2.67 | benchmark run |
| 3 | EH-Net | deep_learning | pending | — | — | — | YAML config |
| 4 | Phase-Sideband | deep_learning | pending | — | — | — | YAML config |

#### electron_tomography — Electron Tomography

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 19.28 | 0.9419 | 0.00 | benchmark run |
| 2 | FBP (SIRT baseline) | traditional | **done** | 25.12 | 0.9525 | 0.41 | benchmark run |
| 3 | IMOD-SIRT-DL | deep_learning | pending | — | — | — | YAML config |
| 4 | SIRT-3D | deep_learning | pending | — | — | — | YAML config |

#### fib_sem — Focused Ion Beam SEM (FIB-SEM)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | CARE | deep_learning | **done** | 4.90 | 0.0019 | 0.16 | benchmark run |
| 2 | Precomputed Baseline | traditional | **done** | 28.11 | 0.9862 | 0.00 | benchmark run |
| 3 | Richardson-Lucy | traditional | **done** | 26.01 | 0.4862 | 0.00 | benchmark run |
| 4 | FIB-SEM-Net | deep_learning | pending | — | — | — | YAML config |

#### sem — Scanning Electron Microscopy (SEM)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 15.75 | 0.7926 | 0.00 | benchmark run |
| 2 | Richardson-Lucy (SEM) | traditional | **done** | 23.17 | 0.4997 | 0.00 | benchmark run |
| 3 | SEM-DL (SegNet) | deep_learning | pending | — | — | — | YAML config |
| 4 | SEM-UNet | deep_learning | pending | — | — | — | YAML config |
| #1 | SEM-FM | deep_learning | reference | 39.2 | 0.960 | — | pwm.platformai.org |
| #2 | SEM-UNet | deep_learning | reference | 37.5 | 0.946 | — | pwm.platformai.org |
| #3 | CARE-SEM | deep_learning | reference | 35.8 | 0.931 | — | pwm.platformai.org |
| #4 | DnCNN-SEM | deep_learning | reference | 34.2 | 0.915 | — | pwm.platformai.org |
| #5 | BM3D-SEM | traditional | reference | 32.5 | 0.896 | — | pwm.platformai.org |
| #6 | Richardson-Lucy | traditional | reference | 30.8 | 0.875 | — | pwm.platformai.org |
| #7 | NLM Denoise | traditional | reference | 29.1 | 0.853 | — | pwm.platformai.org |
| #8 | Wiener Filter | traditional | reference | 27.4 | 0.829 | — | pwm.platformai.org |

#### stem — Scanning Transmission Electron Microscopy (STEM)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 29.97 | 0.9276 | 0.00 | benchmark run |
| 2 | Richardson-Lucy (STEM) | traditional | **done** | 31.01 | 0.9508 | 0.00 | benchmark run |
| 3 | STEM-DL (AtomSegNet) | deep_learning | pending | — | — | — | YAML config |
| 4 | STEM-UNet | deep_learning | pending | — | — | — | YAML config |
| #1 | STEM-FM | deep_learning | reference | 40.1 | 0.964 | — | pwm.platformai.org |
| #2 | STEM-UNet | deep_learning | reference | 38.4 | 0.950 | — | pwm.platformai.org |
| #3 | CARE-STEM | deep_learning | reference | 36.7 | 0.935 | — | pwm.platformai.org |
| #4 | BM3D-STEM | traditional | reference | 34.9 | 0.917 | — | pwm.platformai.org |
| #5 | Wiener-STEM | traditional | reference | 32.8 | 0.895 | — | pwm.platformai.org |
| #6 | Richardson-Lucy | traditional | reference | 30.9 | 0.872 | — | pwm.platformai.org |

#### tem — Transmission Electron Microscopy (TEM)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 25.30 | 0.9190 | 0.00 | benchmark run |
| 2 | FISTA-L2 (CTF correction) | traditional | pending | — | — | — | YAML config |
| 3 | TEM-DL (ePIE-Net) | deep_learning | pending | — | — | — | YAML config |
| 4 | TEM-UNet | deep_learning | pending | — | — | — | YAML config |
| #1 | TEM-FM | deep_learning | reference | 38.8 | 0.957 | — | pwm.platformai.org |
| #2 | TEM-UNet | deep_learning | reference | 37.1 | 0.943 | — | pwm.platformai.org |
| #3 | CARE-TEM | deep_learning | reference | 35.5 | 0.928 | — | pwm.platformai.org |
| #4 | CTF-Wiener | traditional | reference | 33.8 | 0.910 | — | pwm.platformai.org |
| #5 | Richardson-Lucy+CTF | traditional | reference | 32.1 | 0.891 | — | pwm.platformai.org |
| #6 | Phase Plate (Zernike) | traditional | reference | 30.4 | 0.870 | — | pwm.platformai.org |
| #7 | BM3D-TEM | traditional | reference | 28.7 | 0.847 | — | pwm.platformai.org |

### Industrial Inspection

#### acoustic_microscopy — Scanning Acoustic Microscopy (SAM)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 10.04 | -0.0384 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | SAFT-DL | deep_learning | pending | — | — | — | YAML config |

#### active_thermography — Active Thermography (IR)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 6.54 | 0.1897 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | Pulsed-Phase TV | traditional | pending | — | — | — | YAML config |

#### eddy_current — Eddy Current Imaging

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 4.79 | -0.0811 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | ECT-Net | deep_learning | pending | — | — | — | YAML config |

#### industrial_ct — Industrial X-ray CT

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 20.33 | 0.4046 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | IndustrialCT-Net | deep_learning | pending | — | — | — | YAML config |

#### machine_vision — Machine Vision / AOI

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 26.48 | 0.9622 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | PatchCore | deep_learning | pending | — | — | — | YAML config |

#### shearography — Shearography

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 8.02 | -0.0011 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | ShearNet | deep_learning | pending | — | — | — | YAML config |

#### terahertz — Terahertz Imaging (THz)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 37.10 | 0.9963 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | THz-Net | deep_learning | pending | — | — | — | YAML config |

#### ultrasonic_phased_array — Ultrasonic Phased Array (TFM/FMC)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 29.60 | 0.6891 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | TFM-DL | deep_learning | pending | — | — | — | YAML config |

#### xray_ndt — X-ray NDT (Radiography)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 16.72 | 0.8430 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | NDT-DefectNet | deep_learning | pending | — | — | — | YAML config |

#### xrf_imaging — X-ray Fluorescence (XRF) Imaging

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 22.11 | 0.9626 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | XRF-Net | deep_learning | pending | — | — | — | YAML config |

### Medical Imaging

#### angiography — X-ray Angiography

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 11.22 | 0.0435 | 0.00 | benchmark run |
| 2 | FBP (DSA baseline) | traditional | **done** | 12.89 | 0.5828 | 1.35 | benchmark run |
| 3 | DSA-Net | deep_learning | pending | — | — | — | YAML config |
| 4 | VesselSegNet | deep_learning | pending | — | — | — | YAML config |

#### asl_mri — Arterial Spin Labeling (ASL) MRI

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 2.66 | 0.0636 | 0.00 | benchmark run |
| 2 | FBP | traditional | pending | — | — | — | YAML config |
| 3 | DL-Recon | deep_learning | pending | — | — | — | YAML config |
| 4 | ASL-Net | deep_learning | pending | — | — | — | YAML config |
| #1 | ASL-FM | deep_learning | reference | 34.5 | 0.921 | — | pwm.platformai.org |
| #2 | ASL-Net | deep_learning | reference | 32.8 | 0.905 | — | pwm.platformai.org |
| #3 | Bayesian ASL | traditional | reference | 30.5 | 0.882 | — | pwm.platformai.org |
| #4 | FBP (ASL) | traditional | reference | 28.2 | 0.855 | — | pwm.platformai.org |
| #5 | Simple Subtraction | traditional | reference | 25.8 | 0.821 | — | pwm.platformai.org |

#### brachytherapy_img — Brachytherapy Imaging

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 20.48 | 0.2374 | 0.00 | benchmark run |
| 2 | FBP | traditional | pending | — | — | — | YAML config |
| 3 | DL-Recon | deep_learning | pending | — | — | — | YAML config |
| 4 | BrachyNet | deep_learning | pending | — | — | — | YAML config |

#### cbct — Cone-Beam Computed Tomography (CBCT)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | FBP (Ram-Lak filter) | traditional | **done** | 14.93 | 0.3496 | 0.08 | benchmark run |
| 2 | FBP (Shepp-Logan filter) | traditional | **done** | 15.19 | 0.3593 | 0.07 | benchmark run |
| 3 | FDK / FBP | traditional | **done** | 12.69 | 0.4010 | 0.28 | benchmark run |
| 4 | FDK-DL | deep_learning | pending | — | — | — | YAML config |
| 5 | CBCT-UNet | deep_learning | pending | — | — | — | YAML config |
| #1 | CBCT-FM | deep_learning | reference | 38.9 | 0.956 | — | pwm.platformai.org |
| #2 | CTformer-CBCT | deep_learning | reference | 37.1 | 0.944 | — | pwm.platformai.org |
| #3 | CBCT-UNet | deep_learning | reference | 35.5 | 0.931 | — | pwm.platformai.org |
| #4 | RED-CBCT | deep_learning | reference | 34.2 | 0.918 | — | pwm.platformai.org |
| #5 | TV-ADMM-CBCT | traditional | reference | 32.8 | 0.902 | — | pwm.platformai.org |
| #6 | FDK (Hamming) | traditional | reference | 31.3 | 0.885 | — | pwm.platformai.org |
| #7 | FDK / FBP | traditional | reference | 30.1 | 0.868 | — | pwm.platformai.org |
| #8 | FDK (Ram-Lak) | traditional | reference | 29.2 | 0.851 | — | pwm.platformai.org |
| #9 | Backprojection | traditional | reference | 26.5 | 0.816 | — | pwm.platformai.org |

#### cest_mri — CEST MRI

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 31.00 | 0.9859 | 0.00 | benchmark run |
| 2 | FBP | traditional | pending | — | — | — | YAML config |
| 3 | DL-Recon | deep_learning | pending | — | — | — | YAML config |
| 4 | CEST-Net | deep_learning | pending | — | — | — | YAML config |
| #1 | CEST-FM | deep_learning | reference | 36.2 | 0.939 | — | pwm.platformai.org |
| #2 | CEST-Net | deep_learning | reference | 34.5 | 0.922 | — | pwm.platformai.org |
| #3 | Lorentzian Fitting DL | deep_learning | reference | 32.9 | 0.905 | — | pwm.platformai.org |
| #4 | Lorentzian Fitting | traditional | reference | 31.0 | 0.884 | — | pwm.platformai.org |
| #5 | MTRasym | traditional | reference | 29.2 | 0.860 | — | pwm.platformai.org |
| #6 | Direct Saturation | traditional | reference | 27.5 | 0.833 | — | pwm.platformai.org |

#### ceus — Contrast-Enhanced Ultrasound (CEUS)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 24.53 | 0.9206 | 0.00 | benchmark run |
| 2 | FBP | traditional | pending | — | — | — | YAML config |
| 3 | DL-Recon | deep_learning | pending | — | — | — | YAML config |
| 4 | US-DeepSight | deep_learning | pending | — | — | — | YAML config |

#### confocal_endomicroscopy — Confocal Laser Endomicroscopy (CLE)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 34.03 | 0.9927 | 0.00 | benchmark run |
| 2 | FBP | traditional | pending | — | — | — | YAML config |
| 3 | DL-Recon | deep_learning | pending | — | — | — | YAML config |
| 4 | CLE-Net (CARE) | deep_learning | pending | — | — | — | YAML config |

#### ct — X-ray Computed Tomography (CT)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | RED-CNN | deep_learning | **done** | 1.28 | 0.1144 | 0.17 | benchmark run |
| 2 | FBP (Ram-Lak filter) | traditional | **done** | 12.94 | 0.0922 | 0.27 | benchmark run |
| 3 | FBP (Shepp-Logan filter) | traditional | **done** | 13.78 | 0.1053 | 0.16 | benchmark run |
| 4 | SART (10 iterations) | traditional | **done** | 13.75 | 0.2168 | 53.90 | benchmark run |
| 5 | RED-CNN | deep_learning | **done** | 1.19 | 0.1060 | 0.08 | benchmark run |
| 6 | FBP | traditional | **done** | 13.75 | 0.0649 | 0.04 | benchmark run |
| #1 | CT-FM | deep_learning | reference | 44.1 | 0.981 | — | pwm.platformai.org |
| #2 | CTformer | deep_learning | reference | 41.2 | 0.968 | — | pwm.platformai.org |
| #3 | LEARN++ | deep_learning | reference | 40.5 | 0.962 | — | pwm.platformai.org |
| #4 | LEARN | deep_learning | reference | 40.1 | 0.958 | — | pwm.platformai.org |
| #5 | MossFormer-CT | deep_learning | reference | 39.8 | 0.955 | — | pwm.platformai.org |
| #6 | RED-CNN | deep_learning | reference | 38.9 | 0.949 | — | pwm.platformai.org |
| #7 | FBPConvNet | deep_learning | reference | 37.5 | 0.938 | — | pwm.platformai.org |
| #8 | DU-GAN | deep_learning | reference | 37.0 | 0.934 | — | pwm.platformai.org |
| #9 | Noise2Noise-CT | deep_learning | reference | 36.5 | 0.929 | — | pwm.platformai.org |
| #10 | TV-ADMM | traditional | reference | 35.2 | 0.918 | — | pwm.platformai.org |
| #11 | PnP-ADMM | traditional | reference | 34.8 | 0.912 | — | pwm.platformai.org |
| #12 | SART (50 iter) | traditional | reference | 34.1 | 0.905 | — | pwm.platformai.org |
| #13 | SART (20 iter) | traditional | reference | 33.5 | 0.898 | — | pwm.platformai.org |
| #14 | SART (10 iter) | traditional | reference | 32.8 | 0.889 | — | pwm.platformai.org |
| #15 | BM3D-CT | traditional | reference | 31.9 | 0.880 | — | pwm.platformai.org |
| #16 | NLM-CT | traditional | reference | 31.2 | 0.870 | — | pwm.platformai.org |
| #17 | Wiener-CT | traditional | reference | 30.1 | 0.855 | — | pwm.platformai.org |
| #18 | OSEM-CT | traditional | reference | 29.4 | 0.843 | — | pwm.platformai.org |
| #19 | MLEM-CT | traditional | reference | 28.7 | 0.831 | — | pwm.platformai.org |
| #20 | FBP (Hann) | traditional | reference | 27.9 | 0.818 | — | pwm.platformai.org |
| #21 | FBP (Ram-Lak) | traditional | reference | 27.2 | 0.805 | — | pwm.platformai.org |
| #22 | FBP (Shepp-Logan) | traditional | reference | 26.5 | 0.792 | — | pwm.platformai.org |
| #23 | Direct Backprojection | traditional | reference | 25.9 | 0.778 | — | pwm.platformai.org |
| #24 | FBP (cosine) | traditional | reference | 25.2 | 0.761 | — | pwm.platformai.org |

#### dexa — Dual-Energy X-ray Absorptiometry (DEXA)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 9.53 | 0.2550 | 0.00 | benchmark run |
| 2 | FISTA-L2 (dual-energy) | traditional | pending | — | — | — | YAML config |
| 3 | DXA-Net | deep_learning | pending | — | — | — | YAML config |
| 4 | DEXA-UNet | deep_learning | pending | — | — | — | YAML config |

#### diffusion_mri — Diffusion MRI (DTI)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Zero-Filled (IFFT) | traditional | **done** | 11.31 | 0.0002 | 0.00 | benchmark run |
| 2 | SENSE (WLS tensor fit) | traditional | pending | — | — | — | YAML config |
| 3 | q-DL (qDiffusion) | deep_learning | pending | — | — | — | YAML config |
| 4 | SHORE-Net | deep_learning | pending | — | — | — | YAML config |
| #1 | SHORE-Net | deep_learning | reference | 36.5 | 0.941 | — | pwm.platformai.org |
| #2 | q-DL (qDiffusion) | deep_learning | reference | 34.8 | 0.925 | — | pwm.platformai.org |
| #3 | DiffusionDL | deep_learning | reference | 33.2 | 0.909 | — | pwm.platformai.org |
| #4 | CS-dMRI | traditional | reference | 31.5 | 0.889 | — | pwm.platformai.org |
| #5 | SENSE (WLS tensor fit) | traditional | reference | 29.8 | 0.868 | — | pwm.platformai.org |
| #6 | SHORE (analytical) | traditional | reference | 28.1 | 0.845 | — | pwm.platformai.org |
| #7 | DTI (least squares) | traditional | reference | 26.5 | 0.821 | — | pwm.platformai.org |

#### digital_breast_tomo — Digital Breast Tomosynthesis (DBT)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | -36.04 | 0.0001 | 0.00 | benchmark run |
| 2 | FBP | traditional | pending | — | — | — | YAML config |
| 3 | DL-Recon | deep_learning | pending | — | — | — | YAML config |
| 4 | DBT-DL | deep_learning | pending | — | — | — | YAML config |

#### doppler_ultrasound — Doppler Ultrasound

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | UDoppler-Net | deep_learning | pending | — | — | — | YAML config |
| 2 | Doppler CFAR | deep_learning | pending | — | — | — | YAML config |

#### dot — Diffuse Optical Tomography (DOT)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | L-BFGS-TV | deep_learning | pending | — | — | — | YAML config |
| 2 | DOT-Net | deep_learning | pending | — | — | — | YAML config |

#### elastography — Shear-Wave Elastography

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 5.69 | 0.0091 | 0.00 | benchmark run |
| 2 | SENSE (displacement field) | traditional | pending | — | — | — | YAML config |
| 3 | MRE-Net | deep_learning | pending | — | — | — | YAML config |
| 4 | NLSI-Solver | deep_learning | pending | — | — | — | YAML config |

#### endoscopy — Fiber Bundle Endoscopy

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Reconstruction | traditional | **done** | 4.10 | 0.3912 | 0.00 | benchmark run |
| 2 | Richardson-Lucy (20 iter) | traditional | **done** | 11.75 | 0.8796 | 0.04 | benchmark run |
| 3 | Richardson-Lucy (50 iter) | traditional | **done** | 10.41 | 0.8225 | 0.12 | benchmark run |
| 4 | FISTA-L2 (endoscopy) | traditional | pending | — | — | — | YAML config |
| 5 | EndoMapper-Net | deep_learning | pending | — | — | — | YAML config |
| 6 | AF-SfMLearner | deep_learning | pending | — | — | — | YAML config |

#### fluoroscopy — Fluoroscopy

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 43.48 | 0.9997 | 0.00 | benchmark run |
| 2 | FBP (fluoroscopy) | traditional | **done** | 8.58 | 0.1239 | 0.35 | benchmark run |
| 3 | FluoroNet | deep_learning | pending | — | — | — | YAML config |
| 4 | X-ray CNN | deep_learning | pending | — | — | — | YAML config |

#### fmri — Functional MRI (BOLD fMRI)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Zero-Filled (IFFT) | traditional | **done** | 4.93 | -0.5617 | 0.00 | benchmark run |
| 2 | SENSE (fMRI) | traditional | pending | — | — | — | YAML config |
| 3 | fMRI-Transformer | deep_learning | pending | — | — | — | YAML config |
| 4 | DeepBold | deep_learning | pending | — | — | — | YAML config |
| #1 | fMRI-FM | deep_learning | reference | 38.5 | 0.953 | — | pwm.platformai.org |
| #2 | SwinMR-fMRI | deep_learning | reference | 36.9 | 0.939 | — | pwm.platformai.org |
| #3 | U-Net-fMRI | deep_learning | reference | 35.4 | 0.924 | — | pwm.platformai.org |
| #4 | CS-fMRI (Wavelet) | traditional | reference | 33.2 | 0.903 | — | pwm.platformai.org |
| #5 | SENSE-fMRI | traditional | reference | 31.5 | 0.882 | — | pwm.platformai.org |
| #6 | GRAPPA-fMRI | traditional | reference | 30.0 | 0.861 | — | pwm.platformai.org |
| #7 | Zero-Filled IFFT | traditional | reference | 26.0 | 0.812 | — | pwm.platformai.org |

#### fundus — Fundus Camera

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Wiener Filter (precomputed) | traditional | **done** | 22.02 | 0.9248 | 0.00 | benchmark run |
| 2 | Richardson-Lucy (20 iter) | traditional | **done** | 35.02 | 0.9965 | 0.04 | benchmark run |
| 3 | Richardson-Lucy (50 iter) | traditional | **done** | 35.93 | 0.9972 | 0.11 | benchmark run |
| 4 | Richardson-Lucy | traditional | **done** | 30.58 | 0.9090 | 0.00 | benchmark run |
| 5 | RETFound | deep_learning | pending | — | — | — | YAML config |
| 6 | DR-Grade-Net | deep_learning | pending | — | — | — | YAML config |

#### ivus — Intravascular Ultrasound (IVUS)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 19.83 | 0.8902 | 0.00 | benchmark run |
| 2 | FBP | traditional | pending | — | — | — | YAML config |
| 3 | DL-Recon | deep_learning | pending | — | — | — | YAML config |
| 4 | IVUS-Net | deep_learning | pending | — | — | — | YAML config |

#### mammography — Mammography

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Reconstruction | traditional | **done** | 20.94 | 0.8580 | 0.00 | benchmark run |
| 2 | FBP (mammography) | traditional | **done** | 4.08 | 0.0047 | 0.08 | benchmark run |
| 3 | MammoNet (GatorTron) | deep_learning | pending | — | — | — | YAML config |
| 4 | Mammo-ResNet | deep_learning | pending | — | — | — | YAML config |

#### mr_elastography — MR Elastography (MRE)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 6.01 | 0.0984 | 0.00 | benchmark run |
| 2 | FBP | traditional | pending | — | — | — | YAML config |
| 3 | DL-Recon | deep_learning | pending | — | — | — | YAML config |
| 4 | MRE-Net | deep_learning | pending | — | — | — | YAML config |
| #1 | MRE-FM | deep_learning | reference | 35.8 | 0.937 | — | pwm.platformai.org |
| #2 | MRE-Net | deep_learning | reference | 33.9 | 0.920 | — | pwm.platformai.org |
| #3 | NLSI-Solver | traditional | reference | 31.5 | 0.898 | — | pwm.platformai.org |
| #4 | SENSE (displacement field) | traditional | reference | 29.2 | 0.872 | — | pwm.platformai.org |
| #5 | Direct Inversion | traditional | reference | 26.8 | 0.841 | — | pwm.platformai.org |

#### mr_fingerprinting — MR Fingerprinting (MRF)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 1.84 | 0.0693 | 0.00 | benchmark run |
| 2 | FBP | traditional | pending | — | — | — | YAML config |
| 3 | DL-Recon | deep_learning | pending | — | — | — | YAML config |
| 4 | MRF-Net | deep_learning | pending | — | — | — | YAML config |
| #1 | MRF-FM | deep_learning | reference | 37.2 | 0.945 | — | pwm.platformai.org |
| #2 | DeepMRF | deep_learning | reference | 35.6 | 0.930 | — | pwm.platformai.org |
| #3 | BLOCH-Net | deep_learning | reference | 34.1 | 0.915 | — | pwm.platformai.org |
| #4 | Dictionary Matching | traditional | reference | 32.5 | 0.897 | — | pwm.platformai.org |
| #5 | SVD Compression | traditional | reference | 30.8 | 0.877 | — | pwm.platformai.org |
| #6 | Low-Rank Approximation | traditional | reference | 29.2 | 0.855 | — | pwm.platformai.org |

#### mra — MR Angiography (MRA)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 12.10 | 0.2673 | 0.00 | benchmark run |
| 2 | FBP | traditional | pending | — | — | — | YAML config |
| 3 | DL-Recon | deep_learning | pending | — | — | — | YAML config |
| 4 | MRA-VesselNet | deep_learning | pending | — | — | — | YAML config |
| #1 | MRA-FM | deep_learning | reference | 40.2 | 0.965 | — | pwm.platformai.org |
| #2 | VarNet-MRA | deep_learning | reference | 38.7 | 0.951 | — | pwm.platformai.org |
| #3 | E2E-VarNet-MRA | deep_learning | reference | 37.3 | 0.937 | — | pwm.platformai.org |
| #4 | SENSE-MRA | traditional | reference | 35.0 | 0.918 | — | pwm.platformai.org |
| #5 | GRAPPA-MRA | traditional | reference | 33.2 | 0.898 | — | pwm.platformai.org |
| #6 | Zero-Filled IFFT | traditional | reference | 28.5 | 0.851 | — | pwm.platformai.org |

#### mri — Magnetic Resonance Imaging (MRI)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | CS-MRI (Wavelet L1) | traditional | **done** | 13.02 | 0.0006 | 0.05 | benchmark run |
| 2 | SENSE (parallel imaging) | traditional | **done** | 13.05 | 0.0010 | 0.04 | benchmark run |
| 3 | Zero-Filled (IFFT) | traditional | **done** | 13.01 | 0.0004 | 0.11 | benchmark run |
| 4 | SENSE | traditional | pending | — | — | — | YAML config |
| #1 | SwinMR++ | deep_learning | reference | 43.8 | 0.979 | — | pwm.platformai.org |
| #2 | E2E-VarNet | deep_learning | reference | 42.1 | 0.972 | — | pwm.platformai.org |
| #3 | VarNet | deep_learning | reference | 41.5 | 0.968 | — | pwm.platformai.org |
| #4 | MoDL | deep_learning | reference | 40.9 | 0.963 | — | pwm.platformai.org |
| #5 | XPDNet | deep_learning | reference | 40.3 | 0.958 | — | pwm.platformai.org |
| #6 | SwinMR | deep_learning | reference | 39.8 | 0.953 | — | pwm.platformai.org |
| #7 | ReconFormer | deep_learning | reference | 39.2 | 0.947 | — | pwm.platformai.org |
| #8 | Cascaded U-Net | deep_learning | reference | 38.7 | 0.942 | — | pwm.platformai.org |
| #9 | E2E-VarNet (fastMRI) | deep_learning | reference | 38.2 | 0.937 | — | pwm.platformai.org |
| #10 | U-Net MRI | deep_learning | reference | 37.5 | 0.929 | — | pwm.platformai.org |
| #11 | ISTA-Net+ | deep_learning | reference | 37.0 | 0.923 | — | pwm.platformai.org |
| #12 | ADMM-Net | deep_learning | reference | 36.4 | 0.917 | — | pwm.platformai.org |
| #13 | D5C5 | deep_learning | reference | 35.9 | 0.911 | — | pwm.platformai.org |
| #14 | DCCNN | deep_learning | reference | 35.3 | 0.904 | — | pwm.platformai.org |
| #15 | CS-MRI (Wavelet) | traditional | reference | 34.8 | 0.897 | — | pwm.platformai.org |
| #16 | GRAPPA | traditional | reference | 34.2 | 0.890 | — | pwm.platformai.org |
| #17 | SENSE | traditional | reference | 33.7 | 0.882 | — | pwm.platformai.org |
| #18 | ESPIRIT | traditional | reference | 33.1 | 0.875 | — | pwm.platformai.org |
| #19 | TV-Reg | traditional | reference | 32.5 | 0.867 | — | pwm.platformai.org |
| #20 | LORAKS | traditional | reference | 31.8 | 0.858 | — | pwm.platformai.org |
| #21 | SPIRiT | traditional | reference | 31.2 | 0.849 | — | pwm.platformai.org |
| #22 | L+S Decomp | traditional | reference | 30.5 | 0.839 | — | pwm.platformai.org |
| #23 | PICS (BART) | traditional | reference | 29.9 | 0.829 | — | pwm.platformai.org |
| #24 | k-t SPARSE-SENSE | traditional | reference | 29.2 | 0.818 | — | pwm.platformai.org |
| #25 | BM3D-MRI | traditional | reference | 28.6 | 0.807 | — | pwm.platformai.org |
| #26 | NLM-MRI | traditional | reference | 28.0 | 0.796 | — | pwm.platformai.org |
| #27 | Zero-Filled (IFFT) | traditional | reference | 26.0 | 0.761 | — | pwm.platformai.org |

#### mrs — MR Spectroscopy (MRS)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 1.88 | 0.0676 | 0.00 | benchmark run |
| 2 | SENSE (spectroscopy) | traditional | pending | — | — | — | YAML config |
| 3 | MRS-Net | deep_learning | pending | — | — | — | YAML config |
| 4 | HLSVD-MRS | deep_learning | pending | — | — | — | YAML config |

#### nirs_brain — Functional Near-Infrared Spectroscopy (fNIRS)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 14.48 | 0.8761 | 0.00 | benchmark run |
| 2 | FBP | traditional | pending | — | — | — | YAML config |
| 3 | DL-Recon | deep_learning | pending | — | — | — | YAML config |
| 4 | fNIRS-Net | deep_learning | pending | — | — | — | YAML config |

#### oct — Optical Coherence Tomography (OCT)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Spectral Estimation | deep_learning | **done** | 10.16 | 0.1411 | 1.01 | benchmark run |
| 2 | B-scan Direct (noisy) | traditional | **done** | 23.10 | 0.9439 | 0.00 | benchmark run |
| 3 | B-scan Ideal (noiseless) | traditional | **done** | 23.48 | 0.9482 | 0.00 | benchmark run |
| 4 | OCT Denoising Net | deep_learning | **done** | 10.27 | 0.1422 | 0.00 | benchmark run |
| 5 | OCT Denoising Net | deep_learning | **done** | 10.27 | 0.1422 | 0.00 | benchmark run |
| 6 | FFT Recon | traditional | **done** | 10.27 | 0.1422 | 0.00 | benchmark run |
| #1 | ScoreOCT | deep_learning | reference | 38.0 | 0.959 | — | pwm.platformai.org |
| #2 | OCT-FM | deep_learning | reference | 36.8 | 0.948 | — | pwm.platformai.org |
| #3 | OCT-Transformer | deep_learning | reference | 35.4 | 0.935 | — | pwm.platformai.org |
| #4 | DnCNN-OCT | deep_learning | reference | 34.2 | 0.921 | — | pwm.platformai.org |
| #5 | BM3D-OCT | traditional | reference | 32.5 | 0.903 | — | pwm.platformai.org |
| #6 | NLM-OCT | traditional | reference | 31.0 | 0.886 | — | pwm.platformai.org |
| #7 | Wiener-OCT | traditional | reference | 29.5 | 0.868 | — | pwm.platformai.org |
| #8 | B-scan Average | traditional | reference | 28.0 | 0.847 | — | pwm.platformai.org |

#### octa — OCT Angiography (OCTA)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 16.78 | 0.4326 | 0.00 | benchmark run |
| 2 | FFT Recon (OCTA) | traditional | **done** | 13.33 | 0.0566 | 0.00 | benchmark run |
| 3 | OCTA-Net | deep_learning | pending | — | — | — | YAML config |
| 4 | OCTA-FF | deep_learning | pending | — | — | — | YAML config |

#### pet — Positron Emission Tomography (PET)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | FBP (Ram-Lak filter) | traditional | **done** | 9.29 | 0.1813 | 0.09 | benchmark run |
| 2 | FBP (Shepp-Logan filter) | traditional | **done** | 11.86 | 0.2681 | 0.07 | benchmark run |
| 3 | FBP (precomputed) | traditional | **done** | 33.09 | 0.9325 | 0.00 | benchmark run |
| 4 | FBP (emission tomography) | traditional | **done** | 15.42 | 0.0116 | 0.07 | benchmark run |
| 5 | NeuroLF-PET | deep_learning | pending | — | — | — | YAML config |
| 6 | PET-DL (U-Net) | deep_learning | pending | — | — | — | YAML config |
| #1 | NeuroLF-PET | deep_learning | reference | 39.2 | 0.962 | — | pwm.platformai.org |
| #2 | U-Net-PET | deep_learning | reference | 36.8 | 0.948 | — | pwm.platformai.org |
| #3 | PET-DL (ResNet) | deep_learning | reference | 35.4 | 0.935 | — | pwm.platformai.org |
| #4 | DPIR-PET | deep_learning | reference | 34.1 | 0.921 | — | pwm.platformai.org |
| #5 | MAP-EM | traditional | reference | 32.5 | 0.903 | — | pwm.platformai.org |
| #6 | OSEM (128 iter) | traditional | reference | 31.2 | 0.887 | — | pwm.platformai.org |
| #7 | OSEM (64 iter) | traditional | reference | 30.5 | 0.878 | — | pwm.platformai.org |
| #8 | MLEM (100 iter) | traditional | reference | 29.8 | 0.866 | — | pwm.platformai.org |
| #9 | FBP (Ramp) | traditional | reference | 27.3 | 0.832 | — | pwm.platformai.org |
| #10 | Direct Backprojection | traditional | reference | 24.8 | 0.798 | — | pwm.platformai.org |

#### photoacoustic — Photoacoustic Imaging

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 19.11 | 0.2490 | 0.00 | benchmark run |
| 2 | Back Projection | traditional | **done** | 18.55 | 0.3658 | 0.00 | benchmark run |
| 3 | Time Reversal | deep_learning | pending | — | — | — | YAML config |
| 4 | Deep-PAT | deep_learning | pending | — | — | — | YAML config |
| 5 | Deep-PAT | deep_learning | pending | — | — | — | YAML config |
| #1 | PAM-FM | deep_learning | reference | 37.5 | 0.948 | — | pwm.platformai.org |
| #2 | PAM-U-Net | deep_learning | reference | 35.8 | 0.933 | — | pwm.platformai.org |
| #3 | PAM-CNN | deep_learning | reference | 34.1 | 0.917 | — | pwm.platformai.org |
| #4 | Model-Based (TV) | traditional | reference | 32.5 | 0.899 | — | pwm.platformai.org |
| #5 | Backprojection (Time Reversal) | traditional | reference | 30.5 | 0.876 | — | pwm.platformai.org |
| #6 | Delay-and-Sum | traditional | reference | 28.2 | 0.849 | — | pwm.platformai.org |

#### portal_imaging — Portal Imaging (EPID)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 10.49 | 0.4088 | 0.00 | benchmark run |
| 2 | FBP | traditional | pending | — | — | — | YAML config |
| 3 | DL-Recon | deep_learning | pending | — | — | — | YAML config |
| 4 | PortalDL | deep_learning | pending | — | — | — | YAML config |

#### proton_therapy_img — Proton Therapy Imaging

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 17.85 | 0.7117 | 0.00 | benchmark run |
| 2 | FBP | traditional | pending | — | — | — | YAML config |
| 3 | DL-Recon | deep_learning | pending | — | — | — | YAML config |
| 4 | ProtonTherapy-Net | deep_learning | pending | — | — | — | YAML config |

#### spect — Single Photon Emission CT (SPECT)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | FBP (Ram-Lak filter) | traditional | **done** | -6.54 | 0.0101 | 0.06 | benchmark run |
| 2 | FBP (precomputed) | traditional | **done** | 30.03 | 0.9523 | 0.00 | benchmark run |
| 3 | FBP (emission tomography) | traditional | **done** | 10.77 | 0.0669 | 0.36 | benchmark run |
| 4 | SPECT-DL (OSEM+) | deep_learning | pending | — | — | — | YAML config |
| 5 | SPECT-UNet | deep_learning | pending | — | — | — | YAML config |
| #1 | SPECT-FM | deep_learning | reference | 37.5 | 0.951 | — | pwm.platformai.org |
| #2 | U-Net-SPECT | deep_learning | reference | 35.2 | 0.938 | — | pwm.platformai.org |
| #3 | PnP-ADMM-SPECT | traditional | reference | 33.0 | 0.918 | — | pwm.platformai.org |
| #4 | OSEM-SPECT (64 iter) | traditional | reference | 31.5 | 0.902 | — | pwm.platformai.org |
| #5 | MLEM-SPECT (100 iter) | traditional | reference | 30.2 | 0.886 | — | pwm.platformai.org |
| #6 | MAP-SPECT | traditional | reference | 29.0 | 0.870 | — | pwm.platformai.org |
| #7 | Chang Attenuation | traditional | reference | 27.8 | 0.851 | — | pwm.platformai.org |
| #8 | FBP (Butterworth) | traditional | reference | 26.5 | 0.831 | — | pwm.platformai.org |
| #9 | FBP (Ram-Lak) | traditional | reference | 25.1 | 0.812 | — | pwm.platformai.org |
| #10 | Backprojection | traditional | reference | 22.3 | 0.781 | — | pwm.platformai.org |

#### spectral_ct — Photon-Counting Spectral CT

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 12.30 | 0.1106 | 0.00 | benchmark run |
| 2 | FBP | traditional | pending | — | — | — | YAML config |
| 3 | DL-Recon | deep_learning | pending | — | — | — | YAML config |
| 4 | SpectralCT-Net | deep_learning | pending | — | — | — | YAML config |

#### swi — Susceptibility-Weighted Imaging (SWI)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 1.94 | 0.0677 | 0.00 | benchmark run |
| 2 | FBP | traditional | pending | — | — | — | YAML config |
| 3 | DL-Recon | deep_learning | pending | — | — | — | YAML config |
| 4 | SWI-Net | deep_learning | pending | — | — | — | YAML config |
| #1 | SWI-FM | deep_learning | reference | 39.5 | 0.957 | — | pwm.platformai.org |
| #2 | VarNet-SWI | deep_learning | reference | 37.8 | 0.943 | — | pwm.platformai.org |
| #3 | SENSE-SWI | traditional | reference | 35.2 | 0.924 | — | pwm.platformai.org |
| #4 | Phase Mask (Homodyne) | traditional | reference | 32.8 | 0.900 | — | pwm.platformai.org |
| #5 | Zero-Filled IFFT | traditional | reference | 28.9 | 0.862 | — | pwm.platformai.org |

#### ultrasound — Ultrasound B-mode Imaging

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Richardson-Lucy (20 iter) | traditional | **done** | 14.57 | 0.1559 | 0.05 | benchmark run |
| 2 | Richardson-Lucy (50 iter) | traditional | **done** | 14.12 | 0.1323 | 0.11 | benchmark run |
| 3 | FISTA-L2 (ultrasound) | traditional | pending | — | — | — | YAML config |
| 4 | US-UNet (DeepUS) | deep_learning | pending | — | — | — | YAML config |
| 5 | US-CNN | deep_learning | pending | — | — | — | YAML config |
| #1 | ScoreUS | deep_learning | reference | 36.3 | 0.944 | — | pwm.platformai.org |
| #2 | US-FM | deep_learning | reference | 35.1 | 0.932 | — | pwm.platformai.org |
| #3 | US-UNet (DeepUS) | deep_learning | reference | 33.8 | 0.919 | — | pwm.platformai.org |
| #4 | US-CNN | deep_learning | reference | 32.5 | 0.905 | — | pwm.platformai.org |
| #5 | DAS Beamforming | traditional | reference | 30.8 | 0.887 | — | pwm.platformai.org |
| #6 | ADMM-US | traditional | reference | 29.5 | 0.870 | — | pwm.platformai.org |
| #7 | FISTA-L2 | traditional | reference | 28.2 | 0.853 | — | pwm.platformai.org |
| #8 | Richardson-Lucy | traditional | reference | 26.9 | 0.833 | — | pwm.platformai.org |

#### xray_radiography — X-ray Radiography

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 26.31 | 0.9844 | 0.00 | benchmark run |
| 2 | FBP (X-ray radiography) | traditional | **done** | 4.54 | -0.0019 | 0.06 | benchmark run |
| 3 | CheXNet | deep_learning | pending | — | — | — | YAML config |
| 4 | X-ray UNet | deep_learning | pending | — | — | — | YAML config |

### Microscopy

#### confocal_3d — Confocal 3D Z-Stack

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | 3D CARE | deep_learning | **done** | 0.07 | 0.0043 | 0.57 | benchmark run |
| 2 | CARE-3D | deep_learning | **done** | 0.07 | 0.0043 | 0.41 | benchmark run |
| 3 | Precomputed Baseline | traditional | **done** | 17.83 | 0.0530 | 0.00 | benchmark run |
| 4 | Richardson-Lucy (20 iter) | traditional | **done** | -26.42 | 0.0000 | 0.04 | benchmark run |
| 5 | CARE-3D (slice-wise) | deep_learning | **done** | 27.27 | 0.8317 | 0.39 | benchmark run |
| 6 | 3D Richardson-Lucy | traditional | **done** | 0.29 | 0.0042 | 0.00 | benchmark run |
| #1 | CARE-3D | deep_learning | reference | 38.2 | 0.954 | — | pwm.platformai.org |
| #2 | CARE-3D (slice-wise) | deep_learning | reference | 36.5 | 0.939 | — | pwm.platformai.org |
| #3 | 3D CARE | deep_learning | reference | 35.0 | 0.924 | — | pwm.platformai.org |
| #4 | DeconvNet-3D | deep_learning | reference | 33.5 | 0.908 | — | pwm.platformai.org |
| #5 | 3D Richardson-Lucy | traditional | reference | 31.8 | 0.888 | — | pwm.platformai.org |
| #6 | 3D Wiener | traditional | reference | 29.9 | 0.864 | — | pwm.platformai.org |
| #7 | 3D BM4D | traditional | reference | 28.2 | 0.839 | — | pwm.platformai.org |

#### confocal_livecell — Confocal Live-Cell Microscopy

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | CARE | deep_learning | **done** | 13.92 | 0.3589 | 0.15 | benchmark run |
| 2 | CARE | deep_learning | **done** | 14.61 | 0.2807 | 0.15 | benchmark run |
| 3 | Precomputed Baseline | traditional | **done** | 31.34 | 0.9870 | 0.00 | benchmark run |
| 4 | CARE | deep_learning | **done** | 15.59 | 0.2430 | 0.14 | benchmark run |
| 5 | Richardson-Lucy | traditional | **done** | 32.28 | 0.8670 | 0.00 | benchmark run |

#### dark_field — Dark-Field Microscopy

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | CARE | deep_learning | **done** | 10.46 | 0.6101 | 0.14 | benchmark run |
| 2 | Precomputed Baseline | traditional | **done** | 25.11 | 0.9781 | 0.00 | benchmark run |
| 3 | Richardson-Lucy | traditional | **done** | 20.62 | 0.7815 | 0.00 | benchmark run |
| 4 | DF-UNet | traditional | pending | — | — | — | YAML config |

#### dic — Differential Interference Contrast (DIC)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | CARE | deep_learning | **done** | 15.03 | 0.3956 | 0.16 | benchmark run |
| 2 | Precomputed Baseline | traditional | **done** | 10.86 | -0.3388 | 0.00 | benchmark run |
| 3 | Richardson-Lucy | traditional | **done** | 15.62 | 0.3801 | 0.00 | benchmark run |
| 4 | DIC-Net | deep_learning | pending | — | — | — | YAML config |

#### dna_paint — DNA-PAINT Super-Resolution

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | CARE | deep_learning | **done** | 27.79 | 0.1984 | 0.17 | benchmark run |
| 2 | Precomputed Baseline | traditional | **done** | 28.53 | 0.3552 | 0.00 | benchmark run |
| 3 | Richardson-Lucy | traditional | **done** | 22.86 | 0.0260 | 0.00 | benchmark run |
| 4 | DECODE-PAINT | deep_learning | pending | — | — | — | YAML config |

#### expansion — Expansion Microscopy (ExM)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | CARE | deep_learning | **done** | 15.93 | 0.6087 | 0.15 | benchmark run |
| 2 | Precomputed Baseline | traditional | **done** | 33.33 | 0.9823 | 0.00 | benchmark run |
| 3 | Richardson-Lucy | traditional | **done** | 33.89 | 0.6181 | 0.00 | benchmark run |
| 4 | EXpansionNet | deep_learning | pending | — | — | — | YAML config |

#### flim — Fluorescence Lifetime Imaging (FLIM)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 30.74 | 0.9901 | 0.00 | benchmark run |
| 2 | Phasor Analysis | traditional | **done** | 1.24 | 0.0555 | 0.01 | benchmark run |
| 3 | FLIMNet | deep_learning | pending | — | — | — | YAML config |
| 4 | FLIMNet | deep_learning | pending | — | — | — | YAML config |

#### fpm — Fourier Ptychographic Microscopy (FPM)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 16.94 | 0.7943 | 0.00 | benchmark run |
| 2 | Sequential Phase Retrieval | traditional | **done** | 5.21 | 0.0237 | 0.00 | benchmark run |
| 3 | Gradient Descent FPM | deep_learning | pending | — | — | — | YAML config |
| 4 | Fourier Ptychnet | deep_learning | pending | — | — | — | YAML config |
| 5 | Fourier Ptychnet | deep_learning | pending | — | — | — | YAML config |

#### ism — Image Scanning Microscopy (ISM)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | CARE | deep_learning | **done** | 0.51 | 0.0286 | 0.21 | benchmark run |
| 2 | Precomputed Baseline | traditional | **done** | -50.06 | 0.0000 | 0.00 | benchmark run |
| 3 | Richardson-Lucy | traditional | **done** | 3.07 | 0.1516 | 0.00 | benchmark run |
| 4 | ISM-Reassignment-Net | deep_learning | pending | — | — | — | YAML config |

#### lattice_lightsheet — Lattice Light-Sheet Microscopy

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | CARE | deep_learning | **done** | 14.45 | 0.7656 | 0.18 | benchmark run |
| 2 | Precomputed Baseline | traditional | **done** | 21.33 | 0.7759 | 0.00 | benchmark run |
| 3 | Richardson-Lucy | traditional | **done** | 25.11 | 0.3079 | 0.00 | benchmark run |
| 4 | LLSM-CARE | deep_learning | pending | — | — | — | YAML config |

#### lightsheet — Light-Sheet Fluorescence Microscopy (LSFM)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | VSNR | deep_learning | **done** | 0.22 | 0.0043 | 0.07 | benchmark run |
| 2 | DeStripe | deep_learning | **done** | 0.21 | 0.0045 | 16.92 | benchmark run |
| 3 | Fourier Notch Filter | traditional | **done** | -28.21 | 0.0000 | 0.00 | benchmark run |
| 4 | Precomputed Baseline | traditional | **done** | 20.03 | 0.0553 | 0.00 | benchmark run |
| 5 | Richardson-Lucy (20 iter) | traditional | **done** | -33.41 | 0.0000 | 0.05 | benchmark run |
| 6 | DeStripe | deep_learning | **done** | 0.21 | 0.0045 | 17.85 | benchmark run |
| 7 | Fourier Notch Filter | traditional | **done** | 0.21 | 0.0045 | 0.50 | benchmark run |

#### minflux — MINFLUX Nanoscopy

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | CARE | deep_learning | **done** | 29.21 | 0.7051 | 0.35 | benchmark run |
| 2 | Precomputed Baseline | traditional | **done** | 29.52 | 0.4336 | 0.00 | benchmark run |
| 3 | Richardson-Lucy | traditional | **done** | 29.50 | 0.7052 | 0.00 | benchmark run |
| 4 | MINFLUX-Net | deep_learning | pending | — | — | — | YAML config |

#### palm_storm — PALM/STORM Single-Molecule Localization

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 32.42 | 0.6094 | 0.00 | benchmark run |
| 2 | Richardson-Lucy (20 iter) | traditional | **done** | 32.42 | 0.5904 | 0.04 | benchmark run |
| 3 | Richardson-Lucy (STORM/PALM) | traditional | **done** | 0.03 | 0.0005 | 0.00 | benchmark run |
| 4 | DECODE-SMLM | deep_learning | pending | — | — | — | YAML config |
| 5 | DeepSTORM | deep_learning | pending | — | — | — | YAML config |

#### phase_contrast — Phase Contrast Microscopy

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | CARE | deep_learning | **done** | 8.10 | 0.3458 | 0.24 | benchmark run |
| 2 | Precomputed Baseline | traditional | **done** | 45.56 | 0.9991 | 0.00 | benchmark run |
| 3 | Richardson-Lucy | traditional | **done** | 8.10 | 0.3458 | 0.00 | benchmark run |
| 4 | PhaseNet | deep_learning | pending | — | — | — | YAML config |

#### polarization — Polarization Microscopy

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 15.83 | 0.6265 | 0.00 | benchmark run |
| 2 | PnP-HQS | traditional | **done** | 8.42 | 0.0892 | 0.00 | benchmark run |
| 3 | PolarNet | deep_learning | pending | — | — | — | YAML config |
| 4 | Stokes-NN | deep_learning | pending | — | — | — | YAML config |

#### shg — Second Harmonic Generation (SHG) Microscopy

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | CARE | deep_learning | **done** | 17.20 | 0.1742 | 0.48 | benchmark run |
| 2 | Precomputed Baseline | traditional | **done** | 23.03 | 0.7974 | 0.00 | benchmark run |
| 3 | Richardson-Lucy | traditional | **done** | 22.30 | 0.5919 | 0.00 | benchmark run |
| 4 | SHG-CARE | deep_learning | pending | — | — | — | YAML config |

#### sim — Structured Illumination Microscopy (SIM)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 21.58 | 0.1863 | 0.00 | benchmark run |
| 2 | Wiener-SIM | traditional | **done** | 0.08 | 0.0056 | 0.00 | benchmark run |
| 3 | Wiener SIM Reconstruction | traditional | **done** | 6.16 | 0.0174 | 0.26 | benchmark run |

#### spinning_disk — Spinning Disk Confocal Microscopy

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | CARE | deep_learning | **done** | 14.23 | 0.3440 | 0.47 | benchmark run |
| 2 | Precomputed Baseline | traditional | **done** | 30.61 | 0.9835 | 0.00 | benchmark run |
| 3 | Richardson-Lucy | traditional | **done** | 29.47 | 0.7581 | 0.00 | benchmark run |
| 4 | SD-CARE | deep_learning | pending | — | — | — | YAML config |

#### sted — STED Microscopy

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 24.98 | 0.8484 | 0.00 | benchmark run |
| 2 | Richardson-Lucy (20 iter) | traditional | **done** | -38.33 | 0.0000 | 0.04 | benchmark run |
| 3 | Richardson-Lucy (STED) | traditional | **done** | 0.26 | 0.0017 | 0.00 | benchmark run |
| 4 | STED-Net (CARE) | deep_learning | pending | — | — | — | YAML config |
| 5 | RCAN-STED | deep_learning | pending | — | — | — | YAML config |

#### three_photon — Three-Photon Microscopy

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | CARE | deep_learning | **done** | 14.53 | 0.3513 | 0.45 | benchmark run |
| 2 | Precomputed Baseline | traditional | **done** | 20.80 | 0.8419 | 0.00 | benchmark run |
| 3 | Richardson-Lucy | traditional | **done** | 14.65 | 0.3353 | 0.00 | benchmark run |
| 4 | 3P-Net (CARE) | deep_learning | pending | — | — | — | YAML config |

#### tirf — TIRF Microscopy

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 31.24 | 0.6216 | 0.00 | benchmark run |
| 2 | Richardson-Lucy (TIRF) | traditional | **done** | 27.72 | 0.1106 | 0.00 | benchmark run |
| 3 | TIRF-Net (CARE) | deep_learning | pending | — | — | — | YAML config |
| 4 | TIRF-SRRF | deep_learning | pending | — | — | — | YAML config |

#### two_photon — Two-Photon / Multiphoton Microscopy

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 33.76 | 0.9867 | 0.00 | benchmark run |
| 2 | Richardson-Lucy (20 iter) | traditional | **done** | -46.98 | 0.0000 | 0.04 | benchmark run |
| 3 | Richardson-Lucy (2P) | traditional | **done** | 0.94 | 0.0073 | 0.00 | benchmark run |
| 4 | 2P-Net (CARE) | deep_learning | pending | — | — | — | YAML config |
| 5 | 2P-DeepInterp | deep_learning | pending | — | — | — | YAML config |

#### widefield — Widefield Fluorescence Microscopy

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | CARE | deep_learning | **done** | 14.45 | 0.7656 | 0.47 | benchmark run |
| 2 | CARE | deep_learning | **done** | 14.45 | 0.7656 | 0.31 | benchmark run |
| 3 | Precomputed Baseline | traditional | **done** | 24.98 | 0.9091 | 0.00 | benchmark run |
| 4 | CARE | deep_learning | **done** | 14.45 | 0.7656 | 0.42 | benchmark run |
| 5 | Richardson-Lucy | traditional | **done** | 24.08 | 0.2696 | 0.00 | benchmark run |

#### widefield_lowdose — Low-Dose Widefield Microscopy

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | CARE | deep_learning | **done** | 12.63 | 0.5013 | 0.29 | benchmark run |
| 2 | Precomputed Baseline | traditional | **done** | 18.84 | 0.6755 | 0.00 | benchmark run |
| 3 | BM3D + RL | traditional | **done** | 28.96 | 0.9402 | 0.00 | benchmark run |
| 4 | Noise2Void | deep_learning | pending | — | — | — | YAML config |
| 5 | Noise2Void | deep_learning | pending | — | — | — | YAML config |

### Multi-Modal Fusion

#### clem — Correlative Light-Electron Microscopy (CLEM)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 17.00 | 0.7297 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | CLEM-Net | deep_learning | pending | — | — | — | YAML config |

#### ct_fluorescence — CT + Fluorescence (FLIT)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | -37.64 | 0.0002 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | XFCT-Net | deep_learning | pending | — | — | — | YAML config |

#### pet_ct — PET/CT Fusion

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 12.98 | 0.0656 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | PET-CT-Fusion-Net | deep_learning | pending | — | — | — | YAML config |

#### pet_mr — PET/MR Fusion

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 10.98 | 0.0165 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | PET-MR-DeepJoint | deep_learning | pending | — | — | — | YAML config |

#### spect_ct — SPECT/CT Fusion

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 11.38 | 0.0239 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | SPECT-CT-Net | deep_learning | pending | — | — | — | YAML config |

#### us_mri — US/MRI Fusion

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 7.56 | -0.0694 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | US-MRI-Net | deep_learning | pending | — | — | — | YAML config |

### Neural Rendering

#### gaussian_splatting — 3D Gaussian Splatting (3DGS)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | 3DGS (full) | deep_learning | **done** | inf | 1.0000 | 0.00 | benchmark run |
| 2 | Direct Render Baseline | traditional | **done** | 0.00 | 0.0000 | 0.00 | benchmark run |
| 3 | NeRF (baseline comparison) | deep_learning | **done** | inf | 1.0000 | 240.44 | benchmark run |
| 4 | 3DGS (compact) | deep_learning | **done** | inf | 1.0000 | 0.00 | benchmark run |
| 5 | EWA Splatting | traditional | **done** | inf | 1.0000 | 0.00 | benchmark run |

#### nerf — Neural Radiance Fields (NeRF)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 28.99 | 0.9913 | 0.00 | benchmark run |
| 2 | SfM + MVS | traditional | **done** | 21.37 | 0.8758 | 210.61 | benchmark run |
| #1 | NeRFactor2 | deep_learning | reference | 35.9 | 0.938 | — | pwm.platformai.org |
| #2 | Instant-NGP | deep_learning | reference | 34.5 | 0.924 | — | pwm.platformai.org |
| #3 | 3D Gaussian Splatting | deep_learning | reference | 33.2 | 0.910 | — | pwm.platformai.org |
| #4 | TensoRF | deep_learning | reference | 32.0 | 0.896 | — | pwm.platformai.org |
| #5 | Mip-NeRF 360 | deep_learning | reference | 30.8 | 0.881 | — | pwm.platformai.org |
| #6 | NeRF (vanilla) | deep_learning | reference | 29.5 | 0.863 | — | pwm.platformai.org |

### Quantum Imaging

#### entangled_photon — Entangled Photon Microscopy

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 31.82 | 0.9688 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | QGI-DL | deep_learning | pending | — | — | — | YAML config |

#### ghost_imaging — Ghost Imaging

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 6.63 | 0.1947 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | GI-Net | deep_learning | pending | — | — | — | YAML config |
| #1 | Ghost-ViT | deep_learning | reference | 30.1 | 0.892 | — | pwm.platformai.org |
| #2 | Ghost-FM | deep_learning | reference | 28.9 | 0.878 | — | pwm.platformai.org |
| #3 | U-Net-Ghost | deep_learning | reference | 27.5 | 0.861 | — | pwm.platformai.org |
| #4 | Differential Ghost Imaging | traditional | reference | 25.2 | 0.834 | — | pwm.platformai.org |
| #5 | Compressed Sensing GI | traditional | reference | 23.8 | 0.812 | — | pwm.platformai.org |
| #6 | Correlation GI | traditional | reference | 21.9 | 0.784 | — | pwm.platformai.org |

#### quantum_illumination — Quantum Illumination

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 20.22 | 0.7859 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | QI-DL | deep_learning | pending | — | — | — | YAML config |

### Remote Sensing

#### gpr — Ground-Penetrating Radar (GPR)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 10.60 | 0.0059 | 0.00 | benchmark run |
| 2 | RDA | traditional | pending | — | — | — | YAML config |
| 3 | SAR-DL | deep_learning | pending | — | — | — | YAML config |
| 4 | GPR-Net | deep_learning | pending | — | — | — | YAML config |

#### hyperspectral_remote — Hyperspectral Remote Sensing

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 29.15 | 0.9768 | 0.00 | benchmark run |
| 2 | RDA | traditional | pending | — | — | — | YAML config |
| 3 | SAR-DL | deep_learning | pending | — | — | — | YAML config |
| 4 | SST-USRNet | deep_learning | pending | — | — | — | YAML config |

#### insar — Interferometric SAR (InSAR)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Wrapped Phase Baseline | traditional | **done** | 31.83 | 0.9933 | 0.00 | benchmark run |
| 2 | RDA | traditional | pending | — | — | — | YAML config |
| 3 | SAR-DL | deep_learning | pending | — | — | — | YAML config |
| 4 | InSAR-Net | deep_learning | pending | — | — | — | YAML config |
| #1 | InSAR-FM | deep_learning | reference | 32.5 | 0.918 | — | pwm.platformai.org |
| #2 | DeepInSAR | deep_learning | reference | 30.8 | 0.899 | — | pwm.platformai.org |
| #3 | SNAPHU (Statistical) | traditional | reference | 28.2 | 0.871 | — | pwm.platformai.org |
| #4 | Goldstein Phase Unwrap | traditional | reference | 25.9 | 0.843 | — | pwm.platformai.org |

#### multispectral_sat — Multispectral Satellite Imaging

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Bicubic Upsampling | traditional | **done** | 10.79 | 0.1002 | 0.01 | benchmark run |
| 2 | RDA | traditional | pending | — | — | — | YAML config |
| 3 | SAR-DL | deep_learning | pending | — | — | — | YAML config |
| 4 | MS-Pansharpening-DL | deep_learning | pending | — | — | — | YAML config |

#### ocean_color — Ocean Color Remote Sensing

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 44.10 | 0.9998 | 0.00 | benchmark run |
| 2 | RDA | traditional | pending | — | — | — | YAML config |
| 3 | SAR-DL | deep_learning | pending | — | — | — | YAML config |
| 4 | OC-Net | deep_learning | pending | — | — | — | YAML config |

#### passive_microwave — Passive Microwave Radiometry

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 9.19 | 0.5946 | 0.00 | benchmark run |
| 2 | RDA | traditional | pending | — | — | — | YAML config |
| 3 | SAR-DL | deep_learning | pending | — | — | — | YAML config |
| 4 | PM-Net | deep_learning | pending | — | — | — | YAML config |

#### polsar — Polarimetric SAR (PolSAR)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 3.45 | -0.0175 | 0.00 | benchmark run |
| 2 | RDA | traditional | pending | — | — | — | YAML config |
| 3 | SAR-DL | deep_learning | pending | — | — | — | YAML config |
| 4 | PolSAR-Net | deep_learning | pending | — | — | — | YAML config |
| #1 | PolSAR-FM | deep_learning | reference | 36.2 | 0.942 | — | pwm.platformai.org |
| #2 | DiffusionPolSAR | deep_learning | reference | 34.9 | 0.930 | — | pwm.platformai.org |
| #3 | PolSAR-Transformer | deep_learning | reference | 33.6 | 0.917 | — | pwm.platformai.org |
| #4 | PolSAR-CNN | deep_learning | reference | 32.4 | 0.904 | — | pwm.platformai.org |
| #5 | Coherency Matrix BM3D | traditional | reference | 30.8 | 0.887 | — | pwm.platformai.org |
| #6 | PolSAR NL-Filter | traditional | reference | 29.5 | 0.871 | — | pwm.platformai.org |
| #7 | Refined Lee | traditional | reference | 28.2 | 0.854 | — | pwm.platformai.org |
| #8 | Boxcar (PolSAR) | traditional | reference | 26.9 | 0.836 | — | pwm.platformai.org |
| #9 | IDAN Filter | traditional | reference | 25.7 | 0.817 | — | pwm.platformai.org |
| #10 | Lee Filter | traditional | reference | 24.5 | 0.797 | — | pwm.platformai.org |
| #11 | Gamma-MAP | traditional | reference | 23.3 | 0.776 | — | pwm.platformai.org |
| #12 | Kuan Filter | traditional | reference | 22.1 | 0.754 | — | pwm.platformai.org |
| #13 | Pauli Decomposition | traditional | reference | 20.9 | 0.731 | — | pwm.platformai.org |

#### radio_interferometry — Radio Interferometry (VLBI)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 23.25 | 0.2029 | 0.00 | benchmark run |
| 2 | RDA | traditional | pending | — | — | — | YAML config |
| 3 | SAR-DL | deep_learning | pending | — | — | — | YAML config |
| 4 | R2D2 (interferometry) | deep_learning | pending | — | — | — | YAML config |
| #1 | RI-FM | deep_learning | reference | 35.5 | 0.936 | — | pwm.platformai.org |
| #2 | CLEAN-RI-Net | deep_learning | reference | 33.8 | 0.920 | — | pwm.platformai.org |
| #3 | WSCLEAN-RI | traditional | reference | 32.1 | 0.902 | — | pwm.platformai.org |
| #4 | MS-CLEAN-RI | traditional | reference | 30.4 | 0.882 | — | pwm.platformai.org |
| #5 | CLEAN (Hogbom) | traditional | reference | 28.8 | 0.860 | — | pwm.platformai.org |
| #6 | NUFFT Gridding | traditional | reference | 27.1 | 0.836 | — | pwm.platformai.org |

#### sar — Synthetic Aperture Radar (SAR)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 17.31 | 0.7046 | 0.00 | benchmark run |
| 2 | FBP (SAR backprojection) | traditional | **done** | 13.56 | 0.2879 | 0.16 | benchmark run |
| 3 | SAR-DL (PolSF) | deep_learning | pending | — | — | — | YAML config |
| 4 | SAR-CNN | deep_learning | pending | — | — | — | YAML config |
| #1 | DiffusionSAR | deep_learning | reference | 35.4 | 0.938 | — | pwm.platformai.org |
| #2 | SAR-FM | deep_learning | reference | 34.2 | 0.927 | — | pwm.platformai.org |
| #3 | SAR-Transformer | deep_learning | reference | 33.0 | 0.915 | — | pwm.platformai.org |
| #4 | SAR-CNN | deep_learning | reference | 31.8 | 0.901 | — | pwm.platformai.org |
| #5 | ID-CNN | deep_learning | reference | 30.7 | 0.888 | — | pwm.platformai.org |
| #6 | SAR-BM3D | traditional | reference | 29.5 | 0.873 | — | pwm.platformai.org |
| #7 | NL-SAR | traditional | reference | 28.3 | 0.858 | — | pwm.platformai.org |
| #8 | PPB (SAR) | traditional | reference | 27.2 | 0.842 | — | pwm.platformai.org |
| #9 | NLSAR | traditional | reference | 26.1 | 0.826 | — | pwm.platformai.org |
| #10 | Lee Filter | traditional | reference | 25.0 | 0.809 | — | pwm.platformai.org |
| #11 | Frost Filter | traditional | reference | 23.9 | 0.791 | — | pwm.platformai.org |
| #12 | Kuan Filter | traditional | reference | 22.8 | 0.772 | — | pwm.platformai.org |
| #13 | Boxcar Filter | traditional | reference | 21.7 | 0.751 | — | pwm.platformai.org |

#### sonar — Sonar Imaging

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 10.32 | 0.5149 | 0.00 | benchmark run |
| 2 | FISTA-L2 (DAS) | traditional | pending | — | — | — | YAML config |
| 3 | SonarSR-Net | deep_learning | pending | — | — | — | YAML config |
| 4 | Sonar-CNN | deep_learning | pending | — | — | — | YAML config |

#### weather_radar — Weather / Doppler Radar

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 26.85 | 0.9155 | 0.00 | benchmark run |
| 2 | RDA | traditional | pending | — | — | — | YAML config |
| 3 | SAR-DL | deep_learning | pending | — | — | — | YAML config |
| 4 | NowcastNet | deep_learning | pending | — | — | — | YAML config |

### Scanning Probe Microscopy

#### afm — Atomic Force Microscopy (AFM)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | CARE | deep_learning | **done** | 15.58 | 0.3770 | 1.41 | benchmark run |
| 2 | Precomputed Baseline | traditional | **done** | 19.01 | 0.8537 | 0.00 | benchmark run |
| 3 | Richardson-Lucy | traditional | **done** | 31.32 | 0.7815 | 0.00 | benchmark run |
| 4 | AFM-UNet | deep_learning | pending | — | — | — | YAML config |
| #1 | AFM-FM | deep_learning | reference | 38.5 | 0.955 | — | pwm.platformai.org |
| #2 | AFM-UNet | deep_learning | reference | 36.8 | 0.940 | — | pwm.platformai.org |
| #3 | CARE (AFM) | deep_learning | reference | 35.2 | 0.924 | — | pwm.platformai.org |
| #4 | BM3D-AFM | traditional | reference | 33.5 | 0.906 | — | pwm.platformai.org |
| #5 | Richardson-Lucy (AFM) | traditional | reference | 31.8 | 0.885 | — | pwm.platformai.org |
| #6 | Wiener Deconvolution | traditional | reference | 29.5 | 0.858 | — | pwm.platformai.org |
| #7 | Median Filter | traditional | reference | 26.8 | 0.821 | — | pwm.platformai.org |

#### mfm — Magnetic Force Microscopy (MFM)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | CARE | deep_learning | **done** | 5.06 | 0.0008 | 0.46 | benchmark run |
| 2 | Precomputed Baseline | traditional | **done** | 34.33 | 0.2871 | 0.00 | benchmark run |
| 3 | Richardson-Lucy | traditional | **done** | 9.52 | 0.3740 | 0.00 | benchmark run |
| 4 | MFM-UNet | deep_learning | pending | — | — | — | YAML config |

#### nsom — Near-field Scanning Optical Microscopy (NSOM)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | CARE | deep_learning | **done** | 17.68 | 0.7562 | 0.31 | benchmark run |
| 2 | Precomputed Baseline | traditional | **done** | 19.63 | 0.7328 | 0.00 | benchmark run |
| 3 | Richardson-Lucy | traditional | **done** | 22.28 | 0.2438 | 0.00 | benchmark run |
| 4 | NSOM-Net | deep_learning | pending | — | — | — | YAML config |

#### stm — Scanning Tunneling Microscopy (STM)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | CARE | deep_learning | **done** | 6.96 | 0.0000 | 0.45 | benchmark run |
| 2 | Precomputed Baseline | traditional | **done** | 17.89 | 0.8025 | 0.00 | benchmark run |
| 3 | Richardson-Lucy | traditional | **done** | 23.28 | 0.9600 | 0.00 | benchmark run |
| 4 | STM-Net | deep_learning | pending | — | — | — | YAML config |
| #1 | STM-FM | deep_learning | reference | 41.5 | 0.967 | — | pwm.platformai.org |
| #2 | STM-UNet | deep_learning | reference | 39.8 | 0.953 | — | pwm.platformai.org |
| #3 | CARE-STM | deep_learning | reference | 38.1 | 0.938 | — | pwm.platformai.org |
| #4 | BM3D-STM | traditional | reference | 36.2 | 0.920 | — | pwm.platformai.org |
| #5 | Wiener-STM | traditional | reference | 34.3 | 0.900 | — | pwm.platformai.org |
| #6 | Richardson-Lucy | traditional | reference | 32.4 | 0.878 | — | pwm.platformai.org |

### Scientific Instrumentation

#### atom_probe — Atom Probe Tomography (APT)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 40.23 | 0.9878 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | APT-Net | deep_learning | pending | — | — | — | YAML config |

#### cathodoluminescence — Cathodoluminescence (CL) Imaging

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 28.87 | 0.9772 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | CL-Net | deep_learning | pending | — | — | — | YAML config |

#### cryo_em — Cryo-EM Single Particle Analysis

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Wiener Filter (precomputed) | traditional | **done** | 19.21 | 0.0300 | 0.00 | benchmark run |
| 2 | Richardson-Lucy + CTF (20 iter) | traditional | **done** | 13.18 | 0.4136 | 0.23 | benchmark run |
| 3 | Adjoint | traditional | pending | — | — | — | YAML config |
| 4 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 5 | CryoDRGN | deep_learning | pending | — | — | — | YAML config |
| #1 | DiffusionCryo | deep_learning | reference | 39.8 | 0.961 | — | pwm.platformai.org |
| #2 | CryoDRGN-2 | deep_learning | reference | 38.2 | 0.948 | — | pwm.platformai.org |
| #3 | CryoSPARC-NN | deep_learning | reference | 36.9 | 0.935 | — | pwm.platformai.org |
| #4 | CryoDRGN | deep_learning | reference | 35.5 | 0.921 | — | pwm.platformai.org |
| #5 | RELION-4 (DL) | deep_learning | reference | 34.2 | 0.907 | — | pwm.platformai.org |
| #6 | RELION-4 (ADMM) | traditional | reference | 32.8 | 0.891 | — | pwm.platformai.org |
| #7 | CryoSPARC (MAPEM) | traditional | reference | 31.5 | 0.875 | — | pwm.platformai.org |
| #8 | RELION (Gold-Std) | traditional | reference | 30.1 | 0.857 | — | pwm.platformai.org |
| #9 | Wiener + CTF correction | traditional | reference | 28.4 | 0.834 | — | pwm.platformai.org |
| #10 | Phase Flipping + CTF | traditional | reference | 26.8 | 0.810 | — | pwm.platformai.org |

#### maldi_msi — MALDI Mass Spectrometry Imaging

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 26.30 | 0.9418 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | MSI-UNet | deep_learning | pending | — | — | — | YAML config |

#### muon_tomo — Muon Tomography

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 5.17 | -0.0128 | 0.00 | benchmark run |
| 2 | FBP (muon tomography) | traditional | **done** | 3.15 | 0.0019 | 0.16 | benchmark run |
| 3 | POCA-DL | deep_learning | pending | — | — | — | YAML config |
| 4 | EM-POCA | deep_learning | pending | — | — | — | YAML config |

#### neutron_diffraction — Neutron Diffraction

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 8.55 | 0.0116 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | NeutronDiff-Net | deep_learning | pending | — | — | — | YAML config |

#### neutron_tomo — Neutron Radiography / Tomography

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | -5.66 | 0.0503 | 0.00 | benchmark run |
| 2 | FBP (neutron tomography) | traditional | **done** | 4.35 | 0.0210 | 0.09 | benchmark run |
| 3 | NeuTomo-DL | deep_learning | pending | — | — | — | YAML config |
| 4 | GRIDREC-Neutron | deep_learning | pending | — | — | — | YAML config |

#### proton_radiography — Proton Radiography

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 4.11 | -0.0000 | 0.00 | benchmark run |
| 2 | FBP (proton radiography) | traditional | **done** | 10.85 | 0.0397 | 0.06 | benchmark run |
| 3 | ProtonRecon-Net | deep_learning | pending | — | — | — | YAML config |
| 4 | FBP-Proton | deep_learning | pending | — | — | — | YAML config |

#### saxs — Small-Angle X-ray Scattering (SAXS)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 8.42 | 0.0542 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | SAXS-VAE | deep_learning | pending | — | — | — | YAML config |

#### waxs — Wide-Angle X-ray Scattering (WAXS)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 20.63 | 0.0694 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | WAXS-Net | deep_learning | pending | — | — | — | YAML config |

#### xray_crystallography — X-ray Crystallography

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 22.37 | 0.0651 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | AlphaFold-SF | deep_learning | pending | — | — | — | YAML config |
| #1 | XC-FM | deep_learning | reference | 42.5 | 0.974 | — | pwm.platformai.org |
| #2 | PhaseNet-XC | deep_learning | reference | 40.8 | 0.961 | — | pwm.platformai.org |
| #3 | SHELX-DL | deep_learning | reference | 39.2 | 0.947 | — | pwm.platformai.org |
| #4 | Direct Methods (SIR) | traditional | reference | 37.5 | 0.931 | — | pwm.platformai.org |
| #5 | Patterson Synthesis | traditional | reference | 35.8 | 0.913 | — | pwm.platformai.org |
| #6 | Maximum Entropy Method | traditional | reference | 34.1 | 0.893 | — | pwm.platformai.org |

#### xrf_tomo — X-ray Fluorescence Tomography

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 15.64 | 0.8431 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | XRFT-Net | deep_learning | pending | — | — | — | YAML config |

### Spectroscopy & Spectral Imaging

#### brillouin — Brillouin Microscopy

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 35.84 | 0.9959 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | Brillouin-Net | deep_learning | pending | — | — | — | YAML config |

#### cars — Coherent Anti-Stokes Raman (CARS) Microscopy

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 14.19 | 0.0040 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | CARS-DeepSpec | deep_learning | pending | — | — | — | YAML config |

#### desi — DESI Mass Spectrometry Imaging

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 15.13 | 0.3130 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | DESI-SegNet | deep_learning | pending | — | — | — | YAML config |

#### ftir_imaging — FTIR Spectroscopic Imaging

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 14.78 | 0.8058 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | FTIR-UNet | deep_learning | pending | — | — | — | YAML config |

#### libs — Laser-Induced Breakdown Spectroscopy (LIBS) Imaging

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 18.02 | 0.5987 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | LIBS-CNN | deep_learning | pending | — | — | — | YAML config |

#### raman_imaging — Raman Imaging / Microscopy

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 14.11 | 0.2149 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | RamanNet | deep_learning | pending | — | — | — | YAML config |

#### sims — Secondary Ion Mass Spectrometry (SIMS) Imaging

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 20.50 | 0.9749 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | SIMS-Net | deep_learning | pending | — | — | — | YAML config |

#### srs — Stimulated Raman Scattering (SRS) Microscopy

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 29.08 | 0.9779 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | SRS-DeepSpec | deep_learning | pending | — | — | — | YAML config |

### Ultrafast Imaging

#### cup — Compressed Ultrafast Photography (CUP)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | -2.34 | 0.1202 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | E2E-CUP | traditional | pending | — | — | — | YAML config |

#### pump_probe — Pump-Probe Microscopy

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 18.24 | 0.7781 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | PumpProbe-Net | deep_learning | pending | — | — | — | YAML config |

#### streak_camera — Streak Camera Imaging

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 14.29 | 0.1114 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | StreakNet | deep_learning | pending | — | — | — | YAML config |

#### xfel_sfx — XFEL Serial Femtosecond Crystallography (SFX)

| # | Algorithm | Type | Status | PSNR (dB) | SSIM | Time(s) | Source |
|---|-----------|------|--------|-----------|------|---------|--------|
| 1 | Precomputed Baseline | traditional | **done** | 24.08 | 0.9753 | 0.00 | benchmark run |
| 2 | Adjoint | traditional | pending | — | — | — | YAML config |
| 3 | PnP-ADMM | deep_learning | pending | — | — | — | YAML config |
| 4 | SFX-Net | deep_learning | pending | — | — | — | YAML config |
