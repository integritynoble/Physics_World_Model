# PWM Imaging Modality Registry

**168 modalities across 19 categories, registered in `packages/pwm_core/contrib/modalities.yaml`**

This document is the complete reference for all imaging modalities supported by the Physics World Model (PWM) framework. Each modality can be evaluated through LIP (Living Imaging Physics) Arena via `pwm evaluate --modality <id>`. For the evaluation protocol, see [targeting_system.md](targeting_system.md). For per-modality benchmark details (B1-B4), see [pwm_modality_benchmarks_detailed.md](pwm_modality_benchmarks_detailed.md).

---

## Validation Status Legend

| Status | Meaning |
|--------|---------|
| **Validated** | Full 4-Scenario Protocol completed; flagship paper results available; included in LIP Arena rolling baseline |
| **Registered** | OperatorGraph template exists in `modalities.yaml`; sealed-simulator ready; awaiting 4-Scenario validation |
| **Planned** | Category allocated; forward model specified; dataset acquisition in progress |

---

## 1. Microscopy (24 modalities)

| # | ID | Full Name | Forward Model | Default Solver | Status |
|---|-----|-----------|---------------|----------------|--------|
| 1 | `widefield` | Widefield Fluorescence Microscopy | linear_operator | richardson_lucy | Registered |
| 2 | `widefield_lowdose` | Low-Dose Widefield Microscopy | linear_operator | pnp_hqs | Registered |
| 3 | `confocal_livecell` | Confocal Live-Cell Microscopy | linear_operator | richardson_lucy | Registered |
| 4 | `confocal_3d` | Confocal 3D Z-Stack | linear_operator | richardson_lucy_3d | Registered |
| 5 | `sim` | Structured Illumination Microscopy (SIM) | linear_operator | wiener_sim | Registered |
| 6 | `lightsheet` | Light-Sheet Fluorescence Microscopy (LSFM) | linear_operator | fourier_notch_destripe | Registered |
| 7 | `flim` | Fluorescence Lifetime Imaging (FLIM) | nonlinear_operator | phasor | Registered |
| 8 | `fpm` | Fourier Ptychographic Microscopy (FPM) | nonlinear_operator | sequential_phase_retrieval | Registered |
| 9 | `two_photon` | Two-Photon / Multiphoton Microscopy | nonlinear_operator | richardson_lucy | Registered |
| 10 | `sted` | STED Microscopy | nonlinear_operator | richardson_lucy | Registered |
| 11 | `palm_storm` | PALM/STORM Single-Molecule Localization | nonlinear_operator | thunderstorm | Registered |
| 12 | `tirf` | TIRF Microscopy | linear_operator | richardson_lucy | Registered |
| 13 | `polarization` | Polarization Microscopy | linear_operator | pnp_hqs | Registered |
| 14 | `expansion` | Expansion Microscopy (ExM) | linear_operator | richardson_lucy | Registered |
| 15 | `minflux` | MINFLUX Nanoscopy | nonlinear_operator | mle_localization | Registered |
| 16 | `ism` | Image Scanning Microscopy (ISM) | linear_operator | pixel_reassignment | Registered |
| 17 | `phase_contrast` | Phase Contrast Microscopy | linear_operator | halo_removal | Registered |
| 18 | `dic` | Differential Interference Contrast (DIC) | linear_operator | dic_gradient_integration | Registered |
| 19 | `dark_field` | Dark-Field Microscopy | linear_operator | richardson_lucy | Registered |
| 20 | `lattice_lightsheet` | Lattice Light-Sheet Microscopy | linear_operator | richardson_lucy_3d | Registered |
| 21 | `shg` | Second Harmonic Generation (SHG) Microscopy | nonlinear_operator | pnp_hqs | Registered |
| 22 | `spinning_disk` | Spinning Disk Confocal Microscopy | linear_operator | richardson_lucy | Registered |
| 23 | `three_photon` | Three-Photon Microscopy | nonlinear_operator | richardson_lucy | Registered |
| 24 | `dna_paint` | DNA-PAINT Super-Resolution | nonlinear_operator | thunderstorm | Registered |

---

## 2. Compressive Imaging (4 modalities)

| # | ID | Full Name | Forward Model | Default Solver | Status |
|---|-----|-----------|---------------|----------------|--------|
| 25 | `cassi` | Coded Aperture Snapshot Spectral Imaging (CASSI) | linear_operator | mst | **Validated** |
| 26 | `spc` | Single-Pixel Camera (SPC) | explicit_matrix | pnp_fista | **Validated** |
| 27 | `cacti` | Coded Aperture Compressive Temporal Imaging (CACTI) | linear_operator | gap_tv | **Validated** |
| 28 | `matrix` | Generic Matrix Sensing | explicit_matrix | fista_l2 | Registered |

### Validated Datasets

| Modality | Simulation Benchmark | Experimental Data |
|----------|---------------------|-------------------|
| CASSI | KAIST TSA (10 scenes, 256x256x28, 450-650 nm) | TSA Real (5 scenes, 660x660x28) |
| SPC | Set11 (11 images, 256x256, 25% sampling) | -- |
| CACTI | 6 standard videos (256x256x8) | EfficientSCI Real (4 scenes, 512x512, cr=10) |

---

## 3. Medical Imaging (37 modalities)

| # | ID | Full Name | Forward Model | Default Solver | Status |
|---|-----|-----------|---------------|----------------|--------|
| 29 | `ct` | X-ray Computed Tomography (CT) | linear_operator | fbp | **Validated** |
| 30 | `mri` | Magnetic Resonance Imaging (MRI) | linear_operator | sense | **Validated** |
| 31 | `xray_radiography` | X-ray Radiography | nonlinear_operator | tv_fista | Registered |
| 32 | `ultrasound` | Ultrasound B-mode Imaging | linear_operator | das_beamform | Planned |
| 33 | `pet` | Positron Emission Tomography (PET) | linear_operator | mlem | Registered |
| 34 | `spect` | Single Photon Emission CT (SPECT) | linear_operator | mlem | Registered |
| 35 | `fluoroscopy` | Fluoroscopy | nonlinear_operator | tv_fista | Registered |
| 36 | `mammography` | Mammography | nonlinear_operator | tv_fista | Registered |
| 37 | `dexa` | Dual-Energy X-ray Absorptiometry (DEXA) | nonlinear_operator | dual_energy_decomposition | Registered |
| 38 | `cbct` | Cone-Beam Computed Tomography (CBCT) | linear_operator | fdk | Registered |
| 39 | `angiography` | X-ray Angiography | nonlinear_operator | dsa_subtraction | Registered |
| 40 | `dot` | Diffuse Optical Tomography (DOT) | linear_operator | born_approx | Registered |
| 41 | `photoacoustic` | Photoacoustic Imaging | linear_operator | back_projection | Planned |
| 42 | `oct` | Optical Coherence Tomography (OCT) | nonlinear_operator | fft_recon | Planned |
| 43 | `fmri` | Functional MRI (BOLD fMRI) | linear_operator | sense | Registered |
| 44 | `mrs` | MR Spectroscopy (MRS) | linear_operator | lcmodel | Registered |
| 45 | `diffusion_mri` | Diffusion MRI (DTI) | linear_operator | weighted_least_squares | Registered |
| 46 | `doppler_ultrasound` | Doppler Ultrasound | nonlinear_operator | autocorrelation_estimator | Registered |
| 47 | `elastography` | Shear-Wave Elastography | nonlinear_operator | time_of_flight_inversion | Registered |
| 48 | `endoscopy` | Fiber Bundle Endoscopy | nonlinear_operator | tv_fista | Registered |
| 49 | `fundus` | Fundus Camera | linear_operator | richardson_lucy | Registered |
| 50 | `octa` | OCT Angiography (OCTA) | nonlinear_operator | tv_fista | Registered |
| 51 | `proton_therapy_img` | Proton Therapy Imaging | nonlinear_operator | back_projection | Registered |
| 52 | `brachytherapy_img` | Brachytherapy Imaging | nonlinear_operator | tg43_dose | Registered |
| 53 | `portal_imaging` | Portal Imaging (EPID) | linear_operator | back_projection | Registered |
| 54 | `spectral_ct` | Photon-Counting Spectral CT | linear_operator | material_decomposition | Planned |
| 55 | `mr_elastography` | MR Elastography (MRE) | linear_operator | lfe_inversion | Registered |
| 56 | `cest_mri` | CEST MRI | nonlinear_operator | z_spectrum_fit | Registered |
| 57 | `asl_mri` | Arterial Spin Labeling (ASL) MRI | linear_operator | perfusion_quantification | Registered |
| 58 | `mra` | MR Angiography (MRA) | linear_operator | mip_recon | Registered |
| 59 | `swi` | Susceptibility-Weighted Imaging (SWI) | linear_operator | swi_phase_mask | Registered |
| 60 | `mr_fingerprinting` | MR Fingerprinting (MRF) | nonlinear_operator | dictionary_matching | Registered |
| 61 | `ivus` | Intravascular Ultrasound (IVUS) | nonlinear_operator | polar_recon | Registered |
| 62 | `ceus` | Contrast-Enhanced Ultrasound (CEUS) | nonlinear_operator | contrast_specific | Registered |
| 63 | `digital_breast_tomo` | Digital Breast Tomosynthesis (DBT) | linear_operator | back_projection | Registered |
| 64 | `confocal_endomicroscopy` | Confocal Laser Endomicroscopy (CLE) | nonlinear_operator | fiber_deconvolution | Registered |
| 65 | `nirs_brain` | Functional Near-Infrared Spectroscopy (fNIRS) | nonlinear_operator | modified_beer_lambert | Registered |

### Validated Datasets

| Modality | Simulation Benchmark | Experimental Data |
|----------|---------------------|-------------------|
| CT | -- | FIPS Walnut Micro-CT (1200 proj x 2296 det, Zenodo); Helsinki Tomography Challenge 2022 (721 proj x 560 det, Zenodo) |
| MRI | Synthetic 8-coil | M4Raw Multi-Coil Brain (256x256, 4 coils, R=2/R=4, Zenodo 8056074) |

---

## 4. Coherent Imaging (5 modalities)

| # | ID | Full Name | Forward Model | Default Solver | Status |
|---|-----|-----------|---------------|----------------|--------|
| 66 | `ptychography` | Ptychographic Imaging | nonlinear_operator | epie | **Validated** |
| 67 | `holography` | Digital Holographic Microscopy | nonlinear_operator | angular_spectrum | Registered |
| 68 | `phase_retrieval` | Coherent Diffractive Imaging / Phase Retrieval | nonlinear_operator | hio | Registered |
| 69 | `odt` | Optical Diffraction Tomography (ODT) | nonlinear_operator | rytov_inversion | Registered |
| 70 | `talbot_lau` | Talbot-Lau X-ray Grating Interferometry | nonlinear_operator | phase_stepping | Registered |

### Validated Datasets

| Modality | Simulation Benchmark | Experimental Data |
|----------|---------------------|-------------------|
| Ptychography | -- | 4D-STEM SrTiO3 [001] (128x128 scan, 300 kV, Zenodo 5113449) |

---

## 5. Computational Photography (5 modalities)

| # | ID | Full Name | Forward Model | Default Solver | Status |
|---|-----|-----------|---------------|----------------|--------|
| 71 | `lensless` | Lensless (Diffuser Camera) Imaging | linear_operator | admm_tv | **Validated** |
| 72 | `panorama` | Panorama Multi-Focus Fusion | linear_operator | laplacian_pyramid_fusion | Registered |
| 73 | `coded_exposure` | Coded Exposure / Flutter Shutter | linear_operator | wiener_deblur | Registered |
| 74 | `event_camera` | Event Camera / Dynamic Vision Sensor (DVS) | nonlinear_operator | event_to_frame | Registered |
| 75 | `hdr_imaging` | High Dynamic Range (HDR) Imaging | nonlinear_operator | hdr_merge | Registered |

### Validated Datasets

| Modality | Simulation Benchmark | Experimental Data |
|----------|---------------------|-------------------|
| Lensless | DiffuserCam (256x256) | -- |

---

## 6. Computational Optics (2 modalities)

| # | ID | Full Name | Forward Model | Default Solver | Status |
|---|-----|-----------|---------------|----------------|--------|
| 76 | `light_field` | Light Field Imaging | linear_operator | shift_and_sum | Registered |
| 77 | `integral` | Integral Photography | linear_operator | depth_estimation | Registered |

---

## 7. Neural Rendering (2 modalities)

| # | ID | Full Name | Forward Model | Default Solver | Status |
|---|-----|-----------|---------------|----------------|--------|
| 78 | `nerf` | Neural Radiance Fields (NeRF) | nonlinear_operator | nerf_mlp | Registered |
| 79 | `gaussian_splatting` | 3D Gaussian Splatting (3DGS) | nonlinear_operator | gaussian_splatting_3dgs | Registered |

---

## 8. Electron Microscopy (11 modalities)

| # | ID | Full Name | Forward Model | Default Solver | Status |
|---|-----|-----------|---------------|----------------|--------|
| 80 | `sem` | Scanning Electron Microscopy (SEM) | nonlinear_operator | direct_imaging | Registered |
| 81 | `tem` | Transmission Electron Microscopy (TEM) | nonlinear_operator | ctf_correction | Registered |
| 82 | `electron_tomography` | Electron Tomography | linear_operator | sirt | Registered |
| 83 | `stem` | Scanning Transmission Electron Microscopy (STEM) | linear_operator | direct_imaging | Registered |
| 84 | `electron_diffraction` | 4D-STEM Electron Diffraction | nonlinear_operator | ptychography_epie | Registered |
| 85 | `ebsd` | Electron Backscatter Diffraction (EBSD) | nonlinear_operator | hough_indexing | Registered |
| 86 | `eels` | Electron Energy Loss Spectroscopy (EELS) | linear_operator | fourier_ratio | Registered |
| 87 | `electron_holography` | Electron Holography | nonlinear_operator | fourier_sideband | Registered |
| 88 | `cryo_et` | Cryo-Electron Tomography (Cryo-ET) | linear_operator | sirt | Registered |
| 89 | `fib_sem` | Focused Ion Beam SEM (FIB-SEM) | nonlinear_operator | stack_alignment | Registered |
| 90 | `edx_mapping` | STEM-EDX Elemental Mapping | nonlinear_operator | cliff_lorimer | Registered |

---

## 9. Depth Imaging (5 modalities)

| # | ID | Full Name | Forward Model | Default Solver | Status |
|---|-----|-----------|---------------|----------------|--------|
| 91 | `tof_camera` | Time-of-Flight Depth Camera | nonlinear_operator | tv_fista | Registered |
| 92 | `lidar` | LiDAR Scanner | nonlinear_operator | point_cloud_recon | Registered |
| 93 | `structured_light` | Structured-Light Depth Camera | nonlinear_operator | phase_unwrap | Registered |
| 94 | `photometric_stereo` | Photometric Stereo | nonlinear_operator | normal_estimation | Registered |
| 95 | `flash_lidar` | Flash LiDAR | nonlinear_operator | depth_map_recon | Registered |

---

## 10. Remote Sensing (11 modalities)

| # | ID | Full Name | Forward Model | Default Solver | Status |
|---|-----|-----------|---------------|----------------|--------|
| 96 | `sar` | Synthetic Aperture Radar (SAR) | linear_operator | backprojection | Planned |
| 97 | `sonar` | Sonar Imaging | linear_operator | beamform_das | Registered |
| 98 | `hyperspectral_remote` | Hyperspectral Remote Sensing | linear_operator | spectral_unmixing | Registered |
| 99 | `multispectral_sat` | Multispectral Satellite Imaging | linear_operator | pan_sharpening | Registered |
| 100 | `gpr` | Ground-Penetrating Radar (GPR) | nonlinear_operator | migration | Registered |
| 101 | `weather_radar` | Weather / Doppler Radar | nonlinear_operator | reflectivity_estimation | Registered |
| 102 | `radio_interferometry` | Radio Interferometry (VLBI) | linear_operator | clean | Registered |
| 103 | `passive_microwave` | Passive Microwave Radiometry | linear_operator | deconvolution | Registered |
| 104 | `insar` | Interferometric SAR (InSAR) | nonlinear_operator | phase_unwrap | Registered |
| 105 | `polsar` | Polarimetric SAR (PolSAR) | nonlinear_operator | polarimetric_decomposition | Registered |
| 106 | `ocean_color` | Ocean Color Remote Sensing | nonlinear_operator | atmospheric_correction | Registered |

---

## 11. Industrial Inspection (10 modalities)

| # | ID | Full Name | Forward Model | Default Solver | Status |
|---|-----|-----------|---------------|----------------|--------|
| 107 | `industrial_ct` | Industrial X-ray CT | linear_operator | fbp | Registered |
| 108 | `xray_ndt` | X-ray NDT (Radiography) | nonlinear_operator | contrast_enhancement | Registered |
| 109 | `ultrasonic_phased_array` | Ultrasonic Phased Array (TFM/FMC) | linear_operator | total_focusing_method | Registered |
| 110 | `eddy_current` | Eddy Current Imaging | nonlinear_operator | impedance_inversion | Registered |
| 111 | `active_thermography` | Active Thermography (IR) | nonlinear_operator | thermal_diffusivity_inversion | Registered |
| 112 | `terahertz` | Terahertz Imaging (THz) | nonlinear_operator | deconvolution | Registered |
| 113 | `machine_vision` | Machine Vision / AOI | linear_operator | defect_detection | Registered |
| 114 | `xrf_imaging` | X-ray Fluorescence (XRF) Imaging | nonlinear_operator | element_quantification | Registered |
| 115 | `shearography` | Shearography | nonlinear_operator | phase_unwrap | Registered |
| 116 | `acoustic_microscopy` | Scanning Acoustic Microscopy (SAM) | nonlinear_operator | c_scan_recon | Registered |

---

## 12. Scientific Instrumentation (12 modalities)

| # | ID | Full Name | Forward Model | Default Solver | Status |
|---|-----|-----------|---------------|----------------|--------|
| 117 | `xray_crystallography` | X-ray Crystallography | nonlinear_operator | direct_methods | Registered |
| 118 | `saxs` | Small-Angle X-ray Scattering (SAXS) | nonlinear_operator | desmearing | Registered |
| 119 | `maldi_msi` | MALDI Mass Spectrometry Imaging | nonlinear_operator | peak_picking | Registered |
| 120 | `atom_probe` | Atom Probe Tomography (APT) | nonlinear_operator | trajectory_recon | Registered |
| 121 | `cryo_em` | Cryo-EM Single Particle Analysis | nonlinear_operator | ctf_3d_refinement | Registered |
| 122 | `neutron_tomo` | Neutron Radiography / Tomography | nonlinear_operator | filtered_back_projection | Registered |
| 123 | `proton_radiography` | Proton Radiography | nonlinear_operator | filtered_back_projection | Registered |
| 124 | `muon_tomo` | Muon Tomography | nonlinear_operator | poca_reconstruction | Registered |
| 125 | `waxs` | Wide-Angle X-ray Scattering (WAXS) | nonlinear_operator | azimuthal_integration | Registered |
| 126 | `xrf_tomo` | X-ray Fluorescence Tomography | nonlinear_operator | fbp_self_absorption | Registered |
| 127 | `neutron_diffraction` | Neutron Diffraction | nonlinear_operator | rietveld_refinement | Registered |
| 128 | `cathodoluminescence` | Cathodoluminescence (CL) Imaging | nonlinear_operator | hyperspectral_unmixing | Registered |

---

## 13. Broader Experimental Science (11 modalities)

| # | ID | Full Name | Forward Model | Default Solver | Status |
|---|-----|-----------|---------------|----------------|--------|
| 129 | `adaptive_optics` | Adaptive Optics (AO) Imaging | nonlinear_operator | psf_deconvolution | Registered |
| 130 | `seismic_tomo` | Seismic Tomography | linear_operator | travel_time_inversion | Registered |
| 131 | `gravitational_wave` | Gravitational Wave Detection | nonlinear_operator | matched_filter | Registered |
| 132 | `particle_calorimetry` | Particle Calorimetry | nonlinear_operator | energy_recon | Registered |
| 133 | `radio_astronomy` | Radio Aperture Synthesis | linear_operator | clean | Registered |
| 134 | `acoustic_emission` | Acoustic Emission Testing (AE) | nonlinear_operator | source_localization | Registered |
| 135 | `magnetic_particle` | Magnetic Particle Imaging (MPI) | linear_operator | system_function_inversion | Registered |
| 136 | `impedance_tomo` | Electrical Impedance Tomography (EIT) | nonlinear_operator | gauss_newton | Registered |
| 137 | `fwi` | Full-Waveform Inversion (FWI) | nonlinear_operator | adjoint_state | Planned |
| 138 | `ocean_acoustic_tomo` | Ocean Acoustic Tomography | linear_operator | travel_time_inversion | Planned |
| 139 | `bioluminescence_tomo` | Bioluminescence Tomography (BLT) | nonlinear_operator | diffusion_inversion | Registered |

---

## 14. Spectroscopy & Spectral Imaging (8 modalities)

| # | ID | Full Name | Forward Model | Default Solver | Status |
|---|-----|-----------|---------------|----------------|--------|
| 140 | `raman_imaging` | Raman Imaging / Microscopy | nonlinear_operator | spectral_unmixing | Registered |
| 141 | `cars` | Coherent Anti-Stokes Raman (CARS) Microscopy | nonlinear_operator | nrb_removal | Registered |
| 142 | `srs` | Stimulated Raman Scattering (SRS) Microscopy | nonlinear_operator | lock_in_demod | Registered |
| 143 | `ftir_imaging` | FTIR Spectroscopic Imaging | linear_operator | interferogram_fft | Registered |
| 144 | `libs` | Laser-Induced Breakdown Spectroscopy (LIBS) Imaging | nonlinear_operator | element_quantification | Registered |
| 145 | `brillouin` | Brillouin Microscopy | nonlinear_operator | lorentz_fit | Registered |
| 146 | `sims` | Secondary Ion Mass Spectrometry (SIMS) Imaging | nonlinear_operator | mass_image_recon | Registered |
| 147 | `desi` | DESI Mass Spectrometry Imaging | nonlinear_operator | mass_image_recon | Registered |

---

## 15. Ultrafast Imaging (4 modalities)

| # | ID | Full Name | Forward Model | Default Solver | Status |
|---|-----|-----------|---------------|----------------|--------|
| 148 | `streak_camera` | Streak Camera Imaging | nonlinear_operator | spatiotemporal_recon | Registered |
| 149 | `pump_probe` | Pump-Probe Microscopy | nonlinear_operator | transient_absorption | Registered |
| 150 | `cup` | Compressed Ultrafast Photography (CUP) | linear_operator | cup_recon | Registered |
| 151 | `xfel_sfx` | XFEL Serial Femtosecond Crystallography (SFX) | nonlinear_operator | indexing_merge | Registered |

---

## 16. Quantum Imaging (3 modalities)

| # | ID | Full Name | Forward Model | Default Solver | Status |
|---|-----|-----------|---------------|----------------|--------|
| 152 | `ghost_imaging` | Ghost Imaging | linear_operator | correlation_recon | Registered |
| 153 | `quantum_illumination` | Quantum Illumination | nonlinear_operator | quantum_detector | Planned |
| 154 | `entangled_photon` | Entangled Photon Microscopy | nonlinear_operator | coincidence_recon | Planned |

---

## 17. Multi-Modal Fusion (6 modalities)

| # | ID | Full Name | Forward Model | Default Solver | Status |
|---|-----|-----------|---------------|----------------|--------|
| 155 | `pet_ct` | PET/CT Fusion | linear_operator | joint_recon_petct | Registered |
| 156 | `pet_mr` | PET/MR Fusion | linear_operator | joint_recon_petmr | Registered |
| 157 | `spect_ct` | SPECT/CT Fusion | linear_operator | joint_recon_spectct | Registered |
| 158 | `us_mri` | US/MRI Fusion | nonlinear_operator | registration_fusion | Registered |
| 159 | `ct_fluorescence` | CT + Fluorescence (FLIT) | nonlinear_operator | joint_recon_flit | Registered |
| 160 | `clem` | Correlative Light-Electron Microscopy (CLEM) | nonlinear_operator | overlay_registration | Registered |

---

## 18. Scanning Probe Microscopy (4 modalities)

| # | ID | Full Name | Forward Model | Default Solver | Status |
|---|-----|-----------|---------------|----------------|--------|
| 161 | `afm` | Atomic Force Microscopy (AFM) | nonlinear_operator | tip_deconvolution | Registered |
| 162 | `stm` | Scanning Tunneling Microscopy (STM) | nonlinear_operator | ldos_normalization | Registered |
| 163 | `nsom` | Near-field Scanning Optical Microscopy (NSOM) | nonlinear_operator | near_field_recon | Registered |
| 164 | `mfm` | Magnetic Force Microscopy (MFM) | nonlinear_operator | lift_mode_recon | Registered |

---

## 19. Astronomy & Space Imaging (4 modalities)

| # | ID | Full Name | Forward Model | Default Solver | Status |
|---|-----|-----------|---------------|----------------|--------|
| 165 | `coronagraphy` | Stellar Coronagraphy | nonlinear_operator | adi_speckle_subtraction | Registered |
| 166 | `lucky_imaging` | Lucky Imaging | nonlinear_operator | shift_and_add | Registered |
| 167 | `eht_imaging` | Event Horizon Telescope (EHT) Imaging | linear_operator | rml_clean | Registered |
| 168 | `solar_imaging` | Solar EUV/X-ray Imaging | nonlinear_operator | dem_inversion | Registered |

---

## Summary

| Category | Count | Validated | Registered | Planned |
|----------|------:|----------:|-----------:|--------:|
| Microscopy | 24 | 0 | 24 | 0 |
| Compressive Imaging | 4 | 3 | 1 | 0 |
| Medical Imaging | 37 | 2 | 32 | 3 |
| Coherent Imaging | 5 | 1 | 4 | 0 |
| Computational Photography | 5 | 1 | 4 | 0 |
| Computational Optics | 2 | 0 | 2 | 0 |
| Neural Rendering | 2 | 0 | 2 | 0 |
| Electron Microscopy | 11 | 0 | 11 | 0 |
| Depth Imaging | 5 | 0 | 5 | 0 |
| Remote Sensing | 11 | 0 | 10 | 1 |
| Industrial Inspection | 10 | 0 | 10 | 0 |
| Scientific Instrumentation | 12 | 0 | 12 | 0 |
| Broader Experimental Science | 11 | 0 | 9 | 2 |
| Spectroscopy & Spectral Imaging | 8 | 0 | 8 | 0 |
| Ultrafast Imaging | 4 | 0 | 4 | 0 |
| Quantum Imaging | 3 | 0 | 1 | 2 |
| Multi-Modal Fusion | 6 | 0 | 6 | 0 |
| Scanning Probe Microscopy | 4 | 0 | 4 | 0 |
| Astronomy & Space Imaging | 4 | 0 | 4 | 0 |
| **Total** | **168** | **7** | **153** | **8** |

**7 validated modalities** form the current LIP Arena rolling baseline: CASSI, CACTI, SPC, Lensless, CT, Ptychography, MRI.

All 168 modalities share the same OperatorGraph IR, the same 11 physical primitives, the same Triad decomposition, and the same 4-Scenario Protocol. A solver that works on one modality can be submitted to LIP Arena for any modality -- the framework is universal.
