# PWM Imaging Modality Registry

**64 modalities across 13 categories, registered in `packages/pwm_core/contrib/modalities.yaml`**

This document is the complete reference for all imaging modalities supported by the Physics World Model (PWM) framework. Each modality can be evaluated through LIP (Living Imaging Physics) Arena via `pwm evaluate --modality <id>`. For the evaluation protocol, see [targeting_system.md](targeting_system.md).

---

## Validation Status Legend

| Status | Meaning |
|--------|---------|
| **Validated** | Full 4-Scenario Protocol completed; flagship paper results available; included in LIP Arena rolling baseline |
| **Registered** | OperatorGraph template exists in `modalities.yaml`; sealed-simulator ready; awaiting 4-Scenario validation |
| **Planned** | Category allocated; forward model specified; dataset acquisition in progress |

---

## 1. Microscopy (13 modalities)

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

---

## 2. Compressive Imaging (4 modalities)

| # | ID | Full Name | Forward Model | Default Solver | Status |
|---|-----|-----------|---------------|----------------|--------|
| 14 | `cassi` | Coded Aperture Snapshot Spectral Imaging (CASSI) | linear_operator | mst | **Validated** |
| 15 | `spc` | Single-Pixel Camera (SPC) | explicit_matrix | pnp_fista | **Validated** |
| 16 | `cacti` | Coded Aperture Compressive Temporal Imaging (CACTI) | linear_operator | gap_tv | **Validated** |
| 17 | `matrix` | Generic Matrix Sensing | explicit_matrix | fista_l2 | Registered |

### Validated Datasets

| Modality | Simulation Benchmark | Experimental Data |
|----------|---------------------|-------------------|
| CASSI | KAIST TSA (10 scenes, 256x256x28, 450-650 nm) | TSA Real (5 scenes, 660x660x28) |
| SPC | Set11 (11 images, 256x256, 25% sampling) | -- |
| CACTI | 6 standard videos (256x256x8) | EfficientSCI Real (4 scenes, 512x512, cr=10) |

---

## 3. Medical Imaging (17 modalities)

| # | ID | Full Name | Forward Model | Default Solver | Status |
|---|-----|-----------|---------------|----------------|--------|
| 18 | `ct` | X-ray Computed Tomography (CT) | linear_operator | fbp | **Validated** |
| 19 | `mri` | Magnetic Resonance Imaging (MRI) | linear_operator | sense | **Validated** |
| 20 | `xray_radiography` | X-ray Radiography | nonlinear_operator | tv_fista | Registered |
| 21 | `ultrasound` | Ultrasound Imaging | linear_operator | tv_fista | Planned |
| 22 | `pet` | Positron Emission Tomography (PET) | linear_operator | mlem | Registered |
| 23 | `spect` | Single Photon Emission CT (SPECT) | linear_operator | mlem | Registered |
| 24 | `fluoroscopy` | Fluoroscopy | nonlinear_operator | tv_fista | Registered |
| 25 | `mammography` | Mammography | nonlinear_operator | tv_fista | Registered |
| 26 | `dexa` | Dual-Energy X-ray Absorptiometry (DEXA) | nonlinear_operator | dual_energy_decomposition | Registered |
| 27 | `cbct` | Cone-Beam Computed Tomography (CBCT) | linear_operator | fdk | Registered |
| 28 | `angiography` | X-ray Angiography | nonlinear_operator | dsa_subtraction | Registered |
| 29 | `dot` | Diffuse Optical Tomography (DOT) | linear_operator | born_approx | Registered |
| 30 | `photoacoustic` | Photoacoustic Imaging | linear_operator | back_projection | Planned |
| 31 | `oct` | Optical Coherence Tomography (OCT) | nonlinear_operator | fft_recon | Planned |
| 32 | `fmri` | Functional MRI (BOLD fMRI) | linear_operator | sense | Registered |
| 33 | `mrs` | MR Spectroscopy (MRS) | linear_operator | lcmodel | Registered |
| 34 | `diffusion_mri` | Diffusion MRI (DTI) | linear_operator | weighted_least_squares | Registered |

### Validated Datasets

| Modality | Simulation Benchmark | Experimental Data |
|----------|---------------------|-------------------|
| CT | -- | FIPS Walnut Micro-CT (1200 proj x 2296 det, Zenodo); Helsinki Tomography Challenge 2022 (721 proj x 560 det, Zenodo) |
| MRI | Synthetic 8-coil | M4Raw Multi-Coil Brain (256x256, 4 coils, R=2/R=4, Zenodo 8056074) |

---

## 4. Coherent Imaging (3 modalities)

| # | ID | Full Name | Forward Model | Default Solver | Status |
|---|-----|-----------|---------------|----------------|--------|
| 35 | `ptychography` | Ptychographic Imaging | nonlinear_operator | epie | **Validated** |
| 36 | `holography` | Digital Holographic Microscopy | nonlinear_operator | angular_spectrum | Registered |
| 37 | `phase_retrieval` | Coherent Diffractive Imaging / Phase Retrieval | nonlinear_operator | hio | Registered |

### Validated Datasets

| Modality | Simulation Benchmark | Experimental Data |
|----------|---------------------|-------------------|
| Ptychography | -- | 4D-STEM SrTiO3 [001] (128x128 scan, 300 kV, Zenodo 5113449) |

---

## 5. Neural Rendering (2 modalities)

| # | ID | Full Name | Forward Model | Default Solver | Status |
|---|-----|-----------|---------------|----------------|--------|
| 38 | `nerf` | Neural Radiance Fields (NeRF) | nonlinear_operator | nerf_mlp | Registered |
| 39 | `gaussian_splatting` | 3D Gaussian Splatting (3DGS) | nonlinear_operator | gaussian_splatting_3dgs | Registered |

---

## 6. Computational Photography (2 modalities)

| # | ID | Full Name | Forward Model | Default Solver | Status |
|---|-----|-----------|---------------|----------------|--------|
| 40 | `lensless` | Lensless (Diffuser Camera) Imaging | linear_operator | admm_tv | **Validated** |
| 41 | `panorama` | Panorama Multi-Focus Fusion | linear_operator | laplacian_pyramid_fusion | Registered |

### Validated Datasets

| Modality | Simulation Benchmark | Experimental Data |
|----------|---------------------|-------------------|
| Lensless | DiffuserCam (256x256) | -- |

---

## 7. Computational (2 modalities)

| # | ID | Full Name | Forward Model | Default Solver | Status |
|---|-----|-----------|---------------|----------------|--------|
| 42 | `light_field` | Light Field Imaging | linear_operator | shift_and_sum | Registered |
| 43 | `integral` | Integral Photography | linear_operator | depth_estimation | Registered |

---

## 8. Electron Microscopy (8 modalities)

| # | ID | Full Name | Forward Model | Default Solver | Status |
|---|-----|-----------|---------------|----------------|--------|
| 44 | `sem` | Scanning Electron Microscopy (SEM) | nonlinear_operator | direct_imaging | Registered |
| 45 | `tem` | Transmission Electron Microscopy (TEM) | nonlinear_operator | ctf_correction | Registered |
| 46 | `electron_tomography` | Electron Tomography | linear_operator | sirt | Registered |
| 47 | `stem` | Scanning Transmission Electron Microscopy (STEM) | linear_operator | direct_imaging | Registered |
| 48 | `electron_diffraction` | 4D-STEM Electron Diffraction | nonlinear_operator | ptychography_epie | Registered |
| 49 | `ebsd` | Electron Backscatter Diffraction (EBSD) | nonlinear_operator | hough_indexing | Registered |
| 50 | `eels` | Electron Energy Loss Spectroscopy (EELS) | linear_operator | fourier_ratio | Registered |
| 51 | `electron_holography` | Electron Holography | nonlinear_operator | fourier_sideband | Registered |

---

## 9. Clinical Optics (3 modalities)

| # | ID | Full Name | Forward Model | Default Solver | Status |
|---|-----|-----------|---------------|----------------|--------|
| 52 | `endoscopy` | Fiber Bundle Endoscopy | nonlinear_operator | tv_fista | Registered |
| 53 | `fundus` | Fundus Camera | linear_operator | richardson_lucy | Registered |
| 54 | `octa` | OCT Angiography (OCTA) | nonlinear_operator | tv_fista | Registered |

---

## 10. Depth Imaging (3 modalities)

| # | ID | Full Name | Forward Model | Default Solver | Status |
|---|-----|-----------|---------------|----------------|--------|
| 55 | `tof_camera` | Time-of-Flight Depth Camera | nonlinear_operator | tv_fista | Registered |
| 56 | `lidar` | LiDAR Scanner | nonlinear_operator | tv_fista | Registered |
| 57 | `structured_light` | Structured-Light Depth Camera | nonlinear_operator | phase_unwrap | Registered |

---

## 11. Medical Ultrasound (2 modalities)

| # | ID | Full Name | Forward Model | Default Solver | Status |
|---|-----|-----------|---------------|----------------|--------|
| 58 | `doppler_ultrasound` | Doppler Ultrasound | nonlinear_operator | autocorrelation_estimator | Registered |
| 59 | `elastography` | Shear-Wave Elastography | nonlinear_operator | time_of_flight_inversion | Registered |

---

## 12. Remote Sensing (2 modalities)

| # | ID | Full Name | Forward Model | Default Solver | Status |
|---|-----|-----------|---------------|----------------|--------|
| 60 | `sar` | Synthetic Aperture Radar (SAR) | linear_operator | backprojection | Planned |
| 61 | `sonar` | Sonar Imaging | linear_operator | beamform_das | Registered |

---

## 13. Particle Imaging (3 modalities)

| # | ID | Full Name | Forward Model | Default Solver | Status |
|---|-----|-----------|---------------|----------------|--------|
| 62 | `neutron_tomo` | Neutron Radiography / Tomography | nonlinear_operator | filtered_back_projection | Registered |
| 63 | `proton_radiography` | Proton Radiography | nonlinear_operator | filtered_back_projection | Registered |
| 64 | `muon_tomo` | Muon Tomography | nonlinear_operator | poca_reconstruction | Registered |

---

## Summary

| Category | Count | Validated | Registered | Planned |
|----------|------:|----------:|-----------:|--------:|
| Microscopy | 13 | 0 | 13 | 0 |
| Compressive Imaging | 4 | 3 | 1 | 0 |
| Medical Imaging | 17 | 2 | 12 | 3 |
| Coherent Imaging | 3 | 1 | 2 | 0 |
| Neural Rendering | 2 | 0 | 2 | 0 |
| Computational Photography | 2 | 1 | 1 | 0 |
| Computational | 2 | 0 | 2 | 0 |
| Electron Microscopy | 8 | 0 | 8 | 0 |
| Clinical Optics | 3 | 0 | 3 | 0 |
| Depth Imaging | 3 | 0 | 3 | 0 |
| Medical Ultrasound | 2 | 0 | 2 | 0 |
| Remote Sensing | 2 | 0 | 1 | 1 |
| Particle Imaging | 3 | 0 | 3 | 0 |
| **Total** | **64** | **7** | **53** | **4** |

**7 validated modalities** form the current LIP Arena rolling baseline: CASSI, CACTI, SPC, Lensless, CT, Ptychography, MRI.

All 64 modalities share the same OperatorGraph IR, the same 11 physical primitives, the same Triad decomposition, and the same 4-Scenario Protocol. A solver that works on one modality can be submitted to LIP Arena for any modality -- the framework is universal.
