# Physics World Model (PWM) — Benchmark Overview

The **Physics World Model Benchmark** is a comprehensive evaluation platform for blind reconstruction algorithms across computational imaging modalities. It covers **65 imaging variants** spanning **21 scientific categories**, from medical CT and MRI to electron microscopy, quantum imaging, and neural rendering.

**Platform:** [https://pwm.platformai.org/benchmark](https://pwm.platformai.org/benchmark)

---

## Table of Contents

1. [The Blind Reconstruction Challenge](#1-the-blind-reconstruction-challenge)
2. [Benchmark Structure](#2-benchmark-structure)
3. [Scoring](#3-scoring)
4. [Complete Modality Catalog](#4-complete-modality-catalog)
5. [Forward Model Types](#5-forward-model-types)
6. [Noise Models](#6-noise-models)
7. [Dataset Format](#7-dataset-format)
8. [Getting Started](#8-getting-started)

---

## 1. The Blind Reconstruction Challenge

In real-world imaging, the actual system parameters differ from the ideal model assumed during reconstruction. The **Blind Reconstruction Challenge** tests whether algorithms can:

1. **Reconstruct** the original signal `x` from corrupted measurements `y = H(spec) @ x + noise`
2. **Estimate** the unknown mismatch parameters (the "spec") that caused the corruption

Each modality defines a set of physically meaningful mismatch parameters (e.g., center offset in CT, B0 inhomogeneity in MRI) with known ranges but unknown true values.

### Three-Tier Evaluation

| Tier | Access | Ground Truth | Cost | Purpose |
|------|--------|-------------|------|---------|
| **Public** | Full download | Yes (x_true + true_spec) | Free | Development and debugging |
| **Dev** | Download (blind) | No | Free | Blind evaluation, scored server-side |
| **Hidden** | Server-side only | No | 10 credits | Final leaderboard ranking |

All three tiers contain the same scenes but use **different mismatch realizations** (different true_spec values and noise seeds) to prevent overfitting across tiers.

---

## 2. Benchmark Structure

### Overview

- **65 imaging variants** across 21 categories
- **7 forward model types** (radon, kspace, psf, ctf, mask, tip)
- **4 noise models** (gaussian, poisson, poisson_gaussian, speckle)
- **3 tiers per variant** (public, dev, hidden) = **195 challenge datasets**
- **Unified scoring** across all modalities

### Category Summary

| Category | # Variants | Signal Shape | Noise Model | Forward Model |
|----------|-----------|-------------|-------------|---------------|
| Compressive | 5 | 256x256 | poisson_gaussian | mask |
| Medical | 15 | 128x128x64 | poisson | radon / kspace |
| Medical Ultrasound | 3 | 256x256 | speckle | psf |
| Coherent | 3 | 256x256 | gaussian | psf |
| Microscopy | 13 | 256x256 | poisson_gaussian | psf |
| Electron Microscopy | 8 | 512x512 | poisson | ctf |
| Clinical Optics | 4 | 256x256 | gaussian | psf |
| Computational | 2 | 128x128 | gaussian | psf |
| Computational Photography | 2 | 256x256 | poisson_gaussian | psf |
| Neural Rendering | 2 | 800x800 | gaussian | psf |
| Depth Imaging | 3 | 256x256 | gaussian | psf |
| Remote Sensing | 2 | 512x512 | speckle | kspace |
| Particle Imaging | 3 | 128x128x64 | poisson | radon |

---

## 3. Scoring

All variants use a unified scoring formula:

```
Score = 0.4 x PSNR_norm + 0.4 x SSIM + 0.2 x (1 - ||y - H_hat @ x_hat|| / ||y||)
```

| Component | Weight | Measures |
|-----------|--------|----------|
| **PSNR_norm** | 40% | Peak Signal-to-Noise Ratio (normalized) |
| **SSIM** | 40% | Structural Similarity Index |
| **Consistency** | 20% | Measurement fidelity under estimated model |

The consistency term rewards algorithms that not only reconstruct well but also correctly estimate the mismatch parameters.

---

## 4. Complete Modality Catalog

### Compressive Imaging (5 variants)

| Variant Key | Display Name | Full Name | Mismatch Parameters | Scenes |
|-------------|-------------|-----------|---------------------|--------|
| `sd_cassi` | SD-CASSI | Single-Disperser CASSI | mask_dx, mask_dy, mask_theta, disp_a1, disp_alpha, sigma_read, dark_current, gain | 10 |
| `cacti` | CACTI | Coded Aperture Compressive Temporal Imaging | mask_dx, mask_dy, mask_theta, clock_offset, duty_cycle, gain | 6 |
| `spc_block` | SPC-Block | Single-Pixel Camera (Block Sensing) | gain_alpha, sigma_y | 11 |
| `spc_kronecker` | SPC-Kronecker | Single-Pixel Camera (Kronecker Sensing) | gain_alpha, sigma_y | 11 |
| `matrix` | Matrix | Generic Matrix Sensing | matrix_perturb, gain, sigma_y | 5 |

**Baselines (SD-CASSI):** MST-L (20.83/0.744 PSNR/SSIM), HDNet (21.88/0.756), PnP-HSICNN (20.40/0.574), GAP-TV (20.96/0.612)

**Baselines (CACTI):** EfficientSCI (27.38/0.927), ELP-Unfolding (26.50/0.910), PnP-FFDNet (20.15/0.650), GAP-TV (14.81/0.303)

**Baselines (SPC):** ISTA-Net (27.45/0.760), HATNet (26.80/0.745), PnP-DRUNet (24.10/0.690), FISTA-TV (19.02/0.584)

### Medical Imaging — Projection-Based (11 variants)

| Variant Key | Display Name | Full Name | Mismatch Parameters |
|-------------|-------------|-----------|---------------------|
| `ct` | CT | X-ray Computed Tomography | center_offset, angle_error, beam_hardening, detector_tilt |
| `cbct` | CBCT | Cone-Beam CT | center_offset, source_dist, cone_angle, detector_tilt |
| `pet` | PET | Positron Emission Tomography | attenuation, scatter_frac, timing_res, normalization |
| `spect` | SPECT | Single Photon Emission CT | center_offset, collimator_septal, attenuation, scatter |
| `xray_radiography` | X-ray Radiography | X-ray Radiography | source_dist, beam_hardening, scatter |
| `mammography` | Mammography | Mammography | compression, anode_angle, scatter |
| `fluoroscopy` | Fluoroscopy | Fluoroscopy | motion_blur, lag, gain_drift |
| `angiography` | X-ray Angiography | X-ray Angiography | contrast_timing, motion, scatter |
| `dexa` | DEXA | Dual-Energy X-ray Absorptiometry | energy_offset, soft_tissue, beam_overlap |
| `dot` | DOT | Diffuse Optical Tomography | mu_a, mu_s, source_pos |
| `photoacoustic` | Photoacoustic | Photoacoustic Imaging | sos, fluence, sensor_response |

### Medical Imaging — Fourier-Based (4 variants)

| Variant Key | Display Name | Full Name | Mismatch Parameters |
|-------------|-------------|-----------|---------------------|
| `mri` | MRI | Magnetic Resonance Imaging | B0_inhomog, gradient_nonlin, coil_sensitivity, k_trajectory |
| `fmri` | fMRI | Functional MRI (BOLD) | B0_inhomog, head_motion, hemodynamic_delay, physiological_noise |
| `diffusion_mri` | Diffusion MRI | Diffusion MRI (DTI) | b_value_error, eddy_current, gradient_direction, susceptibility |
| `mrs` | MRS | MR Spectroscopy | linewidth, freq_drift, phase_error, baseline |

### Medical Ultrasound (3 variants)

| Variant Key | Display Name | Full Name | Mismatch Parameters |
|-------------|-------------|-----------|---------------------|
| `ultrasound` | Ultrasound | B-mode Ultrasound | sos, attenuation, element_sensitivity, phase_aberration |
| `doppler_ultrasound` | Doppler Ultrasound | Doppler Ultrasound | sos, doppler_angle, wall_filter, prf |
| `elastography` | Elastography | Shear-Wave Elastography | shear_speed, push_duration, tissue_viscosity |

### Coherent Imaging (3 variants)

| Variant Key | Display Name | Full Name | Mismatch Parameters |
|-------------|-------------|-----------|---------------------|
| `holography` | Holography | Digital Holographic Microscopy | wavelength, prop_distance, tilt |
| `ptychography` | Ptychography | Ptychographic Imaging | probe_error, position_error, partial_coherence |
| `phase_retrieval` | CDI | Coherent Diffractive Imaging | support, saturation, missing_center |

### Microscopy (13 variants)

| Variant Key | Display Name | Full Name | Mismatch Parameters |
|-------------|-------------|-----------|---------------------|
| `widefield` | Widefield | Widefield Fluorescence | psf_sigma, defocus, background |
| `widefield_lowdose` | Widefield Low-Dose | Low-Dose Widefield | psf_sigma, photon_budget, read_noise |
| `confocal_livecell` | Confocal Live-Cell | Confocal Live-Cell | pinhole, refractive_index, photobleaching |
| `confocal_3d` | Confocal 3D | Confocal 3D Z-Stack | z_step, spherical_aberr, refractive_index |
| `lightsheet` | Light-Sheet | Light-Sheet Fluorescence | sheet_thickness, sheet_tilt, stripe_artifact |
| `two_photon` | Two-Photon | Two-Photon Microscopy | pulse_width, gdd, scattering |
| `sted` | STED | STED Microscopy | depletion_power, donut_alignment, saturation_intensity |
| `tirf` | TIRF | TIRF Microscopy | incidence_angle, penetration_depth, refractive_index |
| `sim` | SIM | Structured Illumination | pattern_phase, pattern_freq, modulation_depth |
| `fpm` | FPM | Fourier Ptychographic Microscopy | led_position, na_error, defocus |
| `flim` | FLIM | Fluorescence Lifetime Imaging | irf_width, time_bin, afterpulsing |
| `palm_storm` | PALM/STORM | Single-Molecule Localization | psf_model, emitter_density, drift |
| `polarization` | Polarization | Polarization Microscopy | extinction_ratio, retardance, alignment |

### Electron Microscopy (8 variants)

| Variant Key | Display Name | Full Name | Mismatch Parameters |
|-------------|-------------|-----------|---------------------|
| `sem` | SEM | Scanning Electron Microscopy | beam_energy, stigmatism, working_distance |
| `tem` | TEM | Transmission Electron Microscopy | defocus, Cs, beam_tilt |
| `stem` | STEM | Scanning TEM | probe_size, convergence_angle, scan_distortion |
| `electron_tomography` | Electron Tomo | Electron Tomography | tilt_angle, tilt_axis, defocus_gradient |
| `electron_diffraction` | 4D-STEM | 4D-STEM Electron Diffraction | camera_length, center_offset, elliptical_distortion |
| `electron_holography` | Electron Holography | Electron Holography | biprism_voltage, fringe_spacing, partial_coherence |
| `ebsd` | EBSD | Electron Backscatter Diffraction | pattern_center, sample_tilt, detector_distance |
| `eels` | EELS | Electron Energy Loss Spectroscopy | energy_dispersion, zero_loss_shift, aberration |

### Clinical Optics (4 variants)

| Variant Key | Display Name | Full Name | Mismatch Parameters |
|-------------|-------------|-----------|---------------------|
| `oct` | OCT | Optical Coherence Tomography | dispersion, reference_delay, spectral_roll_off |
| `octa` | OCTA | OCT Angiography | inter_bscan_time, bulk_motion, decorrelation_threshold |
| `fundus` | Fundus | Fundus Camera | pupil_dilation, focus, vignetting |
| `endoscopy` | Endoscopy | Fiber Bundle Endoscopy | fiber_coupling, core_spacing, bending_loss |

### Computational Imaging (2 variants)

| Variant Key | Display Name | Full Name | Mismatch Parameters |
|-------------|-------------|-----------|---------------------|
| `light_field` | Light Field | Light Field Imaging | microlens_pitch, main_lens_f, vignetting |
| `integral` | Integral | Integral Photography | lens_pitch, gap_distance, aberration |

### Computational Photography (2 variants)

| Variant Key | Display Name | Full Name | Mismatch Parameters |
|-------------|-------------|-----------|---------------------|
| `lensless` | Lensless | Lensless (Diffuser Camera) | diffuser_psf, sensor_distance, wavelength |
| `panorama` | Panorama | Panorama Multi-Focus Fusion | focus_step, registration, exposure_variation |

### Neural Rendering (2 variants)

| Variant Key | Display Name | Full Name | Mismatch Parameters |
|-------------|-------------|-----------|---------------------|
| `nerf` | NeRF | Neural Radiance Fields | camera_pose, focal_length, distortion, exposure |
| `gaussian_splatting` | 3DGS | 3D Gaussian Splatting | camera_pose, focal_length, point_cloud_init |

### Depth Imaging (3 variants)

| Variant Key | Display Name | Full Name | Mismatch Parameters |
|-------------|-------------|-----------|---------------------|
| `tof_camera` | ToF Camera | Time-of-Flight Depth Camera | modulation_freq, multipath, phase_nonlinearity |
| `structured_light` | Structured Light | Structured-Light Depth Camera | baseline, pattern_distortion, ambient_ir |
| `lidar` | LiDAR | LiDAR Scanner | timing_jitter, beam_divergence, range_walk |

### Remote Sensing (2 variants)

| Variant Key | Display Name | Full Name | Mismatch Parameters |
|-------------|-------------|-----------|---------------------|
| `sar` | SAR | Synthetic Aperture Radar | motion_error, phase_error, range_cell_migration |
| `sonar` | Sonar | Sonar Imaging | sound_speed_profile, multipath, array_calibration |

### Particle Imaging (3 variants)

| Variant Key | Display Name | Full Name | Mismatch Parameters |
|-------------|-------------|-----------|---------------------|
| `neutron_tomo` | Neutron Tomo | Neutron Tomography | beam_spectrum, scatter_correction, rotation_offset |
| `proton_radiography` | Proton Radiography | Proton Radiography | energy_loss, scattering, range_straggling |
| `muon_tomo` | Muon Tomo | Muon Tomography | angular_resolution, momentum_estimate, detector_efficiency |

---

## 5. Forward Model Types

The benchmark uses 7 physics-based forward models to simulate measurements:

| Runner Type | Forward Model | H_ideal Format | Used By |
|-------------|--------------|----------------|---------|
| **radon** | Radon transform (sinogram projection) | Projection angles array | CT, CBCT, PET, SPECT, Particle Imaging |
| **kspace** | Fourier undersampling | k-space sampling mask | MRI, fMRI, Diffusion MRI, MRS, SAR, Sonar |
| **psf** | PSF convolution | PSF kernel | Microscopy, Clinical Optics, Ultrasound, Depth Imaging, Neural Rendering |
| **ctf** | Contrast Transfer Function | CTF parameter array | TEM, SEM, STEM, Electron Diffraction/Holography |
| **mask** | Binary coded aperture encoding | Binary mask | SD-CASSI, CACTI, SPC, Quantum, Ultrafast |
| **tip** | Scanning probe tip convolution | Tip kernel | AFM, STM, Scanning Probe |

---

## 6. Noise Models

| Model | Description | Parameters | Used By |
|-------|-------------|------------|---------|
| **gaussian** | Additive white Gaussian noise | sigma | Coherent, Clinical Optics, Computational, Depth, Neural Rendering |
| **poisson** | Shot noise (photon counting) | peak_counts | Medical (CT/PET/SPECT), Electron Microscopy, Astronomy, Quantum |
| **poisson_gaussian** | Combined shot + read noise | poisson_alpha, gaussian_sigma | Compressive (CASSI/CACTI), Microscopy, Ultrafast |
| **speckle** | Multiplicative Rayleigh noise | n_looks | Medical Ultrasound, Remote Sensing (SAR/Sonar) |

---

## 7. Dataset Format

All challenge datasets use HDF5 format:

```
file.h5
├── [file attributes]
│   ├── variant     = "ct"
│   ├── tier        = "public"
│   ├── version     = "1.0"
│   └── runner_type = "radon"
│
├── sample_00/
│   ├── y           [dataset]    — measurements (corrupted by mismatch + noise)
│   ├── H_ideal     [dataset]    — ideal forward model operator
│   ├── x_true      [dataset]    — ground truth (public + hidden only)
│   ├── spec_ranges [attribute]  — JSON: parameter ranges visible to contestant
│   ├── metadata    [attribute]  — JSON: scene info, dimensions, noise model
│   └── true_spec   [attribute]  — JSON: true mismatch values (public + hidden only)
│
├── sample_01/
│   └── ...
```

### Submission Format

```
submission.h5
├── sample_00/
│   ├── x_hat          [dataset]    — reconstructed signal
│   └── corrected_spec [attribute]  — JSON: estimated mismatch parameters
```

### Tier Data Visibility

| Field | Public | Dev | Hidden |
|-------|--------|-----|--------|
| y (measurements) | Yes | Yes | Server-side |
| H_ideal (forward model) | Yes | Yes | Server-side |
| spec_ranges (parameter ranges) | Yes | Yes | Server-side |
| metadata | Yes | Yes | Server-side |
| x_true (ground truth) | Yes | **No** | Server-side |
| true_spec (true parameters) | Yes | **No** | Server-side |

---

## 8. Getting Started

### Prerequisites

```bash
pip install numpy h5py scipy
```

### Example files

Download example algorithms and datasets from the platform:

- **CT Baseline Algorithm** (`ct_baseline_algorithm.py`) — Filtered Back-Projection with mismatch estimation
- **MRI Baseline Algorithm** (`mri_baseline_algorithm.py`) — Zero-filled IFFT + iterative soft-thresholding
- **Example HDF5 files** — Small 32x32 datasets for CT and MRI (public, dev, submission formats)

Available at: [https://pwm.platformai.org/static/examples/](https://pwm.platformai.org/static/examples/)

### Quick start

```bash
# Download a challenge dataset from the platform
# Then run the baseline:
python ct_baseline_algorithm.py ct_challenge_dev.h5 my_submission.h5

# Upload my_submission.h5 to the platform for scoring
```

### Detailed guide

See [GUIDE.md](platform/pwm_platform/static/examples/GUIDE.md) for step-by-step instructions on competition participation and dataset contribution.

---

## Citation

If you use the PWM benchmark in your research, please cite:

```
Physics World Model: A Unified Benchmark for Blind Reconstruction
Under Forward-Model Mismatch in Computational Imaging
```

## Contact

- **Platform:** [https://pwm.platformai.org](https://pwm.platformai.org)
- **Email:** platformaigpt@gmail.com
