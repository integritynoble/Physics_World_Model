# PWM Benchmark — Internal Reference (CONFIDENTIAL)

> **DO NOT publish this file.** It contains true_spec values, seeds, tier interpolation
> fractions, GCS paths, and other server-side details that would compromise the
> blind evaluation integrity.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Tier Generation Logic](#2-tier-generation-logic)
3. [Hand-Crafted Challenge Configs (4 variants)](#3-hand-crafted-challenge-configs)
4. [Auto-Generated Challenge Configs (61 variants)](#4-auto-generated-challenge-configs)
5. [Complete Mismatch Parameter Reference](#5-complete-mismatch-parameter-reference)
6. [GCS Storage & Access](#6-gcs-storage--access)
7. [Key Implementation Files](#7-key-implementation-files)

---

## 1. Architecture Overview

```
platform/
├── scripts/
│   ├── generate_challenge_datasets.py    # HDF5 generation pipeline
│   └── generate_example_data.py          # Example data for docs
├── pwm_platform/
│   ├── services/benchmark_database/
│   │   ├── _variant_registry.py          # 65 variant metadata (compact)
│   │   ├── _challenge_data.py            # Challenge configs + auto-generation
│   │   ├── _factory.py                   # Expands registry → full VARIANT_DATABASE
│   │   └── _algorithm_catalog.py         # Algorithm overrides & leaderboard
│   ├── routers/gcs_proxy.py              # GCS → HTTP proxy for .h5 downloads
│   ├── templates/
│   │   ├── compete.html                  # Competition page
│   │   └── contribute.html               # Contribution page
│   └── static/
│       ├── examples/                     # Example algorithms + small HDF5 files
│       └── benchmark-data/challenge-data/v1.0/  # (empty — all in GCS)
```

**Total:** 65 variants x 3 tiers = 195 HDF5 files in GCS (~1.35 GiB)

---

## 2. Tier Generation Logic

### Mismatch Interpolation Fractions

Each tier's true_spec is derived by interpolating between `nominal` and `perturbed` values:

```python
true_value = nominal + fraction * (perturbed - nominal)
```

| Tier | Fraction | Difficulty | Seeds |
|------|----------|------------|-------|
| **Public** | 0.50 (50%) | Moderate | 1001 |
| **Dev** | 0.30 (30%) | Mild | 2001 |
| **Hidden** | 0.80 (80%) | Severe | 3001 |

### Spec Range Derivation

Contestant-visible spec_ranges are `±1.5 * delta` around the nominal value, where `delta = |perturbed - nominal|`. This gives contestants a range wider than the actual perturbation, adding uncertainty.

Per-tier spec_ranges are re-centered on that tier's true_spec value (same width, different center).

### Visible Data per Tier

| Tier | y | H_ideal | spec_ranges | x_true | true_spec |
|------|---|---------|-------------|--------|-----------|
| Public | Yes | Yes | Yes | **Yes** | **Yes** |
| Dev | Yes | Yes | Yes | No | No |
| Hidden | — (server-side) | — | — | — (has all, used for eval) | — |

---

## 3. Hand-Crafted Challenge Configs

These 4 variants have manually specified true_spec values and baselines from InverseNet validation:

### 3.1 SD-CASSI

- **Data source:** `datasets/TSA_simu_data/Truth/` (10 KAIST hyperspectral scenes)
- **Signal shape:** 256 x 256 x 28
- **Noise:** poisson_gaussian (alpha=1.0, sigma=0.01)

**Spec ranges (contestant-visible):**

| Parameter | Min | Max | Unit |
|-----------|-----|-----|------|
| mask_dx | 0.3 | 0.7 | px |
| mask_dy | 0.1 | 0.5 | px |
| mask_rotation | 0.0 | 0.2 | deg |
| dispersion_slope | 1.90 | 2.15 | px/band |
| dispersion_axis | 0.0 | 0.3 | deg |

**True spec per tier:**

| Parameter | Public (50%) | Dev (30%) | Hidden (80%) |
|-----------|-------------|-----------|--------------|
| mask_dx | 0.50 | 0.40 | 0.60 |
| mask_dy | 0.30 | 0.20 | 0.40 |
| mask_rotation | 0.10 | 0.05 | 0.15 |
| dispersion_slope | 2.02 | 2.08 | 1.95 |
| dispersion_axis | 0.15 | 0.10 | 0.22 |

**Baselines (Scenario II — with mismatch):**

| Method | PSNR | SSIM |
|--------|------|------|
| MST-L | 20.83 | 0.744 |
| HDNet | 21.88 | 0.756 |
| PnP-HSICNN | 20.40 | 0.574 |
| GAP-TV | 20.96 | 0.612 |

**Baselines (Scenario III — oracle spec):**

| Method | PSNR | SSIM |
|--------|------|------|
| MST-L | 27.33 | 0.881 |
| HDNet | 21.88 | 0.756 |
| PnP-HSICNN | 23.08 | 0.702 |
| GAP-TV | 21.72 | 0.688 |

### 3.2 CACTI

- **Data source:** `datasets/CACTI/simulation/` (6 video scenes: kobe, traffic, runner, drop, crash, aerial)
- **Signal shape:** 256 x 256 x 8
- **Noise:** poisson_gaussian (alpha=1.0, sigma=0.01)

**Spec ranges (contestant-visible):**

| Parameter | Min | Max | Unit |
|-----------|-----|-----|------|
| mask_dx | 0.2 | 0.8 | px |
| mask_dy | 0.1 | 0.5 | px |
| mask_rotation | 0.0 | 0.3 | deg |
| mask_blur | 0.0 | 0.5 | px |
| clock_offset | -0.1 | 0.1 | frames |
| gain_drift | 0.95 | 1.05 | — |
| offset_drift | -0.02 | 0.02 | — |

**True spec per tier:**

| Parameter | Public | Dev | Hidden |
|-----------|--------|-----|--------|
| mask_dx | 0.50 | 0.35 | 0.65 |
| mask_dy | 0.30 | 0.20 | 0.40 |
| mask_rotation | 0.15 | 0.08 | 0.22 |
| mask_blur | 0.20 | 0.10 | 0.35 |
| clock_offset | 0.05 | -0.03 | 0.08 |
| gain_drift | 1.02 | 0.98 | 1.04 |
| offset_drift | 0.01 | -0.01 | 0.015 |

**Baselines (Scenario II):**

| Method | PSNR | SSIM |
|--------|------|------|
| EfficientSCI | 27.38 | 0.927 |
| ELP-Unfolding | 26.50 | 0.910 |
| PnP-FFDNet | 20.15 | 0.650 |
| GAP-TV | 14.81 | 0.303 |

**Baselines (Scenario III):**

| Method | PSNR | SSIM |
|--------|------|------|
| EfficientSCI | 35.39 | 0.973 |
| ELP-Unfolding | 34.09 | 0.965 |
| PnP-FFDNet | 29.28 | 0.910 |
| GAP-TV | 26.75 | 0.870 |

### 3.3 SPC-Block

- **Data source:** `datasets/SPC/Set11/` (11 grayscale images)
- **Signal shape:** 256 x 256
- **Noise:** gaussian (sigma=0.03)

**Spec ranges:**

| Parameter | Min | Max | Unit |
|-----------|-----|-----|------|
| gain_decay_alpha | 0.001 | 0.01 | 1/measurement |
| noise_sigma | 0.01 | 0.05 | — |

**True spec per tier:**

| Parameter | Public | Dev | Hidden |
|-----------|--------|-----|--------|
| gain_decay_alpha | 0.005 | 0.003 | 0.008 |
| noise_sigma | 0.03 | 0.02 | 0.04 |

**Baselines (Scenario II / III):**

| Method | Sc.II PSNR | Sc.II SSIM | Sc.III PSNR | Sc.III SSIM |
|--------|-----------|-----------|------------|------------|
| ISTA-Net | 27.45 | 0.760 | 31.85 | 0.916 |
| HATNet | 26.80 | 0.745 | 30.98 | 0.905 |
| PnP-DRUNet | 24.10 | 0.690 | 30.53 | 0.895 |
| FISTA-TV | 19.02 | 0.584 | 28.06 | 0.850 |

### 3.4 SPC-Kronecker

Same as SPC-Block (shared data source, same spec ranges and baselines, different sensing matrix structure).

---

## 4. Auto-Generated Challenge Configs

The remaining 61 variants have their challenge configs auto-generated from the variant registry's `mismatch_params` via `generate_challenge_config()` in `_challenge_data.py`.

### Auto-Generation Formula

For each mismatch parameter with `(nominal, perturbed)`:

1. **spec_range** = `[nominal - 1.5*delta, nominal + 1.5*delta]` where `delta = |perturbed - nominal|`
2. **true_spec[public]** = `nominal + 0.50 * (perturbed - nominal)`
3. **true_spec[dev]** = `nominal + 0.30 * (perturbed - nominal)`
4. **true_spec[hidden]** = `nominal + 0.80 * (perturbed - nominal)`

### Category Defaults

| Category | Noise Model | Signal Shape | Scenes | Runner |
|----------|-------------|-------------|--------|--------|
| compressive | poisson_gaussian | [256, 256] | 5 | mask |
| medical | poisson | [128, 128, 64] | 3 | radon |
| medical_ultrasound | speckle | [256, 256] | 3 | psf |
| coherent | gaussian | [256, 256] | 5 | psf |
| microscopy | poisson_gaussian | [256, 256] | 5 | psf |
| electron_microscopy | poisson | [512, 512] | 3 | ctf |
| clinical_optics | gaussian | [256, 256] | 3 | psf |
| computational | gaussian | [128, 128] | 5 | psf |
| computational_photography | poisson_gaussian | [256, 256] | 5 | psf |
| neural_rendering | gaussian | [800, 800] | 5 | psf |
| depth_imaging | gaussian | [256, 256] | 5 | psf |
| remote_sensing | speckle | [512, 512] | 3 | kspace |
| particle_imaging | poisson | [128, 128, 64] | 3 | radon |
| scanning_probe | gaussian | [256, 256] | 3 | tip |
| industrial_inspection | gaussian | [256, 256] | 3 | psf |
| spectroscopy | gaussian | [256, 256] | 5 | psf |
| astronomy | poisson | [512, 512] | 3 | psf |
| ultrafast | poisson_gaussian | [256, 256] | 5 | mask |
| quantum | poisson | [128, 128] | 3 | mask |
| experimental_science | gaussian | [256, 256] | 5 | psf |
| scientific_instrumentation | gaussian | [256, 256] | 5 | psf |
| multi_modal_fusion | gaussian | [256, 256] | 3 | psf |

---

## 5. Complete Mismatch Parameter Reference

### All 65 Variants — Nominal & Perturbed Values

Below is the complete `(nominal, perturbed)` pair for every mismatch parameter. The auto-generation uses these to derive true_spec per tier.

#### Compressive

**sd_cassi:** mask_dx (0, 0.5), mask_dy (0, 0.3), mask_theta (0, 0.1), disp_a1 (2.0, 2.02), disp_alpha (0, 0.15), sigma_read (5.0, 8.0), dark_current (0.1, 0.5), gain (1.0, 1.03)

**cacti:** mask_dx (0, 0.5), mask_dy (0, 0.3), mask_theta (0, 0.1), clock_offset (0, 0.05), duty_cycle (1.0, 0.95), gain (1.0, 1.02)

**spc_block / spc_kronecker:** gain_alpha (0, 0.0015), sigma_y (0, 0.03)

**matrix:** matrix_perturb (0, 0.01), gain (1.0, 1.03), sigma_y (0, 0.02)

#### Medical — Projection-Based

**ct:** center_offset (0, 1.5), angle_error (0, 0.5), beam_hardening (0, 0.03), detector_tilt (0, 0.2)

**cbct:** center_offset (0, 2.0), source_dist (0, 1.0), cone_angle (0, 0.3), detector_tilt (0, 0.5)

**pet:** attenuation (0, 5.0), scatter_frac (0.3, 0.35), timing_res (200, 250), normalization (0, 2.0)

**spect:** center_offset (0, 1.5), collimator_septal (0, 0.02), attenuation (0, 5.0), scatter (0.2, 0.25)

**xray_radiography:** source_dist (0, 5.0), beam_hardening (0, 0.02), scatter (0, 0.05)

**mammography:** compression (0, 2.0), anode_angle (0, 0.5), scatter (0.3, 0.35)

**fluoroscopy:** motion_blur (0, 5.0), lag (0, 3.0), gain_drift (0, 0.5)

**angiography:** contrast_timing (0, 0.5), motion (0, 2.0), scatter (0, 0.05)

**dexa:** energy_offset (0, 1.0), soft_tissue (0, 3.0), beam_overlap (0, 0.02)

**dot:** mu_a (0, 10.0), mu_s (0, 8.0), source_pos (0, 1.0)

**photoacoustic:** sos (1540, 1560), fluence (0, 10.0), sensor_response (0, 5.0)

#### Medical — Fourier-Based

**mri:** B0_inhomog (0, 1.5), gradient_nonlin (0, 2.0), coil_sensitivity (0, 5.0), k_trajectory (0, 1.0)

**fmri:** B0_inhomog (0, 2.0), head_motion (0, 1.0), hemodynamic_delay (6.0, 7.0), physiological_noise (0, 0.02)

**diffusion_mri:** b_value_error (0, 3.0), eddy_current (0, 0.5), gradient_direction (0, 1.0), susceptibility (0, 1.0)

**mrs:** linewidth (0, 2.0), freq_drift (0, 1.5), phase_error (0, 5.0), baseline (0, 0.05)

#### Medical Ultrasound

**ultrasound:** sos (1540, 1560), attenuation (0.5, 0.6), element_sensitivity (0, 5.0), phase_aberration (0, 0.3)

**doppler_ultrasound:** sos (1540, 1555), doppler_angle (0, 5.0), wall_filter (50, 80), prf (0, 1.0)

**elastography:** shear_speed (0, 0.3), push_duration (0, 10), tissue_viscosity (0, 15.0)

#### Coherent

**holography:** wavelength (0, 0.5), prop_distance (0, 5.0), tilt (0, 0.5)

**ptychography:** probe_error (0, 5.0), position_error (0, 10.0), partial_coherence (0, 5.0)

**phase_retrieval:** support (0, 3.0), saturation (0, 5.0), missing_center (0, 3)

#### Microscopy

**widefield:** psf_sigma (0, 10.0), defocus (0, 0.5), background (0, 50)

**widefield_lowdose:** psf_sigma (0, 10.0), photon_budget (0, 20.0), read_noise (1.5, 2.5)

**confocal_livecell:** pinhole (0, 5.0), refractive_index (1.515, 1.52), photobleaching (0, 5.0)

**confocal_3d:** z_step (0, 50), spherical_aberr (0, 0.1), refractive_index (1.515, 1.525)

**lightsheet:** sheet_thickness (0, 1.0), sheet_tilt (0, 0.5), stripe_artifact (0, 0.1)

**two_photon:** pulse_width (100, 140), gdd (0, 500), scattering (0, 10.0)

**sted:** depletion_power (0, 10.0), donut_alignment (0, 10), saturation_intensity (0, 8.0)

**tirf:** incidence_angle (0, 0.3), penetration_depth (0, 20), refractive_index (1.515, 1.52)

**sim:** pattern_phase (0, 0.05), pattern_freq (0, 1.0), modulation_depth (0, 5.0)

**fpm:** led_position (0, 0.1), na_error (0.1, 0.105), defocus (0, 2.0)

**flim:** irf_width (0, 20), time_bin (0, 5), afterpulsing (0, 0.005)

**palm_storm:** psf_model (0, 5.0), emitter_density (0, 20.0), drift (0, 0.5)

**polarization:** extinction_ratio (0, 0.5), retardance (0, 2.0), alignment (0, 0.5)

#### Electron Microscopy

**sem:** beam_energy (0, 0.1), stigmatism (0, 5.0), working_distance (0, 0.1)

**tem:** defocus (0, 50), Cs (0, 0.01), beam_tilt (0, 0.5)

**stem:** probe_size (0, 0.1), convergence_angle (0, 0.5), scan_distortion (0, 0.5)

**electron_tomography:** tilt_angle (0, 0.5), tilt_axis (0, 0.3), defocus_gradient (0, 10)

**electron_diffraction:** camera_length (0, 2.0), center_offset (0, 1.0), elliptical_distortion (0, 0.005)

**electron_holography:** biprism_voltage (0, 2.0), fringe_spacing (0, 0.1), partial_coherence (0, 5.0)

**ebsd:** pattern_center (0, 2.0), sample_tilt (70, 70.5), detector_distance (0, 0.5)

**eels:** energy_dispersion (0, 0.002), zero_loss_shift (0, 0.3), aberration (0, 2.0)

#### Clinical Optics

**oct:** dispersion (0, 200), reference_delay (0, 5.0), spectral_roll_off (0, 1.0)

**octa:** inter_bscan_time (0, 0.5), bulk_motion (0, 0.2), decorrelation_threshold (0.5, 0.55)

**fundus:** pupil_dilation (0, 0.5), focus (0, 0.25), vignetting (0, 5.0)

**endoscopy:** fiber_coupling (0, 5.0), core_spacing (0, 0.5), bending_loss (0, 0.3)

#### Computational

**light_field:** microlens_pitch (0, 0.5), main_lens_f (0, 0.1), vignetting (0, 3.0)

**integral:** lens_pitch (0, 1.0), gap_distance (0, 5.0), aberration (0, 0.1)

#### Computational Photography

**lensless:** diffuser_psf (0, 5.0), sensor_distance (0, 0.2), wavelength (0, 5.0)

**panorama:** focus_step (0, 2.0), registration (0, 0.5), exposure_variation (0, 3.0)

#### Neural Rendering

**nerf:** camera_pose (0, 1.0), focal_length (0, 5.0), distortion (0, 0.01), exposure (0, 10.0)

**gaussian_splatting:** camera_pose (0, 1.0), focal_length (0, 5.0), point_cloud_init (0, 2.0)

#### Depth Imaging

**tof_camera:** modulation_freq (20, 20.1), multipath (0, 5.0), phase_nonlinearity (0, 2.0)

**structured_light:** baseline (0, 0.5), pattern_distortion (0, 1.0), ambient_ir (0, 3.0)

**lidar:** timing_jitter (0, 50), beam_divergence (0, 0.1), range_walk (0, 1.0)

#### Remote Sensing

**sar:** motion_error (0, 2.0), phase_error (0, 0.3), range_cell_migration (0, 0.5)

**sonar:** sound_speed_profile (0, 5.0), multipath (0, 2), array_calibration (0, 3.0)

#### Particle Imaging

**neutron_tomo:** beam_spectrum (0, 3.0), scatter_correction (0, 5.0), rotation_offset (0, 1.0)

**proton_radiography:** energy_loss (0, 2.0), scattering (0, 5.0), range_straggling (0, 3.0)

**muon_tomo:** angular_resolution (0, 2.0), momentum_estimate (0, 10.0), detector_efficiency (0, 3.0)

---

## 6. GCS Storage & Access

### Bucket

```
gs://pwm-benchmark-datasets/
```

### File Path Pattern

```
challenge-data/v1.0/{variant_key}_challenge_{tier}.h5
```

Examples:
```
challenge-data/v1.0/ct_challenge_public.h5
challenge-data/v1.0/ct_challenge_dev.h5
challenge-data/v1.0/ct_challenge_hidden.h5
challenge-data/v1.0/sd_cassi_challenge_public.h5
...
```

### Total: 195 files (65 variants x 3 tiers)

### Access Chain

1. **nginx** tries local file at `/var/www/pwm-downloads/{path}`
2. Falls back to `@gcs_app` (FastAPI GCS proxy at `routers/gcs_proxy.py`)
3. Proxy fetches from `gs://pwm-benchmark-datasets/{path}`
4. Streams to client with appropriate content-type headers

### No Local Copies

All challenge `.h5` files are served from GCS only. The local directory `static/benchmark-data/challenge-data/v1.0/` is empty.

---

## 7. Key Implementation Files

| File | Purpose |
|------|---------|
| `_variant_registry.py` | 65 variant metadata (display names, mismatch_params with nominal/perturbed) |
| `_challenge_data.py` | 4 hand-crafted configs + auto-generation from mismatch_params |
| `_factory.py` | Expands compact registry entries into full VARIANT_DATABASE |
| `_algorithm_catalog.py` | Algorithm overrides, leaderboard builders |
| `generate_challenge_datasets.py` | HDF5 generation with 7 runner-specific forward model pipelines |
| `generate_example_data.py` | Small 32x32 example HDF5 files for CT/MRI |
| `gcs_proxy.py` | GCS streaming proxy for .h5 downloads |
| `gcs_store.py` | `GCSDatasetStore` class for upload/download to GCS bucket |
| `generate_challenge_datasets.py` | Forward models: `_forward_radon_fast`, `_forward_kspace`, `_forward_psf`, `_forward_ctf`, `_forward_mask`, `_forward_tip` |

### Generating New Challenge Data

```bash
cd platform

# Generate all variants
python scripts/generate_challenge_datasets.py --variant all --upload-gcs

# Generate one variant
python scripts/generate_challenge_datasets.py --variant ct --upload-gcs

# Generate by category
python scripts/generate_challenge_datasets.py --category medical --upload-gcs

# GCS-only (don't save locally)
python scripts/generate_challenge_datasets.py --variant all --gcs-only --upload-gcs
```

### Regenerating Example Data

```bash
cd platform
python scripts/generate_example_data.py
# Output: pwm_platform/static/examples/*.h5
```
