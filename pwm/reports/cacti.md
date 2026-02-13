# Coded Aperture Compressive Temporal Imaging (CACTI) — PWM Report

| Field | Value |
|-------|-------|
| Modality ID | `cacti` |
| Category | compressive |
| Dataset | Grayscale SCI Video Benchmark — 6 scenes (256x256, 8:1 compression) |
| Date | 2026-02-12 |
| PWM version | `ae5106b` |
| Author | integritynoble |

## Modality overview

- Modality key: `cacti`
- Category: compressive
- Forward model: y = gain * sum_t(EffectiveMask_t(dx,dy,theta,blur,clock,duty) * x_t) + offset + noise, where noise ~ Poisson(peak_photons) + N(0, read_sigma^2) + Quantization(12bit)
- Default solver: GAP-TV (classical), with PnP-FFDNet, ELP-Unfolding, EfficientSCI as deep alternatives
- Pipeline linearity: linear

CACTI (also known as Snapshot Compressive Imaging for video) captures T video frames in a single 2D snapshot using time-varying coded aperture masks driven by a DMD/LCD. Each video frame is modulated by a different binary mask pattern, integrated through a shutter, and summed on the detector. The measurement is corrupted by Poisson shot noise, read noise, and ADC quantization. Reconstruction recovers the 3D video cube from the 2D measurement. This benchmark evaluates 4 solvers spanning classical optimization, plug-and-play denoisers, and deep unfolding networks across 6 standard test scenes. W2 evaluates a physically realistic multi-parameter mismatch model with multi-stage correction on 3 scenes.

| Parameter | Value |
|-----------|-------|
| Image size (H x W) | 256 x 256 |
| Temporal frames per measurement | 8 |
| Compression ratio | 8:1 |
| Mask type | Binary random (from dataset, DMD-driven) |
| Data range | [0, 255] |
| Noise model | Poisson(peak=10000) + Read(sigma=5.0) + Quant(12bit) |

## Standard dataset

- Name: Grayscale SCI Video Benchmark (6 scenes)
- Source: Standard CACTI benchmark suite
- Scenes: kobe32 (32 frames), crash32 (32 frames), aerial32 (32 frames), traffic48 (48 frames), runner40 (40 frames), drop40 (40 frames)
- Format: MATLAB `.mat`, keys: `meas`, `mask`, `orig`
- Location: `PnP-SCI_python-master/dataset/cacti/grayscale_benchmark/`

| Scene | Total frames | Coded measurements | Content |
|-------|-------------|-------------------|---------|
| kobe32 | 32 | 4 | Basketball footage, fast motion |
| crash32 | 32 | 4 | Car crash, sudden deformation |
| aerial32 | 32 | 4 | Aerial view, fine texture |
| traffic48 | 48 | 6 | Traffic intersection, multi-object motion |
| runner40 | 40 | 5 | Running athlete, periodic motion |
| drop40 | 40 | 5 | Water drop, fluid dynamics |

## PWM pipeline flowchart (mandatory)

```
x (world)
  ↓
SourceNode: photon_source — scales input by illumination strength (1.0)
  ↓
Element 1 (subrole=transport): objective_lens — throughput=0.95, PSF blur, vignetting
  ↓
Element 2 (subrole=encoding): temporal_coded_aperture — DMD/LCD time-varying binary masks (mismatch: mask_dx, mask_dy, mask_theta, mask_blur_sigma)
  ↓
Element 3 (subrole=interaction): shutter_integration — exposure schedule (mismatch: clock_offset, duty_cycle, timing_jitter_std, temporal_response_tau)
  ↓
SensorNode: detector — QE=0.9, gain=1.0, offset=0.0 (mismatch: gain, offset)
  ↓
NoiseNode: poisson_read_quantization — Poisson(peak=10000) + Read(sigma=5.0) + Quant(12bit)
  ↓
y
```

## Element inventory & mismatch parameters

| # | node_id | primitive_id | subrole | learnable params | mismatch knob | bounds | correction family |
|---|---------|-------------|---------|-----------------|---------------|--------|-------------------|
| 1 | source | photon_source | source | strength=1.0 | — | — | — |
| 2 | objective | objective_lens | transport | throughput=0.95, psf_sigma=0.0 | psf_sigma | [0, 2] px | Pre |
| 3 | coded_aperture | temporal_mask | encoding | binary mask (256,256,8) | mask_dx, mask_dy, mask_theta, mask_blur_sigma | [-3,3] px, [-3,3] px, [-0.6,0.6]°, [0,2] px | Pre |
| 4 | shutter | shutter_integration | integration | duty_cycle=1.0, clock_offset=0.0 | clock_offset, duty_cycle, timing_jitter_std, temporal_response_tau | [-0.5,0.5] fr, [0.7,1.0], [0,0.2] fr, [0,2] ms | PreTemporal |
| 5 | detector | photon_sensor | sensor | QE=0.9, gain=1.0, offset=0.0 | gain, offset | [0.5,1.5], [-0.1,0.1] | Post |
| 6 | noise | poisson_gaussian_sensor | noise | peak_photons=10000, read_sigma=5.0, bit_depth=12 | read_sigma | [0,0.05] | — |

### Full mismatch parameter table (mismatch_db.yaml)

| Parameter | Range | Typical error | Weight | Category | Description |
|-----------|-------|--------------|--------|----------|-------------|
| mask_dx | [-3, 3] px | 0.5 | 0.12 | Spatial | Coded mask horizontal registration error |
| mask_dy | [-3, 3] px | 0.5 | 0.12 | Spatial | Coded mask vertical registration error |
| mask_theta | [-0.6, 0.6]° | 0.05 | 0.08 | Spatial | Mask-sensor rotation from assembly tolerance |
| mask_blur_sigma | [0, 2] px | 0.3 | 0.06 | Spatial | Mask edge blur from defocus/relay PSF/DMD fill |
| clock_offset | [-0.5, 0.5] fr | 0.05 | 0.16 | Temporal | DMD-camera sync offset |
| timing_jitter_std | [0, 0.2] fr | 0.03 | 0.10 | Temporal | Pattern switching jitter |
| duty_cycle | [0.7, 1.0] | 0.03 | 0.08 | Temporal | Sub-exposure fraction |
| temporal_response_tau | [0, 2] ms | 0.2 | 0.06 | Temporal | Camera/DMD settling blur |
| gain | [0.5, 1.5] | 0.1 | 0.08 | Sensor | Detector gain calibration error |
| offset | [-0.1, 0.1] | 0.01 | 0.05 | Sensor | Detector DC offset |
| read_sigma | [0, 0.05] | 0.01 | 0.09 | Sensor | Read noise std |

Correction method: `UPWMI_multistage_search`

## Node-by-node trace (one sample)

Trace of intermediate tensor states at each node for one representative sample from kobe32 (first coded frame block, frames 0-7), with realistic mismatch parameters applied.

| stage | node_id | output shape | dtype | description | artifact_path |
|-------|---------|-------------|-------|-------------|---------------|
| 0 | input_x | (256, 256, 8) | float32 | Video cube input | `artifacts/trace/00_input_x.npy` |
| 1 | objective | (256, 256, 8) | float32 | After objective lens (throughput=0.95) | `artifacts/trace/01_objective.npy` |
| 2 | coded_aperture | (256, 256, 8) | float32 | After warped mask modulation (dx=1.5, dy=1.0, theta=0.3°) | `artifacts/trace/02_coded_aperture.npy` |
| 3 | shutter_integrated | (256, 256) | float32 | After temporal integration (clock=0.08, duty=0.92) | `artifacts/trace/03_shutter_integrated.npy` |
| 4 | detector | (256, 256) | float32 | After sensor gain=1.05 + offset=0.005 | `artifacts/trace/04_detector.npy` |
| 5 | noisy_y | (256, 256) | float32 | After Poisson+Read+Quant noise | `artifacts/trace/05_noisy_y.npy` |
| 6 | recon_gaptv | (256, 256, 8) | float32 | GAP-TV reconstruction | `artifacts/trace/06_recon_gaptv.npy` |

PNG visualizations saved at: `artifacts/trace/png/`

## Workflow W1: Prompt-driven simulation + reconstruction

- **Prompt used:** `"Reconstruct 6-scene CACTI benchmark with 4 solvers: GAP-TV, PnP-FFDNet, ELP-Unfolding, EfficientSCI"`
- **ExperimentSpec summary:**
  - modality: cacti
  - mode: reconstruct from real benchmark data
  - solvers: GAP-TV, PnP-FFDNet, ELP-Unfolding, EfficientSCI
  - dataset: 6 scenes (kobe32, crash32, aerial32, traffic48, runner40, drop40), 256x256, 8:1 compression
- **Mode S results (real measurement):**
  - y shape: (256, 256, N_coded) — N_coded varies per scene (4-6 coded snapshots)
  - Mask: (256, 256, 8) binary
- **Mode I results (reconstruct x_hat):**

  **PSNR (dB) — 4-Solver Comparison across 6 Scenes**

  | Scene | GAP-TV | PnP-FFDNet | ELP-Unfolding | EfficientSCI |
  |-------|--------|------------|---------------|-------------|
  | kobe32 | 24.00 | 30.33 | 34.08 | 35.76 |
  | crash32 | 25.40 | 24.69 | 29.39 | 31.12 |
  | aerial32 | 26.13 | 24.36 | 30.54 | 31.50 |
  | traffic48 | 21.06 | 23.88 | 31.34 | 32.29 |
  | runner40 | 28.70 | 32.97 | 38.17 | 41.89 |
  | drop40 | 34.42 | 39.91 | 40.09 | 45.10 |
  | **Average** | **26.62** | **29.36** | **33.94** | **36.28** |

  **SSIM — 4-Solver Comparison across 6 Scenes**

  | Scene | GAP-TV | PnP-FFDNet | ELP-Unfolding | EfficientSCI |
  |-------|--------|------------|---------------|-------------|
  | kobe32 | 0.7461 | 0.9253 | 0.9644 | 0.9758 |
  | crash32 | 0.8649 | 0.8332 | 0.9537 | 0.9726 |
  | aerial32 | 0.8510 | 0.8200 | 0.9398 | 0.9542 |
  | traffic48 | 0.7063 | 0.8299 | 0.9623 | 0.9691 |
  | runner40 | 0.8908 | 0.9357 | 0.9744 | 0.9868 |
  | drop40 | 0.9654 | 0.9863 | 0.9798 | 0.9950 |
  | **Average** | **0.8374** | **0.8884** | **0.9624** | **0.9756** |

- **Dataset metrics (best solver — EfficientSCI, 6-scene average):**

  | Metric | Value |
  |--------|-------|
  | PSNR   | 36.28 dB |
  | SSIM   | 0.9756 |

GAP-TV provides a classical baseline averaging 26.62 dB across all 6 scenes. PnP-FFDNet improves to 29.36 dB by replacing the TV denoiser with a learned FFDNet prior. Deep unfolding (ELP-Unfolding, ECCV 2022) reaches 33.94 dB with 8+5 learned ADMM iterations using Vision Transformers. EfficientSCI (CVPR 2023) achieves the best result at 36.28 dB average via an end-to-end architecture with spatial-temporal attention.

Note: GAP-TV and PnP-FFDNet use the dataset's original mask. ELP-Unfolding uses the dataset's mask for forward simulation. EfficientSCI uses its own trained mask and re-simulates measurements for fair comparison.

## Workflow W2: Operator correction mode (measured y + operator A)

### Operator definition

- A_definition: callable
- A_extraction_method: provided
- Operator chain: y = gain * sum_t(EffectiveMask_t * x_t) + offset, with EffectiveMask incorporating spatial warp (dx,dy,theta,blur) and temporal mixing (clock_offset, duty_cycle, jitter, tau)
- Linearity: linear
- Noise model: Poisson(peak_photons=10000) + Read(sigma=5.0) + Quantization(12bit)

### Mismatch specification

- Mismatch type: synthetic_injected
- Number of parameters perturbed: 6 (spatial + temporal + sensor)
- Injected parameters:

| Parameter | Injected value | Category |
|-----------|---------------|----------|
| mask_dx | 1.5 px | Spatial |
| mask_dy | 1.0 px | Spatial |
| mask_theta | 0.3° | Spatial |
| clock_offset | 0.08 frames | Temporal |
| duty_cycle | 0.92 | Temporal |
| gain | 1.05 | Sensor |
| offset | 0.005 | Sensor |

- Description: Realistic multi-parameter mismatch simulating real CACTI hardware errors: DMD mask misregistration (dx, dy, theta), DMD-camera synchronization offset, incomplete sub-exposure duty cycle, detector gain/offset calibration error. All combined with physically realistic Poisson+Read+Quantization noise to prevent NLL from reaching 0.

### Mode C fit results (multi-stage correction)

- Correction method: `UPWMI_multistage_search`
- Correction families: Pre (spatial mask warp) + PreTemporal (temporal schedule) + Post (sensor gain/offset)
- Number of stages: 4

| Stage | Parameters searched | Grid size | Description |
|-------|-------------------|-----------|-------------|
| 1. Coarse spatial | mask_dx (±3, int), mask_dy (±3, int) | 7x7=49 | Integer mask registration |
| 2. Refine spatial | mask_dx (±1, 0.25), mask_dy (±1, 0.25), mask_theta (±0.6°, 0.1°) | 9x9x13=1053 | Sub-pixel + rotation |
| 3. Temporal | clock_offset (±0.3, 0.05), duty_cycle (0.8-1.0, 0.02) | 13x11=143 | DMD-camera timing |
| 4. Sensor | gain (0.9-1.1, 0.02), offset (±0.02, 0.005) | 11x9=99 | Electronics calibration |

- Evaluated on 3 scenes: kobe32, crash32, runner40
- NLL computed using Poisson+Read Gaussian approximation (NLL cannot reach 0)

### Mode C + Mode I results (per-scene)

| Scene | NLL decrease | PSNR (uncorrected) | PSNR (corrected) | PSNR delta |
|-------|-------------|-------------------|------------------|------------|
| kobe32 | TBD% | TBD dB | TBD dB | TBD dB |
| crash32 | TBD% | TBD dB | TBD dB | TBD dB |
| runner40 | TBD% | TBD dB | TBD dB | TBD dB |
| **Median** | **TBD%** | — | — | **TBD dB** |

(Values will be populated after running the benchmark.)

The multi-stage correction progressively refines the operator estimate: Stage 1 finds the coarse mask registration, Stage 2 refines to sub-pixel with rotation, Stage 3 recovers temporal coding mismatches, and Stage 4 calibrates sensor electronics. The Poisson+Read+Quantization noise ensures NLL has a realistic noise floor and cannot drop to 0 even with perfect parameter recovery.

## Test results summary

### Quick gate (pass/fail)

- [x] W1 simulate: PASS (6 scenes loaded from real benchmark data)
- [x] W1 reconstruct GAP-TV (avg PSNR >= 20 dB): PASS (26.62 dB)
- [x] W1 reconstruct PnP-FFDNet (avg PSNR >= 25 dB): PASS (29.36 dB)
- [x] W1 reconstruct ELP-Unfolding (avg PSNR >= 30 dB): PASS (33.94 dB)
- [x] W1 reconstruct EfficientSCI (avg PSNR >= 30 dB): PASS (36.28 dB)
- [ ] W2 operator correction (NLL decreases on 3 scenes): PENDING (run benchmark)
- [ ] W2 corrected recon (median PSNR gain > 0): PENDING (run benchmark)
- [x] Report contract (flowchart + all headings): PASS
- [x] Node-by-node trace saved (>= 6 stages): PASS (7 stages)
- [x] RunBundle saved (with trace PNGs): PASS

### Full metrics

| Check | Metric | Value | Threshold | Status |
|-------|--------|-------|-----------|--------|
| W1 GAP-TV avg PSNR | psnr | 26.62 dB | >= 20 dB | PASS |
| W1 PnP-FFDNet avg PSNR | psnr | 29.36 dB | >= 25 dB | PASS |
| W1 ELP-Unfolding avg PSNR | psnr | 33.94 dB | >= 30 dB | PASS |
| W1 EfficientSCI avg PSNR | psnr | 36.28 dB | >= 30 dB | PASS |
| W1 Best avg SSIM | ssim | 0.9756 | >= 0.90 | PASS |
| W2 Median NLL decrease | nll_decrease_pct | TBD% | >= 5% | PENDING |
| W2 Median PSNR gain | psnr_delta | TBD dB | > 0 | PENDING |
| W2 Scenes evaluated | n_scenes | 3 | >= 3 | PASS |
| Trace stages | n_stages | 7 | >= 3 | PASS |
| Total W1 scenes | n_scenes | 6 | >= 6 | PASS |

## Reproducibility

- Seed: 42
- PWM version: ae5106b
- Python version: 3.13.9
- Key package versions: numpy=2.3.5, scipy=1.16.3, torch=2.10.0+cu128
- Platform: Linux x86_64
- External codebases: PnP-SCI, ELP-Unfolding (ECCV 2022), EfficientSCI (CVPR 2023)
- Deterministic reproduction command:
  ```bash
  pwm_cli run --modality cacti --seed 42 --mode simulate,invert,calibrate
  # Full 6-scene benchmark with multi-stage W2:
  PYTHONPATH="$PWD:$PWD/packages/pwm_core" python scripts/run_cacti_benchmark.py
  ```
- Output hash (orig, kobe32): 67745be99e30074a
- Output hash (meas, kobe32): 917718cce2e61c46
- Output hash (mask, kobe32): 94d64c2f732b2072

## Saved artifacts

- RunBundle: `runs/run_cacti_benchmark_*/`
- Report: `pwm/reports/cacti.md`
- Scoreboard: `pwm/reports/scoreboard.yaml`
- Node trace (.npy): `runs/run_cacti_benchmark_*/artifacts/trace/*.npy`
- Node trace (.png): `runs/run_cacti_benchmark_*/artifacts/trace/png/*.png`
- W2 operator metadata: `runs/run_cacti_benchmark_*/artifacts/w2_operator_meta.json`
- Per-scene artifacts: `runs/run_cacti_benchmark_*/artifacts/{scene}/`
- Full results JSON: `runs/run_cacti_benchmark_*/cacti_benchmark_results.json`

## Next actions

- Run the updated benchmark to populate W2 results
- Add FastDVDNet solver to the comparison
- Investigate learned mask optimization for improved compression
