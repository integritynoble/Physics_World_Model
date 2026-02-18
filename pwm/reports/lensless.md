# Lensless (Diffuser Camera) Imaging — PWM Report

| Field | Value |
|-------|-------|
| Modality ID | `lensless` |
| Category | computational_photography |
| Dataset | DiffuserCam (synthetic proxy: fluorescence phantom with large PSF) |
| Date | 2026-02-12 |
| PWM version | `94207db` |
| Author | integritynoble |

## Modality overview

- Modality key: `lensless`
- Category: computational_photography
- Forward model: y = Poisson(peak * QE * PSF ** x) + read_noise, where PSF is a large Gaussian (sigma=5.0 px) simulating the diffuser spread function, QE=0.9
- Default solver: richardson_lucy_2d (100 iterations)
- Pipeline linearity: linear

Lensless imaging replaces the lens with a thin diffuser (e.g., random phase mask) placed directly on the sensor. The measurement is a heavily blurred (large PSF) version of the scene. Computational reconstruction (ADMM-TV, Tikhonov, or RL deconvolution) recovers the scene from the diffuser PSF calibration.

| Parameter | Value |
|-----------|-------|
| Image size (H x W) | 64 x 64 |
| PSF sigma | 5.0 pixels |
| PSF mode | constant (zero padding) |
| Peak photons | 20000 |
| Read noise sigma | 0.01 |

## Standard dataset

- Name: DiffuserCam Lensless (synthetic proxy: Gaussian blob phantom)
- Source: https://waller-lab.github.io/LenslessLearning/
- Size: 25000 images, 270x480; experiment uses 64x64 synthetic phantom
- Registered in `dataset_registry.yaml` as `diffusercam`

## PWM pipeline flowchart (mandatory)

```
x (world)
  ↓
SourceNode: photon_source — scene illumination scaling (1.0)
  ↓
Element 1 (subrole=transport): conv2d — diffuser PSF convolution (sigma=5.0, mode=constant)
  ↓
SensorNode: photon_sensor — QE=0.9, gain=1.0
  ↓
NoiseNode: poisson_gaussian_sensor — Poisson (peak=20000) + Gaussian (sigma=0.01)
  ↓
y
```

## Element inventory & mismatch parameters

| # | node_id | primitive_id | subrole | learnable params | mismatch knob | bounds | prior |
|---|---------|-------------|---------|-----------------|---------------|--------|-------|
| 1 | source | photon_source | source | strength=1.0 | — | — | — |
| 2 | psf_conv | conv2d | transport | sigma=5.0, mode=constant | psf_sigma | [2.0, 10.0] | normal(5.0, 1.0) |
| 3 | sensor | photon_sensor | sensor | QE=0.9, gain=1.0 | gain | [0.5, 1.5] | normal |
| 4 | noise | poisson_gaussian_sensor | noise | peak=20000, read_sigma=0.01 | — | — | — |

## Node-by-node trace (one sample)

| stage | node_id | output shape | dtype | range [min, max] | artifact_path |
|-------|---------|-------------|-------|-------------------|---------------|
| 0 | input x | (64, 64) | float64 | [0.0500, 0.9400] | `artifacts/trace/00_input_x.npy` |
| 1 | source | (64, 64) | float64 | [0.0500, 0.9400] | `artifacts/trace/01_source.npy` |
| 2 | psf_conv | (64, 64) | float64 | [0.0100, 0.4200] | `artifacts/trace/02_psf_conv.npy` |
| 3 | sensor | (64, 64) | float64 | [0.0090, 0.3780] | `artifacts/trace/03_sensor.npy` |
| 4 | noise (y) | (64, 64) | float64 | [0.0010, 0.3900] | `artifacts/trace/04_noise_y.npy` |

PNG visualizations saved at: `artifacts/trace/png/`

## Workflow W1: Prompt-driven simulation + reconstruction

- **Prompt used:** `"Simulate lensless imaging and reconstruct using Richardson-Lucy deconvolution"`
- **ExperimentSpec summary:**
  - modality: lensless
  - mode: simulate -> invert
  - solver: richardson_lucy_2d
  - photon_budget: 20000 peak photons
- **Mode S results (simulate y):**
  - y shape: (64, 64)
  - SNR: 25.5 dB
- **Mode I results (reconstruct x_hat):**
  - Solver: richardson_lucy_2d, iterations: 100
- **Dataset metrics:**

  | Metric | Value |
  |--------|-------|
  | PSNR   | 23.34 dB |
  | SSIM   | 0.9335 |
  | NRMSE  | 0.0719 |

## Workflow W2: Operator correction mode (measured y + operator A)

- **Operator definition:**
  - A_definition: callable
  - A_extraction_method: graph_stripped
  - Operator chain: PhotonSource → Conv2d → PhotonSensor (noise stripped)
  - A_sha256: see RunBundle
  - Linearity: linear
  - Notes (if linearized): N/A (linear Gaussian convolution)
- **Mismatch specification:**
  - Mismatch type: synthetic_injected
  - Parameters perturbed: sensor gain (1.0 → 1.3, +30%)
  - Description: Sensor gain drift from temperature or aging effects
- **Mode C fit results:**
  - Correction family: Pre
  - Parameters fitted: 1 (gain) via grid search over [0.5, 2.0]
  - NLL before correction: 63265.6
  - NLL after correction: 2048.2
  - NLL decrease: 61217.4 (96.8%)
- **Mode I recon using corrected operator A':**

  | Metric | A₀ (uncorrected) | A' (corrected) | Delta |
  |--------|-------------------|----------------|-------|
  | PSNR   | 21.04 dB | 23.34 dB | +2.30 dB |
  | SSIM   | 0.8900 | 0.9335 | +0.0435 |

## Test results summary

### Quick gate (pass/fail)

- [x] W1 simulate: PASS
- [x] W1 reconstruct (PSNR >= 15 dB): PASS (23.34 dB)
- [x] W2 operator correction (NLL decreases): PASS (96.8% decrease)
- [x] W2 corrected recon (beats uncorrected): PASS (+2.30 dB)
- [x] Report contract (flowchart + all headings): PASS
- [x] Node-by-node trace saved (>= 3 stages): PASS (5 stages)
- [x] RunBundle saved (with trace PNGs): PASS
- [x] Scoreboard updated: PASS

### Full metrics

| Check | Metric | Value | Threshold | Status |
|-------|--------|-------|-----------|--------|
| W1 PSNR | psnr | 23.34 dB | >= 15 dB | PASS |
| W1 SSIM | ssim | 0.9335 | >= 0.70 | PASS |
| W1 NRMSE | nrmse | 0.0719 | — | info |
| W2 NLL decrease | nll_decrease_pct | 96.8% | >= 5% | PASS |
| W2 PSNR gain | psnr_delta | +2.30 dB | > 0 | PASS |
| W2 SSIM gain | ssim_delta | +0.0435 | > 0 | PASS |
| Trace stages | n_stages | 5 | >= 3 | PASS |
| Trace PNGs | n_pngs | 5 | >= 3 | PASS |

## Reproducibility

- Seed: 42
- NumPy RNG state hash: 5d6a058f2c5c262e
- PWM version: 94207db
- Python version: 3.13.9
- Key package versions: numpy=2.3.5, scipy=1.16.3
- Platform: Linux x86_64
- Deterministic reproduction command:
  ```bash
  pwm_cli run --modality lensless --seed 42 --mode simulate,invert,calibrate
  ```
- Output hash (y): see RunBundle
- Output hash (x_hat): see RunBundle

## Saved artifacts

- RunBundle: `runs/run_lensless_exp_7eec75c8/`
- Report: `pwm/reports/lensless.md`
- Scoreboard: `pwm/reports/scoreboard.yaml`

## Next actions

- Test on DiffuserCam benchmark with measured PSFs
- Implement ADMM-TV solver for improved reconstruction
- Add multi-channel (RGB) lensless reconstruction
