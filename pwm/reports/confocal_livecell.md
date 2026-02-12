# Confocal Live-Cell Microscopy — PWM Report

| Field | Value |
|-------|-------|
| Modality ID | `confocal_livecell` |
| Category | microscopy |
| Dataset | DeepBacs fluorescence (synthetic proxy: live-cell phantom) |
| Date | 2026-02-12 |
| PWM version | `65e5739` |
| Author | integritynoble |

## Modality overview

- Modality key: `confocal_livecell`
- Category: microscopy
- Forward model: y = Poisson(peak * QE * PSF ** x) + read_noise, where PSF is a tight Gaussian (sigma=1.2 px) from confocal pinhole, peak=5000 photons
- Default solver: richardson_lucy_2d (50 iterations)
- Pipeline linearity: linear

Confocal microscopy uses a pinhole to reject out-of-focus light, producing a tighter PSF than widefield. Live-cell imaging requires moderate photon budgets to avoid phototoxicity.

| Parameter | Value |
|-----------|-------|
| Image size (H x W) | 64 x 64 |
| PSF sigma | 1.2 pixels |
| Peak photons | 5000 |
| Read noise sigma | 0.02 |
| Noise model | Poisson-Gaussian |

## Standard dataset

- Name: DeepBacs fluorescence (synthetic proxy: live-cell phantom)
- Source: https://zenodo.org/record/5764540
- Size: 900 images, 512x512 confocal; experiment uses 64x64 synthetic phantom
- Registered in `dataset_registry.yaml` as `deepbacs_fluor`

## PWM pipeline flowchart (mandatory)

```
x (world)
  ↓
SourceNode: photon_source — excitation scaling (1.0)
  ↓
Element 1 (subrole=transport): conv2d — confocal PSF convolution (sigma=1.2)
  ↓
SensorNode: photon_sensor — QE=0.9, gain=1.0
  ↓
NoiseNode: poisson_gaussian_sensor — Poisson (peak=5000) + Gaussian (sigma=0.02)
  ↓
y
```

## Element inventory & mismatch parameters

| # | node_id | primitive_id | subrole | learnable params | mismatch knob | bounds | prior |
|---|---------|-------------|---------|-----------------|---------------|--------|-------|
| 1 | source | photon_source | source | strength=1.0 | — | — | — |
| 2 | blur | conv2d | transport | sigma=1.2 | psf_sigma | [0.3, 2.5] | normal(1.2, 0.2) |
| 3 | sensor | photon_sensor | sensor | QE=0.9, gain=1.0 | gain | [0.6, 1.4] | normal |
| 4 | noise | poisson_gaussian_sensor | noise | peak=5000, read_sigma=0.02 | — | — | — |

## Node-by-node trace (one sample)

| stage | node_id | output shape | dtype | range [min, max] | artifact_path |
|-------|---------|-------------|-------|-------------------|---------------|
| 0 | input x | (64, 64) | float64 | [0.0500, 0.9497] | `artifacts/trace/00_input_x.npy` |
| 1 | source | (64, 64) | float64 | [0.0500, 0.9497] | `artifacts/trace/01_source.npy` |
| 2 | blur | (64, 64) | float64 | [0.0503, 0.8850] | `artifacts/trace/02_blur.npy` |
| 3 | sensor | (64, 64) | float64 | [0.0453, 0.7965] | `artifacts/trace/03_sensor.npy` |
| 4 | noise (y) | (64, 64) | float64 | [0.0020, 0.8120] | `artifacts/trace/04_noise_y.npy` |

PNG visualizations saved at: `artifacts/trace/png/`

## Workflow W1: Prompt-driven simulation + reconstruction

- **Prompt used:** `"Simulate confocal live-cell microscopy and reconstruct using Richardson-Lucy"`
- **ExperimentSpec summary:**
  - modality: confocal_livecell
  - mode: simulate -> invert
  - solver: richardson_lucy_2d
  - photon_budget: 5000
- **Mode S results (simulate y):**
  - y shape: (64, 64)
  - SNR: 18.2 dB
- **Mode I results (reconstruct x_hat):**
  - Solver: richardson_lucy_2d, iterations: 50
- **Dataset metrics:**

  | Metric | Value |
  |--------|-------|
  | PSNR   | 25.09 dB |
  | SSIM   | 0.9378 |
  | NRMSE  | 0.0591 |

## Workflow W2: Operator correction mode (measured y + operator A)

- **Operator definition:**
  - A_definition: callable
  - A_extraction_method: graph_stripped
  - Operator chain: PhotonSource → Conv2d → PhotonSensor (noise stripped)
  - A_sha256: c57775d7e456b32c
  - Linearity: linear
  - Notes (if linearized): N/A (linear Gaussian convolution)
- **Mismatch specification:**
  - Mismatch type: synthetic_injected
  - Parameters perturbed: PSF sigma (1.2 → 2.0, +67% wider)
  - Description: Pinhole-NA mismatch causing wider PSF than calibrated
- **Mode C fit results:**
  - Correction family: Pre
  - Parameters fitted: 1 (PSF sigma) via grid search [0.5, 3.0]
  - Best fitted sigma: 2.0
  - NLL before correction: 2320.6
  - NLL after correction: 2048.3
  - NLL decrease: 272.3 (11.7%)
- **Mode I recon using corrected operator A':**

  | Metric | A₀ (uncorrected) | A' (corrected) | Delta |
  |--------|-------------------|----------------|-------|
  | PSNR   | 23.73 dB | 25.33 dB | +1.60 dB |
  | SSIM   | 0.9174 | 0.9410 | +0.0236 |

## Test results summary

### Quick gate (pass/fail)

- [x] W1 simulate: PASS
- [x] W1 reconstruct (PSNR >= 18 dB): PASS (25.09 dB)
- [x] W2 operator correction (NLL decreases): PASS (11.7%)
- [x] W2 corrected recon (beats uncorrected): PASS (+1.60 dB)
- [x] Report contract (flowchart + all headings): PASS
- [x] Node-by-node trace saved (>= 3 stages): PASS (5 stages)
- [x] RunBundle saved (with trace PNGs): PASS
- [x] Scoreboard updated: PASS

### Full metrics

| Check | Metric | Value | Threshold | Status |
|-------|--------|-------|-----------|--------|
| W1 PSNR | psnr | 25.09 dB | >= 18 dB | PASS |
| W1 SSIM | ssim | 0.9378 | >= 0.80 | PASS |
| W1 NRMSE | nrmse | 0.0591 | — | info |
| W2 NLL decrease | nll_decrease_pct | 11.7% | >= 5% | PASS |
| W2 PSNR gain | psnr_delta | +1.60 dB | > 0 | PASS |
| W2 SSIM gain | ssim_delta | +0.0236 | > 0 | PASS |
| Trace stages | n_stages | 5 | >= 3 | PASS |
| Trace PNGs | n_pngs | 5 | >= 3 | PASS |

## Reproducibility

- Seed: 42
- NumPy RNG state hash: 5d6a058f2c5c262e
- PWM version: 65e5739
- Python version: 3.13.9
- Key package versions: numpy=2.3.5, scipy=1.16.3
- Platform: Linux x86_64
- Deterministic reproduction command:
  ```bash
  pwm_cli run --modality confocal_livecell --seed 42 --mode simulate,invert,calibrate
  ```
- Output hash (y): see RunBundle
- Output hash (x_hat): see RunBundle

## Saved artifacts

- RunBundle: `runs/run_confocal_livecell_exp_335a8198/`
- Report: `pwm/reports/confocal_livecell.md`
- Scoreboard: `pwm/reports/scoreboard.yaml`
- Node trace (.npy): `runs/run_confocal_livecell_exp_335a8198/artifacts/trace/*.npy`
- Node trace (.png): `runs/run_confocal_livecell_exp_335a8198/artifacts/trace/png/*.png`
- W2 operator metadata: `runs/run_confocal_livecell_exp_335a8198/artifacts/w2_operator_meta.json`

## Next actions

- Test on DeepBacs benchmark at 512x512 resolution
- Add CARE and Noise2Void deep learning denoisers
- Implement time-lapse confocal with temporal regularization
