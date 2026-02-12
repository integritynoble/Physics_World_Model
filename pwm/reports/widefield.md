# Widefield Fluorescence Microscopy — PWM Report

| Field | Value |
|-------|-------|
| Modality ID | `widefield` |
| Category | microscopy |
| Dataset | BioSR F-Actin (synthetic proxy: fluorescence phantom) |
| Date | 2026-02-12 |
| PWM version | `7394757` |
| Author | integritynoble |

## Modality overview

- Modality key: `widefield`
- Category: microscopy
- Forward model: y = Poisson(peak * QE * PSF ** x) + read_noise, where x is a 2D fluorescence distribution (H, W), PSF is a Gaussian point spread function (sigma=2.0 px), QE is quantum efficiency (0.9), and noise follows a Poisson-Gaussian model
- Default solver: richardson_lucy_2d (Richardson-Lucy deconvolution, 50 iterations)
- Pipeline linearity: linear

Widefield fluorescence microscopy illuminates the entire field of view simultaneously. The image is formed by convolution of the specimen fluorescence distribution with the system point spread function (PSF). Out-of-focus blur is the primary degradation. Deconvolution via Richardson-Lucy restores resolution by iteratively inverting the PSF convolution.

| Parameter | Value |
|-----------|-------|
| Image size (H x W) | 64 x 64 (N=4096) |
| PSF type | Gaussian, sigma=2.0 pixels |
| PSF FWHM | ~4.7 pixels |
| Photon budget (peak) | 10000 |
| Read noise sigma | 0.01 |
| Noise model | Poisson-Gaussian |

## Standard dataset

- Name: BioSR F-Actin (synthetic proxy used here: fluorescence phantom with Gaussian blob structures)
- Source: https://figshare.com/articles/dataset/BioSR/13744429
- Size: 400 images, 502x502 fluorescence; experiment uses 64x64 synthetic phantom
- Registered in `dataset_registry.yaml` as `biosr_factin`

For this baseline experiment, a deterministic fluorescence phantom (seed=42, cell body + nucleus + filaments + puncta) is used. Real-world widefield benchmarks on BioSR achieve 27+ dB PSNR with Richardson-Lucy and 31+ dB with CARE deep learning.

## PWM pipeline flowchart (mandatory)

```
x (world)
  ↓
SourceNode: photon_source — scales fluorescence by excitation strength (1.0)
  ↓
Element 1 (subrole=transport): conv2d — Gaussian PSF convolution (sigma=2.0, mode=reflect)
  ↓
SensorNode: photon_sensor — QE=0.9, gain=1.0, converts photon signal to electrons
  ↓
NoiseNode: poisson_gaussian_sensor — Poisson shot noise (peak=10000) + Gaussian read noise (sigma=0.01)
  ↓
y
```

## Element inventory & mismatch parameters

| # | node_id | primitive_id | subrole | learnable params | mismatch knob | bounds | prior |
|---|---------|-------------|---------|-----------------|---------------|--------|-------|
| 1 | source | photon_source | source | strength=1.0 | — | — | — |
| 2 | blur | conv2d | transport | sigma=2.0, mode=reflect | psf_sigma | [0.5, 3.0] | normal(2.0, 0.3) |
| 3 | sensor | photon_sensor | sensor | QE=0.9, gain=1.0 | qe_drift, gain | [0.8, 1.0], [0.9, 1.1] | normal |
| 4 | noise | poisson_gaussian_sensor | noise | peak_photons=10000, read_sigma=0.01 | — | — | — |

## Node-by-node trace (one sample)

Trace of intermediate tensor states at each node for one representative sample (seed=42).
Each row references a saved artifact in the RunBundle.

| stage | node_id | output shape | dtype | range [min, max] | artifact_path |
|-------|---------|-------------|-------|-------------------|---------------|
| 0 | input x | (64, 64) | float64 | [0.0500, 0.9392] | `artifacts/trace/00_input_x.npy` |
| 1 | source | (64, 64) | float64 | [0.0500, 0.9392] | `artifacts/trace/01_source.npy` |
| 2 | blur | (64, 64) | float64 | [0.0501, 0.8193] | `artifacts/trace/02_blur.npy` |
| 3 | sensor | (64, 64) | float64 | [0.0451, 0.7374] | `artifacts/trace/03_sensor.npy` |
| 4 | noise (y) | (64, 64) | float64 | [0.0059, 0.7376] | `artifacts/trace/04_noise_y.npy` |

PNG visualizations saved at: `artifacts/trace/png/`

## Workflow W1: Prompt-driven simulation + reconstruction

- **Prompt used:** `"Simulate widefield fluorescence microscopy and reconstruct using Richardson-Lucy deconvolution"`
- **ExperimentSpec summary:**
  - modality: widefield
  - mode: simulate -> invert
  - solver: richardson_lucy_2d
  - photon_budget: 10000 peak photons
- **Mode S results (simulate y):**
  - y shape: (64, 64)
  - y range: [0.0059, 0.7376]
  - SNR: 25.0 dB
- **Mode I results (reconstruct x_hat):**
  - x_hat shape: (64, 64)
  - Solver: richardson_lucy_2d, iterations: 50
- **Dataset metrics:**

  | Metric | Value |
  |--------|-------|
  | PSNR   | 26.33 dB |
  | SSIM   | 0.9527 |
  | NRMSE  | 0.0509 |

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
  - Parameters perturbed: PSF sigma (2.0 → 2.5, +25% wider blur)
  - Description: Objective defocus or wavelength mismatch causes PSF to be 25% wider than nominal calibration
- **Mode C fit results:**
  - Correction family: Pre
  - Parameters fitted: 1 parameter (PSF sigma) via grid search over [1.0, 4.0]
  - Best fitted sigma: 2.5
  - NLL before correction: 2682.1
  - NLL after correction: 2048.1
  - NLL decrease: 634.0 (23.6%)
- **Mode I recon using corrected operator A':**

  | Metric | A₀ (uncorrected) | A' (corrected) | Delta |
  |--------|-------------------|----------------|-------|
  | PSNR   | 25.35 dB | 25.55 dB | +0.20 dB |
  | SSIM   | 0.9391 | 0.9434 | +0.0043 |

## Test results summary

### Quick gate (pass/fail)

- [x] W1 simulate: PASS
- [x] W1 reconstruct (PSNR >= 18 dB): PASS (26.33 dB)
- [x] W2 operator correction (NLL decreases): PASS (23.6% decrease)
- [x] W2 corrected recon (beats uncorrected): PASS (+0.20 dB)
- [x] Report contract (flowchart + all headings): PASS
- [x] Node-by-node trace saved (>= 3 stages): PASS (5 stages)
- [x] RunBundle saved (with trace PNGs): PASS
- [x] Scoreboard updated: PASS

### Full metrics

| Check | Metric | Value | Threshold | Status |
|-------|--------|-------|-----------|--------|
| W1 PSNR | psnr | 26.33 dB | >= 18 dB | PASS |
| W1 SSIM | ssim | 0.9527 | >= 0.80 | PASS |
| W1 NRMSE | nrmse | 0.0509 | — | info |
| W2 NLL decrease | nll_decrease_pct | 23.6% | >= 5% | PASS |
| W2 PSNR gain | psnr_delta | +0.20 dB | > 0 | PASS |
| W2 SSIM gain | ssim_delta | +0.0043 | > 0 | PASS |
| Trace stages | n_stages | 5 | >= 3 | PASS |
| Trace PNGs | n_pngs | 5 | >= 3 | PASS |
| W1 wall time | w1_seconds | 0.02 s | — | info |
| W2 wall time | w2_seconds | 0.05 s | — | info |

## Reproducibility

- Seed: 42
- NumPy RNG state hash: 5d6a058f2c5c262e
- PWM version: 7394757
- Python version: 3.13.9
- Key package versions: numpy=2.3.5, scipy=1.16.3
- Platform: Linux x86_64
- Deterministic reproduction command:
  ```bash
  pwm_cli run --modality widefield --seed 42 --mode simulate,invert,calibrate
  ```
- Output hash (y): see RunBundle
- Output hash (x_hat): see RunBundle

## Saved artifacts

- RunBundle: `runs/run_widefield_exp_985bcedd/`
- Report: `pwm/reports/widefield.md`
- Scoreboard: `pwm/reports/scoreboard.yaml`
- Node trace (.npy): `runs/run_widefield_exp_985bcedd/artifacts/trace/*.npy`
- Node trace (.png): `runs/run_widefield_exp_985bcedd/artifacts/trace/png/*.png`
- W2 operator metadata: `runs/run_widefield_exp_985bcedd/artifacts/w2_operator_meta.json`
- Ground truth: `runs/run_widefield_exp_985bcedd/artifacts/x_true.npy`
- Measurement: `runs/run_widefield_exp_985bcedd/artifacts/y.npy`
- W1 reconstruction: `runs/run_widefield_exp_985bcedd/artifacts/x_hat.npy`
- W2 reconstructions: `runs/run_widefield_exp_985bcedd/artifacts/x_hat_w2_uncorrected.npy`, `x_hat_w2_corrected.npy`

## Next actions

- Test on BioSR F-Actin benchmark at 502x502 resolution
- Add CARE deep learning solver for improved PSNR
- Implement 3D widefield deconvolution with z-stack data
- Extend to multi-channel fluorescence
