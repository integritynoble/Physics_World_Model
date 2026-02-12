# Light-Sheet Fluorescence Microscopy — PWM Report

| Field | Value |
|-------|-------|
| Modality ID | `lightsheet` |
| Category | microscopy |
| Dataset | OpenSPIM Demo (synthetic proxy: fluorescence phantom) |
| Date | 2026-02-12 |
| PWM version | `94207db` |
| Author | integritynoble |

## Modality overview

- Modality key: `lightsheet`
- Category: microscopy
- Forward model: y = Poisson(peak * QE * PSF ** x) + read_noise, where PSF is a tight Gaussian (sigma=1.0 px) representing the thin light-sheet sectioning, QE=0.9
- Default solver: richardson_lucy_2d (50 iterations)
- Pipeline linearity: linear

Light-sheet fluorescence microscopy (LSFM/SPIM) illuminates the sample with a thin sheet of light perpendicular to the detection axis. This provides intrinsic optical sectioning with minimal photobleaching. The detection PSF is tight (sigma=1.0 px) due to the selective illumination plane.

| Parameter | Value |
|-----------|-------|
| Image size (H x W) | 64 x 64 |
| PSF sigma | 1.0 pixels |
| Peak photons | 15000 |
| Read noise sigma | 0.01 |

## Standard dataset

- Name: OpenSPIM Demo (synthetic proxy: fluorescence phantom)
- Source: https://openspim.org/table_of_contents
- Size: 50 images, 512x512x200; experiment uses 64x64 synthetic phantom
- Registered in `dataset_registry.yaml` as `openspim_demo`

## PWM pipeline flowchart (mandatory)

```
x (world)
  ↓
SourceNode: photon_source — excitation scaling (1.0)
  ↓
Element 1 (subrole=transport): conv2d — lateral PSF convolution (sigma=1.0, mode=reflect)
  ↓
SensorNode: photon_sensor — QE=0.9, gain=1.0
  ↓
NoiseNode: poisson_gaussian_sensor — Poisson (peak=15000) + Gaussian (sigma=0.01)
  ↓
y
```

## Element inventory & mismatch parameters

| # | node_id | primitive_id | subrole | learnable params | mismatch knob | bounds | prior |
|---|---------|-------------|---------|-----------------|---------------|--------|-------|
| 1 | source | photon_source | source | strength=1.0 | — | — | — |
| 2 | blur | conv2d | transport | sigma=1.0, mode=reflect | psf_sigma | [0.3, 3.0] | normal(1.0, 0.2) |
| 3 | sensor | photon_sensor | sensor | QE=0.9, gain=1.0 | gain | [0.8, 1.2] | normal |
| 4 | noise | poisson_gaussian_sensor | noise | peak=15000, read_sigma=0.01 | — | — | — |

## Node-by-node trace (one sample)

| stage | node_id | output shape | dtype | range [min, max] | artifact_path |
|-------|---------|-------------|-------|-------------------|---------------|
| 0 | input x | (64, 64) | float64 | [0.0500, 0.9400] | `artifacts/trace/00_input_x.npy` |
| 1 | source | (64, 64) | float64 | [0.0500, 0.9400] | `artifacts/trace/01_source.npy` |
| 2 | blur | (64, 64) | float64 | [0.0500, 0.9300] | `artifacts/trace/02_blur.npy` |
| 3 | sensor | (64, 64) | float64 | [0.0450, 0.8370] | `artifacts/trace/03_sensor.npy` |
| 4 | noise (y) | (64, 64) | float64 | [0.0200, 0.8500] | `artifacts/trace/04_noise_y.npy` |

PNG visualizations saved at: `artifacts/trace/png/`

## Workflow W1: Prompt-driven simulation + reconstruction

- **Prompt used:** `"Simulate light-sheet microscopy and reconstruct using Richardson-Lucy deconvolution"`
- **ExperimentSpec summary:**
  - modality: lightsheet
  - mode: simulate -> invert
  - solver: richardson_lucy_2d
  - photon_budget: 15000 peak photons
- **Mode S results (simulate y):**
  - y shape: (64, 64)
  - SNR: 25.3 dB
- **Mode I results (reconstruct x_hat):**
  - Solver: richardson_lucy_2d, iterations: 50
- **Dataset metrics:**

  | Metric | Value |
  |--------|-------|
  | PSNR   | 24.78 dB |
  | SSIM   | 0.9634 |
  | NRMSE  | 0.0621 |

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
  - Parameters perturbed: PSF sigma (1.0 → 2.0, +100%)
  - Description: Sheet-detection focus misalignment causing wider detection PSF
- **Mode C fit results:**
  - Correction family: Pre
  - Parameters fitted: 1 (PSF sigma) via grid search over [0.5, 3.0]
  - NLL before correction: 3796.9
  - NLL after correction: 2048.2
  - NLL decrease: 1748.7 (46.1%)
- **Mode I recon using corrected operator A':**

  | Metric | A₀ (uncorrected) | A' (corrected) | Delta |
  |--------|-------------------|----------------|-------|
  | PSNR   | 23.61 dB | 24.78 dB | +1.17 dB |
  | SSIM   | 0.9500 | 0.9634 | +0.0134 |

## Test results summary

### Quick gate (pass/fail)

- [x] W1 simulate: PASS
- [x] W1 reconstruct (PSNR >= 18 dB): PASS (24.78 dB)
- [x] W2 operator correction (NLL decreases): PASS (46.1% decrease)
- [x] W2 corrected recon (beats uncorrected): PASS (+1.17 dB)
- [x] Report contract (flowchart + all headings): PASS
- [x] Node-by-node trace saved (>= 3 stages): PASS (5 stages)
- [x] RunBundle saved (with trace PNGs): PASS
- [x] Scoreboard updated: PASS

### Full metrics

| Check | Metric | Value | Threshold | Status |
|-------|--------|-------|-----------|--------|
| W1 PSNR | psnr | 24.78 dB | >= 18 dB | PASS |
| W1 SSIM | ssim | 0.9634 | >= 0.80 | PASS |
| W1 NRMSE | nrmse | 0.0621 | — | info |
| W2 NLL decrease | nll_decrease_pct | 46.1% | >= 5% | PASS |
| W2 PSNR gain | psnr_delta | +1.17 dB | > 0 | PASS |
| W2 SSIM gain | ssim_delta | +0.0134 | > 0 | PASS |
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
  pwm_cli run --modality lightsheet --seed 42 --mode simulate,invert,calibrate
  ```
- Output hash (y): see RunBundle
- Output hash (x_hat): see RunBundle

## Saved artifacts

- RunBundle: `runs/run_lightsheet_exp_5f9f97c7/`
- Report: `pwm/reports/lightsheet.md`
- Scoreboard: `pwm/reports/scoreboard.yaml`

## Next actions

- Implement 3D light-sheet reconstruction with z-stack data
- Add Fourier notch destriping solver for stripe artifact removal
- Test on OpenSPIM benchmark data
