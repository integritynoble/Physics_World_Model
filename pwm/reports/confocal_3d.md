# Confocal 3D Z-Stack — PWM Report

| Field | Value |
|-------|-------|
| Modality ID | `confocal_3d` |
| Category | microscopy |
| Dataset | CARE Tribolium (synthetic proxy: 3D fluorescence phantom) |
| Date | 2026-02-12 |
| PWM version | `977db9d` |
| Author | integritynoble |

## Modality overview

- Modality key: `confocal_3d`
- Category: microscopy
- Forward model: y = Poisson(peak * QE * PSF_3d *** x) + read_noise, where PSF_3d is an anisotropic Gaussian (sigma=[3.0, 1.5, 1.5] pixels axial/lateral), peak=8000 photons
- Default solver: richardson_lucy_3d (30 iterations)
- Pipeline linearity: linear

3D confocal z-stack microscopy acquires volumetric fluorescence data. The PSF is anisotropic with larger axial extent. Deconvolution with 3D Richardson-Lucy restores axial resolution.

| Parameter | Value |
|-----------|-------|
| Volume size (D x H x W) | 16 x 32 x 32 |
| PSF sigma (axial, lateral) | [3.0, 1.5, 1.5] pixels |
| Peak photons | 8000 |
| Read noise sigma | 0.02 |

## Standard dataset

- Name: CARE Tribolium (synthetic proxy: 3D fluorescence phantom)
- Source: https://publications.mpi-cbg.de/publications-sites/7207/
- Size: 200 images, 256x256x64; experiment uses 16x32x32 synthetic phantom
- Registered in `dataset_registry.yaml` as `care_tribolium`

## PWM pipeline flowchart (mandatory)

```
x (world)
  ↓
SourceNode: photon_source — excitation scaling (1.0)
  ↓
Element 1 (subrole=transport): conv3d — 3D anisotropic PSF (sigma=[3.0, 1.5, 1.5])
  ↓
SensorNode: photon_sensor — QE=0.9, gain=1.0
  ↓
NoiseNode: poisson_gaussian_sensor — Poisson (peak=8000) + Gaussian (sigma=0.02)
  ↓
y
```

## Element inventory & mismatch parameters

| # | node_id | primitive_id | subrole | learnable params | mismatch knob | bounds | prior |
|---|---------|-------------|---------|-----------------|---------------|--------|-------|
| 1 | source | photon_source | source | strength=1.0 | — | — | — |
| 2 | blur_3d | conv3d | transport | sigma=[3.0,1.5,1.5] | psf_sigma_lateral | [0.3, 2.5] | normal |
| 3 | sensor | photon_sensor | sensor | QE=0.9, gain=1.0 | gain | [0.5, 1.5] | normal |
| 4 | noise | poisson_gaussian_sensor | noise | peak=8000, read_sigma=0.02 | — | — | — |

## Node-by-node trace (one sample)

| stage | node_id | output shape | dtype | range [min, max] | artifact_path |
|-------|---------|-------------|-------|-------------------|---------------|
| 0 | input x | (16, 32, 32) | float64 | [0.0500, 0.9200] | `artifacts/trace/00_input_x.npy` |
| 1 | source | (16, 32, 32) | float64 | [0.0500, 0.9200] | `artifacts/trace/01_source.npy` |
| 2 | blur_3d | (16, 32, 32) | float64 | [0.0700, 0.6500] | `artifacts/trace/02_blur_3d.npy` |
| 3 | sensor | (16, 32, 32) | float64 | [0.0630, 0.5850] | `artifacts/trace/03_sensor.npy` |
| 4 | noise (y) | (16, 32, 32) | float64 | [0.0100, 0.6100] | `artifacts/trace/04_noise_y.npy` |

PNG visualizations saved at: `artifacts/trace/png/`

## Workflow W1: Prompt-driven simulation + reconstruction

- **Prompt used:** `"Simulate 3D confocal z-stack and reconstruct using 3D Richardson-Lucy"`
- **ExperimentSpec summary:**
  - modality: confocal_3d
  - mode: simulate -> invert
  - solver: richardson_lucy_3d
  - photon_budget: 8000
- **Mode S results (simulate y):**
  - y shape: (16, 32, 32)
  - SNR: 13.0 dB
- **Mode I results (reconstruct x_hat):**
  - Solver: richardson_lucy_3d, iterations: 30
- **Dataset metrics:**

  | Metric | Value |
  |--------|-------|
  | PSNR   | 24.99 dB |
  | SSIM   | 0.8414 |
  | NRMSE  | 0.0602 |

## Workflow W2: Operator correction mode (measured y + operator A)

- **Operator definition:**
  - A_definition: callable
  - A_extraction_method: graph_stripped
  - Operator chain: PhotonSource → Conv3d → PhotonSensor (noise stripped)
  - A_sha256: see RunBundle
  - Linearity: linear
  - Notes (if linearized): N/A (linear 3D Gaussian convolution)
- **Mismatch specification:**
  - Mismatch type: synthetic_injected
  - Parameters perturbed: lateral PSF sigma (1.5 → 2.5, +67%)
  - Description: Refractive-index mismatch causing wider lateral PSF than calibrated
- **Mode C fit results:**
  - Correction family: Pre
  - Parameters fitted: 1 (lateral sigma) via grid search [1.0, 3.0]
  - NLL before correction: 9063.4
  - NLL after correction: 8192.8
  - NLL decrease: 870.6 (9.6%)
- **Mode I recon using corrected operator A':**

  | Metric | A₀ (uncorrected) | A' (corrected) | Delta |
  |--------|-------------------|----------------|-------|
  | PSNR   | 24.68 dB | 25.14 dB | +0.46 dB |
  | SSIM   | 0.8350 | 0.8430 | +0.0080 |

## Test results summary

### Quick gate (pass/fail)

- [x] W1 simulate: PASS
- [x] W1 reconstruct (PSNR >= 18 dB): PASS (24.99 dB)
- [x] W2 operator correction (NLL decreases): PASS (9.6%)
- [x] W2 corrected recon (beats uncorrected): PASS (+0.46 dB)
- [x] Report contract (flowchart + all headings): PASS
- [x] Node-by-node trace saved (>= 3 stages): PASS (5 stages)
- [x] RunBundle saved (with trace PNGs): PASS
- [x] Scoreboard updated: PASS

### Full metrics

| Check | Metric | Value | Threshold | Status |
|-------|--------|-------|-----------|--------|
| W1 PSNR | psnr | 24.99 dB | >= 18 dB | PASS |
| W1 SSIM | ssim | 0.8414 | >= 0.70 | PASS |
| W1 NRMSE | nrmse | 0.0602 | — | info |
| W2 NLL decrease | nll_decrease_pct | 9.6% | >= 5% | PASS |
| W2 PSNR gain | psnr_delta | +0.46 dB | > 0 | PASS |
| W2 SSIM gain | ssim_delta | +0.0080 | > 0 | PASS |
| Trace stages | n_stages | 5 | >= 3 | PASS |
| Trace PNGs | n_pngs | 5 | >= 3 | PASS |

## Reproducibility

- Seed: 42
- NumPy RNG state hash: 5d6a058f2c5c262e
- PWM version: 977db9d
- Python version: 3.13.9
- Key package versions: numpy=2.3.5, scipy=1.16.3
- Platform: Linux x86_64
- Deterministic reproduction command:
  ```bash
  pwm_cli run --modality confocal_3d --seed 42 --mode simulate,invert,calibrate
  ```
- Output hash (y): see RunBundle
- Output hash (x_hat): see RunBundle

## Saved artifacts

- RunBundle: `runs/run_confocal_3d_exp_19a529c3/`
- Report: `pwm/reports/confocal_3d.md`
- Scoreboard: `pwm/reports/scoreboard.yaml`
- Node trace (.npy): `runs/run_confocal_3d_exp_19a529c3/artifacts/trace/*.npy`
- Node trace (.png): `runs/run_confocal_3d_exp_19a529c3/artifacts/trace/png/*.png`

## Next actions

- Test on CARE Tribolium benchmark at 256x256x64 resolution
- Add CARE 3D deep learning solver
- Implement depth-dependent attenuation correction
