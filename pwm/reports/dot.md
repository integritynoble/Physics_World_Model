# Diffuse Optical Tomography (DOT) — PWM Report

| Field | Value |
|-------|-------|
| Modality ID | `dot` |
| Category | medical |
| Dataset | Diffuse phantom (synthetic proxy: Gaussian blob absorption map) |
| Date | 2026-02-12 |
| PWM version | `199aab9` |
| Author | integritynoble |

## Modality overview

- Modality key: `dot`
- Category: medical
- Forward model: y = Sensor(PSF ** x) + noise, where PSF is a large Gaussian (sigma=8.0 px) modeling photon diffusion through tissue, Sensor has QE=0.9, and noise is Poisson-Gaussian
- Default solver: richardson_lucy_2d (RL deconvolution, 100 iterations)
- Pipeline linearity: linear

Diffuse optical tomography (DOT) illuminates tissue with near-infrared (NIR) light and measures the diffusely transmitted or reflected photon distribution. Photon diffusion through scattering media is modeled as a large-kernel convolution (Gaussian PSF with sigma=8 px), representing the Green's function of the diffusion equation. Reconstruction uses Richardson-Lucy deconvolution to recover the absorption map.

| Parameter | Value |
|-----------|-------|
| Image size (H x W) | 64 x 64 |
| Diffusion PSF sigma | 8.0 pixels |
| PSF mode | constant (zero padding) |
| Peak photons | 5000 |
| Read noise sigma | 0.02 |
| Quantum efficiency | 0.9 |

## Standard dataset

- Name: Diffuse phantom (synthetic proxy: Gaussian blob absorption map)
- Source: synthetic
- Size: 64x64 phantom
- Registered in `dataset_registry.yaml` as `diffuse_phantom`

## PWM pipeline flowchart (mandatory)

```
x (world)
  ↓
SourceNode: photon_source — NIR illumination scaling (1.0)
  ↓
Element 1 (subrole=transport): conv2d — diffusion PSF convolution (sigma=8.0, mode=constant)
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
| 2 | diffuse | conv2d | transport | sigma=8.0, mode=constant | psf_sigma | [4.0, 16.0] | normal(8.0, 2.0) |
| 3 | sensor | photon_sensor | sensor | QE=0.9, gain=1.0 | gain | [0.5, 1.5] | normal |
| 4 | noise | poisson_gaussian_sensor | noise | peak=5000, read_sigma=0.02 | — | — | — |

## Node-by-node trace (one sample)

| stage | node_id | output shape | dtype | range [min, max] | artifact_path |
|-------|---------|-------------|-------|-------------------|---------------|
| 0 | input x | (64, 64) | float64 | [0.0500, 0.9400] | `artifacts/trace/00_input_x.npy` |
| 1 | source | (64, 64) | float64 | [0.0500, 0.9400] | `artifacts/trace/01_source.npy` |
| 2 | diffuse | (64, 64) | float64 | [0.0050, 0.3200] | `artifacts/trace/02_diffuse.npy` |
| 3 | sensor | (64, 64) | float64 | [0.0045, 0.2880] | `artifacts/trace/03_sensor.npy` |
| 4 | noise (y) | (64, 64) | float64 | [0.0010, 0.3000] | `artifacts/trace/04_noise_y.npy` |

PNG visualizations saved at: `artifacts/trace/png/`

## Workflow W1: Prompt-driven simulation + reconstruction

- **Prompt used:** `"Simulate diffuse optical tomography and reconstruct using Richardson-Lucy deconvolution"`
- **ExperimentSpec summary:**
  - modality: dot
  - mode: simulate -> invert
  - solver: richardson_lucy_2d
  - photon_budget: 5000 peak photons
- **Mode S results (simulate y):**
  - y shape: (64, 64)
  - y range: [0.0010, 0.3000]
  - SNR: 17.4 dB
- **Mode I results (reconstruct x_hat):**
  - x_hat shape: (64, 64)
  - Solver: richardson_lucy_2d, iterations: 100
- **Dataset metrics:**

  | Metric | Value |
  |--------|-------|
  | PSNR   | 17.13 dB |
  | SSIM   | 0.8211 |
  | NRMSE  | 0.1507 |

## Workflow W2: Operator correction mode (measured y + operator A)

- **Operator definition:**
  - A_definition: callable
  - A_extraction_method: graph_stripped
  - Operator chain: PhotonSource → Conv2d → PhotonSensor (noise stripped)
  - A_sha256: ce1180b2a4f93e71
  - Linearity: linear
  - Notes (if linearized): N/A (linear Gaussian convolution)
- **Mismatch specification:**
  - Mismatch type: synthetic_injected
  - Parameters perturbed: sensor gain (1.0 → 1.3, +30%)
  - Description: Detector gain drift from temperature or PMT aging effects
- **Mode C fit results:**
  - Correction family: Pre
  - Parameters fitted: 1 (gain) via grid search over [0.5, 2.0]
  - NLL before correction: 11477.7
  - NLL after correction: 2048.3
  - NLL decrease: 9429.4 (82.2%)
- **Mode I recon using corrected operator A':**

  | Metric | A₀ (uncorrected) | A' (corrected) | Delta |
  |--------|-------------------|----------------|-------|
  | PSNR   | 14.97 dB | 17.13 dB | +2.16 dB |
  | SSIM   | 0.7600 | 0.8211 | +0.0611 |

## Test results summary

### Quick gate (pass/fail)

- [x] W1 simulate: PASS
- [x] W1 reconstruct (PSNR >= 12 dB): PASS (17.13 dB)
- [x] W2 operator correction (NLL decreases): PASS (82.2% decrease)
- [x] W2 corrected recon (beats uncorrected): PASS (+2.16 dB)
- [x] Report contract (flowchart + all headings): PASS
- [x] Node-by-node trace saved (>= 3 stages): PASS (5 stages)
- [x] RunBundle saved (with trace PNGs): PASS
- [x] Scoreboard updated: PASS

### Full metrics

| Check | Metric | Value | Threshold | Status |
|-------|--------|-------|-----------|--------|
| W1 PSNR | psnr | 17.13 dB | >= 12 dB | PASS |
| W1 SSIM | ssim | 0.8211 | >= 0.50 | PASS |
| W1 NRMSE | nrmse | 0.1507 | — | info |
| W2 NLL decrease | nll_decrease_pct | 82.2% | >= 5% | PASS |
| W2 PSNR gain | psnr_delta | +2.16 dB | > 0 | PASS |
| W2 SSIM gain | ssim_delta | +0.0611 | > 0 | PASS |
| Trace stages | n_stages | 5 | >= 3 | PASS |
| Trace PNGs | n_pngs | 5 | >= 3 | PASS |
| W1 wall time | w1_seconds | 0.05 s | — | info |
| W2 wall time | w2_seconds | 0.08 s | — | info |

## Reproducibility

- Seed: 42
- NumPy RNG state hash: 5d6a058f2c5c262e
- PWM version: 199aab9
- Python version: 3.13.9
- Key package versions: numpy=2.3.5, scipy=1.16.3
- Platform: Linux x86_64
- Deterministic reproduction command:
  ```bash
  pwm_cli run --modality dot --seed 42 --mode simulate,invert,calibrate
  ```
- Output hash (y): see RunBundle
- Output hash (x_hat): see RunBundle

## Saved artifacts

- RunBundle: `runs/run_dot_exp_ce1180b2/`
- Report: `pwm/reports/dot.md`
- Scoreboard: `pwm/reports/scoreboard.yaml`
- Node trace (.npy): `runs/run_dot_exp_ce1180b2/artifacts/trace/*.npy`
- Node trace (.png): `runs/run_dot_exp_ce1180b2/artifacts/trace/png/*.png`
- W2 operator metadata: `runs/run_dot_exp_ce1180b2/artifacts/w2_operator_meta.json`
- Ground truth: `runs/run_dot_exp_ce1180b2/artifacts/x_true.npy`
- Measurement: `runs/run_dot_exp_ce1180b2/artifacts/y.npy`
- W1 reconstruction: `runs/run_dot_exp_ce1180b2/artifacts/x_hat.npy`
- W2 reconstructions: `runs/run_dot_exp_ce1180b2/artifacts/x_hat_w2_uncorrected.npy`, `x_hat_w2_corrected.npy`

## Next actions

- Test on real DOT data with multi-source/multi-detector geometry
- Implement diffusion equation solver for forward model (replace PSF approximation)
- Add multi-wavelength DOT for spectroscopic imaging
