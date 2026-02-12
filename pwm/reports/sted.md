# STED — PWM Report

| Field | Value |
|-------|-------|
| Modality ID | `sted` |
| Category | microscopy |
| Dataset | sted_benchmark (synthetic proxy: phantom) |
| Date | 2026-02-12 |
| PWM version | `7394757` |
| Author | integritynoble |

## Modality overview

- Modality key: `sted`
- Category: microscopy
- Forward model: y = Sensor(Conv2D(SaturationDepletion(PhotonSource(x)))) + n
- Default solver: richardson_lucy_2d
- Pipeline linearity: nonlinear

STED imaging pipeline with 2 element(s) in the forward chain. Reconstruction uses richardson_lucy_2d. Tested on a 64x64 synthetic phantom with seed=42.

| Parameter | Value |
|-----------|-------|
| Image size | 64 x 64 |
| Output shape | 64 x 64 |
| Noise model | Poisson-Gaussian |

## Standard dataset

- Name: sted_benchmark (synthetic proxy: Gaussian-blob phantom)
- Source: synthetic
- Size: 1 image, 64x64
- Registered in `dataset_registry.yaml` as `sted_benchmark`

For this baseline experiment, a deterministic phantom (seed=42, smooth Gaussian blobs on a 64x64 grid) is used.

## PWM pipeline flowchart (mandatory)

```
x (world)
  ↓
SourceNode: photon_source — strength=1.0
  ↓
Element 1 (subrole=interaction): saturation_depletion — depletion_factor=0.5, psf_sigma=1.0
  ↓
Element 2 (subrole=transport): conv2d — sigma=0.8, mode=reflect
  ↓
SensorNode: photon_sensor — quantum_efficiency=0.45, gain=1.0
  ↓
NoiseNode: poisson_gaussian_sensor — peak_photons=3000.0, read_sigma=0.03, seed=0
  ↓
y
```

## Element inventory & mismatch parameters

| # | node_id | primitive_id | subrole | learnable params | mismatch knob | bounds | prior |
|---|---------|-------------|---------|-----------------|---------------|--------|-------|
| 1 | source | photon_source | source | strength=1.0 | — | — | — |
| 2 | depletion | saturation_depletion | interaction | depletion_factor=0.5, psf_sigma=1.0 | gain | [0.5, 2.0] | uniform |
| 3 | blur | conv2d | transport | sigma=0.8, mode=reflect | — | — | — |
| 4 | sensor | photon_sensor | sensor | quantum_efficiency=0.45, gain=1.0 | gain | [0.5, 2.0] | uniform |
| 5 | noise | poisson_gaussian_sensor | noise | peak_photons=3000.0, read_sigma=0.03, seed=0 | — | — | — |

## Node-by-node trace (one sample)

Trace of intermediate tensor states at each node for one representative sample (seed=42).
Each row references a saved artifact in the RunBundle.

| stage | node_id | output shape | dtype | range [min, max] | artifact_path |
|-------|---------|-------------|-------|-------------------|---------------|
| 0 | input x | (64, 64) | float64 | [0.0000, 1.0000] | `artifacts/trace/00_input_x.npy` |
| 1 | source | (64, 64) | float64 | [0.0000, 1.0000] | `artifacts/trace/01_source.npy` |
| 2 | depletion | (64, 64) | float64 | [0.0000, 1.0000] | `artifacts/trace/02_depletion.npy` |
| 3 | blur | (64, 64) | float64 | [0.0000, 1.0000] | `artifacts/trace/03_blur.npy` |
| 4 | sensor | (64, 64) | float64 | [0.0000, 1.0000] | `artifacts/trace/04_sensor.npy` |
| 5 | noise (y) | (64, 64) | float64 | [0.0000, 1.0000] | `artifacts/trace/05_noise_y.npy` |

PNG visualizations saved at: `artifacts/trace/png/`

## Workflow W1: Prompt-driven simulation + reconstruction

- **Prompt used:** `"Simulate sted measurement of a phantom and reconstruct"`
- **ExperimentSpec summary:**
  - modality: sted
  - mode: simulate -> invert
  - solver: richardson_lucy_2d
- **Mode S results (simulate y):**
  - y shape: (64, 64)
  - SNR: 3.9 dB
- **Mode I results (reconstruct x_hat):**
  - x_hat shape: (64, 64)
  - Solver: richardson_lucy_2d
- **Dataset metrics:**

  | Metric | Value |
  |--------|-------|
  | PSNR   | 11.83 dB |
  | SSIM   | 0.1727 |
  | NRMSE  | 0.2773 |

## Workflow W2: Operator correction mode (measured y + operator A)

- **Operator definition:**
  - A_definition: callable
  - A_extraction_method: graph_stripped
  - Operator chain: forward model stripped of noise node
  - A_sha256: 9ebcc7e46be345f7
  - Linearity: nonlinear
  - Notes (if linearized): N/A
- **Mismatch specification:**
  - Mismatch type: synthetic_injected
  - Parameters perturbed: gain drift (1.0 -> 1.3)
  - Description: Synthetic parameter drift injected for calibration testing
- **Mode C fit results:**
  - Correction family: Pre
  - Parameters fitted: 1 parameter via grid search
  - NLL before correction: 2492.4
  - NLL after correction: 2048.1
  - NLL decrease: 17.8%
- **Mode I recon using corrected operator A':**

  | Metric | A₀ (uncorrected) | A' (corrected) | Delta |
  |--------|-------------------|----------------|-------|
  | PSNR   | 11.83 dB | 11.40 dB | -0.43 dB |

## Test results summary

### Quick gate (pass/fail)

- [x] W1 simulate: PASS
- [x] W1 reconstruct (PSNR reported): PASS (11.83 dB)
- [x] W2 operator correction (NLL decrease): PASS (17.8%)
- [x] Report contract (flowchart + all headings): PASS
- [x] Node-by-node trace saved (>= 3 stages): PASS (6 stages)
- [x] RunBundle saved (with trace PNGs): PASS
- [x] Scoreboard updated: PASS

### Full metrics

| Check | Metric | Value | Threshold | Status |
|-------|--------|-------|-----------|--------|
| W1 PSNR | psnr | 11.83 dB | >= 1 dB | PASS |
| W1 SSIM | ssim | 0.1727 | — | info |
| W1 NRMSE | nrmse | 0.2773 | — | info |
| W2 NLL decrease | nll_decrease_pct | 17.8% | >= 0% | PASS |
| W2 PSNR delta | psnr_delta | -0.43 dB | — | info |
| Trace stages | n_stages | 6 | >= 3 | PASS |

## Reproducibility

- Seed: 42
- PWM version: 7394757
- Python version: 3.13.9
- Key package versions: numpy=2.3.5, scipy=1.16.3
- Platform: Linux x86_64
- Deterministic reproduction command:
  ```bash
  pwm_cli run --modality sted --seed 42 --mode simulate,invert,calibrate
  ```
- Output hash (y): e50a9ccad50d189c
- Output hash (x_hat): b8f0fbaf9e423baa

## Saved artifacts

- RunBundle: `runs/run_sted_exp_40ed0df0/`
- Report: `pwm/reports/sted.md`
- Scoreboard: `pwm/reports/scoreboard.yaml`
- Node trace (.npy): `runs/run_sted_exp_40ed0df0/artifacts/trace/*.npy`
- Node trace (.png): `runs/run_sted_exp_40ed0df0/artifacts/trace/png/*.png`
- W2 operator metadata: `runs/run_sted_exp_40ed0df0/artifacts/w2_operator_meta.json`

## Next actions

- Test on real-world STED datasets at full resolution
- Add advanced reconstruction solvers for improved quality
- Investigate additional mismatch parameters
