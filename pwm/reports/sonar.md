# Sonar — PWM Report

| Field | Value |
|-------|-------|
| Modality ID | `sonar` |
| Category | radar_sonar |
| Dataset | sonar_benchmark (synthetic proxy: phantom) |
| Date | 2026-02-12 |
| PWM version | `7394757` |
| Author | integritynoble |

## Modality overview

- Modality key: `sonar`
- Category: radar_sonar
- Forward model: y = Sensor(BeamformDelay(AcousticSource(x))) + n
- Default solver: adjoint_backprojection
- Pipeline linearity: linear

Sonar imaging pipeline with 1 element(s) in the forward chain. Reconstruction uses adjoint_backprojection. Tested on a 64x64 synthetic phantom with seed=42.

| Parameter | Value |
|-----------|-------|
| Image size | 64 x 64 |
| Output shape | 64 |
| Noise model | Poisson-Gaussian |

## Standard dataset

- Name: sonar_benchmark (synthetic proxy: Gaussian-blob phantom)
- Source: synthetic
- Size: 1 image, 64x64
- Registered in `dataset_registry.yaml` as `sonar_benchmark`

For this baseline experiment, a deterministic phantom (seed=42, smooth Gaussian blobs on a 64x64 grid) is used.

## PWM pipeline flowchart (mandatory)

```
x (world)
  ↓
SourceNode: acoustic_source — strength=1.0
  ↓
Element 1 (subrole=transport): beamform_delay — n_elements=32
  ↓
SensorNode: transducer_sensor — sensitivity=1.0
  ↓
NoiseNode: poisson_gaussian_sensor — peak_photons=10000.0, read_sigma=0.02, seed=0
  ↓
y
```

## Element inventory & mismatch parameters

| # | node_id | primitive_id | subrole | learnable params | mismatch knob | bounds | prior |
|---|---------|-------------|---------|-----------------|---------------|--------|-------|
| 1 | source | acoustic_source | source | strength=1.0 | — | — | — |
| 2 | beamform | beamform_delay | transport | n_elements=32 | gain | [0.5, 2.0] | uniform |
| 3 | sensor | transducer_sensor | sensor | sensitivity=1.0 | gain | [0.5, 2.0] | uniform |
| 4 | noise | poisson_gaussian_sensor | noise | peak_photons=10000.0, read_sigma=0.02, seed=0 | — | — | — |

## Node-by-node trace (one sample)

Trace of intermediate tensor states at each node for one representative sample (seed=42).
Each row references a saved artifact in the RunBundle.

| stage | node_id | output shape | dtype | range [min, max] | artifact_path |
|-------|---------|-------------|-------|-------------------|---------------|
| 0 | input x | (64, 64) | float64 | [0.0000, 1.0000] | `artifacts/trace/00_input_x.npy` |
| 1 | source | (64, 64) | float64 | [0.0000, 1.0000] | `artifacts/trace/01_source.npy` |
| 2 | beamform | (64,) | float64 | [0.0000, 1.0000] | `artifacts/trace/02_beamform.npy` |
| 3 | sensor | (64,) | float64 | [0.0000, 1.0000] | `artifacts/trace/03_sensor.npy` |
| 4 | noise (y) | (64,) | float64 | [0.0000, 1.0000] | `artifacts/trace/04_noise_y.npy` |

PNG visualizations saved at: `artifacts/trace/png/`

## Workflow W1: Prompt-driven simulation + reconstruction

- **Prompt used:** `"Simulate sonar measurement of a phantom and reconstruct"`
- **ExperimentSpec summary:**
  - modality: sonar
  - mode: simulate -> invert
  - solver: adjoint_backprojection
- **Mode S results (simulate y):**
  - y shape: (64,)
  - SNR: 18.3 dB
- **Mode I results (reconstruct x_hat):**
  - x_hat shape: (64, 64)
  - Solver: adjoint_backprojection
- **Dataset metrics:**

  | Metric | Value |
  |--------|-------|
  | PSNR   | 8.81 dB |
  | SSIM   | 0.4249 |
  | NRMSE  | 0.3927 |

## Workflow W2: Operator correction mode (measured y + operator A)

- **Operator definition:**
  - A_definition: callable
  - A_extraction_method: graph_stripped
  - Operator chain: forward model stripped of noise node
  - A_sha256: 4f4287c4b3316a76
  - Linearity: linear
  - Notes (if linearized): N/A
- **Mismatch specification:**
  - Mismatch type: synthetic_injected
  - Parameters perturbed: sensitivity drift (1.0 -> 1.3)
  - Description: Synthetic parameter drift injected for calibration testing
- **Mode C fit results:**
  - Correction family: Pre
  - Parameters fitted: 1 parameter via grid search
  - NLL before correction: 217.2
  - NLL after correction: 32.1
  - NLL decrease: 85.2%
- **Mode I recon using corrected operator A':**

  | Metric | A₀ (uncorrected) | A' (corrected) | Delta |
  |--------|-------------------|----------------|-------|
  | PSNR   | 8.81 dB | 4.09 dB | -4.72 dB |

## Test results summary

### Quick gate (pass/fail)

- [x] W1 simulate: PASS
- [x] W1 reconstruct (PSNR reported): PASS (8.81 dB)
- [x] W2 operator correction (NLL decrease): PASS (85.2%)
- [x] Report contract (flowchart + all headings): PASS
- [x] Node-by-node trace saved (>= 3 stages): PASS (5 stages)
- [x] RunBundle saved (with trace PNGs): PASS
- [x] Scoreboard updated: PASS

### Full metrics

| Check | Metric | Value | Threshold | Status |
|-------|--------|-------|-----------|--------|
| W1 PSNR | psnr | 8.81 dB | >= 1 dB | PASS |
| W1 SSIM | ssim | 0.4249 | — | info |
| W1 NRMSE | nrmse | 0.3927 | — | info |
| W2 NLL decrease | nll_decrease_pct | 85.2% | >= 0% | PASS |
| W2 PSNR delta | psnr_delta | -4.72 dB | — | info |
| Trace stages | n_stages | 5 | >= 3 | PASS |

## Reproducibility

- Seed: 42
- PWM version: 7394757
- Python version: 3.13.9
- Key package versions: numpy=2.3.5, scipy=1.16.3
- Platform: Linux x86_64
- Deterministic reproduction command:
  ```bash
  pwm_cli run --modality sonar --seed 42 --mode simulate,invert,calibrate
  ```
- Output hash (y): 8d3cecd249473f25
- Output hash (x_hat): 2ee14cdf17f9d18b

## Saved artifacts

- RunBundle: `runs/run_sonar_exp_d571f604/`
- Report: `pwm/reports/sonar.md`
- Scoreboard: `pwm/reports/scoreboard.yaml`
- Node trace (.npy): `runs/run_sonar_exp_d571f604/artifacts/trace/*.npy`
- Node trace (.png): `runs/run_sonar_exp_d571f604/artifacts/trace/png/*.png`
- W2 operator metadata: `runs/run_sonar_exp_d571f604/artifacts/w2_operator_meta.json`

## Next actions

- Test on real-world Sonar datasets at full resolution
- Add advanced reconstruction solvers for improved quality
- Investigate additional mismatch parameters
