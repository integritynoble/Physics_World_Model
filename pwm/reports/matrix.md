# Matrix — PWM Report

| Field | Value |
|-------|-------|
| Modality ID | `matrix` |
| Category | compressive |
| Dataset | matrix_benchmark (synthetic proxy: phantom) |
| Date | 2026-02-12 |
| PWM version | `7394757` |
| Author | integritynoble |

## Modality overview

- Modality key: `matrix`
- Category: compressive
- Forward model: y = Sensor(RandomMask(GenericSource(x))) + n
- Default solver: pseudo_inverse
- Pipeline linearity: linear

Matrix imaging pipeline with 1 element(s) in the forward chain. Reconstruction uses pseudo_inverse. Tested on a 64x64 synthetic phantom with seed=42.

| Parameter | Value |
|-----------|-------|
| Image size | 64 x 64 |
| Output shape | 1024 |
| Noise model | Poisson-Gaussian |

## Standard dataset

- Name: matrix_benchmark (synthetic proxy: Gaussian-blob phantom)
- Source: synthetic
- Size: 1 image, 64x64
- Registered in `dataset_registry.yaml` as `matrix_benchmark`

For this baseline experiment, a deterministic phantom (seed=42, smooth Gaussian blobs on a 64x64 grid) is used.

## PWM pipeline flowchart (mandatory)

```
x (world)
  ↓
SourceNode: generic_source — strength=1.0
  ↓
Element 1 (subrole=encoding): random_mask — seed=42, H=64, W=64, sampling_rate=0.25
  ↓
SensorNode: generic_sensor — gain=1.0
  ↓
NoiseNode: poisson_gaussian_sensor — peak_photons=10000.0, read_sigma=0.01, seed=0
  ↓
y
```

## Element inventory & mismatch parameters

| # | node_id | primitive_id | subrole | learnable params | mismatch knob | bounds | prior |
|---|---------|-------------|---------|-----------------|---------------|--------|-------|
| 1 | source | generic_source | source | strength=1.0 | — | — | — |
| 2 | project | random_mask | encoding | seed=42, H=64, W=64, sampling_rate=0.25 | gain | [0.5, 2.0] | uniform |
| 3 | sensor | generic_sensor | sensor | gain=1.0 | gain | [0.5, 2.0] | uniform |
| 4 | noise | poisson_gaussian_sensor | noise | peak_photons=10000.0, read_sigma=0.01, seed=0 | — | — | — |

## Node-by-node trace (one sample)

Trace of intermediate tensor states at each node for one representative sample (seed=42).
Each row references a saved artifact in the RunBundle.

| stage | node_id | output shape | dtype | range [min, max] | artifact_path |
|-------|---------|-------------|-------|-------------------|---------------|
| 0 | input x | (64, 64) | float64 | [0.0000, 1.0000] | `artifacts/trace/00_input_x.npy` |
| 1 | source | (64, 64) | float64 | [0.0000, 1.0000] | `artifacts/trace/01_source.npy` |
| 2 | project | (1024,) | float64 | [0.0000, 1.0000] | `artifacts/trace/02_project.npy` |
| 3 | sensor | (1024,) | float64 | [0.0000, 1.0000] | `artifacts/trace/03_sensor.npy` |
| 4 | noise (y) | (1024,) | float64 | [0.0000, 1.0000] | `artifacts/trace/04_noise_y.npy` |

PNG visualizations saved at: `artifacts/trace/png/`

## Workflow W1: Prompt-driven simulation + reconstruction

- **Prompt used:** `"Simulate matrix measurement of a phantom and reconstruct"`
- **ExperimentSpec summary:**
  - modality: matrix
  - mode: simulate -> invert
  - solver: pseudo_inverse
- **Mode S results (simulate y):**
  - y shape: (1024,)
  - SNR: -3.3 dB
- **Mode I results (reconstruct x_hat):**
  - x_hat shape: (64, 64)
  - Solver: pseudo_inverse
- **Dataset metrics:**

  | Metric | Value |
  |--------|-------|
  | PSNR   | 10.41 dB |
  | SSIM   | -0.0059 |
  | NRMSE  | 0.3266 |

## Workflow W2: Operator correction mode (measured y + operator A)

- **Operator definition:**
  - A_definition: callable
  - A_extraction_method: graph_stripped
  - Operator chain: forward model stripped of noise node
  - A_sha256: 7957779ebeacbe25
  - Linearity: linear
  - Notes (if linearized): N/A
- **Mismatch specification:**
  - Mismatch type: synthetic_injected
  - Parameters perturbed: gain drift (1.0 -> 1.3)
  - Description: Synthetic parameter drift injected for calibration testing
- **Mode C fit results:**
  - Correction family: Pre
  - Parameters fitted: 1 parameter via grid search
  - NLL before correction: 600.3
  - NLL after correction: 424.6
  - NLL decrease: 29.3%
- **Mode I recon using corrected operator A':**

  | Metric | A₀ (uncorrected) | A' (corrected) | Delta |
  |--------|-------------------|----------------|-------|
  | PSNR   | 10.41 dB | 10.12 dB | -0.29 dB |

## Test results summary

### Quick gate (pass/fail)

- [x] W1 simulate: PASS
- [x] W1 reconstruct (PSNR reported): PASS (10.41 dB)
- [x] W2 operator correction (NLL decrease): PASS (29.3%)
- [x] Report contract (flowchart + all headings): PASS
- [x] Node-by-node trace saved (>= 3 stages): PASS (5 stages)
- [x] RunBundle saved (with trace PNGs): PASS
- [x] Scoreboard updated: PASS

### Full metrics

| Check | Metric | Value | Threshold | Status |
|-------|--------|-------|-----------|--------|
| W1 PSNR | psnr | 10.41 dB | >= 1 dB | PASS |
| W1 SSIM | ssim | -0.0059 | — | info |
| W1 NRMSE | nrmse | 0.3266 | — | info |
| W2 NLL decrease | nll_decrease_pct | 29.3% | >= 0% | PASS |
| W2 PSNR delta | psnr_delta | -0.29 dB | — | info |
| Trace stages | n_stages | 5 | >= 3 | PASS |

## Reproducibility

- Seed: 42
- PWM version: 7394757
- Python version: 3.13.9
- Key package versions: numpy=2.3.5, scipy=1.16.3
- Platform: Linux x86_64
- Deterministic reproduction command:
  ```bash
  pwm_cli run --modality matrix --seed 42 --mode simulate,invert,calibrate
  ```
- Output hash (y): 4e00b2923bf30342
- Output hash (x_hat): c4250660441823e9

## Saved artifacts

- RunBundle: `runs/run_matrix_exp_5298bf5b/`
- Report: `pwm/reports/matrix.md`
- Scoreboard: `pwm/reports/scoreboard.yaml`
- Node trace (.npy): `runs/run_matrix_exp_5298bf5b/artifacts/trace/*.npy`
- Node trace (.png): `runs/run_matrix_exp_5298bf5b/artifacts/trace/png/*.png`
- W2 operator metadata: `runs/run_matrix_exp_5298bf5b/artifacts/w2_operator_meta.json`

## Next actions

- Test on real-world Matrix datasets at full resolution
- Add advanced reconstruction solvers for improved quality
- Investigate additional mismatch parameters
