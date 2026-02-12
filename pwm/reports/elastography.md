# Elastography — PWM Report

| Field | Value |
|-------|-------|
| Modality ID | `elastography` |
| Category | ultrasound_variant |
| Dataset | elastography_benchmark (synthetic proxy: phantom) |
| Date | 2026-02-12 |
| PWM version | `7394757` |
| Author | integritynoble |

## Modality overview

- Modality key: `elastography`
- Category: ultrasound_variant
- Forward model: y = Sensor(AcousticProp(ElasticWave(AcousticSource(x)))) + n
- Default solver: adjoint_backprojection
- Pipeline linearity: linear

Elastography imaging pipeline with 2 element(s) in the forward chain. Reconstruction uses adjoint_backprojection. Tested on a 64x64 synthetic phantom with seed=42.

| Parameter | Value |
|-----------|-------|
| Image size | 64 x 64 |
| Output shape | 64 |
| Noise model | Poisson-Gaussian |

## Standard dataset

- Name: elastography_benchmark (synthetic proxy: Gaussian-blob phantom)
- Source: synthetic
- Size: 1 image, 64x64
- Registered in `dataset_registry.yaml` as `elastography_benchmark`

For this baseline experiment, a deterministic phantom (seed=42, smooth Gaussian blobs on a 64x64 grid) is used.

## PWM pipeline flowchart (mandatory)

```
x (world)
  ↓
SourceNode: acoustic_source — strength=1.0
  ↓
Element 1 (subrole=interaction): elastic_wave_model — shear_speed_m_per_s=3.0, push_duration_us=100.0
  ↓
Element 2 (subrole=transport): acoustic_propagation — speed_of_sound=1540.0, n_sensors=32, x_shape=[64, 64]
  ↓
SensorNode: acoustic_receive_sensor — sensitivity=1.0
  ↓
NoiseNode: gaussian_sensor_noise — sigma=0.015, seed=0
  ↓
y
```

## Element inventory & mismatch parameters

| # | node_id | primitive_id | subrole | learnable params | mismatch knob | bounds | prior |
|---|---------|-------------|---------|-----------------|---------------|--------|-------|
| 1 | source | acoustic_source | source | strength=1.0 | — | — | — |
| 2 | elastic | elastic_wave_model | interaction | shear_speed_m_per_s=3.0, push_duration_us=100.0 | gain | [0.5, 2.0] | uniform |
| 3 | propagate | acoustic_propagation | transport | speed_of_sound=1540.0, n_sensors=32, x_shape=[64, 64] | — | — | — |
| 4 | sensor | acoustic_receive_sensor | sensor | sensitivity=1.0 | gain | [0.5, 2.0] | uniform |
| 5 | noise | gaussian_sensor_noise | noise | sigma=0.015, seed=0 | — | — | — |

## Node-by-node trace (one sample)

Trace of intermediate tensor states at each node for one representative sample (seed=42).
Each row references a saved artifact in the RunBundle.

| stage | node_id | output shape | dtype | range [min, max] | artifact_path |
|-------|---------|-------------|-------|-------------------|---------------|
| 0 | input x | (64, 64) | float64 | [0.0000, 1.0000] | `artifacts/trace/00_input_x.npy` |
| 1 | source | (64, 64) | float64 | [0.0000, 1.0000] | `artifacts/trace/01_source.npy` |
| 2 | elastic | (64,) | float64 | [0.0000, 1.0000] | `artifacts/trace/02_elastic.npy` |
| 3 | propagate | (64,) | float64 | [0.0000, 1.0000] | `artifacts/trace/03_propagate.npy` |
| 4 | sensor | (64,) | float64 | [0.0000, 1.0000] | `artifacts/trace/04_sensor.npy` |
| 5 | noise (y) | (64,) | float64 | [0.0000, 1.0000] | `artifacts/trace/05_noise_y.npy` |

PNG visualizations saved at: `artifacts/trace/png/`

## Workflow W1: Prompt-driven simulation + reconstruction

- **Prompt used:** `"Simulate elastography measurement of a phantom and reconstruct"`
- **ExperimentSpec summary:**
  - modality: elastography
  - mode: simulate -> invert
  - solver: adjoint_backprojection
- **Mode S results (simulate y):**
  - y shape: (64,)
  - SNR: 60.6 dB
- **Mode I results (reconstruct x_hat):**
  - x_hat shape: (64, 64)
  - Solver: adjoint_backprojection
- **Dataset metrics:**

  | Metric | Value |
  |--------|-------|
  | PSNR   | 10.64 dB |
  | SSIM   | 0.6226 |
  | NRMSE  | 0.3179 |

## Workflow W2: Operator correction mode (measured y + operator A)

- **Operator definition:**
  - A_definition: callable
  - A_extraction_method: graph_stripped
  - Operator chain: forward model stripped of noise node
  - A_sha256: 39fc751852741889
  - Linearity: linear
  - Notes (if linearized): N/A
- **Mismatch specification:**
  - Mismatch type: synthetic_injected
  - Parameters perturbed: sensitivity drift (1.0 -> 1.3)
  - Description: Synthetic parameter drift injected for calibration testing
- **Mode C fit results:**
  - Correction family: Pre
  - Parameters fitted: 1 parameter via grid search
  - NLL before correction: 106396960.8
  - NLL after correction: 1024.2
  - NLL decrease: 100.0%
- **Mode I recon using corrected operator A':**

  | Metric | A₀ (uncorrected) | A' (corrected) | Delta |
  |--------|-------------------|----------------|-------|
  | PSNR   | 10.64 dB | 53.81 dB | +43.17 dB |

## Test results summary

### Quick gate (pass/fail)

- [x] W1 simulate: PASS
- [x] W1 reconstruct (PSNR reported): PASS (10.64 dB)
- [x] W2 operator correction (NLL decrease): PASS (100.0%)
- [x] Report contract (flowchart + all headings): PASS
- [x] Node-by-node trace saved (>= 3 stages): PASS (6 stages)
- [x] RunBundle saved (with trace PNGs): PASS
- [x] Scoreboard updated: PASS

### Full metrics

| Check | Metric | Value | Threshold | Status |
|-------|--------|-------|-----------|--------|
| W1 PSNR | psnr | 10.64 dB | >= 1 dB | PASS |
| W1 SSIM | ssim | 0.6226 | — | info |
| W1 NRMSE | nrmse | 0.3179 | — | info |
| W2 NLL decrease | nll_decrease_pct | 100.0% | >= 0% | PASS |
| W2 PSNR delta | psnr_delta | +43.17 dB | — | info |
| Trace stages | n_stages | 6 | >= 3 | PASS |

## Reproducibility

- Seed: 42
- PWM version: 7394757
- Python version: 3.13.9
- Key package versions: numpy=2.3.5, scipy=1.16.3
- Platform: Linux x86_64
- Deterministic reproduction command:
  ```bash
  pwm_cli run --modality elastography --seed 42 --mode simulate,invert,calibrate
  ```
- Output hash (y): eb04bef65a683174
- Output hash (x_hat): 9b03bb06f8df7ce9

## Saved artifacts

- RunBundle: `runs/run_elastography_exp_26f87d54/`
- Report: `pwm/reports/elastography.md`
- Scoreboard: `pwm/reports/scoreboard.yaml`
- Node trace (.npy): `runs/run_elastography_exp_26f87d54/artifacts/trace/*.npy`
- Node trace (.png): `runs/run_elastography_exp_26f87d54/artifacts/trace/png/*.png`
- W2 operator metadata: `runs/run_elastography_exp_26f87d54/artifacts/w2_operator_meta.json`

## Next actions

- Test on real-world Elastography datasets at full resolution
- Add advanced reconstruction solvers for improved quality
- Investigate additional mismatch parameters
