# MRS — PWM Report

| Field | Value |
|-------|-------|
| Modality ID | `mrs` |
| Category | mri_variant |
| Dataset | mrs_benchmark (synthetic proxy: phantom) |
| Date | 2026-02-12 |
| PWM version | `7394757` |
| Author | integritynoble |

## Modality overview

- Modality key: `mrs`
- Category: mri_variant
- Forward model: y = Sensor(SequenceBlock(MRI_kspace(SpinSource(x)))) + n
- Default solver: adjoint_ifft
- Pipeline linearity: linear

MRS imaging pipeline with 2 element(s) in the forward chain. Reconstruction uses adjoint_ifft. Tested on a 64x64 synthetic phantom with seed=42.

| Parameter | Value |
|-----------|-------|
| Image size | 64 x 64 |
| Output shape | 64 x 64 |
| Noise model | Poisson-Gaussian |

## Standard dataset

- Name: mrs_benchmark (synthetic proxy: Gaussian-blob phantom)
- Source: synthetic
- Size: 1 image, 64x64
- Registered in `dataset_registry.yaml` as `mrs_benchmark`

For this baseline experiment, a deterministic phantom (seed=42, smooth Gaussian blobs on a 64x64 grid) is used.

## PWM pipeline flowchart (mandatory)

```
x (world)
  ↓
SourceNode: spin_source — strength=1.0
  ↓
Element 1 (subrole=encoding): mri_kspace — H=64, W=64, sampling_rate=1.0, seed=42
  ↓
Element 2 (subrole=encoding): sequence_block — TR_ms=3000.0, TE_ms=135.0, flip_angle_deg=90.0
  ↓
SensorNode: coil_sensor — sensitivity=1.0
  ↓
NoiseNode: complex_gaussian_sensor — sigma=0.02, seed=0
  ↓
y
```

## Element inventory & mismatch parameters

| # | node_id | primitive_id | subrole | learnable params | mismatch knob | bounds | prior |
|---|---------|-------------|---------|-----------------|---------------|--------|-------|
| 1 | source | spin_source | source | strength=1.0 | — | — | — |
| 2 | kspace | mri_kspace | encoding | H=64, W=64, sampling_rate=1.0, seed=42 | gain | [0.5, 2.0] | uniform |
| 3 | sequence | sequence_block | encoding | TR_ms=3000.0, TE_ms=135.0, flip_angle_deg=90.0 | — | — | — |
| 4 | sensor | coil_sensor | sensor | sensitivity=1.0 | gain | [0.5, 2.0] | uniform |
| 5 | noise | complex_gaussian_sensor | noise | sigma=0.02, seed=0 | — | — | — |

## Node-by-node trace (one sample)

Trace of intermediate tensor states at each node for one representative sample (seed=42).
Each row references a saved artifact in the RunBundle.

| stage | node_id | output shape | dtype | range [min, max] | artifact_path |
|-------|---------|-------------|-------|-------------------|---------------|
| 0 | input x | (64, 64) | float64 | [0.0000, 1.0000] | `artifacts/trace/00_input_x.npy` |
| 1 | source | (64, 64) | float64 | [0.0000, 1.0000] | `artifacts/trace/01_source.npy` |
| 2 | kspace | (64, 64) | float64 | [0.0000, 1.0000] | `artifacts/trace/02_kspace.npy` |
| 3 | sequence | (64, 64) | float64 | [0.0000, 1.0000] | `artifacts/trace/03_sequence.npy` |
| 4 | sensor | (64, 64) | float64 | [0.0000, 1.0000] | `artifacts/trace/04_sensor.npy` |
| 5 | noise (y) | (64, 64) | float64 | [0.0000, 1.0000] | `artifacts/trace/05_noise_y.npy` |

PNG visualizations saved at: `artifacts/trace/png/`

## Workflow W1: Prompt-driven simulation + reconstruction

- **Prompt used:** `"Simulate mrs measurement of a phantom and reconstruct"`
- **ExperimentSpec summary:**
  - modality: mrs
  - mode: simulate -> invert
  - solver: adjoint_ifft
- **Mode S results (simulate y):**
  - y shape: (64, 64)
  - SNR: 33.7 dB
- **Mode I results (reconstruct x_hat):**
  - x_hat shape: (64, 64)
  - Solver: adjoint_ifft
- **Dataset metrics:**

  | Metric | Value |
  |--------|-------|
  | PSNR   | 22.89 dB |
  | SSIM   | 0.9562 |
  | NRMSE  | 0.0776 |

## Workflow W2: Operator correction mode (measured y + operator A)

- **Operator definition:**
  - A_definition: callable
  - A_extraction_method: graph_stripped
  - Operator chain: forward model stripped of noise node
  - A_sha256: d8704249d566123a
  - Linearity: linear
  - Notes (if linearized): N/A
- **Mismatch specification:**
  - Mismatch type: synthetic_injected
  - Parameters perturbed: sensitivity drift (1.0 -> 1.3)
  - Description: Synthetic parameter drift injected for calibration testing
- **Mode C fit results:**
  - Correction family: Pre
  - Parameters fitted: 1 parameter via grid search
  - NLL before correction: 2066595.3
  - NLL after correction: 9708.8
  - NLL decrease: 99.5%
- **Mode I recon using corrected operator A':**

  | Metric | A₀ (uncorrected) | A' (corrected) | Delta |
  |--------|-------------------|----------------|-------|
  | PSNR   | 22.89 dB | 35.40 dB | +12.51 dB |

## Test results summary

### Quick gate (pass/fail)

- [x] W1 simulate: PASS
- [x] W1 reconstruct (PSNR reported): PASS (22.89 dB)
- [x] W2 operator correction (NLL decrease): PASS (99.5%)
- [x] Report contract (flowchart + all headings): PASS
- [x] Node-by-node trace saved (>= 3 stages): PASS (6 stages)
- [x] RunBundle saved (with trace PNGs): PASS
- [x] Scoreboard updated: PASS

### Full metrics

| Check | Metric | Value | Threshold | Status |
|-------|--------|-------|-----------|--------|
| W1 PSNR | psnr | 22.89 dB | >= 1 dB | PASS |
| W1 SSIM | ssim | 0.9562 | — | info |
| W1 NRMSE | nrmse | 0.0776 | — | info |
| W2 NLL decrease | nll_decrease_pct | 99.5% | >= 0% | PASS |
| W2 PSNR delta | psnr_delta | +12.51 dB | — | info |
| Trace stages | n_stages | 6 | >= 3 | PASS |

## Reproducibility

- Seed: 42
- PWM version: 7394757
- Python version: 3.13.9
- Key package versions: numpy=2.3.5, scipy=1.16.3
- Platform: Linux x86_64
- Deterministic reproduction command:
  ```bash
  pwm_cli run --modality mrs --seed 42 --mode simulate,invert,calibrate
  ```
- Output hash (y): f0ea832e2b6dcab7
- Output hash (x_hat): 1569b29e2865d905

## Saved artifacts

- RunBundle: `runs/run_mrs_exp_59fd660e/`
- Report: `pwm/reports/mrs.md`
- Scoreboard: `pwm/reports/scoreboard.yaml`
- Node trace (.npy): `runs/run_mrs_exp_59fd660e/artifacts/trace/*.npy`
- Node trace (.png): `runs/run_mrs_exp_59fd660e/artifacts/trace/png/*.png`
- W2 operator metadata: `runs/run_mrs_exp_59fd660e/artifacts/w2_operator_meta.json`

## Next actions

- Test on real-world MRS datasets at full resolution
- Add advanced reconstruction solvers for improved quality
- Investigate additional mismatch parameters
