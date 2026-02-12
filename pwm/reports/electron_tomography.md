# Electron Tomography — PWM Report

| Field | Value |
|-------|-------|
| Modality ID | `electron_tomography` |
| Category | electron |
| Dataset | electron_tomo_benchmark (synthetic proxy: phantom) |
| Date | 2026-02-12 |
| PWM version | `7394757` |
| Author | integritynoble |

## Modality overview

- Modality key: `electron_tomography`
- Category: electron
- Forward model: y = Sensor(ThinObjectPhase(ElectronBeamSource(x))) + n
- Default solver: pseudo_inverse
- Pipeline linearity: nonlinear

Electron Tomography imaging pipeline with 1 element(s) in the forward chain. Reconstruction uses pseudo_inverse. Tested on a 64x64 synthetic phantom with seed=42.

| Parameter | Value |
|-----------|-------|
| Image size | 64 x 64 |
| Output shape | 64 x 64 |
| Noise model | Poisson-Gaussian |

## Standard dataset

- Name: electron_tomo_benchmark (synthetic proxy: Gaussian-blob phantom)
- Source: synthetic
- Size: 1 image, 64x64
- Registered in `dataset_registry.yaml` as `electron_tomo_benchmark`

For this baseline experiment, a deterministic phantom (seed=42, smooth Gaussian blobs on a 64x64 grid) is used.

## PWM pipeline flowchart (mandatory)

```
x (world)
  ↓
SourceNode: electron_beam_source — accelerating_voltage_kv=300.0, beam_current_na=0.1
  ↓
Element 1 (subrole=interaction): thin_object_phase — sigma=0.00729, thickness_nm=50.0
  ↓
SensorNode: electron_detector_sensor — detector_type=BF, collection_efficiency=0.7, gain=1.0
  ↓
NoiseNode: gaussian_sensor_noise — sigma=0.02
  ↓
y
```

## Element inventory & mismatch parameters

| # | node_id | primitive_id | subrole | learnable params | mismatch knob | bounds | prior |
|---|---------|-------------|---------|-----------------|---------------|--------|-------|
| 1 | source | electron_beam_source | source | accelerating_voltage_kv=300.0, beam_current_na=0.1 | — | — | — |
| 2 | interaction | thin_object_phase | interaction | sigma=0.00729, thickness_nm=50.0 | gain | [0.5, 2.0] | uniform |
| 3 | sensor | electron_detector_sensor | sensor | detector_type=BF, collection_efficiency=0.7, gain=1.0 | gain | [0.5, 2.0] | uniform |
| 4 | noise | gaussian_sensor_noise | noise | sigma=0.02 | — | — | — |

## Node-by-node trace (one sample)

Trace of intermediate tensor states at each node for one representative sample (seed=42).
Each row references a saved artifact in the RunBundle.

| stage | node_id | output shape | dtype | range [min, max] | artifact_path |
|-------|---------|-------------|-------|-------------------|---------------|
| 0 | input x | (64, 64) | float64 | [0.0000, 1.0000] | `artifacts/trace/00_input_x.npy` |
| 1 | source | (64, 64) | float64 | [0.0000, 1.0000] | `artifacts/trace/01_source.npy` |
| 2 | interaction | (64, 64) | float64 | [0.0000, 1.0000] | `artifacts/trace/02_interaction.npy` |
| 3 | sensor | (64, 64) | float64 | [0.0000, 1.0000] | `artifacts/trace/03_sensor.npy` |
| 4 | noise (y) | (64, 64) | float64 | [0.0000, 1.0000] | `artifacts/trace/04_noise_y.npy` |

PNG visualizations saved at: `artifacts/trace/png/`

## Workflow W1: Prompt-driven simulation + reconstruction

- **Prompt used:** `"Simulate electron_tomography measurement of a phantom and reconstruct"`
- **ExperimentSpec summary:**
  - modality: electron_tomography
  - mode: simulate -> invert
  - solver: pseudo_inverse
- **Mode S results (simulate y):**
  - y shape: (64, 64)
  - SNR: 10.9 dB
- **Mode I results (reconstruct x_hat):**
  - x_hat shape: (64, 64)
  - Solver: pseudo_inverse
- **Dataset metrics:**

  | Metric | Value |
  |--------|-------|
  | PSNR   | 8.83 dB |
  | SSIM   | 0.0039 |
  | NRMSE  | 0.3917 |

## Workflow W2: Operator correction mode (measured y + operator A)

- **Operator definition:**
  - A_definition: callable
  - A_extraction_method: graph_stripped
  - Operator chain: forward model stripped of noise node
  - A_sha256: 54264c2882211e3d
  - Linearity: nonlinear
  - Notes (if linearized): N/A
- **Mismatch specification:**
  - Mismatch type: synthetic_injected
  - Parameters perturbed: gain drift (1.0 -> 1.3)
  - Description: Synthetic parameter drift injected for calibration testing
- **Mode C fit results:**
  - Correction family: Pre
  - Parameters fitted: 1 parameter via grid search
  - NLL before correction: 4331.6
  - NLL after correction: 2048.1
  - NLL decrease: 52.7%
- **Mode I recon using corrected operator A':**

  | Metric | A₀ (uncorrected) | A' (corrected) | Delta |
  |--------|-------------------|----------------|-------|
  | PSNR   | 8.83 dB | 8.83 dB | +0.00 dB |

## Test results summary

### Quick gate (pass/fail)

- [x] W1 simulate: PASS
- [x] W1 reconstruct (PSNR reported): PASS (8.83 dB)
- [x] W2 operator correction (NLL decrease): PASS (52.7%)
- [x] Report contract (flowchart + all headings): PASS
- [x] Node-by-node trace saved (>= 3 stages): PASS (5 stages)
- [x] RunBundle saved (with trace PNGs): PASS
- [x] Scoreboard updated: PASS

### Full metrics

| Check | Metric | Value | Threshold | Status |
|-------|--------|-------|-----------|--------|
| W1 PSNR | psnr | 8.83 dB | >= 1 dB | PASS |
| W1 SSIM | ssim | 0.0039 | — | info |
| W1 NRMSE | nrmse | 0.3917 | — | info |
| W2 NLL decrease | nll_decrease_pct | 52.7% | >= 0% | PASS |
| W2 PSNR delta | psnr_delta | +0.00 dB | — | info |
| Trace stages | n_stages | 5 | >= 3 | PASS |

## Reproducibility

- Seed: 42
- PWM version: 7394757
- Python version: 3.13.9
- Key package versions: numpy=2.3.5, scipy=1.16.3
- Platform: Linux x86_64
- Deterministic reproduction command:
  ```bash
  pwm_cli run --modality electron_tomography --seed 42 --mode simulate,invert,calibrate
  ```
- Output hash (y): 78c12ef8a04bb952
- Output hash (x_hat): c0410e807a5ee002

## Saved artifacts

- RunBundle: `runs/run_electron_tomography_exp_f969255f/`
- Report: `pwm/reports/electron_tomography.md`
- Scoreboard: `pwm/reports/scoreboard.yaml`
- Node trace (.npy): `runs/run_electron_tomography_exp_f969255f/artifacts/trace/*.npy`
- Node trace (.png): `runs/run_electron_tomography_exp_f969255f/artifacts/trace/png/*.png`
- W2 operator metadata: `runs/run_electron_tomography_exp_f969255f/artifacts/w2_operator_meta.json`

## Next actions

- Test on real-world Electron Tomography datasets at full resolution
- Add advanced reconstruction solvers for improved quality
- Investigate additional mismatch parameters
