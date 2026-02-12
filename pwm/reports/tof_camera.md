# ToF Camera — PWM Report

| Field | Value |
|-------|-------|
| Modality ID | `tof_camera` |
| Category | depth_tof |
| Dataset | tof_benchmark (synthetic proxy: phantom) |
| Date | 2026-02-12 |
| PWM version | `7394757` |
| Author | integritynoble |

## Modality overview

- Modality key: `tof_camera`
- Category: depth_tof
- Forward model: y = Sensor(ToFGate(PhotonSource(x))) + n
- Default solver: pseudo_inverse
- Pipeline linearity: linear

ToF Camera imaging pipeline with 1 element(s) in the forward chain. Reconstruction uses pseudo_inverse. Tested on a 64x64 synthetic phantom with seed=42.

| Parameter | Value |
|-----------|-------|
| Image size | 64 x 64 |
| Output shape | 64 x 64 |
| Noise model | Poisson-Gaussian |

## Standard dataset

- Name: tof_benchmark (synthetic proxy: Gaussian-blob phantom)
- Source: synthetic
- Size: 1 image, 64x64
- Registered in `dataset_registry.yaml` as `tof_benchmark`

For this baseline experiment, a deterministic phantom (seed=42, smooth Gaussian blobs on a 64x64 grid) is used.

## PWM pipeline flowchart (mandatory)

```
x (world)
  ↓
SourceNode: photon_source — strength=1.0
  ↓
Element 1 (subrole=encoding): tof_gate — n_bins=64, bin_width_ns=1.0, timing_jitter_ns=0.1
  ↓
SensorNode: spad_tof_sensor — n_bins=64, dead_time_ns=50.0, qe=0.3
  ↓
NoiseNode: poisson_gaussian_sensor — peak_photons=5000.0, read_sigma=0.02, seed=0
  ↓
y
```

## Element inventory & mismatch parameters

| # | node_id | primitive_id | subrole | learnable params | mismatch knob | bounds | prior |
|---|---------|-------------|---------|-----------------|---------------|--------|-------|
| 1 | source | photon_source | source | strength=1.0 | — | — | — |
| 2 | tof_gate | tof_gate | encoding | n_bins=64, bin_width_ns=1.0, timing_jitter_ns=0.1 | gain | [0.5, 2.0] | uniform |
| 3 | sensor | spad_tof_sensor | sensor | n_bins=64, dead_time_ns=50.0, qe=0.3 | gain | [0.5, 2.0] | uniform |
| 4 | noise | poisson_gaussian_sensor | noise | peak_photons=5000.0, read_sigma=0.02, seed=0 | — | — | — |

## Node-by-node trace (one sample)

Trace of intermediate tensor states at each node for one representative sample (seed=42).
Each row references a saved artifact in the RunBundle.

| stage | node_id | output shape | dtype | range [min, max] | artifact_path |
|-------|---------|-------------|-------|-------------------|---------------|
| 0 | input x | (64, 64) | float64 | [0.0000, 1.0000] | `artifacts/trace/00_input_x.npy` |
| 1 | source | (64, 64) | float64 | [0.0000, 1.0000] | `artifacts/trace/01_source.npy` |
| 2 | tof_gate | (64, 64) | float64 | [0.0000, 1.0000] | `artifacts/trace/02_tof_gate.npy` |
| 3 | sensor | (64, 64) | float64 | [0.0000, 1.0000] | `artifacts/trace/03_sensor.npy` |
| 4 | noise (y) | (64, 64) | float64 | [0.0000, 1.0000] | `artifacts/trace/04_noise_y.npy` |

PNG visualizations saved at: `artifacts/trace/png/`

## Workflow W1: Prompt-driven simulation + reconstruction

- **Prompt used:** `"Simulate tof_camera measurement of a phantom and reconstruct"`
- **ExperimentSpec summary:**
  - modality: tof_camera
  - mode: simulate -> invert
  - solver: pseudo_inverse
- **Mode S results (simulate y):**
  - y shape: (64, 64)
  - SNR: 26.5 dB
- **Mode I results (reconstruct x_hat):**
  - x_hat shape: (64, 64)
  - Solver: pseudo_inverse
- **Dataset metrics:**

  | Metric | Value |
  |--------|-------|
  | PSNR   | 35.92 dB |
  | SSIM   | 0.9974 |
  | NRMSE  | 0.0173 |

## Workflow W2: Operator correction mode (measured y + operator A)

- **Operator definition:**
  - A_definition: callable
  - A_extraction_method: graph_stripped
  - Operator chain: forward model stripped of noise node
  - A_sha256: 48675dcb22b3d783
  - Linearity: linear
  - Notes (if linearized): N/A
- **Mismatch specification:**
  - Mismatch type: synthetic_injected
  - Parameters perturbed: gain drift (1.0 -> 1.3)
  - Description: Synthetic parameter drift injected for calibration testing
- **Mode C fit results:**
  - Correction family: Pre
  - Parameters fitted: 1 parameter via grid search
  - NLL before correction: 64509.6
  - NLL after correction: 2048.0
  - NLL decrease: 96.8%
- **Mode I recon using corrected operator A':**

  | Metric | A₀ (uncorrected) | A' (corrected) | Delta |
  |--------|-------------------|----------------|-------|
  | PSNR   | 35.92 dB | 60.20 dB | +24.28 dB |

## Test results summary

### Quick gate (pass/fail)

- [x] W1 simulate: PASS
- [x] W1 reconstruct (PSNR reported): PASS (35.92 dB)
- [x] W2 operator correction (NLL decrease): PASS (96.8%)
- [x] Report contract (flowchart + all headings): PASS
- [x] Node-by-node trace saved (>= 3 stages): PASS (5 stages)
- [x] RunBundle saved (with trace PNGs): PASS
- [x] Scoreboard updated: PASS

### Full metrics

| Check | Metric | Value | Threshold | Status |
|-------|--------|-------|-----------|--------|
| W1 PSNR | psnr | 35.92 dB | >= 1 dB | PASS |
| W1 SSIM | ssim | 0.9974 | — | info |
| W1 NRMSE | nrmse | 0.0173 | — | info |
| W2 NLL decrease | nll_decrease_pct | 96.8% | >= 0% | PASS |
| W2 PSNR delta | psnr_delta | +24.28 dB | — | info |
| Trace stages | n_stages | 5 | >= 3 | PASS |

## Reproducibility

- Seed: 42
- PWM version: 7394757
- Python version: 3.13.9
- Key package versions: numpy=2.3.5, scipy=1.16.3
- Platform: Linux x86_64
- Deterministic reproduction command:
  ```bash
  pwm_cli run --modality tof_camera --seed 42 --mode simulate,invert,calibrate
  ```
- Output hash (y): 30d81d3c73d1b434
- Output hash (x_hat): bf472f8104504a6f

## Saved artifacts

- RunBundle: `runs/run_tof_camera_exp_05de8930/`
- Report: `pwm/reports/tof_camera.md`
- Scoreboard: `pwm/reports/scoreboard.yaml`
- Node trace (.npy): `runs/run_tof_camera_exp_05de8930/artifacts/trace/*.npy`
- Node trace (.png): `runs/run_tof_camera_exp_05de8930/artifacts/trace/png/*.png`
- W2 operator metadata: `runs/run_tof_camera_exp_05de8930/artifacts/w2_operator_meta.json`

## Next actions

- Test on real-world ToF Camera datasets at full resolution
- Add advanced reconstruction solvers for improved quality
- Investigate additional mismatch parameters
