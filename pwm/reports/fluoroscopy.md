# Fluoroscopy — PWM Report

| Field | Value |
|-------|-------|
| Modality ID | `fluoroscopy` |
| Category | xray_variant |
| Dataset | fluoroscopy_benchmark (synthetic proxy: phantom) |
| Date | 2026-02-12 |
| PWM version | `7394757` |
| Author | integritynoble |

## Modality overview

- Modality key: `fluoroscopy`
- Category: xray_variant
- Forward model: y = Sensor(FluoroInteg(BeerLambert(XRaySource(x)))) + n
- Default solver: pseudo_inverse
- Pipeline linearity: nonlinear

Fluoroscopy imaging pipeline with 2 element(s) in the forward chain. Reconstruction uses pseudo_inverse. Tested on a 64x64 synthetic phantom with seed=42.

| Parameter | Value |
|-----------|-------|
| Image size | 64 x 64 |
| Output shape | 64 x 64 |
| Noise model | Poisson-Gaussian |

## Standard dataset

- Name: fluoroscopy_benchmark (synthetic proxy: Gaussian-blob phantom)
- Source: synthetic
- Size: 1 image, 64x64
- Registered in `dataset_registry.yaml` as `fluoroscopy_benchmark`

For this baseline experiment, a deterministic phantom (seed=42, smooth Gaussian blobs on a 64x64 grid) is used.

## PWM pipeline flowchart (mandatory)

```
x (world)
  ↓
SourceNode: xray_source — strength=1.0
  ↓
Element 1 (subrole=transduction): beer_lambert — I_0=5000.0
  ↓
Element 2 (subrole=encoding): fluoro_temporal_integrator — n_frames=8, decay=0.9
  ↓
SensorNode: xray_detector_sensor — scintillator_efficiency=0.7, gain=1.0
  ↓
NoiseNode: poisson_gaussian_sensor — peak_photons=20000.0, read_sigma=0.01, seed=0
  ↓
y
```

## Element inventory & mismatch parameters

| # | node_id | primitive_id | subrole | learnable params | mismatch knob | bounds | prior |
|---|---------|-------------|---------|-----------------|---------------|--------|-------|
| 1 | source | xray_source | source | strength=1.0 | — | — | — |
| 2 | transmission | beer_lambert | transduction | I_0=5000.0 | gain | [0.5, 2.0] | uniform |
| 3 | temporal_int | fluoro_temporal_integrator | encoding | n_frames=8, decay=0.9 | — | — | — |
| 4 | sensor | xray_detector_sensor | sensor | scintillator_efficiency=0.7, gain=1.0 | gain | [0.5, 2.0] | uniform |
| 5 | noise | poisson_gaussian_sensor | noise | peak_photons=20000.0, read_sigma=0.01, seed=0 | — | — | — |

## Node-by-node trace (one sample)

Trace of intermediate tensor states at each node for one representative sample (seed=42).
Each row references a saved artifact in the RunBundle.

| stage | node_id | output shape | dtype | range [min, max] | artifact_path |
|-------|---------|-------------|-------|-------------------|---------------|
| 0 | input x | (64, 64) | float64 | [0.0000, 1.0000] | `artifacts/trace/00_input_x.npy` |
| 1 | source | (64, 64) | float64 | [0.0000, 1.0000] | `artifacts/trace/01_source.npy` |
| 2 | transmission | (64, 64) | float64 | [0.0000, 1.0000] | `artifacts/trace/02_transmission.npy` |
| 3 | temporal_int | (64, 64) | float64 | [0.0000, 1.0000] | `artifacts/trace/03_temporal_int.npy` |
| 4 | sensor | (64, 64) | float64 | [0.0000, 1.0000] | `artifacts/trace/04_sensor.npy` |
| 5 | noise (y) | (64, 64) | float64 | [0.0000, 1.0000] | `artifacts/trace/05_noise_y.npy` |

PNG visualizations saved at: `artifacts/trace/png/`

## Workflow W1: Prompt-driven simulation + reconstruction

- **Prompt used:** `"Simulate fluoroscopy measurement of a phantom and reconstruct"`
- **ExperimentSpec summary:**
  - modality: fluoroscopy
  - mode: simulate -> invert
  - solver: pseudo_inverse
- **Mode S results (simulate y):**
  - y shape: (64, 64)
  - SNR: 77.9 dB
- **Mode I results (reconstruct x_hat):**
  - x_hat shape: (64, 64)
  - Solver: pseudo_inverse
- **Dataset metrics:**

  | Metric | Value |
  |--------|-------|
  | PSNR   | 72.28 dB |
  | SSIM   | 1.0000 |
  | NRMSE  | 0.0003 |

## Workflow W2: Operator correction mode (measured y + operator A)

- **Operator definition:**
  - A_definition: callable
  - A_extraction_method: graph_stripped
  - Operator chain: forward model stripped of noise node
  - A_sha256: e5804b63b7cea5b6
  - Linearity: nonlinear
  - Notes (if linearized): N/A
- **Mismatch specification:**
  - Mismatch type: synthetic_injected
  - Parameters perturbed: I_0 drift (5000 -> 6500)
  - Description: Synthetic parameter drift injected for calibration testing
- **Mode C fit results:**
  - Correction family: Pre
  - Parameters fitted: 1 parameter via grid search
  - NLL before correction: 8732844959.8
  - NLL after correction: 2048.2
  - NLL decrease: 100.0%
- **Mode I recon using corrected operator A':**

  | Metric | A₀ (uncorrected) | A' (corrected) | Delta |
  |--------|-------------------|----------------|-------|
  | PSNR   | 72.28 dB | 72.28 dB | +0.00 dB |

## Test results summary

### Quick gate (pass/fail)

- [x] W1 simulate: PASS
- [x] W1 reconstruct (PSNR reported): PASS (72.28 dB)
- [x] W2 operator correction (NLL decrease): PASS (100.0%)
- [x] Report contract (flowchart + all headings): PASS
- [x] Node-by-node trace saved (>= 3 stages): PASS (6 stages)
- [x] RunBundle saved (with trace PNGs): PASS
- [x] Scoreboard updated: PASS

### Full metrics

| Check | Metric | Value | Threshold | Status |
|-------|--------|-------|-----------|--------|
| W1 PSNR | psnr | 72.28 dB | >= 1 dB | PASS |
| W1 SSIM | ssim | 1.0000 | — | info |
| W1 NRMSE | nrmse | 0.0003 | — | info |
| W2 NLL decrease | nll_decrease_pct | 100.0% | >= 0% | PASS |
| W2 PSNR delta | psnr_delta | +0.00 dB | — | info |
| Trace stages | n_stages | 6 | >= 3 | PASS |

## Reproducibility

- Seed: 42
- PWM version: 7394757
- Python version: 3.13.9
- Key package versions: numpy=2.3.5, scipy=1.16.3
- Platform: Linux x86_64
- Deterministic reproduction command:
  ```bash
  pwm_cli run --modality fluoroscopy --seed 42 --mode simulate,invert,calibrate
  ```
- Output hash (y): 5169beef45e8b4b6
- Output hash (x_hat): 6217c105c441e19b

## Saved artifacts

- RunBundle: `runs/run_fluoroscopy_exp_56ee32d4/`
- Report: `pwm/reports/fluoroscopy.md`
- Scoreboard: `pwm/reports/scoreboard.yaml`
- Node trace (.npy): `runs/run_fluoroscopy_exp_56ee32d4/artifacts/trace/*.npy`
- Node trace (.png): `runs/run_fluoroscopy_exp_56ee32d4/artifacts/trace/png/*.png`
- W2 operator metadata: `runs/run_fluoroscopy_exp_56ee32d4/artifacts/w2_operator_meta.json`

## Next actions

- Test on real-world Fluoroscopy datasets at full resolution
- Add advanced reconstruction solvers for improved quality
- Investigate additional mismatch parameters
