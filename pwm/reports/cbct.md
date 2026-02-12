# CBCT — PWM Report

| Field | Value |
|-------|-------|
| Modality ID | `cbct` |
| Category | xray_variant |
| Dataset | cbct_benchmark (synthetic proxy: phantom) |
| Date | 2026-02-12 |
| PWM version | `7394757` |
| Author | integritynoble |

## Modality overview

- Modality key: `cbct`
- Category: xray_variant
- Forward model: y = Sensor(BeerLambert(Radon(XRaySource(x)))) + n
- Default solver: fbp_2d
- Pipeline linearity: nonlinear

CBCT imaging pipeline with 2 element(s) in the forward chain. Reconstruction uses fbp_2d. Tested on a 64x64 synthetic phantom with seed=42.

| Parameter | Value |
|-----------|-------|
| Image size | 64 x 64 |
| Output shape | 180 x 64 |
| Noise model | Poisson-Gaussian |

## Standard dataset

- Name: cbct_benchmark (synthetic proxy: Gaussian-blob phantom)
- Source: synthetic
- Size: 1 image, 64x64
- Registered in `dataset_registry.yaml` as `cbct_benchmark`

For this baseline experiment, a deterministic phantom (seed=42, smooth Gaussian blobs on a 64x64 grid) is used.

## PWM pipeline flowchart (mandatory)

```
x (world)
  ↓
SourceNode: xray_source — strength=1.0
  ↓
Element 1 (subrole=transport): ct_radon — n_angles=180, H=64, W=64
  ↓
Element 2 (subrole=transduction): beer_lambert — I_0=8000.0
  ↓
SensorNode: photon_sensor — quantum_efficiency=0.85, gain=1.0
  ↓
NoiseNode: poisson_only_sensor — peak_photons=50000.0, seed=0
  ↓
y
```

## Element inventory & mismatch parameters

| # | node_id | primitive_id | subrole | learnable params | mismatch knob | bounds | prior |
|---|---------|-------------|---------|-----------------|---------------|--------|-------|
| 1 | source | xray_source | source | strength=1.0 | — | — | — |
| 2 | radon | ct_radon | transport | n_angles=180, H=64, W=64 | gain | [0.5, 2.0] | uniform |
| 3 | beer_lambert | beer_lambert | transduction | I_0=8000.0 | — | — | — |
| 4 | sensor | photon_sensor | sensor | quantum_efficiency=0.85, gain=1.0 | gain | [0.5, 2.0] | uniform |
| 5 | noise | poisson_only_sensor | noise | peak_photons=50000.0, seed=0 | — | — | — |

## Node-by-node trace (one sample)

Trace of intermediate tensor states at each node for one representative sample (seed=42).
Each row references a saved artifact in the RunBundle.

| stage | node_id | output shape | dtype | range [min, max] | artifact_path |
|-------|---------|-------------|-------|-------------------|---------------|
| 0 | input x | (64, 64) | float64 | [0.0000, 1.0000] | `artifacts/trace/00_input_x.npy` |
| 1 | source | (64, 64) | float64 | [0.0000, 1.0000] | `artifacts/trace/01_source.npy` |
| 2 | radon | (180, 64) | float64 | [0.0000, 1.0000] | `artifacts/trace/02_radon.npy` |
| 3 | beer_lambert | (180, 64) | float64 | [0.0000, 1.0000] | `artifacts/trace/03_beer_lambert.npy` |
| 4 | sensor | (180, 64) | float64 | [0.0000, 1.0000] | `artifacts/trace/04_sensor.npy` |
| 5 | noise (y) | (180, 64) | float64 | [0.0000, 1.0000] | `artifacts/trace/05_noise_y.npy` |

PNG visualizations saved at: `artifacts/trace/png/`

## Workflow W1: Prompt-driven simulation + reconstruction

- **Prompt used:** `"Simulate cbct measurement of a phantom and reconstruct"`
- **ExperimentSpec summary:**
  - modality: cbct
  - mode: simulate -> invert
  - solver: fbp_2d
- **Mode S results (simulate y):**
  - y shape: (180, 64)
  - SNR: 77.4 dB
- **Mode I results (reconstruct x_hat):**
  - x_hat shape: (64, 64)
  - Solver: fbp_2d
- **Dataset metrics:**

  | Metric | Value |
  |--------|-------|
  | PSNR   | 7.38 dB |
  | SSIM   | -0.5592 |
  | NRMSE  | 0.4630 |

## Workflow W2: Operator correction mode (measured y + operator A)

- **Operator definition:**
  - A_definition: callable
  - A_extraction_method: graph_stripped
  - Operator chain: forward model stripped of noise node
  - A_sha256: b4d1ab1b7b7b2643
  - Linearity: nonlinear
  - Notes (if linearized): N/A
- **Mismatch specification:**
  - Mismatch type: synthetic_injected
  - Parameters perturbed: I_0 drift (8000 -> 10400)
  - Description: Synthetic parameter drift injected for calibration testing
- **Mode C fit results:**
  - Correction family: Pre
  - Parameters fitted: 1 parameter via grid search
  - NLL before correction: 19735511206.1
  - NLL after correction: 5761.1
  - NLL decrease: 100.0%
- **Mode I recon using corrected operator A':**

  | Metric | A₀ (uncorrected) | A' (corrected) | Delta |
  |--------|-------------------|----------------|-------|
  | PSNR   | 7.38 dB | 7.38 dB | +0.00 dB |

## Test results summary

### Quick gate (pass/fail)

- [x] W1 simulate: PASS
- [x] W1 reconstruct (PSNR reported): PASS (7.38 dB)
- [x] W2 operator correction (NLL decrease): PASS (100.0%)
- [x] Report contract (flowchart + all headings): PASS
- [x] Node-by-node trace saved (>= 3 stages): PASS (6 stages)
- [x] RunBundle saved (with trace PNGs): PASS
- [x] Scoreboard updated: PASS

### Full metrics

| Check | Metric | Value | Threshold | Status |
|-------|--------|-------|-----------|--------|
| W1 PSNR | psnr | 7.38 dB | >= 1 dB | PASS |
| W1 SSIM | ssim | -0.5592 | — | info |
| W1 NRMSE | nrmse | 0.4630 | — | info |
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
  pwm_cli run --modality cbct --seed 42 --mode simulate,invert,calibrate
  ```
- Output hash (y): 1b21efb027356338
- Output hash (x_hat): 8bdd40a137ce0764

## Saved artifacts

- RunBundle: `runs/run_cbct_exp_a328448e/`
- Report: `pwm/reports/cbct.md`
- Scoreboard: `pwm/reports/scoreboard.yaml`
- Node trace (.npy): `runs/run_cbct_exp_a328448e/artifacts/trace/*.npy`
- Node trace (.png): `runs/run_cbct_exp_a328448e/artifacts/trace/png/*.png`
- W2 operator metadata: `runs/run_cbct_exp_a328448e/artifacts/w2_operator_meta.json`

## Next actions

- Test on real-world CBCT datasets at full resolution
- Add advanced reconstruction solvers for improved quality
- Investigate additional mismatch parameters
