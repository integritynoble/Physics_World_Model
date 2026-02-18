# DEXA — PWM Report

| Field | Value |
|-------|-------|
| Modality ID | `dexa` |
| Category | xray_variant |
| Dataset | dexa_benchmark (synthetic proxy: phantom) |
| Date | 2026-02-12 |
| PWM version | `7394757` |
| Author | integritynoble |

## Modality overview

- Modality key: `dexa`
- Category: xray_variant
- Forward model: y = Sensor(DualEnergyBeerLambert(XRaySource(x))) + n
- Default solver: pseudo_inverse
- Pipeline linearity: nonlinear

DEXA imaging pipeline with 1 element(s) in the forward chain. Reconstruction uses pseudo_inverse. Tested on a 64x64 synthetic phantom with seed=42.

| Parameter | Value |
|-----------|-------|
| Image size | 64 x 64 |
| Output shape | 64 x 64 |
| Noise model | Poisson-Gaussian |

## Standard dataset

- Name: dexa_benchmark (synthetic proxy: Gaussian-blob phantom)
- Source: synthetic
- Size: 1 image, 64x64
- Registered in `dataset_registry.yaml` as `dexa_benchmark`

For this baseline experiment, a deterministic phantom (seed=42, smooth Gaussian blobs on a 64x64 grid) is used.

## PWM pipeline flowchart (mandatory)

```
x (world)
  ↓
SourceNode: xray_source — strength=1.0
  ↓
Element 1 (subrole=transduction): dual_energy_beer_lambert — I_0_low=5000.0, I_0_high=8000.0
  ↓
SensorNode: photon_sensor — quantum_efficiency=0.85, gain=1.0
  ↓
NoiseNode: poisson_gaussian_sensor — peak_photons=25000.0, read_sigma=0.008, seed=0
  ↓
y
```

## Element inventory & mismatch parameters

| # | node_id | primitive_id | subrole | learnable params | mismatch knob | bounds | prior |
|---|---------|-------------|---------|-----------------|---------------|--------|-------|
| 1 | source | xray_source | source | strength=1.0 | — | — | — |
| 2 | dual_transmission | dual_energy_beer_lambert | transduction | I_0_low=5000.0, I_0_high=8000.0 | gain | [0.5, 2.0] | uniform |
| 3 | sensor | photon_sensor | sensor | quantum_efficiency=0.85, gain=1.0 | gain | [0.5, 2.0] | uniform |
| 4 | noise | poisson_gaussian_sensor | noise | peak_photons=25000.0, read_sigma=0.008, seed=0 | — | — | — |

## Node-by-node trace (one sample)

Trace of intermediate tensor states at each node for one representative sample (seed=42).
Each row references a saved artifact in the RunBundle.

| stage | node_id | output shape | dtype | range [min, max] | artifact_path |
|-------|---------|-------------|-------|-------------------|---------------|
| 0 | input x | (64, 64) | float64 | [0.0000, 1.0000] | `artifacts/trace/00_input_x.npy` |
| 1 | source | (64, 64) | float64 | [0.0000, 1.0000] | `artifacts/trace/01_source.npy` |
| 2 | dual_transmission | (64, 64) | float64 | [0.0000, 1.0000] | `artifacts/trace/02_dual_transmission.npy` |
| 3 | sensor | (64, 64) | float64 | [0.0000, 1.0000] | `artifacts/trace/03_sensor.npy` |
| 4 | noise (y) | (64, 64) | float64 | [0.0000, 1.0000] | `artifacts/trace/04_noise_y.npy` |

PNG visualizations saved at: `artifacts/trace/png/`

## Workflow W1: Prompt-driven simulation + reconstruction

- **Prompt used:** `"Simulate dexa measurement of a phantom and reconstruct"`
- **ExperimentSpec summary:**
  - modality: dexa
  - mode: simulate -> invert
  - solver: pseudo_inverse
- **Mode S results (simulate y):**
  - y shape: (64, 64)
  - SNR: 81.3 dB
- **Mode I results (reconstruct x_hat):**
  - x_hat shape: (64, 64)
  - Solver: pseudo_inverse
- **Dataset metrics:**

  | Metric | Value |
  |--------|-------|
  | PSNR   | 59.65 dB |
  | SSIM   | 1.0000 |
  | NRMSE  | 0.0011 |

## Workflow W2: Operator correction mode (measured y + operator A)

- **Operator definition:**
  - A_definition: callable
  - A_extraction_method: graph_stripped
  - Operator chain: forward model stripped of noise node
  - A_sha256: f0cfb9d8418ff368
  - Linearity: nonlinear
  - Notes (if linearized): N/A
- **Mismatch specification:**
  - Mismatch type: synthetic_injected
  - Parameters perturbed: I_0_low drift (5000 -> 6500)
  - Description: Synthetic parameter drift injected for calibration testing
- **Mode C fit results:**
  - Correction family: Pre
  - Parameters fitted: 1 parameter via grid search
  - NLL before correction: 19078073641.2
  - NLL after correction: 2048.1
  - NLL decrease: 100.0%
- **Mode I recon using corrected operator A':**

  | Metric | A₀ (uncorrected) | A' (corrected) | Delta |
  |--------|-------------------|----------------|-------|
  | PSNR   | 59.65 dB | 140.76 dB | +81.11 dB |

## Test results summary

### Quick gate (pass/fail)

- [x] W1 simulate: PASS
- [x] W1 reconstruct (PSNR reported): PASS (59.65 dB)
- [x] W2 operator correction (NLL decrease): PASS (100.0%)
- [x] Report contract (flowchart + all headings): PASS
- [x] Node-by-node trace saved (>= 3 stages): PASS (5 stages)
- [x] RunBundle saved (with trace PNGs): PASS
- [x] Scoreboard updated: PASS

### Full metrics

| Check | Metric | Value | Threshold | Status |
|-------|--------|-------|-----------|--------|
| W1 PSNR | psnr | 59.65 dB | >= 1 dB | PASS |
| W1 SSIM | ssim | 1.0000 | — | info |
| W1 NRMSE | nrmse | 0.0011 | — | info |
| W2 NLL decrease | nll_decrease_pct | 100.0% | >= 0% | PASS |
| W2 PSNR delta | psnr_delta | +81.11 dB | — | info |
| Trace stages | n_stages | 5 | >= 3 | PASS |

## Reproducibility

- Seed: 42
- PWM version: 7394757
- Python version: 3.13.9
- Key package versions: numpy=2.3.5, scipy=1.16.3
- Platform: Linux x86_64
- Deterministic reproduction command:
  ```bash
  pwm_cli run --modality dexa --seed 42 --mode simulate,invert,calibrate
  ```
- Output hash (y): bd3d6be998740199
- Output hash (x_hat): d15495c131267282

## Saved artifacts

- RunBundle: `runs/run_dexa_exp_95127813/`
- Report: `pwm/reports/dexa.md`
- Scoreboard: `pwm/reports/scoreboard.yaml`
- Node trace (.npy): `runs/run_dexa_exp_95127813/artifacts/trace/*.npy`
- Node trace (.png): `runs/run_dexa_exp_95127813/artifacts/trace/png/*.png`
- W2 operator metadata: `runs/run_dexa_exp_95127813/artifacts/w2_operator_meta.json`

## Next actions

- Test on real-world DEXA datasets at full resolution
- Add advanced reconstruction solvers for improved quality
- Investigate additional mismatch parameters
