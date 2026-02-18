# Single-Photon Emission CT (SPECT) — PWM Report

| Field | Value |
|-------|-------|
| Modality ID | `spect` |
| Category | nuclear |
| Dataset | SPECT benchmark (synthetic proxy: emission activity phantom) |
| Date | 2026-02-12 |
| PWM version | `199aab9` |
| Author | integritynoble |

## Modality overview

- Modality key: `spect`
- Category: nuclear
- Forward model: y = Noise(Sensor(EmissionProjection(Source(x)))), where EmissionProjection computes sinogram projections at 32 angles through a parallel-hole collimator, PhotonSensor detects with QE=0.85, and noise is Poisson (peak=100000)
- Default solver: adjoint backprojection
- Pipeline linearity: linear

Single-photon emission computed tomography (SPECT) detects gamma rays emitted by radiotracers (e.g., 99mTc) through a collimated gamma camera rotating around the patient. The emission projection operator computes line integrals of the activity distribution at 32 angles through a parallel-hole collimator. Unlike PET, SPECT does not have coincidence detection, so scatter correction is simpler but spatial resolution is lower.

| Parameter | Value |
|-----------|-------|
| Image size (H x W) | 64 x 64 |
| Projection angles | 32 |
| Measurements (M) | 2048 (32 x 64) |
| Quantum efficiency | 0.85 |
| Peak photons | 100000 |
| Noise model | Poisson only |

## Standard dataset

- Name: SPECT benchmark (synthetic proxy: emission activity phantom)
- Source: synthetic
- Size: 64x64 phantom
- Registered in `dataset_registry.yaml` as `spect_benchmark`

## PWM pipeline flowchart (mandatory)

```
x (world)
  ↓
SourceNode: generic_source — emission activity scaling (1.0)
  ↓
Element 1 (subrole=transport): emission_projection — collimated sinogram projection (n_angles=32)
  ↓
SensorNode: photon_sensor — QE=0.85, gain=1.0
  ↓
NoiseNode: poisson_only_sensor — Poisson shot noise (peak=100000)
  ↓
y
```

## Element inventory & mismatch parameters

| # | node_id | primitive_id | subrole | learnable params | mismatch knob | bounds | prior |
|---|---------|-------------|---------|-----------------|---------------|--------|-------|
| 1 | source | generic_source | source | strength=1.0 | — | — | — |
| 2 | projection | emission_projection | transport | n_angles=32, x_shape=[64,64] | projection_angle_offset | [-2, 2] | normal(0, 0.5) |
| 3 | sensor | photon_sensor | sensor | QE=0.85, gain=1.0 | gain | [0.5, 1.5] | normal |
| 4 | noise | poisson_only_sensor | noise | peak=100000 | — | — | — |

## Node-by-node trace (one sample)

| stage | node_id | output shape | dtype | range [min, max] | artifact_path |
|-------|---------|-------------|-------|-------------------|---------------|
| 0 | input x | (64, 64) | float64 | [0.0000, 0.5000] | `artifacts/trace/00_input_x.npy` |
| 1 | source | (64, 64) | float64 | [0.0000, 0.5000] | `artifacts/trace/01_source.npy` |
| 2 | projection | (32, 64) | float64 | [0.0000, 15.6000] | `artifacts/trace/02_projection.npy` |
| 3 | sensor | (32, 64) | float64 | [0.0000, 13.2600] | `artifacts/trace/03_sensor.npy` |
| 4 | noise (y) | (32, 64) | float64 | [0.0000, 14.0000] | `artifacts/trace/04_noise_y.npy` |

PNG visualizations saved at: `artifacts/trace/png/`

## Workflow W1: Prompt-driven simulation + reconstruction

- **Prompt used:** `"Simulate SPECT emission scan and reconstruct using adjoint backprojection"`
- **ExperimentSpec summary:**
  - modality: spect
  - mode: simulate -> invert
  - solver: adjoint backprojection
  - photon_budget: peak=100000
- **Mode S results (simulate y):**
  - y shape: (32, 64)
  - y range: [0.0000, 14.0000]
  - SNR: 60.2 dB
- **Mode I results (reconstruct x_hat):**
  - x_hat shape: (64, 64)
  - Solver: adjoint backprojection
- **Dataset metrics:**

  | Metric | Value |
  |--------|-------|
  | PSNR   | 10.64 dB |
  | SSIM   | 0.6227 |
  | NRMSE  | 0.3178 |

Note: The moderate PSNR reflects the limited angular sampling (32 angles) and simple adjoint reconstruction. OSEM with collimator-detector response modeling achieves 20+ dB on clinical SPECT data.

## Workflow W2: Operator correction mode (measured y + operator A)

- **Operator definition:**
  - A_definition: callable
  - A_extraction_method: graph_stripped
  - Operator chain: GenericSource → EmissionProjection → PhotonSensor (noise stripped)
  - A_sha256: 17579c0a6b2d8e45
  - Linearity: linear
  - Notes (if linearized): N/A
- **Mismatch specification:**
  - Mismatch type: synthetic_injected
  - Parameters perturbed: sensor gain (1.0 → 1.3, +30%)
  - Description: Detector gain drift from scintillator crystal degradation or PMT aging
- **Mode C fit results:**
  - Correction family: Pre
  - Parameters fitted: 1 (gain) via grid search over [0.5, 2.0]
  - NLL before correction: 78186639.1
  - NLL after correction: 1024.1
  - NLL decrease: 78185615.0 (100.0%)
- **Mode I recon using corrected operator A':**

  | Metric | A₀ (uncorrected) | A' (corrected) | Delta |
  |--------|-------------------|----------------|-------|
  | PSNR   | -52.82 dB | 10.64 dB | +63.46 dB |
  | SSIM   | 0.0100 | 0.6227 | +0.6127 |

## Test results summary

### Quick gate (pass/fail)

- [x] W1 simulate: PASS
- [x] W1 reconstruct (PSNR >= 5 dB): PASS (10.64 dB)
- [x] W2 operator correction (NLL decreases): PASS (100.0% decrease)
- [x] W2 corrected recon (beats uncorrected): PASS (+63.46 dB)
- [x] Report contract (flowchart + all headings): PASS
- [x] Node-by-node trace saved (>= 3 stages): PASS (5 stages)
- [x] RunBundle saved (with trace PNGs): PASS
- [x] Scoreboard updated: PASS

### Full metrics

| Check | Metric | Value | Threshold | Status |
|-------|--------|-------|-----------|--------|
| W1 PSNR | psnr | 10.64 dB | >= 5 dB | PASS |
| W1 SSIM | ssim | 0.6227 | >= 0.20 | PASS |
| W1 NRMSE | nrmse | 0.3178 | — | info |
| W2 NLL decrease | nll_decrease_pct | 100.0% | >= 5% | PASS |
| W2 PSNR gain | psnr_delta | +63.46 dB | > 0 | PASS |
| W2 SSIM gain | ssim_delta | +0.6127 | > 0 | PASS |
| Trace stages | n_stages | 5 | >= 3 | PASS |
| Trace PNGs | n_pngs | 5 | >= 3 | PASS |
| W1 wall time | w1_seconds | 0.03 s | — | info |
| W2 wall time | w2_seconds | 0.05 s | — | info |

## Reproducibility

- Seed: 42
- NumPy RNG state hash: 5d6a058f2c5c262e
- PWM version: 199aab9
- Python version: 3.13.9
- Key package versions: numpy=2.3.5, scipy=1.16.3
- Platform: Linux x86_64
- Deterministic reproduction command:
  ```bash
  pwm_cli run --modality spect --seed 42 --mode simulate,invert,calibrate
  ```
- Output hash (y): see RunBundle
- Output hash (x_hat): see RunBundle

## Saved artifacts

- RunBundle: `runs/run_spect_exp_17579c0a/`
- Report: `pwm/reports/spect.md`
- Scoreboard: `pwm/reports/scoreboard.yaml`
- Node trace (.npy): `runs/run_spect_exp_17579c0a/artifacts/trace/*.npy`
- Node trace (.png): `runs/run_spect_exp_17579c0a/artifacts/trace/png/*.png`
- W2 operator metadata: `runs/run_spect_exp_17579c0a/artifacts/w2_operator_meta.json`
- Ground truth: `runs/run_spect_exp_17579c0a/artifacts/x_true.npy`
- Measurement: `runs/run_spect_exp_17579c0a/artifacts/y.npy`
- W1 reconstruction: `runs/run_spect_exp_17579c0a/artifacts/x_hat.npy`
- W2 reconstructions: `runs/run_spect_exp_17579c0a/artifacts/x_hat_w2_uncorrected.npy`, `x_hat_w2_corrected.npy`

## Next actions

- Implement OSEM iterative reconstruction with collimator-detector response modeling
- Add attenuation correction using co-registered CT map
- Test on XCAT phantom SPECT simulation benchmark
