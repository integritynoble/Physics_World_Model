# Positron Emission Tomography (PET) — PWM Report

| Field | Value |
|-------|-------|
| Modality ID | `pet` |
| Category | nuclear |
| Dataset | PET benchmark (synthetic proxy: emission activity phantom) |
| Date | 2026-02-12 |
| PWM version | `199aab9` |
| Author | integritynoble |

## Modality overview

- Modality key: `pet`
- Category: nuclear
- Forward model: y = Noise(Sensor(Scatter(EmissionProjection(Source(x))))), where EmissionProjection computes sinogram projections at 32 angles, ScatterModel adds Compton scatter (frac=0.15, sigma=3), PhotonSensor detects with QE=0.85, and noise is Poisson (peak=100000)
- Default solver: adjoint backprojection
- Pipeline linearity: linear

Positron emission tomography (PET) detects coincident 511 keV annihilation photons from positron-emitting radiotracers. The emission projection operator computes line-integral projections of the radiotracer activity distribution at multiple detector angles. Compton scatter adds a smooth background. Reconstruction uses adjoint backprojection or iterative MLEM.

| Parameter | Value |
|-----------|-------|
| Image size (H x W) | 64 x 64 |
| Projection angles | 32 |
| Measurements (M) | 2048 (32 x 64) |
| Scatter fraction | 0.15 |
| Scatter kernel sigma | 3.0 pixels |
| Quantum efficiency | 0.85 |
| Peak photons | 100000 |
| Noise model | Poisson only |

## Standard dataset

- Name: PET benchmark (synthetic proxy: emission activity phantom)
- Source: synthetic
- Size: 64x64 phantom
- Registered in `dataset_registry.yaml` as `pet_benchmark`

## PWM pipeline flowchart (mandatory)

```
x (world)
  ↓
SourceNode: generic_source — emission activity scaling (1.0)
  ↓
Element 1 (subrole=transport): emission_projection — sinogram projection (n_angles=32)
  ↓
Element 2 (subrole=transport): scatter_model — Compton scatter (frac=0.15, sigma=3.0)
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
| 3 | scatter | scatter_model | transport | scatter_fraction=0.15, kernel_sigma=3.0 | scatter_fraction | [0.0, 0.3] | uniform |
| 4 | sensor | photon_sensor | sensor | QE=0.85, gain=1.0 | gain | [0.5, 1.5] | normal |
| 5 | noise | poisson_only_sensor | noise | peak=100000 | — | — | — |

## Node-by-node trace (one sample)

| stage | node_id | output shape | dtype | range [min, max] | artifact_path |
|-------|---------|-------------|-------|-------------------|---------------|
| 0 | input x | (64, 64) | float64 | [0.0000, 0.5000] | `artifacts/trace/00_input_x.npy` |
| 1 | source | (64, 64) | float64 | [0.0000, 0.5000] | `artifacts/trace/01_source.npy` |
| 2 | projection | (32, 64) | float64 | [0.0000, 15.6000] | `artifacts/trace/02_projection.npy` |
| 3 | scatter | (32, 64) | float64 | [0.1000, 15.8000] | `artifacts/trace/03_scatter.npy` |
| 4 | sensor | (32, 64) | float64 | [0.0850, 13.4300] | `artifacts/trace/04_sensor.npy` |
| 5 | noise (y) | (32, 64) | float64 | [0.0000, 14.0000] | `artifacts/trace/05_noise_y.npy` |

PNG visualizations saved at: `artifacts/trace/png/`

## Workflow W1: Prompt-driven simulation + reconstruction

- **Prompt used:** `"Simulate PET emission scan and reconstruct using adjoint backprojection"`
- **ExperimentSpec summary:**
  - modality: pet
  - mode: simulate -> invert
  - solver: adjoint backprojection
  - photon_budget: peak=100000
- **Mode S results (simulate y):**
  - y shape: (32, 64)
  - y range: [0.0000, 14.0000]
  - SNR: 60.8 dB
- **Mode I results (reconstruct x_hat):**
  - x_hat shape: (64, 64)
  - Solver: adjoint backprojection
- **Dataset metrics:**

  | Metric | Value |
  |--------|-------|
  | PSNR   | 10.53 dB |
  | SSIM   | 0.6160 |
  | NRMSE  | 0.3219 |

Note: The moderate PSNR reflects the limited angular sampling (32 angles) and simple adjoint reconstruction. OSEM with 32+ subsets achieves 25+ dB on clinical PET data.

## Workflow W2: Operator correction mode (measured y + operator A)

- **Operator definition:**
  - A_definition: callable
  - A_extraction_method: graph_stripped
  - Operator chain: GenericSource → EmissionProjection → ScatterModel → PhotonSensor (noise stripped)
  - A_sha256: 8d3e857b4a1c62f9
  - Linearity: linear
  - Notes (if linearized): N/A
- **Mismatch specification:**
  - Mismatch type: synthetic_injected
  - Parameters perturbed: sensor gain (1.0 → 1.3, +30%)
  - Description: Detector gain drift from scintillator crystal degradation or PMT aging
- **Mode C fit results:**
  - Correction family: Pre
  - Parameters fitted: 1 (gain) via grid search over [0.5, 2.0]
  - NLL before correction: 89448782.7
  - NLL after correction: 1024.1
  - NLL decrease: 89447758.6 (100.0%)
- **Mode I recon using corrected operator A':**

  | Metric | A₀ (uncorrected) | A' (corrected) | Delta |
  |--------|-------------------|----------------|-------|
  | PSNR   | -55.23 dB | 10.53 dB | +65.76 dB |
  | SSIM   | 0.0100 | 0.6160 | +0.6060 |

## Test results summary

### Quick gate (pass/fail)

- [x] W1 simulate: PASS
- [x] W1 reconstruct (PSNR >= 5 dB): PASS (10.53 dB)
- [x] W2 operator correction (NLL decreases): PASS (100.0% decrease)
- [x] W2 corrected recon (beats uncorrected): PASS (+65.76 dB)
- [x] Report contract (flowchart + all headings): PASS
- [x] Node-by-node trace saved (>= 3 stages): PASS (6 stages)
- [x] RunBundle saved (with trace PNGs): PASS
- [x] Scoreboard updated: PASS

### Full metrics

| Check | Metric | Value | Threshold | Status |
|-------|--------|-------|-----------|--------|
| W1 PSNR | psnr | 10.53 dB | >= 5 dB | PASS |
| W1 SSIM | ssim | 0.6160 | >= 0.20 | PASS |
| W1 NRMSE | nrmse | 0.3219 | — | info |
| W2 NLL decrease | nll_decrease_pct | 100.0% | >= 5% | PASS |
| W2 PSNR gain | psnr_delta | +65.76 dB | > 0 | PASS |
| W2 SSIM gain | ssim_delta | +0.6060 | > 0 | PASS |
| Trace stages | n_stages | 6 | >= 3 | PASS |
| Trace PNGs | n_pngs | 6 | >= 3 | PASS |
| W1 wall time | w1_seconds | 0.04 s | — | info |
| W2 wall time | w2_seconds | 0.06 s | — | info |

## Reproducibility

- Seed: 42
- NumPy RNG state hash: 5d6a058f2c5c262e
- PWM version: 199aab9
- Python version: 3.13.9
- Key package versions: numpy=2.3.5, scipy=1.16.3
- Platform: Linux x86_64
- Deterministic reproduction command:
  ```bash
  pwm_cli run --modality pet --seed 42 --mode simulate,invert,calibrate
  ```
- Output hash (y): see RunBundle
- Output hash (x_hat): see RunBundle

## Saved artifacts

- RunBundle: `runs/run_pet_exp_8d3e857b/`
- Report: `pwm/reports/pet.md`
- Scoreboard: `pwm/reports/scoreboard.yaml`
- Node trace (.npy): `runs/run_pet_exp_8d3e857b/artifacts/trace/*.npy`
- Node trace (.png): `runs/run_pet_exp_8d3e857b/artifacts/trace/png/*.png`
- W2 operator metadata: `runs/run_pet_exp_8d3e857b/artifacts/w2_operator_meta.json`
- Ground truth: `runs/run_pet_exp_8d3e857b/artifacts/x_true.npy`
- Measurement: `runs/run_pet_exp_8d3e857b/artifacts/y.npy`
- W1 reconstruction: `runs/run_pet_exp_8d3e857b/artifacts/x_hat.npy`
- W2 reconstructions: `runs/run_pet_exp_8d3e857b/artifacts/x_hat_w2_uncorrected.npy`, `x_hat_w2_corrected.npy`

## Next actions

- Implement OSEM iterative reconstruction for improved PET image quality
- Add attenuation correction using co-registered CT map
- Test on Brainweb PET simulation benchmark
