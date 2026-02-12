# Digital Holographic Microscopy — PWM Report

| Field | Value |
|-------|-------|
| Modality ID | `holography` |
| Category | coherent |
| Dataset | DHM benchmark (synthetic proxy: complex transmission phantom) |
| Date | 2026-02-12 |
| PWM version | `a473f9a` |
| Author | integritynoble |

## Modality overview

- Modality key: `holography`
- Category: coherent
- Forward model: y = Sensor(Fresnel(x)) + noise, where Fresnel is Fresnel propagation (λ=0.633 µm, z=5 mm, dx=3.45 µm), Sensor has QE=0.9, and noise is additive Gaussian (σ=0.005)
- Default solver: adjoint backpropagation (linear)
- Pipeline linearity: linear

Digital holographic microscopy (DHM) records the interference pattern (hologram) between a reference wave and the object wave after free-space propagation. Since the measurement is linear (no magnitude-squared), direct numerical backpropagation via the adjoint Fresnel operator recovers the object field.

| Parameter | Value |
|-----------|-------|
| Image size (H x W) | 64 x 64 |
| Propagation model | Fresnel (single-FFT) |
| Wavelength | 0.633e-6 m |
| Propagation distance | 5.0e-3 m |
| Pixel size | 3.45e-6 m |
| Peak photons | 50000 |
| Read noise sigma | 0.005 |
| Quantum efficiency | 0.9 |
| Noise model | Additive Gaussian |

## Standard dataset

- Name: DHM benchmark (synthetic proxy: Gaussian blob phantom)
- Source: synthetic
- Size: 64x64 phantom
- Registered in `dataset_registry.yaml` as `dhm_benchmark`

## PWM pipeline flowchart (mandatory)

```
x (world)
  ↓
SourceNode: photon_source — scales input by illumination strength (1.0)
  ↓
Element 1 (subrole=transport): fresnel_prop — Fresnel propagation (λ=0.633e-6, z=5.0e-3, dx=3.45e-6)
  ↓
SensorNode: photon_sensor — QE=0.9, gain=1.0, converts photon signal to electrons
  ↓
NoiseNode: poisson_gaussian_sensor — additive Gaussian noise (sigma=0.005)
  ↓
y
```

## Element inventory & mismatch parameters

| # | node_id | primitive_id | subrole | learnable params | mismatch knob | bounds | prior |
|---|---------|-------------|---------|-----------------|---------------|--------|-------|
| 1 | source | photon_source | source | strength=1.0 | — | — | — |
| 2 | prop | fresnel_prop | transport | wavelength=0.633e-6, distance=5.0e-3, pixel_size=3.45e-6 | distance | [3.0e-3, 10.0e-3] | normal(5.0e-3, 7.5e-4) |
| 3 | sensor | photon_sensor | sensor | QE=0.9, gain=1.0 | qe_drift, gain | [0.8, 1.0], [0.9, 1.1] | normal |
| 4 | noise | poisson_gaussian_sensor | noise | read_sigma=0.005 | — | — | — |

## Node-by-node trace (one sample)

| stage | node_id | output shape | dtype | range [min, max] | artifact_path |
|-------|---------|-------------|-------|-------------------|---------------|
| 0 | input x | (64, 64) | float64 | [0.0500, 0.9392] | `artifacts/trace/00_input_x.npy` |
| 1 | source | (64, 64) | float64 | [0.0500, 0.9392] | `artifacts/trace/01_source.npy` |
| 2 | prop | (64, 64) | complex128 | [0.0000, 0.9500] | `artifacts/trace/02_prop.npy` |
| 3 | sensor | (64, 64) | float64 | [0.0000, 0.8550] | `artifacts/trace/03_sensor.npy` |
| 4 | noise (y) | (64, 64) | float64 | [0.0000, 0.8600] | `artifacts/trace/04_noise_y.npy` |

PNG visualizations saved at: `artifacts/trace/png/`

## Workflow W1: Prompt-driven simulation + reconstruction

- **Prompt used:** `"Simulate digital holographic microscopy and reconstruct using adjoint backpropagation"`
- **ExperimentSpec summary:**
  - modality: holography
  - mode: simulate -> invert
  - solver: adjoint backpropagation (linear)
  - photon_budget: 50000 peak photons
- **Mode S results (simulate y):**
  - y shape: (64, 64)
  - y range: [0.0000, 0.8600]
  - SNR: 33.3 dB
- **Mode I results (reconstruct x_hat):**
  - x_hat shape: (64, 64)
  - Solver: adjoint, linear
- **Dataset metrics:**

  | Metric | Value |
  |--------|-------|
  | PSNR   | 20.99 dB |
  | SSIM   | 0.8519 |
  | NRMSE  | 0.0967 |

## Workflow W2: Operator correction mode (measured y + operator A)

- **Operator definition:**
  - A_definition: callable
  - A_extraction_method: graph_stripped
  - Operator chain: PhotonSource → FresnelProp → PhotonSensor (noise stripped)
  - A_sha256: 56423a8b1f2e7d90
  - Linearity: linear
  - Notes (if linearized): N/A
- **Mismatch specification:**
  - Mismatch type: synthetic_injected
  - Parameters perturbed: Fresnel propagation distance (5.0e-3 → 7.0e-3, +40%)
  - Description: Sample-to-sensor distance miscalibration causes defocus
- **Mode C fit results:**
  - Correction family: Pre
  - Parameters fitted: 1 parameter (distance) via grid search over [3.0e-3, 10.0e-3]
  - Best fitted distance: 7.0e-3
  - NLL before correction: 37181.1
  - NLL after correction: 968.2
  - NLL decrease: 36212.9 (97.4%)
- **Mode I recon using corrected operator A':**

  | Metric | A₀ (uncorrected) | A' (corrected) | Delta |
  |--------|-------------------|----------------|-------|
  | PSNR   | 20.94 dB | 20.99 dB | +0.05 dB |

## Test results summary

### Quick gate (pass/fail)

- [x] W1 simulate: PASS
- [x] W1 reconstruct (PSNR >= 18 dB): PASS (20.99 dB)
- [x] W2 operator correction (NLL decreases): PASS (97.4% decrease)
- [x] W2 corrected recon (beats uncorrected): PASS (+0.05 dB)
- [x] Report contract (flowchart + all headings): PASS
- [x] Node-by-node trace saved (>= 3 stages): PASS (5 stages)
- [x] RunBundle saved (with trace PNGs): PASS
- [x] Scoreboard updated: PASS

### Full metrics

| Check | Metric | Value | Threshold | Status |
|-------|--------|-------|-----------|--------|
| W1 PSNR | psnr | 20.99 dB | >= 18 dB | PASS |
| W1 SSIM | ssim | 0.8519 | >= 0.50 | PASS |
| W1 NRMSE | nrmse | 0.0967 | — | info |
| W2 NLL decrease | nll_decrease_pct | 97.4% | >= 5% | PASS |
| W2 PSNR gain | psnr_delta | +0.05 dB | > 0 | PASS |
| Trace stages | n_stages | 5 | >= 3 | PASS |
| Trace PNGs | n_pngs | 5 | >= 3 | PASS |
| W1 wall time | w1_seconds | 0.03 s | — | info |
| W2 wall time | w2_seconds | 0.06 s | — | info |

## Reproducibility

- Seed: 42
- NumPy RNG state hash: 5d6a058f2c5c262e
- PWM version: a473f9a
- Python version: 3.13.9
- Key package versions: numpy=2.3.5, scipy=1.16.3
- Platform: Linux x86_64
- Deterministic reproduction command:
  ```bash
  pwm_cli run --modality holography --seed 42 --mode simulate,invert,calibrate
  ```
- Output hash (y): see RunBundle
- Output hash (x_hat): see RunBundle

## Saved artifacts

- RunBundle: `runs/run_holography_exp_56423a8b/`
- Report: `pwm/reports/holography.md`
- Scoreboard: `pwm/reports/scoreboard.yaml`
- W2 reconstructions: `runs/run_holography_exp_56423a8b/artifacts/x_hat_w2_uncorrected.npy`, `x_hat_w2_corrected.npy`

## Next actions

- Test on real holographic data with off-axis recording geometry
- Add twin-image removal algorithm
- Extend to multi-wavelength DHM for quantitative phase imaging
