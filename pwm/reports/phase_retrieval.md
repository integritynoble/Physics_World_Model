# Phase Retrieval (CDI) — PWM Report

| Field | Value |
|-------|-------|
| Modality ID | `phase_retrieval` |
| Category | coherent |
| Dataset | CDI benchmark (synthetic proxy: complex transmission phantom) |
| Date | 2026-02-12 |
| PWM version | `a473f9a` |
| Author | integritynoble |

## Modality overview

- Modality key: `phase_retrieval`
- Category: coherent
- Forward model: y = |Angular_Spectrum(x)|² + noise, where Angular_Spectrum is angular spectrum propagation (λ=0.5 µm, z=1 mm, dx=1 µm), |·|² is intensity detection (magnitude_sq), and noise follows a Poisson-Gaussian model
- Default solver: backpropagation (sqrt → adjoint), nonlinear
- Pipeline linearity: nonlinear (magnitude_sq)

Coherent diffraction imaging (CDI) / phase retrieval records the far-field intensity pattern of a coherently illuminated specimen. Since only intensity (magnitude squared) is measured, the phase information is lost and must be recovered iteratively.

| Parameter | Value |
|-----------|-------|
| Image size (H x W) | 64 x 64 |
| Propagation model | Angular spectrum (double-FFT) |
| Wavelength | 0.5e-6 m |
| Propagation distance | 1.0e-3 m |
| Pixel size | 1.0e-6 m |
| Peak photons | 10000 |
| Read noise sigma | 0.01 |
| Quantum efficiency | 0.9 |
| Noise model | Poisson-Gaussian |

## Standard dataset

- Name: CDI benchmark (synthetic proxy: Gaussian blob phantom)
- Source: synthetic
- Size: 64x64 phantom
- Registered in `dataset_registry.yaml` as `cdi_benchmark`

## PWM pipeline flowchart (mandatory)

```
x (world)
  ↓
SourceNode: photon_source — scales input by illumination strength (1.0)
  ↓
Element 1 (subrole=transport): angular_spectrum — Angular spectrum propagation (λ=0.5e-6, z=1.0e-3, dx=1.0e-6)
  ↓
Element 2 (subrole=transport): magnitude_sq — intensity detection |·|² (nonlinear)
  ↓
SensorNode: photon_sensor — QE=0.9, gain=1.0, converts photon signal to electrons
  ↓
NoiseNode: poisson_gaussian_sensor — Poisson shot noise (peak=10000) + Gaussian read noise (sigma=0.01)
  ↓
y
```

## Element inventory & mismatch parameters

| # | node_id | primitive_id | subrole | learnable params | mismatch knob | bounds | prior |
|---|---------|-------------|---------|-----------------|---------------|--------|-------|
| 1 | source | photon_source | source | strength=1.0 | — | — | — |
| 2 | prop | angular_spectrum | transport | wavelength=0.5e-6, distance=1.0e-3, pixel_size=1.0e-6 | distance | [0.5e-3, 2.0e-3] | normal |
| 3 | intensity | magnitude_sq | transport | — | — | — | — |
| 4 | sensor | photon_sensor | sensor | QE=0.9, gain=1.0 | gain | [0.5, 2.0] | normal(1.0, 0.1) |
| 5 | noise | poisson_gaussian_sensor | noise | peak_photons=10000, read_sigma=0.01 | — | — | — |

## Node-by-node trace (one sample)

| stage | node_id | output shape | dtype | range [min, max] | artifact_path |
|-------|---------|-------------|-------|-------------------|---------------|
| 0 | input x | (64, 64) | float64 | [0.0500, 0.9392] | `artifacts/trace/00_input_x.npy` |
| 1 | source | (64, 64) | float64 | [0.0500, 0.9392] | `artifacts/trace/01_source.npy` |
| 2 | prop | (64, 64) | complex128 | [0.0000, 0.9400] | `artifacts/trace/02_prop.npy` |
| 3 | intensity | (64, 64) | float64 | [0.0000, 0.8836] | `artifacts/trace/03_intensity.npy` |
| 4 | sensor | (64, 64) | float64 | [0.0000, 0.7952] | `artifacts/trace/04_sensor.npy` |
| 5 | noise (y) | (64, 64) | float64 | [0.0000, 0.7980] | `artifacts/trace/05_noise_y.npy` |

PNG visualizations saved at: `artifacts/trace/png/`

## Workflow W1: Prompt-driven simulation + reconstruction

- **Prompt used:** `"Simulate coherent diffraction imaging and reconstruct using backpropagation phase retrieval"`
- **ExperimentSpec summary:**
  - modality: phase_retrieval
  - mode: simulate -> invert
  - solver: backpropagation (sqrt → adjoint)
  - photon_budget: 10000 peak photons
- **Mode S results (simulate y):**
  - y shape: (64, 64)
  - y range: [0.0000, 0.7980]
  - SNR: 9.3 dB
- **Mode I results (reconstruct x_hat):**
  - x_hat shape: (64, 64)
  - Solver: backpropagation, nonlinear
- **Dataset metrics:**

  | Metric | Value |
  |--------|-------|
  | PSNR   | 16.17 dB |
  | SSIM   | 0.5812 |
  | NRMSE  | 0.1676 |

## Workflow W2: Operator correction mode (measured y + operator A)

- **Operator definition:**
  - A_definition: callable
  - A_extraction_method: graph_stripped
  - Operator chain: PhotonSource → AngularSpectrum → MagnitudeSq → PhotonSensor (noise stripped)
  - A_sha256: 78129139a4c6e5d2
  - Linearity: nonlinear (magnitude_sq)
  - Notes (if linearized): Nonlinear forward model; magnitude_sq prevents direct linear inversion
- **Mismatch specification:**
  - Mismatch type: synthetic_injected
  - Parameters perturbed: Sensor gain (1.0 → 1.3, +30%)
  - Description: Sensor gain miscalibration changes overall intensity scale
- **Mode C fit results:**
  - Correction family: Pre
  - Parameters fitted: 1 parameter (gain) via grid search over [0.5, 2.0]
  - Best fitted gain: 1.3
  - NLL before correction: 11367.1
  - NLL after correction: 2504.2
  - NLL decrease: 8862.9 (77.9%)
- **Mode I recon using corrected operator A':**

  | Metric | A₀ (uncorrected) | A' (corrected) | Delta |
  |--------|-------------------|----------------|-------|
  | PSNR   | 14.92 dB | 16.17 dB | +1.25 dB |

## Test results summary

### Quick gate (pass/fail)

- [x] W1 simulate: PASS
- [x] W1 reconstruct (PSNR >= 14 dB): PASS (16.17 dB)
- [x] W2 operator correction (NLL decreases): PASS (77.9% decrease)
- [x] W2 corrected recon (beats uncorrected): PASS (+1.25 dB)
- [x] Report contract (flowchart + all headings): PASS
- [x] Node-by-node trace saved (>= 3 stages): PASS (6 stages)
- [x] RunBundle saved (with trace PNGs): PASS
- [x] Scoreboard updated: PASS

### Full metrics

| Check | Metric | Value | Threshold | Status |
|-------|--------|-------|-----------|--------|
| W1 PSNR | psnr | 16.17 dB | >= 14 dB | PASS |
| W1 SSIM | ssim | 0.5812 | >= 0.30 | PASS |
| W1 NRMSE | nrmse | 0.1676 | — | info |
| W2 NLL decrease | nll_decrease_pct | 77.9% | >= 5% | PASS |
| W2 PSNR gain | psnr_delta | +1.25 dB | > 0 | PASS |
| Trace stages | n_stages | 6 | >= 3 | PASS |
| Trace PNGs | n_pngs | 6 | >= 3 | PASS |
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
  pwm_cli run --modality phase_retrieval --seed 42 --mode simulate,invert,calibrate
  ```
- Output hash (y): see RunBundle
- Output hash (x_hat): see RunBundle

## Saved artifacts

- RunBundle: `runs/run_phase_retrieval_exp_78129139/`
- Report: `pwm/reports/phase_retrieval.md`
- Scoreboard: `pwm/reports/scoreboard.yaml`

## Next actions

- Implement HIO and ER solvers for improved convergence
- Test with oversampling constraints
- Add support for ptychographic-CDI hybrid reconstruction
