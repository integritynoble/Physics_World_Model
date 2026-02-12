# Optical Coherence Tomography (OCT) — PWM Report

| Field | Value |
|-------|-------|
| Modality ID | `oct` |
| Category | coherent |
| Dataset | OCT benchmark (synthetic proxy: tissue phantom) |
| Date | 2026-02-12 |
| PWM version | `a473f9a` |
| Author | integritynoble |

## Modality overview

- Modality key: `oct`
- Category: coherent
- Forward model: y = Sensor(Angular_Spectrum(x)) + noise, where Angular_Spectrum models low-coherence interferometric propagation (λ=1.3 µm, z=1 mm, dx=5 µm), Sensor has QE=0.9, and noise is additive Gaussian (σ=0.005)
- Default solver: adjoint backpropagation (linear)
- Pipeline linearity: linear

Optical coherence tomography (OCT) uses low-coherence interferometry to produce cross-sectional images of tissue microstructure. The measurement is linear since OCT uses heterodyne detection that preserves the complex field. Direct adjoint backpropagation recovers the object reflectivity profile.

| Parameter | Value |
|-----------|-------|
| Image size (H x W) | 64 x 64 |
| Propagation model | Angular spectrum (double-FFT) |
| Wavelength | 1.3e-6 m |
| Propagation distance | 1.0e-3 m |
| Pixel size | 5.0e-6 m |
| Peak photons | 50000 |
| Read noise sigma | 0.005 |
| Quantum efficiency | 0.9 |
| Noise model | Additive Gaussian |

## Standard dataset

- Name: OCT benchmark (synthetic proxy: Gaussian blob phantom)
- Source: synthetic
- Size: 64x64 phantom
- Registered in `dataset_registry.yaml` as `oct_benchmark`

## PWM pipeline flowchart (mandatory)

```
x (world)
  ↓
SourceNode: photon_source — scales input by illumination strength (1.0)
  ↓
Element 1 (subrole=transport): angular_spectrum — Angular spectrum propagation (λ=1.3e-6, z=1.0e-3, dx=5.0e-6)
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
| 2 | prop | angular_spectrum | transport | wavelength=1.3e-6, distance=1.0e-3, pixel_size=5.0e-6 | distance | [0.5e-3, 2.5e-3] | normal(1.0e-3, 2.0e-4) |
| 3 | sensor | photon_sensor | sensor | QE=0.9, gain=1.0 | qe_drift, gain | [0.8, 1.0], [0.9, 1.1] | normal |
| 4 | noise | poisson_gaussian_sensor | noise | read_sigma=0.005 | — | — | — |

## Node-by-node trace (one sample)

| stage | node_id | output shape | dtype | range [min, max] | artifact_path |
|-------|---------|-------------|-------|-------------------|---------------|
| 0 | input x | (64, 64) | float64 | [0.0500, 0.9392] | `artifacts/trace/00_input_x.npy` |
| 1 | source | (64, 64) | float64 | [0.0500, 0.9392] | `artifacts/trace/01_source.npy` |
| 2 | prop | (64, 64) | complex128 | [0.0000, 0.9400] | `artifacts/trace/02_prop.npy` |
| 3 | sensor | (64, 64) | float64 | [0.0000, 0.8460] | `artifacts/trace/03_sensor.npy` |
| 4 | noise (y) | (64, 64) | float64 | [0.0000, 0.8500] | `artifacts/trace/04_noise_y.npy` |

PNG visualizations saved at: `artifacts/trace/png/`

## Workflow W1: Prompt-driven simulation + reconstruction

- **Prompt used:** `"Simulate OCT measurement and reconstruct using adjoint backpropagation"`
- **ExperimentSpec summary:**
  - modality: oct
  - mode: simulate -> invert
  - solver: adjoint backpropagation (linear)
  - photon_budget: 50000 peak photons
- **Mode S results (simulate y):**
  - y shape: (64, 64)
  - y range: [0.0000, 0.8500]
  - SNR: 33.3 dB
- **Mode I results (reconstruct x_hat):**
  - x_hat shape: (64, 64)
  - Solver: adjoint, linear
- **Dataset metrics:**

  | Metric | Value |
  |--------|-------|
  | PSNR   | 11.81 dB |
  | SSIM   | 0.2962 |
  | NRMSE  | 0.2778 |

## Workflow W2: Operator correction mode (measured y + operator A)

- **Operator definition:**
  - A_definition: callable
  - A_extraction_method: graph_stripped
  - Operator chain: PhotonSource → AngularSpectrum → PhotonSensor (noise stripped)
  - A_sha256: d64324bd5e9a2c18
  - Linearity: linear
  - Notes (if linearized): N/A
- **Mismatch specification:**
  - Mismatch type: synthetic_injected
  - Parameters perturbed: Angular spectrum propagation distance (1.0e-3 → 1.5e-3, +50%)
  - Description: Sample-to-detector distance miscalibration causes defocus
- **Mode C fit results:**
  - Correction family: Pre
  - Parameters fitted: 1 parameter (distance) via grid search over [0.5e-3, 2.5e-3]
  - Best fitted distance: 1.5e-3
  - NLL before correction: 2067619.9
  - NLL after correction: 7994.5
  - NLL decrease: 2059625.4 (99.6%)
- **Mode I recon using corrected operator A':**

  | Metric | A₀ (uncorrected) | A' (corrected) | Delta |
  |--------|-------------------|----------------|-------|
  | PSNR   | 4.13 dB | 11.81 dB | +7.68 dB |

## Test results summary

### Quick gate (pass/fail)

- [x] W1 simulate: PASS
- [x] W1 reconstruct (PSNR >= 10 dB): PASS (11.81 dB)
- [x] W2 operator correction (NLL decreases): PASS (99.6% decrease)
- [x] W2 corrected recon (beats uncorrected): PASS (+7.68 dB)
- [x] Report contract (flowchart + all headings): PASS
- [x] Node-by-node trace saved (>= 3 stages): PASS (5 stages)
- [x] RunBundle saved (with trace PNGs): PASS
- [x] Scoreboard updated: PASS

### Full metrics

| Check | Metric | Value | Threshold | Status |
|-------|--------|-------|-----------|--------|
| W1 PSNR | psnr | 11.81 dB | >= 10 dB | PASS |
| W1 SSIM | ssim | 0.2962 | >= 0.20 | PASS |
| W1 NRMSE | nrmse | 0.2778 | — | info |
| W2 NLL decrease | nll_decrease_pct | 99.6% | >= 5% | PASS |
| W2 PSNR gain | psnr_delta | +7.68 dB | > 0 | PASS |
| Trace stages | n_stages | 5 | >= 3 | PASS |
| Trace PNGs | n_pngs | 5 | >= 3 | PASS |
| W1 wall time | w1_seconds | 0.03 s | — | info |
| W2 wall time | w2_seconds | 0.08 s | — | info |

## Reproducibility

- Seed: 42
- NumPy RNG state hash: 5d6a058f2c5c262e
- PWM version: a473f9a
- Python version: 3.13.9
- Key package versions: numpy=2.3.5, scipy=1.16.3
- Platform: Linux x86_64
- Deterministic reproduction command:
  ```bash
  pwm_cli run --modality oct --seed 42 --mode simulate,invert,calibrate
  ```
- Output hash (y): see RunBundle
- Output hash (x_hat): see RunBundle

## Saved artifacts

- RunBundle: `runs/run_oct_exp_d64324bd/`
- Report: `pwm/reports/oct.md`
- Scoreboard: `pwm/reports/scoreboard.yaml`

## Next actions

- Implement spectral-domain OCT with FFT-based depth reconstruction
- Add dispersion compensation for improved axial resolution
- Test with multi-layer tissue phantom for depth-resolved imaging
