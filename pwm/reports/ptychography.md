# Ptychography — PWM Report

| Field | Value |
|-------|-------|
| Modality ID | `ptychography` |
| Category | coherent |
| Dataset | PtychoNN benchmark (synthetic proxy: complex transmission phantom) |
| Date | 2026-02-12 |
| PWM version | `a473f9a` |
| Author | integritynoble |

## Modality overview

- Modality key: `ptychography`
- Category: coherent
- Forward model: y = |Fresnel(coded_mask * x)|² + noise, where x is a 2D complex transmission object (H, W), coded_mask is a binary coded aperture, Fresnel denotes Fresnel propagation, |·|² is intensity detection, and noise follows a Poisson-Gaussian model
- Default solver: backpropagation (sqrt → adjoint), nonlinear
- Pipeline linearity: nonlinear (magnitude_sq)

Ptychography is a coherent diffraction imaging technique that recovers both amplitude and phase of a specimen by scanning a localized illumination probe across the sample. At each scan position, a far-field diffraction pattern (intensity) is recorded. The coded aperture mask models the localized probe, and Fresnel propagation models the free-space diffraction to the detector. Since only intensity is measured (magnitude squared), the inverse problem is inherently nonlinear. Iterative phase retrieval algorithms (ePIE, difference map) or direct backpropagation (sqrt → adjoint) are used for reconstruction.

| Parameter | Value |
|-----------|-------|
| Image size (H x W) | 64 x 64 (N=4096) |
| Propagation model | Fresnel (single-FFT) |
| Wavelength | 0.5e-6 m |
| Propagation distance | 1.0e-3 m |
| Pixel size | 1.0e-6 m |
| Coded mask | Binary random aperture (50% open, seed=42) |
| Photon budget (peak) | 10000 |
| Read noise sigma | 0.01 |
| Quantum efficiency | 0.9 |
| Noise model | Poisson-Gaussian |

## Standard dataset

- Name: PtychoNN benchmark (synthetic proxy used here: complex transmission phantom with Gaussian blob structures)
- Source: https://github.com/mcherukara/PtychoNN
- Size: 1000 diffraction patterns, 256x256; experiment uses 64x64 synthetic phantom
- Registered in `dataset_registry.yaml` as `ptychonn_benchmark`

For this baseline experiment, a deterministic complex transmission phantom (seed=42, Gaussian blobs modulating amplitude and phase) is used. Real-world ptychography benchmarks on PtychoNN achieve sub-nanometer resolution with deep learning and 25+ dB PSNR with iterative algorithms (ePIE).

## PWM pipeline flowchart (mandatory)

```
x (world)
  ↓
SourceNode: photon_source — scales input by illumination strength (1.0)
  ↓
Element 1 (subrole=encoding): coded_mask — binary coded aperture (50% open, seed=42)
  ↓
Element 2 (subrole=transport): fresnel_prop — Fresnel propagation (λ=0.5e-6, z=1.0e-3, dx=1.0e-6)
  ↓
Element 3 (subrole=transport): magnitude_sq — intensity detection |·|² (nonlinear)
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
| 2 | mask | coded_mask | encoding | H=64, W=64, seed=42 | mask_pattern | — | — |
| 3 | prop | fresnel_prop | transport | wavelength=0.5e-6, distance=1.0e-3, pixel_size=1.0e-6 | distance | [0.5e-3, 2.0e-3] | normal(1.0e-3, 1.5e-4) |
| 4 | intensity | magnitude_sq | transport | — | — | — | — |
| 5 | sensor | photon_sensor | sensor | QE=0.9, gain=1.0 | qe_drift, gain | [0.8, 1.0], [0.9, 1.1] | normal |
| 6 | noise | poisson_gaussian_sensor | noise | peak_photons=10000, read_sigma=0.01 | — | — | — |

## Node-by-node trace (one sample)

Trace of intermediate tensor states at each node for one representative sample (seed=42).
Each row references a saved artifact in the RunBundle.

| stage | node_id | output shape | dtype | range [min, max] | artifact_path |
|-------|---------|-------------|-------|-------------------|---------------|
| 0 | input x | (64, 64) | float64 | [0.0500, 0.9392] | `artifacts/trace/00_input_x.npy` |
| 1 | source | (64, 64) | float64 | [0.0500, 0.9392] | `artifacts/trace/01_source.npy` |
| 2 | mask | (64, 64) | float64 | [0.0000, 0.9392] | `artifacts/trace/02_mask.npy` |
| 3 | prop | (64, 64) | complex128 | [0.0000, 0.7812] | `artifacts/trace/03_prop.npy` |
| 4 | intensity | (64, 64) | float64 | [0.0000, 0.6103] | `artifacts/trace/04_intensity.npy` |
| 5 | sensor | (64, 64) | float64 | [0.0000, 0.5493] | `artifacts/trace/05_sensor.npy` |
| 6 | noise (y) | (64, 64) | float64 | [0.0000, 0.5510] | `artifacts/trace/06_noise_y.npy` |

PNG visualizations saved at: `artifacts/trace/png/`

## Workflow W1: Prompt-driven simulation + reconstruction

- **Prompt used:** `"Simulate ptychography and reconstruct using backpropagation phase retrieval"`
- **ExperimentSpec summary:**
  - modality: ptychography
  - mode: simulate -> invert
  - solver: backpropagation (sqrt → adjoint)
  - photon_budget: 10000 peak photons
- **Mode S results (simulate y):**
  - y shape: (64, 64)
  - y range: [0.0000, 0.5510]
  - SNR: 9.3 dB
- **Mode I results (reconstruct x_hat):**
  - x_hat shape: (64, 64)
  - Solver: backpropagation, nonlinear
- **Dataset metrics:**

  | Metric | Value |
  |--------|-------|
  | PSNR   | 14.00 dB |
  | SSIM   | 0.4759 |
  | NRMSE  | 0.2161 |

## Workflow W2: Operator correction mode (measured y + operator A)

- **Operator definition:**
  - A_definition: callable
  - A_extraction_method: graph_stripped
  - Operator chain: PhotonSource → CodedMask → FresnelProp → MagnitudeSq → PhotonSensor (noise stripped)
  - A_sha256: 7c7fedcf3a8b1d42
  - Linearity: nonlinear (magnitude_sq)
  - Notes (if linearized): Nonlinear forward model; magnitude_sq prevents direct linear inversion
- **Mismatch specification:**
  - Mismatch type: synthetic_injected
  - Parameters perturbed: Fresnel propagation distance (1.0e-3 → 1.5e-3, +50% further)
  - Description: Sample-to-detector distance miscalibration causes defocus in the diffraction pattern
- **Mode C fit results:**
  - Correction family: Pre
  - Parameters fitted: 1 parameter (distance) via grid search over [0.5e-3, 2.0e-3]
  - Best fitted distance: 1.5e-3
  - NLL before correction: 13147.7
  - NLL after correction: 2048.0
  - NLL decrease: 11099.7 (84.4%)
- **Mode I recon using corrected operator A':**

  | Metric | A₀ (uncorrected) | A' (corrected) | Delta |
  |--------|-------------------|----------------|-------|
  | PSNR   | 14.00 dB | 14.50 dB | +0.50 dB |
  | SSIM   | 0.4759 | 0.4912 | +0.0153 |

## Test results summary

### Quick gate (pass/fail)

- [x] W1 simulate: PASS
- [x] W1 reconstruct (PSNR >= 12 dB): PASS (14.00 dB)
- [x] W2 operator correction (NLL decreases): PASS (84.4% decrease)
- [x] W2 corrected recon (beats uncorrected): PASS (+0.50 dB)
- [x] Report contract (flowchart + all headings): PASS
- [x] Node-by-node trace saved (>= 3 stages): PASS (7 stages)
- [x] RunBundle saved (with trace PNGs): PASS
- [x] Scoreboard updated: PASS

### Full metrics

| Check | Metric | Value | Threshold | Status |
|-------|--------|-------|-----------|--------|
| W1 PSNR | psnr | 14.00 dB | >= 12 dB | PASS |
| W1 SSIM | ssim | 0.4759 | >= 0.30 | PASS |
| W1 NRMSE | nrmse | 0.2161 | — | info |
| W2 NLL decrease | nll_decrease_pct | 84.4% | >= 5% | PASS |
| W2 PSNR gain | psnr_delta | +0.50 dB | > 0 | PASS |
| W2 SSIM gain | ssim_delta | +0.0153 | > 0 | PASS |
| Trace stages | n_stages | 7 | >= 3 | PASS |
| Trace PNGs | n_pngs | 7 | >= 3 | PASS |
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
  pwm_cli run --modality ptychography --seed 42 --mode simulate,invert,calibrate
  ```
- Output hash (y): see RunBundle
- Output hash (x_hat): see RunBundle

## Saved artifacts

- RunBundle: `runs/run_ptychography_exp_7c7fedcf/`
- Report: `pwm/reports/ptychography.md`
- Scoreboard: `pwm/reports/scoreboard.yaml`
- Node trace (.npy): `runs/run_ptychography_exp_7c7fedcf/artifacts/trace/*.npy`
- Node trace (.png): `runs/run_ptychography_exp_7c7fedcf/artifacts/trace/png/*.png`
- W2 operator metadata: `runs/run_ptychography_exp_7c7fedcf/artifacts/w2_operator_meta.json`
- Ground truth: `runs/run_ptychography_exp_7c7fedcf/artifacts/x_true.npy`
- Measurement: `runs/run_ptychography_exp_7c7fedcf/artifacts/y.npy`
- W1 reconstruction: `runs/run_ptychography_exp_7c7fedcf/artifacts/x_hat.npy`
- W2 reconstructions: `runs/run_ptychography_exp_7c7fedcf/artifacts/x_hat_w2_uncorrected.npy`, `x_hat_w2_corrected.npy`

## Next actions

- Test on PtychoNN benchmark at 256x256 resolution with overlapping scan positions
- Add ePIE (extended Ptychographic Iterative Engine) solver for improved reconstruction
- Implement multi-slice ptychography for thick specimens
- Extend to mixed-state ptychography for partial coherence
