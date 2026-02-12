# Magnetic Resonance Imaging (MRI) — PWM Report

| Field | Value |
|-------|-------|
| Modality ID | `mri` |
| Category | medical |
| Dataset | fastMRI (synthetic proxy: brain-like phantom) |
| Date | 2026-02-12 |
| PWM version | `1dd6f76` |
| Author | integritynoble |

## Modality overview

- Modality key: `mri`
- Category: medical
- Forward model: y = Mask * FFT(S * x) + n, where x is a 2D image (H, W), FFT is the 2D Fourier transform, Mask is a k-space undersampling pattern (25% rate with ACS center), S is coil sensitivity (scalar), and n ~ CN(0, sigma^2 I) complex Gaussian
- Default solver: cs_mri_wavelet (Compressed Sensing MRI with wavelet sparsity and FISTA)
- Pipeline linearity: linear

MRI acquires data in the spatial frequency domain (k-space) using gradient magnetic fields. Accelerated MRI undersamples k-space using a variable-density random mask that preserves the central auto-calibration region. Reconstruction recovers the image from undersampled complex k-space data using CS-MRI with wavelet sparsity regularization.

| Parameter | Value |
|-----------|-------|
| Image size (H x W) | 64 x 64 (N=4096) |
| Sampling rate | 25% (with ACS center fully sampled) |
| Effective mask density | 30.4% |
| Acceleration factor | ~3.3x |
| Data type | Complex (k-space) |
| Noise model | Complex Gaussian, sigma=0.005 |

## Standard dataset

- Name: fastMRI Knee (synthetic proxy used here: brain-like phantom)
- Source: https://fastmri.med.nyu.edu/
- Size: 1,594 volumes, 320x320; experiment uses 64x64 synthetic phantom
- Registered in `dataset_registry.yaml` as `fastmri_knee`

For this baseline experiment, a deterministic brain-like phantom (seed=42, skull ellipse with 6 internal structures simulating white/gray matter, CSF, and lesions) is used. Real-world MRI benchmarks on fastMRI achieve 35+ dB PSNR with deep learning solvers (VarNet, MoDL).

## PWM pipeline flowchart (mandatory)

```
x (world)
  ↓
SourceNode: spin_source — RF excitation scaling (strength=1.0)
  ↓
Element 1 (subrole=encoding): mri_kspace — 2D FFT + random undersampling mask (25% rate, ACS center)
  ↓
SensorNode: coil_sensor — coil sensitivity weighting (sensitivity=1.0)
  ↓
NoiseNode: complex_gaussian_sensor — complex Gaussian noise (sigma=0.005)
  ↓
y
```

## Element inventory & mismatch parameters

| # | node_id | primitive_id | subrole | learnable params | mismatch knob | bounds | prior |
|---|---------|-------------|---------|-----------------|---------------|--------|-------|
| 1 | source | spin_source | source | strength=1.0 | — | — | — |
| 2 | kspace | mri_kspace | encoding | H=64, W=64, sampling_rate=0.25, seed=42 | k_space_trajectory_error, mask_seed | [0, 0.05], [0, 200] | normal |
| 3 | sensor | coil_sensor | sensor | sensitivity=1.0 | coil_sensitivity_error | [0, 0.5] | normal(0, 0.1) |
| 4 | noise | complex_gaussian_sensor | noise | sigma=0.005 | — | — | — |

## Node-by-node trace (one sample)

Trace of intermediate tensor states at each node for one representative sample (seed=42).
Each row references a saved artifact in the RunBundle.

| stage | node_id | output shape | dtype | range [min, max] | artifact_path |
|-------|---------|-------------|-------|-------------------|---------------|
| 0 | input x | (64, 64) | float64 | [0.0000, 0.9500] | `artifacts/trace/00_input_x.npy` |
| 1 | source | (64, 64) | float64 | [0.0000, 0.9500] | `artifacts/trace/01_source.npy` |
| 2 | kspace | (64, 64) | complex128 | [0.0000, 1302.1500] | `artifacts/trace/02_kspace.npy` |
| 3 | sensor | (64, 64) | complex128 | [0.0000, 1302.1500] | `artifacts/trace/03_sensor.npy` |
| 4 | noise (y) | (64, 64) | complex128 | [0.0001, 1302.1500] | `artifacts/trace/04_noise_y.npy` |

PNG visualizations saved at: `artifacts/trace/png/`

## Workflow W1: Prompt-driven simulation + reconstruction

- **Prompt used:** `"Simulate accelerated MRI scan at 4x acceleration and reconstruct using CS-MRI"`
- **ExperimentSpec summary:**
  - modality: mri
  - mode: simulate -> invert
  - solver: cs_mri_wavelet
  - photon_budget: N/A (thermal noise model)
- **Mode S results (simulate y):**
  - y shape: (64, 64) complex128
  - |y| range: [0.0001, 1302.1500]
  - SNR: 71.9 dB
- **Mode I results (reconstruct x_hat):**
  - x_hat shape: (64, 64)
  - Solver: cs_mri_wavelet (FISTA with wavelet L1 sparsity), iterations: 50, lambda: 0.005
- **Dataset metrics:**

  | Metric | Value |
  |--------|-------|
  | PSNR   | 29.37 dB |
  | SSIM   | 0.9944 |
  | NRMSE  | 0.0340 |

## Workflow W2: Operator correction mode (measured y + operator A)

- **Operator definition:**
  - A_definition: callable
  - A_extraction_method: graph_stripped
  - Operator chain: SpinSource → MRIKspace → CoilSensor (noise stripped)
  - A_sha256: 53daf0cd161ccb1b
  - Linearity: linear
  - Notes (if linearized): N/A (linear Fourier + masking)
- **Mismatch specification:**
  - Mismatch type: synthetic_injected
  - Parameters perturbed: k-space undersampling mask seed (42 → 99, entirely different mask pattern)
  - Description: Gradient trajectory error — the k-space sampling mask changes from seed 42 to 99, simulating a different undersampling pattern due to gradient timing errors
- **Mode C fit results:**
  - Correction family: Pre
  - Parameters fitted: 1 parameter (mask seed) via grid search over seeds [30..120]
  - Best fitted seed: 99
  - NLL before correction: 943026991.6
  - NLL after correction: 8235.0
  - NLL decrease: 943018756.6 (100.0%)
- **Mode I recon using corrected operator A':**

  | Metric | A₀ (uncorrected) | A' (corrected) | Delta |
  |--------|-------------------|----------------|-------|
  | PSNR   | 25.02 dB | 29.23 dB | +4.22 dB |
  | SSIM   | 0.9847 | 0.9942 | +0.0095 |

## Test results summary

### Quick gate (pass/fail)

- [x] W1 simulate: PASS
- [x] W1 reconstruct (PSNR >= 18 dB): PASS (29.37 dB)
- [x] W2 operator correction (NLL decreases): PASS (100.0% decrease)
- [x] W2 corrected recon (beats uncorrected): PASS (+4.22 dB)
- [x] Report contract (flowchart + all headings): PASS
- [x] Node-by-node trace saved (>= 3 stages): PASS (5 stages)
- [x] RunBundle saved (with trace PNGs): PASS
- [x] Scoreboard updated: PASS

### Full metrics

| Check | Metric | Value | Threshold | Status |
|-------|--------|-------|-----------|--------|
| W1 PSNR | psnr | 29.37 dB | >= 18 dB | PASS |
| W1 SSIM | ssim | 0.9944 | >= 0.80 | PASS |
| W1 NRMSE | nrmse | 0.0340 | — | info |
| W2 NLL decrease | nll_decrease_pct | 100.0% | >= 5% | PASS |
| W2 PSNR gain | psnr_delta | +4.22 dB | > 0 | PASS |
| W2 SSIM gain | ssim_delta | +0.0095 | > 0 | PASS |
| Trace stages | n_stages | 5 | >= 3 | PASS |
| Trace PNGs | n_pngs | 5 | >= 3 | PASS |
| W1 wall time | w1_seconds | 0.02 s | — | info |
| W2 wall time | w2_seconds | 0.06 s | — | info |

## Reproducibility

- Seed: 42
- NumPy RNG state hash: 5d6a058f2c5c262e
- PWM version: 1dd6f76
- Python version: 3.13.9
- Key package versions: numpy=2.3.5, scipy=1.16.3
- Platform: Linux x86_64
- Deterministic reproduction command:
  ```bash
  pwm_cli run --modality mri --seed 42 --mode simulate,invert,calibrate
  ```
- Output hash (y): see RunBundle
- Output hash (x_hat): see RunBundle

## Saved artifacts

- RunBundle: `runs/run_mri_exp_f5f0882a/`
- Report: `pwm/reports/mri.md`
- Scoreboard: `pwm/reports/scoreboard.yaml`
- Node trace (.npy): `runs/run_mri_exp_f5f0882a/artifacts/trace/*.npy`
- Node trace (.png): `runs/run_mri_exp_f5f0882a/artifacts/trace/png/*.png`
- W2 operator metadata: `runs/run_mri_exp_f5f0882a/artifacts/w2_operator_meta.json`
- Ground truth: `runs/run_mri_exp_f5f0882a/artifacts/x_true.npy`
- Measurement: `runs/run_mri_exp_f5f0882a/artifacts/y.npy`
- W1 reconstruction: `runs/run_mri_exp_f5f0882a/artifacts/x_hat.npy`
- W2 reconstructions: `runs/run_mri_exp_f5f0882a/artifacts/x_hat_w2_uncorrected.npy`, `x_hat_w2_corrected.npy`

## Next actions

- Integrate SENSE parallel imaging with multi-coil data
- Test on fastMRI knee benchmark at 320x320 resolution
- Add VarNet and MoDL deep learning solvers
- Extend to multi-coil (n_coils=8) configuration
