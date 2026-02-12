# Single-Pixel Camera (SPC) — PWM Report

| Field | Value |
|-------|-------|
| Modality ID | `spc` |
| Category | compressive |
| Dataset | Set11 (synthetic proxy: pixel-sparse phantom) |
| Date | 2026-02-12 |
| PWM version | `ff34c58b9731` |
| Author | integritynoble |

## Modality overview

- Modality key: `spc`
- Category: compressive
- Forward model: y = Phi * x + n, where Phi is a random +/-1 Bernoulli measurement matrix (M x N), x is a sparse scene (N pixels), and n ~ N(0, sigma^2 I)
- Default solver: fista_l1 (ISTA/FISTA with L1 soft-thresholding for basis pursuit)
- Pipeline linearity: linear

The single-pixel camera (SPC) acquires M < N random linear projections of an N-pixel scene using a spatial light modulator (DMD). Reconstruction exploits signal sparsity via L1 minimization (basis pursuit / FISTA). The measurement matrix uses +/-1 Bernoulli entries normalized by 1/sqrt(N).

| Parameter | Value |
|-----------|-------|
| Image size (H x W) | 64 x 64 (N=4096) |
| Sampling rate | 15% |
| Measurements (M) | 614 |
| Compression ratio | 6.67:1 |
| Sparsity (k) | 100 nonzero pixels (2.4%) |
| Noise model | Additive Gaussian, sigma=0.01 |

## Standard dataset

- Name: Set11 Natural Images (synthetic proxy used here: pixel-sparse phantom)
- Source: https://github.com/jianzhangcs/SCSNet
- Size: 11 images, 256x256 grayscale (Set11); experiment uses 64x64 synthetic sparse phantom
- Registered in `dataset_registry.yaml` as `set11`

For this baseline experiment, a deterministic pixel-sparse phantom (seed=42, k=100 random bright pixels in [0.2, 1.0] on a 64x64 grid) is used. This matches the L1 sparsity assumption of the FISTA solver. Real-world SPC benchmarks use Set11 with DL-based solvers (ISTA-Net+, HATNet) achieving 31+ dB PSNR.

## PWM pipeline flowchart (mandatory)

```
x (world)
  ↓
SourceNode: photon_source — scales input by illumination strength (1.0)
  ↓
Element 1 (subrole=encoding): random_mask — random +/-1 Bernoulli matrix, 15% sampling
  ↓
SensorNode: photon_sensor — QE=0.9, gain=1.0, converts photon signal to electrons
  ↓
NoiseNode: poisson_gaussian_sensor — additive Gaussian noise (sigma=0.01)
  ↓
y
```

## Element inventory & mismatch parameters

| # | node_id | primitive_id | subrole | learnable params | mismatch knob | bounds | prior |
|---|---------|-------------|---------|-----------------|---------------|--------|-------|
| 1 | source | photon_source | source | strength=1.0 | — | — | — |
| 2 | measure | random_mask | encoding | seed=42, H=64, W=64, sampling_rate=0.15 | per_row_gain | [0.8, 1.2] | uniform |
| 3 | sensor | photon_sensor | sensor | QE=0.9, gain=1.0 | qe_drift | [0.8, 1.0] | normal(0.9, 0.02) |
| 4 | noise | poisson_gaussian_sensor | noise | read_sigma=0.01 | — | — | — |

## Node-by-node trace (one sample)

Trace of intermediate tensor states at each node for one representative sample (seed=42).
Each row references a saved artifact in the RunBundle.

| stage | node_id | output shape | dtype | range [min, max] | artifact_path |
|-------|---------|-------------|-------|-------------------|---------------|
| 0 | input x | (64, 64) | float64 | [0.0000, 0.9946] | `artifacts/trace/00_input_x.npy` |
| 1 | source | (64, 64) | float64 | [0.0000, 0.9946] | `artifacts/trace/01_source.npy` |
| 2 | measure | (614,) | float64 | [-0.3010, 0.3172] | `artifacts/trace/02_measure.npy` |
| 3 | sensor | (614,) | float64 | [-0.2709, 0.2855] | `artifacts/trace/03_sensor.npy` |
| 4 | noise (y) | (614,) | float64 | [-0.2793, 0.2878] | `artifacts/trace/04_noise_y.npy` |

PNG visualizations saved at: `artifacts/trace/png/`

## Workflow W1: Prompt-driven simulation + reconstruction

- **Prompt used:** `"Simulate single-pixel camera measurement of a sparse scene at 15% sampling and reconstruct"`
- **ExperimentSpec summary:**
  - modality: spc
  - mode: simulate -> invert
  - solver: fista_l1
  - photon_budget: N/A (Gaussian noise model)
- **Mode S results (simulate y):**
  - y shape: (614,)
  - y range: [-0.2793, 0.2878]
  - SNR: 19.2 dB
- **Mode I results (reconstruct x_hat):**
  - x_hat shape: (64, 64)
  - Solver: fista_l1 (FISTA with L1 soft-thresholding), iterations: 1000, lambda: 5e-4
- **Dataset metrics:**

  | Metric | Value |
  |--------|-------|
  | PSNR   | 30.20 dB |
  | SSIM   | 0.9070 |
  | NRMSE  | 0.0311 |

## Workflow W2: Operator correction mode (measured y + operator A)

- **Operator definition:**
  - A_definition: matrix
  - A_extraction_method: graph_stripped
  - Shape: (614, 4096)
  - A_sha256: 50ad8eaeb3d49ff4
  - Linearity: linear
  - Notes (if linearized): N/A
- **Mismatch specification:**
  - Mismatch type: synthetic_injected
  - Parameters perturbed: per-row gain drift in [0.9, 1.1] (614 independent factors)
  - Description: DMD nonuniformity — each measurement pattern row scaled by an independent random gain factor drawn from Uniform(0.9, 1.1), modeling per-pattern intensity variation
- **Mode C fit results:**
  - Correction family: Pre
  - Parameters fitted: 614 per-row gain factors (fitted via y_measured / y_predicted ratio)
  - NLL before correction: 377.5
  - NLL after correction: 40.1
  - NLL decrease: 337.4 (89.4%)
- **Mode I recon using corrected operator A':**

  | Metric | A₀ (uncorrected) | A' (corrected) | Delta |
  |--------|-------------------|----------------|-------|
  | PSNR   | 31.90 dB | 41.35 dB | +9.45 dB |
  | SSIM   | 0.9436 | 0.9939 | +0.0503 |

## Test results summary

### Quick gate (pass/fail)

- [x] W1 simulate: PASS
- [x] W1 reconstruct (PSNR >= 18 dB): PASS (30.20 dB)
- [x] W2 operator correction (NLL decreases): PASS (89.4% decrease)
- [x] W2 corrected recon (beats uncorrected): PASS (+9.45 dB)
- [x] Report contract (flowchart + all headings): PASS
- [x] Node-by-node trace saved (>= 3 stages): PASS (5 stages)
- [x] RunBundle saved (with trace PNGs): PASS
- [x] Scoreboard updated: PASS

### Full metrics

| Check | Metric | Value | Threshold | Status |
|-------|--------|-------|-----------|--------|
| W1 PSNR | psnr | 30.20 dB | >= 18 dB | PASS |
| W1 SSIM | ssim | 0.9070 | >= 0.80 | PASS |
| W1 NRMSE | nrmse | 0.0311 | — | info |
| W2 NLL decrease | nll_decrease_pct | 89.4% | >= 5% | PASS |
| W2 PSNR gain | psnr_delta | +9.45 dB | > 0 | PASS |
| W2 SSIM gain | ssim_delta | +0.0503 | > 0 | PASS |
| Trace stages | n_stages | 5 | >= 3 | PASS |
| Trace PNGs | n_pngs | 5 | >= 3 | PASS |
| W1 wall time | w1_seconds | 1.56 s | — | info |
| W2 wall time | w2_seconds | 3.16 s | — | info |

## Reproducibility

- Seed: 42
- NumPy RNG state hash: 5d6a058f2c5c262e
- PWM version: ff34c58b9731
- Python version: 3.13.9
- Key package versions: numpy=2.3.5, scipy=1.16.3
- Platform: Linux x86_64
- Deterministic reproduction command:
  ```bash
  pwm_cli run --modality spc --seed 42 --mode simulate,invert,calibrate
  ```
- Output hash (y): 32a6de1615ca1341
- Output hash (x_hat): 7652902db9e54ad4

## Saved artifacts

- RunBundle: `runs/run_spc_exp_4b56ea18/`
- Report: `pwm/reports/spc.md`
- Scoreboard: `pwm/reports/scoreboard.yaml`
- Node trace (.npy): `runs/run_spc_exp_4b56ea18/artifacts/trace/*.npy`
- Node trace (.png): `runs/run_spc_exp_4b56ea18/artifacts/trace/png/*.png`
- W2 operator metadata: `runs/run_spc_exp_4b56ea18/artifacts/w2_operator_meta.json`
- Ground truth: `runs/run_spc_exp_4b56ea18/artifacts/x_true.npy`
- Measurement: `runs/run_spc_exp_4b56ea18/artifacts/y.npy`
- W1 reconstruction: `runs/run_spc_exp_4b56ea18/artifacts/x_hat.npy`
- W2 reconstructions: `runs/run_spc_exp_4b56ea18/artifacts/x_hat_w2_uncorrected.npy`, `x_hat_w2_corrected.npy`
- Operators: `runs/run_spc_exp_4b56ea18/artifacts/A_original.npy`, `A_corrected.npy`

## Next actions

- Integrate DL-based SPC solvers (ISTA-Net+, HATNet) for natural image reconstruction
- Test on Set11 benchmark at 15% and 25% sampling rates
- Add wavelet/DCT sparsifying transform for non-sparse scenes
- Proceed to next modality: CASSI
