# Fluorescence Lifetime Imaging (FLIM) — PWM Report

| Field | Value |
|-------|-------|
| Modality ID | `flim` |
| Category | microscopy |
| Dataset | FLIM-FRET Benchmark (synthetic proxy: fluorescence phantom with temporal gating) |
| Date | 2026-02-12 |
| PWM version | `94207db` |
| Author | integritynoble |

## Modality overview

- Modality key: `flim`
- Category: microscopy
- Forward model: y = Poisson(peak * QE * gate * PSF ** x) + read_noise, where gate is a temporal binary mask, PSF is Gaussian (sigma=2.0 px), QE=0.9, peak=3000
- Default solver: richardson_lucy_2d (50 iterations)
- Pipeline linearity: linear

Fluorescence Lifetime Imaging Microscopy (FLIM) measures fluorescence decay kinetics at each pixel. The temporal gating masks select time windows of photon arrivals, providing lifetime contrast. The low photon budget (3000) and temporal gating reduce effective signal, making reconstruction challenging.

| Parameter | Value |
|-----------|-------|
| Image size (H x W) | 64 x 64 |
| PSF sigma | 2.0 pixels |
| Temporal gate | binary mask (T=1, seed=42) |
| Peak photons | 3000 |
| Read noise sigma | 0.03 |

## Standard dataset

- Name: FLIM-FRET Benchmark (synthetic proxy: fluorescence phantom)
- Source: https://zenodo.org/record/8139025
- Size: 100 images, 256x256 with lifetime curves; experiment uses 64x64 phantom
- Registered in `dataset_registry.yaml` as `flim_fret_benchmark`

## PWM pipeline flowchart (mandatory)

```
x (world)
  ↓
SourceNode: photon_source — pulsed excitation scaling (1.0)
  ↓
Element 1 (subrole=transport): conv2d — PSF convolution (sigma=2.0, mode=reflect)
  ↓
Element 2 (subrole=encoding): temporal_mask — temporal gating (T=1, binary mask)
  ↓
SensorNode: photon_sensor — QE=0.9, gain=1.0 (TCSPC detector)
  ↓
NoiseNode: poisson_gaussian_sensor — Poisson (peak=3000) + Gaussian (sigma=0.03)
  ↓
y
```

## Element inventory & mismatch parameters

| # | node_id | primitive_id | subrole | learnable params | mismatch knob | bounds | prior |
|---|---------|-------------|---------|-----------------|---------------|--------|-------|
| 1 | source | photon_source | source | strength=1.0 | — | — | — |
| 2 | blur | conv2d | transport | sigma=2.0, mode=reflect | psf_sigma | [0.5, 4.0] | normal(2.0, 0.3) |
| 3 | gate | temporal_mask | encoding | T=1, seed=42 | — | — | — |
| 4 | sensor | photon_sensor | sensor | QE=0.9, gain=1.0 | gain | [0.5, 2.0] | normal |
| 5 | noise | poisson_gaussian_sensor | noise | peak=3000, read_sigma=0.03 | — | — | — |

## Node-by-node trace (one sample)

| stage | node_id | output shape | dtype | range [min, max] | artifact_path |
|-------|---------|-------------|-------|-------------------|---------------|
| 0 | input x | (64, 64) | float64 | [0.0500, 0.9400] | `artifacts/trace/00_input_x.npy` |
| 1 | source | (64, 64) | float64 | [0.0500, 0.9400] | `artifacts/trace/01_source.npy` |
| 2 | blur | (64, 64) | float64 | [0.0500, 0.7800] | `artifacts/trace/02_blur.npy` |
| 3 | gate | (64, 64) | float64 | [0.0000, 0.7800] | `artifacts/trace/03_gate.npy` |
| 4 | sensor | (64, 64) | float64 | [0.0000, 0.7020] | `artifacts/trace/04_sensor.npy` |
| 5 | noise (y) | (64, 64) | float64 | [0.0000, 0.7200] | `artifacts/trace/05_noise_y.npy` |

PNG visualizations saved at: `artifacts/trace/png/`

## Workflow W1: Prompt-driven simulation + reconstruction

- **Prompt used:** `"Simulate FLIM with temporal gating and reconstruct using Richardson-Lucy deconvolution"`
- **ExperimentSpec summary:**
  - modality: flim
  - mode: simulate -> invert
  - solver: richardson_lucy_2d
  - photon_budget: 3000 peak photons
- **Mode S results (simulate y):**
  - y shape: (64, 64)
  - SNR: 12.7 dB
- **Mode I results (reconstruct x_hat):**
  - Solver: richardson_lucy_2d, iterations: 50
- **Dataset metrics:**

  | Metric | Value |
  |--------|-------|
  | PSNR   | 13.54 dB |
  | SSIM   | 0.4600 |
  | NRMSE  | 0.2266 |

Note: FLIM's low photon budget (3000) combined with temporal gating (binary mask) significantly reduces effective signal, limiting single-frame reconstruction. Real FLIM uses phasor analysis or MLE fitting on multi-time-bin data for better results.

## Workflow W2: Operator correction mode (measured y + operator A)

- **Operator definition:**
  - A_definition: callable
  - A_extraction_method: graph_stripped
  - Operator chain: PhotonSource → Conv2d → TemporalMask → PhotonSensor (noise stripped)
  - A_sha256: see RunBundle
  - Linearity: linear
  - Notes (if linearized): N/A (linear convolution + binary mask)
- **Mismatch specification:**
  - Mismatch type: synthetic_injected
  - Parameters perturbed: sensor gain (1.0 → 1.4, +40%)
  - Description: TCSPC detector gain drift from high-voltage supply aging
- **Mode C fit results:**
  - Correction family: Pre
  - Parameters fitted: 1 (gain) via grid search over [0.5, 2.0]
  - NLL before correction: 8036.8
  - NLL after correction: 2048.3
  - NLL decrease: 5988.5 (74.5%)
- **Mode I recon using corrected operator A':**

  | Metric | A₀ (uncorrected) | A' (corrected) | Delta |
  |--------|-------------------|----------------|-------|
  | PSNR   | 13.40 dB | 13.54 dB | +0.14 dB |
  | SSIM   | 0.4500 | 0.4600 | +0.0100 |

## Test results summary

### Quick gate (pass/fail)

- [x] W1 simulate: PASS
- [x] W1 reconstruct (PSNR >= 12 dB): PASS (13.54 dB)
- [x] W2 operator correction (NLL decreases): PASS (74.5% decrease)
- [x] W2 corrected recon (beats uncorrected): PASS (+0.14 dB)
- [x] Report contract (flowchart + all headings): PASS
- [x] Node-by-node trace saved (>= 3 stages): PASS (6 stages)
- [x] RunBundle saved (with trace PNGs): PASS
- [x] Scoreboard updated: PASS

### Full metrics

| Check | Metric | Value | Threshold | Status |
|-------|--------|-------|-----------|--------|
| W1 PSNR | psnr | 13.54 dB | >= 12 dB | PASS |
| W1 SSIM | ssim | 0.4600 | — | info |
| W1 NRMSE | nrmse | 0.2266 | — | info |
| W2 NLL decrease | nll_decrease_pct | 74.5% | >= 5% | PASS |
| W2 PSNR gain | psnr_delta | +0.14 dB | > 0 | PASS |
| W2 SSIM gain | ssim_delta | +0.0100 | > 0 | PASS |
| Trace stages | n_stages | 6 | >= 3 | PASS |
| Trace PNGs | n_pngs | 6 | >= 3 | PASS |

## Reproducibility

- Seed: 42
- NumPy RNG state hash: 5d6a058f2c5c262e
- PWM version: 94207db
- Python version: 3.13.9
- Key package versions: numpy=2.3.5, scipy=1.16.3
- Platform: Linux x86_64
- Deterministic reproduction command:
  ```bash
  pwm_cli run --modality flim --seed 42 --mode simulate,invert,calibrate
  ```
- Output hash (y): see RunBundle
- Output hash (x_hat): see RunBundle

## Saved artifacts

- RunBundle: `runs/run_flim_exp_9360bf8f/`
- Report: `pwm/reports/flim.md`
- Scoreboard: `pwm/reports/scoreboard.yaml`

## Next actions

- Implement phasor analysis solver for lifetime extraction
- Test on FLIM-FRET benchmark with multi-time-bin data
- Add MLE fitting for multi-exponential decay recovery
