# Low-Dose Widefield Fluorescence Microscopy — PWM Report

| Field | Value |
|-------|-------|
| Modality ID | `widefield_lowdose` |
| Category | microscopy |
| Dataset | BioSR Low-SNR (synthetic proxy: fluorescence phantom) |
| Date | 2026-02-12 |
| PWM version | `af26b05` |
| Author | integritynoble |

## Modality overview

- Modality key: `widefield_lowdose`
- Category: microscopy
- Forward model: y = Poisson(peak * QE * gain * PSF ** x) + read_noise, where x is a 2D fluorescence distribution (H, W), PSF is Gaussian (sigma=3.0 px), peak=1000 photons, read_sigma=0.05
- Default solver: richardson_lucy_2d (Richardson-Lucy deconvolution, 30 iterations)
- Pipeline linearity: linear

Low-dose widefield fluorescence microscopy operates at reduced illumination to minimize phototoxicity and photobleaching. Images are dominated by shot noise and read noise. Reconstruction requires careful deconvolution with noise-aware regularization.

| Parameter | Value |
|-----------|-------|
| Image size (H x W) | 64 x 64 (N=4096) |
| PSF type | Gaussian, sigma=3.0 pixels |
| Photon budget (peak) | 1000 |
| Read noise sigma | 0.05 |
| Noise model | Poisson-Gaussian |
| SNR | 11.1 dB |

## Standard dataset

- Name: BioSR Low-SNR subset (synthetic proxy used here: fluorescence phantom)
- Source: https://figshare.com/articles/dataset/BioSR/13744429
- Size: 400 images, 502x502 fluorescence (low photon count); experiment uses 64x64 synthetic phantom
- Registered in `dataset_registry.yaml` as `biosr_lowsnr`

For this baseline experiment, a deterministic fluorescence phantom (seed=42) is used with reduced photon count (1000 vs 10000 in standard widefield).

## PWM pipeline flowchart (mandatory)

```
x (world)
  ↓
SourceNode: photon_source — scales fluorescence by excitation strength (1.0)
  ↓
Element 1 (subrole=transport): conv2d — Gaussian PSF convolution (sigma=3.0, mode=reflect)
  ↓
SensorNode: photon_sensor — QE=0.9, gain=1.0, converts photon signal to electrons
  ↓
NoiseNode: poisson_gaussian_sensor — Poisson shot noise (peak=1000) + Gaussian read noise (sigma=0.05)
  ↓
y
```

## Element inventory & mismatch parameters

| # | node_id | primitive_id | subrole | learnable params | mismatch knob | bounds | prior |
|---|---------|-------------|---------|-----------------|---------------|--------|-------|
| 1 | source | photon_source | source | strength=1.0 | — | — | — |
| 2 | blur | conv2d | transport | sigma=3.0, mode=reflect | psf_sigma | [0.5, 3.0] | normal(3.0, 0.3) |
| 3 | sensor | photon_sensor | sensor | QE=0.9, gain=1.0 | gain | [0.5, 1.5] | normal(1.0, 0.1) |
| 4 | noise | poisson_gaussian_sensor | noise | peak_photons=1000, read_sigma=0.05 | — | — | — |

## Node-by-node trace (one sample)

Trace of intermediate tensor states at each node for one representative sample (seed=42).
Each row references a saved artifact in the RunBundle.

| stage | node_id | output shape | dtype | range [min, max] | artifact_path |
|-------|---------|-------------|-------|-------------------|---------------|
| 0 | input x | (64, 64) | float64 | [0.0500, 0.9392] | `artifacts/trace/00_input_x.npy` |
| 1 | source | (64, 64) | float64 | [0.0500, 0.9392] | `artifacts/trace/01_source.npy` |
| 2 | blur | (64, 64) | float64 | [0.0547, 0.7621] | `artifacts/trace/02_blur.npy` |
| 3 | sensor | (64, 64) | float64 | [0.0493, 0.6859] | `artifacts/trace/03_sensor.npy` |
| 4 | noise (y) | (64, 64) | float64 | [-0.1254, 0.7114] | `artifacts/trace/04_noise_y.npy` |

PNG visualizations saved at: `artifacts/trace/png/`

## Workflow W1: Prompt-driven simulation + reconstruction

- **Prompt used:** `"Simulate low-dose widefield fluorescence microscopy and reconstruct using Richardson-Lucy"`
- **ExperimentSpec summary:**
  - modality: widefield_lowdose
  - mode: simulate -> invert
  - solver: richardson_lucy_2d
  - photon_budget: 1000 peak photons
- **Mode S results (simulate y):**
  - y shape: (64, 64)
  - y range: [-0.1254, 0.7114]
  - SNR: 11.1 dB
- **Mode I results (reconstruct x_hat):**
  - x_hat shape: (64, 64)
  - Solver: richardson_lucy_2d, iterations: 30
- **Dataset metrics:**

  | Metric | Value |
  |--------|-------|
  | PSNR   | 23.89 dB |
  | SSIM   | 0.9161 |
  | NRMSE  | 0.0675 |

## Workflow W2: Operator correction mode (measured y + operator A)

- **Operator definition:**
  - A_definition: callable
  - A_extraction_method: graph_stripped
  - Operator chain: PhotonSource → Conv2d → PhotonSensor (noise stripped)
  - A_sha256: c57775d7e456b32c
  - Linearity: linear
  - Notes (if linearized): N/A (linear Gaussian convolution)
- **Mismatch specification:**
  - Mismatch type: synthetic_injected
  - Parameters perturbed: Detector gain (1.0 → 1.3, +30%)
  - Description: Camera gain miscalibration due to aging or temperature drift, causing 30% intensity overestimation
- **Mode C fit results:**
  - Correction family: Pre
  - Parameters fitted: 1 parameter (gain) via grid search over [0.5, 1.5]
  - Best fitted gain: 1.30
  - NLL before correction: 4292.9
  - NLL after correction: 2048.0
  - NLL decrease: 2244.9 (52.3%)
- **Mode I recon using corrected operator A':**

  | Metric | A₀ (uncorrected) | A' (corrected) | Delta |
  |--------|-------------------|----------------|-------|
  | PSNR   | 22.51 dB | 24.53 dB | +2.02 dB |
  | SSIM   | 0.8950 | 0.9260 | +0.0310 |

## Test results summary

### Quick gate (pass/fail)

- [x] W1 simulate: PASS
- [x] W1 reconstruct (PSNR >= 18 dB): PASS (23.89 dB)
- [x] W2 operator correction (NLL decreases): PASS (52.3% decrease)
- [x] W2 corrected recon (beats uncorrected): PASS (+2.02 dB)
- [x] Report contract (flowchart + all headings): PASS
- [x] Node-by-node trace saved (>= 3 stages): PASS (5 stages)
- [x] RunBundle saved (with trace PNGs): PASS
- [x] Scoreboard updated: PASS

### Full metrics

| Check | Metric | Value | Threshold | Status |
|-------|--------|-------|-----------|--------|
| W1 PSNR | psnr | 23.89 dB | >= 18 dB | PASS |
| W1 SSIM | ssim | 0.9161 | >= 0.80 | PASS |
| W1 NRMSE | nrmse | 0.0675 | — | info |
| W2 NLL decrease | nll_decrease_pct | 52.3% | >= 5% | PASS |
| W2 PSNR gain | psnr_delta | +2.02 dB | > 0 | PASS |
| W2 SSIM gain | ssim_delta | +0.0310 | > 0 | PASS |
| Trace stages | n_stages | 5 | >= 3 | PASS |
| Trace PNGs | n_pngs | 5 | >= 3 | PASS |
| W1 wall time | w1_seconds | 0.02 s | — | info |
| W2 wall time | w2_seconds | 0.04 s | — | info |

## Reproducibility

- Seed: 42
- NumPy RNG state hash: 5d6a058f2c5c262e
- PWM version: af26b05
- Python version: 3.13.9
- Key package versions: numpy=2.3.5, scipy=1.16.3
- Platform: Linux x86_64
- Deterministic reproduction command:
  ```bash
  pwm_cli run --modality widefield_lowdose --seed 42 --mode simulate,invert,calibrate
  ```
- Output hash (y): see RunBundle
- Output hash (x_hat): see RunBundle

## Saved artifacts

- RunBundle: `runs/run_widefield_lowdose_exp_2c006df8/`
- Report: `pwm/reports/widefield_lowdose.md`
- Scoreboard: `pwm/reports/scoreboard.yaml`
- Node trace (.npy): `runs/run_widefield_lowdose_exp_2c006df8/artifacts/trace/*.npy`
- Node trace (.png): `runs/run_widefield_lowdose_exp_2c006df8/artifacts/trace/png/*.png`
- W2 operator metadata: `runs/run_widefield_lowdose_exp_2c006df8/artifacts/w2_operator_meta.json`
- Ground truth: `runs/run_widefield_lowdose_exp_2c006df8/artifacts/x_true.npy`
- Measurement: `runs/run_widefield_lowdose_exp_2c006df8/artifacts/y.npy`
- W1 reconstruction: `runs/run_widefield_lowdose_exp_2c006df8/artifacts/x_hat.npy`
- W2 reconstructions: `runs/run_widefield_lowdose_exp_2c006df8/artifacts/x_hat_w2_uncorrected.npy`, `x_hat_w2_corrected.npy`

## Next actions

- Test on BioSR Low-SNR benchmark at 502x502 resolution
- Add Noise2Void and CARE deep learning denoisers
- Implement BM3D + RL pipeline for improved low-dose reconstruction
- Test extreme low-dose regime (100 photons)
