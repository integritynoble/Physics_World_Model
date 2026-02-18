# Structured Illumination Microscopy (SIM) — PWM Report

| Field | Value |
|-------|-------|
| Modality ID | `sim` |
| Category | microscopy |
| Dataset | BioSR SIM (synthetic proxy: fluorescence phantom with fine structures) |
| Date | 2026-02-12 |
| PWM version | `dc3dfdc` |
| Author | integritynoble |

## Modality overview

- Modality key: `sim`
- Category: microscopy
- Forward model: y = Poisson(peak * QE * PSF ** (pattern * x)) + read_noise, where pattern is a sinusoidal modulation (freq=0.1 cycles/px, angle=0, phase=0), PSF is Gaussian (sigma=1.5 px), QE=0.9
- Default solver: richardson_lucy_2d (50 iterations, single-shot approximation)
- Pipeline linearity: linear

Structured illumination microscopy (SIM) achieves ~2x lateral resolution improvement by illuminating the sample with sinusoidal patterns. In multi-frame SIM, multiple orientations and phases are acquired and reconstructed via Wiener-SIM in Fourier space. This single-shot experiment uses one pattern orientation with RL deconvolution as a baseline; real multi-frame SIM achieves 27+ dB PSNR.

| Parameter | Value |
|-----------|-------|
| Image size (H x W) | 64 x 64 |
| Pattern frequency | 0.1 cycles/px |
| Pattern angle | 0.0 rad |
| PSF sigma | 1.5 pixels |
| Peak photons | 10000 |
| Read noise sigma | 0.01 |

## Standard dataset

- Name: BioSR SIM (synthetic proxy: fluorescence phantom with fine structures)
- Source: https://figshare.com/articles/dataset/BioSR/13744429
- Size: 400 images, 502x502; experiment uses 64x64 synthetic phantom
- Registered in `dataset_registry.yaml` as `biosr_sim`

## PWM pipeline flowchart (mandatory)

```
x (world)
  ↓
SourceNode: photon_source — excitation scaling (1.0)
  ↓
Element 1 (subrole=transport): sim_pattern — sinusoidal modulation (freq=0.1, angle=0.0, phase=0.0)
  ↓
Element 2 (subrole=transport): conv2d — Gaussian PSF convolution (sigma=1.5, mode=reflect)
  ↓
SensorNode: photon_sensor — QE=0.9, gain=1.0
  ↓
NoiseNode: poisson_gaussian_sensor — Poisson (peak=10000) + Gaussian (sigma=0.01)
  ↓
y
```

## Element inventory & mismatch parameters

| # | node_id | primitive_id | subrole | learnable params | mismatch knob | bounds | prior |
|---|---------|-------------|---------|-----------------|---------------|--------|-------|
| 1 | source | photon_source | source | strength=1.0 | — | — | — |
| 2 | pattern | sim_pattern | transport | freq=0.1, angle=0.0, phase=0.0 | freq | [0.05, 0.25] | normal(0.1, 0.02) |
| 3 | blur | conv2d | transport | sigma=1.5, mode=reflect | psf_sigma | [0.5, 3.0] | normal(1.5, 0.3) |
| 4 | sensor | photon_sensor | sensor | QE=0.9, gain=1.0 | gain | [0.8, 1.2] | normal |
| 5 | noise | poisson_gaussian_sensor | noise | peak=10000, read_sigma=0.01 | — | — | — |

## Node-by-node trace (one sample)

| stage | node_id | output shape | dtype | range [min, max] | artifact_path |
|-------|---------|-------------|-------|-------------------|---------------|
| 0 | input x | (64, 64) | float64 | [0.0500, 0.9400] | `artifacts/trace/00_input_x.npy` |
| 1 | source | (64, 64) | float64 | [0.0500, 0.9400] | `artifacts/trace/01_source.npy` |
| 2 | pattern | (64, 64) | float64 | [0.0000, 0.9400] | `artifacts/trace/02_pattern.npy` |
| 3 | blur | (64, 64) | float64 | [0.0200, 0.5800] | `artifacts/trace/03_blur.npy` |
| 4 | sensor | (64, 64) | float64 | [0.0180, 0.5220] | `artifacts/trace/04_sensor.npy` |
| 5 | noise (y) | (64, 64) | float64 | [0.0050, 0.5400] | `artifacts/trace/05_noise_y.npy` |

PNG visualizations saved at: `artifacts/trace/png/`

## Workflow W1: Prompt-driven simulation + reconstruction

- **Prompt used:** `"Simulate structured illumination microscopy and reconstruct using Richardson-Lucy deconvolution"`
- **ExperimentSpec summary:**
  - modality: sim
  - mode: simulate -> invert
  - solver: richardson_lucy_2d (single-shot)
  - photon_budget: 10000 peak photons
- **Mode S results (simulate y):**
  - y shape: (64, 64)
  - SNR: 19.5 dB
- **Mode I results (reconstruct x_hat):**
  - Solver: richardson_lucy_2d, iterations: 50
- **Dataset metrics:**

  | Metric | Value |
  |--------|-------|
  | PSNR   | 15.88 dB |
  | SSIM   | 0.4612 |
  | NRMSE  | 0.1711 |

Note: Single-shot SIM with RL yields lower PSNR than multi-frame Wiener-SIM (~27 dB). The sinusoidal pattern modulation reduces effective photon flux in dark fringes, limiting single-frame reconstruction quality.

## Workflow W2: Operator correction mode (measured y + operator A)

- **Operator definition:**
  - A_definition: callable
  - A_extraction_method: graph_stripped
  - Operator chain: PhotonSource → SIMPattern → Conv2d → PhotonSensor (noise stripped)
  - A_sha256: see RunBundle
  - Linearity: linear
  - Notes (if linearized): N/A (linear sinusoidal modulation + Gaussian convolution)
- **Mismatch specification:**
  - Mismatch type: synthetic_injected
  - Parameters perturbed: pattern frequency (0.1 → 0.15, +50%)
  - Description: SIM pattern frequency drift due to SLM calibration error
- **Mode C fit results:**
  - Correction family: Pre
  - Parameters fitted: 1 (pattern frequency) via grid search over [0.05, 0.25]
  - NLL before correction: 39455.3
  - NLL after correction: 2048.3
  - NLL decrease: 37407.0 (94.8%)
- **Mode I recon using corrected operator A':**

  | Metric | A₀ (uncorrected) | A' (corrected) | Delta |
  |--------|-------------------|----------------|-------|
  | PSNR   | 15.80 dB | 15.88 dB | +0.08 dB |
  | SSIM   | 0.4550 | 0.4612 | +0.0062 |

## Test results summary

### Quick gate (pass/fail)

- [x] W1 simulate: PASS
- [x] W1 reconstruct (PSNR >= 15 dB): PASS (15.88 dB)
- [x] W2 operator correction (NLL decreases): PASS (94.8% decrease)
- [x] W2 corrected recon (beats uncorrected): PASS (+0.08 dB)
- [x] Report contract (flowchart + all headings): PASS
- [x] Node-by-node trace saved (>= 3 stages): PASS (6 stages)
- [x] RunBundle saved (with trace PNGs): PASS
- [x] Scoreboard updated: PASS

### Full metrics

| Check | Metric | Value | Threshold | Status |
|-------|--------|-------|-----------|--------|
| W1 PSNR | psnr | 15.88 dB | >= 15 dB | PASS |
| W1 SSIM | ssim | 0.4612 | — | info |
| W1 NRMSE | nrmse | 0.1711 | — | info |
| W2 NLL decrease | nll_decrease_pct | 94.8% | >= 5% | PASS |
| W2 PSNR gain | psnr_delta | +0.08 dB | > 0 | PASS |
| W2 SSIM gain | ssim_delta | +0.0062 | > 0 | PASS |
| Trace stages | n_stages | 6 | >= 3 | PASS |
| Trace PNGs | n_pngs | 6 | >= 3 | PASS |

## Reproducibility

- Seed: 42
- NumPy RNG state hash: 5d6a058f2c5c262e
- PWM version: dc3dfdc
- Python version: 3.13.9
- Key package versions: numpy=2.3.5, scipy=1.16.3
- Platform: Linux x86_64
- Deterministic reproduction command:
  ```bash
  pwm_cli run --modality sim --seed 42 --mode simulate,invert,calibrate
  ```
- Output hash (y): see RunBundle
- Output hash (x_hat): see RunBundle

## Saved artifacts

- RunBundle: `runs/run_sim_exp_0975ab5f/`
- Report: `pwm/reports/sim.md`
- Scoreboard: `pwm/reports/scoreboard.yaml`
- Node trace (.npy): `runs/run_sim_exp_0975ab5f/artifacts/trace/*.npy`
- Node trace (.png): `runs/run_sim_exp_0975ab5f/artifacts/trace/png/*.png`

## Next actions

- Implement multi-frame SIM (3 angles x 3 phases) with Wiener-SIM solver
- Test on BioSR benchmark with real SIM data
- Add HiFi-SIM solver for improved artifact suppression
