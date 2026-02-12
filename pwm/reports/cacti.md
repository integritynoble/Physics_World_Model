# Coded Aperture Compressive Temporal Imaging (CACTI) — PWM Report

| Field | Value |
|-------|-------|
| Modality ID | `cacti` |
| Category | compressive |
| Dataset | Grayscale Video (synthetic proxy: moving-blob video phantom) |
| Date | 2026-02-12 |
| PWM version | `7394757db7a4` |
| Author | integritynoble |

## Modality overview

- Modality key: `cacti`
- Category: compressive
- Forward model: y = Sensor(sum_t(Mask_t * Source(x_t))) + n, where x is a 3D video cube (H, W, T), Mask_t is a time-varying binary mask (rolled from base), summation collapses temporal axis, and n ~ N(0, sigma^2 I)
- Default solver: gap_tv_cacti (GAP-TV with spatial+temporal Total Variation regularization)
- Pipeline linearity: linear

CACTI (also known as Snapshot Compressive Imaging for video) captures T video frames in a single 2D snapshot using time-varying coded aperture masks. Each video frame is modulated by a different shifted version of a base binary mask, and all masked frames are summed on the detector. Reconstruction recovers the 3D video cube from the 2D measurement using GAP-TV with spatial and temporal TV regularization.

| Parameter | Value |
|-----------|-------|
| Image size (H x W x T) | 64 x 64 x 8 (N=32768) |
| Measurements (M) | 4096 (64 x 64) |
| Compression ratio | 8:1 |
| Temporal frames | 8 |
| Mask density | 50% (binary random) |
| Mask shift | Vertical roll, 1 pixel/frame |
| Noise model | Additive Gaussian, sigma=0.01 |

## Standard dataset

- Name: Grayscale Video (synthetic proxy used here: moving-blob video phantom)
- Source: DAVIS 2017 video segmentation dataset (480p)
- Size: 90 video sequences, 480x854; experiment uses 64x64x8 synthetic phantom
- Registered in `dataset_registry.yaml` as `davis_video`

For this baseline experiment, a deterministic video phantom (seed=42, 5 Gaussian-blob objects with smooth temporal motion, mapped to 8 frames) is used. The smooth spatiotemporal structure matches the TV regularization assumption of GAP-TV. Real-world CACTI benchmarks on DAVIS achieve 30+ dB PSNR with DL-based solvers (EfficientSCI, RevSCI).

## PWM pipeline flowchart (mandatory)

```
x (world)
  ↓
SourceNode: photon_source — scales input by illumination strength (1.0)
  ↓
Element 1 (subrole=encoding): temporal_mask — time-varying binary masks, compresses T frames to 1
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
| 2 | mask | temporal_mask | encoding | seed=42, H=64, W=64, T=8 | mask_shift, temporal_jitter | [-4, 4], [-2, 2] | uniform |
| 3 | sensor | photon_sensor | sensor | QE=0.9, gain=1.0 | qe_drift, gain | [0.8, 1.0], [0.9, 1.1] | normal |
| 4 | noise | poisson_gaussian_sensor | noise | read_sigma=0.01 | — | — | — |

## Node-by-node trace (one sample)

Trace of intermediate tensor states at each node for one representative sample (seed=42).
Each row references a saved artifact in the RunBundle.

| stage | node_id | output shape | dtype | range [min, max] | artifact_path |
|-------|---------|-------------|-------|-------------------|---------------|
| 0 | input x | (64, 64, 8) | float64 | [0.0500, 0.9500] | `artifacts/trace/00_input_x.npy` |
| 1 | source | (64, 64, 8) | float64 | [0.0500, 0.9500] | `artifacts/trace/01_source.npy` |
| 2 | mask | (64, 64) | float64 | [0.0000, 4.4849] | `artifacts/trace/02_mask.npy` |
| 3 | sensor | (64, 64) | float64 | [0.0000, 4.0364] | `artifacts/trace/03_sensor.npy` |
| 4 | noise (y) | (64, 64) | float64 | [-0.0155, 4.0376] | `artifacts/trace/04_noise_y.npy` |

PNG visualizations saved at: `artifacts/trace/png/`

## Workflow W1: Prompt-driven simulation + reconstruction

- **Prompt used:** `"Simulate CACTI measurement of an 8-frame video scene and reconstruct using GAP-TV"`
- **ExperimentSpec summary:**
  - modality: cacti
  - mode: simulate -> invert
  - solver: gap_tv_cacti
  - photon_budget: N/A (Gaussian noise model)
- **Mode S results (simulate y):**
  - y shape: (64, 64)
  - y range: [-0.0155, 4.0376]
  - SNR: 36.4 dB
- **Mode I results (reconstruct x_hat):**
  - x_hat shape: (64, 64, 8)
  - Solver: gap_tv_cacti (GAP-TV with spatial+temporal TV denoising), iterations: 50, lambda: 0.001
- **Dataset metrics:**

  | Metric | Value |
  |--------|-------|
  | PSNR   | 17.43 dB |
  | SSIM   | 0.3999 |
  | NRMSE  | 0.1493 |

Note: CACTI operates at 8:1 temporal compression (32768 unknowns from 4096 measurements). The GAP-TV solver uses mask structure directly (not generic operator) for efficient reconstruction. Published GAP-TV baselines on DAVIS achieve ~30 dB with optimized implementations; our generic solver provides a correct baseline at reduced scale.

## Workflow W2: Operator correction mode (measured y + operator A)

- **Operator definition:**
  - A_definition: callable
  - A_extraction_method: graph_stripped
  - Operator chain: PhotonSource → TemporalMask → PhotonSensor (noise stripped)
  - A_sha256: 4ac899bfa1fa4b47
  - Linearity: linear
  - Notes (if linearized): N/A
- **Mismatch specification:**
  - Mismatch type: synthetic_injected
  - Parameters perturbed: mask timing jitter (seed 42 → 99, entirely different mask realization)
  - Description: Mask-detector synchronization error — the mask seed changes from 42 to 99, simulating a completely different mask pattern due to timing desynchronization
- **Mode C fit results:**
  - Correction family: Pre
  - Parameters fitted: 1 parameter (mask seed) via grid search over seeds [30..120]
  - Best fitted seed: 99
  - NLL before correction: 2317467.2
  - NLL after correction: 1997.0
  - NLL decrease: 2315470.2 (99.9%)
- **Mode I recon using corrected operator A':**

  | Metric | A₀ (uncorrected) | A' (corrected) | Delta |
  |--------|-------------------|----------------|-------|
  | PSNR   | 16.48 dB | 17.36 dB | +0.88 dB |
  | SSIM   | 0.3332 | 0.3891 | +0.0559 |

## Test results summary

### Quick gate (pass/fail)

- [x] W1 simulate: PASS
- [x] W1 reconstruct (PSNR >= 8 dB): PASS (17.43 dB)
- [x] W2 operator correction (NLL decreases): PASS (99.9% decrease)
- [x] W2 corrected recon (beats uncorrected): PASS (+0.88 dB)
- [x] Report contract (flowchart + all headings): PASS
- [x] Node-by-node trace saved (>= 3 stages): PASS (5 stages)
- [x] RunBundle saved (with trace PNGs): PASS
- [x] Scoreboard updated: PASS

### Full metrics

| Check | Metric | Value | Threshold | Status |
|-------|--------|-------|-----------|--------|
| W1 PSNR | psnr | 17.43 dB | >= 8 dB | PASS |
| W1 SSIM | ssim | 0.3999 | >= 0.10 | PASS |
| W1 NRMSE | nrmse | 0.1493 | — | info |
| W2 NLL decrease | nll_decrease_pct | 99.9% | >= 5% | PASS |
| W2 PSNR gain | psnr_delta | +0.88 dB | > 0 | PASS |
| W2 SSIM gain | ssim_delta | +0.0559 | > 0 | PASS |
| Trace stages | n_stages | 5 | >= 3 | PASS |
| Trace PNGs | n_pngs | 5 | >= 3 | PASS |
| W1 wall time | w1_seconds | 0.59 s | — | info |
| W2 wall time | w2_seconds | 1.94 s | — | info |

## Reproducibility

- Seed: 42
- NumPy RNG state hash: 5d6a058f2c5c262e
- PWM version: 7394757db7a4
- Python version: 3.13.9
- Key package versions: numpy=2.3.5, scipy=1.16.3
- Platform: Linux x86_64
- Deterministic reproduction command:
  ```bash
  pwm_cli run --modality cacti --seed 42 --mode simulate,invert,calibrate
  ```
- Output hash (y): see RunBundle
- Output hash (x_hat): see RunBundle

## Saved artifacts

- RunBundle: `runs/run_cacti_exp_fb9cb1ac/`
- Report: `pwm/reports/cacti.md`
- Scoreboard: `pwm/reports/scoreboard.yaml`
- Node trace (.npy): `runs/run_cacti_exp_fb9cb1ac/artifacts/trace/*.npy`
- Node trace (.png): `runs/run_cacti_exp_fb9cb1ac/artifacts/trace/png/*.png`
- W2 operator metadata: `runs/run_cacti_exp_fb9cb1ac/artifacts/w2_operator_meta.json`
- Ground truth: `runs/run_cacti_exp_fb9cb1ac/artifacts/x_true.npy`
- Measurement: `runs/run_cacti_exp_fb9cb1ac/artifacts/y.npy`
- W1 reconstruction: `runs/run_cacti_exp_fb9cb1ac/artifacts/x_hat.npy`
- W2 reconstructions: `runs/run_cacti_exp_fb9cb1ac/artifacts/x_hat_w2_uncorrected.npy`, `x_hat_w2_corrected.npy`

## Next actions

- Integrate EfficientSCI deep learning solver for improved CACTI reconstruction
- Test on DAVIS benchmark at 256x256x8 resolution
- Add RevSCI and PnP-FFDNet solvers
- Proceed to next modality: CT
