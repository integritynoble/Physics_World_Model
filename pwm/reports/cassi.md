# Coded Aperture Snapshot Spectral Imaging (CASSI) — PWM Report

| Field | Value |
|-------|-------|
| Modality ID | `cassi` |
| Category | compressive |
| Dataset | TSA Simulation Benchmark — 10 scenes (256x256x28, step=2 dispersion) |
| Date | 2026-02-12 |
| PWM version | `ae5106be6a89` |
| Author | integritynoble |

## Modality overview

- Modality key: `cassi`
- Category: compressive
- Forward model: y = PSF * sum_l(shift(warp(Mask,dx,dy,theta) * x, a1,a2,alpha)) + noise, where x is a 3D spectral cube (H, W, L), Mask is a coded aperture (float-valued) warped by affine transform (dx,dy,theta), shift disperses each band with parametric curve delta(l)=a1*l+a2*l^2 along axis angle alpha, PSF is Gaussian blur (sigma), and noise is Poisson shot + Gaussian read noise
- Default solver: GAP-TV (classical), with HDNet, MST-S, MST-L as deep alternatives
- Pipeline linearity: linear

CASSI captures a single 2D snapshot of a 3D hyperspectral scene using a coded aperture mask and spectral disperser. The mask modulates the spectral cube element-wise, the disperser shifts each band spatially by a parametric dispersion curve, and integration collapses the spectral axis to produce a 2D measurement. An optional PSF blur and Poisson+read noise model complete the realistic forward chain. Reconstruction recovers the 3D cube from the 2D measurement. This benchmark evaluates 4 solvers across 10 test scenes (W1) and 5 mismatch correction scenarios (W2).

| Parameter | Value |
|-----------|-------|
| Image size (H x W x L) | 256 x 256 x 28 |
| Measurement size | 256 x 310 |
| Compression ratio | 28:1 |
| Spectral bands | 28 |
| Dispersion curve (nominal) | delta(l) = 2.0*l + 0.0*l^2, axis=0° |
| Mask type | Float-valued coded aperture [0.007, 1.0] |
| Data range | [0, ~0.91] |

## Standard dataset

- Name: TSA Simulation Benchmark (10 scenes)
- Source: MST (CVPR 2022) benchmark suite
- Scenes: scene01–scene10, each 256x256x28, float32
- Format: MATLAB `.mat`, key: `img`; mask: `mask.mat` (256x256), `mask_3d_shift.mat` (256x310x28)
- Location: `MST-main/datasets/TSA_simu_data/`

| Scene | Shape | Value range | Content |
|-------|-------|------------|---------|
| scene01 | (256,256,28) | [0.004, 0.913] | Natural scene |
| scene02 | (256,256,28) | [0.002, 0.627] | Natural scene |
| scene03 | (256,256,28) | [0.004, 0.748] | Natural scene |
| scene04 | (256,256,28) | [0.002, 0.885] | Natural scene |
| scene05 | (256,256,28) | [0.003, 0.905] | Natural scene |
| scene06 | (256,256,28) | [0.001, 1.058] | Natural scene |
| scene07 | (256,256,28) | [0.008, 0.534] | Natural scene |
| scene08 | (256,256,28) | [0.000, 1.053] | Natural scene |
| scene09 | (256,256,28) | [0.003, 0.844] | Natural scene |
| scene10 | (256,256,28) | [0.003, 0.934] | Natural scene |

## PWM pipeline flowchart (mandatory)

```
x (world)
  ↓
SourceNode: photon_source — illumination (strength=1.0)
  ↓
Element 1 (subrole=encoding): mask_transform — affine-warped coded aperture (dx,dy,θ)
  ↓
Element 2 (subrole=encoding): parametric_dispersion — Δu(l)=a1·l+a2·l², axis angle α
  ↓
Element 3 (subrole=encoding): frame_integration — sums along spectral axis (28→1)
  ↓
Element 4 (subrole=transport): psf_blur — Gaussian PSF convolution (σ)
  ↓
SensorNode: photon_sensor — QE=0.9, gain=1.0
  ↓
NoiseNode: poisson_read_sensor — Poisson shot noise (gain=1000) + Gaussian read noise (σ=5.0)
  ↓
y
```

## Element inventory & mismatch parameters

| # | node_id | primitive_id | subrole | learnable params | mismatch knob | bounds | prior |
|---|---------|-------------|---------|-----------------|---------------|--------|-------|
| 1 | source | photon_source | source | strength=1.0 | — | — | — |
| 2 | mask | mask_transform | encoding | mask (256,256), float [0.007,1.0] | mask_dx, mask_dy, mask_theta | [-5,5]px, [-5,5]px, [-3°,3°] | uniform |
| 3 | disperse | parametric_dispersion | encoding | a1=2.0, a2=0.0, α=0° | disp_a1, disp_a2, disp_alpha | [1.8,2.3], [-0.01,0.01], [-5°,5°] | normal |
| 4 | integrate | frame_integration | encoding | axis=-1, L=28 | — | — | — |
| 5 | psf | psf_blur | transport | sigma=0 | psf_sigma | [0, 3] px | half-normal |
| 6 | sensor | photon_sensor | sensor | QE=0.9, gain=1.0 | qe_drift, gain | [0.8, 1.0], [0.9, 1.1] | normal |
| 7 | noise | poisson_read_sensor | noise | photon_gain=1000, read_σ=5.0 | — | — | — |

## Node-by-node trace (one sample)

Trace of intermediate tensor states at each node for one representative sample from scene01, showing the full 8-stage parametric pipeline.

| stage | node_id | output shape | dtype | range [min, max] | artifact_path |
|-------|---------|-------------|-------|-------------------|---------------|
| 0 | input_x | (256, 256, 28) | float32 | [0.0043, 0.9133] | `artifacts/trace/00_input_x.npy` |
| 1 | mask_warped | (256, 256) | float32 | [0.0000, 1.0000] | `artifacts/trace/01_mask_warped.npy` |
| 2 | masked | (256, 256, 28) | float32 | [0.0000, 0.9040] | `artifacts/trace/02_masked.npy` |
| 3 | dispersed | (256, 310, 28) | float32 | [0.0000, 0.9040] | `artifacts/trace/03_dispersed.npy` |
| 4 | integrated | (256, 310) | float32 | [0.0002, 7.4487] | `artifacts/trace/04_integrated.npy` |
| 5 | psf_blurred | (256, 310) | float32 | [0.0003, 7.2541] | `artifacts/trace/05_psf_blurred.npy` |
| 6 | noisy_y | (256, 310) | float32 | [-0.0250, 7.3012] | `artifacts/trace/06_noisy_y.npy` |
| 7 | recon_gaptv | (256, 256, 28) | float32 | [0.0000, 1.0000] | `artifacts/trace/07_recon_gaptv.npy` |

PNG visualizations saved at: `artifacts/trace/png/`

## Workflow W1: Prompt-driven simulation + reconstruction

- **Prompt used:** `"Reconstruct 10-scene CASSI benchmark with 4 solvers: GAP-TV, HDNet, MST-S, MST-L"`
- **ExperimentSpec summary:**
  - modality: cassi
  - mode: reconstruct from real benchmark data
  - solvers: GAP-TV, HDNet (CVPR 2022), MST-S (CVPR 2022), MST-L (CVPR 2022)
  - dataset: 10 scenes (scene01–scene10), 256x256x28, step=2 dispersion
- **Mode S results (real measurement):**
  - y shape: (256, 310) per scene — spectral dispersion widens W from 256 to 310
  - Mask: (256, 256) float-valued, shifted to (256, 310, 28) for 3D encoding
- **Mode I results (reconstruct x_hat):**

  **PSNR (dB) — 4-Solver Comparison across 10 Scenes**

  | Scene | GAP-TV | HDNet | MST-S | MST-L |
  |-------|--------|-------|-------|-------|
  | scene01 | 15.41 | 35.17 | 34.78 | 35.42 |
  | scene02 | 15.33 | 35.73 | 34.44 | 35.90 |
  | scene03 | 14.42 | 36.13 | 33.83 | 34.92 |
  | scene04 | 15.86 | 42.78 | 42.11 | 42.24 |
  | scene05 | 14.53 | 32.72 | 31.78 | 32.50 |
  | scene06 | 14.77 | 34.53 | 33.77 | 34.74 |
  | scene07 | 14.41 | 33.70 | 32.38 | 33.44 |
  | scene08 | 15.07 | 32.49 | 31.87 | 32.91 |
  | scene09 | 14.42 | 34.93 | 34.14 | 35.05 |
  | scene10 | 15.02 | 32.39 | 31.87 | 32.74 |
  | **Average** | **14.92** | **35.06** | **34.10** | **34.99** |

  **SSIM — 4-Solver Comparison across 10 Scenes**

  | Scene | GAP-TV | HDNet | MST-S | MST-L |
  |-------|--------|-------|-------|-------|
  | scene01 | 0.1917 | 0.9358 | 0.9295 | 0.9419 |
  | scene02 | 0.1844 | 0.9421 | 0.9234 | 0.9451 |
  | scene03 | 0.1711 | 0.9421 | 0.9273 | 0.9480 |
  | scene04 | 0.2389 | 0.9764 | 0.9693 | 0.9751 |
  | scene05 | 0.1793 | 0.9457 | 0.9271 | 0.9448 |
  | scene06 | 0.2131 | 0.9542 | 0.9407 | 0.9541 |
  | scene07 | 0.1685 | 0.9232 | 0.9057 | 0.9221 |
  | scene08 | 0.2224 | 0.9467 | 0.9362 | 0.9510 |
  | scene09 | 0.1658 | 0.9409 | 0.9273 | 0.9375 |
  | scene10 | 0.2107 | 0.9441 | 0.9286 | 0.9459 |
  | **Average** | **0.1946** | **0.9451** | **0.9315** | **0.9466** |

- **Dataset metrics (best solver — HDNet, 10-scene average):**

  | Metric | Value |
  |--------|-------|
  | PSNR   | 35.06 dB |
  | SSIM   | 0.9451 |

GAP-TV provides a classical baseline averaging 14.92 dB across all 10 scenes. The low PSNR reflects the extreme 28:1 spectral compression ratio — far harder than temporal 8:1 in CACTI. HDNet (CVPR 2022) achieves 35.06 dB using a learned spectral prior with frequency-domain learning. MST-S (CVPR 2022, small transformer) reaches 34.10 dB via mask-guided spectral-wise self-attention. MST-L (CVPR 2022, large transformer) achieves 34.99 dB with deeper attention blocks. Both HDNet and MST-L perform comparably at ~35 dB.

Note: Deep solvers use the MST evaluation protocol — measurement is generated via shift+sum with step=2, then unfolded via shift_back. GAP-TV uses the same measurement generation but reconstructs via iterative GAP with TV denoising.

## Workflow W2: Operator correction mode (measured y + operator A)

- **Operator definition:**
  - A_definition: callable
  - A_extraction_method: provided
  - Operator chain: y = PSF(sum_l(shift(warp(mask,dx,dy,theta) * x, a1,a2,alpha))) + noise
  - A_sha256: 1a9dbae46d79a133
  - Linearity: linear
  - Notes (if linearized): N/A
- **Mismatch specification:**
  - Mismatch type: synthetic_injected
  - Parameters perturbed: 5 realistic calibration failure modes — mask translation (dx,dy), mask rotation (theta), dispersion slope (a1), dispersion axis angle (alpha), PSF blur (sigma)
  - Noise model: Poisson shot noise (gain=1000) + Gaussian read noise (sigma=5.0)
  - Description: Five independent mismatch scenarios covering all real SD-CASSI calibration failure modes, each with Poisson+read noise for physical realism
- **Mode C fit results:**
  - Correction family: Pre
  - Grid search calibration per scenario

  - NLL before correction: 55395964.0 (W2d, worst-case scenario)
  - NLL after correction: 483205.2 (W2d, corrected)

  **5-Scenario Mismatch Summary**

  | Scenario | Mismatch | Injected | Recovered | NLL decrease | PSNR uncorr | PSNR corr | PSNR A₀ vs A' delta | SAM uncorr | SAM corr |
  |----------|----------|----------|-----------|-------------|------------|----------|---------------------|-----------|---------|
  | W2a | Mask (dx,dy) | (2,1) px | (2,1) px | 98.7% | 15.24 dB | 15.36 dB | +0.12 dB | 45.41° | 45.15° |
  | W2b | Mask rotation | 1.0° | 1.0° | 97.6% | 15.23 dB | 19.00 dB | +3.77 dB | 46.01° | 38.55° |
  | W2c | Disp slope | a1=2.15 | a1=2.15 | 93.9% | 15.27 dB | 20.76 dB | +5.49 dB | 45.61° | 39.31° |
  | W2d | Disp axis | alpha=2° | alpha=2° | 99.1% | 15.01 dB | 22.05 dB | +7.04 dB | 47.49° | 36.08° |
  | W2e | PSF blur | sigma=1.5 | sigma=1.5 | 97.9% | 15.25 dB | 15.25 dB | +0.00 dB | 45.78° | 45.45° |

- **Mode I recon using corrected operator A':**

  **W2a — Mask Translation (dx=2, dy=1)**

  | Metric | A₀ (uncorrected) | A' (corrected) | Delta |
  |--------|-------------------|----------------|-------|
  | PSNR   | 15.24 dB | 15.36 dB | +0.12 dB |
  | SSIM   | 0.1462 | 0.1442 | -0.0020 |
  | SAM    | 45.41° | 45.15° | -0.26° |
  | Residual | 0.1755 | 0.0203 | -0.1552 |

  Grid search: 11x7=77 trials over dx in [-5,5], dy in [-3,3]. Exact recovery of injected shift (2,1) px. Modest PSNR gain because mask translation preserves most spectral structure.

  **W2b — Mask Rotation (theta=1.0°)**

  | Metric | A₀ (uncorrected) | A' (corrected) | Delta |
  |--------|-------------------|----------------|-------|
  | PSNR   | 15.23 dB | 19.00 dB | +3.77 dB |
  | SSIM   | 0.1394 | 0.2056 | +0.0662 |
  | SAM    | 46.01° | 38.55° | -7.45° |
  | Residual | 0.1313 | 0.0204 | -0.1109 |

  Grid search: 13 trials over theta in [-3°,3°] step 0.5°. Exact recovery of 1.0° rotation. Significant PSNR gain because rotation decorrelates the mask from the correct pixel grid.

  **W2c — Dispersion Slope (a1=2.15 vs nominal 2.0)**

  | Metric | A₀ (uncorrected) | A' (corrected) | Delta |
  |--------|-------------------|----------------|-------|
  | PSNR   | 15.27 dB | 20.76 dB | +5.49 dB |
  | SSIM   | 0.1493 | 0.3447 | +0.1954 |
  | SAM    | 45.61° | 39.31° | -6.30° |
  | Residual | 0.0828 | 0.0205 | -0.0623 |

  Grid search: 22 trials over a1 in [1.8, 2.3] step 0.025. Exact recovery of a1=2.15. Large PSNR gain because dispersion slope error causes cross-band spectral leakage during reconstruction.

  **W2d — Dispersion Axis Angle (alpha=2°)**

  | Metric | A₀ (uncorrected) | A' (corrected) | Delta |
  |--------|-------------------|----------------|-------|
  | PSNR   | 15.01 dB | 22.05 dB | +7.04 dB |
  | SSIM   | 0.1307 | 0.4120 | +0.2813 |
  | SAM    | 47.49° | 36.08° | -11.41° |
  | Residual | 0.2195 | 0.0205 | -0.1990 |

  Grid search: 11 trials over alpha in [-5°,5°] step 1°. Exact recovery of alpha=2°. Strongest correction effect — axis misalignment causes 2D spectral smearing that completely confounds the 1D reconstruction model.

  **W2e — PSF Blur (sigma=1.5 px)**

  | Metric | A₀ (uncorrected) | A' (corrected) | Delta |
  |--------|-------------------|----------------|-------|
  | PSNR   | 15.25 dB | 15.25 dB | +0.00 dB |
  | SSIM   | 0.1313 | 0.1342 | +0.0029 |
  | SAM    | 45.78° | 45.45° | -0.34° |
  | Residual | 0.1454 | 0.0209 | -0.1245 |

  Grid search: 13 trials over sigma in [0, 3] step 0.25. Exact recovery of sigma=1.5 px (NLL decrease 97.9%). PSNR unchanged because Wiener deblurring amplifies noise at high frequencies, offsetting blur correction. The residual improvement (0.1454 to 0.0209) confirms the PSF parameter is correctly identified.

All 5 scenarios achieve exact parameter recovery via NLL grid search (93.9–99.1% NLL decrease). Dispersion mismatch (W2c, W2d) produces the largest reconstruction gains (+5.49 to +7.04 dB) because dispersion errors cause spectral cross-talk. Mask transform errors (W2a, W2b) have moderate impact. PSF blur correction is limited by deconvolution noise amplification. (All W2 evaluated on scene01 with GAP-TV.)

## Test results summary

### Quick gate (pass/fail)

- [x] W1 simulate: PASS (10 scenes loaded from real benchmark data)
- [x] W1 reconstruct GAP-TV (avg PSNR >= 10 dB): PASS (14.92 dB)
- [x] W1 reconstruct HDNet (avg PSNR >= 30 dB): PASS (35.06 dB)
- [x] W1 reconstruct MST-S (avg PSNR >= 30 dB): PASS (34.10 dB)
- [x] W1 reconstruct MST-L (avg PSNR >= 30 dB): PASS (34.99 dB)
- [x] W2 operator correction W2a (NLL decreases): PASS (98.7% decrease)
- [x] W2 operator correction W2b (NLL decreases): PASS (97.6% decrease)
- [x] W2 operator correction W2c (NLL decreases): PASS (93.9% decrease)
- [x] W2 operator correction W2d (NLL decreases): PASS (99.1% decrease)
- [x] W2 operator correction W2e (NLL decreases): PASS (97.9% decrease)
- [x] W2 best corrected recon (beats uncorrected PSNR): PASS (+7.04 dB, W2d)
- [x] Report contract (flowchart + all headings): PASS
- [x] Node-by-node trace saved (>= 3 stages): PASS (8 stages)
- [x] RunBundle saved (with trace PNGs): PASS

### Full metrics

| Check | Metric | Value | Threshold | Status |
|-------|--------|-------|-----------|--------|
| W1 GAP-TV avg PSNR | psnr | 14.92 dB | >= 10 dB | PASS |
| W1 HDNet avg PSNR | psnr | 35.06 dB | >= 30 dB | PASS |
| W1 MST-S avg PSNR | psnr | 34.10 dB | >= 30 dB | PASS |
| W1 MST-L avg PSNR | psnr | 34.99 dB | >= 30 dB | PASS |
| W1 Best avg SSIM | ssim | 0.9466 | >= 0.90 | PASS |
| W2a NLL decrease | nll_decrease_pct | 98.7% | >= 5% | PASS |
| W2b NLL decrease | nll_decrease_pct | 97.6% | >= 5% | PASS |
| W2c NLL decrease | nll_decrease_pct | 93.9% | >= 5% | PASS |
| W2d NLL decrease | nll_decrease_pct | 99.1% | >= 5% | PASS |
| W2e NLL decrease | nll_decrease_pct | 97.9% | >= 5% | PASS |
| W2d PSNR gain | psnr_delta | +7.04 dB | > 0 | PASS |
| Trace stages | n_stages | 8 | >= 3 | PASS |
| Trace PNGs | n_pngs | 8 | >= 3 | PASS |
| Total scenes | n_scenes | 10 | >= 10 | PASS |

## Reproducibility

- Seed: 42
- PWM version: ae5106be6a89
- Python version: 3.13.9
- Key package versions: numpy=2.3.5, scipy=1.16.3, torch=2.10.0+cu128
- Platform: Linux x86_64
- External codebases: MST (CVPR 2022), HDNet (CVPR 2022), PnP-CASSI (GAP-TV utilities)
- Deterministic reproduction command:
  ```bash
  pwm_cli run --modality cassi --seed 42 --mode simulate,invert,calibrate
  # Full 10-scene benchmark:
  PYTHONPATH="$PWD:$PWD/packages/pwm_core" python scripts/run_cassi_benchmark.py
  ```
- Output hash (mask): 1a9dbae46d79a133
- Output hash (mask_3d_shift): 8673a0e07958f984

## Saved artifacts

- RunBundle: `runs/run_cassi_benchmark_1d6d9d00/`
- Report: `pwm/reports/cassi.md`
- Scoreboard: `pwm/reports/scoreboard.yaml`
- Node trace (.npy): `runs/run_cassi_benchmark_1d6d9d00/artifacts/trace/*.npy`
- Node trace (.png): `runs/run_cassi_benchmark_1d6d9d00/artifacts/trace/png/*.png`
- W2 operator metadata: `runs/run_cassi_benchmark_1d6d9d00/artifacts/w2_operator_meta.json`
- Per-scene artifacts: `runs/run_cassi_benchmark_1d6d9d00/artifacts/{scene}/`
- Full results JSON: `runs/run_cassi_benchmark_1d6d9d00/cassi_benchmark_results.json`

## Next actions

- Add joint multi-parameter grid search (simultaneous dx+dy+theta+a1+alpha)
- Add CST (ECCV 2022) and DAUHST (NeurIPS 2022) solvers to W1 comparison
- Investigate GPU-accelerated differentiable forward model for gradient-based calibration
- Proceed to next modality benchmark
