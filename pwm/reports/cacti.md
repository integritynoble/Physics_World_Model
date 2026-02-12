# Coded Aperture Compressive Temporal Imaging (CACTI) — PWM Report

| Field | Value |
|-------|-------|
| Modality ID | `cacti` |
| Category | compressive |
| Dataset | Grayscale SCI Video Benchmark — 6 scenes (256x256, 8:1 compression) |
| Date | 2026-02-12 |
| PWM version | `14090be` |
| Author | integritynoble |

## Modality overview

- Modality key: `cacti`
- Category: compressive
- Forward model: y = sum_t(Mask_t * x_t) + n, where x is a 3D video cube (H, W, T), Mask_t is a time-varying binary mask, summation collapses temporal axis, and n ~ N(0, sigma^2 I)
- Default solver: GAP-TV (classical), with PnP-FFDNet, ELP-Unfolding, EfficientSCI as deep alternatives
- Pipeline linearity: linear

CACTI (also known as Snapshot Compressive Imaging for video) captures T video frames in a single 2D snapshot using time-varying coded aperture masks. Each video frame is modulated by a different binary mask pattern, and all masked frames are summed on the detector. Reconstruction recovers the 3D video cube from the 2D measurement. This benchmark evaluates 4 solvers spanning classical optimization, plug-and-play denoisers, and deep unfolding networks across 6 standard test scenes.

| Parameter | Value |
|-----------|-------|
| Image size (H x W) | 256 x 256 |
| Temporal frames per measurement | 8 |
| Compression ratio | 8:1 |
| Mask type | Binary random (from dataset) |
| Data range | [0, 255] |

## Standard dataset

- Name: Grayscale SCI Video Benchmark (6 scenes)
- Source: Standard CACTI benchmark suite
- Scenes: kobe32 (32 frames), crash32 (32 frames), aerial32 (32 frames), traffic48 (48 frames), runner40 (40 frames), drop40 (40 frames)
- Format: MATLAB `.mat`, keys: `meas`, `mask`, `orig`
- Location: `PnP-SCI_python-master/dataset/cacti/grayscale_benchmark/`

| Scene | Total frames | Coded measurements | Content |
|-------|-------------|-------------------|---------|
| kobe32 | 32 | 4 | Basketball footage, fast motion |
| crash32 | 32 | 4 | Car crash, sudden deformation |
| aerial32 | 32 | 4 | Aerial view, fine texture |
| traffic48 | 48 | 6 | Traffic intersection, multi-object motion |
| runner40 | 40 | 5 | Running athlete, periodic motion |
| drop40 | 40 | 5 | Water drop, fluid dynamics |

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
NoiseNode: poisson_gaussian_sensor — additive Gaussian noise
  ↓
y
```

## Element inventory & mismatch parameters

| # | node_id | primitive_id | subrole | learnable params | mismatch knob | bounds | prior |
|---|---------|-------------|---------|-----------------|---------------|--------|-------|
| 1 | source | photon_source | source | strength=1.0 | — | — | — |
| 2 | mask | temporal_mask | encoding | binary mask (256,256,8) | mask_shift (vertical) | [-5, 5] px | uniform |
| 3 | sensor | photon_sensor | sensor | QE=0.9, gain=1.0 | qe_drift, gain | [0.8, 1.0], [0.9, 1.1] | normal |
| 4 | noise | poisson_gaussian_sensor | noise | read_sigma | — | — | — |

## Node-by-node trace (one sample)

Trace of intermediate tensor states at each node for one representative sample from kobe32 (first coded frame block, frames 0-7).

| stage | node_id | output shape | dtype | range [min, max] | artifact_path |
|-------|---------|-------------|-------|-------------------|---------------|
| 0 | input_x | (256, 256, 8) | float32 | [0.00, 255.00] | `artifacts/trace/00_input_x.npy` |
| 1 | masked | (256, 256, 8) | float32 | [0.00, 255.00] | `artifacts/trace/01_masked.npy` |
| 2 | measurement | (256, 256, 1) | float32 | [0.00, 1558.14] | `artifacts/trace/02_measurement.npy` |
| 3 | recon_gaptv | (256, 256, 8) | float32 | [0.00, 255.00] | `artifacts/trace/03_recon_gaptv.npy` |

PNG visualizations saved at: `artifacts/trace/png/`

## Workflow W1: Prompt-driven simulation + reconstruction

- **Prompt used:** `"Reconstruct 6-scene CACTI benchmark with 4 solvers: GAP-TV, PnP-FFDNet, ELP-Unfolding, EfficientSCI"`
- **ExperimentSpec summary:**
  - modality: cacti
  - mode: reconstruct from real benchmark data
  - solvers: GAP-TV, PnP-FFDNet, ELP-Unfolding, EfficientSCI
  - dataset: 6 scenes (kobe32, crash32, aerial32, traffic48, runner40, drop40), 256x256, 8:1 compression
- **Mode S results (real measurement):**
  - y shape: (256, 256, N_coded) — N_coded varies per scene (4-6 coded snapshots)
  - Mask: (256, 256, 8) binary
- **Mode I results (reconstruct x_hat):**

  **PSNR (dB) — 4-Solver Comparison across 6 Scenes**

  | Scene | GAP-TV | PnP-FFDNet | ELP-Unfolding | EfficientSCI |
  |-------|--------|------------|---------------|-------------|
  | kobe32 | 24.00 | 30.33 | 34.08 | 35.76 |
  | crash32 | 25.40 | 24.69 | 29.39 | 31.12 |
  | aerial32 | 26.13 | 24.36 | 30.54 | 31.50 |
  | traffic48 | 21.06 | 23.88 | 31.34 | 32.29 |
  | runner40 | 28.70 | 32.97 | 38.17 | 41.89 |
  | drop40 | 34.42 | 39.91 | 40.09 | 45.10 |
  | **Average** | **26.62** | **29.36** | **33.94** | **36.28** |

  **SSIM — 4-Solver Comparison across 6 Scenes**

  | Scene | GAP-TV | PnP-FFDNet | ELP-Unfolding | EfficientSCI |
  |-------|--------|------------|---------------|-------------|
  | kobe32 | 0.7461 | 0.9253 | 0.9644 | 0.9758 |
  | crash32 | 0.8649 | 0.8332 | 0.9537 | 0.9726 |
  | aerial32 | 0.8510 | 0.8200 | 0.9398 | 0.9542 |
  | traffic48 | 0.7063 | 0.8299 | 0.9623 | 0.9691 |
  | runner40 | 0.8908 | 0.9357 | 0.9744 | 0.9868 |
  | drop40 | 0.9654 | 0.9863 | 0.9798 | 0.9950 |
  | **Average** | **0.8374** | **0.8884** | **0.9624** | **0.9756** |

- **Dataset metrics (best solver — EfficientSCI, 6-scene average):**

  | Metric | Value |
  |--------|-------|
  | PSNR   | 36.28 dB |
  | SSIM   | 0.9756 |

GAP-TV provides a classical baseline averaging 26.62 dB across all 6 scenes. PnP-FFDNet improves to 29.36 dB by replacing the TV denoiser with a learned FFDNet prior. Deep unfolding (ELP-Unfolding, ECCV 2022) reaches 33.94 dB with 8+5 learned ADMM iterations using Vision Transformers. EfficientSCI (CVPR 2023) achieves the best result at 36.28 dB average via an end-to-end architecture with spatial-temporal attention.

Note: GAP-TV and PnP-FFDNet use the dataset's original mask. ELP-Unfolding uses the dataset's mask for forward simulation. EfficientSCI uses its own trained mask and re-simulates measurements for fair comparison.

## Workflow W2: Operator correction mode (measured y + operator A)

- **Operator definition:**
  - A_definition: callable
  - A_extraction_method: provided
  - Operator chain: y = sum_t(mask_t * x_t) — binary mask modulation + temporal sum
  - A_sha256: 94d64c2f732b2072
  - Linearity: linear
  - Notes (if linearized): N/A
- **Mismatch specification:**
  - Mismatch type: synthetic_injected
  - Parameters perturbed: mask-detector vertical shift (2 pixels)
  - Description: Mask-detector misalignment — the binary mask array is vertically shifted by 2 pixels relative to its nominal position, simulating a physical registration error between the coded aperture mask and the detector array
- **Mode C fit results:**
  - Correction family: Pre
  - Parameters fitted: 1 parameter (vertical shift in pixels) via grid search over [-5, +5]
  - Best fitted shift: 2 px (exact recovery)
  - NLL before correction: 98828576.0
  - NLL after correction: 0.0
  - NLL decrease: 98828576.0 (100.0%)
- **Mode I recon using corrected operator A':**

  | Metric | A₀ (uncorrected) | A' (corrected) | Delta |
  |--------|-------------------|----------------|-------|
  | PSNR   | 15.76 dB | 24.00 dB | +8.24 dB |
  | SSIM   | 0.2944 | 0.7458 | +0.4514 |

The 2-pixel mask shift degrades reconstruction by 8.24 dB. The grid search over integer shifts exactly recovers the injected perturbation, restoring full GAP-TV performance. (W2 evaluated on kobe32.)

## Test results summary

### Quick gate (pass/fail)

- [x] W1 simulate: PASS (6 scenes loaded from real benchmark data)
- [x] W1 reconstruct GAP-TV (avg PSNR >= 20 dB): PASS (26.62 dB)
- [x] W1 reconstruct PnP-FFDNet (avg PSNR >= 25 dB): PASS (29.36 dB)
- [x] W1 reconstruct ELP-Unfolding (avg PSNR >= 30 dB): PASS (33.94 dB)
- [x] W1 reconstruct EfficientSCI (avg PSNR >= 30 dB): PASS (36.28 dB)
- [x] W2 operator correction (NLL decreases): PASS (100.0% decrease)
- [x] W2 corrected recon (beats uncorrected): PASS (+8.24 dB)
- [x] Report contract (flowchart + all headings): PASS
- [x] Node-by-node trace saved (>= 3 stages): PASS (4 stages)
- [x] RunBundle saved (with trace PNGs): PASS

### Full metrics

| Check | Metric | Value | Threshold | Status |
|-------|--------|-------|-----------|--------|
| W1 GAP-TV avg PSNR | psnr | 26.62 dB | >= 20 dB | PASS |
| W1 PnP-FFDNet avg PSNR | psnr | 29.36 dB | >= 25 dB | PASS |
| W1 ELP-Unfolding avg PSNR | psnr | 33.94 dB | >= 30 dB | PASS |
| W1 EfficientSCI avg PSNR | psnr | 36.28 dB | >= 30 dB | PASS |
| W1 Best avg SSIM | ssim | 0.9756 | >= 0.90 | PASS |
| W2 NLL decrease | nll_decrease_pct | 100.0% | >= 5% | PASS |
| W2 PSNR gain | psnr_delta | +8.24 dB | > 0 | PASS |
| W2 SSIM gain | ssim_delta | +0.4514 | > 0 | PASS |
| Trace stages | n_stages | 4 | >= 3 | PASS |
| Trace PNGs | n_pngs | 4 | >= 3 | PASS |
| Total scenes | n_scenes | 6 | >= 6 | PASS |

## Reproducibility

- Seed: 42
- PWM version: 14090be
- Python version: 3.13.9
- Key package versions: numpy=2.3.5, scipy=1.16.3, torch=2.10.0+cu128
- Platform: Linux x86_64
- External codebases: PnP-SCI, ELP-Unfolding (ECCV 2022), EfficientSCI (CVPR 2023)
- Deterministic reproduction command:
  ```bash
  pwm_cli run --modality cacti --seed 42 --mode simulate,invert,calibrate
  # Full 6-scene benchmark:
  PYTHONPATH="$PWD:$PWD/packages/pwm_core" python scripts/run_cacti_benchmark.py
  ```
- Output hash (orig, kobe32): 67745be99e30074a
- Output hash (meas, kobe32): 917718cce2e61c46
- Output hash (mask, kobe32): 94d64c2f732b2072

## Saved artifacts

- RunBundle: `runs/run_cacti_benchmark_c8a40b01/`
- Report: `pwm/reports/cacti.md`
- Scoreboard: `pwm/reports/scoreboard.yaml`
- Node trace (.npy): `runs/run_cacti_benchmark_c8a40b01/artifacts/trace/*.npy`
- Node trace (.png): `runs/run_cacti_benchmark_c8a40b01/artifacts/trace/png/*.png`
- W2 operator metadata: `runs/run_cacti_benchmark_c8a40b01/artifacts/w2_operator_meta.json`
- Per-scene artifacts: `runs/run_cacti_benchmark_c8a40b01/artifacts/{scene}/`
- Full results JSON: `runs/run_cacti_benchmark_c8a40b01/cacti_benchmark_results.json`

## Next actions

- Add FastDVDNet solver to the comparison
- Investigate learned mask optimization for improved compression
- Proceed to next modality benchmark
