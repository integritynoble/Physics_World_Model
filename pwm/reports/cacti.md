# Coded Aperture Compressive Temporal Imaging (CACTI) — PWM Report

| Field | Value |
|-------|-------|
| Modality ID | `cacti` |
| Category | compressive |
| Dataset | Grayscale Benchmark — kobe32 (256x256, 32 frames, 8:1 compression) |
| Date | 2026-02-12 |
| PWM version | `fdab88c35ac3` |
| Author | integritynoble |

## Modality overview

- Modality key: `cacti`
- Category: compressive
- Forward model: y = sum_t(Mask_t * x_t) + n, where x is a 3D video cube (H, W, T), Mask_t is a time-varying binary mask, summation collapses temporal axis, and n ~ N(0, sigma^2 I)
- Default solver: GAP-TV (classical), with PnP-FFDNet, ELP-Unfolding, EfficientSCI as deep alternatives
- Pipeline linearity: linear

CACTI (also known as Snapshot Compressive Imaging for video) captures T video frames in a single 2D snapshot using time-varying coded aperture masks. Each video frame is modulated by a different binary mask pattern, and all masked frames are summed on the detector. Reconstruction recovers the 3D video cube from the 2D measurement. This benchmark evaluates 4 solvers spanning classical optimization, plug-and-play denoisers, and deep unfolding networks.

| Parameter | Value |
|-----------|-------|
| Image size (H x W) | 256 x 256 |
| Temporal frames per measurement | 8 |
| Coded measurements | 4 |
| Total frames | 32 |
| Compression ratio | 8:1 |
| Mask type | Binary random (from dataset) |
| Data range | [0, 255] |

## Standard dataset

- Name: Grayscale SCI Video Benchmark (kobe32)
- Source: Standard CACTI benchmark suite (Kobe Bryant basketball footage, 32 frames at 256x256)
- Size: meas (256,256,4), mask (256,256,8), orig (256,256,32)
- Format: MATLAB `.mat`, keys: `meas`, `mask`, `orig`
- Location: `PnP-SCI_python-master/dataset/cacti/grayscale_benchmark/kobe32_cacti.mat`

The kobe32 scene is a standard benchmark for video SCI reconstruction containing fast motion and fine textures. The dataset provides 4 coded 2D measurements, each encoding 8 consecutive frames via binary masks, for a total of 32 ground-truth frames. Six scenes are available (kobe32, crash32, aerial32, traffic48, runner40, drop40); this report uses kobe32 as the primary benchmark.

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

- **Prompt used:** `"Reconstruct kobe32 CACTI benchmark with 4 solvers: GAP-TV, PnP-FFDNet, ELP-Unfolding, EfficientSCI"`
- **ExperimentSpec summary:**
  - modality: cacti
  - mode: reconstruct from real benchmark data
  - solvers: GAP-TV, PnP-FFDNet, ELP-Unfolding, EfficientSCI
  - dataset: kobe32 (256x256x32, 4 coded measurements, 8:1 compression)
- **Mode S results (real measurement):**
  - y shape: (256, 256, 4) — 4 coded snapshot measurements
  - y range: [0.00, 1558.14]
  - Mask: (256, 256, 8) binary
- **Mode I results (reconstruct x_hat):**

  **4-Solver Comparison Table (kobe32, 32 frames)**

  | Solver | Type | PSNR (dB) | SSIM | Time (s) |
  |--------|------|-----------|------|----------|
  | GAP-TV | Classical optimization | 24.00 | 0.7461 | 13.1 |
  | PnP-FFDNet | Plug-and-play deep denoiser | 30.33 | 0.9253 | 8.7 |
  | ELP-Unfolding | Deep unfolding (ECCV 2022) | 34.08 | 0.9644 | 1.9 |
  | EfficientSCI | End-to-end (CVPR 2023) | 35.76 | 0.9758 | 1.6 |

- **Dataset metrics (best solver — EfficientSCI):**

  | Metric | Value |
  |--------|-------|
  | PSNR   | 35.76 dB |
  | SSIM   | 0.9758 |

GAP-TV provides a classical baseline at 24.00 dB. PnP-FFDNet improves +6.33 dB by replacing the TV denoiser with a learned FFDNet prior. Deep unfolding (ELP-Unfolding, ECCV 2022) reaches 34.08 dB with 8+5 learned ADMM iterations using Vision Transformers. EfficientSCI (CVPR 2023) achieves the best result at 35.76 dB via an end-to-end architecture with spatial-temporal attention.

Note: ELP-Unfolding and EfficientSCI use the dataset's mask for forward simulation. EfficientSCI uses its own trained mask and re-simulates measurements for fair comparison.

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

The 2-pixel mask shift degrades reconstruction by 8.24 dB. The grid search over integer shifts exactly recovers the injected perturbation, restoring full GAP-TV performance.

## Test results summary

### Quick gate (pass/fail)

- [x] W1 simulate: PASS (real benchmark data loaded)
- [x] W1 reconstruct GAP-TV (PSNR >= 20 dB): PASS (24.00 dB)
- [x] W1 reconstruct PnP-FFDNet (PSNR >= 28 dB): PASS (30.33 dB)
- [x] W1 reconstruct ELP-Unfolding (PSNR >= 32 dB): PASS (34.08 dB)
- [x] W1 reconstruct EfficientSCI (PSNR >= 32 dB): PASS (35.76 dB)
- [x] W2 operator correction (NLL decreases): PASS (100.0% decrease)
- [x] W2 corrected recon (beats uncorrected): PASS (+8.24 dB)
- [x] Report contract (flowchart + all headings): PASS
- [x] Node-by-node trace saved (>= 3 stages): PASS (4 stages)
- [x] RunBundle saved (with trace PNGs): PASS

### Full metrics

| Check | Metric | Value | Threshold | Status |
|-------|--------|-------|-----------|--------|
| W1 GAP-TV PSNR | psnr | 24.00 dB | >= 20 dB | PASS |
| W1 PnP-FFDNet PSNR | psnr | 30.33 dB | >= 28 dB | PASS |
| W1 ELP-Unfolding PSNR | psnr | 34.08 dB | >= 32 dB | PASS |
| W1 EfficientSCI PSNR | psnr | 35.76 dB | >= 32 dB | PASS |
| W1 Best SSIM | ssim | 0.9758 | >= 0.90 | PASS |
| W2 NLL decrease | nll_decrease_pct | 100.0% | >= 5% | PASS |
| W2 PSNR gain | psnr_delta | +8.24 dB | > 0 | PASS |
| W2 SSIM gain | ssim_delta | +0.4514 | > 0 | PASS |
| Trace stages | n_stages | 4 | >= 3 | PASS |
| Trace PNGs | n_pngs | 4 | >= 3 | PASS |
| W1 GAP-TV wall time | w1_gaptv_seconds | 13.1 s | — | info |
| W1 EfficientSCI wall time | w1_esci_seconds | 1.6 s | — | info |

## Reproducibility

- Seed: 42
- PWM version: fdab88c35ac3
- Python version: 3.13.9
- Key package versions: numpy=2.3.5, scipy=1.16.3, torch=2.10.0+cu128
- Platform: Linux x86_64
- External codebases: PnP-SCI, ELP-Unfolding (ECCV 2022), EfficientSCI (CVPR 2023)
- Deterministic reproduction command:
  ```bash
  pwm_cli run --modality cacti --seed 42 --mode simulate,invert,calibrate
  ```
- Output hash (orig): 67745be99e30074a
- Output hash (meas): 917718cce2e61c46
- Output hash (mask): 94d64c2f732b2072

## Saved artifacts

- RunBundle: `runs/run_cacti_benchmark_a68c6589/`
- Report: `pwm/reports/cacti.md`
- Scoreboard: `pwm/reports/scoreboard.yaml`
- Node trace (.npy): `runs/run_cacti_benchmark_a68c6589/artifacts/trace/*.npy`
- Node trace (.png): `runs/run_cacti_benchmark_a68c6589/artifacts/trace/png/*.png`
- W2 operator metadata: `runs/run_cacti_benchmark_a68c6589/artifacts/w2_operator_meta.json`
- Ground truth: `runs/run_cacti_benchmark_a68c6589/artifacts/x_true.npy`
- Measurement: `runs/run_cacti_benchmark_a68c6589/artifacts/meas.npy`
- W1 GAP-TV reconstruction: `runs/run_cacti_benchmark_a68c6589/artifacts/x_hat_gap_tv.npy`
- W1 PnP-FFDNet reconstruction: `runs/run_cacti_benchmark_a68c6589/artifacts/x_hat_pnp_ffdnet.npy`
- W1 ELP-Unfolding reconstruction: `runs/run_cacti_benchmark_a68c6589/artifacts/x_hat_elp_unfolding.npy`
- W1 EfficientSCI reconstruction: `runs/run_cacti_benchmark_a68c6589/artifacts/x_hat_efficientsci.npy`
- W2 reconstructions: `runs/run_cacti_benchmark_a68c6589/artifacts/x_hat_w2_uncorrected.npy`, `x_hat_w2_corrected.npy`
- Full results JSON: `runs/run_cacti_benchmark_a68c6589/cacti_benchmark_results.json`

## Next actions

- Extend benchmark to all 6 scenes (crash32, aerial32, traffic48, runner40, drop40)
- Add FastDVDNet solver to the comparison
- Investigate learned mask optimization for improved compression
- Proceed to next modality benchmark
