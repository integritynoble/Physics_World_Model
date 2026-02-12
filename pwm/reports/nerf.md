# NeRF — PWM Report

| Field | Value |
|-------|-------|
| Modality ID | `nerf` |
| Category | volumetric |
| Dataset | nerf_benchmark (synthetic proxy: phantom) |
| Date | 2026-02-12 |
| PWM version | `7394757` |
| Author | integritynoble |

## Modality overview

- Modality key: `nerf`
- Category: volumetric
- Forward model: y = Sensor(VolumeRendering(GenericSource(x))) + n
- Default solver: pseudo_inverse
- Pipeline linearity: nonlinear

NeRF imaging pipeline with 1 element(s) in the forward chain. Reconstruction uses pseudo_inverse. Tested on a 64x64 synthetic phantom with seed=42.

| Parameter | Value |
|-----------|-------|
| Image size | 16 x 64 x 64 |
| Output shape | 64 x 64 |
| Noise model | Poisson-Gaussian |

## Standard dataset

- Name: nerf_benchmark (synthetic proxy: Gaussian-blob phantom)
- Source: synthetic
- Size: 1 image, 16x64x64
- Registered in `dataset_registry.yaml` as `nerf_benchmark`

For this baseline experiment, a deterministic phantom (seed=42, smooth Gaussian blobs on a 16x64x64 grid) is used.

## PWM pipeline flowchart (mandatory)

```
x (world)
  ↓
SourceNode: generic_source — strength=1.0
  ↓
Element 1 (subrole=transport): volume_rendering_stub — n_views=1, render_mode=mip
  ↓
SensorNode: generic_sensor — gain=1.0
  ↓
NoiseNode: gaussian_sensor_noise — sigma=0.01, seed=0
  ↓
y
```

## Element inventory & mismatch parameters

| # | node_id | primitive_id | subrole | learnable params | mismatch knob | bounds | prior |
|---|---------|-------------|---------|-----------------|---------------|--------|-------|
| 1 | source | generic_source | source | strength=1.0 | — | — | — |
| 2 | volume_render | volume_rendering_stub | transport | n_views=1, render_mode=mip | gain | [0.5, 2.0] | uniform |
| 3 | sensor | generic_sensor | sensor | gain=1.0 | gain | [0.5, 2.0] | uniform |
| 4 | noise | gaussian_sensor_noise | noise | sigma=0.01, seed=0 | — | — | — |

## Node-by-node trace (one sample)

Trace of intermediate tensor states at each node for one representative sample (seed=42).
Each row references a saved artifact in the RunBundle.

| stage | node_id | output shape | dtype | range [min, max] | artifact_path |
|-------|---------|-------------|-------|-------------------|---------------|
| 0 | input x | (16, 64, 64) | float64 | [0.0000, 1.0000] | `artifacts/trace/00_input_x.npy` |
| 1 | source | (16, 64, 64) | float64 | [0.0000, 1.0000] | `artifacts/trace/01_source.npy` |
| 2 | volume_render | (64, 64) | float64 | [0.0000, 1.0000] | `artifacts/trace/02_volume_render.npy` |
| 3 | sensor | (64, 64) | float64 | [0.0000, 1.0000] | `artifacts/trace/03_sensor.npy` |
| 4 | noise (y) | (64, 64) | float64 | [0.0000, 1.0000] | `artifacts/trace/04_noise_y.npy` |

PNG visualizations saved at: `artifacts/trace/png/`

## Workflow W1: Prompt-driven simulation + reconstruction

- **Prompt used:** `"Simulate nerf measurement of a phantom and reconstruct"`
- **ExperimentSpec summary:**
  - modality: nerf
  - mode: simulate -> invert
  - solver: pseudo_inverse
- **Mode S results (simulate y):**
  - y shape: (64, 64)
  - SNR: 26.3 dB
- **Mode I results (reconstruct x_hat):**
  - x_hat shape: (16, 64, 64)
  - Solver: pseudo_inverse
- **Dataset metrics:**

  | Metric | Value |
  |--------|-------|
  | PSNR   | 36.28 dB |
  | SSIM   | 0.9976 |
  | NRMSE  | 0.0166 |

## Workflow W2: Operator correction mode (measured y + operator A)

- **Operator definition:**
  - A_definition: callable
  - A_extraction_method: graph_stripped
  - Operator chain: forward model stripped of noise node
  - A_sha256: c19b264ffc2025da
  - Linearity: nonlinear
  - Notes (if linearized): N/A
- **Mismatch specification:**
  - Mismatch type: synthetic_injected
  - Parameters perturbed: gain drift (1.0 -> 1.3)
  - Description: Synthetic parameter drift injected for calibration testing
- **Mode C fit results:**
  - Correction family: Pre
  - Parameters fitted: 1 parameter via grid search
  - NLL before correction: 82244.3
  - NLL after correction: 2048.1
  - NLL decrease: 97.5%
- **Mode I recon using corrected operator A':**

  | Metric | A₀ (uncorrected) | A' (corrected) | Delta |
  |--------|-------------------|----------------|-------|
  | PSNR   | 36.28 dB | 58.70 dB | +22.42 dB |

## Test results summary

### Quick gate (pass/fail)

- [x] W1 simulate: PASS
- [x] W1 reconstruct (PSNR reported): PASS (36.28 dB)
- [x] W2 operator correction (NLL decrease): PASS (97.5%)
- [x] Report contract (flowchart + all headings): PASS
- [x] Node-by-node trace saved (>= 3 stages): PASS (5 stages)
- [x] RunBundle saved (with trace PNGs): PASS
- [x] Scoreboard updated: PASS

### Full metrics

| Check | Metric | Value | Threshold | Status |
|-------|--------|-------|-----------|--------|
| W1 PSNR | psnr | 36.28 dB | >= 1 dB | PASS |
| W1 SSIM | ssim | 0.9976 | — | info |
| W1 NRMSE | nrmse | 0.0166 | — | info |
| W2 NLL decrease | nll_decrease_pct | 97.5% | >= 0% | PASS |
| W2 PSNR delta | psnr_delta | +22.42 dB | — | info |
| Trace stages | n_stages | 5 | >= 3 | PASS |

## Reproducibility

- Seed: 42
- PWM version: 7394757
- Python version: 3.13.9
- Key package versions: numpy=2.3.5, scipy=1.16.3
- Platform: Linux x86_64
- Deterministic reproduction command:
  ```bash
  pwm_cli run --modality nerf --seed 42 --mode simulate,invert,calibrate
  ```
- Output hash (y): bef856ca73b30b23
- Output hash (x_hat): 8d9913d1e86a2ff4

## Saved artifacts

- RunBundle: `runs/run_nerf_exp_daf02bfb/`
- Report: `pwm/reports/nerf.md`
- Scoreboard: `pwm/reports/scoreboard.yaml`
- Node trace (.npy): `runs/run_nerf_exp_daf02bfb/artifacts/trace/*.npy`
- Node trace (.png): `runs/run_nerf_exp_daf02bfb/artifacts/trace/png/*.png`
- W2 operator metadata: `runs/run_nerf_exp_daf02bfb/artifacts/w2_operator_meta.json`

## Next actions

- Test on real-world NeRF datasets at full resolution
- Add advanced reconstruction solvers for improved quality
- Investigate additional mismatch parameters
