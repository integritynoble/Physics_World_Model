# Electron Diffraction — PWM Report

| Field | Value |
|-------|-------|
| Modality ID | `electron_diffraction` |
| Category | electron |
| Dataset | electron_diff_benchmark (synthetic proxy: phantom) |
| Date | 2026-02-12 |
| PWM version | `7394757` |
| Author | integritynoble |

## Modality overview

- Modality key: `electron_diffraction`
- Category: electron
- Forward model: y = Sensor(DiffractionCamera(GenericSource(x))) + n
- Default solver: pseudo_inverse
- Pipeline linearity: nonlinear

Electron Diffraction imaging pipeline with 1 element(s) in the forward chain. Reconstruction uses pseudo_inverse. Tested on a 64x64 synthetic phantom with seed=42.

| Parameter | Value |
|-----------|-------|
| Image size | 64 x 64 |
| Output shape | 64 x 64 |
| Noise model | Poisson-Gaussian |

## Standard dataset

- Name: electron_diff_benchmark (synthetic proxy: Gaussian-blob phantom)
- Source: synthetic
- Size: 1 image, 64x64
- Registered in `dataset_registry.yaml` as `electron_diff_benchmark`

For this baseline experiment, a deterministic phantom (seed=42, smooth Gaussian blobs on a 64x64 grid) is used.

## PWM pipeline flowchart (mandatory)

```
x (world)
  ↓
SourceNode: generic_source — strength=1.0
  ↓
Element 1 (subrole=transport): diffraction_camera — detector_psf_sigma=0.5
  ↓
SensorNode: energy_resolving_detector — n_channels=256, energy_range_ev=[0, 2000], qe=0.8
  ↓
NoiseNode: poisson_gaussian_sensor — peak_photons=10000.0, read_sigma=0.01, seed=0
  ↓
y
```

## Element inventory & mismatch parameters

| # | node_id | primitive_id | subrole | learnable params | mismatch knob | bounds | prior |
|---|---------|-------------|---------|-----------------|---------------|--------|-------|
| 1 | source | generic_source | source | strength=1.0 | — | — | — |
| 2 | diffract | diffraction_camera | transport | detector_psf_sigma=0.5 | gain | [0.5, 2.0] | uniform |
| 3 | sensor | energy_resolving_detector | sensor | n_channels=256, energy_range_ev=[0, 2000], qe=0.8 | gain | [0.5, 2.0] | uniform |
| 4 | noise | poisson_gaussian_sensor | noise | peak_photons=10000.0, read_sigma=0.01, seed=0 | — | — | — |

## Node-by-node trace (one sample)

Trace of intermediate tensor states at each node for one representative sample (seed=42).
Each row references a saved artifact in the RunBundle.

| stage | node_id | output shape | dtype | range [min, max] | artifact_path |
|-------|---------|-------------|-------|-------------------|---------------|
| 0 | input x | (64, 64) | float64 | [0.0000, 1.0000] | `artifacts/trace/00_input_x.npy` |
| 1 | source | (64, 64) | float64 | [0.0000, 1.0000] | `artifacts/trace/01_source.npy` |
| 2 | diffract | (64, 64) | float64 | [0.0000, 1.0000] | `artifacts/trace/02_diffract.npy` |
| 3 | sensor | (64, 64) | float64 | [0.0000, 1.0000] | `artifacts/trace/03_sensor.npy` |
| 4 | noise (y) | (64, 64) | float64 | [0.0000, 1.0000] | `artifacts/trace/04_noise_y.npy` |

PNG visualizations saved at: `artifacts/trace/png/`

## Workflow W1: Prompt-driven simulation + reconstruction

- **Prompt used:** `"Simulate electron_diffraction measurement of a phantom and reconstruct"`
- **ExperimentSpec summary:**
  - modality: electron_diffraction
  - mode: simulate -> invert
  - solver: pseudo_inverse
- **Mode S results (simulate y):**
  - y shape: (64, 64)
  - SNR: 90.3 dB
- **Mode I results (reconstruct x_hat):**
  - x_hat shape: (64, 64)
  - Solver: pseudo_inverse
- **Dataset metrics:**

  | Metric | Value |
  |--------|-------|
  | PSNR   | 10.10 dB |
  | SSIM   | 0.0003 |
  | NRMSE  | 0.3383 |

## Workflow W2: Operator correction mode (measured y + operator A)

- **Operator definition:**
  - A_definition: callable
  - A_extraction_method: graph_stripped
  - Operator chain: forward model stripped of noise node
  - A_sha256: c8a0460d0f43380b
  - Linearity: nonlinear
  - Notes (if linearized): N/A
- **Mismatch specification:**
  - Mismatch type: synthetic_injected
  - Parameters perturbed: gain drift (1.0 -> 1.3)
  - Description: Synthetic parameter drift injected for calibration testing
- **Mode C fit results:**
  - Correction family: Pre
  - Parameters fitted: 1 parameter via grid search
  - NLL before correction: 96431598809.8
  - NLL after correction: 2048.2
  - NLL decrease: 100.0%
- **Mode I recon using corrected operator A':**

  | Metric | A₀ (uncorrected) | A' (corrected) | Delta |
  |--------|-------------------|----------------|-------|
  | PSNR   | 10.10 dB | 10.10 dB | +0.00 dB |

## Test results summary

### Quick gate (pass/fail)

- [x] W1 simulate: PASS
- [x] W1 reconstruct (PSNR reported): PASS (10.10 dB)
- [x] W2 operator correction (NLL decrease): PASS (100.0%)
- [x] Report contract (flowchart + all headings): PASS
- [x] Node-by-node trace saved (>= 3 stages): PASS (5 stages)
- [x] RunBundle saved (with trace PNGs): PASS
- [x] Scoreboard updated: PASS

### Full metrics

| Check | Metric | Value | Threshold | Status |
|-------|--------|-------|-----------|--------|
| W1 PSNR | psnr | 10.10 dB | >= 1 dB | PASS |
| W1 SSIM | ssim | 0.0003 | — | info |
| W1 NRMSE | nrmse | 0.3383 | — | info |
| W2 NLL decrease | nll_decrease_pct | 100.0% | >= 0% | PASS |
| W2 PSNR delta | psnr_delta | +0.00 dB | — | info |
| Trace stages | n_stages | 5 | >= 3 | PASS |

## Reproducibility

- Seed: 42
- PWM version: 7394757
- Python version: 3.13.9
- Key package versions: numpy=2.3.5, scipy=1.16.3
- Platform: Linux x86_64
- Deterministic reproduction command:
  ```bash
  pwm_cli run --modality electron_diffraction --seed 42 --mode simulate,invert,calibrate
  ```
- Output hash (y): d361dcbeb1c3340e
- Output hash (x_hat): 0798d37d545e7958

## Saved artifacts

- RunBundle: `runs/run_electron_diffraction_exp_3f4ff4f9/`
- Report: `pwm/reports/electron_diffraction.md`
- Scoreboard: `pwm/reports/scoreboard.yaml`
- Node trace (.npy): `runs/run_electron_diffraction_exp_3f4ff4f9/artifacts/trace/*.npy`
- Node trace (.png): `runs/run_electron_diffraction_exp_3f4ff4f9/artifacts/trace/png/*.png`
- W2 operator metadata: `runs/run_electron_diffraction_exp_3f4ff4f9/artifacts/w2_operator_meta.json`

## Next actions

- Test on real-world Electron Diffraction datasets at full resolution
- Add advanced reconstruction solvers for improved quality
- Investigate additional mismatch parameters
