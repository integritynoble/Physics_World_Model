# X-ray Radiography — PWM Report

| Field | Value |
|-------|-------|
| Modality ID | `xray_radiography` |
| Category | medical |
| Dataset | X-ray benchmark (synthetic proxy: attenuation phantom) |
| Date | 2026-02-12 |
| PWM version | `199aab9` |
| Author | integritynoble |

## Modality overview

- Modality key: `xray_radiography`
- Category: medical
- Forward model: y = Sensor(Scatter(I_0 * exp(-mu * x))) + noise, where x is a 2D attenuation map, Beer-Lambert converts attenuation to transmitted intensity, scatter adds low-frequency background, and the detector has scintillator efficiency 0.8
- Default solver: Beer-Lambert inversion (-log)
- Pipeline linearity: nonlinear (Beer-Lambert exponential)

X-ray radiography produces a single 2D projection image by transmitting X-rays through an object. The transmitted intensity follows the Beer-Lambert law: I = I_0 * exp(-mu * x), where mu is the linear attenuation coefficient and x is the projected thickness. Compton scatter adds a low-frequency background. Reconstruction inverts the Beer-Lambert transform via -log(y/I_0) to recover the attenuation map.

| Parameter | Value |
|-----------|-------|
| Image size (H x W) | 64 x 64 |
| Incident intensity (I_0) | 10000 |
| Scatter fraction | 0.1 |
| Scatter kernel sigma | 5.0 pixels |
| Scintillator efficiency | 0.8 |
| Peak photons | 50000 |
| Read noise sigma | 0.005 |

## Standard dataset

- Name: X-ray benchmark (synthetic proxy: smooth attenuation phantom)
- Source: synthetic
- Size: 64x64 phantom
- Registered in `dataset_registry.yaml` as `xray_benchmark`

## PWM pipeline flowchart (mandatory)

```
x (world)
  ↓
SourceNode: xray_source — X-ray tube intensity scaling (1.0)
  ↓
Element 1 (subrole=transduction): beer_lambert — I = I_0 * exp(-x), nonlinear attenuation (I_0=10000)
  ↓
Element 2 (subrole=transport): scatter_model — Compton scatter (frac=0.1, sigma=5.0)
  ↓
SensorNode: xray_detector_sensor — scintillator efficiency=0.8, gain=1.0
  ↓
NoiseNode: poisson_gaussian_sensor — Poisson (peak=50000) + Gaussian (sigma=0.005)
  ↓
y
```

## Element inventory & mismatch parameters

| # | node_id | primitive_id | subrole | learnable params | mismatch knob | bounds | prior |
|---|---------|-------------|---------|-----------------|---------------|--------|-------|
| 1 | source | xray_source | source | strength=1.0 | — | — | — |
| 2 | transmission | beer_lambert | transduction | I_0=10000 | I_0_drift | [5000, 20000] | log_normal |
| 3 | scatter | scatter_model | transport | scatter_fraction=0.1, kernel_sigma=5.0 | scatter_fraction | [0.0, 0.3] | uniform |
| 4 | sensor | xray_detector_sensor | sensor | scintillator_efficiency=0.8, gain=1.0 | gain | [0.5, 1.5] | normal |
| 5 | noise | poisson_gaussian_sensor | noise | peak=50000, read_sigma=0.005 | — | — | — |

## Node-by-node trace (one sample)

| stage | node_id | output shape | dtype | range [min, max] | artifact_path |
|-------|---------|-------------|-------|-------------------|---------------|
| 0 | input x | (64, 64) | float64 | [0.0000, 0.5000] | `artifacts/trace/00_input_x.npy` |
| 1 | source | (64, 64) | float64 | [0.0000, 0.5000] | `artifacts/trace/01_source.npy` |
| 2 | transmission | (64, 64) | float64 | [6065.3, 10000.0] | `artifacts/trace/02_transmission.npy` |
| 3 | scatter | (64, 64) | float64 | [6200.0, 10100.0] | `artifacts/trace/03_scatter.npy` |
| 4 | sensor | (64, 64) | float64 | [4960.0, 8080.0] | `artifacts/trace/04_sensor.npy` |
| 5 | noise (y) | (64, 64) | float64 | [4900.0, 8150.0] | `artifacts/trace/05_noise_y.npy` |

PNG visualizations saved at: `artifacts/trace/png/`

## Workflow W1: Prompt-driven simulation + reconstruction

- **Prompt used:** `"Simulate X-ray radiography of a phantom and reconstruct via Beer-Lambert inversion"`
- **ExperimentSpec summary:**
  - modality: xray_radiography
  - mode: simulate -> invert
  - solver: Beer-Lambert inversion (-log)
  - photon_budget: I_0=10000
- **Mode S results (simulate y):**
  - y shape: (64, 64)
  - y range: [4900.0, 8150.0]
  - SNR: 85.9 dB
- **Mode I results (reconstruct x_hat):**
  - x_hat shape: (64, 64)
  - Solver: Beer-Lambert inversion, x_hat = -log(y / I_0)
- **Dataset metrics:**

  | Metric | Value |
  |--------|-------|
  | PSNR   | 20.67 dB |
  | SSIM   | 0.8779 |
  | NRMSE  | 0.1001 |

## Workflow W2: Operator correction mode (measured y + operator A)

- **Operator definition:**
  - A_definition: callable
  - A_extraction_method: graph_stripped
  - Operator chain: XRaySource → BeerLambert → ScatterModel → XRayDetectorSensor (noise stripped)
  - A_sha256: bca7c215f8e243a1
  - Linearity: nonlinear (Beer-Lambert exponential)
  - Notes (if linearized): Beer-Lambert exp(-x) is nonlinear; inversion via -log
- **Mismatch specification:**
  - Mismatch type: synthetic_injected
  - Parameters perturbed: sensor gain (1.0 → 1.3, +30%)
  - Description: Detector gain drift from flat-panel aging or temperature effects
- **Mode C fit results:**
  - Correction family: Pre
  - Parameters fitted: 1 (gain) via grid search over [0.5, 2.0]
  - NLL before correction: 54855769425.5
  - NLL after correction: 2048.2
  - NLL decrease: 54855767377.3 (100.0%)
- **Mode I recon using corrected operator A':**

  | Metric | A₀ (uncorrected) | A' (corrected) | Delta |
  |--------|-------------------|----------------|-------|
  | PSNR   | 11.48 dB | 20.67 dB | +9.19 dB |
  | SSIM   | 0.5100 | 0.8779 | +0.3679 |

## Test results summary

### Quick gate (pass/fail)

- [x] W1 simulate: PASS
- [x] W1 reconstruct (PSNR >= 15 dB): PASS (20.67 dB)
- [x] W2 operator correction (NLL decreases): PASS (100.0% decrease)
- [x] W2 corrected recon (beats uncorrected): PASS (+9.19 dB)
- [x] Report contract (flowchart + all headings): PASS
- [x] Node-by-node trace saved (>= 3 stages): PASS (6 stages)
- [x] RunBundle saved (with trace PNGs): PASS
- [x] Scoreboard updated: PASS

### Full metrics

| Check | Metric | Value | Threshold | Status |
|-------|--------|-------|-----------|--------|
| W1 PSNR | psnr | 20.67 dB | >= 15 dB | PASS |
| W1 SSIM | ssim | 0.8779 | >= 0.60 | PASS |
| W1 NRMSE | nrmse | 0.1001 | — | info |
| W2 NLL decrease | nll_decrease_pct | 100.0% | >= 5% | PASS |
| W2 PSNR gain | psnr_delta | +9.19 dB | > 0 | PASS |
| W2 SSIM gain | ssim_delta | +0.3679 | > 0 | PASS |
| Trace stages | n_stages | 6 | >= 3 | PASS |
| Trace PNGs | n_pngs | 6 | >= 3 | PASS |
| W1 wall time | w1_seconds | 0.02 s | — | info |
| W2 wall time | w2_seconds | 0.04 s | — | info |

## Reproducibility

- Seed: 42
- NumPy RNG state hash: 5d6a058f2c5c262e
- PWM version: 199aab9
- Python version: 3.13.9
- Key package versions: numpy=2.3.5, scipy=1.16.3
- Platform: Linux x86_64
- Deterministic reproduction command:
  ```bash
  pwm_cli run --modality xray_radiography --seed 42 --mode simulate,invert,calibrate
  ```
- Output hash (y): see RunBundle
- Output hash (x_hat): see RunBundle

## Saved artifacts

- RunBundle: `runs/run_xray_radiography_exp_bca7c215/`
- Report: `pwm/reports/xray_radiography.md`
- Scoreboard: `pwm/reports/scoreboard.yaml`
- Node trace (.npy): `runs/run_xray_radiography_exp_bca7c215/artifacts/trace/*.npy`
- Node trace (.png): `runs/run_xray_radiography_exp_bca7c215/artifacts/trace/png/*.png`
- W2 operator metadata: `runs/run_xray_radiography_exp_bca7c215/artifacts/w2_operator_meta.json`
- Ground truth: `runs/run_xray_radiography_exp_bca7c215/artifacts/x_true.npy`
- Measurement: `runs/run_xray_radiography_exp_bca7c215/artifacts/y.npy`
- W1 reconstruction: `runs/run_xray_radiography_exp_bca7c215/artifacts/x_hat.npy`
- W2 reconstructions: `runs/run_xray_radiography_exp_bca7c215/artifacts/x_hat_w2_uncorrected.npy`, `x_hat_w2_corrected.npy`

## Next actions

- Test on real chest X-ray datasets (CheXpert, MIMIC-CXR)
- Add dual-energy X-ray radiography for material decomposition
- Implement anti-scatter grid modeling for improved scatter correction
