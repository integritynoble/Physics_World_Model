# InverseNet ECCV: CASSI Validation Plan

**Document Version:** 4.0
**Date:** 2026-02-17
**Status:** IN PROGRESS -- 5-parameter mismatch (mask affine + dispersion)
**Purpose:** Comprehensive validation of CASSI reconstruction methods under full 5-parameter operator mismatch

---

## Executive Summary

This document details the validation framework for CASSI (Coded Aperture Snapshot Spectral Imaging) reconstruction methods in the context of the InverseNet ECCV paper. The benchmark compares **4 reconstruction methods** across **3 scenarios** using **10 KAIST hyperspectral scenes**, evaluating reconstruction quality under realistic **5-parameter operator mismatch** (mask affine + dispersion) without calibration.

**Key Features:**
- **3 Scenarios:** I (Ideal), II (Assumed/Baseline), IV (Truth Forward Model / Oracle)
- **5 Mismatch Parameters:** mask_dx, mask_dy, mask_theta, disp_a1, disp_alpha
- **4 Methods:** GAP-TV (classical), HDNet, MST-S, MST-L (deep learning)
- **10 Scenes:** 256x256x28 hyperspectral KAIST dataset
- **Metrics:** PSNR (dB), SSIM (0-1), SAM (degrees)
- **Total Reconstructions:** 120 (10 scenes x 3 scenarios x 4 methods)

**Rationale:** The InverseNet benchmark compares reconstruction methods under operator mismatch WITHOUT calibration. Scenario III (operator correction via Algorithms 1 & 2) is skipped. Scenario IV (Truth Forward Model) provides the oracle performance when true mismatch parameters are known, establishing an upper bound for corrupted measurements.

**Expected Hierarchy:** PSNR_I > PSNR_IV > PSNR_II (for mask-aware methods)

---

## 1. Problem Formulation

### 1.1 Forward Model

CASSI forward model with step-2 spectral dispersion:

```
y = H_true(x) + n
```

Where:
- **x** in R^{256x256x28}: True hyperspectral scene
- **H_true**: Fast CASSI forward with step=2 dispersion
- **y** in R^{256x310}: Measurement (W_ext = 256 + (28-1)*2 = 310)
- **n**: Poisson shot + Gaussian read noise

### 1.2 Operator Mismatch

In practice, the reconstruction operator `H_assumed` differs from truth `H_true` due to 5 mismatch factors from W1-W5 analysis (cassi_plan.md):

| Factor | Parameter | Range | Impact (dB) | Source |
|--------|-----------|-------|-------------|--------|
| Mask x-shift | mask_dx | [-3, 3] px | 0.12 | W1: mechanical assembly tolerance |
| Mask y-shift | mask_dy | [-3, 3] px | 0.12 | W1: mechanical assembly tolerance |
| Mask rotation | mask_theta | [-1, 1] deg | 3.77 | W2: optical bench rotation |
| Dispersion slope | disp_a1 | [1.95, 2.05] px/band | 5.49 | W4: prism slope / thermal drift |
| Dispersion angle | disp_alpha | [-1, 1] deg | 7.04 | W5: dispersion axis offset / prism settling |

**Total potential mismatch impact:** ~16.54 dB (sum of all 5 factors at worst case)

**Parameter grouping:**
- **Group 1 (Mask Affine):** mask_dx, mask_dy, mask_theta -- combined into one affine warp
- **Group 2 (Dispersion):** disp_a1, disp_alpha -- modify spectral dispersion encoding

### 1.3 Measurement Generation

For Scenarios II & IV, we inject all 5 mismatch parameters into the measurement:

```
y_corrupt = H_mismatch(x) + n
```

Where H_mismatch applies:
- **Mask affine:** dx=0.5 px, dy=0.3 px, theta=0.1 deg (mask misalignment)
- **Dispersion:** a1=2.02 px/band (2% slope drift), alpha=0.15 deg (axis offset)

This creates degradation from both mask misalignment AND dispersion model error.

---

## 2. Scenario Definitions

### Scenario I: Ideal

**Purpose:** Theoretical upper bound for perfect measurements

**Configuration:**
- **Measurement:** y_ideal from ideal mask (TSA simulation data), no noise
- **Reconstruction:** Each method using ideal mask
- **Mismatch:** None (dx=0, dy=0, theta=0, a1=2.0, alpha=0)

### Scenario II: Assumed/Baseline (Uncorrected Mismatch)

**Purpose:** Realistic baseline showing degradation from uncorrected operator mismatch

**Configuration:**
- **Measurement:** y_corrupt with injected 5-parameter mismatch + low noise
  - Mask affine: dx=0.5 px, dy=0.3 px, theta=0.1 deg
  - Dispersion: a1=2.02 px/band (nominal 2.0 + 1% drift), alpha=0.15 deg (axis offset)
  - Noise: Poisson (alpha=100,000) + Gaussian (sigma=0.01)
- **Reconstruction:** Each method assuming ideal parameters (dx=0, dy=0, theta=0, a1=2.0, alpha=0)
- **Key insight:** Methods don't "know" about ANY mismatch (mask OR dispersion), so reconstruction is degraded

### Scenario IV: Truth Forward Model (Oracle)

**Purpose:** Upper bound for corrupted measurements when all 5 true mismatch parameters are known

**Configuration:**
- **Measurement:** Same y_corrupt as Scenario II (all 5 mismatch factors applied)
- **Reconstruction:** Each method using TRUE parameters:
  - Mask warped with true (dx=0.5, dy=0.3, theta=0.1)
  - Dispersion with true (a1=2.02, alpha=0.15)
- **Key insight:** Shows recovery possible if system were perfectly characterized (all 5 parameters known)

**Note:** Scenario III (operator correction via Algorithms 1 & 2) is intentionally skipped for the InverseNet benchmark, which focuses on reconstruction comparison rather than calibration.

### Comparison: Scenario Hierarchy

For each mask-aware method:
```
PSNR_I (ideal) > PSNR_IV (oracle) > PSNR_II (baseline)
```

For mask-oblivious HDNet: PSNR_IV = PSNR_II (no oracle benefit)

**Gaps quantify:**
- **Gap I->II:** Full 5-parameter mismatch impact (mask affine + dispersion, method-dependent)
- **Gap II->IV:** Oracle recovery when all 5 true parameters are known (depends on mask/dispersion-awareness)
- **Gap IV->I:** Residual noise/solver limitation

---

## 3. Mismatch Parameters (5 Factors)

### Injected Mismatch

**Group 1: Mask Affine (W1-W2 factors)**

| Parameter | Value | Range | Impact | Rationale |
|-----------|-------|-------|--------|-----------|
| mask_dx | 0.5 px | [-3, 3] px | 0.12 dB | Moderate sub-pixel horizontal shift (mechanical tolerance) |
| mask_dy | 0.3 px | [-3, 3] px | 0.12 dB | Moderate sub-pixel vertical shift |
| mask_theta | 0.1 deg | [-1, 1] deg | 3.77 dB | Moderate rotation (optical bench twist) |

**Group 2: Dispersion (W4-W5 factors)**

| Parameter | Value | Range | Impact | Rationale |
|-----------|-------|-------|--------|-----------|
| disp_a1 | 2.02 px/band | [1.95, 2.05] | 5.49 dB | 1% drift from nominal (thermal drift on prism slope) |
| disp_alpha | 0.15 deg | [-1, 1] deg | 7.04 dB | Dispersion axis offset (prism settling after assembly) |

**Combined mismatch impact table (from cassi_plan.md Algorithm 2 estimates):**

| Parameter | Ground Truth | Estimated | Error | Impact (dB) |
|-----------|-------------|-----------|-------|-------------|
| mask_dx | in [-3, 3] px | +/-0.05-0.1 px | ~0.1 px | 0.12 |
| mask_dy | in [-3, 3] px | +/-0.05-0.1 px | ~0.1 px | 0.12 |
| mask_theta | in [-1, 1] deg | +/-0.02-0.05 deg | ~0.05 deg | 3.77 |
| disp_a1 | nominal=2.0 | +/-0.001 px/band | ~0.001 | 5.49 |
| disp_alpha | nominal=0 deg | +/-0.02-0.05 deg | ~0.05 deg | 7.04 |

**Design rationale:**
- All 5 mismatch factors from W1-W5 analysis are now included
- Mask affine parameters (dx, dy, theta) capture mechanical assembly tolerance
- Dispersion parameters (a1, alpha) capture prism/grating optical drift
- Dispersion mismatch has HIGHER individual impact (5.49 + 7.04 = 12.53 dB) than mask affine (0.12 + 0.12 + 3.77 = 4.01 dB)
- Selected values are moderate (not worst-case) to show meaningful but realistic degradation
- Strong enough to demonstrate that dispersion mismatch is the dominant degradation source

### Noise Model

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Photon peak (alpha) | 100,000 | Low noise to isolate mismatch effect |
| Read noise (sigma) | 0.01 | Minimal read noise |

**Design rationale:** Low noise regime ensures the dominant degradation source is operator mismatch (5 parameters), not photon noise.

### Bounds and Uncertainty

From cassi_plan.md W1-W5 analysis (ALL 5 parameters now active):
```
Group 1 (Mask Affine):
  mask_dx    in [-3, 3] px       -> selected 0.5 px (moderate)
  mask_dy    in [-3, 3] px       -> selected 0.3 px (moderate)
  mask_theta in [-1, 1] deg      -> selected 0.1 deg (moderate)

Group 2 (Dispersion):
  disp_a1    in [1.95, 2.05]     -> selected 2.02 (1% drift from nominal 2.0)
  disp_alpha in [-1, 1] deg      -> selected 0.15 deg (moderate axis offset)
```

---

## 4. Reconstruction Methods

### Method 1: GAP-TV (Classical Baseline)

**Category:** Iterative algebraic reconstruction (mask-aware)

**Implementation:** Shifted-domain accelerated GAP with Chambolle TV denoiser

**Parameters:**
- Iterations: 50
- TV weight (lambda): 0.01
- Step: 2 (dispersion stride)
- Operates on 3D shifted cubes (H, W_ext, nC)

**Expected Performance:** ~32 dB (Scenario I), ~28-30 dB (Scenario II)

---

### Method 2: HDNet (Deep Learning, Mask-Oblivious)

**Category:** Dual-domain network with ResBlocks + SDL attention

**Architecture:**
- Head: conv(28, dim, 3) -> 16 ResBlocks with SDL attention + EFF -> Tail: conv(dim, 28, 3)
- Input: 28-channel initial estimate from shift_back (mask NOT used in forward)
- Pretrained weights: hdnet.pth (~2.37M params)

**Expected Performance:** ~35 dB (Scenario I), ~31 dB (Scenario II)

**Key characteristic:** Mask-oblivious -- Scenario IV = Scenario II always.

---

### Method 3: MST-S (Transformer Small, Mask-Aware)

**Category:** Mask-guided Spectral Transformer

**Architecture:**
- Multi-stage Transformer, stage=2, blocks=[2,2,2]
- ~0.9M parameters
- Explicitly takes shifted mask as second input

**Expected Performance:** ~34 dB (Scenario I), ~30 dB (Scenario II), ~31 dB (Scenario IV)

---

### Method 4: MST-L (Transformer Large, Mask-Aware)

**Category:** Mask-guided Spectral Transformer

**Architecture:**
- Multi-stage Transformer, stage=2, blocks=[4,7,5]
- ~2.0M parameters
- State-of-the-art on clean reconstructions
- Explicitly takes shifted mask as second input

**Expected Performance:** ~36 dB (Scenario I), ~32 dB (Scenario II), ~33 dB (Scenario IV)

---

## 5. Forward Model Specification

### Fast CASSI Forward (Step=2, with Dispersion Parameters)

**Ideal dispersion model (a1=2.0, alpha=0):**
```
y[:, 2k:2k+W] += mask * x[:,:,k]   for k = 0..27
```

**Corrupted dispersion model (a1=2.02, alpha=0.15 deg):**
```
For band k:
  shift_k = round(a1 * (k - k_center))                         # Dispersion with modified slope
  Rotate dispersion axis by alpha degrees                        # Axis offset
  y[:, shift_k:shift_k+W] += warp(mask, dx, dy, theta) * x[:,:,k]
```

**Measurement size:** (256, 310) where W_ext = W + (nC-1)*step = 256 + 27*2 = 310

**Dispersion parameter effects:**
- `disp_a1` controls pixels-per-band shift: nominal=2.0, corrupted=2.02 (cumulative error grows with band index)
- `disp_alpha` rotates the dispersion axis: nominal=0, corrupted=0.15 deg (creates 2D spectral spread instead of purely horizontal)

### Mask and Dispersion Handling

**Scenario I (Ideal):**
- Mask source: TSA simulation mask (`TSA_simu_data/mask.mat`)
- No mismatch: dx=0, dy=0, theta=0, a1=2.0, alpha=0

**Scenario II (Baseline):**
- Measurement: generated with ALL 5 mismatch factors:
  - Corrupted mask (warped by dx=0.5, dy=0.3, theta=0.1)
  - Corrupted dispersion (a1=2.02, alpha=0.15 deg)
- Reconstruction: uses ideal parameters (assumes no mismatch in mask OR dispersion)

**Scenario IV (Oracle):**
- Measurement: same y_corrupt as Scenario II
- Reconstruction: uses TRUE mismatch parameters for both mask AND dispersion
  - Mask: warped with true (dx=0.5, dy=0.3, theta=0.1)
  - Dispersion: a1=2.02, alpha=0.15 deg

### Noise Model

**Poisson + Gaussian:**
```
y_noisy = Poisson(alpha * y_clean) / alpha + Gaussian(0, sigma)
```

**Parameters:**
- Photon peak (alpha): 100,000 (low noise regime)
- Read noise std (sigma): 0.01

---

## 6. Evaluation Metrics

### PSNR (Peak Signal-to-Noise Ratio)

```
PSNR = 10 * log10(max_val^2 / MSE)  [dB]
```

Where max_val = 1.0 (data normalized to [0,1])

### SSIM (Structural Similarity)

Computed on grayscale images (mean across spectral dimension) using 11x11 window.

### SAM (Spectral Angle Mapper)

```
SAM = mean(arccos(dot(x_true, x_recon) / (||x_true|| * ||x_recon||)))  [degrees]
```

Computed per-pixel, averaged over all spatial locations.

---

## 7. Expected Results

### PSNR Hierarchy (per method)

With all 5 mismatch parameters active (mask affine + dispersion), degradation is significantly larger than mask-only:

- **Gap I->II:** 5-10 dB degradation (combined mask + dispersion mismatch impact)
- **Gap II->IV:** 3-6 dB recovery (oracle knows all 5 true parameters)
- **Gap IV->I:** 1-3 dB residual (noise + measurement corruption)

**Note:** Dispersion mismatch (a1, alpha) contributes more degradation than mask affine (dx, dy, theta) due to cumulative error across 28 bands.

### Method Ranking (all scenarios)

1. **MST-L:** ~36 dB (I), ~28 dB (II), ~33 dB (IV)
2. **HDNet:** ~35 dB (I), ~27 dB (II), ~27 dB (IV) -- mask-oblivious, no oracle benefit
3. **MST-S:** ~34 dB (I), ~26 dB (II), ~31 dB (IV)
4. **GAP-TV:** ~32 dB (I), ~24 dB (II), ~29 dB (IV)

### Key Insights for Paper

- **Dispersion mismatch dominates:** disp_a1 (5.49 dB) and disp_alpha (7.04 dB) have higher individual impact than all 3 mask parameters combined (4.01 dB)
- Deep learning methods maintain advantage even under 5-parameter mismatch
- Scenario IV validates that methods can utilize corrected operators for BOTH mask and dispersion
- Gap II->IV is larger with dispersion mismatch, quantifying the importance of dispersion calibration
- **Mask-oblivious methods (HDNet):** Scenario IV = Scenario II, as they don't use mask or dispersion parameters in reconstruction

---

## 8. Deliverables

### Data Files

1. **results/cassi_validation_results.json** (10 scenes x 3 scenarios x 4 methods)
   - Per-scene PSNR, SSIM, SAM for all method/scenario combinations
   - Per-scene gap metrics (I->II, II->IV, IV->I)

2. **results/cassi_summary.json** (aggregated statistics)
   - Mean PSNR/SSIM/SAM per scenario per method with standard deviations
   - Gap means and standard deviations

### Visualization Files (7 figures)

3. **figures/cassi/scenario_comparison.png** -- PSNR bar chart (4 methods x 3 scenarios)
4. **figures/cassi/method_comparison_heatmap.png** -- PSNR + SSIM heatmaps
5. **figures/cassi/gap_comparison.png** -- Degradation (I->II) and recovery (II->IV) bar charts
6. **figures/cassi/psnr_distribution.png** -- PSNR boxplot across scenes
7. **figures/cassi/per_scene_psnr.png** -- Per-scene PSNR line plots (2x2 grid)
8. **figures/cassi/ssim_comparison.png** -- SSIM bar chart across scenarios
9. **figures/cassi/oracle_gain_per_scene.png** -- Oracle gain (II->IV) per scene

### Table Files

10. **tables/cassi_results_table.csv** -- LaTeX-ready results table

---

## 9. Implementation Files

### Main Scripts
- `scripts/validate_cassi_inversenet.py` -- Primary 4-method validation engine
- `scripts/generate_cassi_figures.py` -- Creates 7 PNG figures from results JSON

### Documentation
- `cassi_plan_inversenet.md` -- This file

---

## 10. Execution Details

| Metric | Value |
|--------|-------|
| Total scenes | 10 KAIST TSA simulated |
| Total reconstructions | 120 (10 x 3 x 4) |
| Mismatch parameters | 5 (mask_dx, mask_dy, mask_theta, disp_a1, disp_alpha) |
| Mismatch values | dx=0.5, dy=0.3, theta=0.1, a1=2.02, alpha=0.15 |
| Device | NVIDIA CUDA GPU |
| GAP-TV config | 50 iter, lam=0.01, step=2 |
| DL models | Pretrained weights, inference only |

---

## 11. Quality Assurance

### Verification Checks

1. **Dataset Loading:** All 10 scenes load correctly (256x256x28)
2. **PSNR Hierarchy:** I > IV > II confirmed for all mask-aware methods
3. **HDNet invariance:** Scenario IV = Scenario II for all scenes (mask-oblivious confirmed)
4. **Consistency:** Results reproducible across runs

### Architectural Classification

| Method | Mask-Aware | Oracle Benefit | Robustness Under Mismatch |
|--------|-----------|---------------|--------------------------|
| GAP-TV | Yes (Phi in forward/adjoint) | Moderate | Moderate |
| HDNet | No (ignores mask input) | None | High (most robust) |
| MST-S | Yes (shifted mask input) | High | Low (sensitive) |
| MST-L | Yes (shifted mask input) | Highest | Low (most sensitive) |

---

## 12. Citation & References

- KAIST HSI Dataset: Choi et al., "High-quality hyperspectral reconstruction using a spectral prior"
- MST: Cai et al., "Mask-guided Spectral-wise Transformer for Efficient Hyperspectral Image Reconstruction" (CVPR 2022)
- HDNet: Hu et al., "HDNet: High-resolution Dual-domain Learning for Spectral Compressive Imaging" (CVPR 2022)
- GAP-TV: Yuan, "Generalized alternating projection based total variation minimization for compressive sensing" (2016)

---

**Document prepared for InverseNet ECCV benchmark -- Version 4.0**
*Updated 2026-02-17: Full 5-parameter mismatch (mask_dx=0.5, mask_dy=0.3, mask_theta=0.1, disp_a1=2.02, disp_alpha=0.15)*
