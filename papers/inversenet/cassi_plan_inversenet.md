# InverseNet ECCV: CASSI Validation Plan

**Document Version:** 2.0
**Date:** 2026-02-15
**Status:** VALIDATED -- All results finalized
**Purpose:** Comprehensive validation of CASSI reconstruction methods under operator mismatch

---

## Executive Summary

This document details the validation framework for CASSI (Coded Aperture Snapshot Spectral Imaging) reconstruction methods in the context of the InverseNet ECCV paper. The benchmark compares **4 reconstruction methods** across **3 scenarios** using **10 KAIST hyperspectral scenes**, evaluating reconstruction quality under realistic operator mismatch without calibration.

**Key Features:**
- **3 Scenarios:** I (Ideal), II (Assumed/Baseline), III (Oracle/Truth Forward Model)
- **4 Methods:** GAP-TV (classical), HDNet, MST-S, MST-L (deep learning)
- **10 Scenes:** 256x256x28 hyperspectral KAIST dataset
- **Metrics:** PSNR (dB), SSIM (0-1)
- **Total Reconstructions:** 120 (10 scenes x 3 scenarios x 4 methods)

**Central Finding:** Mask-awareness is the decisive factor for oracle recovery. MST-L recovers 84% of its mismatch loss (+13.68 dB) when given oracle mask knowledge, while mask-oblivious HDNet recovers 0%.

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

In practice, the reconstruction operator `H_assumed` differs from truth `H_true` due to:

| Factor | Parameter | Range | Impact |
|--------|-----------|-------|--------|
| Mask x-shift | dx | +/-3 px | ~0.12 dB/px |
| Mask y-shift | dy | +/-3 px | ~0.12 dB/px |
| Mask rotation | theta | +/-1 deg | ~3.77 dB/degree |

### 1.3 Measurement Generation

For Scenarios II & III, we inject mismatch into the measurement:

```
y_corrupt = H_mismatch(x) + n
```

Where H_mismatch applies true misalignment parameters (dx=1.5, dy=1.0, theta=0.3), creating degradation that reconstructors must overcome.

---

## 2. Scenario Definitions

### Scenario I: Ideal

**Purpose:** Theoretical upper bound for perfect measurements

**Configuration:**
- **Measurement:** y_ideal from ideal mask (TSA simulation data), no noise
- **Reconstruction:** Each method using ideal mask
- **Mismatch:** None (dx=0, dy=0, theta=0)

**Validated PSNR (10-scene mean +/- std):**
- GAP-TV: 25.45 +/- 2.81 dB
- HDNet: 34.66 +/- 2.62 dB
- MST-S: 33.98 +/- 2.50 dB
- MST-L: 34.81 +/- 2.11 dB

### Scenario II: Assumed/Baseline (Uncorrected Mismatch)

**Purpose:** Realistic baseline showing degradation from uncorrected operator mismatch

**Configuration:**
- **Measurement:** y_corrupt with injected mismatch + low noise
  - Mismatch: dx=1.5 px, dy=1.0 px, theta=0.3 deg
  - Noise: Poisson (alpha=100,000) + Gaussian (sigma=0.01)
- **Reconstruction:** Each method assuming ideal mask (dx=0, dy=0, theta=0)
- **Key insight:** Methods don't "know" about mismatch, so reconstruction is degraded

**Validated PSNR (10-scene mean +/- std):**
- GAP-TV: 20.10 +/- 1.54 dB (degradation: -5.35 dB)
- HDNet: 25.94 +/- 1.97 dB (degradation: -8.72 dB)
- MST-S: 18.54 +/- 2.04 dB (degradation: -15.45 dB)
- MST-L: 18.45 +/- 1.96 dB (degradation: -16.36 dB)

### Scenario III: Oracle (Truth Forward Model)

**Purpose:** Upper bound for corrupted measurements when true mismatch is known

**Configuration:**
- **Measurement:** Same y_corrupt as Scenario II
- **Reconstruction:** Each method using the TRUE corrupted mask (with mismatch applied)
- **Key insight:** Shows recovery possible if system were perfectly characterized

**Validated PSNR (10-scene mean +/- std):**
- GAP-TV: 24.08 +/- 2.04 dB (recovery: +3.98 dB, 74% of loss)
- HDNet: 25.94 +/- 1.97 dB (recovery: +0.00 dB, mask-oblivious)
- MST-S: 31.08 +/- 1.96 dB (recovery: +12.55 dB, 81% of loss)
- MST-L: 32.13 +/- 1.35 dB (recovery: +13.68 dB, 84% of loss)

### Comparison: Scenario Hierarchy

For each mask-aware method:
```
PSNR_I (ideal) > PSNR_III (oracle) > PSNR_II (baseline)
```

For mask-oblivious HDNet: PSNR_III = PSNR_II (no oracle benefit)

**Gaps quantify:**
- **Gap I->II:** Mismatch impact (5-16 dB, method-dependent)
- **Gap II->III:** Oracle recovery (0-13.7 dB, depends on mask-awareness)
- **Gap III->I:** Residual noise/solver limitation (1.4-2.9 dB for mask-aware methods)

---

## 3. Mismatch Parameters

### Injected Mismatch

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| dx | 1.5 px | Moderate-strong horizontal shift |
| dy | 1.0 px | Moderate vertical shift |
| theta | 0.3 deg | Moderate rotation |

**Design rationale:**
- Chosen to produce clear separation between scenarios (II->III gap of +3.98 to +13.68 dB)
- Physically realistic: corresponds to ~0.5 mm mechanical error at typical pixel pitch
- Strong enough to show dramatic MST degradation (-16 dB) but not so severe that all methods collapse

### Noise Model

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Photon peak (alpha) | 100,000 | Low noise to isolate mismatch effect |
| Read noise (sigma) | 0.01 | Minimal read noise |

**Design rationale:** Low noise regime ensures the dominant degradation source is operator mismatch, not photon noise. This produces small residual gaps (III->I: 1.4-2.9 dB), confirming that mismatch is the primary performance bottleneck.

### Bounds and Uncertainty

From cassi_plan.md W1-W5 analysis:
```
dx in [-3, 3] px       -> selected 1.5 px (moderate-strong)
dy in [-3, 3] px       -> selected 1.0 px (moderate)
theta in [-1, 1] deg   -> selected 0.3 deg (moderate)
a1 in [1.95, 2.05]     -> not corrected in this benchmark
alpha in [-1, 1] deg   -> not corrected in this benchmark
```

---

## 4. Reconstruction Methods

### Method 1: GAP-TV (Classical Baseline)

**Category:** Iterative algebraic reconstruction (mask-aware)

**Implementation:** Shifted-domain accelerated GAP with Chambolle TV denoiser

**Parameters:**
- Iterations: 100
- TV weight: 4.0 (tuned via sweep from 0.01-10.0)
- Operates on 3D shifted cubes (H, W_ext, nC)
- Isotropic Chambolle 2004 denoiser in unshifted domain

**Validated Performance:**

| Scenario | PSNR (dB) | SSIM |
|----------|-----------|------|
| I (Ideal) | 25.45 +/- 2.81 | 0.755 +/- 0.082 |
| II (Baseline) | 20.10 +/- 1.54 | 0.598 +/- 0.081 |
| III (Oracle) | 24.08 +/- 2.04 | 0.733 +/- 0.084 |

**Gap II->III:** +3.98 dB (74% recovery)

---

### Method 2: HDNet (Deep Learning, Mask-Oblivious)

**Category:** Dual-domain network with ResBlocks + SDL attention

**Implementation:** Original architecture from MST-main (loaded via importlib)

**Architecture:**
- Head: conv(28, 64, 3) -> 16 ResBlocks with SDL attention + EFF -> Tail: conv(64, 28, 3)
- Input: 28-channel initial estimate from shift_back (mask NOT used)
- Pretrained weights: hdnet.pth

**Validated Performance:**

| Scenario | PSNR (dB) | SSIM |
|----------|-----------|------|
| I (Ideal) | 34.66 +/- 2.62 | 0.965 +/- 0.011 |
| II (Baseline) | 25.94 +/- 1.97 | 0.836 +/- 0.049 |
| III (Oracle) | 25.94 +/- 1.97 | 0.836 +/- 0.049 |

**Gap II->III:** +0.00 dB (mask-oblivious -- ignores mask entirely)

**Key finding:** HDNet's forward() ignores the mask input parameter. Scenario III = Scenario II always. Despite being the most robust under uncorrected mismatch (25.94 dB in Scenario II), it cannot benefit from oracle knowledge.

---

### Method 3: MST-S (Transformer Small, Mask-Aware)

**Category:** Mask-guided Spectral Transformer

**Implementation:** `create_mst(variant='mst_s')` with pretrained weights

**Architecture:**
- Multi-stage Transformer, stage=2, blocks=[2,2,2]
- ~0.9M parameters
- Explicitly takes shifted mask as second input

**Validated Performance:**

| Scenario | PSNR (dB) | SSIM |
|----------|-----------|------|
| I (Ideal) | 33.98 +/- 2.50 | 0.959 +/- 0.014 |
| II (Baseline) | 18.54 +/- 2.04 | 0.656 +/- 0.074 |
| III (Oracle) | 31.08 +/- 1.96 | 0.923 +/- 0.020 |

**Gap II->III:** +12.55 dB (81% recovery)

---

### Method 4: MST-L (Transformer Large, Mask-Aware)

**Category:** Mask-guided Spectral Transformer

**Implementation:** `create_mst(variant='mst_l')` with pretrained weights

**Architecture:**
- Multi-stage Transformer, stage=2, blocks=[4,7,5]
- ~2.0M parameters
- State-of-the-art on clean reconstructions
- Explicitly takes shifted mask as second input

**Validated Performance:**

| Scenario | PSNR (dB) | SSIM |
|----------|-----------|------|
| I (Ideal) | 34.81 +/- 2.11 | 0.969 +/- 0.010 |
| II (Baseline) | 18.45 +/- 1.96 | 0.651 +/- 0.072 |
| III (Oracle) | 32.13 +/- 1.35 | 0.932 +/- 0.017 |

**Gap II->III:** +13.68 dB (84% recovery)

---

## 5. Forward Model Specification

### Fast CASSI Forward (Step=2)

**Dispersion model:**
```
y[:, 2k:2k+W] += mask * x[:,:,k]   for k = 0..27
```

**Measurement size:** (256, 310) where W_ext = W + (nC-1)*step = 256 + 27*2 = 310

### Mask Handling

**Scenario I (Ideal):**
- Mask source: TSA simulation mask (`TSA_simu_data/mask.mat`)
- No mismatch: dx=0, dy=0, theta=0

**Scenario II (Baseline):**
- Measurement: generated with corrupted mask (warped by dx=1.5, dy=1.0, theta=0.3)
- Reconstruction: uses ideal mask (assumes no mismatch)

**Scenario III (Oracle):**
- Measurement: same y_corrupt as Scenario II
- Reconstruction: uses corrupted mask (true mismatch applied)

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

**Definition:**
```
PSNR = 10 * log10(max_val^2 / MSE)  [dB]
```

Where:
- max_val = 1.0 (data normalized to [0,1])
- MSE = mean((x_true - x_recon)^2)

**Interpretation:**
- >40 dB: Excellent (human imperceptible)
- 30-40 dB: Good (minor artifacts)
- 20-30 dB: Fair (visible degradation)
- <20 dB: Poor (significant loss)

### SSIM (Structural Similarity)

**Definition:** Luminance/contrast/structure similarity metric

**Implementation:** Computed on grayscale images (mean across spectral dimension)

**Interpretation:**
- 1.0: Perfect reconstruction
- 0.8-1.0: Excellent perceptual quality
- 0.6-0.8: Good quality
- <0.6: Perceptually degraded

---

## 7. Validated Results Summary

### 7.1 PSNR Results (Mean +/- Std across 10 scenes)

| Method | Scenario I | Scenario II | Scenario III | Gap I->II | Gap II->III | Recovery % |
|--------|-----------|-----------|-----------|---------|----------|-----------|
| GAP-TV | 25.45+/-2.81 | 20.10+/-1.54 | 24.08+/-2.04 | 5.35 | **+3.98** | 74% |
| HDNet | 34.66+/-2.62 | 25.94+/-1.97 | 25.94+/-1.97 | 8.72 | **+0.00** | 0% |
| MST-S | 33.98+/-2.50 | 18.54+/-2.04 | 31.08+/-1.96 | 15.45 | **+12.55** | 81% |
| MST-L | 34.81+/-2.11 | 18.45+/-1.96 | 32.13+/-1.35 | 16.36 | **+13.68** | 84% |

### 7.2 SSIM Results

| Method | Scenario I | Scenario II | Scenario III |
|--------|-----------|-----------|-----------|
| GAP-TV | 0.755+/-0.082 | 0.598+/-0.081 | 0.733+/-0.084 |
| HDNet | 0.965+/-0.011 | 0.836+/-0.049 | 0.836+/-0.049 |
| MST-S | 0.959+/-0.014 | 0.656+/-0.074 | 0.923+/-0.020 |
| MST-L | 0.969+/-0.010 | 0.651+/-0.072 | 0.932+/-0.017 |

### 7.3 Key Findings

1. **Mask-awareness determines oracle recovery.** MST models recover 81-84% of mismatch loss with oracle mask; HDNet recovers 0%.

2. **MST models are most sensitive to mismatch** (15-16 dB degradation) but benefit the most from correction (+12.5-13.7 dB). High sensitivity + high recovery = ideal for calibration-assisted pipelines.

3. **HDNet is most robust under uncorrected mismatch** (25.94 dB in Scenario II) because it ignores the mask. But this robustness prevents it from leveraging oracle knowledge.

4. **GAP-TV provides balanced classical baseline** with moderate sensitivity (5.35 dB) and meaningful recovery (+3.98 dB, 74%).

5. **Method ranking inverts between scenarios:**
   - Scenario I: MST-L > HDNet > MST-S > GAP-TV
   - Scenario II: HDNet > GAP-TV > MST-S > MST-L
   - Scenario III: MST-L > MST-S > HDNet > GAP-TV

6. **Residual gaps are small** for mask-aware methods (1.4-2.9 dB), confirming mismatch -- not noise -- is the dominant degradation source.

---

## 8. Deliverables

### Data Files

1. **results/cassi_validation_results.json** (10 scenes x 3 scenarios x 4 methods)
   - Per-scene PSNR, SSIM for all method/scenario combinations
   - Per-scene gap metrics (I->II, II->III, III->I)

2. **results/cassi_summary.json** (aggregated statistics)
   - Mean PSNR/SSIM per scenario per method with standard deviations
   - Gap means and standard deviations

3. **results/phase3_scenario_results.json** (raw per-scene data)
4. **results/phase3_summary.json** (raw summary statistics)

### Visualization Files (7 figures)

5. **figures/cassi/scenario_comparison.png** -- PSNR bar chart (4 methods x 3 scenarios)
6. **figures/cassi/method_comparison_heatmap.png** -- PSNR + SSIM heatmaps
7. **figures/cassi/gap_comparison.png** -- Degradation (I->II) and recovery (II->III) bar charts
8. **figures/cassi/psnr_distribution.png** -- PSNR boxplot across scenes
9. **figures/cassi/per_scene_psnr.png** -- Per-scene PSNR line plots (2x2 grid)
10. **figures/cassi/ssim_comparison.png** -- SSIM bar chart across scenarios
11. **figures/cassi/oracle_gain_per_scene.png** -- Oracle gain (II->III) per scene

### Table Files

12. **tables/cassi_results_table.csv** -- LaTeX-ready results table

### Report

13. **VALIDATION_REPORT.md** -- Comprehensive analysis with per-scene breakdowns

---

## 9. Validation Workflow

### Step 1: Load Dataset

```python
# Load 10 scenes from KAIST
scenes = [load_scene(f"scene{i:02d}") for i in range(1, 11)]  # (256,256,28) each

# Load mask
mask = load_mask("TSA_simu_data/mask.mat")   # Binary coded aperture
```

### Step 2: Validate Each Scene

For each of 10 scenes:

1. **Scenario I:** Ideal measurement (no mismatch, no noise) + ideal mask reconstruction
2. **Scenario II:** Corrupted measurement (mismatch + noise) + ideal mask reconstruction
3. **Scenario III:** Corrupted measurement (same as II) + corrupted mask reconstruction (oracle)

For each scenario, reconstruct with all 4 methods, compute PSNR/SSIM

### Step 3: Aggregate Results

Compute per-method and per-scenario statistics:
- Mean PSNR/SSIM across 10 scenes
- Standard deviations
- Gap metrics (I->II, II->III, III->I)
- Recovery percentages

### Step 4: Generate Visualizations

Create 7 PNG figures and CSV table as specified in Deliverables section

---

## 10. Implementation Files

### Main Script
- `scripts/phase3_scenario_implementation.py` -- Primary 4-method validation engine (~700 lines)

### Visualization Script
- `scripts/generate_cassi_figures.py` -- Creates 7 PNG figures from results JSON

### Legacy Scripts
- `scripts/validate_cassi_inversenet.py` -- Earlier validation script (v1)
- `scripts/validate_cassi_inversenet_v2.py` -- Earlier validation script (v2)

### Documentation
- `cassi_plan_inversenet.md` -- This file
- `VALIDATION_REPORT.md` -- Comprehensive results analysis

---

## 11. Execution Details

| Metric | Value |
|--------|-------|
| Total scenes | 10 KAIST TSA simulated |
| Total reconstructions | 120 (10 x 3 x 4) |
| Execution time | ~15.3 minutes (GPU) |
| Device | NVIDIA CUDA GPU |
| GAP-TV config | 100 iter, tv_weight=4.0, shifted-domain Chambolle TV |
| DL models | Pretrained weights, inference only |

---

## 12. Quality Assurance

### Verification Checks (All Passed)

1. **Dataset Loading:** All 10 scenes load correctly (256x256x28)
2. **PSNR Hierarchy:** I > III > II confirmed for all mask-aware methods across all 10 scenes
3. **HDNet invariance:** Scenario III = Scenario II for all 10 scenes (mask-oblivious confirmed)
4. **Consistency:** Results reproducible across runs
5. **Recovery percentages:** 74% (GAP-TV), 81% (MST-S), 84% (MST-L) -- physically plausible

### Architectural Classification

| Method | Mask-Aware | Oracle Benefit | Best Scenario |
|--------|-----------|---------------|--------------|
| GAP-TV | Yes (Phi in forward/adjoint) | Moderate (+3.98 dB) | I (25.45 dB) |
| HDNet | No (ignores mask input) | None (0.00 dB) | II (25.94 dB, most robust) |
| MST-S | Yes (shifted mask input) | High (+12.55 dB) | III (31.08 dB) |
| MST-L | Yes (shifted mask input) | Highest (+13.68 dB) | III (32.13 dB) |

---

## 13. Citation & References

**Key References:**
- KAIST HSI Dataset: Choi et al., "High-quality hyperspectral reconstruction using a spectral prior"
- MST: Cai et al., "Mask-guided Spectral-wise Transformer for Efficient Hyperspectral Image Reconstruction" (CVPR 2022)
- HDNet: Hu et al., "HDNet: High-resolution Dual-domain Learning for Spectral Compressive Imaging" (CVPR 2022)
- GAP-TV: Yuan, "Generalized alternating projection based total variation minimization for compressive sensing" (2016)
- Metrics: PSNR (ITU-R BT.601), SSIM (Wang et al., 2004)

**Related Documents:**
- `docs/cassi_plan.md` -- Full CASSI calibration plan (v4+)
- `pwm/reports/cassi_report.md` -- Algorithm 1 & 2 validation report

---

## 14. Appendix: Notation

| Symbol | Meaning |
|--------|---------|
| x | Hyperspectral scene (256x256x28) |
| y | Measurement (256x310) |
| H | Forward model operator |
| dx, dy, theta | Mask affine misalignment |
| PSNR | Peak Signal-to-Noise Ratio (dB) |
| SSIM | Structural Similarity Index (0-1) |
| step | Dispersion stride (2 pixels/band) |
| nC | Number of spectral bands (28) |

---

**Document prepared for InverseNet ECCV benchmark -- Version 2.0 (validated)**
*All results validated on 10 KAIST scenes, 2026-02-15.*
