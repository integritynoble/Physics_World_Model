# CASSI Validation Report for InverseNet ECCV Paper

**Date:** 2026-02-15
**Dataset:** KAIST TSA Simulated (10 scenes, 256x256x28)
**Modality:** CASSI (Coded Aperture Snapshot Spectral Imaging)

---

## 1. Experimental Setup

### 1.1 Forward Model

- **Operator:** Fast CASSI forward model with spectral dispersion
- **Resolution:** 256x256 spatial, 28 spectral bands
- **Measurement:** (256, 310) where W_ext = 256 + (28-1)*2
- **Dispersion:** step=2 pixels per spectral band

### 1.2 Mismatch Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| dx | 1.5 px | Horizontal mask shift |
| dy | 1.0 px | Vertical mask shift |
| theta | 0.3 deg | Mask rotation |

These represent moderate-to-strong assembly mismatch, chosen to produce a clear separation between scenarios while remaining physically realistic.

### 1.3 Noise Model

| Parameter | Value |
|-----------|-------|
| Photon peak (alpha) | 100,000 |
| Read noise (sigma) | 0.01 |

Low noise regime to isolate the effect of operator mismatch from noise-induced degradation.

### 1.4 Reconstruction Methods

| Method | Type | Mask-Aware | Description |
|--------|------|------------|-------------|
| GAP-TV | Classical | Yes | Accelerated GAP-TV with Chambolle TV denoiser (shifted-domain, tv_weight=4.0, 100 iter) |
| HDNet | Deep Learning | No | Dual-domain network with ResBlocks + SDL attention (pretrained, 28-ch input) |
| MST-S | Deep Learning | Yes | Mask-guided Spectral Transformer, Small (pretrained, stage=2, blocks=[2,2,2]) |
| MST-L | Deep Learning | Yes | Mask-guided Spectral Transformer, Large (pretrained, stage=2, blocks=[4,7,5]) |

GAP-TV operates in the shifted domain using accelerated proximal gradient with isotropic TV regularization.
HDNet, MST-S, and MST-L use pretrained weights from their respective publications.

### 1.5 Scenarios

| Scenario | Measurement | Reconstruction Mask | Purpose |
|----------|-------------|-------------------|---------|
| I (Ideal) | Ideal mask, no noise | Ideal mask | Upper bound |
| II (Baseline) | Corrupted mask + noise | Ideal mask (wrong!) | Mismatch degradation |
| III (Oracle) | Corrupted mask + noise | Corrupted mask (true) | Oracle recovery |

---

## 2. Results

### 2.1 Summary Table (PSNR, dB)

| Method | Scenario I | Scenario II | Scenario III | Gap I->II | Gap II->III |
|--------|-----------|------------|------------|----------|----------|
| GAP-TV | 25.45 +/- 2.81 | 20.10 +/- 1.54 | 24.08 +/- 2.04 | 5.35 | **+3.98** |
| HDNet | 34.66 +/- 2.62 | 25.94 +/- 1.97 | 25.94 +/- 1.97 | 8.72 | **+0.00** |
| MST-S | 33.98 +/- 2.50 | 18.54 +/- 2.04 | 31.08 +/- 1.96 | 15.45 | **+12.55** |
| MST-L | 34.81 +/- 2.11 | 18.45 +/- 1.96 | 32.13 +/- 1.35 | 16.36 | **+13.68** |

### 2.2 SSIM Results

| Method | Scenario I | Scenario II | Scenario III |
|--------|-----------|------------|------------|
| GAP-TV | 0.755 +/- 0.082 | 0.598 +/- 0.081 | 0.733 +/- 0.084 |
| HDNet | 0.965 +/- 0.011 | 0.836 +/- 0.049 | 0.836 +/- 0.049 |
| MST-S | 0.959 +/- 0.014 | 0.656 +/- 0.074 | 0.923 +/- 0.020 |
| MST-L | 0.969 +/- 0.010 | 0.651 +/- 0.072 | 0.932 +/- 0.017 |

### 2.3 Per-Scene PSNR (dB)

#### GAP-TV

| Scene | Scenario I | Scenario II | Scenario III | Gap I->II | Gap II->III |
|-------|-----------|------------|------------|----------|----------|
| S01 | 26.84 | 22.77 | 26.06 | 4.07 | 3.29 |
| S02 | 25.33 | 20.80 | 24.53 | 4.53 | 3.73 |
| S03 | 25.70 | 17.17 | 22.89 | 8.52 | 5.72 |
| S04 | 33.21 | 22.26 | 29.32 | 10.95 | 7.06 |
| S05 | 24.09 | 19.53 | 23.12 | 4.56 | 3.59 |
| S06 | 23.12 | 19.72 | 22.52 | 3.41 | 2.81 |
| S07 | 24.20 | 19.87 | 23.24 | 4.33 | 3.37 |
| S08 | 22.98 | 20.55 | 22.22 | 2.43 | 1.67 |
| S09 | 24.59 | 18.75 | 23.17 | 5.84 | 4.43 |
| S10 | 24.44 | 19.57 | 23.67 | 4.87 | 4.10 |

#### HDNet

| Scene | Scenario I | Scenario II | Scenario III | Gap I->II | Gap II->III |
|-------|-----------|------------|------------|----------|----------|
| S01 | 34.95 | 29.42 | 29.42 | 5.52 | 0.00 |
| S02 | 35.65 | 27.22 | 27.22 | 8.43 | 0.00 |
| S03 | 35.54 | 22.03 | 22.03 | 13.51 | 0.00 |
| S04 | 41.63 | 25.88 | 25.88 | 15.75 | 0.00 |
| S05 | 32.56 | 24.05 | 24.05 | 8.51 | 0.00 |
| S06 | 34.33 | 27.99 | 27.99 | 6.34 | 0.00 |
| S07 | 33.27 | 24.99 | 24.99 | 8.29 | 0.00 |
| S08 | 32.26 | 25.33 | 25.33 | 6.93 | 0.00 |
| S09 | 34.18 | 25.71 | 25.71 | 8.47 | 0.00 |
| S10 | 32.22 | 26.78 | 26.78 | 5.44 | 0.00 |

#### MST-S

| Scene | Scenario I | Scenario II | Scenario III | Gap I->II | Gap II->III |
|-------|-----------|------------|------------|----------|----------|
| S01 | 34.73 | 21.25 | 32.88 | 13.48 | 11.63 |
| S02 | 34.59 | 20.56 | 31.69 | 14.03 | 11.13 |
| S03 | 34.34 | 14.14 | 27.25 | 20.20 | 13.11 |
| S04 | 40.75 | 20.61 | 34.91 | 20.14 | 14.29 |
| S05 | 32.15 | 16.73 | 29.64 | 15.41 | 12.91 |
| S06 | 33.64 | 18.93 | 32.44 | 14.71 | 13.51 |
| S07 | 32.56 | 18.28 | 30.43 | 14.28 | 12.15 |
| S08 | 31.73 | 19.70 | 30.78 | 12.04 | 11.09 |
| S09 | 33.54 | 17.63 | 30.49 | 15.91 | 12.86 |
| S10 | 31.79 | 17.52 | 30.30 | 14.27 | 12.78 |

#### MST-L

| Scene | Scenario I | Scenario II | Scenario III | Gap I->II | Gap II->III |
|-------|-----------|------------|------------|----------|----------|
| S01 | 35.29 | 21.18 | 32.86 | 14.12 | 11.69 |
| S02 | 36.14 | 20.41 | 32.67 | 15.72 | 12.26 |
| S03 | 35.66 | 14.00 | 30.44 | 21.65 | 16.44 |
| S04 | 40.05 | 20.20 | 35.10 | 19.85 | 14.90 |
| S05 | 32.84 | 17.16 | 30.54 | 15.68 | 13.37 |
| S06 | 34.55 | 18.92 | 33.00 | 15.64 | 14.08 |
| S07 | 33.80 | 18.18 | 31.87 | 15.62 | 13.70 |
| S08 | 32.74 | 19.34 | 31.43 | 13.40 | 12.08 |
| S09 | 34.37 | 17.70 | 32.54 | 16.68 | 14.84 |
| S10 | 32.63 | 17.41 | 30.81 | 15.22 | 13.41 |

---

## 3. Analysis

### 3.1 Scenario Hierarchy Verification

The expected hierarchy **PSNR_I > PSNR_III > PSNR_II** is confirmed for mask-aware methods (GAP-TV, MST-S, MST-L) across all 10 scenes.

For HDNet (mask-oblivious), Scenario III = Scenario II for all scenes, since HDNet does not use the reconstruction mask as input.

### 3.2 Mask-Awareness Classification

A key finding is the distinction between **mask-aware** and **mask-oblivious** reconstruction methods:

| Method | Mask Input | Oracle Gain (II->III) | Classification |
|--------|-----------|---------------------|----------------|
| GAP-TV | Yes (Phi) | +3.98 dB | Mask-aware |
| HDNet | No | +0.00 dB | Mask-oblivious |
| MST-S | Yes (shifted mask) | +12.55 dB | Mask-aware |
| MST-L | Yes (shifted mask) | +13.68 dB | Mask-aware |

**HDNet** takes only the initial spectral estimate (28 channels from shift_back) as input and ignores the mask entirely. Its reconstruction quality under mismatch depends solely on the measurement, not the assumed operator. This makes HDNet inherently unable to benefit from oracle mask knowledge.

**MST** explicitly takes the shifted mask as a second input, enabling dramatic scenario differentiation. **GAP-TV** uses the mask in its forward/adjoint operators, achieving moderate oracle gains.

### 3.3 Mismatch Degradation (Gap I -> II)

| Method | Gap I->II (dB) | Interpretation |
|--------|---------------|----------------|
| GAP-TV | 5.35 +/- 2.41 | Moderate degradation (classical, lower capacity) |
| HDNet | 8.72 +/- 3.20 | Significant degradation |
| MST-S | 15.45 +/- 2.56 | Severe degradation (most mask-dependent) |
| MST-L | 16.36 +/- 2.39 | Severe degradation (most mask-dependent) |

MST models suffer the largest degradation (15-16 dB) because their learned features are tightly coupled to the mask input. When the assumed mask differs significantly from the true measurement mask, MST's attention mechanism produces highly corrupted reconstructions. GAP-TV, as a classical iterative method, is more robust to mismatch but has lower overall capacity.

### 3.4 Oracle Recovery (Gap II -> III)

| Method | Gap II->III (dB) | Recovery as % of degradation |
|--------|----------------|------------------------------|
| GAP-TV | +3.98 +/- 1.43 | 74% of 5.35 dB loss |
| HDNet | +0.00 +/- 0.00 | 0% (mask-oblivious) |
| MST-S | +12.55 +/- 0.99 | 81% of 15.45 dB loss |
| MST-L | +13.68 +/- 1.39 | 84% of 16.36 dB loss |

This is the central result. When provided with the true (oracle) mask parameters:
- **MST-L recovers 84% of its mismatch loss** (+13.68 dB gain), demonstrating that mask-guided transformers are highly effective at utilizing correct operator information.
- **MST-S recovers 81%** (+12.55 dB), confirming this is an architectural property rather than model-size dependent.
- **GAP-TV recovers 74%** (+3.98 dB), showing classical methods also benefit substantially from correct operator knowledge.
- **HDNet recovers 0%**, confirming it is entirely mask-oblivious.

### 3.5 Residual Gap (III -> I)

The residual gap between oracle and ideal represents unrecoverable losses due to noise and measurement corruption:

| Method | Residual Gap (dB) |
|--------|------------------|
| GAP-TV | 1.38 |
| HDNet | 8.72 |
| MST-S | 2.90 |
| MST-L | 2.68 |

GAP-TV and MST models have small residual gaps (1.4-2.9 dB), meaning oracle performance nearly matches ideal. This demonstrates that the forward model mismatch is the dominant source of degradation, not noise or measurement corruption. HDNet's large residual (8.72 dB) reflects its inability to use the oracle mask.

### 3.6 Method Comparison by Scenario

**Scenario I (Ideal, no mismatch):**
1. MST-L: 34.81 dB
2. HDNet: 34.66 dB
3. MST-S: 33.98 dB
4. GAP-TV: 25.45 dB

**Scenario II (Mismatch, no correction):**
1. HDNet: 25.94 dB
2. GAP-TV: 20.10 dB
3. MST-S: 18.54 dB
4. MST-L: 18.45 dB

**Scenario III (Oracle mask):**
1. MST-L: 32.13 dB
2. MST-S: 31.08 dB
3. HDNet: 25.94 dB
4. GAP-TV: 24.08 dB

Under ideal conditions, MST-L leads. Under mismatch without correction, HDNet is most robust (being mask-oblivious, it only suffers from corrupted measurements, not corrupted mask). Under oracle correction, MST-L regains its lead with a dramatic 13.68 dB improvement.

### 3.7 SSIM Analysis

SSIM trends mirror PSNR findings:
- MST models achieve near-ideal SSIM (0.92-0.93) under oracle correction, recovering from 0.65 under mismatch
- GAP-TV SSIM recovery: 0.60 -> 0.73 (+0.14)
- HDNet SSIM unchanged: 0.84 in both scenarios II and III
- The SSIM recovery for MST-L (+0.28, from 0.65 to 0.93) represents a dramatic structural quality improvement

---

## 4. Key Findings for InverseNet Paper

1. **Mask-awareness is the decisive factor for oracle recovery.** MST models recover 81-84% of their mismatch loss (+12.5-13.7 dB) when given oracle mask knowledge. HDNet, being mask-oblivious, recovers 0%. This creates a clear architectural taxonomy for CASSI reconstruction methods.

2. **MST models are most sensitive to operator mismatch** (15-16 dB degradation), but also benefit the most from correction (+12.5-13.7 dB recovery). This high sensitivity/high recovery pattern makes them ideal candidates for calibration-assisted reconstruction.

3. **HDNet is the most robust under uncorrected mismatch** (25.94 dB in Scenario II) because it ignores the mask entirely. However, this robustness comes at the cost of being unable to leverage operator correction, making it suboptimal when calibration is available.

4. **GAP-TV provides a balanced classical baseline** with moderate mismatch sensitivity (5.35 dB) and meaningful oracle recovery (+3.98 dB, 74%). Its performance is consistent and predictable.

5. **The scenario hierarchy I > III > II holds universally** for mask-aware methods across all 10 scenes, validating the experimental framework. The large gaps (especially 13.68 dB for MST-L) provide strong motivation for operator calibration in CASSI systems.

6. **Residual gaps are small for mask-aware methods** (1.4-2.9 dB), confirming that mismatch -- not noise -- is the dominant source of degradation. This validates the low-noise experimental design.

---

## 5. Generated Artifacts

### Results Files
- `results/cassi_validation_results.json` -- Per-scene detailed results (10 scenes, 4 methods)
- `results/cassi_summary.json` -- Aggregated statistics
- `results/phase3_scenario_results.json` -- Raw per-scene data
- `results/phase3_summary.json` -- Raw summary statistics

### Figures (7 total)
- `figures/cassi/scenario_comparison.png` -- PSNR bar chart (4 methods x 3 scenarios)
- `figures/cassi/method_comparison_heatmap.png` -- PSNR + SSIM heatmaps
- `figures/cassi/gap_comparison.png` -- Degradation and recovery bar charts
- `figures/cassi/psnr_distribution.png` -- PSNR boxplot across scenes
- `figures/cassi/per_scene_psnr.png` -- Per-scene PSNR line plots (2x2 grid)
- `figures/cassi/ssim_comparison.png` -- SSIM bar chart across scenarios
- `figures/cassi/oracle_gain_per_scene.png` -- Oracle gain per scene

### Tables
- `tables/cassi_results_table.csv` -- LaTeX-ready results table

---

## 6. Execution Details

- **Total scenes:** 10 KAIST TSA simulated scenes
- **Total reconstructions:** 120 (10 scenes x 3 scenarios x 4 methods)
- **Total execution time:** ~15.3 minutes (GPU)
- **Device:** CUDA GPU
- **Framework:** PWM (Physics World Model) benchmark infrastructure
- **GAP-TV:** 100 iterations, tv_weight=4.0, shifted-domain with Chambolle TV denoiser
- **Deep learning models:** Pretrained weights, inference only
