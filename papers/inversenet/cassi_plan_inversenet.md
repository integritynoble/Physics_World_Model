# InverseNet ECCV: CASSI Validation Plan

**Document Version:** 1.0
**Date:** 2026-02-15
**Purpose:** Comprehensive validation of CASSI reconstruction methods under operator mismatch

---

## Executive Summary

This document details the validation framework for CASSI (Coded Aperture Snapshot Spectral Imaging) reconstruction methods in the context of the InverseNet ECCV paper. The benchmark compares **4 reconstruction methods** across **3 scenarios** using **10 KAIST hyperspectral scenes**, evaluating reconstruction quality under realistic operator mismatch without calibration.

**Key Features:**
- **3 Scenarios:** I (Ideal), II (Assumed/Baseline), III (Truth Forward Model)
- **Skip Scenario III:** Calibration algorithms (Alg1 & Alg2) not needed for Inversenet
- **4 Methods:** GAP-TV (classical), HDNet, MST-S, MST-L (deep learning)
- **10 Scenes:** 256×256×28 hyperspectral KAIST dataset
- **Metrics:** PSNR (dB), SSIM (0-1), SAM (degrees)
- **Total Reconstructions:** 120 (10 scenes × 3 scenarios × 4 methods)

**Expected Outcome:** Quantify reconstruction quality hierarchy and solver robustness to operator mismatch, enabling fair comparison across methods without calibration correction.

---

## 1. Problem Formulation

### 1.1 Forward Model

CASSI forward model with enlarged simulation grid:

```
y = H_true(x) + n
```

Where:
- **x** ∈ ℝ^{256×256×28}: True hyperspectral scene
- **H_true**: SimulatedOperatorEnlargedGrid(N=4 spatial, K=2 spectral, L_expanded=217)
- **y** ∈ ℝ^{256×310}: Measurement (downsampled from enlarged 1024×1240)
- **n**: Poisson shot + Gaussian read noise

### 1.2 Operator Mismatch

In practice, the reconstruction operator `H_assumed` differs from truth `H_true` due to:

| Factor | Parameter | Range | Impact |
|--------|-----------|-------|--------|
| Mask x-shift | dx | ±3 px | ~0.12 dB/px |
| Mask y-shift | dy | ±3 px | ~0.12 dB/px |
| Mask rotation | θ | ±1° | ~3.77 dB/degree |
| Dispersion slope | a₁ | 1.95-2.05 px/band | ~0.2 dB |
| Dispersion offset | α | ±1° | ~0.5 dB |

### 1.3 Measurement Generation

For Scenarios II & III, we inject mismatch into the measurement:

```
y_corrupt = H_mismatch(x) + n
```

Where H_mismatch applies true misalignment parameters, creating degradation that reconstructors must overcome.

---

## 2. Scenario Definitions

### Scenario I: Ideal (Oracle)

**Purpose:** Theoretical upper bound for perfect measurements

**Configuration:**
- **Measurement:** y_ideal from ideal mask (TSA simulation data)
- **Forward model:** SimulatedOperatorEnlargedGrid with ideal mask
- **Reconstruction:** Each method using perfect operator knowledge
- **Mismatch:** None (dx=0, dy=0, θ=0)

**Expected PSNR (clean, no noise):**
- GAP-TV: ~32.0 dB
- HDNet: ~35.0 dB
- MST-S: ~34.0 dB
- MST-L: ~36.0 dB

### Scenario II: Assumed/Baseline (Uncorrected Mismatch)

**Purpose:** Realistic baseline showing degradation from uncorrected operator mismatch

**Configuration:**
- **Measurement:** y_corrupt with injected mismatch + realistic noise
  - Mismatch injected via mask warping: (dx=0.5 px, dy=0.3 px, θ=0.1°)
  - Noise: Poisson (peak=10000) + Gaussian (σ=1.0 dB SNR)
- **Forward model:** SimulatedOperatorEnlargedGrid with real mask
- **Reconstruction:** Each method assuming perfect mask (dx=0, dy=0, θ=0)
- **Key insight:** Methods don't "know" about mismatch, so reconstruction is degraded

**Expected PSNR:**
- All methods degrade ~3-5 dB compared to Scenario I
- Example: GAP-TV ~28 dB, HDNet ~31 dB, MST-S ~30 dB, MST-L ~32 dB

### Scenario III: Truth Forward Model (Oracle Operator)

**Purpose:** Upper bound for corrupted measurements when true mismatch is known

**Configuration:**
- **Measurement:** Same y_corrupt as Scenario II
- **Forward model:** SimulatedOperatorEnlargedGrid with TRUE mismatch parameters
  - Mask: warped by (dx=0.5, dy=0.3, θ=0.1°)
  - Methods use correct operator reflecting actual hardware state
- **Reconstruction:** Each method using oracle operator
- **Key insight:** Shows recovery possible if system were perfectly characterized

**Expected PSNR:**
- Partial recovery from Scenario II (better than baseline but worse than ideal)
- Gap II→III: ~1-2 dB (method-dependent robustness)
- Example: GAP-TV ~30 dB, HDNet ~32 dB, MST-S ~32 dB, MST-L ~33 dB

### Comparison: Scenario Hierarchy

For each method:
```
PSNR_I (ideal) > PSNR_IV (oracle mismatch) > PSNR_II (baseline uncorrected)
```

**Gaps quantify:**
- **Gap I→II:** Mismatch impact (how much measurement quality degrades)
- **Gap II→III:** Operator awareness (how much better with true operator)
- **Gap III→I:** Residual noise/solver limitation

---

## 3. Mismatch Parameters

### Injected Mismatch

**Values:** dx=0.5 px, dy=0.3 px, θ=0.1° (moderate severity)

**Rationale:**
- Realistic assembly tolerance for CASSI prototype (~±0.5 mm mechanical error)
- Equivalent to optical bench rotation ~0.1° (achievable misalignment)
- Expected PSNR degradation: 3-5 dB (verified from cassi_plan.md W1-W3 analysis)
- Sufficient to see measurable solver robustness differences

### Bounds and Uncertainty

From cassi_plan.md W1-W5 analysis:
```
dx ∈ [-3, 3] px       → selected 0.5 px (low-moderate)
dy ∈ [-3, 3] px       → selected 0.3 px (low-moderate)
θ ∈ [-1°, 1°]         → selected 0.1° (very low)
a₁ ∈ [1.95, 2.05]     → not corrected in this benchmark
α ∈ [-1°, 1°]         → not corrected in this benchmark
```

**Why 0.5, 0.3, 0.1°:** This specific combination provides realistic but recoverable mismatch, enabling clear differentiation between reconstruction methods without being so severe that all methods fail.

---

## 4. Reconstruction Methods

### Method 1: GAP-TV (Classical Baseline)

**Category:** Iterative algebraic reconstruction

**Implementation:** `pwm_core.recon.gap_tv.gap_tv_cassi()`

**Parameters:**
- Iterations: 50 (balanced for speed/quality)
- TV weight: 0.05 (standard hyperparameter)
- Prox step size: auto-tuned by algorithm

**Expected Performance:**
- Scenario I: 32.1 ± 0.02 dB
- Scenario II: 28.5 ± 0.05 dB (gap 3.6 dB)
- Scenario III: 29.8 ± 0.04 dB (recovery 1.3 dB)

**Rationale:** Established baseline, no deep learning dependency, widely used in HSI reconstruction

---

### Method 2: HDNet (Deep Unrolled)

**Category:** Deep unrolled algorithm (iterative)

**Implementation:** `pwm_core.recon.hdnet.hdnet_recon_cassi()`

**Architecture:**
- Dual-domain unrolled network (iterative refinement)
- 2.37M learnable parameters
- Pre-trained on clean KAIST data

**Expected Performance:**
- Scenario I: 35.0 ± 0.03 dB
- Scenario II: 31.2 ± 0.06 dB (gap 3.8 dB)
- Scenario III: 32.5 ± 0.05 dB (recovery 1.3 dB)

**Rationale:** Deep unrolling preserves interpretability while leveraging learned priors

---

### Method 3: MST-S (Transformer Small)

**Category:** Vision Transformer (Spectral)

**Implementation:** `pwm_core.recon.mst.create_mst(variant='mst_s')`

**Architecture:**
- Multi-stage Transformer, compact design
- ~0.9M parameters
- Pre-trained on KAIST dataset

**Expected Performance:**
- Scenario I: 34.2 ± 0.02 dB
- Scenario II: 30.5 ± 0.04 dB (gap 3.7 dB)
- Scenario III: 31.8 ± 0.03 dB (recovery 1.3 dB)

**Rationale:** Lightweight transformer, good speed/accuracy trade-off

---

### Method 4: MST-L (Transformer Large)

**Category:** Vision Transformer (Spectral)

**Implementation:** `pwm_core.recon.mst.create_mst(variant='mst_l')`

**Architecture:**
- Multi-stage Transformer, large capacity
- ~2.0M parameters
- Pre-trained on KAIST dataset
- State-of-the-art on clean reconstructions

**Expected Performance:**
- Scenario I: 36.0 ± 0.02 dB
- Scenario II: 32.3 ± 0.05 dB (gap 3.7 dB)
- Scenario III: 33.6 ± 0.04 dB (recovery 1.3 dB)

**Rationale:** Highest capacity model, best baseline reconstruction quality

---

## 5. Forward Model Specification

### SimulatedOperatorEnlargedGrid

**Class:** `pwm_core.calibration.cassi_upwmi_alg12.SimulatedOperatorEnlargedGrid`

**Configuration:**
- **Spatial enlargement factor:** N=4 (256×256 → 1024×1024)
- **Spectral expansion factor:** K=2 (28 → 217 bands)
- **Dispersion stride:** 1 pixel per frame (fine-grained encoding)
- **Measurement size:** 1024×1240 (enlarged space) → 256×310 (downsampled)

### Mask Handling

**Scenario I (Ideal):**
- Mask source: TSA simulation mask (`TSA_simu_data/mask.mat`)
- No mismatch: dx=0, dy=0, θ=0
- Represents perfect laboratory setup

**Scenarios II & III (Real/Corrupted):**
- Mask source: TSA real data mask (`TSA_real_data/mask.mat`)
- For Scenario II: Used as-is (assumes perfect alignment)
- For Scenario III: Warped by (dx=0.5, dy=0.3, θ=0.1°)
- Represents hardware with realistic misalignment

### Noise Model

**Poisson + Gaussian:**
```
y_noisy = Poisson(y_scaled / peak) + Gaussian(0, σ)
```

**Parameters:**
- Photon peak: 10000 (realistic sensor saturation)
- Read noise std: σ=1.0 dB (typical CMOS readout noise)
- Combined SNR: ~6 dB (realistic operating point)

---

## 6. Evaluation Metrics

### PSNR (Peak Signal-to-Noise Ratio)

**Definition:**
```
PSNR = 10 * log₁₀(max_val² / MSE)  [dB]
```

Where:
- max_val = 1.0 (data normalized to [0,1])
- MSE = mean((x_true - x_recon)²)

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

### SAM (Spectral Angle Mapper)

**Definition:** Per-pixel spectral angle between truth and reconstruction
```
SAM = arccos(x_true · x_recon / (||x_true|| ||x_recon||))  [degrees]
```

**Interpretation:**
- <1°: Excellent spectral fidelity
- 1-2°: Good
- 2-5°: Acceptable
- >5°: Poor spectral accuracy

---

## 7. Expected Results Summary

### PSNR Hierarchy (Mean ± Std across 10 scenes)

| Method | Scenario I | Scenario II | Scenario III | Gap I→II | Gap II→III |
|--------|-----------|-----------|-----------|---------|----------|
| GAP-TV | 32.10±0.02 | 28.50±0.05 | 29.80±0.04 | 3.60 | 1.30 |
| HDNet | 35.00±0.03 | 31.20±0.06 | 32.50±0.05 | 3.80 | 1.30 |
| MST-S | 34.20±0.02 | 30.50±0.04 | 31.80±0.03 | 3.70 | 1.30 |
| MST-L | 36.00±0.02 | 32.30±0.05 | 33.60±0.04 | 3.70 | 1.30 |

### Key Insights

1. **Deep learning advantage persistent:** MST-L maintains ~3-4 dB edge over GAP-TV even under mismatch
2. **Mismatch impact uniform:** Gap I→II is ~3.6-3.8 dB across all methods (mismatch is fundamental)
3. **Solver robustness:** Gap II→III ~1.3 dB (moderate recovery with known operator)
4. **Method ranking stable:** MST-L > HDNet > MST-S > GAP-TV in all scenarios

---

## 8. Deliverables

### Data Files

1. **cassi_validation_results.json** (10 scenes × 3 scenarios × 4 methods × 3 metrics)
   - Per-scene detailed results
   - Per-scenario aggregated statistics
   - Parameter recovery information

2. **cassi_summary.json** (aggregated statistics)
   - Mean PSNR/SSIM/SAM per scenario per method
   - Standard deviations across 10 scenes
   - Gaps and recovery metrics

### Visualization Files

3. **figures/cassi/scenario_comparison.png** (bar chart)
   - X-axis: Scenarios (I, II, III)
   - Y-axis: PSNR (dB)
   - Groups: 4 methods (different colors)

4. **figures/cassi/method_comparison.png** (heatmap)
   - Rows: 4 methods (GAP-TV, HDNet, MST-S, MST-L)
   - Cols: 3 scenarios (I, II, III)
   - Values: PSNR (dB, color-coded)

5. **figures/cassi/scene{01-10}_*.png** (120 images)
   - 10 scenes × 3 scenarios × 4 reconstructions
   - Format: RGB rendering (3 representative bands)
   - Organized by scenario: `scene01_scenario_i_gaptv.png` etc.

6. **figures/cassi/spectral_profiles.png** (spectral comparison)
   - Representative pixel location across all methods
   - 3 subplots (one per scenario)
   - Legend showing method colors

### Table Files

7. **tables/cassi_results_table.csv** (LaTeX-ready)
   - Rows: Methods
   - Cols: Mean PSNR per scenario + gaps
   - Format: CSV with ± standard deviations

---

## 9. Validation Workflow

### Step 1: Load Dataset

```python
# Load 10 scenes from KAIST
scenes = [load_scene(f"scene{i:02d}") for i in range(1, 11)]  # (256,256,28) each

# Load masks
mask_ideal = load_mask("TSA_simu_data/mask.mat")   # Ideal
mask_real = load_mask("TSA_real_data/mask.mat")     # Real (with potential misalignment)
```

### Step 2: Validate Each Scene

For each of 10 scenes:

1. **Scenario I:** Ideal measurement & reconstruction
2. **Scenario II:** Corrupted measurement, uncorrected operator
3. **Scenario III:** Corrupted measurement, truth operator

For each scenario, reconstruct with all 4 methods, compute PSNR/SSIM/SAM

### Step 3: Aggregate Results

Compute per-method and per-scenario statistics:
- Mean PSNR/SSIM/SAM
- Standard deviations
- Confidence intervals (if needed)

### Step 4: Generate Visualizations

Create all PNG and CSV output files as specified in Deliverables section

---

## 10. Implementation Files

### Main Script
- `scripts/validate_cassi_inversenet.py` (main validation engine)

### Visualization Script
- `scripts/generate_cassi_figures.py` (creates PNG/CSV outputs)

### Supporting Scripts
- `scripts/plot_utils.py` (common plotting utilities)
- `scripts/loader.py` (dataset loading helpers)

### Documentation
- `cassi_plan_inversenet.md` (this file)

---

## 11. Execution Timeline

| Phase | Duration | Task |
|-------|----------|------|
| Setup | 30 min | Create directory structure, install dependencies |
| Validation | 2 hours | Run validate_cassi_inversenet.py (10 scenes × 3 scenarios × 4 methods) |
| Visualization | 30 min | Generate figures and tables |
| QA | 30 min | Verify results, check for anomalies |
| **Total** | **~3.5 hours** | End-to-end execution |

**GPU requirement:** NVIDIA CUDA GPU recommended (Transformer methods)

---

## 12. Quality Assurance

### Verification Checks

1. **Dataset Loading:** All 10 scenes load correctly (256×256×28)
2. **PSNR Hierarchy:** Verify I > III > II for all methods
3. **Consistency:** Std dev < 0.1 dB across scenes (low noise in results)
4. **Method Ranking:** MST-L > HDNet > MST-S > GAP-TV (established order)
5. **Gap Similarity:** Gap I→II ~3.6-3.8 dB for all methods (uniform mismatch effect)

### Expected Anomalies (if any)

- **Deep learning variance:** Transformer methods may show slight variance across runs
- **Solver convergence:** GAP-TV convergence time varies (~50-100 sec/method)
- **GPU memory:** MST-L requires ~8-12 GB VRAM at full resolution

---

## 13. Citation & References

**Key References:**
- KAIST HSI Dataset: [Citation]
- CASSI forward model: SimulatedOperatorEnlargedGrid (pwm_core.calibration)
- Reconstruction methods: MST-L, HDNet from standard benchmarks
- Metrics: PSNR (ITU-R BT.601), SSIM (Wang et al., 2004), SAM (Pande & Markham, 2016)

**Related Documents:**
- `docs/cassi_plan.md` – Full CASSI calibration plan (v4+)
- `pwm/reports/cassi_report.md` – Algorithm 1 & 2 validation report

---

## 14. Appendix: Notation

| Symbol | Meaning |
|--------|---------|
| x | Hyperspectral scene (256×256×28) |
| y | Measurement (256×310 downsampled) |
| H | Forward model operator |
| dx, dy, θ | Mask affine misalignment |
| N, K | Spatial/spectral enlargement factors |
| PSNR | Peak Signal-to-Noise Ratio (dB) |
| SSIM | Structural Similarity Index (0-1) |
| SAM | Spectral Angle Mapper (degrees) |

---

**Document prepared for InverseNet ECCV benchmark**
*For questions or updates, refer to main PWM project documentation.*
