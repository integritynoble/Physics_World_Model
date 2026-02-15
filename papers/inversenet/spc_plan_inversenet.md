# InverseNet ECCV: SPC (Single-Pixel Camera) Validation Plan

**Document Version:** 1.0
**Date:** 2026-02-15
**Purpose:** Comprehensive validation of single-pixel camera (SPC) reconstruction methods under operator mismatch

---

## Executive Summary

This document details the validation framework for Single-Pixel Camera (SPC) reconstruction methods in the context of the InverseNet ECCV paper. The benchmark compares **3 reconstruction methods** across **3 scenarios** using **Set11 natural images** (11 standard images at 64×64 resolution), evaluating reconstruction quality under realistic operator mismatch without calibration.

**Key Features:**
- **3 Scenarios:** I (Ideal), II (Assumed/Baseline), IV (Truth Forward Model)
- **Skip Scenario III:** Calibration algorithms not needed for Inversenet
- **3 Methods:** ISTA-Net+, HATNet, ADMM (classical)
- **11 Images:** Set11 benchmark (256×256 center-crops to 64×64, no cropping due to dispersion)
- **Compression Ratio:** 15% (614 out of 4096 measurements)
- **Metrics:** PSNR (dB), SSIM (0-1)
- **Total Reconstructions:** 99 (11 images × 3 scenarios × 3 methods)

**Expected Outcome:** Quantify reconstruction quality hierarchy and solver robustness to DMD/sensor operator mismatch, enabling fair comparison across methods without calibration correction.

---

## 1. Problem Formulation

### 1.1 Forward Model

SPC forward model with physical encoding:

```
y = A(x) + n
```

Where:
- **x** ∈ ℝ^{64×64}: 2D grayscale natural image
- **A**: Physical SPC forward operator (DMD patterns + projection optics + sensor)
  - Spatial projection optics: PSF blur, vignetting, throughput
  - DMD pattern encoding: 614 random ±1 binary masks with spatial warp
  - Photodetector: Quantum efficiency, gain, dark current
  - ADC quantization: 14-bit precision
- **y** ∈ ℝ^{614}: Measurement vector (scalar per pattern)
- **n**: Poisson shot + Gaussian read + quantization noise

### 1.2 Operator Mismatch

In practice, the reconstruction operator `A_assumed` differs from truth `A_true` due to:

| Factor | Parameter | Range | Impact |
|--------|-----------|-------|--------|
| DMD x-shift | mask_dx | ±2.0 px | ~0.12 dB/px |
| DMD y-shift | mask_dy | ±2.0 px | ~0.12 dB/px |
| DMD rotation | mask_theta | ±0.6° | ~3.0 dB/degree |
| Clock offset | clock_offset | ±0.25 units | ~0.5 dB |
| Illumination drift | illum_drift_linear | ±0.15 frac/seq | ~0.3 dB |
| Gain miscalibration | gain | ±0.25 ratio | ~0.2 dB |

### 1.3 Measurement Generation

For Scenarios II & IV, we inject mismatch into the measurement:

```
y_corrupt = A_mismatch(x) + n
```

Where A_mismatch applies true misalignment parameters, creating degradation that reconstructors must overcome.

---

## 2. Scenario Definitions

### Scenario I: Ideal (Oracle)

**Purpose:** Theoretical upper bound for perfect measurements

**Configuration:**
- **Measurement:** y_ideal from ideal DMD patterns and sensor
- **Forward model:** Physical SPC operator with ideal parameters
- **Reconstruction:** Each method using perfect operator knowledge
- **Mismatch:** None (mask_dx=0, mask_dy=0, mask_theta=0, clock_offset=0, etc.)

**Expected PSNR (clean, no noise):**
- ADMM: ~28.0 dB
- ISTA-Net+: ~32.0 dB
- HATNet: ~33.0 dB

### Scenario II: Assumed/Baseline (Uncorrected Mismatch)

**Purpose:** Realistic baseline showing degradation from uncorrected operator mismatch

**Configuration:**
- **Measurement:** y_corrupt with injected mismatch + realistic noise
  - Mismatch injected via DMD warping: (mask_dx=0.4 px, mask_dy=0.4 px, mask_theta=0.08°)
  - Additional: clock_offset=0.06, illum_drift_linear=0.04, gain=1.08
  - Noise: Poisson (peak=50000) + Gaussian (σ=0.005) + 14-bit quantization
- **Forward model:** Physical SPC operator with real DMD patterns
- **Reconstruction:** Each method assuming perfect operator (mask_dx=0, etc.)
- **Key insight:** Methods don't "know" about mismatch, so reconstruction is degraded

**Expected PSNR:**
- All methods degrade ~2-4 dB compared to Scenario I
- Example: ADMM ~25 dB, ISTA-Net+ ~29 dB, HATNet ~30 dB

### Scenario IV: Truth Forward Model (Oracle Operator)

**Purpose:** Upper bound for corrupted measurements when true mismatch is known

**Configuration:**
- **Measurement:** Same y_corrupt as Scenario II
- **Forward model:** Physical SPC operator with TRUE mismatch parameters
  - DMD: warped by (mask_dx=0.4, mask_dy=0.4, mask_theta=0.08°)
  - Temporal: clock_offset=0.06, illum_drift_linear=0.04
  - Sensor: gain=1.08
  - Methods use correct operator reflecting actual hardware state
- **Reconstruction:** Each method using oracle operator
- **Key insight:** Shows recovery possible if system were perfectly characterized

**Expected PSNR:**
- Partial recovery from Scenario II (better than baseline but worse than ideal)
- Gap II→IV: ~1-2 dB (method-dependent robustness)
- Example: ADMM ~27 dB, ISTA-Net+ ~30 dB, HATNet ~31 dB

### Comparison: Scenario Hierarchy

For each method:
```
PSNR_I (ideal) > PSNR_IV (oracle mismatch) > PSNR_II (baseline uncorrected)
```

**Gaps quantify:**
- **Gap I→II:** Mismatch impact (how much measurement quality degrades)
- **Gap II→IV:** Operator awareness (how much better with true operator)
- **Gap IV→I:** Residual noise/solver limitation

---

## 3. Mismatch Parameters

### Injected Mismatch

**Values:** mask_dx=0.4 px, mask_dy=0.4 px, mask_theta=0.08°, clock_offset=0.06, illum_drift_linear=0.04, gain=1.08

**Rationale:**
- Realistic assembly tolerance for DMD-based SPC (~±2mm, ~±0.5° mechanical error)
- Clock synchronization typical variance in FPGA timing systems
- Illumination drift from LED power supply ripple
- Gain from photodetector sensitivity variation
- Expected PSNR degradation: 2-4 dB (verified from spc.md analysis)
- Sufficient to see measurable solver robustness differences

### Bounds and Uncertainty

From spc.md W2 analysis:
```
mask_dx ∈ [-2.0, 2.0] px       → selected 0.4 px (low-moderate)
mask_dy ∈ [-2.0, 2.0] px       → selected 0.4 px (low-moderate)
mask_theta ∈ [-0.6, 0.6]°      → selected 0.08° (very low)
clock_offset ∈ [-0.25, 0.25]   → selected 0.06 (low)
illum_drift_linear ∈ [-0.15, 0.15] → selected 0.04 (low)
gain ∈ [0.75, 1.25]            → selected 1.08 (low-moderate)
```

**Why these values:** This specific combination provides realistic but recoverable mismatch, enabling clear differentiation between reconstruction methods without being so severe that all methods fail.

---

## 4. Reconstruction Methods

### Method 1: ADMM (Classical Baseline)

**Category:** Iterative optimization with alternating direction method of multipliers

**Implementation:** `pwm_core.recon.admm.admm_spc()`

**Parameters:**
- Iterations: 100 (balanced for speed/quality)
- Rho: 1.0 (proximal parameter)
- Basis pursuit formulation: min ||x||_1 s.t. ||Ax - y||_2 ≤ ε

**Expected Performance:**
- Scenario I: 28.0 ± 0.05 dB
- Scenario II: 25.2 ± 0.08 dB (gap 2.8 dB)
- Scenario IV: 27.0 ± 0.06 dB (recovery 1.8 dB)

**Rationale:** Established classical baseline, no deep learning dependency, well-understood convergence

---

### Method 2: ISTA-Net+

**Category:** Deep unrolled ISTA algorithm

**Implementation:** `pwm_core.recon.ista_net_plus.ista_net_plus_spc()`

**Architecture:**
- Unrolled ISTA iterations: 30 layers
- Learnable soft-thresholding in each layer
- ~0.5M learnable parameters
- Pre-trained on Set11 and synthetic data

**Expected Performance:**
- Scenario I: 32.0 ± 0.03 dB
- Scenario II: 29.2 ± 0.05 dB (gap 2.8 dB)
- Scenario IV: 30.5 ± 0.04 dB (recovery 1.3 dB)

**Rationale:** Deep unrolling maintains interpretability while leveraging learned priors for signal sparsity

---

### Method 3: HATNet (Hybrid Attention Transformer)

**Category:** Vision Transformer with hybrid attention

**Implementation:** `pwm_core.recon.hatnet.hatnet_spc()`

**Architecture:**
- Hybrid spatial-spectral attention (even though SPC is 2D)
- Multi-stage refinement blocks
- ~1.2M learnable parameters
- Pre-trained on Set11 dataset

**Expected Performance:**
- Scenario I: 33.0 ± 0.02 dB
- Scenario II: 30.2 ± 0.04 dB (gap 2.8 dB)
- Scenario IV: 31.5 ± 0.03 dB (recovery 1.3 dB)

**Rationale:** Transformer-based architecture captures long-range dependencies in sparse reconstruction

---

## 5. Forward Model Specification

### Physical SPC Operator Chain

**Class:** `pwm_core.calibration.spc_operator.PhysicalSPCOperator`

**Configuration:**
- **Image size:** 64×64 (N=4096 pixels)
- **Measurement size:** 614 (M=614 random projections, 15% compression)
- **DMD pattern type:** Random ±1 binary masks
- **Projection optics:**
  - Throughput: 0.95
  - PSF blur sigma: 0.8 pixels
  - Vignetting: 0.02
- **Photodetector:**
  - Quantum efficiency: 0.85
  - Gain: 1.0 (nominal)
  - Dark current: 0.01 (normalized)

### Mask Handling

**Scenario I (Ideal):**
- Mask source: Ideal random ±1 binary patterns
- No mismatch: mask_dx=0, mask_dy=0, mask_theta=0
- Represents perfect laboratory setup

**Scenarios II & IV (Real/Corrupted):**
- Mask source: Real DMD pattern set with potential misalignment
- For Scenario II: Used as-is (assumes perfect alignment)
- For Scenario IV: Warped by (mask_dx=0.4, mask_dy=0.4, mask_theta=0.08°)
- Represents hardware with realistic misalignment

### Noise Model

**Poisson + Gaussian + Quantization:**
```
y_noisy = Quantize(Poisson(y_scaled / peak) + Gaussian(0, σ), bits=14)
```

**Parameters:**
- Photon peak: 50000 (realistic photodiode saturation)
- Read noise std: σ=0.005 (typical CMOS readout noise)
- ADC bit depth: 14 (standard industrial sensor)
- Combined SNR: ~10 dB (realistic operating point)

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

**Implementation:** Computed on full 64×64 images

**Interpretation:**
- 1.0: Perfect reconstruction
- 0.8-1.0: Excellent perceptual quality
- 0.6-0.8: Good quality
- <0.6: Perceptually degraded

---

## 7. Expected Results Summary

### PSNR Hierarchy (Mean ± Std across 11 images)

| Method | Scenario I | Scenario II | Scenario IV | Gap I→II | Gap II→IV |
|--------|-----------|-----------|-----------|---------|----------|
| ADMM | 28.00±0.05 | 25.20±0.08 | 27.00±0.06 | 2.80 | 1.80 |
| ISTA-Net+ | 32.00±0.03 | 29.20±0.05 | 30.50±0.04 | 2.80 | 1.30 |
| HATNet | 33.00±0.02 | 30.20±0.04 | 31.50±0.03 | 2.80 | 1.30 |

### Key Insights

1. **Deep learning advantage persistent:** HATNet maintains ~5 dB edge over ADMM even under mismatch
2. **Mismatch impact uniform:** Gap I→II is ~2.8 dB across all methods (mismatch is fundamental)
3. **Solver robustness:** Gap II→IV ~1.3-1.8 dB (moderate recovery with known operator)
4. **Method ranking stable:** HATNet > ISTA-Net+ > ADMM in all scenarios

---

## 8. Deliverables

### Data Files

1. **spc_validation_results.json** (11 images × 3 scenarios × 3 methods × 2 metrics)
   - Per-image detailed results
   - Per-scenario aggregated statistics
   - Parameter recovery information

2. **spc_summary.json** (aggregated statistics)
   - Mean PSNR/SSIM per scenario per method
   - Standard deviations across 11 images
   - Gaps and recovery metrics

### Visualization Files

3. **figures/spc/scenario_comparison.png** (bar chart)
   - X-axis: Scenarios (I, II, IV)
   - Y-axis: PSNR (dB)
   - Groups: 3 methods (different colors)

4. **figures/spc/method_comparison.png** (heatmap)
   - Rows: 3 methods (ADMM, ISTA-Net+, HATNet)
   - Cols: 3 scenarios (I, II, IV)
   - Values: PSNR (dB, color-coded)

5. **figures/spc/image{01-11}_*.png** (99 images)
   - 11 images × 3 scenarios × 3 reconstructions
   - Format: Grayscale 64×64
   - Organized by scenario

### Table Files

6. **tables/spc_results_table.csv** (LaTeX-ready)
   - Rows: Methods
   - Cols: Mean PSNR per scenario + gaps
   - Format: CSV with ± standard deviations

---

## 9. Validation Workflow

### Step 1: Load Dataset

```python
# Load 11 Set11 images
images = [load_set11_image(i) for i in range(1, 12)]  # (256,256) each

# Center-crop to 64x64 (no full crop needed due to dispersion)
images_64 = [img[96:160, 96:160] for img in images]  # (64,64) each

# Load DMD patterns
patterns_ideal = load_dmd_patterns("ideal_set")   # Ideal
patterns_real = load_dmd_patterns("real_set")     # Real (with potential misalignment)
```

### Step 2: Validate Each Image

For each of 11 images:

1. **Scenario I:** Ideal measurement & reconstruction
2. **Scenario II:** Corrupted measurement, uncorrected operator
3. **Scenario IV:** Corrupted measurement, truth operator

For each scenario, reconstruct with all 3 methods, compute PSNR/SSIM

### Step 3: Aggregate Results

Compute per-method and per-scenario statistics:
- Mean PSNR/SSIM
- Standard deviations
- Confidence intervals (if needed)

### Step 4: Generate Visualizations

Create all PNG and CSV output files as specified in Deliverables section

---

## 10. Implementation Files

### Main Script
- `scripts/validate_spc_inversenet.py` (main validation engine)

### Visualization Script
- `scripts/generate_spc_figures.py` (creates PNG/CSV outputs)

### Supporting Scripts
- `scripts/plot_utils.py` (common plotting utilities)
- `scripts/loader.py` (Set11 dataset loading helpers)

### Documentation
- `spc_plan_inversenet.md` (this file)

---

## 11. Execution Timeline

| Phase | Duration | Task |
|-------|----------|------|
| Setup | 20 min | Create directory structure, install dependencies |
| Validation | 90 min | Run validate_spc_inversenet.py (11 images × 3 scenarios × 3 methods) |
| Visualization | 20 min | Generate figures and tables |
| QA | 20 min | Verify results, check for anomalies |
| **Total** | **~2.5 hours** | End-to-end execution |

**GPU requirement:** NVIDIA CUDA GPU recommended for HATNet/ISTA-Net+ (optional for ADMM)

---

## 12. Quality Assurance

### Verification Checks

1. **Dataset Loading:** All 11 images load correctly (64×64)
2. **PSNR Hierarchy:** Verify I > IV > II for all methods
3. **Consistency:** Std dev < 0.1 dB across images (low noise in results)
4. **Method Ranking:** HATNet > ISTA-Net+ > ADMM (established order)
5. **Gap Similarity:** Gap I→II ~2.8 dB for all methods (uniform mismatch effect)

### Expected Anomalies (if any)

- **Deep learning variance:** Transformer methods may show slight variance across runs
- **Solver convergence:** ADMM convergence time varies (~10-30 sec/method)
- **GPU memory:** HATNet requires ~4-6 GB VRAM at full capacity

---

## 13. Citation & References

**Key References:**
- Set11 Dataset: Natural image benchmark for compressive sensing
- SPC forward model: PhysicalSPCOperator (pwm_core.calibration)
- Reconstruction methods: ISTA-Net+, HATNet from standard benchmarks
- Metrics: PSNR (ITU-R BT.601), SSIM (Wang et al., 2004)

**Related Documents:**
- `pwm/reports/spc.md` – Full SPC modality report
- `pwm/reports/spc_baseline_results.json` – Baseline metrics

---

## 14. Appendix: Notation

| Symbol | Meaning |
|--------|---------|
| x | 2D grayscale image (64×64) |
| y | Measurement vector (614 scalars) |
| A | Forward model operator (projection + sensor) |
| mask_dx, mask_dy, mask_theta | DMD affine misalignment |
| M, N | Measurement count, image pixel count |
| PSNR | Peak Signal-to-Noise Ratio (dB) |
| SSIM | Structural Similarity Index (0-1) |

---

**Document prepared for InverseNet ECCV benchmark**
*For questions or updates, refer to main PWM project documentation.*
