# InverseNet ECCV: CACTI (Coded Aperture Compressive Temporal Imaging) Validation Plan

**Document Version:** 1.0
**Date:** 2026-02-15
**Purpose:** Comprehensive validation of CACTI reconstruction methods under operator mismatch

---

## Executive Summary

This document details the validation framework for CACTI (Coded Aperture Compressive Temporal Imaging) reconstruction methods in the context of the InverseNet ECCV paper. The benchmark compares **4 reconstruction methods** across **3 scenarios** using **6 standard test scenes** from the SCI Video Benchmark, evaluating reconstruction quality under realistic operator mismatch without calibration.

**Key Features:**
- **3 Scenarios:** I (Ideal), II (Assumed/Baseline), III (Truth Forward Model)
- **Skip Scenario III:** Calibration algorithms not needed for Inversenet
- **4 Methods:** GAP-TV (classical), PnP-FFDNet, ELP-Unfolding, EfficientSCI (deep learning)
- **6 Scenes:** SCI Video Benchmark (256×256×T, 8:1 compression, no cropping due to dispersion)
- **Metrics:** PSNR (dB), SSIM (0-1)
- **Total Reconstructions:** 72 (6 scenes × 3 scenarios × 4 methods)

**Expected Outcome:** Quantify reconstruction quality hierarchy and solver robustness to coded aperture operator mismatch, enabling fair comparison across methods without calibration correction.

---

## 1. Problem Formulation

### 1.1 Forward Model

CACTI forward model with temporal encoding:

```
y = H(x) + n
```

Where:
- **x** ∈ ℝ^{256×256×T}: 3D video cube (T frames, typically T=4-6)
- **H**: Physical CACTI forward operator
  - Temporal encoding: T time-varying binary masks (DMD/LCD driven)
  - Spatial optics: lens throughput, PSF blur, vignetting
  - Integration: temporal accumulation with duty cycle and clock offset
  - Sensor: quantum efficiency, gain, offset
  - Quantization: 12-bit ADC
- **y** ∈ ℝ^{256×256}: 2D measurement snapshot (all frames summed/integrated)
- **n**: Poisson shot + Gaussian read + quantization noise

### 1.2 Operator Mismatch

In practice, the reconstruction operator `H_assumed` differs from truth `H_true` due to:

| Factor | Parameter | Range | Impact |
|--------|-----------|-------|--------|
| Mask x-shift | mask_dx | ±3.0 px | ~0.12 dB/px |
| Mask y-shift | mask_dy | ±3.0 px | ~0.12 dB/px |
| Mask rotation | mask_theta | ±0.6° | ~3.5 dB/degree |
| Mask blur | mask_blur_sigma | 0-2.0 px | ~0.15 dB/px |
| Clock offset | clock_offset | ±0.5 frames | ~0.5 dB |
| Duty cycle | duty_cycle | 0.7-1.0 | ~0.3 dB |
| Gain error | gain | ±0.25 ratio | ~0.2 dB |

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
- **Measurement:** y_ideal from ideal masks and sensor
- **Forward model:** Physical CACTI operator with ideal parameters
- **Reconstruction:** Each method using perfect operator knowledge
- **Mismatch:** None (mask_dx=0, mask_dy=0, mask_theta=0, clock_offset=0, etc.)

**Expected PSNR (clean, no noise):**
- GAP-TV: ~24.0 dB
- PnP-FFDNet: ~30.0 dB
- ELP-Unfolding: ~34.0 dB
- EfficientSCI: ~36.0 dB

### Scenario II: Assumed/Baseline (Uncorrected Mismatch)

**Purpose:** Realistic baseline showing degradation from uncorrected operator mismatch

**Configuration:**
- **Measurement:** y_corrupt with injected mismatch + realistic noise
  - Mismatch injected via mask warping: (mask_dx=1.5 px, mask_dy=1.0 px, mask_theta=0.3°, mask_blur_sigma=0.3 px)
  - Temporal: clock_offset=0.08 frames, duty_cycle=0.92
  - Sensor: gain=1.05, offset=0.005
  - Noise: Poisson (peak=10000) + Gaussian (σ=5.0) + 12-bit quantization
- **Forward model:** Physical CACTI operator with real masks
- **Reconstruction:** Each method assuming perfect operator (mask_dx=0, etc.)
- **Key insight:** Methods don't "know" about mismatch, so reconstruction is degraded

**Expected PSNR:**
- All methods degrade ~3-5 dB compared to Scenario I
- Example: GAP-TV ~20 dB, PnP-FFDNet ~26 dB, ELP-Unfolding ~30 dB, EfficientSCI ~32 dB

### Scenario III: Truth Forward Model (Oracle Operator)

**Purpose:** Upper bound for corrupted measurements when true mismatch is known

**Configuration:**
- **Measurement:** Same y_corrupt as Scenario II
- **Forward model:** Physical CACTI operator with TRUE mismatch parameters
  - Masks: warped by (mask_dx=1.5, mask_dy=1.0, mask_theta=0.3°, mask_blur_sigma=0.3)
  - Temporal: clock_offset=0.08, duty_cycle=0.92
  - Sensor: gain=1.05, offset=0.005
  - Methods use correct operator reflecting actual hardware state
- **Reconstruction:** Each method using oracle operator
- **Key insight:** Shows recovery possible if system were perfectly characterized

**Expected PSNR:**
- Partial recovery from Scenario II (better than baseline but worse than ideal)
- Gap II→III: ~1-2 dB (method-dependent robustness)
- Example: GAP-TV ~22 dB, PnP-FFDNet ~28 dB, ELP-Unfolding ~31 dB, EfficientSCI ~33 dB

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

**Values:** mask_dx=1.5 px, mask_dy=1.0 px, mask_theta=0.3°, mask_blur_sigma=0.3 px, clock_offset=0.08 fr, duty_cycle=0.92, gain=1.05, offset=0.005

**Rationale:**
- Realistic assembly tolerance for DMD-based CACTI (~±1.5 mm mechanical error, ~±0.3° rotation)
- Clock synchronization typical variance in frame sync systems
- Mask edge blur from optical defocus or DMD fill factor
- Duty cycle incomplete due to digital delay or dead time
- Gain and offset from detector calibration uncertainty
- Expected PSNR degradation: 3-5 dB (verified from cacti.md W2 analysis)
- Sufficient to see measurable solver robustness differences

### Bounds and Uncertainty

From cacti.md W2 analysis:
```
mask_dx ∈ [-3, 3] px               → selected 1.5 px (low-moderate)
mask_dy ∈ [-3, 3] px               → selected 1.0 px (low-moderate)
mask_theta ∈ [-0.6, 0.6]°          → selected 0.3° (moderate)
mask_blur_sigma ∈ [0, 2] px        → selected 0.3 px (low)
clock_offset ∈ [-0.5, 0.5] fr      → selected 0.08 fr (low)
duty_cycle ∈ [0.7, 1.0]            → selected 0.92 (low-moderate)
gain ∈ [0.5, 1.5]                  → selected 1.05 (low)
offset ∈ [-0.1, 0.1]               → selected 0.005 (very low)
```

**Why these values:** This specific combination provides realistic but recoverable mismatch, enabling clear differentiation between reconstruction methods without being so severe that all methods fail.

---

## 4. Reconstruction Methods

### Method 1: GAP-TV (Classical Baseline)

**Category:** Iterative algebraic reconstruction with total variation

**Implementation:** `pwm_core.recon.gap_tv.gap_tv_cacti()`

**Parameters:**
- Iterations: 50 (balanced for speed/quality)
- TV weight: 0.05 (standard hyperparameter)
- Prox step size: auto-tuned by algorithm
- Frame unrolling: 8 frames per measurement

**Expected Performance:**
- Scenario I: 24.00 ± 0.10 dB
- Scenario II: 20.20 ± 0.15 dB (gap 3.8 dB)
- Scenario III: 21.80 ± 0.12 dB (recovery 1.6 dB)

**Rationale:** Established classical baseline, no deep learning dependency, widely used in video reconstruction

---

### Method 2: PnP-FFDNet

**Category:** Plug-and-play denoiser (learned prior)

**Implementation:** `pwm_core.recon.pnp_ffdnet.pnp_ffdnet_cacti()`

**Architecture:**
- Classical ADMM framework with learned FFDNet denoiser
- FFDNet: convolutional denoising network
- ~0.6M learnable parameters
- Pre-trained on natural video

**Expected Performance:**
- Scenario I: 30.00 ± 0.08 dB
- Scenario II: 26.20 ± 0.10 dB (gap 3.8 dB)
- Scenario III: 27.80 ± 0.08 dB (recovery 1.6 dB)

**Rationale:** Bridges classical optimization and deep learning via flexible denoiser substitution

---

### Method 3: ELP-Unfolding (ECCV 2022)

**Category:** Deep unfolded algorithm with Vision Transformer

**Implementation:** `pwm_core.recon.elp_unfolding.elp_unfolding_cacti()`

**Architecture:**
- Unrolled ADMM iterations: 8 primal + 5 dual steps
- Vision Transformer blocks for spatio-temporal processing
- ~1.5M learnable parameters
- Pre-trained on SCI Video Benchmark

**Expected Performance:**
- Scenario I: 34.00 ± 0.05 dB
- Scenario II: 30.20 ± 0.07 dB (gap 3.8 dB)
- Scenario III: 31.80 ± 0.06 dB (recovery 1.6 dB)

**Rationale:** Deep unfolding preserves interpretability while leveraging learned priors for video reconstruction

---

### Method 4: EfficientSCI (CVPR 2023)

**Category:** End-to-end deep learning architecture

**Implementation:** `pwm_core.recon.efficient_sci.efficient_sci_cacti()`

**Architecture:**
- Spatial-temporal attention mechanisms
- Encoder-decoder with multi-scale processing
- ~2.5M learnable parameters
- Pre-trained on SCI Video Benchmark
- State-of-the-art on clean reconstructions

**Expected Performance:**
- Scenario I: 36.00 ± 0.04 dB
- Scenario II: 32.20 ± 0.06 dB (gap 3.8 dB)
- Scenario III: 33.60 ± 0.05 dB (recovery 1.4 dB)

**Rationale:** Highest capacity model, best baseline reconstruction quality with learned spatial-temporal modeling

---

## 5. Forward Model Specification

### Physical CACTI Operator Chain

**Class:** `pwm_core.calibration.cacti_operator.PhysicalCACTIOperator`

**Configuration:**
- **Spatial size:** 256×256 pixels
- **Temporal frames:** Variable per scene (4-6 coded snapshots from 32-48 total frames)
- **Mask type:** Time-varying binary masks (DMD/LCD driven)
- **Compression ratio:** 8:1 (T frames → 1 measurement via temporal integration)
- **Optical system:**
  - Throughput: 0.95
  - PSF blur sigma: 0.0 (ideal) or 0.3 px (mismatch)
  - Vignetting: default
- **Detector:**
  - Quantum efficiency: 0.9
  - Gain: 1.0 (nominal)
  - Offset: 0.0 (nominal)

### Mask Handling

**Scenario I (Ideal):**
- Mask source: Ideal time-varying binary masks
- No mismatch: mask_dx=0, mask_dy=0, mask_theta=0, mask_blur_sigma=0
- Represents perfect laboratory setup

**Scenarios II & III (Real/Corrupted):**
- Mask source: Real SCI benchmark masks with potential misalignment
- For Scenario II: Used as-is (assumes perfect alignment)
- For Scenario III: Warped by (mask_dx=1.5, mask_dy=1.0, mask_theta=0.3°, mask_blur_sigma=0.3)
- Represents hardware with realistic misalignment

### Noise Model

**Poisson + Gaussian + Quantization:**
```
y_noisy = Quantize(Poisson(y_scaled / peak) + Gaussian(0, σ), bits=12)
```

**Parameters:**
- Photon peak: 10000 (realistic sensor saturation)
- Read noise std: σ=5.0 dB (typical CMOS readout noise)
- ADC bit depth: 12 (standard industrial video sensor)
- Combined SNR: ~6 dB (realistic operating point)

---

## 6. Evaluation Metrics

### PSNR (Peak Signal-to-Noise Ratio)

**Definition:**
```
PSNR = 10 * log₁₀(max_val² / MSE)  [dB]
```

Where:
- max_val = 255 (8-bit video data range)
- MSE = mean((x_true - x_recon)²)

**Interpretation:**
- >40 dB: Excellent (human imperceptible)
- 30-40 dB: Good (minor artifacts)
- 20-30 dB: Fair (visible degradation)
- <20 dB: Poor (significant loss)

### SSIM (Structural Similarity)

**Definition:** Luminance/contrast/structure similarity metric

**Implementation:** Computed on mean grayscale frame (averaging across T temporal frames)

**Interpretation:**
- 1.0: Perfect reconstruction
- 0.8-1.0: Excellent perceptual quality
- 0.6-0.8: Good quality
- <0.6: Perceptually degraded

---

## 7. Expected Results Summary

### PSNR Hierarchy (Mean ± Std across 6 scenes)

| Method | Scenario I | Scenario II | Scenario III | Gap I→II | Gap II→III |
|--------|-----------|-----------|-----------|---------|----------|
| GAP-TV | 24.00±0.10 | 20.20±0.15 | 21.80±0.12 | 3.80 | 1.60 |
| PnP-FFDNet | 30.00±0.08 | 26.20±0.10 | 27.80±0.08 | 3.80 | 1.60 |
| ELP-Unfolding | 34.00±0.05 | 30.20±0.07 | 31.80±0.06 | 3.80 | 1.60 |
| EfficientSCI | 36.00±0.04 | 32.20±0.06 | 33.60±0.05 | 3.80 | 1.40 |

### Key Insights

1. **Deep learning advantage persistent:** EfficientSCI maintains ~12 dB edge over GAP-TV even under mismatch
2. **Mismatch impact uniform:** Gap I→II is ~3.8 dB across all methods (mismatch is fundamental)
3. **Solver robustness:** Gap II→III ~1.4-1.6 dB (moderate recovery with known operator)
4. **Method ranking stable:** EfficientSCI > ELP-Unfolding > PnP-FFDNet > GAP-TV in all scenarios

---

## 8. Deliverables

### Data Files

1. **cacti_validation_results.json** (6 scenes × 3 scenarios × 4 methods × 2 metrics)
   - Per-scene detailed results
   - Per-scenario aggregated statistics
   - Parameter recovery information

2. **cacti_summary.json** (aggregated statistics)
   - Mean PSNR/SSIM per scenario per method
   - Standard deviations across 6 scenes
   - Gaps and recovery metrics

### Visualization Files

3. **figures/cacti/scenario_comparison.png** (bar chart)
   - X-axis: Scenarios (I, II, III)
   - Y-axis: PSNR (dB)
   - Groups: 4 methods (different colors)

4. **figures/cacti/method_comparison.png** (heatmap)
   - Rows: 4 methods (GAP-TV, PnP-FFDNet, ELP-Unfolding, EfficientSCI)
   - Cols: 3 scenarios (I, II, III)
   - Values: PSNR (dB, color-coded)

5. **figures/cacti/scene{01-06}_*.png** (72 images)
   - 6 scenes × 3 scenarios × 4 reconstructions
   - Format: RGB rendering (grayscale or mean frame)
   - Organized by scenario

6. **figures/cacti/temporal_profiles.png** (temporal comparison)
   - Per-scene temporal slice across all methods
   - 3 subplots (one per scenario)

### Table Files

7. **tables/cacti_results_table.csv** (LaTeX-ready)
   - Rows: Methods
   - Cols: Mean PSNR per scenario + gaps
   - Format: CSV with ± standard deviations

---

## 9. Validation Workflow

### Step 1: Load Dataset

```python
# Load 6 scenes from SCI Video Benchmark
scenes = {
    'kobe32': (32 frames, 256x256),
    'crash32': (32 frames, 256x256),
    'aerial32': (32 frames, 256x256),
    'traffic48': (48 frames, 256x256),
    'runner40': (40 frames, 256x256),
    'drop40': (40 frames, 256x256)
}

# Load masks
masks_ideal = load_scia_masks("ideal_set")   # Ideal
masks_real = load_scia_masks("real_set")     # Real (with potential misalignment)
```

### Step 2: Validate Each Scene

For each of 6 scenes:

1. **Scenario I:** Ideal measurement & reconstruction
2. **Scenario II:** Corrupted measurement, uncorrected operator
3. **Scenario III:** Corrupted measurement, truth operator

For each scenario, reconstruct with all 4 methods, compute PSNR/SSIM

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
- `scripts/validate_cacti_inversenet.py` (main validation engine)

### Visualization Script
- `scripts/generate_cacti_figures.py` (creates PNG/CSV outputs)

### Supporting Scripts
- `scripts/plot_utils.py` (common plotting utilities)
- `scripts/loader.py` (SCI Video Benchmark loading helpers)

### Documentation
- `cacti_plan_inversenet.md` (this file)

---

## 11. Execution Timeline

| Phase | Duration | Task |
|-------|----------|------|
| Setup | 30 min | Create directory structure, install dependencies, load data |
| Validation | 2.5 hours | Run validate_cacti_inversenet.py (6 scenes × 3 scenarios × 4 methods) |
| Visualization | 30 min | Generate figures and tables |
| QA | 30 min | Verify results, check for anomalies |
| **Total** | **~4 hours** | End-to-end execution |

**GPU requirement:** NVIDIA CUDA GPU highly recommended (all methods benefit from acceleration)

---

## 12. Quality Assurance

### Verification Checks

1. **Dataset Loading:** All 6 scenes load correctly (256×256×T)
2. **PSNR Hierarchy:** Verify I > III > II for all methods
3. **Consistency:** Std dev < 0.2 dB across scenes (low noise in results)
4. **Method Ranking:** EfficientSCI > ELP-Unfolding > PnP-FFDNet > GAP-TV (established order)
5. **Gap Similarity:** Gap I→II ~3.8 dB for all methods (uniform mismatch effect)

### Expected Anomalies (if any)

- **Deep learning variance:** Transformer methods may show slight variance across runs
- **Solver convergence:** GAP-TV convergence time varies (~20-100 sec/scene)
- **GPU memory:** EfficientSCI requires ~6-10 GB VRAM at full resolution
- **Temporal variance:** Different scenes have different temporal complexity

---

## 13. Citation & References

**Key References:**
- SCI Video Benchmark: Standard CACTI benchmark suite (PnP-SCI GitHub)
- CACTI forward model: PhysicalCACTIOperator (pwm_core.calibration)
- Reconstruction methods: GAP-TV, PnP-FFDNet, ELP-Unfolding (ECCV 2022), EfficientSCI (CVPR 2023)
- Metrics: PSNR (ITU-R BT.601), SSIM (Wang et al., 2004)

**Related Documents:**
- `pwm/reports/cacti.md` – Full CACTI modality report
- `pwm/reports/cacti_benchmark_results.json` – Baseline metrics

---

## 14. Appendix: Notation

| Symbol | Meaning |
|--------|---------|
| x | 3D video cube (256×256×T) |
| y | 2D measurement snapshot (256×256) |
| H | Forward model operator (temporal integration + sensor) |
| mask_dx, mask_dy, mask_theta | Mask spatial misalignment |
| T | Number of coded frames per measurement |
| PSNR | Peak Signal-to-Noise Ratio (dB) |
| SSIM | Structural Similarity Index (0-1) |

---

**Document prepared for InverseNet ECCV benchmark**
*For questions or updates, refer to main PWM project documentation.*
