# InverseNet ECCV: SPC (Single-Pixel Camera) Validation Plan

**Document Version:** 2.0
**Date:** 2026-02-15
**Purpose:** Comprehensive validation of single-pixel camera (SPC) reconstruction methods under operator mismatch

---

## Executive Summary

This document details the validation framework for Single-Pixel Camera (SPC) reconstruction methods in the context of the InverseNet ECCV paper. The benchmark compares **3 reconstruction methods** across **3 scenarios** using **Set11 natural images** (11 standard images at 256×256 resolution), evaluating reconstruction quality under realistic operator mismatch without calibration.

**Key Features:**
- **3 Scenarios:** I (Ideal), II (Assumed/Baseline), III (Truth Forward Model)
- **Skip Scenario III:** Calibration algorithms not needed for Inversenet
- **3 Methods:** ISTA-Net+, HATNet, ADMM (classical)
- **11 Images:** Set11 benchmark (256×256 native resolution, no cropping needed)
- **Block-based CS:** 64×64 blocks (N=4096 pixels per block), 16 blocks per image
- **Compression Ratio:** 25% (M=1024 measurements per block out of N=4096)
- **Metrics:** PSNR (dB), SSIM (0-1)
- **Total Reconstructions:** 99 (11 images × 3 scenarios × 3 methods)

**Expected Outcome:** Quantify reconstruction quality hierarchy and solver robustness to DMD/sensor operator mismatch, enabling fair comparison across methods without calibration correction.

---

## 1. Problem Formulation

### 1.1 Forward Model

SPC forward model with block-based physical encoding:

```
y_b = A(x_b) + n    for each block b = 1, ..., 16
```

Where:
- **x** ∈ ℝ^{256×256}: 2D grayscale natural image (full Set11 resolution)
- **x_b** ∈ ℝ^{64×64}: Block b of the image (16 non-overlapping blocks per image)
- **A**: Physical SPC forward operator (DMD patterns + projection optics + sensor)
  - Spatial projection optics: PSF blur, vignetting, throughput
  - DMD pattern encoding: 1024 random ±1 binary masks with spatial warp (25% of N=4096)
  - Photodetector: Quantum efficiency, gain, dark current
  - ADC quantization: 14-bit precision
- **y_b** ∈ ℝ^{1024}: Measurement vector per block (scalar per pattern)
- **n**: Poisson shot + Gaussian read + quantization noise

**Block-based CS approach:** Each 256×256 image is divided into 16 non-overlapping 64×64 blocks (4×4 grid). Each block is measured and reconstructed independently, then stitched to form the full 256×256 reconstruction. This follows standard SPC practice (cf. run_all.py) and keeps the measurement matrix manageable (1024×4096 per block).

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

For Scenarios II & III, we inject mismatch into the measurement:

```
y_corrupt_b = A_mismatch(x_b) + n    for each block b = 1, ..., 16
```

Where A_mismatch applies true misalignment parameters, creating degradation that reconstructors must overcome. Mismatch is applied uniformly across all blocks (same DMD misalignment affects all blocks equally).

---

## 2. Scenario Definitions

### Scenario I: Ideal (Oracle)

**Purpose:** Theoretical upper bound for perfect measurements

**Configuration:**
- **Measurement:** y_ideal from ideal DMD patterns and sensor
- **Forward model:** Physical SPC operator with ideal parameters
- **Reconstruction:** Each method using perfect operator knowledge
- **Mismatch:** None (mask_dx=0, mask_dy=0, mask_theta=0, clock_offset=0, etc.)

**Expected PSNR (clean, no noise, 25% sampling):**
- ADMM: ~30.0 dB
- ISTA-Net+: ~34.0 dB
- HATNet: ~35.0 dB

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

**Expected PSNR (25% sampling):**
- All methods degrade ~2-4 dB compared to Scenario I
- Example: ADMM ~27 dB, ISTA-Net+ ~31 dB, HATNet ~32 dB

### Scenario III: Truth Forward Model (Oracle Operator)

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

**Expected PSNR (25% sampling):**
- Partial recovery from Scenario II (better than baseline but worse than ideal)
- Gap II→III: ~1-2 dB (method-dependent robustness)
- Example: ADMM ~29 dB, ISTA-Net+ ~32.5 dB, HATNet ~33.5 dB

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
- Block size: 64×64 (N=4096), M=1024 (25% compression)
- Basis pursuit formulation: min ||x||_1 s.t. ||Ax - y||_2 ≤ ε

**Expected Performance (25% sampling, 256×256 images):**
- Scenario I: 30.0 ± 1.5 dB
- Scenario II: 27.0 ± 1.5 dB (gap 3.0 dB)
- Scenario III: 29.0 ± 1.5 dB (recovery 2.0 dB)

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
- Block-based: processes 64×64 blocks independently

**Expected Performance (25% sampling, 256×256 images):**
- Scenario I: 34.0 ± 1.0 dB
- Scenario II: 31.0 ± 1.2 dB (gap 3.0 dB)
- Scenario III: 32.5 ± 1.0 dB (recovery 1.5 dB)

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
- Block-based: processes 64×64 blocks independently

**Expected Performance (25% sampling, 256×256 images):**
- Scenario I: 35.0 ± 0.8 dB
- Scenario II: 32.0 ± 1.0 dB (gap 3.0 dB)
- Scenario III: 33.5 ± 0.9 dB (recovery 1.5 dB)

**Rationale:** Transformer-based architecture captures long-range dependencies in sparse reconstruction

---

## 5. Forward Model Specification

### Physical SPC Operator Chain

**Class:** `pwm_core.calibration.spc_operator.PhysicalSPCOperator`

**Configuration:**
- **Full image size:** 256×256 (from Set11, native resolution, no cropping)
- **Block size:** 64×64 (N=4096 pixels per block)
- **Blocks per image:** 16 (4×4 grid of non-overlapping blocks)
- **Measurement size per block:** 1024 (M=1024 random projections, 25% compression)
- **Total measurements per image:** 16,384 (16 blocks × 1024)
- **DMD pattern type:** Random ±1 binary masks (Gaussian, row-normalized)
- **Measurement matrix:** Φ ∈ ℝ^{1024×4096} per block (row-normalized Gaussian)
- **Projection optics:**
  - Throughput: 0.95
  - PSF blur sigma: 0.8 pixels
  - Vignetting: 0.02
- **Photodetector:**
  - Quantum efficiency: 0.85
  - Gain: 1.0 (nominal)
  - Dark current: 0.01 (normalized)

### Block-Based Measurement

**Block partitioning:**
```
256×256 image → 4×4 grid of 64×64 blocks

Block (0,0): image[  0: 64,   0: 64]
Block (0,1): image[  0: 64,  64:128]
Block (0,2): image[  0: 64, 128:192]
Block (0,3): image[  0: 64, 192:256]
Block (1,0): image[ 64:128,  0: 64]
...
Block (3,3): image[192:256, 192:256]
```

**Measurement matrix creation (following run_all.py):**
```python
np.random.seed(42)
Phi = np.random.randn(1024, 4096).astype(np.float32)
row_norms = np.linalg.norm(Phi, axis=1, keepdims=True)
Phi_norm = Phi / np.maximum(row_norms, 1e-8)
```

Same Φ matrix is used for all blocks (shared DMD pattern set).

### Mask Handling

**Scenario I (Ideal):**
- Mask source: Ideal random ±1 binary patterns
- No mismatch: mask_dx=0, mask_dy=0, mask_theta=0
- Represents perfect laboratory setup

**Scenarios II & III (Real/Corrupted):**
- Mask source: Real DMD pattern set with potential misalignment
- For Scenario II: Used as-is (assumes perfect alignment)
- For Scenario III: Warped by (mask_dx=0.4, mask_dy=0.4, mask_theta=0.08°)
- Represents hardware with realistic misalignment
- Mismatch applied uniformly across all 16 blocks (same DMD shift affects entire field)

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

**Implementation:** Computed on full 256×256 reconstructed images (stitched from 64×64 blocks)

**Interpretation:**
- 1.0: Perfect reconstruction
- 0.8-1.0: Excellent perceptual quality
- 0.6-0.8: Good quality
- <0.6: Perceptually degraded

---

## 7. Expected Results Summary

### PSNR Hierarchy (Mean ± Std across 11 images, 25% sampling, 256×256)

| Method | Scenario I | Scenario II | Scenario III | Gap I→II | Gap II→III |
|--------|-----------|-----------|-----------|---------|----------|
| ADMM | 30.0±1.5 | 27.0±1.5 | 29.0±1.5 | 3.0 | 2.0 |
| ISTA-Net+ | 34.0±1.0 | 31.0±1.2 | 32.5±1.0 | 3.0 | 1.5 |
| HATNet | 35.0±0.8 | 32.0±1.0 | 33.5±0.9 | 3.0 | 1.5 |

### Key Insights

1. **Deep learning advantage persistent:** HATNet maintains ~5 dB edge over ADMM even under mismatch
2. **Mismatch impact uniform:** Gap I→II is ~3.0 dB across all methods (mismatch is fundamental)
3. **Solver robustness:** Gap II→III ~1.5-2.0 dB (moderate recovery with known operator)
4. **Method ranking stable:** HATNet > ISTA-Net+ > ADMM in all scenarios
5. **25% sampling advantage:** Higher compression ratio (vs 15%) gives ~2 dB improvement across all methods

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
   - X-axis: Scenarios (I, II, III)
   - Y-axis: PSNR (dB)
   - Groups: 3 methods (different colors)

4. **figures/spc/method_comparison.png** (heatmap)
   - Rows: 3 methods (ADMM, ISTA-Net+, HATNet)
   - Cols: 3 scenarios (I, II, III)
   - Values: PSNR (dB, color-coded)

5. **figures/spc/image{01-11}_*.png** (99 images)
   - 11 images × 3 scenarios × 3 reconstructions
   - Format: Grayscale 256×256 (stitched from 16 blocks of 64×64)
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
# Load 11 Set11 images at native 256×256 resolution
images = [load_set11_image(i) for i in range(1, 12)]  # (256,256) each
# No cropping — use full 256×256 images

# Create measurement matrix (shared across all blocks)
np.random.seed(42)
block_size = 64
n_pix = block_size * block_size  # 4096
sampling_rate = 0.25
m = int(n_pix * sampling_rate)   # 1024
Phi = np.random.randn(m, n_pix).astype(np.float32)
row_norms = np.linalg.norm(Phi, axis=1, keepdims=True)
Phi_norm = Phi / np.maximum(row_norms, 1e-8)  # Row-normalized (following run_all.py)
```

### Step 2: Block Partition and Measure

```python
# For each 256×256 image, partition into 16 non-overlapping 64×64 blocks
for img in images:
    blocks = []
    for bi in range(4):          # 4 rows of blocks
        for bj in range(4):      # 4 columns of blocks
            block = img[bi*64:(bi+1)*64, bj*64:(bj+1)*64]  # (64,64)
            blocks.append(block)  # 16 blocks total

    # Measure each block: y_b = Phi_norm @ x_b.flatten() + noise
    for block in blocks:
        x_flat = block.flatten()  # (4096,)
        y = Phi_norm @ x_flat     # (1024,)
        y += noise                # Add noise model
```

### Step 3: Validate Each Image

For each of 11 images (all 16 blocks per image):

1. **Scenario I:** Ideal measurement & reconstruction (per block)
2. **Scenario II:** Corrupted measurement, uncorrected operator (per block)
3. **Scenario III:** Corrupted measurement, truth operator (per block)

For each scenario, reconstruct all 16 blocks with all 3 methods, stitch into 256×256, compute PSNR/SSIM on full image

### Step 4: Aggregate Results

Compute per-method and per-scenario statistics:
- Mean PSNR/SSIM
- Standard deviations
- Confidence intervals (if needed)

### Step 5: Generate Visualizations

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
| Validation | 3 hours | Run validate_spc_inversenet.py (11 images × 16 blocks × 3 scenarios × 3 methods) |
| Visualization | 20 min | Generate figures and tables |
| QA | 20 min | Verify results, check for anomalies |
| **Total** | **~4 hours** | End-to-end execution |

**Note:** 256×256 images with 16 blocks each increases computation ~16× over single-block. Total: 11 × 16 × 3 × 3 = 1584 block reconstructions.

**GPU requirement:** NVIDIA CUDA GPU recommended for HATNet/ISTA-Net+ (optional for ADMM)

---

## 12. Quality Assurance

### Verification Checks

1. **Dataset Loading:** All 11 images load correctly at 256×256
2. **Block Partitioning:** 16 non-overlapping 64×64 blocks per image, no border artifacts
3. **PSNR Hierarchy:** Verify I > III > II for all methods
4. **Consistency:** Std dev < 2.0 dB across images (reasonable variance for natural images)
5. **Method Ranking:** HATNet > ISTA-Net+ > ADMM (established order)
6. **Gap Similarity:** Gap I→II ~3.0 dB for all methods (uniform mismatch effect)
7. **Block Stitching:** Reconstructed 256×256 matches ground truth dimensions exactly

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
| x | 2D grayscale image (256×256) |
| x_b | Block b of image (64×64), b = 1..16 |
| y_b | Measurement vector per block (1024 scalars) |
| Φ | Measurement matrix (1024×4096, row-normalized Gaussian) |
| A | Forward model operator (projection + sensor) |
| mask_dx, mask_dy, mask_theta | DMD affine misalignment |
| M | Measurements per block (1024 at 25% compression) |
| N | Pixels per block (4096 = 64×64) |
| PSNR | Peak Signal-to-Noise Ratio (dB), computed on full 256×256 |
| SSIM | Structural Similarity Index (0-1), computed on full 256×256 |

---

**Document prepared for InverseNet ECCV benchmark**
*For questions or updates, refer to main PWM project documentation.*
