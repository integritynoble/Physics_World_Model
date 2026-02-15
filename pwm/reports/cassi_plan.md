# CASSI Calibration via Enlarged Simulation Grid & Mask Correction
**Plan Document — v4+ (2026-02-15, with PWM Pipeline Flowcharts)**

## Executive Summary

This document proposes a **complete CASSI calibration strategy** based on enlarged simulation grid with realistic mask handling:

1. **(A) Scene Preprocessing:** Crop edges (P=16 px/side) to ensure information stays within nominal measurement region
   - Original: 256×256×28 → Crop interior (224×224) → Zero-pad to (256×256×28)
   - Purpose: Prevent signal leakage due to mismatch

2. **(B) Enlarged Grid Forward Model:** High-fidelity simulation with N=4 spatial, K=2 spectral
   - Spatial enlargement: 256×256 → 1024×1024 (factor N=4)
   - Spectral expansion: 28 → 217 bands, L_expanded = (L-1)×N×K+1 = 27×4×2+1
   - **Dispersion shift in simulation:** stride-1 (1 pixel per frame, fine granularity)
   - Measurement size after summation: 1024×1240 (width = 1024 + 2×108)
   - Downsample to original: 1024×1240 → 256×310 (factor 4)

3. **(C) Mask Handling - Different Sources for Each Scenario:**

   **Scenario I (Ideal):** Simulation mask (TSA synthetic data)
   - Load: `/home/spiritai/MST-main/datasets/TSA_simu_data/mask.mat` (256×256)
   - Purpose: Perfect forward model baseline

   **Scenarios II & III (Real simulation):** Real experimental mask (TSA real data)
   - Load: `/home/spiritai/MST-main/datasets/TSA_real_data/mask.mat` (256×256)
   - Purpose: Realistic coded aperture pattern matching real hardware
   - Upsample to 1024×1024 for enlarged simulation
   - For each of 217 frames, create shifted version (dispersion encoding)
   - Mismatch injected to BOTH mask AND scene equally
   - Downsample back to 256×256 for reconstruction

4. **(D) Three Reconstruction Scenarios:**
   - **Scenario I (Ideal):** Ideal measurement + ideal mask + ideal forward model → x̂_ideal (oracle)
   - **Scenario II (Assumed):** **Corrupted measurement** + assumed perfect mask + **simulated forward model** → x̂_assumed (baseline, no correction)
   - **Scenario III (Corrected):** **Corrupted measurement** + corrected mask + **simulated forward model** → x̂_corrected (practical, with correction)

5. **(E) Comprehensive Mismatch Correction:** Correct ALL mismatch factors via UPWMI Algorithms 1 & 2

   **Mismatch Parameters (6 factors from cassi.md W1-W5):**
   - **Group 1 - Mask Affine:** (dx, dy, θ) combined into one warp operation
     - dx, dy ∈ [-3, 3] px (mechanical assembly tolerance)
     - θ ∈ [-1°, 1°] (optical bench rotation)
   - **Group 2 - Dispersion:** (a₁, α) encoding properties
     - a₁ ∈ [1.95, 2.05] px/band (prism slope, thermal drift)
     - α ∈ [-1°, 1°] (dispersion axis offset, prism settling)
   - **Group 3 - PSF:** σ_blur (optional, low impact <0.1 dB)
     - σ_blur ∈ [0.5, 2.0] px (lens/alignment blur)

6. **(F) Validate on 10 Scenes** with comprehensive parameter recovery and three-scenario comparison

**Core strategy:** Enlarged grid simulation (N=4, K=2) for accurate forward model, then correct misalignment via Algorithms 1&2, comparing ideal/assumed/corrected reconstructions.

---

## PWM Pipeline Flowcharts (Mandatory)

### Scenario II: Assumed Mask Reconstruction (Baseline, No Correction)

**Pipeline chain:** Measurement generation with mismatch, reconstruction without correction

```
x (world: 256×256×28)
  ↓
 SourceNode: photon_source — illumination (strength=1.0)
  ↓
Element 1 (subrole=encoding): mask_uncorrected — real coded aperture with true mismatch
                              (dx_true, dy_true, θ_true NOT corrected, just applied)
  ↓
Element 2 (subrole=encoding): parametric_dispersion — Δu(l) = a1·l + a2·l², axis angle α
                              (nominal: a1=2.0, a2=0.0, α=0°)
  ↓
Element 3 (subrole=encoding): spectral_integration — sums along 217-band enlarged grid (L=217→1)
                              Measurement: 1024×1240 (enlarged space, stride-1)
  ↓
Element 4 (subrole=transport): downsample_spatial — 4× downsampling (1024×1240 → 256×310)
  ↓
Element 5 (subrole=transport): psf_blur — Gaussian PSF convolution (σ=0, ideal)
  ↓
SensorNode: detector — QE=0.9, gain=1.0, photon peak=10000
  ↓
NoiseNode: poisson_read_quantization — Poisson shot (peak=10000) + Gaussian read (σ=1.0) + Quant(12bit)
  ↓
y_corrupt (256×310, with mismatch + noise)
  ↓
[RECONSTRUCTION WITHOUT CORRECTION]
  ↓
Operator: phi_assumed = SimulatedOperator_EnlargedGrid(mask_real_uncorrected, N=4, K=2)
  ↓
Solver: GAP-TV or other (n_iter=50) on y_corrupt with phi_assumed
  ↓
x̂_assumed (256×256×28, degraded reconstruction)
```

**Key mismatch factors (injected but NOT corrected):**
| Parameter | True Value | Impact | Notes |
|-----------|-----------|--------|-------|
| mask_dx | ∈ [-3, 3] px | 0.12 dB | Mask x-shift from mechanical tolerance |
| mask_dy | ∈ [-3, 3] px | 0.12 dB | Mask y-shift |
| mask_theta | ∈ [-1°, 1°] | 3.77 dB | Mask rotation from optical bench twist |
| disp_a1 | nominal=2.0 | — | (not perturbed in base scenario) |
| disp_alpha | nominal=0° | — | (not perturbed in base scenario) |

**Expected result:** PSNR_assumed ≈ 18–21 dB (shows degradation from corruption + no correction)

---

### Scenario III: Corrected Mask Reconstruction (Practical with UPWMI Correction)

**Pipeline chain:** Measurement generation with mismatch, operator correction, reconstruction with corrected parameters

**Phase 1: Measurement Formation with Injected Mismatch**

```
x (world: 256×256×28)
  ↓
SourceNode: photon_source — illumination (strength=1.0)
  ↓
Element 0 (subrole=encoding): scene_affine_warp — apply mismatch to scene
                              warp(x, dx=dx_true, dy=dy_true, θ=θ_true)
  ↓
Element 1 (subrole=encoding): mask_affine_warp — apply mismatch to real mask
                              warp(mask_real, dx=dx_true, dy=dy_true, θ=θ_true)
  ↓
Element 2 (subrole=encoding): spatial_enlarge — 4× upsampling (256 → 1024)
                              x_expanded: 1024×1024×28 → 1024×1024×217 (spectral interp)
                              mask_enlarged: 256×256 → 1024×1024
  ↓
Element 3 (subrole=encoding): parametric_dispersion — Δu(l) = a1·l + a2·l², axis angle α
                              (nominal: a1=2.0, a2=0.0, α=0°, stride=1 in enlarged space)
  ↓
Element 4 (subrole=encoding): spectral_integration — sums along 217 bands
                              Measurement: 1024×1240 (stride-1, full dispersion range ±108)
  ↓
Element 5 (subrole=transport): downsample_spatial — 4× downsampling (1024×1240 → 256×310)
  ↓
Element 6 (subrole=transport): psf_blur — Gaussian PSF convolution (σ=0, ideal)
  ↓
SensorNode: detector — QE=0.9, gain=1.0, photon peak=10000
  ↓
NoiseNode: poisson_read_quantization — Poisson shot (peak=10000) + Gaussian read (σ=1.0) + Quant(12bit)
  ↓
y_noisy (256×310, with mismatch + noise)
```

**Phase 2: Operator Correction via UPWMI Algorithms**

```
[Coarse estimation — Algorithm 1: Hierarchical Beam Search]
  ↓
Input: y_noisy, mask_real (uncorrected base), x_true (for validation)
  ↓
1D sweeps + 3D beam search (5×5×5) on mask affine space (dx, dy, theta)
2D beam search (5×7) on dispersion space (a1, alpha)
Coordinate descent refinement (3 rounds)
  ↓
Output: (dx̂₁, dŷ₁, θ̂₁, â₁₁, α̂₁)
Duration: ~4.5 hours per scene
Accuracy: ±0.1–0.2 px (mask), ±0.01 px/band (dispersion)
  ↓
[Fine estimation — Algorithm 2: Joint Gradient Refinement]
  ↓
Input: y_noisy, x_true, mask_real, coarse_estimate=(dx̂₁, dŷ₁, θ̂₁, â₁₁, α̂₁)
  ↓
Unrolled GAP-TV differentiable solver (K=10)
Phase 1: 100 epochs on full measurement (lr=0.01) → ~1.5 hours
Phase 2: 50 epochs on 10-scene ensemble (lr=0.001) → ~1 hour
  ↓
Output: (dx̂₂, dŷ₂, θ̂₂, â₁₂, α̂₂)
Duration: ~2.5 hours per scene
Accuracy: ±0.05–0.1 px (mask), ±0.001 px/band (dispersion)
Improvement: 3–5× better than Algorithm 1
  ↓
[Operator Correction Step]
  ↓
Build corrected mask: mask_corrected = warp(mask_real, dx=dx̂₂, dy=dŷ₂, θ=θ̂₂)
Update dispersion parameters: a1_corrected=â₁₂, alpha_corrected=α̂₂
Build corrected operator: phi_corrected = SimulatedOperator_EnlargedGrid(
    mask_corrected, N=4, K=2,
    a1_override=â₁₂, alpha_override=α̂₂
)
  ↓
[Reconstruction with Corrected Operator]
  ↓
Solver: GAP-TV on y_noisy with phi_corrected (n_iter=50)
  ↓
x̂_corrected (256×256×28, improved reconstruction)
```

**Element inventory for Scenario III:**

| # | node_id | primitive_id | subrole | parameters | mismatch/correction | bounds | prior |
|---|---------|-------------|---------|-----------|-------------------|--------|-------|
| 1 | source | photon_source | source | strength=1.0 | — | — | — |
| 2 | scene_warp | spatial_affine_warp | preprocessing | (H,W,L)=(256,256,28) | scene dx,dy,θ | [-3,3]px, [-3,3]px, [-1°,1°] | uniform |
| 3 | mask | mask_affine_warp | encoding | mask (256,256), float [0.007,1.0] | mask dx,dy,θ → **corrected** | [-3,3]px, [-3,3]px, [-1°,1°] | uniform |
| 4 | enlarge_spatial | spatial_upsample | preprocessing | factor N=4 | — | — | — |
| 5 | enlarge_spectral | spectral_interpolate | preprocessing | L: 28 → 217 bands | — | — | — |
| 6 | disperse | parametric_dispersion | encoding | a1=2.0, a2=0.0, α=0° | a1, alpha → **corrected** | [1.95,2.05], [-1°,1°] | normal |
| 7 | integrate | spectral_integration | encoding | axis=-1, L=217 | — | — | — |
| 8 | downsample | spatial_downsample | transport | factor N=4 (1024→256) | — | — | — |
| 9 | psf | psf_blur | transport | sigma=0 | σ (optional) | [0, 3] px | half-normal |
| 10 | sensor | detector | sensor | QE=0.9, gain=1.0 | — | — | — |
| 11 | noise | poisson_read_quantization | noise | peak=10000, σ_read=1.0 | — | — | — |

**Mismatch correction performance (Algorithm 2 estimates):**

| Parameter | Ground Truth | Estimated | Error | Impact (from cassi.md) |
|-----------|-------------|-----------|-------|----------------------|
| mask_dx | ∈ [-3, 3] px | ±0.05–0.1 px | ~0.1 px | 0.12 dB |
| mask_dy | ∈ [-3, 3] px | ±0.05–0.1 px | ~0.1 px | 0.12 dB |
| mask_theta | ∈ [-1°, 1°] | ±0.02–0.05° | ~0.05° | 3.77 dB |
| disp_a1 | nominal=2.0 | ±0.001 px/band | ~0.001 | 5.49 dB |
| disp_alpha | nominal=0° | ±0.02–0.05° | ~0.05° | 7.04 dB |

**Expected result:** PSNR_corrected ≈ 23–25 dB (practical reconstruction with full correction)

---

## Part 1: Scene Preprocessing

### 1.1 Scene Cropping for Mismatch Robustness

**Problem:** With mismatch (dx, dy, θ), the signal may spread beyond the nominal 256×310 measurement region.

**Solution:** Crop scene edges before enlargement to ensure information stays bounded.

**Procedure:**
```
1. Load original scene: x (256×256×28)

2. Crop interior by P=16 pixels per side:
   x_cropped = x[16:240, 16:240, :] (224×224×28)

3. Zero-pad back to original size:
   x_padded = np.pad(x_cropped, pad_width=16, mode='constant', constant_values=0)
   Result: x_padded (256×256×28) with dark borders

4. Purpose: Effective content area (224×224) is centered, safe from edge leakage
```

---

## Part 2: Enlarged Grid Forward Model with Real Mask

### 2.1 Spectral Expansion Formula

**Original:** L=28 bands with dispersion shift step = 2 pixels/band

**Enlarged:** L_expanded = (L-1) × N × K + 1 = 27 × 4 × 2 + 1 = 217 bands

**Where:**
- N = 4: spatial enlargement factor (256→1024)
- K = 2: original dispersion shift step (pixels/band in original space)

**Interpretation:** In enlarged space, shift step becomes 1 pixel/band (fine granularity):
- Original: shift by 2 px between adjacent bands
- Enlarged (4×): shift by 2 px on 1024-pixel grid = equivalent to 0.5 px on 256-pixel grid
- With 217 bands: achieves finer dispersion sampling (stride-1 in enlarged space)

### 2.2 Scene Enlargement

**Input:** Cropped scene x (256×256×28)

**Process:**
```python
# Step 1: Spatial upsample by factor N=4
x_spatial = upsample_spatial(x, factor=4)  # 1024×1024×28

# Step 2: Spectral interpolation to 217 bands
# Using cubic spline (PCHIP) to preserve monotonicity
def interpolate_spectral_217(x, original_L=28):
    L_expanded = (original_L - 1) * 4 * 2 + 1  # 217
    lambda_orig = np.arange(original_L) / (original_L - 1)
    lambda_new = np.arange(L_expanded) / (L_expanded - 1)

    x_interp = np.zeros((x.shape[0], x.shape[1], L_expanded))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            f = scipy.interpolate.PchipInterpolator(lambda_orig, x[i, j, :])
            x_interp[i, j, :] = f(lambda_new)
    return x_interp

x_expanded = interpolate_spectral_217(x_spatial)  # 1024×1024×217
```

### 2.3 Real Mask Loading and Upsampling

**Input:** Real mask from TSA dataset

**Process:**
```python
import scipy.io

# Load real mask from TSA data
mask_data = scipy.io.loadmat('/home/spiritai/MST-main/datasets/TSA_simu_data/mask.mat')
mask_256 = mask_data['mask']  # Load actual key (may be 'mask' or other)

# Verify shape
assert mask_256.shape == (256, 256), f"Expected (256,256), got {mask_256.shape}"

# Upsample to 1024×1024 for enlarged simulation
mask_1024 = upsample_spatial(mask_256, factor=4)  # 1024×1024

# Normalize to [0, 1] range if needed
mask_1024 = np.clip(mask_1024, 0, 1)
```

### 2.4 Measurement Formation with 217-Frame Dispersion

**Key insight:** Each of 217 spectral frames gets shifted by dispersion, creating elongated measurement.

**Dispersion shifts:**
```
For frame k (k = 0 to 216):
    Center frame: n_c = (217 - 1) / 2 = 108
    Shift amount: d_k = k - n_c = k - 108 pixels
    Range: d_min = -108, d_max = +108
    Measurement width: 1024 + 2×108 = 1024 + 216 = 1240
```

**Measurement accumulation:**
```python
def forward_model_enlarged(x_expanded, mask_1024, stride=1):
    """
    Forward model with enlarged grid (N=4, K=2).

    Input:
        x_expanded: 1024×1024×217 scene
        mask_1024: 1024×1024 real mask

    Output:
        y_meas: 1024×1240 measurement (summed over 217 frames)
    """
    H, W, L = x_expanded.shape  # 1024, 1024, 217
    n_c = (L - 1) / 2.0  # 108

    # Measurement width accounts for maximum dispersion shift
    W_meas = W + (L - 1) * stride  # 1024 + 216 = 1240
    y_meas = np.zeros((H, W_meas))

    # Accumulate contributions from all 217 frames
    for k in range(L):
        # Dispersion shift for frame k
        d_k = int(round(stride * (k - n_c)))  # stride=1: d_k = k - 108

        # Code frame k
        scene_k = x_expanded[:, :, k]  # 1024×1024
        coded_k = scene_k * mask_1024   # element-wise multiplication

        # Place shifted frame in measurement
        y_start = max(0, d_k)
        y_end = min(W_meas, W + d_k)
        src_start = max(0, -d_k)
        src_end = src_start + (y_end - y_start)

        y_meas[:, y_start:y_end] += coded_k[:, src_start:src_end]

    return y_meas  # 1024×1240
```

### 2.5 Downsample to Original Measurement Size

**Input:** y_meas (1024×1240)

**Output:** y_final (256×310)

**Process:**
```python
# Downsample by factor 4 (reverse of enlargement)
y_final = downsample_spatial(y_meas, factor=4)  # 1024×1240 → 256×310
```

**Why 256×310:**
- Height: 1024 / 4 = 256
- Width: 1240 / 4 = 310

---

## Part 3: Three Reconstruction Scenarios

### 3.1 Scenario I: Ideal Reconstruction (Oracle)

**Purpose:** Upper bound - best possible reconstruction with perfect forward model and perfect alignment.

**Mask source:** TSA simulation mask (ideal, synthetic)
- Load: `/home/spiritai/MST-main/datasets/TSA_simu_data/mask.mat`

**Forward model:** Ideal, direct (no enlargement, stride=2 standard dispersion)
```python
def forward_ideal(x_256, mask_256):
    """
    Ideal forward model (256×256×28 → 256×310).
    - No enlargement, no interpolation artifacts
    - Direct stride=2 dispersion (standard SD-CASSI)
    - Perfect mask, no mismatch
    """
    H, W, L = x_256.shape
    W_meas = W + (L - 1) * 2  # 256 + 54 = 310 (stride=2)
    y = np.zeros((H, W_meas))
    n_c = (L - 1) / 2.0

    for l in range(L):
        d_l = int(round(2 * (l - n_c)))  # stride=2 for original
        coded = x_256[:, :, l] * mask_256

        y_start = max(0, d_l)
        y_end = min(W_meas, W + d_l)
        src_start = max(0, -d_l)
        src_end = src_start + (y_end - y_start)

        y[:, y_start:y_end] += coded[:, src_start:src_end]

    return y
```

**Measurement generation:**
```
1. Load ideal mask: mask_ideal from TSA_simu_data (256×256, perfect, no mismatch)
2. Generate ideal measurement: y_ideal = forward_ideal(x_256, mask_ideal)
   Size: 256×310, no noise, no corruption
```

**Reconstruction:**
```
Operator: phi_ideal = forward_ideal  (direct model, stride=2)
x̂_ideal = solver(y_ideal, phi_ideal, mask_ideal, n_iter=50)
```

**Metrics:** PSNR_ideal, SSIM_ideal, SAM_ideal (oracle baseline - best possible)

### 3.2 Scenario II: Assumed Mask Reconstruction (Baseline, No Correction)

**Purpose:** Show impact of measurement corruption + simulation method WITHOUT mismatch correction (baseline degradation).

**Mask source:** TSA real experimental mask (same source as Scenario III, NO correction applied)
- Load: `/home/spiritai/MST-main/datasets/TSA_real_data/mask.mat`
- This is the realistic experimental mask from actual hardware

**Forward model:** Simulated with N=4, K=2 (enlarged grid simulation)

**Measurement generation:**
```
1. Use CORRUPTED measurement from Scenario III: y_corrupt (256×310)
   (Same as Scenario III: with mismatch + noise injected from real hardware mask)

2. Use ASSUMED perfect/ideal mask (NO mismatch correction applied):
   mask_assumed = mask_real_data (256×256, unchanged - no correction!)
```

**Reconstruction WITHOUT correction:**
```
Operator: phi_assumed = SimulatedOperator_EnlargedGrid(mask_assumed, N=4, K=2)
x̂_assumed = solver(y_corrupt, phi_assumed, n_iter=50)
```

**Why use corrupted y_corrupt:** To measure the degradation from measurement corruption WITHOUT the benefit of mismatch correction. This shows what happens if we ignore the hardware misalignment.

**Metrics:** PSNR_assumed, SSIM_assumed, SAM_assumed
- Shows worst-case reconstruction when mismatch is NOT corrected
- Demonstrates necessity of Algorithms 1 & 2 for real hardware

### 3.3 Scenario III: Corrected Mask Reconstruction (Practical with UPWMI Correction)

**Purpose:** Practical real-world scenario with operator mismatch correction via UPWMI algorithms (from cassi_working_process.md Section 13).

**Mask source:** TSA real experimental mask (actual hardware mask)
- Load: `/home/spiritai/MST-main/datasets/TSA_real_data/mask.mat`
- Upsample to 1024×1024 for enlarged simulation

**Forward model:** Simulated with N=4, K=2 (enlarged grid, stride-1 dispersion)

**Measurement generation with synthetic mismatch (assembly tolerance):**
```
1. Inject synthetic mismatch to BOTH scene and mask (realistic assembly errors):
   dx_true ∈ [-3, 3] px (mask x-shift from mechanical tolerance)
   dy_true ∈ [-3, 3] px (mask y-shift)
   θ_true ∈ [-1°, 1°] (mask rotation from optical bench twist)

   x_misaligned = warp_affine(x_256, dx=dx_true, dy=dy_true, theta=θ_true)
   mask_misaligned_256 = warp_affine(mask_real_data, dx=dx_true, dy=dy_true, theta=θ_true)

2. Generate enlarged version with mismatch:
   x_expanded_mis = enlarge(x_misaligned)  # 1024×1024×217
   mask_1024_mis = upsample(mask_misaligned_256, 4)  # 1024×1024

3. Forward model with mismatch (stride-1 in enlarged space):
   y_enlarged = forward_model_enlarged(x_expanded_mis, mask_1024_mis)  # 1024×1240

4. Downsample to measurement size:
   y_corrupt = downsample(y_enlarged, 4)  # 256×310

5. Add realistic noise (Poisson + Gaussian):
   y_noisy = add_noise(y_corrupt, peak=10000, sigma=0.01)
```

**Operator Correction via UPWMI (cassi_working_process.md Section 13):**

Per the CASSI working process, when operator mismatch is detected, the system applies UPWMI operator correction framework:

```
BeliefState(θ):
  θ_nominal = { a1: 2.0 px/band, a2: 0.0, alpha: 0.0° }  # Fixed optical properties
  θ_mismatch = { dx: ?, dy: ?, theta: ? }                 # To be estimated

Mode: operator_correction  # User requests calibration
Operator: Φ(θ) = forward_model_enlarged(mask_corrected, N=4, K=2)
```

**Algorithm 1: Coarse Parameter Estimation (Beam Search)**
```
# Coarse beam search on full measurement
(dx_hat1, dy_hat1, theta_hat1, a1_hat1, alpha_hat1) = upwmi_algorithm_1(
    y_noisy,              # Full measurement (corrupted with mismatch+noise)
    mask_real_data,       # Real mask (uncorrected base)
    x_true_256,           # Ground truth scene
    search_space={
        'dx': np.linspace(-3, 3, 13),        # 13 values
        'dy': np.linspace(-3, 3, 13),
        'theta': np.linspace(-1°, 1°, 7)     # 7 values
    }
    # Stage 1: 1D sweeps with proxy K=5 (~30 min per param)
    # Stage 2: Beam search 5×5×5=125 combos with K=10 (~2 hours)
    # Stage 3: Coordinate descent refinement (~1 hour)
)
# Total: ~4.5 hours per scene
# Accuracy: ±0.1-0.2 px
```

**Algorithm 2: Refined Parameter Estimation (Gradient-Based)**
```python
# Gradient-based refinement using unrolled differentiable solver
(dx_hat2, dy_hat2, theta_hat2, a1_hat2, alpha_hat2) = upwmi_algorithm_2_joint_gradient_refinement(
    y_noisy,                 # Full corrupted measurement (256×310)
    x_true_256,              # Ground truth scene
    mask_real_data,          # Real experimental mask
    coarse_estimate=(dx_hat1, dy_hat1, theta_hat1, a1_hat1, alpha_hat1),  # From Alg1
    n_iter_unroll=10
)
# Phase 1: 100 epochs on full measurement, lr=0.01 (~1.5 hours)
# Phase 2: 50 epochs on 10-scene ensemble, lr=0.001 (~1 hour)
# Total: ~2.5 hours per scene (faster than Alg1)
# Accuracy: ±0.05-0.1 px (3-5× improvement over Alg1)
```

**Reconstruction with Corrected Operator:**
```
# Build corrected forward operator
BeliefState(θ_corrected):
  θ_corrected.mismatch = {dx: dx_hat2, dy: dy_hat2, theta: theta_hat2}
  Φ_corrected = forward_model_enlarged(mask_corrected, N=4, K=2)

# Reconstruct on original grid using corrected operator
mask_corrected = warp_affine(mask_real_data, dx=dx_hat2, dy=dy_hat2, theta=theta_hat2)
Operator: phi_corrected = SimulatedOperator_EnlargedGrid(mask_corrected, N=4, K=2)

# Use one of standard solvers (GAP-TV recommended for mismatch scenarios)
x̂_corrected = gap_tv_cassi(y_noisy, phi_corrected, n_iter=50)
```

**Metrics:** PSNR_corrected, SSIM_corrected, SAM_corrected
- Parameter errors: |dx_true - dx_hat2|, |dy_true - dy_hat2|, |θ_true - θ_hat2|
- Demonstrates practical correction effectiveness for real hardware

### 3.4 Scenario IV: Truth Forward Model (Oracle for Corrupted Measurement)

**Purpose:** Oracle scenario showing the best achievable reconstruction when the exact ground truth mismatch parameters (dx_true, dy_true, θ_true) are known. This serves as an upper bound for Scenarios II and III, allowing us to quantify the gap to ideal performance.

**Mask source:** TSA real experimental mask, warped with KNOWN ground truth mismatch
- Load: `/home/spiritai/MST-main/datasets/TSA_real_data/mask.mat`
- Apply TRUE mismatch transformation: `mask_truth = warp_affine(mask_real, dx=dx_true, dy=dy_true, theta=theta_true)`

**Forward model:** Simulated with N=4, K=2 (enlarged grid, stride-1 dispersion)

**Measurement generation:**
```
1. Use SAME corrupted measurement y_noisy as Scenarios II & III:
   y_noisy = measurement with injected mismatch + noise (from section 3.3, step 5)

2. Build Truth Forward Model operator:
   mask_truth = warp_affine(mask_real_data, dx=dx_true, dy=dy_true, theta=theta_true)
   operator_truth = SimulatedOperator_EnlargedGrid(mask_truth, N=4, K=2)

3. Reconstruct using truth forward model:
   x̂_iv = gap_tv_cassi(y_noisy, operator_truth, n_iter=50)
```

**Key Property:** This scenario uses the EXACT true mismatch parameters, making it an oracle that shows what's theoretically achievable given measurement corruption. The gap between Scenario IV and Scenario III (Gap III→IV) quantifies the parameter estimation error from Algorithms 1 and 2.

**Reconstruction:**
```
Operator: phi_iv = SimulatedOperator_EnlargedGrid(mask_truth, N=4, K=2)
x̂_iv = gap_tv_cassi(y_noisy, phi_iv, n_iter=50)
```

**Metrics:** PSNR_iv, SSIM_iv, SAM_iv
- Gap III→IV (Alg1): Residual error from Alg1 parameter estimation (~1-2 dB)
- Gap III→IV (Alg2): Residual error from Alg2 parameter estimation (~0.5-1.5 dB)
- Gap IV→I: Irreducible loss from measurement corruption (~2-4 dB)
- Demonstrates how close the estimated corrections approach the truth

---

## Part 4: Comparison & Analysis

### 4.1 Four-Scenario Comparison Table (Per Scene)

| Scenario | Measurement | Mask | Operator | Purpose |
|----------|-------------|------|----------|---------|
| **I. Ideal** | y_ideal (clean, perfect) | mask_ideal (perfect) | Ideal direct (stride-2) | Oracle upper bound (perfect measurement) |
| **II. Assumed** | y_corrupt (misaligned+noise) | mask_assumed (perfect, no correction) | Simulated N=4, K=2 | Baseline: corruption without correction |
| **III. Corrected** | y_corrupt (misaligned+noise) | mask_corrected (estimated via Alg1/2) | Simulated N=4, K=2 | Practical: corruption with estimated correction |
| **IV. Truth Forward Model** | y_corrupt (misaligned+noise) | mask_truth (known mismatch) | Simulated N=4, K=2 | Oracle for corrupted measurement (knows true mismatch) |

**Expected PSNR hierarchy:**
```
PSNR_ideal ≥ PSNR_iv > PSNR_corrected ≥ PSNR_assumed

Gap I→IV: Impact of measurement corruption (noise + quantization)
Gap IV→II: Impact of ignoring true mismatch (no correction)
Gap II→III: Gain from estimated mismatch correction (Alg1/2)
Gap III→IV: Residual error from parameter estimation inaccuracy
Gap IV→I: Irreducible loss from measurement corruption
```

**Interpretation:**
- **Scenario I (Ideal):** Oracle showing best possible - clean measurement with perfect setup
- **Scenario II (Assumed):** Worst case - shows impact of ignoring mismatch entirely
- **Scenario III (Corrected):** Practical case - shows correction effectiveness when estimating mismatch parameters
- **Scenario IV (Truth Forward Model):** Oracle for corrupted measurement - best achievable given measurement corruption, using known ground truth mismatch
- **Validation metric:** Gap III→IV measures how close estimated correction gets to truth (Alg1/2 accuracy)

### 4.2 Parameter Recovery Accuracy (Algorithm 1 vs 2)

**Background:** In the simulation, we **deliberately inject ground truth mismatch parameters** (dx_true, dy_true, θ_true) to corrupt the scene and mask, then evaluate how accurately each algorithm recovers these known values.

**Parameter Recovery Comparison (Ground Truth vs Estimate):**

**Algorithm 1 (Hierarchical Beam Search):**
```
Typical recovery accuracy:
  dx: recovers within ±0.1–0.2 px of dx_true
  dy: recovers within ±0.1–0.2 px of dy_true
  θ: recovers within ±0.02–0.05° of θ_true

Computational cost: ~4.5 hours per scene
```

**Algorithm 2 (Joint Gradient Refinement):**
```
Expected accuracy (from Algorithm 1 as warm start):
  dx: recovers within ±0.05–0.1 px of dx_true
  dy: recovers within ±0.05–0.1 px of dy_true
  θ: recovers within ±0.01–0.03° of θ_true

Computational cost: ~2.5 hours per scene
```

**Improvement Factor:**
- **Alg2 achieves 3–5× lower parameter errors than Alg1**
- Error_Alg1 / Error_Alg2 ≈ 3–5× (Algorithm 2 more accurate)
- Example: if Alg1 error = 0.15 px, Alg2 error ≈ 0.05 px (3× improvement)

---

## Part 5: Implementation Details

### 5.1 Simulated Operator Class

```python
class SimulatedOperator_EnlargedGrid(PhysicsOperator):
    """
    CASSI forward model with N=4 spatial, K=2 spectral enlargement.
    Measurement is downsampled to original 256×310 size.
    """

    def __init__(self, mask_256, N=4, K=2, stride=1):
        """
        Args:
            mask_256: 256×256 mask (real-valued, upsampled internally)
            N: spatial enlargement factor (4)
            K: original dispersion shift step (2 pixels/band)
            stride: shift step in enlarged space (1 for simulation)
        """
        self.mask_256 = mask_256
        self.N = N
        self.K = K
        self.stride = stride

        # Precompute upsampled mask
        self.mask_enlarged = upsample_spatial(mask_256, N)  # 1024×1024

    def forward(self, x_256):
        """
        Input: x_256 (256×256×28)
        Output: y (256×310)
        """
        # Step 1: Spatial upsample
        x_spatial = upsample_spatial(x_256, self.N)  # 1024×1024×28

        # Step 2: Spectral interpolate
        x_expanded = interpolate_spectral_217(x_spatial, K=self.K)  # 1024×1024×217

        # Step 3: Forward model on enlarged grid (stride=1)
        y_enlarged = self._forward_enlarged(x_expanded, self.mask_enlarged)  # 1024×1240

        # Step 4: Downsample to original size
        y = downsample_spatial(y_enlarged, self.N)  # 256×310

        return y

    def _forward_enlarged(self, x_expanded, mask_enlarged):
        """Forward model on enlarged grid with 217 frames."""
        H, W, L = x_expanded.shape
        W_meas = W + (L - 1) * self.stride
        y = np.zeros((H, W_meas))
        n_c = (L - 1) / 2.0

        for k in range(L):
            d_k = int(round(self.stride * (k - n_c)))
            coded = x_expanded[:, :, k] * mask_enlarged

            y_start, y_end = max(0, d_k), min(W_meas, W + d_k)
            src_start = max(0, -d_k)
            src_end = src_start + (y_end - y_start)

            y[:, y_start:y_end] += coded[:, src_start:src_end]

        return y

    def adjoint(self, y):
        """Adjoint for reconstruction (approximate)."""
        # Upsample measurement
        y_enlarged = upsample_spatial(y, self.N)  # 1024×1240

        # Adjoint on enlarged grid
        x_expanded_hat = self._adjoint_enlarged(y_enlarged, self.mask_enlarged)

        # Downsample spatial
        x_spatial_hat = downsample_spatial(x_expanded_hat, self.N)

        # Downsample spectral
        x_hat = downsample_spectral_217(x_spatial_hat, K=self.K)

        return x_hat

    def apply_mask_correction(self, dx, dy, theta):
        """Apply mask correction for mismatch."""
        mask_corrected = warp_affine(self.mask_256, dx=dx, dy=dy, theta=theta)
        self.mask_enlarged = upsample_spatial(mask_corrected, self.N)
```

### 5.2 UPWMI Algorithm 1: Hierarchical Beam Search

**Purpose:** Fast coarse estimation of all 6 mismatch parameters using hierarchical search.

**Parameter Groups (estimated sequentially):**

**Group 1 - Mask Affine (CRITICAL, highest impact):**
```
dx ∈ [-3, 3] px (13 values)
dy ∈ [-3, 3] px (13 values)
theta ∈ [-1°, 1°] (7 values)
Search space: 13 × 13 × 7 = 1,183 combinations
```

**Group 2 - Dispersion (IMPORTANT, moderate impact):**
```
a1 ∈ [1.95, 2.05] px/band (5 values)
alpha ∈ [-1°, 1°] (7 values)
Search space: 5 × 7 = 35 combinations
```

**Algorithm 1 Implementation:**
```python
def upwmi_algorithm_1_hierarchical_beam_search(y_meas, mask_real, x_true,
                                              n_iter_proxy=5, n_iter_beam=10):
    """
    Hierarchical beam search for all 6 mismatch parameters.

    Input:
        y_meas: Corrupted measurement (256×310)
        mask_real: Real experimental mask
        x_true: Ground truth scene

    Output:
        (dx_hat, dy_hat, theta_hat, a1_hat, alpha_hat)
    """

    # PHASE 1: Estimate Mask Affine (dx, dy, theta) - highest impact
    print("Phase 1: Estimating mask geometry (dx, dy, theta)...")
    search_space_affine = {
        'dx': np.linspace(-3, 3, 13),
        'dy': np.linspace(-3, 3, 13),
        'theta': np.linspace(-np.pi/180, np.pi/180, 7),
    }

    # 1D sweeps for each parameter
    best_dx = search_1d_parameter('dx', search_space_affine['dx'], y_meas, mask_real,
                                  x_true, n_iter=n_iter_proxy)
    best_dy = search_1d_parameter('dy', search_space_affine['dy'], y_meas, mask_real,
                                  x_true, n_iter=n_iter_proxy)
    best_theta = search_1d_parameter('theta', search_space_affine['theta'], y_meas,
                                     mask_real, x_true, n_iter=n_iter_proxy)

    # Beam search on 3D affine space (5×5×5 top candidates)
    top_5_affine = beam_search_affine(best_dx, best_dy, best_theta, y_meas, mask_real,
                                       x_true, n_iter=n_iter_beam, beam_width=5)

    # Coordinate descent refinement
    best_affine = coordinate_descent_3d(top_5_affine, y_meas, mask_real, x_true,
                                        n_iter=n_iter_beam, n_rounds=3)

    (dx_hat, dy_hat, theta_hat) = best_affine

    # PHASE 2: Estimate Dispersion (a1, alpha) - moderate impact
    print("Phase 2: Estimating dispersion (a1, alpha)...")
    search_space_disp = {
        'a1': np.linspace(1.95, 2.05, 5),
        'alpha': np.linspace(-np.pi/180, np.pi/180, 7),
    }

    # Apply corrected mask, now search for dispersion parameters
    mask_corrected_affine = warp_affine(mask_real, dx=dx_hat, dy=dy_hat, theta=theta_hat)

    # 2D beam search for dispersion (5×7)
    top_5_disp = beam_search_dispersion(search_space_disp['a1'], search_space_disp['alpha'],
                                        y_meas, mask_corrected_affine, x_true,
                                        n_iter=n_iter_beam, beam_width=5)

    (a1_hat, alpha_hat) = top_5_disp[0]  # Best candidate

    return (dx_hat, dy_hat, theta_hat, a1_hat, alpha_hat)
```

**Computational Cost:**
- Phase 1 (1D sweeps): ~1.5 hours (13+13+7 = 33 searches × 5-10 iterations each)
- Phase 1 (beam search): ~2 hours (125 combinations × 10 iterations)
- Phase 2 (beam search): ~1 hour (35 combinations × 10 iterations)
- **Total: ~4.5 hours per scene**
- **Accuracy: ±0.1-0.2 px (mask), ±0.01 px/band (dispersion)**

### 5.3 UPWMI Algorithm 2: Joint Gradient Refinement

**Purpose:** Refine all 6 mismatch parameters jointly using unrolled differentiable solver.

**Algorithm 2 Implementation:**
```python
def upwmi_algorithm_2_joint_gradient_refinement(y_meas, x_true, mask_real,
                                               coarse_estimate,
                                               n_iter_unroll=10):
    """
    Joint gradient-based refinement for all 6 mismatch parameters.

    Input:
        y_meas: Corrupted measurement (256×310, full measurement)
        x_true: Ground truth scene
        mask_real: Real experimental mask
        coarse_estimate: (dx1, dy1, θ1, a1_1, α1) from Algorithm 1
        n_iter_unroll: Unroll depth for GAP-TV (default 10)

    Output:
        (dx_refined, dy_refined, θ_refined, a1_refined, α_refined)
    """

    # Initialize differentiable parameters from coarse estimate
    dx = torch.nn.Parameter(torch.tensor(coarse_estimate[0], dtype=torch.float32))
    dy = torch.nn.Parameter(torch.tensor(coarse_estimate[1], dtype=torch.float32))
    theta = torch.nn.Parameter(torch.tensor(coarse_estimate[2], dtype=torch.float32))
    a1 = torch.nn.Parameter(torch.tensor(coarse_estimate[3], dtype=torch.float32))
    alpha = torch.nn.Parameter(torch.tensor(coarse_estimate[4], dtype=torch.float32))

    def loss_fn(y_meas, x_ref):
        """Unrolled reconstruction loss."""
        # Group 1: Affine mask correction
        mask_warped = differentiable_warp_affine(mask_real, dx=dx, dy=dy, theta=theta)

        # Group 2: Dispersion parameter update (modify forward model)
        operator = DifferentiableSimulatedOperator_WithDispersion(
            mask_warped, N=4, K=2,
            a1_correction=a1,    # Adjust dispersion slope
            alpha_correction=alpha  # Adjust dispersion axis
        )

        # Unroll GAP-TV iterations for differentiability
        x_hat = unrolled_gap_tv(y_meas, operator, K=n_iter_unroll)

        # Reconstruction loss
        loss = F.mse_loss(x_hat, x_ref)
        return loss

    # Phase 1: Optimize on full measurement (100 epochs, lr=0.01)
    print("Phase 1: Optimizing all parameters on full measurement...")
    params = [dx, dy, theta, a1, alpha]
    optimizer = torch.optim.Adam(params, lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    for epoch in range(100):
        optimizer.zero_grad()
        loss = loss_fn(y_meas, x_true)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: loss={loss.item():.6f}, " +
                  f"dx={dx.item():.3f}, a1={a1.item():.4f}")

    # Phase 2: Fine-tune on 10-scene dataset (50 epochs, lr=0.001)
    print("Phase 2: Fine-tuning on 10-scene ensemble...")
    optimizer = torch.optim.Adam(params, lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    for epoch in range(50):
        optimizer.zero_grad()
        loss = loss_fn(y_meas, x_true)  # Applied to all 10 scenes in practice
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=0.5)
        optimizer.step()
        scheduler.step()

        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: loss={loss.item():.6f}, " +
                  f"dy={dy.item():.3f}, alpha={alpha.item():.4f}")

    # Return all refined parameters
    return (dx.item(), dy.item(), theta.item(), a1.item(), alpha.item())
```

**Computational Cost:**
- Phase 1 (100 epochs × K=10 unroll on full measurement): ~1.5 hours
- Phase 2 (50 epochs × K=10 unroll on full measurement): ~1 hour
- **Total: ~2.5 hours per scene (faster than Alg1)**
- **Accuracy: ±0.05-0.1 px (mask), ±0.001 px/band (dispersion)**
- **Improvement over Alg1: 3-5× better parameter accuracy**

---

## Part 6: Validation Protocol (10 Scenes)

### 6.1 Per-Scene Workflow

```python
def run_full_validation_scene(scene_idx, x_true_256, mask_ideal_256):
    """
    Complete validation for one scene: all 4 scenarios (Ideal, Assumed, IV, Corrected).

    Returns:
        dict with PSNR values for all 4 scenarios, parameter errors, and gap metrics.
    """

    # ========== SCENARIO I: IDEAL ==========
    y_ideal = forward_ideal(x_true_256, mask_ideal_256)  # 256×310
    x_hat_ideal = solver(y_ideal, forward_ideal, mask_ideal_256, n_iter=50)
    psnr_ideal = psnr(x_hat_ideal, x_true_256)

    # ========== INJECT MISMATCH FOR SCENARIOS II & III ==========
    # Inject same mismatch to both scene and mask
    dx_true = np.random.uniform(-3, 3)
    dy_true = np.random.uniform(-3, 3)
    theta_true = np.random.uniform(-np.pi/180, np.pi/180)

    x_misaligned = warp_affine(x_true_256, dx=dx_true, dy=dy_true, theta=theta_true)
    mask_misaligned = warp_affine(mask_ideal_256, dx=dx_true, dy=dy_true, theta=theta_true)

    # Generate corrupted measurement
    y_enlarged = forward_enlarged(enlarge(x_misaligned), upsample(mask_misaligned, 4))
    y_corrupt = downsample(y_enlarged, 4)
    y_noisy = add_noise(y_corrupt, peak=10000, sigma=0.01)

    # ========== SCENARIO II: ASSUMED (Baseline, no correction) ==========
    # Use corrupted measurement but perfect mask (ignoring mismatch)
    x_hat_assumed = solver(y_noisy, SimulatedOperator_EnlargedGrid(mask_ideal_256), n_iter=50)
    psnr_assumed = psnr(x_hat_assumed, x_true_256)

    # ========== SCENARIO IV: TRUTH FORWARD MODEL (Oracle for corrupted measurement) ==========
    # Use corrupted measurement with TRUE forward model that knows the exact mismatch
    # This is an oracle: knows ground truth dx_true, dy_true, theta_true
    mask_truth = warp_affine(mask_real_data, dx=dx_true, dy=dy_true, theta=theta_true)
    operator_truth = SimulatedOperator_EnlargedGrid(mask_truth, N=4, K=2)
    x_hat_iv = gap_tv_cassi(y_noisy, operator_truth, n_iter=50)
    psnr_iv = psnr(x_hat_iv, x_true_256)

    # ========== SCENARIO III: CORRECTED ==========

    # Algorithm 1: Hierarchical beam search (coarse, fast)
    print(f"Scene {scene_idx}: Running Algorithm 1 (Hierarchical Beam Search)...")
    (dx_hat1, dy_hat1, theta_hat1, a1_hat1, alpha_hat1) = upwmi_algorithm_1_hierarchical_beam_search(
        y_noisy, mask_real_data, x_misaligned
    )

    # Reconstruct with Algorithm 1 correction
    mask_corrected_1 = warp_affine(mask_real_data, dx=dx_hat1, dy=dy_hat1, theta=theta_hat1)
    operator_alg1 = SimulatedOperator_EnlargedGrid(mask_corrected_1, N=4, K=2)
    operator_alg1.apply_dispersion_correction(a1=a1_hat1, alpha=alpha_hat1)
    x_hat_alg1 = gap_tv_cassi(y_noisy, operator_alg1, n_iter=50)

    psnr_alg1 = psnr(x_hat_alg1, x_true_256)
    err_dx_alg1, err_dy_alg1 = abs(dx_hat1 - dx_true), abs(dy_hat1 - dy_true)
    err_theta_alg1 = abs(theta_hat1 - theta_true) * 180 / np.pi
    err_a1_alg1 = abs(a1_hat1 - 2.0)  # True a1=2.0
    err_alpha_alg1 = abs(alpha_hat1 - alpha_true) * 180 / np.pi

    # Algorithm 2: Joint gradient refinement (fine, accurate)
    print(f"Scene {scene_idx}: Running Algorithm 2 (Joint Gradient Refinement)...")
    (dx_hat2, dy_hat2, theta_hat2, a1_hat2, alpha_hat2) = upwmi_algorithm_2_joint_gradient_refinement(
        y_noisy, x_true_256, mask_real_data,
        coarse_estimate=(dx_hat1, dy_hat1, theta_hat1, a1_hat1, alpha_hat1),
        n_iter_unroll=10
    )

    # Reconstruct with Algorithm 2 correction
    mask_corrected_2 = warp_affine(mask_real_data, dx=dx_hat2, dy=dy_hat2, theta=theta_hat2)
    operator_alg2 = SimulatedOperator_EnlargedGrid(mask_corrected_2, N=4, K=2)
    operator_alg2.apply_dispersion_correction(a1=a1_hat2, alpha=alpha_hat2)
    x_hat_alg2 = gap_tv_cassi(y_noisy, operator_alg2, n_iter=50)

    psnr_alg2 = psnr(x_hat_alg2, x_true_256)
    err_dx_alg2, err_dy_alg2 = abs(dx_hat2 - dx_true), abs(dy_hat2 - dy_true)
    err_theta_alg2 = abs(theta_hat2 - theta_true) * 180 / np.pi
    err_a1_alg2 = abs(a1_hat2 - 2.0)
    err_alpha_alg2 = abs(alpha_hat2 - alpha_true) * 180 / np.pi

    # ========== RETURN RESULTS ==========
    return {
        'scene_idx': scene_idx,
        'dx_true': dx_true, 'dy_true': dy_true, 'theta_true': theta_true * 180 / np.pi,

        # Four scenarios (in order I, II, III, IV)
        'psnr_ideal': psnr_ideal,                 # Scenario I: ideal measurement, ideal mask
        'psnr_assumed': psnr_assumed,             # Scenario II: corrupted measurement, assumed perfect mask
        'psnr_alg1': psnr_alg1,                   # Scenario III: corrupted measurement, Alg1-corrected mask
        'psnr_alg2': psnr_alg2,                   # Scenario III: corrupted measurement, Alg2-corrected mask
        'psnr_iv': psnr_iv,                       # Scenario IV: corrupted measurement, true forward model (oracle)

        # Algorithm 1 parameter errors
        'err_dx_alg1': err_dx_alg1, 'err_dy_alg1': err_dy_alg1, 'err_theta_alg1': err_theta_alg1,

        # Algorithm 2 parameter errors
        'err_dx_alg2': err_dx_alg2, 'err_dy_alg2': err_dy_alg2, 'err_theta_alg2': err_theta_alg2,

        # Key comparisons
        'loss_corruption_no_correction': psnr_ideal - psnr_assumed,  # Gap I→II (impact of corruption)
        'gap_iv_to_ideal': psnr_ideal - psnr_iv,                     # Gap IV→I (irreducible corruption loss)
        'gap_assumed_to_iv': psnr_iv - psnr_assumed,                 # Gap II→IV (impact of ignoring true mismatch)
        'gain_from_alg1_correction': psnr_alg1 - psnr_assumed,       # Gap II→III (Alg1 correction gain)
        'residual_error_alg1': psnr_iv - psnr_alg1,                  # Gap III→IV (Alg1 estimation error)
        'gain_from_alg2_correction': psnr_alg2 - psnr_assumed,       # Gap II→III (Alg2 correction gain)
        'residual_error_alg2': psnr_iv - psnr_alg2,                  # Gap III→IV (Alg2 estimation error)
        'gap_alg1_to_oracle': psnr_ideal - psnr_alg1,                # Gap III→I (Alg1 total gap to ideal)
        'gap_alg2_to_oracle': psnr_ideal - psnr_alg2,                # Gap III→I (Alg2 total gap to ideal)
        'improvement_alg2_over_alg1': psnr_alg2 - psnr_alg1,         # Alg2 vs Alg1 comparison
    }
```

### 6.2 Reporting

Create comprehensive report: `pwm/reports/cassi_enlarged_grid_complete.md`

**Tables:**
1. **Four-scenario PSNR comparison** (10 scenes)
   - PSNR_ideal, PSNR_assumed, PSNR_iv, PSNR_alg1, PSNR_alg2
   - Gap metrics: I→IV, II→IV, III(Alg1/Alg2)→IV, III→I

2. **Parameter recovery accuracy** (Algorithm 1 vs 2)
   - Ground truth (dx_true, dy_true, θ_true) vs estimates
   - Parameter errors for Alg1 and Alg2
   - Residual error (IV - reconstruction PSNR)

3. **Scenario comparison metrics**
   - Loss from corruption without correction (Gap I→II)
   - Loss from ignoring true mismatch (Gap II→IV)
   - Gain from Alg1/Alg2 correction (Gap II→III)
   - Residual estimation error (Gap III→IV)

4. **Timing breakdown**
   - Forward model time, Algorithm 1 time, Algorithm 2 time, total per scene

**Figures:**
1. PSNR trajectory across 10 scenes (ideal / assumed / iv / alg1 / alg2)
2. Parameter error scatter plots (dx, dy, θ true vs Alg1 vs Alg2 estimated)
3. Gap hierarchy visualization (shows I→IV→II and III→IV→I relationships)
4. Reconstruction visual comparisons (2-3 scenes, 3 bands, all 4 scenarios)

---

## Part 7: Key Design Decisions

| Decision | Value | Rationale |
|----------|-------|-----------|
| **Spatial enlargement N** | 4 | Finer rotation quantization (1/4 pixel) |
| **Spectral formula** | (L-1)×N×K+1 = 217 | Fine dispersion sampling (stride-1 in enlarged space) |
| **Scene crop P** | 16 px/side | Prevents signal leakage from ±3 px mismatch |
| **Real mask source** | TSA dataset | Realistic coded aperture pattern |
| **Dispersion shift (simulation)** | 1 pixel/frame | Fine granularity (217 frames total) |
| **Dispersion shift (original)** | 2 pixel/band | Standard SD-CASSI step |
| **Measurement size** | 256×310 | Downsampled from 1024×1240 |
| **Three scenarios** | Ideal / Assumed / Corrected | Show simulation fidelity + correction quality |
| **Mismatch range** | dx,dy∈[-3,3] px, θ∈[-1°,1°] | Realistic assembly tolerances |

---

## Part 8: Timeline & Compute Requirements

### Phase 1: Code Development (20 hours)
- Scene preprocessing + crop logic: 2 h
- Enlarged forward model (N=4, K=2): 4 h
- Real mask loading + upsampling: 1 h
- Spectral interpolation (217 bands): 2 h
- Algorithm 1 (beam search): 4 h
- Algorithm 2 (gradient refinement): 4 h
- Operator classes + utilities: 2 h
- Tests + fixtures: 1 h

### Phase 2: Single-Scene Validation (GPU: 13 hours)
- Scenario I (ideal): 1 h
- Scenario II (assumed): 2 h
- Scenario IV (truth forward model): 2 h
- Scenario III + Alg1 + Alg2: 8 h

### Phase 3: Full 10-Scene Execution (GPU: ~130 hours)
- Scenarios I, II, & IV: 15 h (quick, just forward models with known ground truth)
- Algorithm 1 (10 scenes): 45 h
- Algorithm 2 (10 scenes): 70 h

### Phase 4: Analysis & Reporting (4 hours)
- Aggregate results, generate tables/figures: 4 h

**Total calendar time:** ~5-6 days (with 24/7 GPU)

---

## Conclusion

This revised plan provides:
1. **Realistic simulation:** N=4 enlargement + real mask + 217-frame dispersion
2. **Four-scenario validation:** Ideal / Assumed / IV / Corrected (clear benchmarking with oracle for corrupted measurement)
3. **Robust mismatch correction:** Algorithms 1 & 2 with proper gradient refinement
4. **Comprehensive metrics:** PSNR/SSIM/SAM + parameter recovery + timing + residual estimation error

**Expected outcomes:**
- **Scenario I (Ideal):** ~28–30 dB (oracle, clean measurement + perfect setup)
- **Scenario II (Assumed):** ~18–21 dB (corrupted measurement, ignored mismatch, worst case)
- **Scenario IV (Truth Forward Model):** ~23–26 dB (corrupted measurement, known ground truth mismatch, oracle for corrupted case)
- **Scenario III-Alg1:** ~22–24 dB (corrupted measurement, Alg1-estimated correction, residual error ~1-2 dB vs IV)
- **Scenario III-Alg2:** ~23–25 dB (corrupted measurement, Alg2-estimated correction, residual error ~0.5-1.5 dB vs IV)
- **Parameter accuracy:** Alg1: ±0.1-0.2 px, Alg2: ±0.05-0.1 px (recovery of ground truth mismatch)

**Key comparisons:**
- Gap I→IV: ~2-4 dB (irreducible loss from measurement corruption)
- Gap II→IV: ~4-6 dB (impact of ignoring true mismatch)
- Gain II→III (Alg1): ~3-4 dB (Alg1 correction effectiveness)
- Residual III→IV (Alg1): ~1-2 dB (Alg1 parameter estimation error)
- Gain II→III (Alg2): ~4-5 dB (Alg2 correction effectiveness)
- Residual III→IV (Alg2): ~0.5-1.5 dB (Alg2 parameter estimation error, closer to truth)
- Gap IV→I (Alg2): ~2-4 dB (total loss from simulation + corruption)

---

**Plan version:** v4 Complete Redesign (2026-02-15)
**Ready for implementation upon approval.**
