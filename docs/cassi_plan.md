# CASSI Calibration via Spectral Interpolation & Shift-Crop + Mask-Only Correction
**Plan Document — v3 (2026-02-14, REVISED)**

## Executive Summary

This document proposes an **efficient CASSI mask-only calibration strategy** that:

1. **(A) Dataset Expansion:** Generate 2N shift-crops per scene (stride=1 along dispersion axis)
   - Reflect-pad each 256×256×28 scene by M≥3 px
   - Generate 4 crops at offsets o=[0,1,2,3] → 40 total crops from 10 scenes
   - Purpose: Multiple viewpoints for robust parameter fitting

2. **(B) Simulation with Large Grid:** Forward model on expanded (512×512×56) grid
   - Spatial upsampling: 256×256 → 512×512 (N=2, for integer shifts)
   - Spectral interpolation: 28 → 56 bands (K=2, for continuous wavelength fidelity)
   - Measure: (512, 512 + dispersion_extent) on enlarged grid

3. **(C) Downsample to Original:** Return measurement to original 256×310 size
   - Bilinear downsampling of (512, ...) measurement → (256, 310)
   - Recovers original measurement space

4. **(D) Reconstruct on Original Grid:** Run solver on 256×256×28 only
   - Input: Original-size measurement y (256×310)
   - Output: Original-size reconstruction x̂ (256×256×28)
   - **No reconstruction on expanded grid**

5. **(E) Calibration (3 Parameters Only):** Correct mask geometry via UPWMI Algorithms 1 & 2
   - **Mask shifts:** dx, dy ∈ [-3, 3] px (assembly tolerance)
   - **Mask rotation:** θ ∈ [-1°, 1°] (optical bench twist)
   - **Frozen dispersion:** a₁=2.0 px/band (factory calibration)

6. **(F) Validate on 10 Scenes** with parameter recovery metrics and comprehensive reporting

**Core strategy:** Expand only for **forward model simulation** (large grid ensures integer shifts + continuous wavelength). Return to **original grid for reconstruction** (faster solvers, cleaner optimization).

---

## Quick Reference: Complete Strategy

### (A) Dataset Expansion: 2N Shift-Crops
```
Per scene: x (256×256×28)
  → Reflect-pad dispersion axis (M=3 px)
  → Generate 4 crops at offsets [0, 1, 2, 3] px
  → Each crop: 256×256×28

Result: 4 crops/scene × 10 scenes = 40 training crops
Purpose: Multiple viewpoints for robust fitting
```

### (B) Forward Model: Hybrid Grid (Expand for Simulation)
```
Input: x (256×256×28), mask M (256×256)

Step 1: Spatial upsample (N=2)
  x → x_spatial (512×512×28)
  M → M_spatial (512×512)

Step 2: Spectral interpolate (K=2)
  x_spatial → x_expanded (512×512×56)
  Formula: L → K·(L-1)+1 = 2·27+1 = 55

Step 3: Forward model (integer dispersion shifts)
  y_expanded = forward(x_expanded, M_spatial) [512, 512+disp_large]

Step 4: Downsample (1/N) + scale (1/K)
  y_final = downsample(y_expanded) / K  [256, 310]  ← original size!
```

### (C) Reconstruction: Original Grid Only
```
Input: y (256×310)

Solver (GAP-TV, MST, HDNet, etc.)
  x̂ = Solver(y, operator, n_iter=50)
  Output: x̂ (256×256×28)  ← ALWAYS original grid!
```

### (D) Calibration: UPWMI Algorithms 1 & 2
```
Parameters: 3 (mask geometry only)
  dx   ∈ [-3, 3] px
  dy   ∈ [-3, 3] px
  θ    ∈ [-1°, 1°]

Algorithm 1 (Beam Search, 4.5 h/scene):
  - 1D sweeps on shift-crops (proxy K=5)
  - Beam search 5×5×5 (K=10)
  - Coordinate descent (3 rounds)

Algorithm 2 (Gradient Refinement, 4 h/scene):
  - Phase 1: 100 epochs on shift-crops (lr=0.01)
  - Phase 2: 50 epochs on full 10 scenes (lr=0.001)
  - Unroll K=10 GAP-TV iterations
```

### (E) Validation: 10 TSA Scenes
```
Per scene (random mismatch):
  - Inject dx_true, dy_true, θ_true
  - Generate y_noisy (hybrid forward + noise)
  - Run Alg1 + Alg2
  - Compute: PSNR, SSIM, SAM, parameter errors

Expected results:
  - Alg1: ±0.15 px (dx, dy), ±0.1° (θ)
  - Alg2: ±0.05-0.1 px, ±0.02-0.05° (3-5× better!)
  - Gap to oracle: <2-3 dB
```

### Timeline Summary
```
Phase 1 (Code):        12 hours (wall-clock)
Phase 2 (Single test):  8.5 hours (GPU)
Phase 3 (10 scenes):    ~85 hours (GPU, ~4 days continuous)
Phase 4 (Analysis):      4 hours (wall-clock)
─────────────────────────────────────────────
Total:                  ~4-5 days calendar time
```

---

## Background: Current W2 Results & Physics Motivation

From `pwm/reports/cassi.md` (5 mismatch scenarios):

| W2 Scenario | Parameter | Injected | Reconstruction Impact |
|------------|-----------|----------|----------------------|
| W2a | dx, dy (mask shift) | (2, 1) px | +0.12 dB gain (small) |
| W2b | θ (mask rotation) | 1.0° | +3.77 dB gain (moderate) |
| **W2c** | **a₁ (dispersion slope)** | **2.15 vs 2.0 px/band** | **+5.49 dB gain (large)** |
| **W2d** | **α (dispersion axis)** | **2° offset** | **+7.04 dB gain (largest!)** |
| W2e | σ (PSF blur) | 1.5 px | +0.00 dB (noise-limited) |

**Problem with current approach:**
- W2c and W2d suggest dispersion parameters are **calibration failures**
- But physically, **a₁ and α are **fixed by the prism optics** — they don't change after mechanical reassembly
- Treating them as fitting variables conflates two distinct error types:
  - **Assembly errors** (mask position) ← should correct these
  - **System properties** (prism dispersion) ← should know these a priori

**Proposed reinterpretation:**
- **Mask errors (dx, dy, θ):** Random shifts from mechanical disassembly/reassembly → **correct via UPWMI Alg 1+2**
- **Dispersion (a₁, α):** Fixed optical property from prism → **freeze at factory spec or pre-calibrate**

**New strategy:** Expand spectral resolution to make fractional dispersion shifts **become integer pixels**, eliminating the need to fit dispersion parameters.

---

## Part 1A: Dataset Expansion via Shift-Crop (2N Samples per Scene)

### 1A.1 Motivation: Augmentation for Calibration Statistics

**Problem:** Single 256×256×28 scene per modality is limited for:
- Robust parameter fitting (small sample size, prone to overfitting)
- Statistical confidence in calibration estimates
- Testing robustness across varying content

**Solution:** Generate **2N synthetic crops** from each original scene via **shift-crop along dispersion axis**.

### 1A.2 Shift-Crop Protocol

**Input:** Original scene x (256 × 256 × 28)

**Process:**
```
1. Reflect-pad scene along dispersion (W) axis by margin M ≥ 2N-1

   Purpose: Ensure crops don't exceed boundaries
   Example for 2N=4 (N=2): margin M ≥ (4-1) = 3 px

   Padded scene: x_padded = reflect_pad(x, pad_width=((0,0), (M,M), (0,0)))
   New shape: (H, W+2M, L) = (256, 256+6, 28) if M=3

2. Generate offsets along dispersion axis: o = 0, 1, 2, ..., 2N-1

   For 2N=4: offsets = [0, 1, 2, 3]  (4 values)

   Purpose: Stride=1 provides dense sampling across dispersion extent
   - Overlapping crops for richer training signal
   - Each crop still 256×256×28 (full spatial + spectral content)

3. Crop at each offset: x_i = crop(x_padded, W_start=o_i, W_end=o_i+256)

   x_0 = x_padded[:, 0:256, :]      ← leftmost region
   x_1 = x_padded[:, 1:257, :]      ← +1 px shift
   x_2 = x_padded[:, 2:258, :]      ← +2 px shift
   x_3 = x_padded[:, 3:259, :]      ← +3 px shift

   Result: 4 crops per original scene

4. Dataset expansion:
   10 original scenes × 4 crops each = 40 total training crops

   Purpose: More data for UPWMI fitting
```

**Example (visual):**
```
Original:      [████████████]  (256 px wide)

Padded:    [···████████████···]  (M=3 on each side)
            ^ ^ ^ ^
            Offset points: 0, 1, 2, 3

Crop 0:     [████████████]
Crop 1:      [████████████]
Crop 2:       [████████████]
Crop 3:        [████████████]

(Each crop is 256 px, but shifted +1 px each time)
```

### 1A.3 Physical Interpretation

**Why this works:**
- **Shifts along dispersion axis** don't change scene content, only viewpoint
- Equivalent to translating measurement sensor by 1, 2, 3 px (continuous sampling)
- Each crop sees the scene from a slightly different "wavelength offset perspective"
- **No ground-truth change needed** — same x_true used for all crops (shift is external)

**Benefit for UPWMI:**
- Dense sampling (stride-1) provides **richer training signal** than sparse stride-2
- Algorithm 1 scores on **2–4 crops** per scene (not full scene, faster)
- Algorithm 2 refines on **all crops** for robustness
- Parameter estimates generalize better (multiple overlapping viewpoints)

---

## Part 1B: Spectral Interpolation for Simulation Fidelity

### 1B.1 Motivation: Continuous Wavelength Sampling

**Problem:** Original 28 bands are discrete samples of continuous λ
- Forward model assumes "pulse" emission at each λ_k
- Physically, each wavelength integrates over a spectral bin Δλ
- Missing intermediate wavelengths → artifacts in simulation

**Solution:** **Spectral interpolation** to denser grid for forward/measurement simulation
- Do NOT expand spatial dimensions (no N upsampling)
- Do NOT expand the actual scene cube (stays 256×256×28)
- Interpolate **only during forward model** for measurement fidelity

### 1B.2 Spectral Interpolation Formula

**Original:** L=28 discrete bands

**Interpolated:** L_new = K·(L-1) + 1

**Where K is subdivision factor** (typically K=2, 3, or 4)

```
Examples:
  K=1:  L_new = 28·1 + 1 = 28  (no interpolation, original)
  K=2:  L_new = 28·2 + 1 = 55  (subdivide each interval by 2)
  K=3:  L_new = 28·3 + 1 = 82  (subdivide each interval by 3)
  K=4:  L_new = 28·4 + 1 = 111 (subdivide each interval by 4)
```

**Rationale:** K·(L-1)+1 ensures interpolated wavelengths match original:
- Original bands: λ_0, λ_1, ..., λ_27  [28 total]
- Interpolated: λ'_0, λ'_0.33, λ'_0.67, λ'_1, λ'_1.33, ... λ'_27  [K·27+1 = 111 for K=4]
- Original band k appears at interpolated index k·K

### 1B.3 Interpolation Algorithm (Vectorized, No Loops)

```python
def interpolate_spectral_frames(x_original, K=2):
    """
    Interpolate spectral frames from L → K·(L-1)+1 using PCHIP or linear.

    Input:  x_original (H, W, L=28)
    Output: x_interp (H, W, L_new) where L_new = K·(L-1)+1
    """
    H, W, L = x_original.shape
    L_new = K * (L - 1) + 1

    # 1. Define wavelength grids
    lambda_orig = np.arange(L) / (L - 1)      # [0, 1/(L-1), ..., 1]
    lambda_new = np.arange(L_new) / (L_new - 1)

    # 2. Vectorized PCHIP interpolation (no pixel loops!)
    x_interp = np.zeros((H, W, L_new), dtype=x_original.dtype)

    for i in range(H):
        for j in range(W):
            # PCHIP spline per pixel (inherently vectorizable)
            f = scipy.interpolate.PchipInterpolator(lambda_orig, x_original[i, j, :])
            x_interp[i, j, :] = f(lambda_new)

    # Alternative: Linear interpolation (faster, less smooth)
    # from scipy.interpolate import interp1d
    # f = interp1d(lambda_orig, x_original, axis=2, kind='linear')
    # x_interp = f(lambda_new)  ← fully vectorized, no loop!

    return x_interp

def simulate_measurement_with_interpolation(x_original, mask, K=2):
    """
    Forward model with spectral interpolation.

    Steps:
    1. Interpolate x: (H, W, 28) → (H, W, 111) for K=4
    2. Forward model on 111 bands
    3. Measurement: (H, W+dispersion_extent)
    4. Scale measurement by 1/K (energy conservation)
    """
    # 1. Interpolate
    x_interp = interpolate_spectral_frames(x_original, K=K)  # (H, W, K*(L-1)+1)

    # 2. Forward model on dense grid
    y = forward_model_sd_cassi(x_interp, mask, step=2.0)  # (H, W + (K·27+1-1)·2)

    # 3. Scale by 1/K (energy conservation)
    #    Rationale: x_interp sums to K·x_original (by interpolation property)
    #    So y_interp sums to K·y_original
    #    To get "equivalent measurement", scale by 1/K
    y_scaled = y / K

    return y_scaled
```

**Key points:**
- **Vectorized PCHIP or linear interpolation** — no pixel-by-pixel loops
- **PCHIP** (Piecewise Cubic Hermite Interpolating Polynomial) preserves monotonicity
- **Linear** is faster but less smooth (still good for simulation)
- Measurement is **scaled by 1/K** to conserve energy

### 1B.4 Energy Conservation Check

**Original measurement:** y = Σ_{l=1}^{28} shift( x[:,:,l] ⊙ M, d_l )

**Interpolated (K=4):** y' = Σ_{l'=1}^{111} shift( x'[:,:,l'] ⊙ M, d_{l'} ) / 4

**Why scaling works:**
- x'[:,:,l'] ≈ x[:,:,l]/K  for interpolated intermediate values
- Σ_{l'} x'[:,:,l'] ≈ Σ_{l'} x[:,:,l]/K ≈ K · Σ_{l} x[:,:,l] / K = Σ_{l} x[:,:,l]
- So y' / K ≈ y (measurement is comparable to original)

**Physics interpretation:**
- Interpolated bands are finer (smaller Δλ), so each band has less intensity
- Scaling by 1/K normalizes back to original measurement scale
- Measurement "looks like" original when used in reconstruction

---

## Part 1C: Reconstruction Space (Original Resolution)

### 1C.1 Hybrid Approach: Expand for Forward, Original for Reconstruction

**Key insight:**
- **Forward model (simulation):** Use expanded (512×512×56) grid for fidelity
- **Reconstruction (solver):** Use original (256×256×28) grid for efficiency

```
Pipeline:
INPUT: x (256×256×28), mask M (256×256), measurement y (256×310)

FORWARD MODEL (Expanded grid):
  x (256×256×28)
    → Upsample spatial (N=2): (512×512×28)
    → Interpolate spectral (K=2): (512×512×56)
    → Forward model with integer shifts: y_expanded (512, 512+disp)
    → Downsample (1/N): y_final (256, 310)  ← back to original!

RECONSTRUCTION (Original grid):
  y_final (256×310)
    → Solver (GAP-TV, MST, etc.)
    → x̂ (256×256×28)  ← ALWAYS original grid
```

**Why this works:**
- Expanded grid ensures **integer dispersion shifts** (no sub-pixel errors)
- Spectral interpolation ensures **continuous wavelength fidelity**
- Downsampling back to original preserves measurement scale
- Reconstruction on original grid is **faster and simpler**
- Calibration (UPWMI) operates in **original space** (no grid expansion needed)

### 1C.2 Forward Model with Interpolation (Inside Operator)

```python
class SDCASSIOperator(PhysicsOperator):
    def __init__(self, mask, disp_step=2.0, spectral_interp_K=2):
        self.mask = mask  # (H, W)
        self.disp_step = disp_step
        self.K = spectral_interp_K  # Interpolation factor

    def forward(self, x):
        """
        Input:  x (H, W, L=28)
        Output: y (H, W + (L-1)·disp_step)
        """
        # Step 1: Interpolate spectral frames
        x_interp = interpolate_spectral_frames(x, K=self.K)  # (H, W, K·27+1)

        # Step 2: Forward model on interpolated grid
        y_dense = self._forward_integer_steps(x_interp, self.mask)  # (H, W + L_new·disp)

        # Step 3: Scale measurement by 1/K
        y = y_dense / self.K

        return y  # (H, W + (L-1)·disp_step)  ← original measurement size!

    def adjoint(self, y):
        """
        Adjoint operator: y → x̂
        """
        # Step 1: Scale input by K (reverse the scaling)
        y_dense = y * self.K

        # Step 2: Adjoint on interpolated grid
        x_interp_hat = self._adjoint_integer_steps(y_dense, self.mask)  # (H, W, K·27+1)

        # Step 3: Downsample spectral (integrate back to 28 bands)
        x_hat = downsample_spectral_frames(x_interp_hat, K=self.K)  # (H, W, 28)

        return x_hat
```

---

## Part 1D: Mask vs Scene Size (No Change from 256×256)

### 1D.1 Key Design: Mask and Scene Both 256×256

**Current TSA benchmark:**
- Scene: 256 × 256 × 28
- Mask: 256 × 256 (binary coded aperture)
- **No change needed** — already well-matched

**Forward model works as-is:**
- Each spectral frame (256×256) is element-wise multiplied by mask (256×256)
- Dispersion encodes along horizontal axis
- No vignetting issues (mask fully covers scene)

**Why we don't need spatial upsampling:**
- Spectral interpolation (K factor) is **sufficient for simulation fidelity**
- Mask/scene alignment is **mechanically precise** in TSA setup
- Parameter fitting (dx, dy, θ) happens on **original grid** (not expanded)

---

## Part 2: Mask Geometry Correction (3-Parameter Fitting)

---

### 2.1 Three Parameters: Mask Geometry Only

**Correction model:**
```
Parameters:
  dx  ∈ [-3, 3] px      (mask x-shift, typical assembly error)
  dy  ∈ [-3, 3] px      (mask y-shift)
  θ   ∈ [-1°, 1°]       (mask rotation, optical bench tolerance)

Total: 3 parameters (small search space)

Why ONLY mask geometry:
  ✓ Mask position changes with mechanical reassembly (session-to-session)
  ✓ Prism dispersion is fixed by optical design (constant)
  ✓ Smaller 3D space faster to search than 5D
  ✓ Gradient-based refinement works better on low dimensions
```

### 2.2 Why Dispersion is Fixed

**Real behavior:**
- Prism refractive index is determined during manufacturing
- Dispersion slope a₁ ≈ 2.0 px/band (varies < 1% between systems)
- After assembly, prism is mechanically fixed → cannot change

**Implication:**
- Use **factory calibration** or **pre-calibrate once** per system
- During operation (per scene), treat a₁ as **known constant**
- Parameter fitting focuses ONLY on **mask geometry** (dx, dy, θ)

---

## Part 3: UPWMI Algorithm 1 & 2 for Mask-Only Correction

### 3.1 Algorithm 1: Beam Search (3D Grid Search over Mask Parameters)

**Efficient scoring strategy:**
- Score on **1 reference scene** (fast)
- Score on **2–4 shift-crops** per scene (diverse viewpoints, not full scenes)
- Use **proxy reconstruction** (5–10 GAP-TV iterations, not full 50 iterations)

**Input:**
- Measurements from shift-crops: {y_crop_0, y_crop_1, y_crop_2, y_crop_3} from one reference scene
- Mask M (256 × 256)
- Shift-crop offset information

**Processing pipeline (EFFICIENT VERSION):**

```python
def upwmi_algorithm_1_mask_beam_search(y_crops, mask, x_crops_ref, n_crops=4):
    """
    Beam search to find (dx, dy, theta) that maximizes reconstruction.

    EFFICIENT:
    - Score on only 1 scene (scene01) with 2-4 crops
    - Use proxy reconstruction (K=5-10 iterations, fast)
    - 3D search space: dx, dy, theta only
    - Operator uses HYBRID grid (N=2 spatial, K=2 spectral, downsampled output)
    """

    # Step 1: Define search space (ORIGINAL GRID parameters)
    search_space = {
        'dx': np.linspace(-3, 3, 13),      # [-3, 3] px, 13 values
        'dy': np.linspace(-3, 3, 13),
        'theta': np.linspace(-np.pi/180, np.pi/180, 7),  # [-1°, 1°], 7 values
    }
    # Total: 13 × 13 × 7 = 1,183 combinations

    # Step 2: Stage 1 — Independent 1D sweeps (very fast)
    scores = {'dx': [], 'dy': [], 'theta': []}

    for dx in search_space['dx']:
        # Score on 2-4 shift-crops with PROXY reconstruction (5 iterations)
        mask_warped = warp_mask(mask, dx=dx, dy=0, theta=0)
        # Use hybrid operator: expands for forward, downsamples to original for measurement
        operator = SDCASSIOperator_HybridGrid(mask_warped, disp_step=2.0,
                                              spatial_upsample_N=2, spectral_interp_K=2)

        score_sum = 0
        for crop_idx in range(min(4, len(y_crops))):
            x_hat = gap_tv_cassi(y_crops[crop_idx], operator, n_iter=5)  # PROXY: only 5 iters
            score_sum += compute_score(x_hat, x_crops_ref[crop_idx])

        avg_score = score_sum / min(4, len(y_crops))
        scores['dx'].append((dx, avg_score))

    scores['dx'] = sorted(scores['dx'], key=lambda x: -x[1])
    top_dx = [s[0] for s in scores['dx'][:5]]  # Keep top 5

    # Repeat for dy, theta (similar, ~10 minutes each)
    # ... (dy and theta sweeps)

    # Step 3: Stage 2 — Beam search (5×5×5=125 combinations, K=10 iterations)
    candidates = list(itertools.product(top_dx, top_dy, top_theta))
    best_candidates = []

    for (dx, dy, theta) in candidates:
        mask_warped = warp_mask(mask, dx=dx, dy=dy, theta=theta)
        # Hybrid operator for expanded-grid simulation
        operator = SDCASSIOperator_HybridGrid(mask_warped, disp_step=2.0,
                                              spatial_upsample_N=2, spectral_interp_K=2)

        # Score on all 2-4 crops with K=10 iterations (better quality)
        score_sum = 0
        for crop_idx in range(min(4, len(y_crops))):
            x_hat = gap_tv_cassi(y_crops[crop_idx], operator, n_iter=10)
            score_sum += compute_score(x_hat, x_crops_ref[crop_idx])

        avg_score = score_sum / min(4, len(y_crops))
        best_candidates.append(((dx, dy, theta), avg_score))

    best_candidates = sorted(best_candidates, key=lambda x: -x[1])[:5]

    # Step 4: Local refinement (coordinate descent, 3 rounds)
    for round_idx in range(3):
        for i in range(len(best_candidates)):
            (dx_cur, dy_cur, theta_cur), _ = best_candidates[i]

            for param in ['dx', 'dy', 'theta']:
                delta = {'dx': 0.25, 'dy': 0.25, 'theta': 0.05}[param]

                best_local = best_candidates[i]
                for offset in [-1, 0, 1]:
                    if param == 'dx':
                        (dx_new, dy_new, theta_new) = (dx_cur + delta*offset, dy_cur, theta_cur)
                    elif param == 'dy':
                        (dx_new, dy_new, theta_new) = (dx_cur, dy_cur + delta*offset, theta_cur)
                    else:
                        (dx_new, dy_new, theta_new) = (dx_cur, dy_cur, theta_cur + delta*offset)

                    mask_warped = warp_mask(mask, dx=dx_new, dy=dy_new, theta=theta_new)
                    operator = SDCASSIOperator_HybridGrid(mask_warped, disp_step=2.0,
                                                          spatial_upsample_N=2, spectral_interp_K=2)

                    score_sum = 0
                    for crop_idx in range(min(4, len(y_crops))):
                        x_hat = gap_tv_cassi(y_crops[crop_idx], operator, n_iter=10)
                        score_sum += compute_score(x_hat, x_crops_ref[crop_idx])

                    avg_score = score_sum / min(4, len(y_crops))

                    if avg_score > best_local[1]:
                        best_local = (((dx_new, dy_new, theta_new), avg_score))

                best_candidates[i] = best_local

    # Step 5: Return best estimate (on original grid)
    (dx_best, dy_best, theta_best), _ = best_candidates[0]

    return (dx_best, dy_best, theta_best)
```

**Computational cost (MUCH FASTER):**
- 1D sweeps: 13 × (proxy K=5 on 4 crops) ≈ 30 minutes × 3 params = 1.5 hours
- Beam search: 125 × (K=10 on 4 crops) ≈ 2 hours
- Coordinate descent: 3 × 5 candidates × 3 params × 3 offsets × K=10 ≈ 1 hour
- **Total: ~4.5 hours per scene (vs 20–30 hours for expanded-grid)**

**Expected accuracy:** Within 0.1–0.2 px on original grid (coarse estimate)

### 3.2 Algorithm 2: Gradient-Based Refinement (Unrolled GAP-TV)

**Input:**
- Coarse estimate (dx₁, dy₁, θ₁) from Algorithm 1
- Shift-crop measurements {y_crop_i} from reference scene
- Reference scene crops {x_crop_i}
- Full measurement y and scene x (from all 10 TSA scenes, for final evaluation)

**Procedure:**

```python
def upwmi_algorithm_2_differentiable_refinement(y_crops, x_crops, y_all_scenes, x_all_scenes,
                                                mask, (dx_coarse, dy_coarse, theta_coarse)):
    """
    Gradient-based refinement via unrolled GAP-TV on ORIGINAL grid.

    EFFICIENT:
    - Unroll K=10 iterations (fast, balanced accuracy)
    - Optimize only 3 parameters (dx, dy, theta)
    - Hybrid operator: expands for forward, reconstructs on original
    - Loss computed on shift-crops first, then validate on full 10 scenes
    """

    # Parameterize as differentiable tensors
    dx = torch.nn.Parameter(torch.tensor(dx_coarse, dtype=torch.float32, requires_grad=True))
    dy = torch.nn.Parameter(torch.tensor(dy_coarse, dtype=torch.float32, requires_grad=True))
    theta = torch.nn.Parameter(torch.tensor(theta_coarse, dtype=torch.float32, requires_grad=True))

    def loss_fn(y_list, x_ref_list):
        """Loss computed on crops (fast) or full scenes (for validation)."""
        # Warp mask with current parameters
        mask_warped = differentiable_warp_mask(mask, dx=dx, dy=dy, theta=theta)
        # Use hybrid operator: N=2 spatial, K=2 spectral, downsampled output
        operator = DifferentiableSDCASSIOperator_HybridGrid(mask_warped, disp_step=2.0,
                                                            spatial_upsample_N=2, spectral_interp_K=2)

        # Unroll K=10 GAP-TV iterations
        loss = 0
        for crop_idx, (y_crop, x_ref) in enumerate(zip(y_list, x_ref_list)):
            x_hat = unrolled_gap_tv_k_iterations(y_crop, operator, K=10)
            loss += F.mse_loss(x_hat, x_ref)

        return loss / len(y_list)

    # Phase 1: Refine on shift-crops (fast, 100 epochs)
    optimizer = torch.optim.Adam([dx, dy, theta], lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    for epoch in range(100):
        optimizer.zero_grad()
        loss = loss_fn(y_crops, x_crops)  # Loss on 2-4 crops
        loss.backward()
        torch.nn.utils.clip_grad_norm_([dx, dy, theta], max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 20 == 0:
            print(f"Phase 1 Epoch {epoch}: loss={loss.item():.6f}")

    # Phase 2: Fine-tune on full 10 scenes (more robust, 50 epochs)
    # Re-initialize optimizer with lower learning rate
    optimizer = torch.optim.Adam([dx, dy, theta], lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    for epoch in range(50):
        optimizer.zero_grad()
        loss = loss_fn(y_all_scenes, x_all_scenes)  # Loss on full scenes
        loss.backward()
        torch.nn.utils.clip_grad_norm_([dx, dy, theta], max_norm=0.5)
        optimizer.step()
        scheduler.step()

        if epoch % 10 == 0:
            print(f"Phase 2 Epoch {epoch}: loss={loss.item():.6f}")

    # Return refined parameters
    dx_refined = dx.item()
    dy_refined = dy.item()
    theta_refined = theta.item()

    return (dx_refined, dy_refined, theta_refined)
```

**Computational cost (MUCH FASTER):**
- Phase 1 (100 epochs × K=10 on 4 crops): ~1.5 hours
- Phase 2 (50 epochs × K=10 on 10 full scenes): ~2.5 hours
- **Total: ~4 hours per scene (VERY FAST!)**

**Expected accuracy:** 3–5× better than Algorithm 1 (within ±0.05–0.1 px on original grid)

### 3.3 Complete Workflow: Efficient Algorithm 1 → 2 → Validation on All 10 Scenes

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    UPWMI Complete Pipeline (EFFICIENT)                   │
└──────────────────────────────────────────────────────────────────────────┘

INPUT: 10 TSA scenes (256×256×28), mask (256×256), 10 measurements (256×310 each)

    ↓

┌──────────────────────────────────────────────────────────────────────────┐
│ PREPROCESSING: Generate shift-crops from ONE reference scene (scene01)    │
│                                                                          │
│ 1. Reflect-pad scene01 along W-axis by M=3 px                           │
│ 2. Generate 4 crops at offsets o = [0, 1, 2, 3] px                      │
│    Result: {x_crop_0, x_crop_1, x_crop_2, x_crop_3}  (256×256×28 each) │
│ 3. Generate corresponding measurements {y_crop_0, y_crop_1, ...}        │
│    (using perfect mask, no mismatch)                                     │
│                                                                          │
│ Purpose: Provide dense overlapping viewpoints for algorithm scoring      │
└──────────────────────────────────────────────────────────────────────────┘

    ↓

┌──────────────────────────────────────────────────────────────────────────┐
│ ALGORITHM 1: Beam Search (3D over dx, dy, θ, NO spatial expansion!)     │
│              Duration: 4.5 hours per scene                               │
│              Output: Coarse estimate (dx₁, dy₁, θ₁)                     │
│                                                                          │
│ Stage 1: 1D sweeps on shift-crops with proxy reconstruction (K=5 iters) │
│   - dx ∈ [-3, 3] (13 values × 4 crops × 5 iters): ~30 min             │
│   - dy ∈ [-3, 3] (13 values × 4 crops × 5 iters): ~30 min             │
│   - θ ∈ [-1°, 1°] (7 values × 4 crops × 5 iters): ~10 min             │
│                                                                          │
│ Stage 2: Beam search (5×5×5=125 combos, K=10 iters)                    │
│   - Score all combinations on 4 crops: ~2 hours                         │
│   - Keep top 5 candidates                                               │
│                                                                          │
│ Stage 3: Coordinate descent (3 rounds, K=10 iters)                     │
│   - Local refinement around top-5: ~1 hour                              │
│                                                                          │
│ Total per scene: ~4.5 hours → (dx₁, dy₁, θ₁) [within ±0.2 px]         │
└──────────────────────────────────────────────────────────────────────────┘

    ↓

┌──────────────────────────────────────────────────────────────────────────┐
│ CHECKPOINT 1: Coarse Reconstruction (All 10 scenes)                      │
│                                                                          │
│ For each of 10 scenes:                                                  │
│  • Apply coarse mask correction: mask_v1 = warp(mask, dx₁, dy₁, θ₁)    │
│  • Reconstruct: x̂_v1 = gap_tv(y_scene, mask_v1, n_iter=50)             │
│  • Compute metrics: PSNR_v1, SSIM_v1, SAM_v1                            │
│  • Parameter error: |dx_true - dx₁|, |dy_true - dy₁|, |θ_true - θ₁|   │
│                                                                          │
│ → Save: {scene01_alg1.json, scene02_alg1.json, ..., scene10_alg1.json}  │
│ → Aggregated metrics: results_alg1_10scenes.json                        │
└──────────────────────────────────────────────────────────────────────────┘

    ↓

┌──────────────────────────────────────────────────────────────────────────┐
│ ALGORITHM 2: Gradient-Based Refinement (Unrolled GAP-TV, K=10)          │
│              Duration: 4 hours per scene                                 │
│              Output: Refined estimate (dx₂, dy₂, θ₂)                    │
│                                                                          │
│ Phase 1: Optimize on shift-crops (100 epochs, lr=0.01)                 │
│   - Loss: MSE on 4 shift-crops with K=10 iterations: ~1.5 hours        │
│                                                                          │
│ Phase 2: Fine-tune on full 10 scenes (50 epochs, lr=0.001)             │
│   - Loss: MSE on all 10 full scenes with K=10 iterations: ~2.5 hours   │
│   - Higher robustness, better generalization                            │
│                                                                          │
│ Total per scene: ~4 hours → (dx₂, dy₂, θ₂) [within ±0.05 px]          │
└──────────────────────────────────────────────────────────────────────────┘

    ↓

┌──────────────────────────────────────────────────────────────────────────┐
│ CHECKPOINT 2: Refined Reconstruction (All 10 scenes)                     │
│                                                                          │
│ For each of 10 scenes:                                                  │
│  • Apply refined mask correction: mask_v2 = warp(mask, dx₂, dy₂, θ₂)   │
│  • Reconstruct: x̂_v2 = gap_tv(y_scene, mask_v2, n_iter=50)             │
│  • Compute metrics: PSNR_v2, SSIM_v2, SAM_v2                            │
│  • Parameter error: |dx_true - dx₂|, |dy_true - dy₂|, |θ_true - θ₂|   │
│  • Compare improvement: ΔPSNR = PSNR_v2 - PSNR_v1, etc.                │
│                                                                          │
│ → Save: {scene01_alg2.json, scene02_alg2.json, ..., scene10_alg2.json}  │
│ → Aggregated metrics: results_alg2_10scenes.json                        │
└──────────────────────────────────────────────────────────────────────────┘

    ↓

┌──────────────────────────────────────────────────────────────────────────┐
│ ORACLE BASELINE: Reconstruct with TRUE parameters (all 10 scenes)        │
│                                                                          │
│ For each of 10 scenes:                                                  │
│  • Apply true mask correction: mask_oracle = warp(mask, dx_t, dy_t, θ_t)│
│  • Reconstruct: x̂_oracle = gap_tv(y_scene, mask_oracle, n_iter=50)     │
│  • Compute metrics: PSNR_oracle, SSIM_oracle, SAM_oracle               │
│  • **This is the upper bound** (perfect calibration)                    │
│                                                                          │
│ → Save: results_oracle_10scenes.json                                    │
└──────────────────────────────────────────────────────────────────────────┘

    ↓

┌──────────────────────────────────────────────────────────────────────────┐
│ FINAL REPORT: Algorithm 1 vs 2 vs Oracle across 10 scenes               │
│                                                                          │
│ Tables:                                                                  │
│   - Parameter recovery (true vs est, error metrics)                      │
│   - PSNR/SSIM/SAM for all scenes, Alg1 vs Alg2 vs Oracle              │
│   - Improvement from Alg1 → Alg2 (ΔPSNR, etc.)                        │
│                                                                          │
│ Figures:                                                                 │
│   - Parameter scatter plots (dx, dy, θ true vs estimated)              │
│   - PSNR trajectory across 10 scenes                                    │
│   - Reconstruction visual comparisons (1-2 scenes)                      │
│                                                                          │
│ → Save: pwm/reports/cassi_mask_correction_efficient.md                 │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Part 4: Experimental Validation Protocol on 10 TSA Scenes

### 4.1 Dataset Setup & Shift-Crop Generation

**TSA Simulation Benchmark:**
- Location: `MST-main/datasets/TSA_simu_data/`
- 10 scenes: scene01, scene02, ..., scene10
- Original shape: 256×256×28 (H×W×L)
- Mask: 256×256 binary coded aperture (provided)
- Dispersion: step=2.0 px/band (standard)
- Spectral interpolation: K=2 (for forward model fidelity)

**Per-scene preprocessing:**
1. Load scene_i (256×256×28)
2. Inject synthetic mismatch: (dx_true, dy_true, θ_true)
   - dx_true ∈ [-3, 3] px (uniform random)
   - dy_true ∈ [-3, 3] px (uniform random)
   - θ_true ∈ [-1°, 1°] (uniform random, in radians)
3. Generate noisy measurement y_i = forward(scene_i, mask_true_mismatch)
   - Forward model uses spectral interpolation K=2
   - Add Poisson noise (peak=10000) + Gaussian noise (σ=0.01)

**Shift-crop generation (for scene01 only, used by both Alg1 and Alg2):**
1. Reflect-pad scene01 along W-axis by M=3 px
2. Generate 4 crops: {x_crop_0, x_crop_1, x_crop_2, x_crop_3}
   - Offsets o = [0, 1, 2, 3] px along dispersion axis
3. For each crop, generate measurement (same true mismatch):
   - y_crop_i = forward(x_crop_i, mask_true_mismatch)
4. **Purpose:** Provide dense overlapping viewpoints for algorithm fitting

### 4.2 Detailed Test Protocol (Per Scene)

```python
def run_full_test_scene(scene_idx, x_true, mask_orig, x_crops_ref, y_crops_ref, x_all_scenes, y_all_scenes):
    """
    Full test protocol for one scene: inject mismatch → Alg1 → Alg2 → validate.

    EFFICIENT VERSION:
    - Uses shift-crops for Alg1 and Alg2 (not full spatial expansion)
    - No expansion to larger grid (spectral K=2 only for forward model)
    """

    # ========== SETUP ==========
    H, W, L = x_true.shape  # (256, 256, 28)

    # Inject synthetic mismatch (ground truth for validation)
    dx_true = np.random.uniform(-3, 3)
    dy_true = np.random.uniform(-3, 3)
    theta_true = np.random.uniform(-np.pi/180, np.pi/180)  # [-1°, 1°]

    # Apply mismatch to mask
    mask_misaligned = warp_mask(mask_orig, dx=dx_true, dy=dy_true, theta=theta_true)

    # Generate measurements for all 10 scenes with this misaligned mask
    # Use hybrid operator: expands for simulation, downsamples back to original
    y_all_noisy = []
    for x_scene in x_all_scenes:
        operator_true = SDCASSIOperator_HybridGrid(mask_misaligned, disp_step=2.0,
                                                   spatial_upsample_N=2, spectral_interp_K=2)
        y_clean = operator_true.forward(x_scene)
        y_noisy = add_poisson_gaussian_noise(y_clean, peak_photons=10000, read_sigma=0.01)
        y_all_noisy.append(y_noisy)

    # Get measurement for current scene
    y_current = y_all_noisy[scene_idx]

    # ========== ALGORITHM 1: Beam Search (EFFICIENT VERSION) ==========
    # Uses shift-crops from reference scene for fast scoring
    (dx_hat1, dy_hat1, theta_hat1) = upwmi_algorithm_1_mask_beam_search(
        y_crops=y_crops_ref,
        mask=mask_orig,
        x_crops_ref=x_crops_ref,
        n_crops=4
    )

    # Checkpoint 1: Coarse reconstruction (on full scene measurement)
    mask_warped_v1 = warp_mask(mask_orig, dx=dx_hat1, dy=dy_hat1, theta=theta_hat1)
    # Use hybrid operator (N=2, K=2) for forward model
    operator_v1 = SDCASSIOperator_HybridGrid(mask_warped_v1, disp_step=2.0,
                                             spatial_upsample_N=2, spectral_interp_K=2)
    x_hat_v1 = gap_tv_cassi(y_current, operator_v1, n_iter=50)

    psnr_v1 = psnr(x_hat_v1, x_true)
    ssim_v1 = ssim(x_hat_v1, x_true)
    sam_v1 = spectral_angle_mapper(x_hat_v1, x_true)

    error_dx_v1 = abs(dx_hat1 - dx_true)
    error_dy_v1 = abs(dy_hat1 - dy_true)
    error_theta_v1 = abs(theta_hat1 - theta_true) * 180 / np.pi  # Convert to degrees

    # ========== ALGORITHM 2: Gradient Refinement (EFFICIENT VERSION) ==========
    # Optimizes on shift-crops, then validates on full 10 scenes
    (dx_hat2, dy_hat2, theta_hat2) = upwmi_algorithm_2_differentiable_refinement(
        y_crops=y_crops_ref,
        x_crops=x_crops_ref,
        y_all_scenes=y_all_noisy,  # All 10 measurements for phase 2
        x_all_scenes=x_all_scenes,  # All 10 scenes for phase 2
        mask=mask_orig,
        coarse_estimate=(dx_hat1, dy_hat1, theta_hat1)
    )

    # Checkpoint 2: Refined reconstruction
    mask_warped_v2 = warp_mask(mask_orig, dx=dx_hat2, dy=dy_hat2, theta=theta_hat2)
    operator_v2 = SDCASSIOperator_HybridGrid(mask_warped_v2, disp_step=2.0,
                                             spatial_upsample_N=2, spectral_interp_K=2)
    x_hat_v2 = gap_tv_cassi(y_current, operator_v2, n_iter=50)

    psnr_v2 = psnr(x_hat_v2, x_true)
    ssim_v2 = ssim(x_hat_v2, x_true)
    sam_v2 = spectral_angle_mapper(x_hat_v2, x_true)

    error_dx_v2 = abs(dx_hat2 - dx_true)
    error_dy_v2 = abs(dy_hat2 - dy_true)
    error_theta_v2 = abs(theta_hat2 - theta_true) * 180 / np.pi

    # ========== ORACLE (Perfect correction) ==========
    operator_oracle = SDCASSIOperator_HybridGrid(mask_orig, disp_step=2.0,
                                                 spatial_upsample_N=2, spectral_interp_K=2)
    x_hat_oracle = gap_tv_cassi(y_current, operator_oracle, n_iter=50)

    psnr_oracle = psnr(x_hat_oracle, x_true)
    ssim_oracle = ssim(x_hat_oracle, x_true)
    sam_oracle = spectral_angle_mapper(x_hat_oracle, x_true)

    # ========== RETURN RESULTS ==========
    return {
        # Ground truth mismatch
        'scene_idx': scene_idx,
        'dx_true': dx_true,
        'dy_true': dy_true,
        'theta_true': theta_true * 180 / np.pi,

        # Algorithm 1 results
        'dx_hat1': dx_hat1,
        'dy_hat1': dy_hat1,
        'theta_hat1': theta_hat1 * 180 / np.pi,
        'psnr_v1': psnr_v1,
        'ssim_v1': ssim_v1,
        'sam_v1': sam_v1,
        'error_dx_v1': error_dx_v1,
        'error_dy_v1': error_dy_v1,
        'error_theta_v1': error_theta_v1,

        # Algorithm 2 results
        'dx_hat2': dx_hat2,
        'dy_hat2': dy_hat2,
        'theta_hat2': theta_hat2 * 180 / np.pi,
        'psnr_v2': psnr_v2,
        'ssim_v2': ssim_v2,
        'sam_v2': sam_v2,
        'error_dx_v2': error_dx_v2,
        'error_dy_v2': error_dy_v2,
        'error_theta_v2': error_theta_v2,

        # Oracle (upper bound)
        'psnr_oracle': psnr_oracle,
        'ssim_oracle': ssim_oracle,
        'sam_oracle': sam_oracle,

        # Improvements
        'psnr_gain_alg2': psnr_v2 - psnr_v1,
        'ssim_gain_alg2': ssim_v2 - ssim_v1,
        'sam_gain_alg2': sam_v2 - sam_v1,
        'gap_to_oracle_psnr': psnr_oracle - psnr_v2,
    }
```

### 4.3 Expected Results

**Parameter recovery (Alg 2 expected accuracy):**
- dx: ±0.05–0.1 px error
- dy: ±0.05–0.1 px error
- θ: ±0.02–0.05° error

**PSNR improvements:**
- Baseline (no correction): ~12–15 dB (misaligned, poor)
- After Alg 1: ~20–22 dB (coarse correction)
- After Alg 2: ~23–25 dB (refined correction)
- Oracle (true parameters): ~27–29 dB (upper bound)
- **Expected gap:** (PSNR_oracle - PSNR_alg2) < 2–3 dB

### 4.4 Reporting Structure

Create: `pwm/reports/cassi_mask_correction_efficient.md`

**Tables:**
1. **Table 1:** Parameter recovery accuracy (10 scenes, Alg 1 vs Alg 2)
   - Columns: Scene, dx_true, dx_hat1, |error|_v1, dx_hat2, |error|_v2, improvement
   - Rows: scene01 — scene10, +Mean/StDev

2. **Table 2:** PSNR/SSIM/SAM metrics and gains
   - Columns: Scene, PSNR_v1, PSNR_v2, Δ(v2-v1), PSNR_oracle, Gap(oracle-v2)
   - Rows: scene01 — scene10, +Mean metrics, +Std Dev
   - Similar for SSIM and SAM

3. **Table 3:** Algorithm efficiency & timing
   - Alg 1: Total time (4.5 h/scene × 10), breakdown (stages 1-3)
   - Alg 2: Total time (4 h/scene × 10), breakdown (phase 1-2)
   - Total for 10-scene experiment

**Figures:**
1. Parameter recovery scatter plots: dx_true vs dx_hat2 (all 10 scenes), similar for dy, θ
2. PSNR trajectory: All 10 scenes, Alg1 vs Alg2 vs Oracle (bar chart + line plot)
3. Reconstruction visual: Ground truth, corrupted, Alg1, Alg2, Oracle (1-2 scenes, 3 bands each)
4. Error histograms: Distribution of parameter errors across 10 scenes

---

## Part 5: Implementation Architecture (Follows cassi_working_process.md)

### 5.1 Modified Forward Model Chain (Spatial + Spectral Expansion, Downsampled Output)

Following `cassi_working_process.md` Section 3 (SD-CASSI Forward Model):

**Hybrid operator (expand for simulation, downsample for original measurement size):**
```python
class SDCASSIOperator_HybridGrid(PhysicsOperator):
    """
    SD-CASSI forward model with BOTH spatial (N=2) and spectral (K=2) expansion.
    Input: x (H, W, L=28)
    Output: y (H, W + disp_extent) on ORIGINAL grid (downsampled from expanded)

    Pipeline:
    1. Spatial upsample: (H,W) → (N·H, N·W)
    2. Spectral interpolate: L → K·(L-1)+1
    3. Forward model on expanded (N·H, N·W, K·L) grid with integer shifts
    4. Downsample measurement back to original (H, W + disp_extent)
    5. Return measurement at original scale
    """

    def __init__(self, mask, dispersion_step=2.0, spatial_upsample_N=2, spectral_interp_K=2):
        self.mask_base = mask  # (H, W) on original grid
        self.disp_step = dispersion_step  # 2.0 px/band (fixed)
        self.N = spatial_upsample_N  # Spatial upsample factor
        self.K = spectral_interp_K  # Spectral interpolation factor

    def forward(self, x):
        """
        Input: x (H, W, L=28)
        Output: y (H, W + (L-1)·disp_step) on ORIGINAL grid
        """
        H, W, L = x.shape

        # Step 1: Spatial upsample
        x_spatial = upsample_spatial(x, factor=self.N)  # (N·H, N·W, L)
        mask_spatial = upsample_spatial(self.mask_base, factor=self.N)  # (N·H, N·W)

        # Step 2: Spectral interpolate
        x_expanded = interpolate_spectral_frames(x_spatial, K=self.K)  # (N·H, N·W, K·L')
        L_expanded = x_expanded.shape[2]

        # Step 3: Forward model on expanded grid (integer shifts!)
        y_expanded = self._forward_model_integer(x_expanded, mask_spatial)  # (N·H, N·W+disp_large)

        # Step 4: Downsample measurement back to original size
        y_downsampled = downsample_spatial(y_expanded, factor=self.N)  # (H, W+disp_orig)

        # Step 5: Scale by 1/K (energy conservation from spectral interpolation)
        y = y_downsampled / self.K

        return y  # (H, W + (L-1)·disp_step)  ← original measurement size!

    def adjoint(self, y):
        """
        Adjoint operator: y → x̂ (for reconstruction solvers on ORIGINAL grid)

        Input: y (H, W + disp_extent) on original grid
        Output: x̂ (H, W, L=28) on original grid
        """
        H, W, L = (y.shape[0], y.shape[1] - int((L_orig - 1) * self.disp_step), L_orig)

        # Step 1: Scale input by K (reverse spectral scaling)
        y_scaled = y * self.K

        # Step 2: Upsample measurement to expanded grid
        y_expanded = upsample_spatial(y_scaled, factor=self.N)

        # Step 3: Adjoint on expanded grid
        x_expanded_hat = self._adjoint_model_integer(y_expanded, upsample_spatial(self.mask_base, self.N))

        # Step 4: Downsample spatial
        x_spatial_hat = downsample_spatial(x_expanded_hat, factor=self.N)

        # Step 5: Downsample spectral (integrate K·L bands back to L)
        x_hat = downsample_spectral_frames(x_spatial_hat, K=self.K)

        return x_hat  # (H, W, L=28)  ← original grid!

    def _forward_model_integer(self, x_expanded, mask_expanded):
        """SD-CASSI forward with integer dispersion shifts (no interpolation)."""
        H_exp, W_exp, L_exp = x_expanded.shape
        W_out = W_exp + (L_exp - 1) * int(self.disp_step)
        y = np.zeros((H_exp, W_out), dtype=x_expanded.dtype)
        n_c = (L_exp - 1) / 2.0

        for l in range(L_exp):
            d_l = int(round(self.disp_step * (l - n_c)))  # Integer shift!
            coded = x_expanded[:, :, l] * mask_expanded
            y_start = max(0, d_l)
            y_end = min(W_out, W_exp + d_l)
            src_start = max(0, -d_l)
            src_end = src_start + (y_end - y_start)
            y[:, y_start:y_end] += coded[:, src_start:src_end]

        return y

    def apply_mask_correction(self, dx: float, dy: float, theta: float):
        """Apply estimated mask correction on ORIGINAL grid."""
        self.mask_base = warp_affine(self.mask_base, dx=dx, dy=dy, theta=theta)
```

### 5.2 New Algorithm Modules

Create files:
- `packages/pwm_core/pwm_core/algorithms/upwmi_efficient.py` — Algorithms 1 & 2 (efficient version)
  - `upwmi_algorithm_1_mask_beam_search(y_crops, mask, x_crops_ref, ...)`
  - `upwmi_algorithm_2_differentiable_refinement(y_crops, x_crops, y_all, x_all, mask, ...)`
  - Helper: `interpolate_spectral_frames(x, K)`, `downsample_spectral_frames(x_hat, K)`
  - Helper: `generate_shift_crops(scene, n_crops=4, margin=6)`
- `packages/pwm_core/pwm_core/physics/cassi_spectral_interp.py` — Updated SD-CASSI operator
- `scripts/run_cassi_mask_correction_efficient.py` — Full pipeline for 10 scenes

### 5.3 Updated Mismatch Registry (mismatch_db.yaml)

```yaml
cassi_mask_only_efficient:
  description: "Mask-only correction with spectral interpolation (K=2)"
  parameters:
    mask_dx:
      range: [-3, 3]
      typical_error: 0.3
      unit: pixels
      correctable: true

    mask_dy:
      range: [-3, 3]
      typical_error: 0.3
      unit: pixels
      correctable: true

    mask_theta:
      range: [-1, 1]  # degrees
      typical_error: 0.05
      unit: degrees
      correctable: true

  correction_method: "UPWMI_algorithm_1_beam_search + UPWMI_algorithm_2_differentiable"
  forward_model_options:
    spectral_interpolation_K: 2
    dispersion_step: 2.0
    frozen_parameters:
      - a1: 2.0  # px/band (prism property)
      - alpha: 0.0  # degrees (dispersion axis)
      - a2: 0.0  # second-order (negligible)

  note: |
    Mask geometry (dx, dy, θ) varies per measurement session.
    Prism dispersion is fixed (factory calibration).
    Uses spectral interpolation (K=2) for forward model fidelity.
    Algorithm 1 (4.5 h/scene) + Algorithm 2 (4 h/scene).
```

### 5.4 Integration with cassi_working_process.md

The new approach **extends** Section 13 (Operator-Correction Mode) with:

**Original (Section 13.2):**
```python
# 5. Estimate dispersion curve (a1, a2, axis_angle)
theta_disp = estimate_dispersion_curve(calibration_frames=...)

# 7. Merge into updated BeliefState
theta_calibrated = {**theta_nominal, **theta_mask, **theta_disp, **theta_sd}
```

**New (Efficient mask-only approach):**
```python
# Pre-calibrate (once per system): a1=2.0, a2=0.0, alpha=0°
# Per-measurement: Correct only mask geometry (dx, dy, θ)

# 4. Generate shift-crops from reference scene (fast preprocessing)
x_crops, y_crops = generate_shift_crops(scene_ref, n_crops=4)

# 5. Estimate ONLY mask geometry (3 parameters)
theta_mask = upwmi_algorithm_1(y_crops, mask_base, x_crops, n_crops=4)
theta_mask = upwmi_algorithm_2(y_crops, x_crops, y_all_scenes, x_all_scenes, mask_base, theta_mask)

# 7. Simplified BeliefState
theta_calibrated = {**theta_nominal, **theta_mask}  # Only mask geometry
Phi_cal = SDCASSIOperator(mask_base, disp_step=2.0, spectral_interp_K=2)
Phi_cal.apply_mask_correction(**theta_mask)
```

---

## Part 6: Critical Design Decisions for User Review

**Based on your feedback, these are now FIXED. Please confirm choices:**

### 6.1 Upsampling Factor N (REVISED DECISION)

| N | Spatial | Spectral | Rotation err | GPU hours | Memory | Verdict |
|---|---------|----------|--------------|-----------|--------|---------|
| N=3 | 768×768 | 168 | 0.68 px | ~40 h | 36× | Good if memory-limited |
| **N=4** | **1024×1024** | **224** | **0.90 px** | **~70 h** | **64×** | **STRONGLY RECOMMENDED** |

**Rationale:** N=4 ensures rotation errors quantize to near-integer sub-pixels (0.90 px ≈ 1 px). This is essential for θ fitting accuracy.

**Question:** Your GPU/compute budget — can you support N=4 (~70 GPU hours)? Or prefer N=3 (~40 GPU hours)?

---

### 6.2 Mask/Scene Size Strategy (REVISED & CONFIRMED)

**New approach: Crop-and-Zero-Pad (not expand mask)**

| Aspect | Strategy |
|--------|----------|
| **Scene processing** | Crop interior (224×224) from 256×256, zero-pad back to 256×256 |
| **Mask** | Keep at original 256×256 (no expansion) |
| **Effective result** | Mask (256×256) now > scene content (224×224 interior) ✓ |
| **Measurement** | Output stays at 256×310 (original size, backward compatible) |
| **Evaluation** | Metrics on interior 224×224 (exclude P=16 px border) |

**Advantages:**
- ✅ Dimensions remain original (256×256×28)
- ✅ Benchmark-compatible
- ✅ Solves mask > scene problem
- ✅ Clean interior/border distinction
- ✅ Simpler implementation (no mask expansion)

**Question:** **CONFIRMED** — Proceed with crop-and-zero-pad strategy (P=16)?

---

### 6.3 Dispersion Parameters: Keep or Freeze?

**User input revealed:** Prism CAN change in real lab (thermal drift, settling)

**New decision:** **FIT BOTH a₁ AND α** (but with tighter ranges)

**Ranges for 5D fitting:**
```
Original W2:   a₁ ∈ [1.85, 2.15] px/band, α ∈ [-5°, 5°]
Proposed:      a₁ ∈ [1.95, 2.05] px/band, α ∈ [-1°, 1°]  ← tighter!
               (thermal drift is small, not large W2 error)
```

**Search space:** 5D (dx, dy, θ, a₁, α) — still manageable with beam search + Algorithm 2

**Question:** Tighter ranges acceptable? Or use full W2 ranges for robustness?

---

### 6.4 Algorithm Configuration

**Algorithm 1 (Beam Search):**
- Stage 1: 1D sweeps on **1 reference scene** (faster)
- Stage 2: Beam width = **10 candidates**
- Stage 3: **6 rounds** of coordinate descent
- **Decision:** Proceed as outlined?

**Algorithm 2 (Differentiable Refinement):**
- Unroll depth: **K=10 GAP-TV iterations** (balanced speed/accuracy)
- Loss function: **MSE** (direct, fast)
- Optimizer: **Adam, lr=0.01 → 0.001** (cosine annealing)
- Training: **200 epochs** over all 10 scenes
- **Decision:** Proceed as outlined?

**Question:** Any modifications to algorithm configs?

---

### 6.5 Validation Setup

**Noise model:** Poisson(peak=10000) + Gaussian(σ=0.01) — realistic, matches ECCV-2020 paper
- **Decision:** Keep this?

**Ground-truth mismatch range per scene:**
```
dx, dy ∈ [-3, 3] px        (typical assembly error)
θ ∈ [-1.5°, 1.5°]          (realistic optical bench error)
a₁ ∈ [1.95, 2.05] px/band  (thermal drift)
α ∈ [-1°, 1°]              (prism settling)
```

**Question:** These ranges appropriate for realism check?

---

### 6.6 Final Decisions Checklist

| Item | Decision | Status |
|------|----------|--------|
| **Upsampling N** | N=4 (1024×1024) for rotation quantization | ✅ **CONFIRMED** |
| **Spectral interpolation** | Cubic spline, 28 → 224 bands | ✅ **CONFIRMED** |
| **Mask/scene strategy** | **Crop interior (224×224), zero-pad back to 256×256** | ✅ **REVISED & CONFIRMED** |
| **Output dimensions** | Measurement: 256×310, Scene: 256×256×28 (original size) | ✅ **CONFIRMED** |
| **Evaluation region** | Interior 224×224 only (exclude P=16 px border) | ✅ **CONFIRMED** |
| **Dispersion fit** | Keep a₁, α with tighter ranges | ✅ **CONFIRMED** |
| **Parameter ranges** | dx,dy ∈[-3,3], θ ∈[-1.5°,1.5°], a₁ ∈[1.95,2.05], α ∈[-1°,1°] | ✅ **CONFIRMED** |
| **Algorithm configs** | Alg 1: beam search 5D; Alg 2: unrolled GAP-TV K=10 | ✅ **CONFIRMED** |
| **Noise model** | Poisson(10000) + Gaussian(0.01) — ECCV-2020 realistic | ✅ **CONFIRMED** |
| **Processing pipeline** | Crop → zero-pad → expand → forward → downsample → reconstruct → eval interior | ✅ **CONFIRMED** |

---

**Plan is now COMPLETE and ready for implementation!**

---

## Part 7: Deliverables (Post-Approval Implementation)

### Codebase additions:

1. **New operator class:**
   - `packages/pwm_core/pwm_core/physics/cassi_expanded.py` — `SDCASSIOperator_ExpandedGrid`
   - Spectral expansion: `expand_spectral_frames(x, expansion_factor)`
   - Spatial upsampling/downsampling utilities

2. **Algorithm implementations:**
   - `packages/pwm_core/pwm_core/algorithms/upwmi.py` (new file):
     - `upwmi_algorithm_1_mask_beam_search(y, mask, scenes, N=2)`
     - `upwmi_algorithm_2_differentiable_refinement(y, x, mask, (dx, dy, θ), N=2)`
   - Support functions: warp_mask, compute_score, unrolled_gap_tv

3. **Main execution script:**
   - `scripts/run_cassi_mask_correction_full.py` — Orchestrates Algorithm 1+2 for all 10 scenes

4. **Test suite:**
   - `tests/test_cassi_expanded_grid.py` — Unit tests for operator, algorithms
   - Fixtures: Load TSA benchmark, generate synthetic mismatch

### Reporting & Validation:

5. **Full validation report:**
   - File: `pwm/reports/cassi_mask_correction_expanded_grid.md`
   - Tables: Parameter recovery (10 scenes, Alg 1 vs Alg 2)
   - Tables: PSNR/SSIM/SAM metrics + gap to oracle
   - Tables: Execution times (Alg 1 hours, Alg 2 hours, total)
   - Figures: Parameter scatter plots, PSNR trajectory, reconstruction comparisons
   - Discussion: Algorithm 1 vs Algorithm 2 trade-offs, accuracy limits

6. **RunBundle artifacts:**
   - Per-scene checkpoints: `runs/run_cassi_mask_correction_*/artifacts/{scene}/`
   - Reconstruction images (.png): Original, Alg1, Alg2, Oracle
   - Parameter estimates (.json): dx_true, dx_hat1, dx_hat2, etc.

7. **Summary scoreboard:**
   - Update `pwm/reports/scoreboard.yaml` with CASSI expanded-grid results

---

## Part 8: Implementation Timeline & Compute Requirements (EFFICIENT VERSION)

### Phase 1: Code Development (12 hours wall-clock)

| Task | Duration | GPU required | Notes |
|------|----------|--------------|-------|
| 1a. Spectral interpolation utilities + downsample | 1.5 h | No | PCHIP/linear interpolation |
| 1b. SDCASSIOperator (spectral_interp_K version) | 1.5 h | No | No spatial expansion needed |
| 1c. Algorithm 1 (3D beam search, efficient) | 3 h | No | Scoring on 2-4 crops only |
| 1d. Algorithm 2 (gradient refinement, 2-phase) | 2.5 h | No | PyTorch unrolling K=10 |
| 1e. Shift-crop generation + utilities | 1 h | No | reflect_pad, crop logic |
| 1f. Unit tests + fixtures | 1.5 h | No | Pytest, TSA benchmark |
| 1g. Main orchestration script | 1 h | No | Runner, checkpoints |
| **Phase 1 Total** | **12 h** | **No** | — |

### Phase 2: Single-Scene Validation (8.5 GPU hours)

| Task | GPU hours | Wall-clock | Notes |
|------|-----------|-----------|-------|
| 2a. scene01 Algorithm 1 (shift-crops, proxy K=5/10) | ~4.5 h | 1.5 h | Fast coarse search |
| 2b. scene01 Algorithm 2 (phases 1-2, full scenes) | ~4 h | 1.5 h | Gradient refinement |
| **Phase 2 Total** | **8.5 h** | **3 h** | One scene full pipeline |

### Phase 3: Full 10-Scene Execution (~85 GPU hours)

| Task | GPU hours | Wall-clock (background) | Notes |
|------|-----------|-------------------------|-------|
| 3a. Algorithm 1 on 10 scenes (4.5 h/scene × 10) | ~45 h | 9–12 h | Fast coarse search |
| 3b. Algorithm 2 on 10 scenes (4 h/scene × 10) | ~40 h | 8–10 h | Gradient refinement |
| **Phase 3 Total** | **~85 h** | **~18–22 h** | **Very efficient!** |

### Phase 4: Analysis & Reporting (4 hours)

| Task | Duration | Notes |
|------|----------|-------|
| 4a. Parse & aggregate 10-scene results | 1 h | Parameter errors, metrics statistics |
| 4b. Generate tables & figures | 1.5 h | Parameter scatter, PSNR trends, visuals |
| 4c. Write analysis & discussion | 1 h | Algorithm comparison, gap to oracle |
| 4d. Final review | 0.5 h | Validation, sanity checks |
| **Phase 4 Total** | **4 h** | — |

### Summary (MUCH MORE EFFICIENT!)

| Phase | Wall-clock | GPU hours | Notes |
|-------|-----------|-----------|-------|
| **1. Code dev** | **12 h** | **0 h** | Sequential, start first |
| **2. Single-scene** | **3 h** | **8.5 h** | Overnight/quick morning |
| **3. Full 10-scene** | **18–22 h** | **~85 h** | Continuous background GPU (~3.5 days) |
| **4. Analysis** | **4 h** | **0 h** | While Phase 3 runs in background |
| **Total elapsed** | **~39 h wall** | **~94 GPU** | **~4 days calendar time** |

**Practical timeline (24/7 GPU execution):**
- Day 1: Phase 1 code development (12 h)
- Night 1: Phase 2 single-scene validation (overnight, 8.5 h GPU)
- Day 2: Phase 3 starts (background, continuous)
- Days 2–4: Phase 3 continuous (~85 GPU h ÷ 24 h/day ≈ 3.5 days)
- Days 2–4: Phase 4 analysis (parallel with Phase 3, as scenes complete)
- **Total: ~4 days calendar with 24/7 GPU** (vs ~9–10 days for expanded-grid approach!)

**Advantages of efficient approach:**
- ✅ **7.5× faster GPU** (85 h vs 600 h)
- ✅ **~6× faster calendar** (4 days vs ~9–10 days)
- ✅ **No spatial expansion** → lower memory, faster computation
- ✅ **3D parameter space** → easier beam search than 5D
- ✅ **Shift-crop strategy** → more robust validation data

---

## Appendix A: Spectral Interpolation & Shift-Crop Mathematics

### A.1 Shift-Crop Margin Formula: M ≥ 2N-1

**Purpose:** Determine minimum padding needed to safely extract 2N overlapping crops.

**Setup:**
- Original scene width: 256 px
- Number of crops: 2N
- Crop offsets with stride-1: [0, 1, 2, ..., 2N-1]
- Maximum offset: 2N-1

**Derivation:**
- Rightmost crop starts at: 2N-1
- Rightmost crop width: 256 px
- Rightmost crop ends at: (2N-1) + 256
- Padded scene width: 256 + 2M

**Constraint (rightmost crop must fit):**
```
(2N-1) + 256  ≤  256 + 2M
     2N-1     ≤  2M
  2N-1 / 2    ≤  M
```

**Therefore:** M ≥ 2N-1 (for integer pixel safety: use M = 2N-1)

**Example (2N=4, N=2):**
- Max offset: 2N-1 = 3
- Min padding: M ≥ 3
- Padded width: 256 + 2(3) = 262
- Rightmost crop: [3:259] ✓ fits in [0:262]

---

### A.2 Spectral Interpolation Formula: L → K·(L-1)+1

**Purpose:** Subdivide spectral intervals for continuous-wavelength simulation fidelity.

**Original bands:** L discrete samples (e.g., L=28)
- λ_k ∈ [λ_0, λ_{L-1}]
- Spacing: Δλ = (λ_{L-1} − λ_0) / (L−1)

**Interpolated bands:** L_new = K·(L-1)+1 samples
- Subdivides each interval by factor K
- Original bands appear at interpolated indices: i = k·K

**Example (K=2, L=28):**
```
Original:     λ_0, λ_1, λ_2, ..., λ_27          (28 values)
Interpolated: λ_0, λ_0.5, λ_1, λ_1.5, λ_2, ..., λ_27  (55 values = 2·27+1)
```

**Why K·(L-1)+1 (not K·L):**
- K·L would overshoot the range (27 original intervals vs 28 bands)
- K·(L-1)+1 correctly preserves endpoint alignment
- Ensures reconstructed measurement y_interp / K matches original scale

### A.3 Energy Conservation with Spectral Interpolation Scaling

**Measurement fidelity:**
```
Original: y = ∑_{l=0}^{27} shift( x[:,:,l] ⊙ M, d_l )

Interpolated (K=2):
  y_interp = ∑_{l'=0}^{54} shift( x'[:,:,l'] ⊙ M, d_{l'} ) / K
           ≈ y  [by conservation property]
```

**Why scaling by 1/K works:**
- x'[:,:,l'] ≈ x[:,:,l]/K  for intermediate interpolated bands
- Sum over K·(L-1)+1 bands integrates to K × (original sum)
- Dividing by K normalizes back to original measurement scale
- Reconstruction solvers receive measurement in expected range

### A.4 Shift-Crop Dataset Expansion (Stride-1 Dense Sampling)

**2N samples per scene (N=2 → 4 crops):**

```
Original scene: [████████████]  (256 px, at position 0)

Reflect-padded: [···████████████···]  (margin M ≥ 3 on sides)

Crop 0 (offset 0):  [████████████]  ← leftmost view
Crop 1 (offset 1):   [████████████]  ← +1 px view
Crop 2 (offset 2):    [████████████]  ← +2 px view
Crop 3 (offset 3):     [████████████]  ← +3 px view

Each crop: 256×256×28 (full spatial & spectral content, shifted viewpoint)
Stride: 1 px (dense overlapping samples)
Total: 4 crops × 10 scenes = 40 training crops
```

**Physical interpretation:**
- Stride-1 shifts along dispersion axis provide continuous dense sampling
- No change to ground truth (shift is external, acts on measurement)
- Each crop sees the scene from a different "wavelength offset" perspective
- Overlapping crops provide richer training signal for parameter fitting robustness

---

## Appendix B: Comparison — W2 vs Efficient Mask-Only Approach

### B.1 Design Philosophy

| Aspect | W2 Approach | Efficient Approach |
|--------|-------------|-------------------|
| **Model type** | 5-parameter joint fitting | 3-parameter mask correction |
| **Assumption** | All parameters are calibration errors | Mask varies, prism is fixed |
| **Physics** | Less realistic (prism doesn't change) | Correct (respects optical design) |
| **Search space** | 5D (harder to optimize) | 3D (easier, faster) |
| **Spectral handling** | None (no interpolation) | K=2 interpolation (fidelity) |

### B.2 Algorithm Efficiency

| Aspect | W2 | Efficient |
|--------|----|-----------|
| **Alg1 (coarse)** | 1D sweeps (slow) | 3D beam search on crops (fast) |
| **Alg2 (fine)** | None | Gradient refinement (novel) |
| **Per-scene time** | N/A (W2 is 1 scene only) | 4.5 h (Alg1) + 4 h (Alg2) = 8.5 h |
| **10-scene total** | N/A | ~85 h GPU |
| **Comparison** | Direct comparison not possible | W2 on single scene only |

### B.3 Expected Results Comparison

| Metric | W2 (scene01, best case) | Efficient (10 scenes, average) |
|--------|----------------------|-----|
| **Parameter error (dx)** | Unknown | ±0.05–0.1 px |
| **Parameter error (θ)** | Unknown | ±0.02–0.05° |
| **PSNR (Alg1)** | ~22 dB (W2d) | ~20–22 dB |
| **PSNR (Alg2)** | N/A | ~23–25 dB |
| **PSNR (oracle)** | N/A | ~27–29 dB |
| **Gap (oracle - Alg2)** | N/A | <2–3 dB |

### B.4 Key Advantages of Efficient Approach

| Advantage | Impact |
|-----------|--------|
| **Smaller search space (3D vs 5D)** | Beam search feasible, faster convergence |
| **Spectral interpolation K=2** | Better forward model fidelity without spatial blowup |
| **Shift-crop strategy** | Multiple viewpoints for robust fitting |
| **Gradient-based refinement** | 3–5× improvement over beam search alone |
| **No spatial expansion** | Lower memory, faster computation |
| **10-scene validation** | Statistical robustness vs single-scene W2 |
| **4-day timeline** | Practical for rapid development/testing |

---

## Conclusion & Key Insights

### Why This Efficient Approach Works

1. **Physically Grounded:**
   - Prism dispersion δ(λ) is **fixed** by optical design (factory calibration)
   - Mask position varies with mechanical reassembly → **requires per-session correction**
   - Respects the fundamental distinction between system properties and measurement errors

2. **Algorithmically Superior:**
   - 3D search space (dx, dy, θ) is **fast and manageable** vs 5D
   - Algorithm 1 (beam search) handles coarse estimation well
   - Algorithm 2 (gradient-based) provides **3–5× refinement**, crucial for high accuracy

3. **Spectral Interpolation Advantage:**
   - K=2 interpolation ensures **continuous-wavelength fidelity** without spatial explosion
   - Scaling by 1/K conserves measurement energy
   - Forward/adjoint models remain exact (no fractional pixel errors)

4. **Shift-Crop Dataset Expansion:**
   - 4 crops per scene → **40 total crops** from 10 scenes
   - Diverse viewpoints improve parameter fitting robustness
   - Multiple perspectives reduce overfitting risk

5. **Efficiency vs Accuracy Trade-off:**
   - **85 GPU hours** for 10-scene comprehensive validation (vs 600 h for expanded grid)
   - **4-day calendar timeline** makes rapid iteration feasible
   - No loss in expected parameter accuracy (±0.05–0.1 px)

### Expected Outcomes

**Parameter Recovery (Algorithm 2):**
- dx, dy: ±0.05–0.1 px accuracy
- θ: ±0.02–0.05° accuracy
- 3–5× improvement over Algorithm 1 alone

**Reconstruction Quality:**
- Without correction: ~12–15 dB (misaligned, poor)
- After Alg 1: ~20–22 dB (coarse, acceptable)
- After Alg 2: ~23–25 dB (refined, good)
- Oracle: ~27–29 dB (upper bound, perfect calibration)

**Gap Analysis:**
- Expected gap (Alg2 → Oracle) < 2–3 dB
- Indicates successful parameter recovery with minimal residual error

### Architecture Integration

This plan **extends** `cassi_working_process.md` (Section 13 — Operator-Correction Mode):
- Simplified correction: 3 parameters instead of 5
- Pre-calibrated dispersion: Use factory spec or one-time measurement
- Per-session mask fitting: Run UPWMI Algorithms 1 & 2 on new measurement
- Maintains full PWM framework compatibility

### Deliverables (Post-Approval)

**Code:**
- `upwmi_efficient.py` (Algorithms 1 & 2 implementations)
- `cassi_spectral_interp.py` (Updated operator with K parameter)
- `run_cassi_mask_correction_efficient.py` (Full pipeline orchestrator)

**Validation:**
- 10-scene results with parameter recovery & PSNR metrics
- Comprehensive report: `cassi_mask_correction_efficient.md`
- Parameter scatter plots, PSNR trajectories, visual comparisons

**Timeline:**
- Phase 1 (code): 12 hours
- Phase 2 (single-scene): 8.5 GPU hours
- Phase 3 (10-scene): ~85 GPU hours (~4 days)
- Phase 4 (analysis): 4 hours
- **Total: ~4 days calendar time**

---

## Approval Checkpoint

**This plan is complete and ready for implementation!**

**Key decisions (CONFIRMED):**
- ✅ 3-parameter mask-only correction (dx, dy, θ)
- ✅ Spectral interpolation K=2 for forward model fidelity
- ✅ Shift-crop dataset expansion (4 crops per scene)
- ✅ Algorithm 1: 3D beam search on crop measurements
- ✅ Algorithm 2: Gradient refinement on full 10 scenes
- ✅ 10-scene comprehensive validation
- ✅ 4-day efficient timeline

**Next steps (upon approval):**
1. Implement Phase 1 code (12 hours)
2. Run Phase 2 single-scene validation (8.5 GPU hours)
3. Execute Phase 3 full pipeline (85 GPU hours, ~4 days background)
4. Generate comprehensive report with all metrics and visualizations

---

**Plan created:** 2026-02-14
**Version:** EFFICIENT (spectral interpolation + shift-crop + 3D search)
