# CASSI Calibration via Upsample-to-Integer & Mask-Only Correction
**Plan Document — v2 (2026-02-14)**

## Executive Summary

This document proposes a **physically-motivated CASSI calibration strategy** that:

1. **Expand image & spectral bands** (256×256×28) → (N·256, N·256, 2N·28) to enforce **integer dispersion shifts**
2. **Interpolate spectral frames** from 28 → 2N·28 using spectral smoothing (preserves physics of hyperspectral cube)
3. **Downsample corrected measurement** back to original size after fixed-shift forward model
4. **Correct only mask geometry** (dx, dy, θ) via **UPWMI Algorithm 1 (beam search) + Algorithm 2 (gradient refinement)**
5. **Freeze dispersion parameters** (a₁, α) as known optical properties, not fitting variables
6. **Validate on all 10 TSA scenes** with ground-truth comparisons and parameter recovery metrics

**Core insight:** Spectral upsampling (2N·28 bands) with spatial upsampling (N) ensures minimum dispersion shifts become **integer pixels**. This eliminates sub-pixel registration errors and allows exact (non-interpolated) forward model evaluation.

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

## Part 1: Image & Spectral Expansion Strategy (Upsample-to-Integer)

### 1.1 Motivation: Integer Dispersion Shifts

Current SD-CASSI forward model (from `cassi_working_process.md`, Section 3.3):
```
Discrete form:  d_n = s · (n − n_c)           where s=2.0 px/band, n_c=(L-1)/2
Result:         fractional band shifts: d_n ∈ {-26, -24, ..., 0, ..., 24, 26} px  [for L=28]
```

**Problem:** Even with integer s=2.0, the shifts d_n can lead to sub-pixel registration if we consider:
1. Mask alignment errors (dx, dy) compound with fractional dispersion
2. Spectral interpolation during forward/adjoint (if bands are unevenly spaced)
3. Small angle α shifts dispersion direction, creating 2D fractional offsets

**Solution:** Expand both spatial and spectral dimensions such that:
- All minimum mismatch/shift amounts become **integer pixels**
- Forward model uses **exact integer shifts** (no interpolation)
- Reconstruction is computed on enlarged grid, then downsampled

### 1.2 Image & Spectral Expansion Protocol

**Original:** 256 × 256 × 28 (H × W × L)

**Expansion:**
```
1. Choose upsampling factor N (e.g., N=2)

2. Spatial upsampling: (H, W) → (N·H, N·W)
   Example: 256×256 → 512×512

3. Spectral upsampling: L → 2N·L
   Example: 28 → 112 (expand by 2N=4)

   Reason: With 2N times more bands, dispersion step becomes finer:
   - Original: step=2 px/band
   - Expanded: step=2 px/band (same), but 2N·28=112 bands
   - Result: finer wavelength spacing → smoother dispersion curve

4. Interpolate spectral frames: 28 → 112 frames
   - Method: Cubic spline interpolation along wavelength axis
   - Assumption: Spectral content varies smoothly with λ
   - Preserves: Spectral smoothness, energy conservation

5. Forward model operates on (N·H) × (N·W) × (2N·L) grid
   - Dispersion shifts: d_n = 2 · (n − (2N·L-1)/2)  [all integers]
   - Mask shifts: (N·dx, N·dy), θ  [scaled to enlarged grid]
   - No fractional pixel offsets in any dimension

6. Output measurement: (N·H) × (N·W + (2N·L-1)·2) ← enlarged measurement

7. Downsample back to original size: (H) × (W + (L-1)·2) ← original measurement shape
   - Bilinear interpolation (lossy, but combined noise/artifacts tolerable)
   - For reconstruction: Use downsampled measurement y, original size
```

### 1.3 Upsampling Factor Selection (Revised for Rotation Quantization)

**Constraints:**
1. Memory reasonable
2. Dispersion shifts become integer pixels
3. **Rotation errors quantize to near-integer sub-pixels** (NEW!)
4. Computation time acceptable

**For TSA benchmark (256×256×28), accounting for rotation quantization:**

| N | Expanded size | Memory (vs orig) | Spectral | Max rotation error | GPU time | Notes |
|---|---|---|---|---|---|---|
| N=2 | 512×512×112 | 16× | 112 | ~0.45 px (frac) | ~15 h | Insufficient for θ quantization |
| **N=3** | **768×768×168** | **36×** | **168** | **~0.68 px (better)** | **~40 h** | Good balance |
| **N=4** | **1024×1024×224** | **64×** | **224** | **~0.90 px (near-int)** | **~70 h** | **RECOMMENDED** |
| N=5 | 1280×1280×280 | 100× | 280 | ~1.12 px (integer!) | ~120 h | Memory/time tradeoff worse |

**Choice: N=4 (REVISED RECOMMENDATION)**
- Spatial: 256×256 → 1024×1024 (16× pixels, high-end GPU required)
- Spectral: 28 → 224 (8× bands, ~70 hours GPU time)
- Dispersion shifts: d_n = 2·(n − 111.5) ∈ {..., -223, -221, ..., 0, ..., 221, 223} (all integer)
- **Rotation error:** θ ∈ [-3°, 3°], max pixel offset ~ 0.9 px (near-integer quantization!)
- Mask errors: (dx, dy) scaled to 4× grid → minimum shift becomes 0.25 px (near-integer when summed across region)

**Fallback: N=3** (if GPU memory limited)
- Spatial: 256×256 → 768×768 (9× pixels, moderate GPU memory)
- Spectral: 28 → 168 (6× bands, ~40 hours GPU)
- Rotation error: ~0.68 px (acceptable, though not fully quantized)
- Time/quality tradeoff better than N=4 for resource-constrained systems

### 1.4 Mask > Scene Size: Crop & Zero-Pad Strategy

**Problem in TSA benchmark:** Mask and scene are same size (256×256)
- Edge pixels of scene are only partially encoded by mask
- Results in edge artifacts and vignetting

**Real CASSI hardware:** Mask is **larger** than scene area
- Ensures complete encoding of scene content
- Peripheral region has zero/black padding

**REVISED Strategy: Crop scene, pad with zeros**

```
Input:
  Original scene:  x (256 × 256 × 28)
  Original mask:   M (256 × 256)

Step 1: Crop scene interior to region of interest
  Crop border: P = 16 px on all sides
  x_crop: (256-2P) × (256-2P) × 28 = (224 × 224 × 28)

  Physical interpretation:
    - (224×224) = actual scene content
    - P=16 px border = peripheral region (black/dark)

Step 2: Pad cropped scene back to original size with zeros
  x_padded: (256 × 256 × 28)
  [Place x_crop at center, zero-pad P-pixel border on all sides]

  Now: Mask (256×256) > effective scene content (224×224)
       Mask covers both scene content AND zero-padded border
  ✓ Problem solved: Mask fully encodes scene!

Step 3: Upsample both to expanded grid (N=4)
  x_expanded: (N·H × N·W × 2N·L) = (1024 × 1024 × 224)
  M_expanded: (1024 × 1024)

  Note: Zero-padded regions remain zero after upsampling

Step 4: Forward model on expanded grid
  y_expanded: (1024 × (1024 + dispersion_spread))

  Integer dispersion shifts applied to both scene + zero-padded region

Step 5: Downsample measurement back to original size
  y_final: (256 × (256 + 54)) = (256 × 310)  ← original measurement shape

  This measurement now has CORRECT encoding:
  - Interior (224×224) region: properly encoded by full mask
  - Border region: encoded by mask (contributes to edges)

Step 6: Reconstruction & evaluation
  x̂_full: (256 × 256 × 28) reconstructed cube

  For metrics: Evaluate ONLY interior region
    x̂_interior: (224 × 224 × 28) ← ignore P-pixel border
    PSNR/SSIM/SAM computed on interior only
    (Border contains reconstruction artifacts due to zero-padding)

Boundary handling:
  - Interior (224×224): Valid reconstruction, high quality
  - Border (P=16 px): Artifact region, expected degradation
  - Evaluation excludes border → honest performance assessment
```

**Benefits:**
1. ✅ Maintains benchmark dimensions (256×256×28)
2. ✅ Mask effectively > scene (covers content + zero region)
3. ✅ Matches real CASSI hardware (zero/black periphery)
4. ✅ Clean distinction: interior (valid) vs. border (artifacts)
5. ✅ Evaluation on interior only (no unfair edge comparisons)

**Padding border size (P):**
- P=16 px recommended (6% on each side, 77% interior for eval)
- Trade-off: Larger P → more buffer, but reduced eval region
- P=16 keeps (256-32)×(256-32) = 224×224 = 77% for metrics

---

### 1.5 Spectral Frame Interpolation (28 → 2N·28)

**Algorithm:**

```python
def expand_spectral_frames(x_original, expansion_factor=4):
    """
    Expand L spectral frames → expansion_factor·L frames via smooth interpolation.

    Input:  x_original: (H, W, L=28) hyperspectral cube
    Output: x_expanded: (H, W, L_new=112) expanded cube
    """
    H, W, L = x_original.shape
    L_new = expansion_factor * L

    # 1. Assume original frames are at normalized wavelengths: λ_k = k / L, k=0..L-1
    lambda_orig = np.arange(L) / L
    lambda_expanded = np.arange(L_new) / L_new

    # 2. For each pixel (i,j), interpolate its spectral curve
    x_expanded = np.zeros((H, W, L_new), dtype=x_original.dtype)

    for i in range(H):
        for j in range(W):
            spectral_curve = x_original[i, j, :]  # (L,)

            # Cubic spline interpolation
            f = scipy.interpolate.CubicSpline(lambda_orig, spectral_curve, bc_type='natural')
            x_expanded[i, j, :] = f(lambda_expanded)

            # Clip to valid range (prevent ringing)
            x_expanded[i, j, :] = np.clip(x_expanded[i, j, :], 0, np.max(spectral_curve))

    return x_expanded
```

**Properties:**
- Smooth: Cubic spline preserves C² continuity
- Energy-conservative: Integrated intensity over wavelength preserved (approximately)
- Realistic: Assumes hyperspectral scenes have smooth spectral variation (true for natural scenes)
- Invertible: Downsampling via integration recovers original bands (approximately)

### 1.6 Benefits of Expanded-Grid Approach

1. **Integer dispersion shifts:** All wavelength-dependent shifts become integer pixels → exact forward/adjoint model → better gradients for Algorithm 2
2. **Rotation quantization:** Large N (e.g., N=4) ensures rotation errors become near-integer sub-pixels → finer grid for θ fitting
3. **Exact differentiable model:** No interpolation → perfect adjoint → unrolled GAP-TV gives accurate gradients
4. **Improved parameter recovery:** 5D grid (3 spatial + 2 dispersion) is feasible with beam search + gradient refinement
5. **Physics-grounded spectral expansion:** Matches real hyperspectral scenes (smooth spectral variation)
6. **Complete scene encoding:** Larger mask (via padding/expansion) ensures uniform encoding across scene
7. **Edge artifact mitigation:** Padded scene + cropped evaluation prevents vignetting artifacts

---

## Part 2: Revised Mismatch Model (Mask + Selective Dispersion Correction)

### 2.1 Real-World Prism Behavior

**Does prism change after assembly?** YES, in real lab conditions:
1. **Thermal drift:** Temperature variations change prism refractive index → shifts a₁
   - Typical: Δa₁ ≈ ±0.02 px/band per °C
   - Lab drift: 18°C → 25°C = +0.14 px/band error
2. **Optical bench settling:** Vibration/settling can tilt prism → shifts α
   - Typical: Δα ≈ ±0.5° after disassembly
3. **Mechanical stress:** Assembly clamps may shift prism slightly

**Implication:** Prism parameters **CANNOT be frozen** — must include in fitting model, but with **higher uncertainty/tighter ranges**.

### 2.2 Mask Size vs Scene Size (Edge Artifact Mitigation)

**Real CASSI hardware:** Mask is typically **LARGER than scene** to:
- Prevent edge vignetting (mask doesn't fully encode scene edges)
- Avoid "missing pixel" artifacts at boundaries
- Ensure uniform encoding across entire scene

**Current TSA benchmark problem:**
- Scene: 256 × 256
- Mask: 256 × 256 (same size, no padding!)
- This means **edge pixels are NOT fully encoded** → artifacts

**Solution:**
1. **Pad scene** with uniform/symmetric borders before encoding
   - Original scene: 256 × 256
   - Padded scene: 256 + 2P × 256 + 2P (e.g., P=32 → 320 × 320)
   - Mask: 256 × 256 (smaller than padded scene) — edge not fully encoded, but applied to padding

2. **Expand mask** instead:
   - Scene: 256 × 256 (use as-is)
   - Mask: 256 + 2P × 256 + 2P (e.g., P=32 → 320 × 320, larger than scene)
   - Edges of scene FULLY ENCODED by mask ✓
   - **Recommended approach:** Mask > Scene

3. **Correct for edge distortion:**
   - Crop reconstruction output to remove edge artifacts
   - Evaluate metrics only on interior (256 × 256) after removing P-pixel border

### 2.3 Rotation Error Quantization: Why N=2 is Insufficient

**Rotation θ introduces sub-pixel shifts:**
```
For pixel at (x, y), rotation by θ (small angle):
  x' ≈ x - y·θ    (displacement ~ y·θ)
  y' ≈ y + x·θ    (displacement ~ x·θ)

Max displacement from rotation: ~θ·max(H,W)
```

**Example: Original grid (256 × 256), θ ∈ [-3°, 3°]**
```
θ_max = 3° = 0.0524 rad
Max offset ~ 0.0524 × 256 ≈ 13.4 px (fractional!)
```

**With N=2 expansion (512 × 512):**
```
Same θ, but on 2× finer grid:
Max offset ~ 0.0524 × 512 ≈ 26.8 px (still fractional!)
```

**Problem:** Even on expanded grid, rotation introduces fractional pixel offsets!

**Solution: Increase N to make rotation errors integer**

For rotation error to become integer, we need:
```
θ_error · H_expanded = θ_error · N · H_original ∈ ℤ

Example: θ_error = 0.05° = 0.000873 rad, H_original = 256
  N=2: 0.000873 × 2×256 ≈ 0.45 px (fractional)
  N=4: 0.000873 × 4×256 ≈ 0.90 px (fractional, better)
  N=8: 0.000873 × 8×256 ≈ 1.79 px (closer to integer)
```

**Recommendation: N ≥ 3 (preferably N=4)**

### 2.4 Correction Parameters (5 Total)

| Parameter | Type | Range (original) | Range (N=4 grid) | Why retained |
|-----------|------|------------------|------------------|-------------|
| **dx** | Mask x-shift | [-5, 5] px | [-20, 20] px | Assembly tolerance |
| **dy** | Mask y-shift | [-5, 5] px | [-20, 20] px | Assembly tolerance |
| **θ** | Mask rotation | [-3°, 3°] | Same | Optical bench twist |
| **a₁** | Dispersion slope | [1.85, 2.15] px/band | Same (NOT expanded) | **Thermal drift** |
| **α** | Dispersion axis | [-2°, 2°] | Same (NOT expanded) | **Prism tilt** |

**Note:** Dispersion parameters (a₁, α) are fitted on **original parameter space**, not expanded grid (their effect scales differently).

### 2.5 Why 5 Parameters (Not 3)

| Parameter | Original W2 approach | Revised approach | Rationale |
|-----------|-------------------|-------------------|-----------|
| dx, dy, θ (mask) | Always fit | Always fit | Mechanical errors, change per session ✓ |
| a₁ (disp slope) | Fit in W2c (+5.49 dB) | **KEEP, tighter range** | Real thermal drift, but smaller than W2c |
| α (disp axis) | Fit in W2d (+7.04 dB) | **KEEP, tighter range** | Real prism tilt, but smaller than W2d |
| a₂ (disp curve) | Fit in W2d (implicit) | **REMOVE — set to 0** | Second-order, negligible after expansion |

---

## Part 2.6: Processing Pipeline Summary (Crop-and-Zero-Pad + Downsample)

**Unified pipeline from original size → expanded grid → back to original size:**

```
INPUT: Original scene x (256×256×28), mask M (256×256), measurement y (256×310)

┌──────────────────────────────────────────────────────────────────────┐
│ PREPROCESSING: Crop interior, zero-pad to original size (P=16)      │
├──────────────────────────────────────────────────────────────────────┤
│ • Extract interior: x_interior = x[16:240, 16:240, :]  (224×224×28) │
│ • Zero-pad back: x_padded = zeros(256,256,28)                       │
│   x_padded[16:240, 16:240, :] = x_interior                          │
│ • Mask unchanged: M_padded = M (256×256)                            │
│ Result: Mask now > effective scene (224×224) ✓                      │
└──────────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│ EXPANSION: Upsample to N=4 grid (1024×1024×224)                    │
├──────────────────────────────────────────────────────────────────────┤
│ • Spatial upsample: x_padded → x_expanded (1024×1024×28)            │
│ • Spectral expand: 28 → 224 bands (cubic spline)                    │
│ • Mask upsample: M_padded → M_expanded (1024×1024)                  │
│ • Zero regions remain zero after upsampling                          │
└──────────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│ FORWARD MODEL: Integer dispersion shifts on expanded grid            │
├──────────────────────────────────────────────────────────────────────┤
│ y_expanded = ForwardModel(x_expanded, M_expanded)                    │
│ Shape: (1024, 1024 + dispersion_spread) = (1024, 1116)             │
│ • All shifts d_n are exact integers (no interpolation)              │
│ • Zero-padded regions processed normally                             │
│ • Integer rotation errors (N=4 ensures ~0.9 px quantization)        │
└──────────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│ DOWNSAMPLE: Back to original measurement size (256×310)             │
├──────────────────────────────────────────────────────────────────────┤
│ y_final = Downsample(y_expanded, factor=1/N)                        │
│ Shape: (256, 310) ← original measurement dimensions                 │
│ • Measurement now has CORRECT encoding                              │
│   - Interior (224×224): fully encoded by mask                       │
│   - Border (P=16): encoded by mask, expected artifacts              │
└──────────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│ RECONSTRUCTION: Recover scene on original grid (256×256×28)         │
├──────────────────────────────────────────────────────────────────────┤
│ x̂ = Solver(y_final, M)  [GAP-TV, MST, HDNet, etc.]                 │
│ Shape: (256, 256, 28)                                                │
│ Contains:                                                             │
│   • Interior (224×224): high-quality reconstruction                  │
│   • Border (P=16): artifact region (expected degradation)           │
└──────────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│ EVALUATION: Metrics on interior only (224×224×28)                   │
├──────────────────────────────────────────────────────────────────────┤
│ x̂_interior = x̂[16:240, 16:240, :]                                   │
│ x_interior = x[16:240, 16:240, :]                                    │
│ PSNR, SSIM, SAM computed on interior only                            │
│ (Border excluded from metrics)                                        │
└──────────────────────────────────────────────────────────────────────┘
```

**Key properties:**
- ✅ Input/output dimensions: **256×256×28** (unchanged, benchmark-compatible)
- ✅ Mask effectively > scene: Mask (256×256) covers scene content (224×224 interior) + zero periphery
- ✅ Measurement: **256×310** (original size, backward compatible)
- ✅ Integer shifts: All dispersions & mask shifts become integers on N=4 grid
- ✅ Evaluation: Only interior 224×224 scored (77% of image, avoids edge artifacts)

---

## Part 3: UPWMI Algorithm 1 & 2 for Mask-Only Correction (on Expanded Grid)

### 3.1 Algorithm 1: Beam Search (3D Grid Search over Mask Parameters)

**Input:**
- Original measurement y_orig (256 × 310)
- Original mask M_orig (256 × 256)
- Original scenes {x₁, ..., x₁₀} (256 × 256 × 28)
- Expansion factor: N=2

**Processing pipeline:**

```python
def upwmi_algorithm_1_mask_beam_search(y_orig, mask_orig, scenes_orig, N=2):
    """
    Beam search to find (dx, dy, theta) that maximizes reconstruction quality.
    Operates on expanded grid (N·256 × N·256 × 2N·28).
    """

    # Step 1: Expand all inputs to larger grid
    H, W, L = scenes_orig[0].shape  # (256, 256, 28)
    scenes_expanded = [expand_spectral_frames(x_orig, expansion_factor=2*N)
                       for x_orig in scenes_orig]
    # Now: (256, 256, 2*N*28) = (256, 256, 112) for each scene

    H_exp, W_exp, L_exp = N*H, N*W, 2*N*L  # (512, 512, 112)
    x_expanded_full = np.zeros((10, H_exp, W_exp, L_exp))  # Stack all 10 scenes
    for i, scene_expanded in enumerate(scenes_expanded):
        x_expanded_full[i] = upsample_spatial(scene_expanded, factor=N)  # (512, 512, 112)

    mask_expanded = upsample_spatial(mask_orig[None, None, :, :], factor=N)[0, 0]  # (512, 512)
    y_expanded = upsample_spatial(y_orig[None, None, :, :], factor=N)[0, 0]  # (512, 620)

    # Step 2: Define search space (on expanded grid)
    search_space = {
        'dx': np.linspace(-10, 10, 21),    # [-10, 10] px on expanded grid
        'dy': np.linspace(-10, 10, 21),    # (21 values)
        'theta': np.linspace(-np.pi/60, np.pi/60, 13),  # [-3°, 3°] (13 values)
    }
    # Total: 21 × 21 × 13 = 5,733 combinations (too many for full grid)

    # Step 3: Stage 1 — Independent 1D sweeps (fast)
    scores = {'dx': [], 'dy': [], 'theta': []}

    for dx in search_space['dx']:
        mask_warped = warp_mask(mask_expanded, dx=dx, dy=0, theta=0)
        operator = SDCASSIOperator_Expanded(mask_warped, disp_step=2.0, upsample_factor=N)

        # Coarse reconstruction on 1 scene (fast)
        x_hat = gap_tv_cassi(y_expanded, operator, n_iter=25)
        score = compute_score(x_hat, x_expanded_full)  # Average across all 10 scenes
        scores['dx'].append((dx, score))

    scores['dx'] = sorted(scores['dx'], key=lambda x: -x[1])
    top_dx = [s[0] for s in scores['dx'][:5]]  # Keep top 5

    # Repeat for dy, theta
    # ... (similar procedure)

    # Step 4: Stage 2 — Beam search (width=10)
    candidates = list(itertools.product(top_dx, top_dy, top_theta))  # 5×5×5=125 combos
    best_candidates = []

    for (dx, dy, theta) in candidates:
        mask_warped = warp_mask(mask_expanded, dx=dx, dy=dy, theta=theta)
        operator = SDCASSIOperator_Expanded(mask_warped, disp_step=2.0, upsample_factor=N)

        x_hat = gap_tv_cassi(y_expanded, operator, n_iter=50)
        score = compute_score(x_hat, x_expanded_full)  # All 10 scenes
        best_candidates.append(((dx, dy, theta), score))

    best_candidates = sorted(best_candidates, key=lambda x: -x[1])[:10]

    # Step 5: Stage 3 — Local refinement (coordinate descent, 6 rounds)
    for round_idx in range(6):
        for i in range(len(best_candidates)):
            (dx_cur, dy_cur, theta_cur), _ = best_candidates[i]

            for param in ['dx', 'dy', 'theta']:
                delta = {'dx': 0.5, 'dy': 0.5, 'theta': 0.05}[param]

                # Evaluate neighbors
                for offset in [-1, 0, 1]:
                    if param == 'dx':
                        dx_new, dy_new, theta_new = dx_cur + delta*offset, dy_cur, theta_cur
                    elif param == 'dy':
                        dx_new, dy_new, theta_new = dx_cur, dy_cur + delta*offset, theta_cur
                    else:
                        dx_new, dy_new, theta_new = dx_cur, dy_cur, theta_cur + delta*offset

                    mask_warped = warp_mask(mask_expanded, dx=dx_new, dy=dy_new, theta=theta_new)
                    operator = SDCASSIOperator_Expanded(mask_warped, disp_step=2.0, upsample_factor=N)
                    x_hat = gap_tv_cassi(y_expanded, operator, n_iter=50)
                    score = compute_score(x_hat, x_expanded_full)

                    best_candidates[i] = (((dx_new, dy_new, theta_new), score)
                                         if score > best_candidates[i][1] else best_candidates[i])

    # Step 6: Return best estimate, map back to original scale
    (dx_best, dy_best, theta_best), _ = best_candidates[0]
    dx_best_orig = dx_best / N  # Scale back to original grid
    dy_best_orig = dy_best / N

    return (dx_best_orig, dy_best_orig, theta_best)  # Return on original scale
```

**Computational cost:** ~1000 forward/adjoint calls × 50 GAP-TV iterations on expanded grid (4× spatial, 4× spectral) = ~20–30 hours GPU.

**Expected accuracy:** Within 0.1–0.2 px on original grid.

### 3.2 Algorithm 2: Differentiable Refinement (Gradient-based fine-tuning)

**Input:**
- Coarse estimate (dx₁, dy₁, θ₁) from Algorithm 1
- Expanded measurement y_expanded
- Expanded scenes x_expanded (all 10)
- Expansion factor N=2

**Procedure:**

```python
def upwmi_algorithm_2_differentiable_refinement(y_expanded, x_expanded_full, mask_expanded,
                                                (dx_coarse, dy_coarse, theta_coarse), N=2):
    """
    Gradient-based fine-tuning via unrolled GAP-TV on expanded grid.
    """

    # Parameterize as differentiable tensors
    dx = torch.nn.Parameter(torch.tensor(dx_coarse, dtype=torch.float32, requires_grad=True))
    dy = torch.nn.Parameter(torch.tensor(dy_coarse, dtype=torch.float32, requires_grad=True))
    theta = torch.nn.Parameter(torch.tensor(theta_coarse, dtype=torch.float32, requires_grad=True))

    def loss_fn():
        # Warp mask with current parameters
        mask_warped = differentiable_warp_mask(mask_expanded, dx=dx, dy=dy, theta=theta)
        operator = DifferentiableSDCASSIOperator_Expanded(mask_warped, disp_step=2.0, upsample_factor=N)

        # Unroll K=10 GAP-TV iterations
        x_hat = unrolled_gap_tv_k_iterations(y_expanded, operator, K=10)

        # Loss: MSE across all 10 expanded scenes
        loss = 0
        for scene_idx in range(10):
            loss += F.mse_loss(x_hat, x_expanded_full[scene_idx])

        return loss / 10

    optimizer = torch.optim.Adam([dx, dy, theta], lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    for epoch in range(200):
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([dx, dy, theta], max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: loss={loss.item():.6f}, dx={dx.item():.3f}, dy={dy.item():.3f}")

    # Return refined parameters (scale back to original grid)
    dx_refined = (dx.item()) / N
    dy_refined = (dy.item()) / N
    theta_refined = theta.item()

    return (dx_refined, dy_refined, theta_refined)
```

**Computational cost:** 200 epochs × 10 unrolled GAP-TV iterations on 4× spatial, 4× spectral grid = ~2–4 hours GPU.

**Expected accuracy:** 3–5× better than Algorithm 1 (within ±0.03 px on original grid).

### 3.3 Complete Workflow: Algorithm 1 → 2 → Validation

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          UPWMI Complete Pipeline                        │
└─────────────────────────────────────────────────────────────────────────┘

INPUT: 10 original scenes (256×256×28), original mask (256×256), measurement (256×310)

    ↓

┌─────────────────────────────────────────────────────────────────────────┐
│ PREPROCESSING: Expand to larger grid (N=2)                              │
│                                                                         │
│ • Upsample scenes spatially: 256×256 → 512×512 (bilinear)              │
│ • Expand spectral frames: 28 → 112 (cubic spline interpolation)       │
│ • Upsample mask: 256×256 → 512×512                                     │
│ • Upsample measurement: 256×310 → 512×620                              │
│ • Result: All data now on expanded (512×512×112) grid                  │
└─────────────────────────────────────────────────────────────────────────┘

    ↓

┌─────────────────────────────────────────────────────────────────────────┐
│ ALGORITHM 1: Beam Search (3D grid over dx, dy, θ)                      │
│              Duration: 20-30 hours GPU                                  │
│              Output: Coarse estimate (dx₁, dy₁, θ₁)                    │
│                                                                         │
│ Stage 1a: 1D sweep dx ∈ [-10,10], score on 1 scene       (~2 hours)    │
│ Stage 1b: 1D sweep dy ∈ [-10,10], score on 1 scene       (~2 hours)    │
│ Stage 1c: 1D sweep θ ∈ [-3°,3°], score on 1 scene        (~1 hour)     │
│ Stage 2:  Beam search: top-5 × top-5 × top-5 = 125 combos (~3 hours)   │
│ Stage 3:  Coordinate descent refinement, 6 rounds         (~3 hours)    │
│                                                                         │
│ Total: ~11 hours → (dx₁, dy₁, θ₁) [within ±0.5 px]                    │
└─────────────────────────────────────────────────────────────────────────┘

    ↓

┌─────────────────────────────────────────────────────────────────────────┐
│ CHECKPOINT 1: Coarse Reconstruction & Validation                        │
│                                                                         │
│ • Apply coarse correction: mask_warped_v1 = warp(mask, dx₁, dy₁, θ₁)  │
│ • Reconstruct with GAP-TV (25 iter): x̂_v1 = gap_tv(y, mask_warped_v1)│
│ • Downsample: x̂_v1_orig = downsample(x̂_v1, scale=1/2)                 │
│ • Compute metrics on all 10 original scenes:                            │
│   - PSNR, SSIM, SAM (per-scene + average)                              │
│   - Parameter recovery error: (dx_true - dx₁), etc.                    │
│                                                                         │
│ → Save: results_alg1_checkpoint.json                                   │
└─────────────────────────────────────────────────────────────────────────┘

    ↓

┌─────────────────────────────────────────────────────────────────────────┐
│ ALGORITHM 2: Differentiable Refinement via Unrolled GAP-TV             │
│              Duration: 2-4 hours GPU                                    │
│              Output: Refined estimate (dx₂, dy₂, θ₂)                   │
│                                                                         │
│ • Parameterize: dx, dy, θ as torch.nn.Parameter (requires_grad=True)   │
│ • Unroll 10 GAP-TV iterations inside loss function                     │
│ • Loss: MSE across all 10 expanded scenes (average)                    │
│ • Optimizer: Adam, lr=0.01 → 0.001 (cosine annealing)                 │
│ • Training: 200 epochs, batch_size=implicit (all 10 scenes)            │
│                                                                         │
│ → Output: (dx₂, dy₂, θ₂) [within ±0.05 px]                            │
└─────────────────────────────────────────────────────────────────────────┘

    ↓

┌─────────────────────────────────────────────────────────────────────────┐
│ CHECKPOINT 2: Refined Reconstruction & Validation                       │
│                                                                         │
│ • Apply refined correction: mask_warped_v2 = warp(mask, dx₂, dy₂, θ₂) │
│ • Reconstruct with GAP-TV + MST (if available): x̂_v2                  │
│ • Downsample: x̂_v2_orig = downsample(x̂_v2, scale=1/2)                 │
│ • Compute metrics on all 10 original scenes:                            │
│   - PSNR, SSIM, SAM (per-scene + average)                              │
│   - Compare v1 vs v2: ΔPSNR, ΔSSIM, ΔSAM                              │
│   - Parameter recovery error: (dx_true - dx₂), etc.                    │
│                                                                         │
│ → Save: results_alg2_checkpoint.json                                   │
└─────────────────────────────────────────────────────────────────────────┘

    ↓

┌─────────────────────────────────────────────────────────────────────────┐
│ FINAL REPORT: Compare Algorithm 1 vs Algorithm 2 vs Oracle              │
│                                                                         │
│ Table: PSNR/SSIM/SAM improvements across 10 scenes                      │
│ Figure: Parameter recovery scatter (true vs estimated)                  │
│ Figure: Reconstruction visual comparison                                │
│                                                                         │
│ → Save: cassi_mask_correction_report.md                                │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Part 4: Experimental Validation Protocol on 10 TSA Scenes

### 4.1 Dataset Setup

**TSA Simulation Benchmark:**
- Location: `MST-main/datasets/TSA_simu_data/`
- 10 scenes: scene01, scene02, ..., scene10
- Original shape: 256×256×28 (H×W×L)
- Mask: 256×256 binary coded aperture (provided)
- Dispersion: step=2.0 px/band (standard)

**Preparation:**
1. Load all 10 scenes
2. Inject synthetic mismatch: (dx_true, dy_true, θ_true) randomly sampled
   - dx_true ∈ [-3, 3] px (typical assembly error)
   - dy_true ∈ [-3, 3] px
   - θ_true ∈ [-1.5°, 1.5°] (smaller than W2 for realism)
3. Generate noisy measurement with injected mismatch
4. Run Algorithm 1 + 2 to recover parameters
5. Compare recovered vs injected parameters

### 4.2 Detailed Test Protocol (Per Scene)

```python
def run_full_test_scene(scene_idx, ground_truth_scene):
    """
    Full test protocol for one scene: load → expand → corrupt → correct → validate.
    """

    # ========== SETUP ==========
    x_true = ground_truth_scene  # (256, 256, 28)
    H, W, L = x_true.shape

    # Inject synthetic mismatch (ground truth for validation)
    dx_true = np.random.uniform(-3, 3)
    dy_true = np.random.uniform(-3, 3)
    theta_true = np.random.uniform(-np.pi/120, np.pi/120)  # [-1.5°, 1.5°]

    mask_orig = load_mask()  # (256, 256)
    mask_misaligned = warp_mask(mask_orig, dx=dx_true, dy=dy_true, theta=theta_true)

    # Generate measurement with injected mismatch
    operator_true = SDCASSIOperator(mask_misaligned, disp_step=2.0)
    y_clean = operator_true.forward(x_true)
    y_noisy = add_poisson_read_noise(y_clean, peak_photons=10000, read_sigma=0.01)

    # ========== EXPANSION ==========
    N = 2
    x_expanded = expand_spectral_frames(x_true, expansion_factor=2*N)  # (256,256,112)
    x_expanded = upsample_spatial(x_expanded, factor=N)  # (512,512,112)
    mask_expanded = upsample_spatial(mask_orig, factor=N)  # (512,512) [no mismatch yet]
    y_expanded = upsample_spatial(y_noisy, factor=N)  # (512, 620)

    # ========== ALGORITHM 1: Beam Search ==========
    (dx_hat1, dy_hat1, theta_hat1) = upwmi_algorithm_1_mask_beam_search(
        y_expanded, mask_expanded, x_expanded, N=N
    )

    # Checkpoint 1: Coarse reconstruction
    mask_warped_v1 = warp_mask(mask_orig, dx=dx_hat1, dy=dy_hat1, theta=theta_hat1)
    operator_v1 = SDCASSIOperator(mask_warped_v1, disp_step=2.0)
    x_hat_v1 = gap_tv_cassi(y_noisy, operator_v1, n_iter=50)

    psnr_v1 = psnr(x_hat_v1, x_true)
    ssim_v1 = ssim(x_hat_v1, x_true)
    sam_v1 = spectral_angle_mapper(x_hat_v1, x_true)

    error_dx_v1 = abs(dx_hat1 - dx_true)
    error_dy_v1 = abs(dy_hat1 - dy_true)
    error_theta_v1 = abs(theta_hat1 - theta_true) * 180 / np.pi  # degrees

    # ========== ALGORITHM 2: Gradient Refinement ==========
    (dx_hat2, dy_hat2, theta_hat2) = upwmi_algorithm_2_differentiable_refinement(
        y_expanded, x_expanded, mask_expanded,
        (dx_hat1, dy_hat1, theta_hat1), N=N
    )

    # Checkpoint 2: Refined reconstruction
    mask_warped_v2 = warp_mask(mask_orig, dx=dx_hat2, dy=dy_hat2, theta=theta_hat2)
    operator_v2 = SDCASSIOperator(mask_warped_v2, disp_step=2.0)
    x_hat_v2 = gap_tv_cassi(y_noisy, operator_v2, n_iter=50)

    psnr_v2 = psnr(x_hat_v2, x_true)
    ssim_v2 = ssim(x_hat_v2, x_true)
    sam_v2 = spectral_angle_mapper(x_hat_v2, x_true)

    error_dx_v2 = abs(dx_hat2 - dx_true)
    error_dy_v2 = abs(dy_hat2 - dy_true)
    error_theta_v2 = abs(theta_hat2 - theta_true) * 180 / np.pi

    # ========== ORACLE (Perfect correction) ==========
    operator_oracle = SDCASSIOperator(mask_orig, disp_step=2.0)  # No mismatch
    x_hat_oracle = gap_tv_cassi(y_noisy, operator_oracle, n_iter=50)

    psnr_oracle = psnr(x_hat_oracle, x_true)
    ssim_oracle = ssim(x_hat_oracle, x_true)
    sam_oracle = spectral_angle_mapper(x_hat_oracle, x_true)

    # ========== RETURN RESULTS ==========
    return {
        # Ground truth mismatch
        'dx_true': dx_true, 'dy_true': dy_true, 'theta_true': theta_true * 180 / np.pi,

        # Algorithm 1 results
        'dx_hat1': dx_hat1, 'dy_hat1': dy_hat1, 'theta_hat1': theta_hat1 * 180 / np.pi,
        'psnr_v1': psnr_v1, 'ssim_v1': ssim_v1, 'sam_v1': sam_v1,
        'error_dx_v1': error_dx_v1, 'error_dy_v1': error_dy_v1, 'error_theta_v1': error_theta_v1,

        # Algorithm 2 results
        'dx_hat2': dx_hat2, 'dy_hat2': dy_hat2, 'theta_hat2': theta_hat2 * 180 / np.pi,
        'psnr_v2': psnr_v2, 'ssim_v2': ssim_v2, 'sam_v2': sam_v2,
        'error_dx_v2': error_dx_v2, 'error_dy_v2': error_dy_v2, 'error_theta_v2': error_theta_v2,

        # Oracle (upper bound)
        'psnr_oracle': psnr_oracle, 'ssim_oracle': ssim_oracle, 'sam_oracle': sam_oracle,
    }
```

### 4.3 Expected Results

**Parameter recovery (Alg 2 expected accuracy):**
- dx: ±0.05–0.1 px error
- dy: ±0.05–0.1 px error
- θ: ±0.02–0.05° error

**PSNR improvements:**
- Baseline (no correction): ~15 dB (severely degraded)
- After Alg 1: ~20–23 dB
- After Alg 2: ~24–26 dB
- Oracle (true parameters): ~28–30 dB (gap of ~2–4 dB)

**Gap to oracle:** (PSNR_alg2 - PSNR_oracle) should be < 2 dB (indicating good parameter recovery).

### 4.4 Reporting Structure

Create: `pwm/reports/cassi_mask_correction_expanded_grid.md`

**Tables:**
1. **Table 1:** Parameter recovery accuracy (10 scenes, Alg 1 vs Alg 2)
   - Columns: Scene, dx_true, dx_hat1, error_dx_v1, dx_hat2, error_dx_v2, ...
   - Rows: scene01 — scene10, +Average

2. **Table 2:** PSNR/SSIM/SAM comparison
   - Columns: Scene, PSNR_v1, PSNR_v2, PSNR_oracle, Gap
   - Rows: scene01 — scene10, +Average

3. **Table 3:** Parameter recovery improvement (Alg 1 → Alg 2)
   - Columns: dx error reduction, dy error reduction, θ error reduction
   - Rows: Per-scene averages

**Figures:**
1. Parameter recovery scatter: (true dx, dy, θ) vs (estimated dx, dy, θ)
2. PSNR trajectory: Before correction → Alg 1 → Alg 2 → Oracle (10 scenes)
3. Reconstruction visual: Groundtruth vs Alg 1 vs Alg 2 (1–2 representative scenes)
4. Error maps: Reconstruction error (x̂_alg2 - x_true) for 1 scene, 3 bands

---

## Part 5: Implementation Architecture (Follows cassi_working_process.md)

### 5.1 Modified Forward Model Chain

Following `cassi_working_process.md` Section 3 (SD-CASSI Forward Model):

**Original chain (Section 3.5, Option A):**
```python
class SDCASSIOperator(PhysicsOperator):
    def forward(self, x):
        """y = Σ_l shift_y( X[:,:,l] ⊙ M, d_l )"""
        # Floating-point shifts: d_l = step · (l − n_c)
```

**New chain (Expanded-grid version):**
```python
class SDCASSIOperator_ExpandedGrid(PhysicsOperator):
    """
    SD-CASSI on expanded grid for integer dispersion shifts.
    Follows cassi_working_process.md but with upsampling/downsampling.
    """

    def __init__(self, mask, dispersion_step=2.0, upsample_factor=2):
        self.mask_base = mask  # (H, W)
        self.disp_step = dispersion_step  # 2.0 px/band (fixed)
        self.N = upsample_factor  # 2
        self.L = None  # Set on first forward call

    def forward(self, x):
        """
        Input: x (H, W, L=28)
        Output: y (H, W + (L-1)·step) on downsampled grid
        """
        H, W, L = x.shape
        self.L = L

        # 1. Expand spectral: 28 → 2N·28 = 112
        x_expanded_spec = expand_spectral_frames(x, expansion_factor=2*self.N)
        # x_expanded_spec: (H, W, 2N·L)

        # 2. Upsample spatial: (H,W) → (N·H, N·W)
        x_expanded = upsample_spatial(x_expanded_spec, factor=self.N)
        # x_expanded: (N·H, N·W, 2N·L)

        mask_expanded = upsample_spatial(self.mask_base, factor=self.N)
        # mask_expanded: (N·H, N·W)

        # 3. Forward model on expanded grid (integer shifts only)
        y_expanded = self._forward_integer_shifts(x_expanded, mask_expanded)
        # y_expanded: (N·H, N·W + (2N·L-1)·step)

        # 4. Downsample measurement
        y_downsampled = downsample_spatial(y_expanded, factor=self.N)
        # y_downsampled: (H, W + (L-1)·step) ← original measurement shape

        return y_downsampled

    def _forward_integer_shifts(self, x_expanded, mask_expanded):
        """Dispersion via exact integer pixel shifts (no interpolation)."""
        H_exp, W_exp, L_exp = x_expanded.shape
        W_out = W_exp + (L_exp - 1) * int(self.disp_step)
        y = np.zeros((H_exp, W_out), dtype=x_expanded.dtype)
        n_c = (L_exp - 1) / 2.0

        for l in range(L_exp):
            d_l = int(round(self.disp_step * (l - n_c)))  # Integer shift
            coded_slice = x_expanded[:, :, l] * mask_expanded
            y_start = max(0, d_l)
            y_end = min(W_out, W_exp + d_l)
            src_start = max(0, -d_l)
            src_end = src_start + (y_end - y_start)
            y[:, y_start:y_end] += coded_slice[:, src_start:src_end]

        return y

    def apply_mask_correction(self, dx: float, dy: float, theta: float):
        """Apply estimated mask shift (mask-only correction from UPWMI)."""
        self.mask_corrected = warp_affine(self.mask_base, dx=dx, dy=dy, theta=theta)
        # On next forward(), use mask_corrected instead of mask_base
```

### 5.2 New Algorithm Modules

Create files:
- `packages/pwm_core/pwm_core/algorithms/upwmi.py` — Algorithms 1 & 2
- `packages/pwm_core/pwm_core/physics/cassi_expanded.py` — Expanded-grid operator
- `scripts/run_cassi_mask_correction_full.py` — Full pipeline for 10 scenes

### 5.3 Updated Mismatch Registry (mismatch_db.yaml)

```yaml
cassi_mask_only:
  # New entry for mask-only correction (dispersion frozen)
  parameters:
    mask_dx:
      range: [-5, 5]
      typical_error: 0.5
      weight: 0.35
      correctable: true

    mask_dy:
      range: [-5, 5]
      typical_error: 0.5
      weight: 0.35
      correctable: true

    mask_theta:
      range: [-3, 3]  # degrees
      typical_error: 0.1
      weight: 0.30
      correctable: true

  correction_method: "UPWMI_algorithm_1_beam_search + UPWMI_algorithm_2_differentiable"
  note: |
    Dispersion parameters (a₁, α, a₂) are frozen at factory spec.
    Only mask geometry (dx, dy, θ) is corrected.
    Uses expanded-grid (N=2) for integer dispersion shifts.
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

**New (Expanded-grid approach):**
```python
# Skip dispersion curve fitting entirely
# Use frozen factory spec: a1=2.0, a2=0.0, alpha=0°

# 5. Estimate ONLY mask geometry
theta_mask = upwmi_algorithm_1(y_expanded, mask_expanded, scenes_expanded)
theta_mask = upwmi_algorithm_2(y_expanded, ..., theta_mask)

# 7. Simplified BeliefState
theta_calibrated = {**theta_nominal, **theta_mask}  # No theta_disp, theta_sd
Phi_cal = SDCASSIOperator_ExpandedGrid(mask_base, disp_step=2.0)
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

## Part 8: Implementation Timeline & Compute Requirements (UPDATED for N=4)

### Phase 1: Code Development (14 hours wall-clock)

| Task | Duration | GPU required | Notes |
|------|----------|--------------|-------|
| 1a. SDCASSIOperator_ExpandedGrid + spectral interp (N=4) | 2 h | No | Handle 1024×1024 grid |
| 1b. Mask expansion + padding (Option A) | 1.5 h | No | Scene crop/pad logic |
| 1c. Algorithm 1 (5D beam: dx,dy,θ,a₁,α) | 4 h | No | More complex than 3D |
| 1d. Algorithm 2 (differentiable refinement) | 3 h | No | PyTorch unrolling |
| 1e. Unit tests + fixtures (N=4 scale) | 2 h | No | Pytest, fixtures |
| 1f. Main orchestration + checkpoints | 1.5 h | No | Runner logic |
| **Phase 1 Total** | **14 h** | **No** | — |

### Phase 2: Single-Scene Validation (14 GPU hours)

| Task | GPU hours | Wall-clock | Notes |
|------|-----------|-----------|-------|
| 2a. scene01 full Alg1+Alg2 on N=4 grid | ~14 h | 4 h | Overnight run |
| **Phase 2 Total** | **14 h** | **4 h** | — |

### Phase 3: Full 10-Scene Execution (550–650 GPU hours)

| Task | GPU hours | Wall-clock (background) | Notes |
|------|-----------|-------------------------|-------|
| 3a. Algorithm 1 (5D beam search) on 10 scenes | ~350–400 h | 70–100 h | 35–40 h/scene |
| 3b. Algorithm 2 (gradient refinement) on 10 scenes | ~200–250 h | 40–60 h | 20–25 h/scene |
| **Phase 3 Total** | **550–650 h** | **110–160 h** | ~5–7 days continuous |

### Phase 4: Analysis & Reporting (6 hours)

| Task | Duration | Notes |
|------|----------|-------|
| 4a. Parse & aggregate 10-scene results | 1.5 h | Metrics, parameter errors, stats |
| 4b. Generate tables & figures | 2 h | PSNR/SSIM/SAM plots, scatter plots |
| 4c. Write analysis & discussion | 1.5 h | Algorithm trade-offs, limits |
| 4d. Final review & validation | 0.5 h | Sanity checks |
| **Phase 4 Total** | **6 h** | — |

### Summary

| Phase | Wall-clock | GPU hours | Notes |
|-------|-----------|-----------|-------|
| **1. Code dev** | **14 h** | **0 h** | Sequential, start first |
| **2. Single-scene** | **4 h** | **14 h** | After Phase 1 complete |
| **3. Full 10-scene** | **110–160 h** | **550–650 h** | **Continuous background GPU** |
| **4. Analysis** | **6 h** | **0 h** | Parallel (after 1–2 scenes done) |
| **Total elapsed** | **~200 h** | **~600 GPU** | **~9–10 days calendar** |

**Practical timeline (24/7 GPU execution):**
- Day 1: Phase 1 code development (14 h)
- Day 2: Phase 2 single-scene test (overnight + morning, 14 h GPU)
- Day 2 pm: Phase 3 starts (background)
- Days 3–8: Phase 3 continuous (550–650 GPU h ÷ 24 h/day ≈ 23–27 days single GPU ≈ 2–3 weeks)
- Days 9–10: Phase 4 analysis (after enough scenes completed)
- **Total: ~9–10 days calendar with 24/7 GPU access**

---

## Appendix A: Integer Dispersion Shift Mathematics

### A.1 Original Grid (256 × 256 × 28)

Given:
- Spatial: H=256, W=256
- Spectral: L=28 bands
- Dispersion: s=2.0 px/band (linear, nominal a₂=0)
- Center band: n_c = (L-1)/2 = 13.5

**Dispersion shifts on original grid:**
```
d_l = s · (l − n_c) = 2.0 · (l − 13.5)

l=0:    d_l = −27 px
l=13:   d_l = −1 px
l=14:   d_l = +1 px
l=27:   d_l = +27 px
```

**Observation:** Shifts are integers (except at l=13.5 center), but fractional in continuous space.

### A.2 Expanded Grid (512 × 512 × 112, N=2)

**Spatial upsampling:** 256 → 512 (factor N=2)
- Pixel spacing: 1 unit → 0.5 units (2× finer)

**Spectral expansion:** 28 → 112 (factor 2N=4)
- Via cubic spline interpolation
- Wavelength spacing: finer by 4×
- New center: n'_c = (112-1)/2 = 55.5

**Dispersion shifts on expanded grid:**
```
d'_l = s · (l − n'_c) = 2.0 · (l − 55.5)

l=0:    d'_l = −111 px  (integer)
l=27:   d'_l = −47.0 px → round to −47 px (integer)
l=55:   d'_l = −1 px   (integer)
l=56:   d'_l = +1 px   (integer)
l=111:  d'_l = +111 px (integer)
```

**All shifts are now integers!** This eliminates fractional pixel registration errors.

### A.3 Mask Shift Error Quantization

**Original grid:** Mask shift errors (dx, dy) are fractional, e.g., dx=1.5 px
- Becomes ambiguous: round to 1 or 2 px?
- Introduces sub-pixel registration error

**Expanded grid (N=2):** Same physical misalignment, but on 2× finer grid
- dx=1.5 px (original) → dx'=3.0 px (expanded) **integer!**
- dy=0.7 px (original) → dy'=1.4 px (expanded) → round to ±1 px (error <0.5 px at original scale)

**Benefit:** Mask errors naturally quantize to integers on expanded grid, reducing fitting difficulty.

### A.4 Spectral Interpolation Energy Conservation

**Assumption:** Hyperspectral scenes have smooth spectral variation.

**Integration check:**
```
∫_λ x_expanded(λ) dλ ≈ ∫_λ x_original(λ) dλ  (by Riemann sum approximation)
```

**Example:** Original band l=5 has intensity I₅
- Expanded bands corresponding to λ₅: l'∈{19, 20, 21, 22} (4 bands per original)
- Sum: I'₁₉ + I'₂₀ + I'₂₁ + I'₂₂ ≈ 4·I₅ (approximately, within interpolation error)

**Consequence:** Measurement y_expanded integrates over 4× more bands, but with 1/4 the intensity per band (noise amplification offset by signal concentration).

---

## Appendix B: Detailed Comparison — Current W2 vs Proposed Expanded-Grid Approach

### B.1 Mismatch Model Comparison

| Aspect | Current W2 (5 scenarios) | Proposed Approach |
|--------|-------------------------|-------------------|
| **Fitted parameters** | 5 (dx, dy, θ, a₁, α) | 3 (dx, dy, θ) |
| **Frozen parameters** | None | a₁=2.0, α=0°, a₂=0 |
| **Interpretation** | All are "calibration errors" | Only mask geometry is error; dispersion is system property |
| **Search space** | 5D grid | 3D grid (reduced by 40%) |
| **Grid search cost** | ~50 min (W2 full sweep) | ~15 min (Alg1 beam search) |

### B.2 Algorithm Comparison

| Aspect | Current W2 | Proposed (Alg 1+2) |
|--------|-----------|-------------------|
| **Algorithm 1** | Independent grid search per param (1D sweeps only) | Beam search: 1D → 3D beam (width=10) → coord descent |
| **Algorithm 2** | None (only NLL grid search) | **NEW:** Differentiable refinement via unrolled GAP-TV |
| **Refinement strategy** | NLL-based (measurement residual) | Gradient-based (reconstruction error) |
| **Unroll depth** | N/A | K=10 GAP-TV iterations |
| **Expected accuracy (Alg1)** | ±0.2 px | ±0.15 px |
| **Expected accuracy (Alg2)** | N/A | ±0.05 px **3–4× better** |
| **Computational cost** | ~50 min (1 scene) | ~25 h Alg1 + ~10 h Alg2 (10 scenes, GPU) |

### B.3 Experimental Scope

| Aspect | Current W2 | Proposed |
|--------|-----------|----------|
| **Number of scenes** | 1 (scene01) | **10 (all TSA benchmark)** |
| **Mismatch per scene** | Fixed: W2a–W2e | Random: (dx, dy, θ) each scene |
| **Ground truth** | Known (injected) | Known (injected, randomized) |
| **Validation metrics** | PSNR, SSIM, SAM | PSNR, SSIM, SAM, **parameter recovery error** |
| **Robustness test** | Single scene type | 10 diverse natural scenes |

### B.4 Reconstruction Quality

| Scenario | Current W2 (best case) | Proposed (expected) |
|----------|----------------------|-------------------|
| **Without correction** | 15.01 dB (W2d) | ~15 dB (severe artifacts) |
| **After Algorithm 1** | 22.05 dB (W2d) | ~22 dB (coarse correction) |
| **After Algorithm 2** | N/A | ~25 dB (refined correction) |
| **Oracle (true params)** | 22.05+ dB? | ~28 dB (upper bound) |
| **Gap to oracle** | Unknown | Expected <2 dB |

### B.5 Physics Interpretation

| Aspect | Current W2 | Proposed |
|--------|-----------|----------|
| **Prism dispersion** | Treated as variable (a₁, α fitted) | Treated as **fixed system property** |
| **Reality check** | Does prism change after reassembly? No! | ✓ Correctly treats as constant |
| **Mask position** | Treated as variable (dx, dy, θ fitted) | Treated as **variable (to be corrected)** |
| **Reality check** | Does mask position change after reassembly? Yes! | ✓ Correctly treats as correctable error |
| **Calibration strategy** | Fit all 5 params per measurement | Pre-calibrate dispersion once, fit mask per session |
| **Practical implication** | Slower, higher dim search, less physical | Faster, lower dim search, more realistic |

### B.6 Summary: Why Expanded-Grid Approach is Better

| Criterion | Score |
|-----------|-------|
| **Smaller search space** (3D vs 5D) | ✓ |
| **Physically accurate** (prism is fixed) | ✓ |
| **Better parameter recovery** (Alg2: ±0.05 px) | ✓ |
| **Comprehensive validation** (10 scenes vs 1) | ✓ |
| **Correct algorithm choice** (gradient > grid for high-dim) | ✓ |
| **Addresses sin(α)·δ(λ) naturally** (integer shifts → no angle needed) | ✓ |
| **Practical engineering** (matches real calibration workflow) | ✓ |

---

---

## Conclusion & Key Insights

### Why This Approach Works

1. **Physically Grounded:**
   - Prism dispersion δ(λ) is determined by optical design → **fixed after factory calibration**
   - Mask position varies with mechanical reassembly → **requires per-session correction**
   - This plan respects this fundamental distinction

2. **Algorithmically Superior:**
   - 5D grid search (current W2) is inefficient; 3D grid search (expanded-grid) is faster
   - Algorithm 1 (beam search) handles coarse estimation well
   - Algorithm 2 (gradient-based) provides 3–5× refinement, exploiting differentiable forward model

3. **Integer Shift Advantage:**
   - Spectral expansion (28 → 112 bands) + spatial upsampling (256 → 512) ensures all dispersion shifts become **integers**
   - No fractional pixel interpolation → exact forward/adjoint model → better gradient flow

4. **Comprehensive Validation:**
   - 10 scenes (vs 1 in current W2) provides robust statistical evaluation
   - Randomized mismatch per scene tests generalization
   - Parameter recovery metrics confirm algorithm effectiveness

### Expected Outcomes

**Parameter recovery (Algorithm 2):**
- dx, dy: ±0.05–0.10 px accuracy
- θ: ±0.02–0.05° accuracy
- 3–5× better than Algorithm 1 alone

**Reconstruction quality:**
- Without correction: ~15 dB (unusable)
- After Alg 1: ~22 dB (acceptable)
- After Alg 2: ~25 dB (good, within 3 dB of oracle)

**Practical impact:**
- Faster calibration (3D vs 5D search)
- More reliable parameter estimates (gradient refinement)
- Better matches real lab workflows (prism is fixed, mask varies)

### Architecture Integration

This plan **extends** `cassi_working_process.md` (Section 13 — Operator-Correction Mode):
- Replaces 5-parameter fitting with 3-parameter correction
- Adds spectral expansion preprocessing
- Implements UPWMI Algorithms 1 & 2 for mask-only calibration
- Maintains compatibility with existing PWM framework

---

## Approval & Next Steps

**✅ Ready for Review**

Please review this plan document and provide feedback on:
- **Section 6:** Open questions (N, interpolation method, algorithm configurations)
- **Section 3:** Algorithm details (feasible for your GPU? Any modifications needed?)
- **Part 4:** Validation protocol (appropriate test design?)
- **Part 8:** Timeline (achievable in your schedule?)

**After approval:**
1. You confirm final decisions on open questions
2. I implement all code (Phases 1–4)
3. Full 10-scene validation with comprehensive reporting
4. Integration into `pwm/reports/cassi_mask_correction_expanded_grid.md`

---
