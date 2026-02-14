# CASSI Calibration via Enlarged Simulation Grid & Mask Correction
**Plan Document — v4 (2026-02-15, COMPLETE REDESIGN)**

## Executive Summary

This document proposes a **complete CASSI calibration strategy** based on enlarged simulation grid with realistic mask handling:

1. **(A) Scene Preprocessing:** Crop edges (P=16 px/side) to ensure information stays within nominal measurement region
   - Original: 256×256×28 → Crop interior (224×224) → Zero-pad to (256×256×28)
   - Purpose: Prevent signal leakage due to mismatch

2. **(B) Shift-Crop Dataset Expansion:** Generate 2N=4 crops per scene with stride-1 spacing
   - Reflect-pad scene along dispersion axis by M≥3 px
   - Generate 4 crops at offsets [0, 1, 2, 3] → 40 total crops from 10 scenes
   - Purpose: Multiple viewpoints for robust parameter fitting

3. **(C) Enlarged Grid Forward Model:** High-fidelity simulation with N=4 spatial, K=2 spectral
   - Spatial enlargement: 256×256 → 1024×1024 (factor N=4)
   - Spectral expansion: 28 → 217 bands, L_expanded = (L-1)×N×K+1 = 27×4×2+1
   - **Dispersion shift in simulation:** stride-1 (1 pixel per frame, fine granularity)
   - Measurement size after summation: 1024×1240 (width = 1024 + 2×108)
   - Downsample to original: 1024×1240 → 256×310 (factor 4)

4. **(D) Real Mask Handling:** Use experimental mask from TSA dataset
   - Load mask: `/home/spiritai/MST-main/datasets/TSA_simu_data/mask.mat` (256×256)
   - Upsample to 1024×1024 for enlarged simulation
   - For each of 217 frames, create shifted version (dispersion encoding)
   - Mismatch injected: apply (dx, dy, θ) to BOTH mask AND scene equally
   - Downsample back to 256×256 for reconstruction

5. **(E) Three Reconstruction Scenarios:**
   - **Scenario I (Ideal):** Ideal measurement + ideal mask + ideal forward model → x̂_ideal (oracle)
   - **Scenario II (Assumed):** Ideal measurement + assumed perfect mask + **simulated forward model** → x̂_assumed (baseline)
   - **Scenario III (Corrected):** Corrupted measurement + corrected mask + **simulated forward model** → x̂_corrected (practical)

6. **(F) Calibration (3 Parameters Only):** Correct mask geometry via UPWMI Algorithms 1 & 2
   - **Mask shifts:** dx, dy ∈ [-3, 3] px (assembly tolerance)
   - **Mask rotation:** θ ∈ [-1°, 1°] (optical bench twist)
   - Apply same mismatch to both mask and scene during injection

7. **(G) Validate on 10 Scenes** with parameter recovery and comprehensive three-scenario comparison

**Core strategy:** Enlarged grid simulation (N=4, K=2) for accurate forward model, then correct misalignment via Algorithms 1&2, comparing ideal/assumed/corrected reconstructions.

---

## Part 1: Scene Preprocessing & Shift-Crop Expansion

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

### 1.2 Shift-Crop Dataset Expansion (Stride-1)

**Input:** Preprocessed scene x_padded (256×256×28)

**Process:**
```
1. Reflect-pad scene along dispersion (W) axis by margin M ≥ 2N-1 = 3 px

   Padded scene: x_padded_W = reflect_pad(x_padded, pad_width=((0,0), (3,3), (0,0)))
   New shape: (256, 256+6, 28) = (256, 262, 28)

2. Generate offsets with stride-1: o = [0, 1, 2, 3]

   Purpose: Dense sampling with overlapping crops

3. Crop at each offset: x_i = crop(x_padded_W, W_start=o_i, W_end=o_i+256)

   x_0 = x_padded_W[:, 0:256, :]      ← leftmost region
   x_1 = x_padded_W[:, 1:257, :]      ← +1 px shift
   x_2 = x_padded_W[:, 2:258, :]      ← +2 px shift
   x_3 = x_padded_W[:, 3:259, :]      ← +3 px shift

   Result: 4 crops per original scene (256×256×28 each)

4. Dataset expansion:
   10 original scenes × 4 crops each = 40 total training crops
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

**Purpose:** Upper bound - best possible reconstruction without any simulation effects.

**Forward model:** Ideal, direct (no enlargement)
```python
def forward_ideal(x_256, mask_256):
    """
    Ideal forward model (256×256×28 → 256×310).
    No enlargement, no interpolation artifacts.
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
1. Load ideal mask: mask_ideal (256×256, no mismatch)
2. Generate ideal measurement: y_ideal = forward_ideal(x_256, mask_ideal)
   Size: 256×310, no noise
```

**Reconstruction:**
```
x̂_ideal = solver(y_ideal, forward_ideal, mask_ideal, n_iter=50)
```

**Metrics:** PSNR_ideal, SSIM_ideal, SAM_ideal (oracle baseline)

### 3.2 Scenario II: Assumed Mask Reconstruction (Baseline)

**Purpose:** Show impact of simulation method (enlarged grid) vs ideal forward model.

**Forward model:** Simulated with N=4, K=2 (from Section 2.4)

**Measurement generation:**
```
1. Use SAME ideal measurement from Scenario I: y_ideal (256×310)
   (No noise, no mismatch)

2. Use ASSUMED perfect mask (downsampled from enlarged 1024×1024 version):
   mask_1024 = upsample(mask_ideal, 4)
   mask_assumed = downsample(mask_1024, 4)  # Back to 256×256
```

**Reconstruction:**
```
Operator: phi_assumed = SimulatedOperator(mask_assumed, N=4, K=2)
x̂_assumed = solver(y_ideal, phi_assumed, n_iter=50)
```

**Why use same y_ideal:** To isolate the effect of simulation method vs ideal forward model.

**Metrics:** PSNR_assumed, SSIM_assumed, SAM_assumed
- Compare with x̂_ideal to measure simulation fidelity loss

### 3.3 Scenario III: Corrected Mask Reconstruction (Practical)

**Purpose:** Practical scenario with mismatch correction.

**Measurement generation with mismatch:**
```
1. Inject synthetic mismatch to BOTH scene and mask:
   dx_true, dy_true, θ_true (uniformly random in ranges)

   x_misaligned = warp_affine(x_256, dx=dx_true, dy=dy_true, theta=θ_true)
   mask_misaligned_256 = warp_affine(mask_256, dx=dx_true, dy=dy_true, theta=θ_true)

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

**Correction via Algorithm 1 & 2:**
```
# Algorithm 1: Coarse beam search
(dx_hat1, dy_hat1, theta_hat1) = upwmi_algorithm_1(
    y_crops, mask_ideal, x_crops_ref,
    search_space={'dx': [-3,3], 'dy': [-3,3], 'theta': [-1°,1°]}
)

# Algorithm 2: Gradient refinement
(dx_hat2, dy_hat2, theta_hat2) = upwmi_algorithm_2(
    y_crops, x_crops_ref, y_all_scenes, x_all_scenes,
    mask_ideal, coarse_estimate=(dx_hat1, dy_hat1, theta_hat1)
)
```

**Reconstruction with corrected mask:**
```
# Option A: Corrected mask from Algorithm 1
mask_corrected_1 = warp_affine(mask_ideal, dx=dx_hat1, dy=dy_hat1, theta=theta_hat1)
Operator: phi_corrected_1 = SimulatedOperator(mask_corrected_1, N=4, K=2)
x̂_corrected_1 = solver(y_noisy, phi_corrected_1, n_iter=50)

# Option B: Corrected mask from Algorithm 2 (better)
mask_corrected_2 = warp_affine(mask_ideal, dx=dx_hat2, dy=dy_hat2, theta=theta_hat2)
Operator: phi_corrected_2 = SimulatedOperator(mask_corrected_2, N=4, K=2)
x̂_corrected_2 = solver(y_noisy, phi_corrected_2, n_iter=50)
```

**Metrics:** PSNR_corrected, SSIM_corrected, SAM_corrected, parameter errors

---

## Part 4: Comparison & Analysis

### 4.1 Three-Scenario Comparison Table (Per Scene)

| Scenario | Measurement | Mask | Operator | Metrics | Purpose |
|----------|-------------|------|----------|---------|---------|
| **I. Ideal** | y_ideal (clean) | mask_ideal (true) | Ideal direct | PSNR_ideal | Oracle upper bound |
| **II. Assumed** | y_ideal (clean) | mask_assumed (perfect) | Simulated N=4, K=2 | PSNR_assumed | Simulation fidelity |
| **III. Corrected** | y_noisy (corrupted+noise) | mask_corrected (est.) | Simulated N=4, K=2 | PSNR_corrected | Practical result |

**Expected gaps:**
```
PSNR_ideal ≥ PSNR_assumed ≥ PSNR_corrected

Gap I→II: Impact of enlarged grid simulation (typically <1-2 dB)
Gap II→III: Impact of mismatch + correction (typical 3-5 dB before correction)
Gap III→I: Total loss from simulation + mismatch (typically 5-10 dB)
```

### 4.2 Parameter Recovery Accuracy (Algorithm 1 vs 2)

**From Algorithm 2 (gradient refinement):**
```
Expected accuracy:
  dx: ±0.05–0.1 px error
  dy: ±0.05–0.1 px error
  θ: ±0.02–0.05° error

Algorithm 1 vs 2: Typically 3–5× improvement (Algorithm 2 better)
```

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

### 5.2 UPWMI Algorithm 1: Beam Search

```python
def upwmi_algorithm_1_beam_search(y_crops, mask_ideal, x_crops_ref, n_crops=4):
    """
    Coarse 3D beam search for (dx, dy, theta).

    Input:
        y_crops: 4 measurements from shift-crops (256×310 each)
        mask_ideal: Ideal mask (256×256)
        x_crops_ref: 4 reference crops (256×256×28 each)

    Output:
        (dx_hat, dy_hat, theta_hat)
    """

    search_space = {
        'dx': np.linspace(-3, 3, 13),
        'dy': np.linspace(-3, 3, 13),
        'theta': np.linspace(-np.pi/180, np.pi/180, 7),
    }

    # Stage 1: 1D sweeps with proxy reconstruction (K=5 iterations)
    scores_dx = []
    for dx in search_space['dx']:
        mask_test = warp_affine(mask_ideal, dx=dx, dy=0, theta=0)
        operator = SimulatedOperator_EnlargedGrid(mask_test, N=4, K=2)

        score = 0
        for i in range(n_crops):
            x_hat = gap_tv_cassi(y_crops[i], operator, n_iter=5)
            score += compute_score(x_hat, x_crops_ref[i])

        scores_dx.append((dx, score / n_crops))

    top_dx = [s[0] for s in sorted(scores_dx, key=lambda x: -x[1])[:5]]

    # Repeat for dy, theta (similar logic)
    top_dy = [...]
    top_theta = [...]

    # Stage 2: Beam search (5×5×5=125 combinations, K=10 iterations)
    candidates = list(itertools.product(top_dx, top_dy, top_theta))
    results = []

    for (dx, dy, theta) in candidates:
        mask_test = warp_affine(mask_ideal, dx=dx, dy=dy, theta=theta)
        operator = SimulatedOperator_EnlargedGrid(mask_test, N=4, K=2)

        score = 0
        for i in range(n_crops):
            x_hat = gap_tv_cassi(y_crops[i], operator, n_iter=10)
            score += compute_score(x_hat, x_crops_ref[i])

        results.append(((dx, dy, theta), score / n_crops))

    top_5 = sorted(results, key=lambda x: -x[1])[:5]

    # Stage 3: Coordinate descent refinement (3 rounds)
    best = top_5[0]
    for round_idx in range(3):
        for param in ['dx', 'dy', 'theta']:
            delta = {'dx': 0.25, 'dy': 0.25, 'theta': 0.05 * np.pi/180}[param]
            for offset in [-1, 0, 1]:
                (dx_cur, dy_cur, theta_cur) = best[0]

                if param == 'dx':
                    test_val = (dx_cur + delta*offset, dy_cur, theta_cur)
                elif param == 'dy':
                    test_val = (dx_cur, dy_cur + delta*offset, theta_cur)
                else:
                    test_val = (dx_cur, dy_cur, theta_cur + delta*offset)

                # Evaluate and update best
                # ...

    return best[0]  # (dx_hat, dy_hat, theta_hat)
```

### 5.3 UPWMI Algorithm 2: Gradient Refinement

```python
def upwmi_algorithm_2_gradient_refinement(
    y_crops, x_crops_ref, y_all_scenes, x_all_scenes,
    mask_ideal, coarse_estimate):
    """
    Gradient-based refinement of mask correction parameters.

    Input:
        y_crops, x_crops_ref: 4 shift-crops for phase 1 (fast)
        y_all_scenes, x_all_scenes: all 10 scenes for phase 2 (robust)
        mask_ideal: ideal mask (256×256)
        coarse_estimate: (dx1, dy1, theta1) from Algorithm 1

    Output:
        (dx_refined, dy_refined, theta_refined)
    """

    # Parameterize as differentiable tensors
    dx = torch.nn.Parameter(torch.tensor(coarse_estimate[0], dtype=torch.float32))
    dy = torch.nn.Parameter(torch.tensor(coarse_estimate[1], dtype=torch.float32))
    theta = torch.nn.Parameter(torch.tensor(coarse_estimate[2], dtype=torch.float32))

    def loss_fn(y_list, x_ref_list):
        mask_warped = differentiable_warp_affine(mask_ideal, dx=dx, dy=dy, theta=theta)
        operator = DifferentiableSimulatedOperator(mask_warped, N=4, K=2)

        loss = 0
        for y_meas, x_ref in zip(y_list, x_ref_list):
            x_hat = unrolled_gap_tv(y_meas, operator, K=10)
            loss += F.mse_loss(x_hat, x_ref)

        return loss / len(y_list)

    # Phase 1: Optimize on 4 shift-crops (100 epochs, lr=0.01)
    optimizer = torch.optim.Adam([dx, dy, theta], lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    for epoch in range(100):
        optimizer.zero_grad()
        loss = loss_fn(y_crops, x_crops_ref)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([dx, dy, theta], max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 20 == 0:
            print(f"Phase 1 Epoch {epoch}: loss={loss.item():.6f}")

    # Phase 2: Fine-tune on all 10 scenes (50 epochs, lr=0.001)
    optimizer = torch.optim.Adam([dx, dy, theta], lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    for epoch in range(50):
        optimizer.zero_grad()
        loss = loss_fn(y_all_scenes, x_all_scenes)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([dx, dy, theta], max_norm=0.5)
        optimizer.step()
        scheduler.step()

        if epoch % 10 == 0:
            print(f"Phase 2 Epoch {epoch}: loss={loss.item():.6f}")

    return (dx.item(), dy.item(), theta.item())
```

---

## Part 6: Validation Protocol (10 Scenes)

### 6.1 Per-Scene Workflow

```python
def run_full_validation_scene(scene_idx, x_true_256, mask_ideal_256):
    """
    Complete validation for one scene: all 3 scenarios + corrections.
    """

    # ========== SCENARIO I: IDEAL ==========
    y_ideal = forward_ideal(x_true_256, mask_ideal_256)  # 256×310
    x_hat_ideal = solver(y_ideal, forward_ideal, mask_ideal_256, n_iter=50)
    psnr_ideal = psnr(x_hat_ideal, x_true_256)

    # ========== SCENARIO II: ASSUMED (Simulated operator) ==========
    x_hat_assumed = solver(y_ideal, SimulatedOperator_EnlargedGrid(mask_ideal_256), n_iter=50)
    psnr_assumed = psnr(x_hat_assumed, x_true_256)

    # ========== SCENARIO III: CORRECTED ==========

    # Inject mismatch
    dx_true = np.random.uniform(-3, 3)
    dy_true = np.random.uniform(-3, 3)
    theta_true = np.random.uniform(-np.pi/180, np.pi/180)

    x_misaligned = warp_affine(x_true_256, dx=dx_true, dy=dy_true, theta=theta_true)
    mask_misaligned = warp_affine(mask_ideal_256, dx=dx_true, dy=dy_true, theta=theta_true)

    # Generate corrupted measurement
    y_enlarged = forward_enlarged(enlarge(x_misaligned), upsample(mask_misaligned, 4))
    y_corrupt = downsample(y_enlarged, 4)
    y_noisy = add_noise(y_corrupt, peak=10000, sigma=0.01)

    # Generate shift-crops for Algorithm 1 & 2
    x_crops, y_crops = generate_shift_crops(x_misaligned, n_crops=4)

    # Algorithm 1: Coarse correction
    (dx_hat1, dy_hat1, theta_hat1) = upwmi_algorithm_1(y_crops, mask_ideal_256, x_crops)
    mask_corrected_1 = warp_affine(mask_ideal_256, dx=dx_hat1, dy=dy_hat1, theta=theta_hat1)
    x_hat_alg1 = solver(y_noisy, SimulatedOperator_EnlargedGrid(mask_corrected_1), n_iter=50)

    psnr_alg1 = psnr(x_hat_alg1, x_true_256)
    err_dx_alg1 = abs(dx_hat1 - dx_true)
    err_dy_alg1 = abs(dy_hat1 - dy_true)
    err_theta_alg1 = abs(theta_hat1 - theta_true) * 180 / np.pi

    # Algorithm 2: Refined correction
    (dx_hat2, dy_hat2, theta_hat2) = upwmi_algorithm_2(
        y_crops, x_crops, [y_noisy]*10, [x_misaligned]*10,  # dummy for all scenes
        mask_ideal_256, (dx_hat1, dy_hat1, theta_hat1)
    )
    mask_corrected_2 = warp_affine(mask_ideal_256, dx=dx_hat2, dy=dy_hat2, theta=theta_hat2)
    x_hat_alg2 = solver(y_noisy, SimulatedOperator_EnlargedGrid(mask_corrected_2), n_iter=50)

    psnr_alg2 = psnr(x_hat_alg2, x_true_256)
    err_dx_alg2 = abs(dx_hat2 - dx_true)
    err_dy_alg2 = abs(dy_hat2 - dy_true)
    err_theta_alg2 = abs(theta_hat2 - theta_true) * 180 / np.pi

    # ========== RETURN RESULTS ==========
    return {
        'scene_idx': scene_idx,
        'dx_true': dx_true, 'dy_true': dy_true, 'theta_true': theta_true * 180 / np.pi,

        'psnr_ideal': psnr_ideal,
        'psnr_assumed': psnr_assumed,

        'psnr_alg1': psnr_alg1,
        'err_dx_alg1': err_dx_alg1, 'err_dy_alg1': err_dy_alg1, 'err_theta_alg1': err_theta_alg1,

        'psnr_alg2': psnr_alg2,
        'err_dx_alg2': err_dx_alg2, 'err_dy_alg2': err_dy_alg2, 'err_theta_alg2': err_theta_alg2,

        'gap_ideal_assumed': psnr_ideal - psnr_assumed,
        'gain_alg1_to_alg2': psnr_alg2 - psnr_alg1,
        'gap_alg2_to_ideal': psnr_ideal - psnr_alg2,
    }
```

### 6.2 Reporting

Create comprehensive report: `pwm/reports/cassi_enlarged_grid_complete.md`

**Tables:**
1. **Three-scenario PSNR comparison** (10 scenes)
   - PSNR_ideal, PSNR_assumed, PSNR_alg1, PSNR_alg2, gaps

2. **Parameter recovery accuracy** (Algorithm 1 vs 2)
   - dx_true vs dx_hat, parameter errors, improvement

3. **Timing breakdown**
   - Forward model time, Algorithm 1 time, Algorithm 2 time, total per scene

**Figures:**
1. PSNR trajectory across 10 scenes (ideal / assumed / alg1 / alg2)
2. Parameter error scatter plots (dx, dy, θ true vs estimated)
3. Reconstruction visual comparisons (2-3 scenes, 3 bands)

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

### Phase 2: Single-Scene Validation (GPU: 12 hours)
- Scenario I (ideal): 1 h
- Scenario II (assumed): 3 h
- Scenario III + Alg1 + Alg2: 8 h

### Phase 3: Full 10-Scene Execution (GPU: ~120 hours)
- Scenarios I & II: 10 h (quick, just forward models)
- Algorithm 1 (10 scenes): 45 h
- Algorithm 2 (10 scenes): 65 h

### Phase 4: Analysis & Reporting (4 hours)
- Aggregate results, generate tables/figures: 4 h

**Total calendar time:** ~5-6 days (with 24/7 GPU)

---

## Conclusion

This revised plan provides:
1. **Realistic simulation:** N=4 enlargement + real mask + 217-frame dispersion
2. **Three-scenario validation:** Ideal / Assumed / Corrected (clear benchmarking)
3. **Robust mismatch correction:** Algorithms 1 & 2 with proper gradient refinement
4. **Comprehensive metrics:** PSNR/SSIM/SAM + parameter recovery + timing

**Expected outcomes:**
- Ideal (oracle): ~28–30 dB
- Assumed (baseline): ~27–29 dB (typically <1 dB loss from simulation)
- Corrected (practical): ~23–25 dB after correction, gap <2–3 dB to oracle
- Parameter accuracy: ±0.05–0.1 px (Algorithm 2)

---

**Plan version:** v4 Complete Redesign (2026-02-15)
**Ready for implementation upon approval.**
