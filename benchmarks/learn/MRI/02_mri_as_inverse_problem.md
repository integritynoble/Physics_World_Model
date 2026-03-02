# 02 — MRI as an Inverse Problem

## 1. The Forward Model

### 1.1 Nominal Forward Model

In MRI reconstruction, we observe multi-coil k-space data **y** and want to
recover the image **x**. The nominal (idealised) forward model is:

```
y_c = M · F · S_c · x + n_c       for each coil c = 1, ..., C
```

where:
- **x** ∈ ℝ^(H×W) — the unknown image (magnitude, normalised to [0, 1])
- **S_c** ∈ ℂ^(H×W) — coil sensitivity map for coil c (complex)
- **F** — 2D discrete Fourier transform (centred: `fftshift(fft2(ifftshift(·)))`)
- **M** ∈ {0,1}^(H×W) — binary undersampling mask (Cartesian, 1D along ky)
- **n_c** ∈ ℂ^(H×W) — additive complex Gaussian noise
- **y_c** ∈ ℂ^(H×W) — observed k-space for coil c (zeros at unsampled lines)

In compact notation with all coils stacked:

```
y = A x + n       where  A = M · F · S
```

The adjoint (backward) operator is:

```
A^H y = Σ_c  S_c^H · F^H · M^H · y_c
```

### 1.2 Centred FFT Convention

The PWM codebase uses centred FFTs throughout:

```python
# Forward: image → k-space
def _fft2c(x):
    return fftshift(fft2(ifftshift(x), axes=(-2, -1)), axes=(-2, -1))

# Backward: k-space → image
def _ifft2c(k):
    return fftshift(ifft2(ifftshift(k), axes=(-2, -1)), axes=(-2, -1))
```

See `build_dataset.py`, lines 108–121.

---

## 2. The 4-Knob Mismatch Model

Real MRI data deviates from the nominal model. The PWM benchmark
introduces **four types of physics mismatch** between the true acquisition
and what algorithms assume. This is the key innovation that makes the
benchmark challenging.

### 2.1 True (Mismatched) Forward Model

```
x_warped  = warp(x, δr)                              # 1. gradient nonlinearity
x_mod     = x_warped · exp(i · 2π · B0_hz · TE · b0_map)   # 2. B0 inhomogeneity
S_c_true  = S_c_nominal · (1 + ε_c)                  # 3. coil sensitivity mismatch
y_c       = mask · k_traj_err(F(S_c_true · x_mod)) + n_c   # 4. k-trajectory error
```

Algorithms receive `y_c` (true acquisition) but only know the nominal
model `y = MFS x + n`. The gap between the two is the **model mismatch**.

### 2.2 The Four Mismatch Knobs

#### Knob 1: B₀ Inhomogeneity

```
x_mod(x,y) = x(x,y) · exp(i · 2π · B0_hz · TE · b0_map(x,y))
```

- **b0_map**: smooth spatial map in [-1, 1], stored as `B0_map(320,320)` float32
- **B0_hz**: field inhomogeneity strength in Hz
- **TE**: echo time (25 ms in the benchmark)
- **Effect**: spatially varying phase errors → blurring, signal voids

#### Knob 2: Gradient Nonlinearity

```
x_warped(x,y) = x(x + dx(x,y), y + dy(x,y))
```

- **warp_field**: displacement field `(2, H, W)` in pixels, stored as `warp_field` float32
- **gradient_nonlin_frac**: controls the magnitude of the warp
- **Effect**: geometric distortion, especially at FOV edges

#### Knob 3: Coil Sensitivity Error

```
S_c_true = S_c_nominal · (1 + ε_c)
```

- **ε_c**: smooth perturbation field (complex), magnitude controlled by `coil_sensitivity_frac`
- **Effect**: SENSE ghosting, signal intensity errors

#### Knob 4: k-Trajectory Deviation

```
y_c_true(kx, ky) = y_c_ideal(kx, ky) · exp(i · 2π · k_traj_frac · ky · ramp)
```

- **k_traj_frac**: per-line phase ramp strength
- **Effect**: phase errors in k-space → ghosting, striping

### 2.3 Mismatch Severity by Tier

The `SPEC_RANGES` dictionary in `build_dataset.py` (lines 74–96) defines
the mismatch ranges per tier:

| Parameter | Public (Mild) | Dev (Moderate) | Hidden (Severe) |
|-----------|:------------:|:--------------:|:---------------:|
| B0 inhomogeneity (Hz) | 5 – 15 | 5 – 20 | 20 – 60 |
| Gradient nonlin (frac) | 0.001 – 0.003 | 0.001 – 0.005 | 0.005 – 0.02 |
| Coil sensitivity (frac) | 0.01 – 0.03 | 0.01 – 0.05 | 0.05 – 0.15 |
| k-trajectory (frac) | 0.001 – 0.003 | 0.001 – 0.005 | 0.005 – 0.02 |
| Noise σ (relative) | 0.01 – 0.02 | 0.01 – 0.03 | 0.03 – 0.06 |

**Key insight**: the hidden tier has 4–10× stronger mismatch than the
public tier. This means algorithms that perform well on clean data may
degrade significantly on the hidden tier — exposing their sensitivity to
model mismatch.

---

## 3. The Noise Model

### 3.1 Complex Gaussian Noise

MRI noise is well-modelled as i.i.d. complex Gaussian in k-space:

```
n_c(kx, ky) ~ CN(0, σ²)
```

The noise level σ is set relative to the signal:

```
σ = noise_sigma · std(|y_c[sampled]|)
```

where `noise_sigma` is the parameter from SPEC_RANGES (0.01 – 0.06 across
tiers).

### 3.2 Noise in Image Domain

After inverse FFT, the noise becomes spatially correlated. The noise
variance in the RSS-combined image is:

```
σ_image ≈ σ_kspace / √(N_sampled)
```

where N_sampled is the number of sampled k-space lines.

---

## 4. Undersampling

### 4.1 Why Undersample?

A fully sampled 320×320 k-space with 15 coils requires 320 phase-encode
steps. At a TR of 5 ms per line, this takes 1.6 seconds per slice. For a
volumetric scan with 200 slices, the total time is ~5 minutes.

With 4× acceleration: only 80 PE lines are acquired, reducing scan time
to ~1.3 minutes. But this creates an underdetermined inverse problem.

### 4.2 Variable-Density Cartesian Sampling

The PWM benchmark uses a **variable-density** random Cartesian mask:

```
P(sample ky line) ∝ 1 / (1 + |ky - ky_centre|)
```

- Central lines are always sampled (ACS region = 8% of k-space)
- Outer lines are sampled with probability decreasing with distance from
  centre
- Total acceleration factor R = 4

The mask is generated by `generate_vds_mask()` in `build_dataset.py`
(line 457) and stored as a 1D uint8 array `mask(320,)`.

### 4.3 The ACS Region

The **auto-calibration signal** (ACS) region is the fully sampled centre
of k-space. It serves two purposes:

1. **Coil sensitivity estimation**: ESPIRiT, JSENSE
2. **GRAPPA kernel calibration**: self-consistent k-space interpolation

In the PWM benchmark: ACS = 8% of 320 = 26 lines (indices ~147–173).

---

## 5. Why Reconstruction is Ill-Posed

### 5.1 The Fundamental Problem

With 4× undersampling, we have:
- **Unknowns**: 320 × 320 = 102,400 pixel values
- **Equations**: 15 coils × 80 sampled lines × 320 = 384,000 complex
  measurements

Despite having more measurements than unknowns (overdetermined for each
pixel), the problem is ill-conditioned because:
1. The coil sensitivities are not orthogonal — many measurements are
   linearly dependent
2. Noise amplifies errors in the missing k-space lines
3. Model mismatch means A_true ≠ A_assumed

### 5.2 Condition Number

The condition number κ(A) of the encoding matrix determines noise
amplification:

```
||x_hat - x_true|| / ||x_true|| ≤ κ(A) · ||n|| / ||y||
```

For undersampled MRI, κ(A) >> 1, especially for:
- High acceleration factors (R = 8+)
- Few coils
- Low-rank coil sensitivity profiles
- Missing outer k-space lines

---

## 6. Regularisation Strategies

To stabilise the inverse problem, we add prior information:

### 6.1 Tikhonov Regularisation (L2)

```
min_x  ||y - Ax||² + λ ||x||²
```

- Closed-form solution: x = (A^H A + λI)^(-1) A^H y
- Efficient via conjugate gradient (CG) — this is essentially what SENSE does
- Pro: stable, fast convergence
- Con: over-smooths edges

### 6.2 Sparsity Regularisation (L1)

```
min_x  ||y - Ax||² + λ ||Ψx||₁
```

where Ψ is a sparsifying transform (e.g. wavelets).

- Solved by FISTA or ADMM — this is CS-MRI
- Pro: preserves edges, exploits natural image sparsity
- Con: can produce staircase artefacts

### 6.3 Learned Regularisation (Deep Learning)

Replace the handcrafted regulariser with a neural network:

```
min_x  ||y - Ax||² + λ ||x - D_θ(x)||²
```

- **PnP**: use a pre-trained denoiser D_θ as an implicit prior
- **VarNet/MoDL**: end-to-end unrolled optimisation with learned
  parameters θ
- Pro: state-of-the-art quality
- Con: requires training data, may fail on out-of-distribution inputs

---

## 7. Connection to the PWM Framework

The PWM benchmark is specifically designed to stress-test these
regularisation strategies by introducing model mismatch:

1. **Classical methods** (SENSE, CS-MRI) assume the nominal forward model
   is exact. Mismatch directly corrupts their solutions.
2. **Learned methods** (VarNet, MoDL) are trained on specific data
   distributions. Out-of-distribution mismatch degrades their performance.
3. **PnP methods** separate the forward model from the prior. They can
   potentially adapt to mismatch by using the true forward model, but
   still require accurate knowledge of A.

The benchmark measures how gracefully each algorithm degrades as mismatch
severity increases from public (mild) to hidden (severe).

---

## 8. Summary

| Concept | Mathematical Form |
|---------|------------------|
| Forward model | y = MFSx + n |
| Mismatch model | y = M · k_err(F(S_true · warp(x) · B0_phase)) + n |
| Noise model | n_c ~ CN(0, σ²), σ = noise_sigma · std(\|y\|) |
| Undersampling | R = 4, VDS Cartesian, 8% ACS |
| Regularised inverse | min \|\|y - Ax\|\|² + λ R(x) |

---

*Previous: [01 — MRI Physics Fundamentals](01_mri_physics_fundamentals.md)*
*Next: [03 — Reconstruction Algorithms](03_reconstruction_algorithms.md)*
