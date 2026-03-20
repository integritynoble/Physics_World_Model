# 03 — Reconstruction Algorithms

This file covers the six reconstruction algorithms implemented in the PWM
codebase. For each we give the mathematical formulation, key insight,
implementation reference, and practical notes.

---

## Algorithm Comparison Table

| # | Algorithm | Type | Multi-Coil | Iterations | GPU | Expected PSNR (4×) |
|---|-----------|------|:----------:|:----------:|:---:|:-------------------:|
| 1 | Zero-Filled RSS | Direct | Native | 0 | No | 20–27 dB |
| 2 | SENSE | Iterative (CG) | Native | 30 | Optional | 25–35 dB |
| 3 | CS-MRI | Iterative (FISTA) | Via SENSE | 50 | Optional | 25–35 dB |
| 4 | PnP-HQS | Iterative (HQS) | Via adapter | 30 | Optional | 25–35 dB |
| 5 | VarNet | Learned (unrolled) | Single-coil* | 12 cascades | Yes | 20–28 dB† |
| 6 | MoDL | Learned (unrolled) | Single-coil* | 5 unrolls | Yes | 20–28 dB† |

*Single-coil input via coil combination. †With random weights (no
pretrained checkpoint).

---

## 1. Zero-Filled RSS

### Mathematical Formulation

The simplest reconstruction: inverse FFT each coil, then combine via
root-sum-of-squares:

```
x_c = F^(-1)(y_c)                    # per-coil image (with aliasing)
x_hat = √(Σ_c |x_c|²)              # RSS combination
```

Missing k-space entries are treated as zero (hence "zero-filled").

### Key Insight

No iterative solve — just inverse FFT and magnitude combination. Serves as
the baseline that all other methods must beat.

### Implementation

```python
from pwm_core.recon.mri_solvers import zero_filled_reconstruction

x_hat = zero_filled_reconstruction(
    kspace,      # (n_coils, H, W) complex — multi-coil: returns RSS
    mask=None,   # optional, not used internally for single-coil path
    device=None  # None = auto-detect
)
# Returns: (H, W) float32
```

Source: `mri_solvers.py:368`

### Pros / Cons

- **Pro**: instantaneous, no parameters, always works
- **Con**: aliasing artefacts at R > 1, no noise reduction

---

## 2. SENSE (SENSitivity Encoding)

### Mathematical Formulation

SENSE solves the regularised least-squares problem using conjugate gradient
(CG):

```
min_x  ||y - MFSx||² + λ||x||²
```

Normal equations: `(A^H A + λI) x = A^H y`

where `A = MFS` is the multi-coil encoding operator.

CG iteration:
```
r₀ = A^H y - (A^H A + λI) x₀
p₀ = r₀
for k = 0, 1, ..., K-1:
    α_k = (r_k^H r_k) / (p_k^H (A^H A + λI) p_k)
    x_{k+1} = x_k + α_k p_k
    r_{k+1} = r_k - α_k (A^H A + λI) p_k
    β_k = (r_{k+1}^H r_{k+1}) / (r_k^H r_k)
    p_{k+1} = r_{k+1} + β_k p_k
```

### Key Insight

Uses the coil sensitivity maps to "unfold" aliased images. The
sensitivity diversity between coils provides the missing information.
More coils → better SENSE performance.

### Implementation

```python
from pwm_core.recon.mri_solvers import sense_reconstruction

x_hat = sense_reconstruction(
    kspace,           # (n_coils, H, W) complex64
    sensitivity_maps, # (n_coils, H, W) complex64
    mask,             # (H, W) float32 — 2D mask
    regularization=0.001,
    iterations=30,
    device=None
)
# Returns: (H, W) complex64 — take np.abs() for magnitude
```

Source: `mri_solvers.py:121`

Note: the mask input here is 2D `(H, W)`. Convert the 1D mask:
```python
mask_2d = mask_1d.astype(np.float32).reshape(-1, 1) * np.ones((1, W))
```

### Pros / Cons

- **Pro**: native multi-coil, well-understood theory, fast convergence
- **Con**: sensitive to coil map errors, Tikhonov regularisation over-smooths

---

## 3. CS-MRI (Compressed Sensing MRI)

### Mathematical Formulation

Promotes sparsity in the wavelet domain via L1 regularisation:

```
min_x  ||y - Ax||² + λ ||Ψx||₁
```

Solved by FISTA (Fast Iterative Shrinkage-Thresholding Algorithm):

```
z_{k+1} = prox_{λ/L · ||Ψ·||₁}(x_k - (1/L) A^H(Ax_k - y))
t_{k+1} = (1 + √(1 + 4t_k²)) / 2
x_{k+1} = z_{k+1} + ((t_k - 1) / t_{k+1}) (z_{k+1} - z_k)
```

The proximal operator for L1 is soft-thresholding:

```
prox_{τ ||·||₁}(v) = sign(v) · max(|v| - τ, 0)
```

### Key Insight

Natural images are sparse in wavelet domain — most coefficients are
near zero. L1 regularisation promotes this sparsity, enabling
reconstruction from fewer measurements than Nyquist requires.

### Implementation

```python
from pwm_core.recon.mri_solvers import cs_mri_wavelet

x_hat = cs_mri_wavelet(
    kspace,                # (H,W) single-coil or (n_coils,H,W) multi-coil
    mask,                  # sampling mask
    lam=0.01,              # sparsity weight λ
    iterations=50,         # FISTA iterations
    sensitivity_maps=None, # for multi-coil input
    device=None
)
# Returns: (H, W) complex64
```

Source: `mri_solvers.py:252`

**Multi-coil behaviour**: if `kspace.ndim == 3` and `sensitivity_maps` is
None, the function auto-estimates sensitivity maps and delegates to
`sense_reconstruction` internally (line 279).

### Pros / Cons

- **Pro**: better edge preservation than SENSE, principled sparsity prior
- **Con**: slow (50+ iterations), wavelet basis may not be optimal for all
  structures

---

## 4. PnP-HQS (Plug-and-Play Half-Quadratic Splitting)

### Mathematical Formulation

PnP decouples the data-fidelity term from the regulariser by introducing
an auxiliary variable:

```
min_{x,z}  ||y - Ax||² + ρ||x - z||²      s.t.  z = D(x)
```

Half-quadratic splitting alternates:

```
# x-update (data fidelity + proximity to z)
x_{k+1} = argmin_x  ||y - Ax||² + ρ||x - z_k||²
         ≈ x_k - step · (A^H(Ax_k - y) + ρ(x_k - z_k))   # gradient steps

# z-update (denoising)
z_{k+1} = D_σ(x_{k+1})     # apply denoiser at noise level σ
```

The noise level σ decays across iterations (σ_{k+1} = σ_k · decay).

### Key Insight

Any image denoiser can serve as an implicit regulariser — no need to
define an explicit prior. The denoiser encodes complex image statistics
that are hard to specify analytically.

### Denoiser Cascade

The `get_denoiser()` function (line 165 in `pnp.py`) tries denoisers in
order of quality:

| Priority | Denoiser | Requirements | Quality |
|----------|----------|-------------|---------|
| 1 | DRUNet | PyTorch + deepinv | Best |
| 2 | BM3D | bm3d package | Very good |
| 3 | NLM | scikit-image | Good |
| 4 | Gaussian | scipy (always available) | Baseline |

### Implementation

```python
from pwm_core.recon.pnp import pnp_hqs, get_denoiser

denoiser = get_denoiser("auto", device="cpu")

x_hat = pnp_hqs(
    y,            # measurements (flattened or structured)
    forward,      # A(x) → y callable
    adjoint,      # A^H(y) → x callable
    x_shape,      # (H, W)
    denoiser,     # callable(x, sigma) → denoised_x
    iters=30,
    rho=1.0,      # proximity weight
    sigma=0.1,    # initial denoiser noise level
    sigma_decay=0.9
)
# Returns: (H, W) float32
```

Source: `pnp.py:277`

For multi-coil MRI, you need to wrap coil_maps and mask into forward/
adjoint callables (see `MultiCoilMRIOp` in the benchmark script).

### Pros / Cons

- **Pro**: flexible prior (any denoiser), no training on MRI data needed
- **Con**: convergence not guaranteed for all denoisers, requires tuning
  of ρ, σ, decay

---

## 5. VarNet (Variational Network)

### Mathematical Formulation

VarNet is an end-to-end learned unrolled optimisation network. Each
cascade k performs:

```
# Learned refinement in image domain
Δk_k = UNet_k(k_pred)

# Data consistency with learnable weight
k_{k+1} = k_k - η_k · (k_k - k_ref) · mask - Δk_k
```

where:
- `k_pred` is the current k-space estimate
- `k_ref` is the observed (zero-filled) k-space
- `UNet_k` is a small U-Net (18 base channels, 3 levels)
- `η_k` is a learnable DC weight per cascade
- 12 cascades total

Final output: `x = |F^(-1)(k_final)|` (magnitude).

### Key Insight

Rather than hand-designing the optimisation algorithm, VarNet *learns*
the update rule from data. Each cascade mimics one iteration of a
variational method, but with learned components.

### Architecture Summary

| Component | Details |
|-----------|---------|
| Sensitivity model | 8 base channels, estimates S from ACS |
| Cascade UNet | 18 base channels, 3 encoder/decoder levels |
| Cascades | 12 (each with separate UNet weights) |
| DC weight | 1 learnable scalar per cascade |
| Input | k-space (H,W,2) real-imag stack + mask |
| Output | magnitude image (H,W) float |

### Implementation

```python
from pwm_core.recon.varnet import varnet_recon

x_hat = varnet_recon(
    kspace,           # (H,W) complex — single-coil only
    mask,             # (H,W) or (W,) binary
    weights_path=None, # None → looks for .../weights/varnet/varnet.pth
    n_cascades=12,
    device=None       # None → cuda if available
)
# Returns: (H, W) float32
```

Source: `varnet.py:306`

**Important**: VarNet in the PWM codebase accepts **single-coil** input
only. For multi-coil data, combine first:

```python
# Coil-combine via RSS in image domain, then back to k-space
imgs = np.fft.ifft2(np.fft.ifftshift(y_kspace, axes=(-2,-1)), axes=(-2,-1))
rss = np.sqrt(np.sum(np.abs(imgs)**2, axis=0))
kspace_combined = np.fft.fftshift(np.fft.fft2(rss))
```

**No pretrained weights**: without a trained checkpoint, VarNet runs with
random initialisation. Results will be similar to zero-filled baseline.

### Pros / Cons

- **Pro**: state-of-the-art quality (when trained), end-to-end learned
- **Con**: requires large training dataset, single-coil only in PWM,
  domain shift degrades performance

---

## 6. MoDL (Model-Based Deep Learning)

### Mathematical Formulation

MoDL alternates between a learned denoiser and a closed-form data
consistency (DC) step:

```
for k = 1, ..., K:
    # Denoising step
    z_k = CNN_θ(x_k)

    # Data consistency (closed-form in k-space)
    x_{k+1} = F^(-1)((mask · y + λ · F(z_k)) / (mask + λ))
```

where:
- `CNN_θ` is a shared residual CNN (5 residual blocks, 64 channels)
- `λ` is a **learnable** regularisation parameter (initialised at 0.05)
- The DC step is exact (no iterative solve needed)

### Key Insight

The DC step has a closed-form solution in k-space because the sampling
operator M is diagonal in the Fourier domain. This makes each unroll very
efficient — no inner CG loop needed.

### Architecture Summary

| Component | Details |
|-----------|---------|
| Denoiser CNN | 5 ResBlocks, 64 channels, shared across iterations |
| Unrolls | 5 (default) |
| Lambda | Learnable nn.Parameter, init 0.05 |
| Input | k-space (H,W) complex or (H,W,2) + mask |
| Output | magnitude image (H,W) float |

### Implementation

```python
from pwm_core.recon.modl import modl_recon

x_hat = modl_recon(
    kspace,          # (H,W) complex — single-coil only
    mask,            # (H,W) or (W,) binary
    weights_path=None, # None → looks for .../weights/modl/modl.pth
    n_iter=5,
    device=None
)
# Returns: (H, W) float32
```

Source: `modl.py:229`

**Mask handling**: if `mask.ndim == 1`, it is tiled to `(H, W)`:
`mask_2d[ky, :] = mask_1d[ky]` for all kx.

**No pretrained weights**: same caveat as VarNet — random init gives
near-baseline results.

### Pros / Cons

- **Pro**: efficient DC step (no CG), shared weights reduce parameters
- **Con**: single-coil only in PWM, needs training data, λ must be learned

---

## 7. Multi-Coil Handling Summary

| Algorithm | Input | Multi-Coil Strategy |
|-----------|-------|---------------------|
| Zero-Filled RSS | (C,H,W) complex | Native: IFFT each coil, RSS combine |
| SENSE | (C,H,W) complex | Native: CG with coil sensitivities |
| CS-MRI | (C,H,W) or (H,W) | Auto-delegates to SENSE for multi-coil |
| PnP-HQS | forward/adjoint callables | Via MultiCoilMRIOp adapter class |
| VarNet | (H,W) complex | Requires pre-combination to single-coil |
| MoDL | (H,W) complex | Requires pre-combination to single-coil |

### Coil Combination for Single-Coil Solvers

```python
def coil_combine_rss(y_kspace):
    """Combine multi-coil k-space to single-coil via RSS."""
    imgs = np.fft.ifft2(np.fft.ifftshift(y_kspace, axes=(-2,-1)), axes=(-2,-1))
    rss = np.sqrt(np.sum(np.abs(imgs)**2, axis=0))
    kspace_combined = np.fft.fftshift(np.fft.fft2(rss))
    return kspace_combined  # (H,W) complex
```

---

## 8. When to Use What

| Scenario | Recommended Algorithm | Why |
|----------|-----------------------|-----|
| Quick look / baseline | Zero-Filled RSS | Instantaneous |
| Clinical parallel imaging | SENSE | Robust, well-understood |
| High acceleration (R ≥ 6) | CS-MRI | Sparsity helps at high R |
| Unknown imaging physics | PnP-HQS | Flexible denoiser prior |
| Abundant training data | VarNet or MoDL | Best quality (when trained) |
| Model mismatch present | PnP-HQS or SENSE | Less sensitive to distribution shift |

---

*Previous: [02 — MRI as an Inverse Problem](02_mri_as_inverse_problem.md)*
*Next: [04 — PWM MRI Benchmark](04_pwm_mri_benchmark.md)*
