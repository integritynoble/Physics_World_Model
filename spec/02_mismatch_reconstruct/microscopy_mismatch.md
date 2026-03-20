# Fluorescence Microscopy — Mismatch Correction + Reconstruction

> **Use Case 2: Correct operator mismatch, then reconstruct**
> Applies to: `widefield`, `widefield_lowdose`, `confocal_3d`, `confocal_livecell`
> Primary mismatch: **PSF width (σ)**, **defocus**, **background**

---

## Mismatch Overview

Fluorescence microscopy reconstructions assume a known PSF. Errors in the PSF cause
over-sharpening or blurring artifacts.

| Mismatch Source | Parameter | Range | Typical Error | Correction Method |
|----------------|-----------|-------|---------------|-------------------|
| **PSF sigma** | `psf_sigma` (pixels) | [0.5, 3.0] | ±0.3 px | Grid search / Blind deconvolution |
| **Defocus** | `defocus` (µm) | [-2, +2] | ±0.5 µm | z-stack refocusing |
| **Background** | `background` (norm.) | [0, 0.15] | 0.03 | Constant background subtraction |
| **Gain** | `gain` (unitless) | [0.5, 1.5] | ±0.1 | Flat-field correction |

### Correction Method: Grid Search

1. Generate a grid of candidate PSF sigma values: e.g., [0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
2. For each candidate, run deconvolution and compute image sharpness metric
3. Select sigma that maximizes the metric (sparsity, gradient, or PSNR if GT available)
4. Reconstruct with the calibrated PSF

---

## Mismatch Parameters (User-Provided)

```yaml
# Example mismatch configuration
modality: widefield
psf_sigma:
  start: 1.0           # initial guess (pixels)
  range: [0.5, 3.0]    # search range
  steps: 20            # grid resolution
defocus:
  start: 0.0           # initial guess (µm)
  range: [-2.0, 2.0]
  steps: 10
background:
  value: 0.03          # known background level (normalized)
```

---

## Run Button

```python
# ============================================================
# Widefield Microscopy Mismatch Correction — PWM Run Button
# ============================================================
import sys, os
BASE = os.path.expanduser('~/Physics_World_Model/pwm/public')
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'packages/pwm_core'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# -------------------------------------------------------
# 1. Simulate blurred microscopy image with mismatch
# -------------------------------------------------------
from skimage.data import camera
x_true = camera().astype(np.float32) / 255.0   # (512, 512)

TRUE_SIGMA = 2.0          # True PSF sigma
WRONG_SIGMA = 0.8         # What the algorithm assumes (wrong!)
BACKGROUND = 0.03         # Additive background

# Simulate measurement
y = gaussian_filter(x_true, sigma=TRUE_SIGMA) + BACKGROUND
y += 0.01 * np.random.randn(*y.shape).astype(np.float32)

# -------------------------------------------------------
# 2. User-provided mismatch parameters
# -------------------------------------------------------
# If you know the PSF sigma, set it directly:
KNOWN_PSF_SIGMA = None        # Set to float if known, None to auto-search

# Search parameters (used if KNOWN_PSF_SIGMA is None)
SIGMA_MIN, SIGMA_MAX = 0.5, 4.0
SIGMA_STEPS = 20

# Background level (provide or set to None to estimate)
KNOWN_BACKGROUND = None   # Set to float (e.g. 0.03) or None to estimate

# -------------------------------------------------------
# 3. Estimate background
# -------------------------------------------------------
if KNOWN_BACKGROUND is not None:
    bg = KNOWN_BACKGROUND
else:
    # Simple estimate: median of corner regions
    corners = np.concatenate([y[:20,:20].flat, y[:20,-20:].flat,
                               y[-20:,:20].flat, y[-20:,-20:].flat])
    bg = float(np.median(corners))
    print(f"Estimated background: {bg:.4f}")

y_bg_corrected = np.clip(y - bg, 0, None)

# -------------------------------------------------------
# 4. PSF calibration via grid search
# -------------------------------------------------------
def richardson_lucy(y, psf_sigma, n_iter=30):
    """Simple Richardson-Lucy deconvolution."""
    from scipy.signal import fftconvolve
    psf = np.zeros_like(y); h, w = y.shape
    psf[h//2, w//2] = 1.0
    psf = gaussian_filter(psf, psf_sigma)
    x = np.ones_like(y) * y.mean()
    for _ in range(n_iter):
        conv = gaussian_filter(x, psf_sigma) + 1e-10
        ratio = y / conv
        x *= gaussian_filter(ratio, psf_sigma)
    return np.clip(x, 0, None)

def sharpness(img):
    """Gradient-based sharpness metric."""
    gx = np.diff(img, axis=1); gy = np.diff(img, axis=0)
    return float(np.mean(gx**2) + np.mean(gy**2))

if KNOWN_PSF_SIGMA is not None:
    best_sigma = KNOWN_PSF_SIGMA
    print(f"Using provided PSF sigma: {best_sigma:.3f}")
else:
    print("Calibrating PSF sigma via grid search ...")
    sigmas = np.linspace(SIGMA_MIN, SIGMA_MAX, SIGMA_STEPS)
    scores = []
    for s in sigmas:
        x_deconv = richardson_lucy(y_bg_corrected, s, n_iter=20)
        scores.append(sharpness(x_deconv))
    best_sigma = sigmas[np.argmax(scores)]
    print(f"Best PSF sigma: {best_sigma:.3f} (true: {TRUE_SIGMA:.3f})")

# -------------------------------------------------------
# 5. Final reconstruction
# -------------------------------------------------------
x_wrong = richardson_lucy(y_bg_corrected, WRONG_SIGMA, n_iter=50)
x_corrected = richardson_lucy(y_bg_corrected, best_sigma, n_iter=50)

# -------------------------------------------------------
# 6. Evaluate & Visualize
# -------------------------------------------------------
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
ref_max = x_true.max()
psnr_wrong = psnr_fn(x_true, np.clip(x_wrong, 0, ref_max), data_range=ref_max)
psnr_fixed = psnr_fn(x_true, np.clip(x_corrected, 0, ref_max), data_range=ref_max)
print(f"PSNR (wrong sigma={WRONG_SIGMA:.1f}): {psnr_wrong:.2f} dB")
print(f"PSNR (corrected sigma={best_sigma:.2f}): {psnr_fixed:.2f} dB  (+{psnr_fixed-psnr_wrong:.2f} dB)")

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(y, cmap='gray'); axes[0].set_title(f'Measurement\n(σ_true={TRUE_SIGMA})')
axes[1].imshow(x_wrong, cmap='gray'); axes[1].set_title(f'Wrong σ={WRONG_SIGMA:.1f}\nPSNR={psnr_wrong:.1f}dB')
axes[2].imshow(x_corrected, cmap='gray'); axes[2].set_title(f'Corrected σ={best_sigma:.2f}\nPSNR={psnr_fixed:.1f}dB')
axes[3].imshow(x_true, cmap='gray'); axes[3].set_title('Ground Truth')
plt.tight_layout()
plt.savefig('microscopy_mismatch_correction.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## Expected Output

| Scenario | PSNR |
|----------|------|
| Correct PSF sigma (no mismatch) | ~34 dB |
| Wrong sigma (0.8 vs 2.0), no correction | ~25 dB |
| Grid search corrected sigma | ~32 dB (+7 dB) |

---

## References

- **Mismatch database**: `packages/pwm_core/contrib/mismatch_db.yaml`
- **Richardson-Lucy**: Richardson, JOSA 1972; Lucy, Astron J 1974
- **Blind deconvolution**: Levin et al., CVPR 2009
- **PWM paper**: System Design paper (see `papers/system_design/`)
