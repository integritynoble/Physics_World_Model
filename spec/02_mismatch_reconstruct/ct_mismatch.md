# CT — Mismatch Correction + Reconstruction

> **Use Case 2: Correct operator mismatch, then reconstruct**
> Primary mismatch: **Center-of-Rotation (CoR) offset**

---

## Mismatch Overview

In CT, the **center-of-rotation** (CoR) must be precisely aligned with the detector center.
Mechanical misalignment causes **ring artifacts** in the reconstruction.

| Mismatch Source | Parameter | Range | Typical Error | Correction Method |
|----------------|-----------|-------|---------------|-------------------|
| **CoR offset** | `cor_offset` | [-5, +5] px | ±0.5 px | Cross-correlation (0°/180° projections) |
| Beam hardening | `bh_coeff` | [0, 0.1] | 0.02 | 2nd-order polynomial linearization |
| Scatter | `scatter_frac` | [0, 0.3] | 0.05 | Scatter kernel subtraction |
| Ring artifacts | `ring_width` | [0, 3] px | 1 px | Wavelet-based ring removal |
| Motion blur | `motion_mm` | [0, 2] mm | 0.5 mm | Sinogram smoothing |

### Effect of CoR Mismatch on Reconstruction

| CoR Offset | PSNR Loss | Artifact Type |
|-----------|-----------|---------------|
| 0.0 px | 0 dB (no artifact) | Clean |
| 0.5 px | ~2 dB | Slight blurring |
| 1.0 px | ~5 dB | Noticeable double edges |
| 2.0 px | ~10 dB | Strong ring artifact |
| 5.0 px | ~15 dB | Severe rings / unusable |

---

## Mismatch Parameters

Users can provide:
- **Measurement** `y`: sinogram (n_angles, n_det) — your CT data
- **CoR start value**: initial guess for CoR (default: detector center = n_det/2)
- **Search range**: range of CoR values to test (default: ±5 pixels)
- **Ground truth** (optional): for PSNR/SSIM evaluation

---

## Correction Method: Cross-Correlation

The CoR is estimated by cross-correlating projections at 0° and 180°:

```
P_0(t) = projection at angle 0°
P_180(t) = projection at angle 180°
CoR = argmax_c  cross_corr(P_0, flip(P_180))
```

This is fast (~milliseconds) and accurate to sub-pixel precision.

---

## Run Button

```python
# ============================================================
# CT Mismatch Correction + Reconstruction — PWM Run Button
# ============================================================
import sys, os
BASE = os.path.expanduser('~/Physics_World_Model/pwm/public')
# Colab: BASE = '/content/Physics_World_Model/pwm/public'
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'packages/pwm_core'))

import numpy as np
import matplotlib.pyplot as plt
from pwm_core.mismatch.operators import (
    ct_radon_forward, ct_sart_tv_recon, ct_calibrate_cor, compute_psnr
)

# -------------------------------------------------------
# 1. Prepare data
# -------------------------------------------------------
# Option A: PWM benchmark data
# import gcsfs, h5py
# fs = gcsfs.GCSFileSystem(token='anon')
# with fs.open('gs://pwm-benchmark-datasets/datasets/Benchmark/ct/public/ct_public.h5') as f:
#     with h5py.File(f, 'r') as hf:
#         x_true = hf['x_true'][0]; has_gt = True
# # Re-simulate with known mismatch:
# y_mismatch = ct_radon_forward(x_true, n_angles=180, cor_offset=YOUR_COR_OFFSET)

# Option B: Synthetic phantom with known mismatch
from skimage.data import shepp_logan_phantom
x_true = shepp_logan_phantom().astype(np.float32)
TRUE_COR_OFFSET = 2.5   # pixels — your known or estimated mismatch

print(f"Simulating CT with CoR offset = {TRUE_COR_OFFSET:.2f} px ...")
y_mismatch = ct_radon_forward(x_true, n_angles=180, cor_offset=TRUE_COR_OFFSET)
print(f"Sinogram shape: {y_mismatch.shape}")

# -------------------------------------------------------
# 2. User-provided mismatch parameters (customize here)
# -------------------------------------------------------
# Provide the known offset if you know it, or set to None to auto-calibrate
KNOWN_COR_OFFSET = None   # Set to float (e.g. 2.5) if known, or None to estimate

# Search range for grid search
COR_SEARCH_RANGE = 5.0    # ± pixels
SEARCH_STEPS     = 50     # number of grid points

# -------------------------------------------------------
# 3. Calibrate CoR
# -------------------------------------------------------
if KNOWN_COR_OFFSET is not None:
    cor_corrected = KNOWN_COR_OFFSET
    print(f"Using provided CoR offset: {cor_corrected:.3f} px")
else:
    print("Estimating CoR via cross-correlation ...")
    cor_corrected = ct_calibrate_cor(y_mismatch)
    print(f"Estimated CoR offset: {cor_corrected:.3f} px  (true: {TRUE_COR_OFFSET:.3f})")

# -------------------------------------------------------
# 4. Reconstruct with and without correction
# -------------------------------------------------------
print("\nReconstructing WITHOUT mismatch correction ...")
x_no_corr = ct_sart_tv_recon(y_mismatch, cor_offset=0.0)

print("Reconstructing WITH mismatch correction ...")
x_corrected = ct_sart_tv_recon(y_mismatch, cor_offset=cor_corrected)

# -------------------------------------------------------
# 5. Evaluate
# -------------------------------------------------------
has_gt = True
if has_gt:
    psnr_no_corr  = compute_psnr(x_true, x_no_corr)
    psnr_corrected = compute_psnr(x_true, x_corrected)
    print(f"\nPSNR without correction: {psnr_no_corr:.2f} dB")
    print(f"PSNR with correction:    {psnr_corrected:.2f} dB  (+{psnr_corrected-psnr_no_corr:.2f} dB improvement)")

# -------------------------------------------------------
# 6. Visualize
# -------------------------------------------------------
fig, axes = plt.subplots(1, 3 if has_gt else 2, figsize=(15 if has_gt else 10, 5))
axes[0].imshow(x_no_corr, cmap='gray')
axes[0].set_title(f'No CoR correction\nPSNR: {psnr_no_corr:.1f} dB' if has_gt else 'No correction')
axes[1].imshow(x_corrected, cmap='gray')
axes[1].set_title(f'CoR corrected (Δ={cor_corrected:.2f}px)\nPSNR: {psnr_corrected:.1f} dB' if has_gt else f'Corrected (Δ={cor_corrected:.2f}px)')
if has_gt:
    axes[2].imshow(x_true, cmap='gray'); axes[2].set_title('Ground Truth')
plt.tight_layout()
plt.savefig('ct_mismatch_correction.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: ct_mismatch_correction.png")
```

---

## Expected Output

| Scenario | PSNR |
|----------|------|
| No mismatch (perfect CoR) | ~38 dB (SART-TV) |
| CoR mismatch 2.5 px, no correction | ~26 dB |
| CoR mismatch 2.5 px, after correction | ~37 dB (+11 dB) |

---

## Additional Mismatch Corrections

### Beam Hardening Correction
```python
# Apply 2nd-order polynomial pre-correction
bh_coeff = 0.05  # Typically 0.02–0.08 for clinical CT
y_corrected = y - bh_coeff * y**2
```

### Scatter Subtraction
```python
from scipy.ndimage import gaussian_filter
scatter_frac = 0.05
y_scatter_corrected = y - scatter_frac * gaussian_filter(y, sigma=20)
```

### Ring Artifact Removal
```python
# Wavelet ring removal (apply to sinogram before FBP)
import pywt
# ... wavelet-based sinogram stripe removal
```

---

## References

- **CoR calibration**: Donath et al., JOSA A 2006
- **Beam hardening**: Joseph & Spital, Med. Phys. 1978; Kijewski & Bjärngard 1978
- **Scatter correction**: Zhu et al., Med. Phys. 2009
- **Ring removal**: Münch et al., Optics Express 2009
