# Lensless Imaging — Mismatch Correction + Reconstruction

> **Use Case 2: Correct operator mismatch, then reconstruct**
> Primary mismatch: **PSF shift** (spatial translation of calibrated PSF)

---

## Mismatch Overview

Lensless cameras require an accurate PSF (point spread function) calibrated from a point source.
Thermal drift or mechanical vibration shift the PSF relative to the calibration position.

| Mismatch Source | Parameter | Range | Typical Error | Correction Method |
|----------------|-----------|-------|---------------|-------------------|
| **PSF shift x** | `shift_x` (pixels) | [-5, +5] | ±1 px | Gradient calibration |
| **PSF shift y** | `shift_y` (pixels) | [-5, +5] | ±1 px | Gradient calibration |
| Background | `background` | [0, 0.05] | 0.01 | Flat-field correction |
| Gain | `gain` | [0.9, 1.1] | ±0.02 | Pixel-wise calibration |

---

## Run Button

```python
import sys, os
BASE = os.path.expanduser('~/Physics_World_Model/pwm/public')
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'packages/pwm_core'))

import numpy as np, matplotlib.pyplot as plt
from pwm_core.mismatch.operators import (
    lensless_forward, lensless_admm_tv, lensless_calibrate_shift, compute_psnr
)

H, W = 256, 256
rng = np.random.default_rng(42)

# True PSF (calibrated)
psf = np.zeros((H, W), np.float32)
psf[H//2, W//2] = 1.0
from scipy.ndimage import gaussian_filter
psf = gaussian_filter(psf, sigma=8); psf /= psf.sum()

# Scene
from skimage.data import camera
x_true = camera().astype(np.float32)[:H, :W] / 255.0

# Simulate with PSF shift mismatch
TRUE_SHIFT = (2.3, -1.7)   # True PSF shift in (y, x)
psf_shifted = np.roll(np.roll(psf, int(TRUE_SHIFT[0]), axis=0), int(TRUE_SHIFT[1]), axis=1)
y = lensless_forward(x_true, psf_shifted)

# -------------------------------------------------------
# User parameters
# -------------------------------------------------------
KNOWN_SHIFT = None      # Set to (shift_y, shift_x) if known, None to auto-calibrate
SHIFT_RANGE = 5         # ± pixels to search

# -------------------------------------------------------
# Calibrate PSF shift
# -------------------------------------------------------
if KNOWN_SHIFT is not None:
    shift_est = KNOWN_SHIFT
else:
    print("Calibrating PSF shift via gradient search ...")
    shift_est = lensless_calibrate_shift(y, psf, shift_range=SHIFT_RANGE)
    print(f"Estimated shift: ({shift_est[0]:.2f}, {shift_est[1]:.2f})  (true: {TRUE_SHIFT})")

# Reconstruct with wrong and corrected PSF
psf_wrong = psf   # No shift correction
psf_corrected = np.roll(np.roll(psf, round(shift_est[0]), axis=0), round(shift_est[1]), axis=1)

x_wrong = lensless_admm_tv(y, psf_wrong, iters=50, lam=0.01)
x_corrected = lensless_admm_tv(y, psf_corrected, iters=50, lam=0.01)

psnr_w = compute_psnr(x_true, x_wrong)
psnr_c = compute_psnr(x_true, x_corrected)
print(f"PSNR (no shift correction): {psnr_w:.2f} dB")
print(f"PSNR (shift corrected):     {psnr_c:.2f} dB  (+{psnr_c-psnr_w:.2f} dB)")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(x_wrong, cmap='gray'); axes[0].set_title(f'No correction\nPSNR={psnr_w:.1f}dB')
axes[1].imshow(x_corrected, cmap='gray'); axes[1].set_title(f'Shift corrected\nPSNR={psnr_c:.1f}dB')
axes[2].imshow(x_true, cmap='gray'); axes[2].set_title('Ground Truth')
plt.tight_layout(); plt.savefig('lensless_mismatch.png', dpi=150); plt.show()
```

---

## References

- **PSF calibration**: Antipa et al., "DiffuserCam", Optica 2018
- **Gradient calibration**: Boominathan et al., Optica 2022
