# CASSI — Mismatch Correction + Reconstruction

> **Use Case 2: Correct operator mismatch, then reconstruct**
> Primary mismatch: **Dispersion step error**

---

## Mismatch Overview

CASSI reconstruction requires knowing the exact pixel dispersion step Δ (pixels per wavelength band).
Manufacturing tolerances in the prism cause Δ to differ from the nominal design value.

| Mismatch Source | Parameter | Range | Typical Error | Correction Method |
|----------------|-----------|-------|---------------|-------------------|
| **Dispersion step** | `step` (pixels/band) | [0.7, 1.3] × nominal | ±0.1 px/band | Grid search on reconstruction quality |
| Mask registration | `shift_x`, `shift_y` | [-3, +3] px | ±1 px | Cross-correlation with calibration target |
| Background | `dark_current` | [0, 0.05] | 0.01 | Dark frame subtraction |

---

## Run Button

```python
# ============================================================
# CASSI Mismatch Correction — PWM Run Button
# ============================================================
import sys, os
BASE = os.path.expanduser('~/Physics_World_Model/pwm/public')
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'packages/pwm_core'))

import numpy as np, matplotlib.pyplot as plt
from pwm_core.mismatch.operators import (
    cassi_forward, cassi_gap_denoise, cassi_calibrate_step, compute_psnr
)

H, W, n_bands = 64, 64, 28
rng = np.random.default_rng(42)
mask = rng.integers(0, 2, (H, W)).astype(np.float32)
x_true = rng.random((H, W, n_bands)).astype(np.float32)

TRUE_STEP = 1.0     # True dispersion step (pixels/band)
WRONG_STEP = 1.2    # What algorithm assumes (wrong!)

# Simulate with true step
y = cassi_forward(x_true, mask, step=TRUE_STEP)

# -------------------------------------------------------
# User-provided mismatch parameters
# -------------------------------------------------------
KNOWN_STEP = None        # Set to float if known, None to auto-search
STEP_RANGE = [0.7, 1.3]  # Search range
STEP_STEPS = 20          # Grid points

# -------------------------------------------------------
# Calibrate
# -------------------------------------------------------
if KNOWN_STEP is not None:
    best_step = KNOWN_STEP
else:
    print("Calibrating dispersion step ...")
    best_step = cassi_calibrate_step(y, mask, step_range=STEP_RANGE, n_steps=STEP_STEPS)
    print(f"Estimated step: {best_step:.3f} (true: {TRUE_STEP:.3f})")

# Reconstruct
x_wrong = cassi_gap_denoise(y, mask, step=WRONG_STEP, iters=50)
x_corrected = cassi_gap_denoise(y, mask, step=best_step, iters=50)

psnr_wrong = compute_psnr(x_true.mean(-1), x_wrong)
psnr_fixed = compute_psnr(x_true.mean(-1), x_corrected)
print(f"PSNR (wrong step={WRONG_STEP}): {psnr_wrong:.2f} dB")
print(f"PSNR (corrected step={best_step:.2f}): {psnr_fixed:.2f} dB")
```

---

## References

- **GAP**: Liao et al., ICIP 2014
- **Mismatch DB**: `packages/pwm_core/contrib/mismatch_db.yaml`
