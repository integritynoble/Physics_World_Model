# MRI — Mismatch Correction + Reconstruction

> **Use Case 2: Correct operator mismatch, then reconstruct**
> Primary mismatch: **Coil sensitivity error**, **B0 field drift**

---

## Mismatch Overview

MRI reconstruction (especially parallel imaging) requires accurate coil sensitivity maps.
Errors in these maps cause residual aliasing and noise amplification.

| Mismatch Source | Parameter | Range | Correction Method |
|----------------|-----------|-------|-------------------|
| Coil sensitivity error | `coil_snr_error` | [0, 0.3] | ESPIRiT (ACS auto-calibration) |
| B0 field drift | `b0_drift_hz` | [-50, +50] Hz | B0 field map estimation |
| Eddy current | `eddy_phase_error` | [-π/8, π/8] rad | Navigator correction |
| k-space trajectory | `grad_delay` | [-10, +10] µs | Trajectory calibration |
| Motion | `displacement_mm` | [0, 5] mm | Motion-corrected recon |

---

## Correction Method: ESPIRiT (Auto-Calibration Signal)

ESPIRiT estimates coil sensitivity maps from the fully-sampled **ACS region** (center of k-space):
1. Collect the center K ACS lines at full Nyquist (always sampled)
2. Compute calibration matrix from the ACS data
3. Solve eigenvalue problem → sensitivity maps
4. Use sensitivity maps in SENSE reconstruction

---

## Run Button

```python
# ============================================================
# MRI Mismatch Correction + Reconstruction — PWM Run Button
# ============================================================
import sys, os
BASE = os.path.expanduser('~/Physics_World_Model/pwm/public')
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'packages/pwm_core'))

import numpy as np
import matplotlib.pyplot as plt
from pwm_core.mismatch.operators import (
    mri_generate_coil_sensitivities, mri_forward_sense,
    mri_estimate_sensitivities_acs, mri_sense_recon, compute_psnr
)

# -------------------------------------------------------
# 1. Simulate MRI with coil sensitivity mismatch
# -------------------------------------------------------
H, W, n_coils = 128, 128, 4
rng = np.random.default_rng(42)

# True phantom
from skimage.data import shepp_logan_phantom
x_true = shepp_logan_phantom()[:128, :128].astype(np.complex64)

# True coil sensitivities
sens_true = mri_generate_coil_sensitivities(H, W, n_coils, seed=42)

# Simulate with mismatch: add noise to sensitivity maps
SENS_SNR = 0.1   # Noise level on sensitivity maps (0=perfect, 0.3=severe mismatch)
sens_wrong = sens_true + SENS_SNR * rng.standard_normal(sens_true.shape).astype(np.complex64)

# Undersampling mask (4× Cartesian)
mask = np.zeros((H, W), np.float32)
mask[::4, :] = 1.0; mask[H//2-8:H//2+8, :] = 1.0

# Simulate k-space with true sensitivities
y = mri_forward_sense(x_true, sens_true, mask)
print(f"k-space shape: {y.shape}")

# -------------------------------------------------------
# 2. User-provided mismatch parameters
# -------------------------------------------------------
# Option: provide known coil sensitivities if available
# sens_corrected = np.load('your_coil_maps.npy')

# Or: automatically estimate from ACS lines
print("Estimating coil sensitivities from ACS data ...")
sens_corrected = mri_estimate_sensitivities_acs(y, acs_lines=24)

# -------------------------------------------------------
# 3. Reconstruct with and without correction
# -------------------------------------------------------
x_wrong_sens = mri_sense_recon(y, sens_wrong, mask)
x_corrected  = mri_sense_recon(y, sens_corrected, mask)

# -------------------------------------------------------
# 4. Evaluate
# -------------------------------------------------------
x_true_real = np.abs(x_true)
psnr_wrong  = compute_psnr(x_true_real, np.abs(x_wrong_sens))
psnr_fixed  = compute_psnr(x_true_real, np.abs(x_corrected))
print(f"PSNR (wrong sensitivity): {psnr_wrong:.2f} dB")
print(f"PSNR (corrected):         {psnr_fixed:.2f} dB  (+{psnr_fixed-psnr_wrong:.2f} dB)")

# -------------------------------------------------------
# 5. Visualize
# -------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(np.abs(x_wrong_sens), cmap='gray')
axes[0].set_title(f'Wrong sensitivity ({SENS_SNR:.1f} noise)\nPSNR: {psnr_wrong:.1f} dB')
axes[1].imshow(np.abs(x_corrected), cmap='gray')
axes[1].set_title(f'Corrected (ESPIRiT)\nPSNR: {psnr_fixed:.1f} dB')
axes[2].imshow(x_true_real, cmap='gray')
axes[2].set_title('Ground Truth')
plt.tight_layout()
plt.savefig('mri_mismatch_correction.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## Expected Output

| Scenario | PSNR |
|----------|------|
| Perfect sensitivity (no mismatch) | ~34 dB |
| 10% sensitivity noise, no correction | ~28 dB |
| 10% sensitivity noise, ESPIRiT corrected | ~33 dB (+5 dB) |

---

## References

- **ESPIRiT**: Uecker et al., MRM 2014
- **SENSE**: Pruessmann et al., MRM 1999
- **B0 correction**: Sutton et al., IEEE TMI 2003
