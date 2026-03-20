# Lensless Imaging — System Design Spec

> **Use Case 3: Simulate forward model (with mismatch) + reconstruct**
> Based on: `papers/system_design/outputs/lensless_forward_v1_iter1.md`

---

## System DAG

```
[LED Illumination] → [Diffuser] → [Object] → [CMOS Sensor]
         ↓               ↓                          ↓
   [Spatial           [PSF shift              [Poisson noise
    coherence]         ±1 px]                  Gaussian readout]
```

---

## Element Definitions

### Element: LED Array (`led_source`)
- **Type**: source
- **Parameters**:
  - `wavelength_nm`: 530 (green LED)
  - `n_leds`: 1 (or array)
  - `coherence_length_um`: 10 (partially coherent)
- **Mismatch**: None at source

### Element: Diffuser (`diffuser`)
- **Type**: interaction
- **Parameters**:
  - `type`: random diffuser (egg-shell finish)
  - `diffusion_angle_deg`: ±30
  - `psf_size_px`: 64 (PSF half-width)
- **Mismatch sources**:
  - `psf_shift` [**MEDIUM**]: Thermal drift shifts calibrated PSF by ±1 px
    - Correction: `lensless_calibrate_shift(y, psf_nominal, shift_range=5)`
  - `background` [**LOW**]: Ambient light contamination
    - Correction: Dark frame subtraction

### Element: CMOS Sensor (`sensor`)
- **Type**: detector
- **Parameters**:
  - `pixels`: [270, 270]
  - `pixel_size_um`: 1.67
  - `bit_depth`: 12
  - `full_well_capacity`: 10000 e⁻
- **Noise**:
  - `poisson`: photon shot noise
  - `gaussian`: σ = 2 e⁻ readout noise

---

## Noise Model

```
y = Poisson(I_sensor) + N(0, σ_readout²)
I_sensor = h * x + background
```

where `h` = system PSF (diffuser response).

---

## Run Button

```python
import sys, os
BASE = os.path.expanduser('~/Physics_World_Model/pwm/public')
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'packages/pwm_core'))

import numpy as np, matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from pwm_core.mismatch.operators import (
    lensless_forward, lensless_admm_tv, lensless_calibrate_shift, compute_psnr
)

# 1. Scene and PSF
H, W = 256, 256
from skimage.data import camera
x_true = camera().astype(np.float32)[:H, :W] / 255.0

# Calibrated PSF (from white-light calibration)
psf = np.zeros((H, W), np.float32); psf[H//2, W//2] = 1.0
psf = gaussian_filter(psf, sigma=10); psf /= psf.sum()

# 2. Simulate acquisition with PSF shift mismatch
TRUE_SHIFT_Y, TRUE_SHIFT_X = 1.5, -0.8   # px
psf_true = np.roll(np.roll(psf, round(TRUE_SHIFT_Y), 0), round(TRUE_SHIFT_X), 1)

I0 = 500   # photons per pixel
y = np.random.poisson(I0 * lensless_forward(x_true, psf_true) + 2.0).astype(np.float32)
y = np.log1p(y.astype(np.float32))   # log-compress

# 3. Calibrate PSF shift
shift_est = lensless_calibrate_shift(y, psf, shift_range=5)
psf_corrected = np.roll(np.roll(psf, round(shift_est[0]), 0), round(shift_est[1]), 1)
print(f"Estimated shift: {shift_est}  (true: {TRUE_SHIFT_Y:.1f}, {TRUE_SHIFT_X:.1f})")

# 4. Reconstruct
x_hat = lensless_admm_tv(y, psf_corrected, iters=100, lam=0.01)

psnr = compute_psnr(x_true, x_hat)
print(f"PSNR: {psnr:.2f} dB")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(y, cmap='gray'); axes[0].set_title('Simulated Lensless Measurement')
axes[1].imshow(x_hat, cmap='gray'); axes[1].set_title(f'Reconstruction\nPSNR={psnr:.1f}dB')
axes[2].imshow(x_true, cmap='gray'); axes[2].set_title('Ground Truth')
plt.tight_layout(); plt.savefig('lensless_system.png', dpi=150); plt.show()
```

---

## References

- **DiffuserCam**: Antipa et al., Optica 2018
- **PWM lensless design**: `papers/system_design/outputs/lensless_forward_v1_iter1.md`
