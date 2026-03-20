# MRI — Imaging System Design Spec

> **Use Case 3: Simulate forward model (with mismatch) + reconstruct**

---

## System DAG

```
[RF Coil Array (4 coils)] → [k-space Sampling] → [Coil Data] → [ADC] → y
         ↓                          ↓                  ↓
   [B0 drift           [Undersampling 4×        [Coil sensitivity
    Δf=20 Hz]           Cartesian pattern]       mismatch σ=0.1]
```

---

## Element Definitions

### Element: RF Transmit/Receive Coils (`coil_array`)
- **Type**: source + receiver
- **Parameters**:
  - `n_coils`: 4
  - `geometry`: head/body array
  - `field_strength`: 3T
- **Mismatch sources**:
  - `coil_sensitivity_error` [**MEDIUM**]: Temperature drift → ±10% sensitivity
    - Correction: ESPIRiT auto-calibration from ACS data

### Element: k-space Sampling (`kspace`)
- **Type**: measurement
- **Parameters**:
  - `sampling`: Cartesian
  - `acceleration`: 4×
  - `center_fraction`: 0.08
  - `acs_lines`: 24
- **Mismatch sources**:
  - `gradient_delay` [**LOW**]: Eddy currents → ±5 µs gradient delay
    - Correction: Navigator-based trajectory correction

### Element: B0 Field (`b0_field`)
- **Type**: interaction
- **Parameters**:
  - `nominal_freq_mhz`: 127.7
  - `drift_hz_per_min`: 5
- **Mismatch sources**:
  - `b0_drift` [**LOW**]: Frequency drift during scan → ghosting
    - Correction: Navigator echo between repetitions

---

## Noise Model

```
y = F_u · S · x + η
η ~ N(0, σ²I),  σ corresponds to SNR ~35 dB
```

where `F_u` = undersampled Fourier, `S` = coil sensitivity maps.

---

## Run Button

```python
import sys, os
BASE = os.path.expanduser('~/Physics_World_Model/pwm/public')
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'packages/pwm_core'))

import numpy as np, matplotlib.pyplot as plt
from pwm_core.mismatch.operators import (
    mri_generate_coil_sensitivities, mri_forward_sense,
    mri_estimate_sensitivities_acs, mri_sense_recon
)

H, W, n_coils = 128, 128, 4
from skimage.data import shepp_logan_phantom
x_true = shepp_logan_phantom()[:H, :W].astype(np.complex64)

# True coil sensitivities
sens_true = mri_generate_coil_sensitivities(H, W, n_coils, seed=42)

# 4× Cartesian undersampling with ACS
mask = np.zeros((H, W), np.float32)
mask[::4, :] = 1.0; mask[H//2-12:H//2+12, :] = 1.0   # center 24 ACS lines

# Simulate with sensitivity mismatch
SNR = 35  # dB
sens_wrong = sens_true + 0.1 * np.random.randn(*sens_true.shape).astype(np.complex64)

y = mri_forward_sense(x_true, sens_true, mask)
noise_std = np.abs(x_true).max() / (10**(SNR/20))
y += noise_std * np.random.randn(*y.shape).astype(np.complex64)

# Estimate sensitivity maps from ACS
sens_est = mri_estimate_sensitivities_acs(y, acs_lines=24)

# Reconstruct
x_wrong = mri_sense_recon(y, sens_wrong, mask)
x_corrected = mri_sense_recon(y, sens_est, mask)

from skimage.metrics import peak_signal_noise_ratio
ref = np.abs(x_true)
psnr_w = peak_signal_noise_ratio(ref, np.abs(x_wrong), data_range=ref.max())
psnr_c = peak_signal_noise_ratio(ref, np.abs(x_corrected), data_range=ref.max())
print(f"PSNR (wrong sensitivity): {psnr_w:.2f} dB")
print(f"PSNR (ESPIRiT corrected): {psnr_c:.2f} dB  (+{psnr_c-psnr_w:.2f} dB)")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(np.abs(x_wrong), cmap='gray'); axes[0].set_title(f'Wrong sens.\nPSNR={psnr_w:.1f}dB')
axes[1].imshow(np.abs(x_corrected), cmap='gray'); axes[1].set_title(f'ESPIRiT corrected\nPSNR={psnr_c:.1f}dB')
axes[2].imshow(np.abs(x_true), cmap='gray'); axes[2].set_title('Ground Truth')
plt.tight_layout(); plt.savefig('mri_system_design.png', dpi=150); plt.show()
```

---

## References

- **SENSE**: Pruessmann et al., MRM 1999
- **ESPIRiT**: Uecker et al., MRM 2014
