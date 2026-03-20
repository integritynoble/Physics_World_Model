# CT — Imaging System Design Spec

> **Use Case 3: Simulate forward model (with mismatch) + reconstruct**
> Based on: `papers/system_design/outputs/ct_forward_v1_iter1.md` and `ct_reconstruction_v1_iter1.md`

---

## System Overview

Design a **sparse-view, low-dose CT** imaging system for pediatric chest imaging.
The system simulates all physical elements from X-ray source to detector digitization,
including realistic noise and calibration mismatch sources.

---

## System DAG

```
[X-ray Tube 80kVp] → [Soft Tissue Phantom] → [Parallel-Beam Geometry 60 angles]
       ↓                      ↓                           ↓
  [Beam hardening           [Compton scatter          [CoR offset
   (polychromatic)]          (SPR ~0.3)]               (±0.5 px)]
                                                            ↓
                            → [CsI:Tl Flat-Panel Detector] → [12-bit ADC] → y
                                        ↓
                               [Poisson I0=1e4]
                               [Gaussian σ=3 e⁻]
                               [Dark current 0.05 e⁻/s]
```

---

## Element Definitions

### Element: X-ray Tube Source (`xray_source`)
- **Type**: source
- **Parameters**:
  - `energy_kVp`: 80
  - `flux_photons_per_s`: 500,000
  - `focal_spot_mm`: 0.4
  - `filtration`: 1.5mm Al
  - `spectrum`: polychromatic
- **Mismatch sources**:
  - `beam_hardening` [**HIGH**]: Polychromatic spectrum → cupping artifacts
    - Correction: 2nd-order polynomial linearization from water phantom
- **Connects to**: `tissue_attenuation`

### Element: Soft Tissue Phantom (`tissue_attenuation`)
- **Type**: interaction
- **Parameters**:
  - `model`: beer_lambert
  - `mu_water_cm`: 0.184
  - `material`: pediatric_soft_tissue
- **Mismatch sources**:
  - `scatter` [**MEDIUM**]: Compton scatter background (SPR ~0.3)
    - Correction: Scatter kernel estimation (1D Gaussian convolution)
- **Connects to**: `geometry`

### Element: Parallel-Beam Geometry (`geometry`)
- **Type**: geometry
- **Parameters**:
  - `scan_type`: parallel_beam
  - `num_angles`: 60
  - `angular_range_deg`: 180
  - `detector_pixels`: 256
  - `pixel_pitch_mm`: 0.4
- **Mismatch sources**:
  - `center_of_rotation_offset` [**MEDIUM**]: ±0.5 px mechanical misalignment
    - Correction: Cross-correlation of 0°/180° projection pair
- **Connects to**: `detector`

### Element: CsI:Tl Flat-Panel Detector (`detector`)
- **Type**: detector
- **Parameters**:
  - `scintillator`: CsI:Tl
  - `pixels`: [256, 256]
  - `pixel_pitch_mm`: 0.4
  - `quantum_efficiency`: 0.75
- **Noise**:
  - `poisson`: I₀ = 10,000 photons/pixel
  - `gaussian`: σ = 3 electrons readout
  - `dark_current`: 0.05 e⁻/s, 0.02s exposure
- **Mismatch sources**:
  - `detector_nonuniformity` [**LOW**]: ±2% per-pixel gain variations
    - Correction: Flat-field correction with air scan

### Element: 12-bit ADC (`adc`)
- **Type**: digitization
- **Parameters**:
  - `bit_depth`: 12
  - `dynamic_range_db`: 72

---

## Composite Noise Model

```
y ~ Poisson(I₀ · exp(-H·x)) + N(0, σ_readout²) + Poisson(dark · t_exp)
```

**Measurement shape**: `(256, 60)` sinogram

**Estimated SNR**: ~17 dB at low-dose (I₀=1e4) — noisy but recoverable with TV-ADMM.

---

## Reconstruction Algorithm: TV-ADMM with Mismatch Corrections

See `papers/system_design/outputs/ct_reconstruction_v1_iter1.md` for full details.

**Pipeline**:
1. Apply beam-hardening correction: `y_corr = y - 0.05·y²`
2. Subtract scatter: `y_corr -= 0.1 · G_σ=20(y)`
3. Calibrate CoR: `cor = ct_calibrate_cor(y_corr)`
4. Initialize with FBP: `x₀ = FBP(y_corr, cor=cor)`
5. TV-ADMM iterations (100):
   - Data fidelity: `grad = Rᵀ(R·xₖ - y_corr)`
   - TV proximal: `x_{k+1} = prox_{λTV}(xₖ - η·grad)`, λ=0.01
   - Non-negativity: `x_{k+1} = max(x_{k+1}, 0)`

---

## Run Button

```python
# ============================================================
# CT System Design: Simulate + Reconstruct — PWM Run Button
# ============================================================
import sys, os
BASE = os.path.expanduser('~/Physics_World_Model/pwm/public')
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'packages/pwm_core'))

import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------
# 1. Simulate forward model (CT system with mismatches)
# -------------------------------------------------------
from pwm_core.mismatch.operators import ct_radon_forward, ct_calibrate_cor, ct_sart_tv_recon

from skimage.data import shepp_logan_phantom
x_true = shepp_logan_phantom().astype(np.float32)

# System parameters (from DAG above)
N_ANGLES     = 60
I0           = 1e4        # Low-dose
COR_OFFSET   = 0.5        # Mechanical CoR offset (px)
BH_COEFF     = 0.05       # Beam hardening coefficient
SCATTER_FRAC = 0.05       # Scatter fraction
SIGMA_READOUT = 3.0       # Detector readout noise (electrons)

# Simulate sinogram
print("Simulating CT acquisition (sparse-view, low-dose, with mismatches)...")
y_clean = ct_radon_forward(x_true, n_angles=N_ANGLES, cor_offset=COR_OFFSET)

# Add Poisson noise
y_noisy = np.random.poisson(I0 * np.exp(-y_clean)).astype(np.float32)
y_log = -np.log(y_noisy / I0 + 1e-10)

# Add beam hardening
y_bh = y_log + BH_COEFF * y_log**2

# Add scatter (low-frequency bias)
from scipy.ndimage import gaussian_filter
y_scatter = y_bh + SCATTER_FRAC * gaussian_filter(y_bh, sigma=20)

# Add readout noise
y = y_scatter + (SIGMA_READOUT / I0) * np.random.randn(*y_scatter.shape)
print(f"Simulated sinogram shape: {y.shape}")

# -------------------------------------------------------
# 2. Apply mismatch corrections
# -------------------------------------------------------
print("Applying corrections: beam hardening, scatter, CoR ...")
y_corr = y - BH_COEFF * y**2             # beam hardening
y_corr -= SCATTER_FRAC * gaussian_filter(y_corr, sigma=20)   # scatter
cor_est = ct_calibrate_cor(y_corr)        # CoR calibration
print(f"Estimated CoR offset: {cor_est:.3f} px (true: {COR_OFFSET:.3f})")

# -------------------------------------------------------
# 3. Reconstruct with TV-ADMM
# -------------------------------------------------------
print("Reconstructing with TV-ADMM ...")
x_hat = ct_sart_tv_recon(y_corr, cor_offset=cor_est)

# -------------------------------------------------------
# 4. Evaluate & Visualize
# -------------------------------------------------------
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
ref_max = x_true.max()
psnr = peak_signal_noise_ratio(x_true, x_hat, data_range=ref_max)
ssim = structural_similarity(x_true, x_hat, data_range=ref_max)
print(f"PSNR: {psnr:.2f} dB  |  SSIM: {ssim:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(y.T, cmap='gray', aspect='auto'); axes[0].set_title('Simulated Sinogram (with noise+mismatch)')
axes[1].imshow(x_hat, cmap='gray'); axes[1].set_title(f'Reconstructed (TV-ADMM)\nPSNR={psnr:.1f}dB')
axes[2].imshow(x_true, cmap='gray'); axes[2].set_title('Ground Truth')
plt.tight_layout()
plt.savefig('ct_system_design.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## Using the Multi-Agent System

For LLM-guided system design (requires Gemini API key):

```bash
cd papers/system_design/
python main.py \
  --modality ct \
  --period forward \
  --prompt "Design sparse-view CT system for pediatric chest imaging, 60 angles, low-dose I0=1e4"
```

This runs the **Plan → Judge → Performance** three-agent pipeline and generates a full system design spec.

---

## References

- **PWM System Design Paper**: `papers/system_design/paper.md`
- **CT forward design**: `papers/system_design/outputs/ct_forward_v1_iter1.md`
- **CT reconstruction design**: `papers/system_design/outputs/ct_reconstruction_v1_iter1.md`
- **TV-ADMM**: Sidky & Pan, Phys. Med. Biol. 2008; Boyd et al. 2010
