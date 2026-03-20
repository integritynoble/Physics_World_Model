# CT Physics Simulation

> **Use Case 4: Scientific Simulation**
> Simulate the complete CT forward model: Radon transform + Poisson noise + beam hardening

---

## Forward Model

```
y = -log( Poisson(I₀ · exp(-Radon(x))) / I₀ )
```

| Variable | Description | Typical Value |
|----------|-------------|---------------|
| `x` | 2D attenuation map (phantom) | (256, 256), values in [0, 0.08] mm⁻¹ |
| `Radon(x)` | Radon transform (line integrals) | (n_angles, n_det) |
| `I₀` | Incident photon count | 10⁴–10⁶ |
| `y` | Log-transformed sinogram | (n_angles, n_det) |
| `η` | Poisson noise | Photon-limited |

---

## Simulation Code

```python
# ============================================================
# CT Physics Simulation — PWM Run Button
# ============================================================
import sys, os
BASE = os.path.expanduser('~/Physics_World_Model/pwm/public')
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'packages/pwm_core'))

import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon
from skimage.data import shepp_logan_phantom

# -------------------------------------------------------
# 1. Generate phantom
# -------------------------------------------------------
PHANTOM = 'shepp_logan'   # Options: 'shepp_logan', 'custom', 'real_ct'

if PHANTOM == 'shepp_logan':
    x_true = shepp_logan_phantom().astype(np.float32)
elif PHANTOM == 'custom':
    x_true = np.zeros((256, 256), np.float32)
    x_true[80:180, 80:180] = 0.04   # water (0.02 cm⁻¹)
    x_true[100:160, 100:160] = 0.08  # bone (0.08 cm⁻¹)
    # Add noise: x_true += 0.005 * np.random.randn(*x_true.shape)
    x_true = np.clip(x_true, 0, None)

print(f"Phantom shape: {x_true.shape}, range: [{x_true.min():.4f}, {x_true.max():.4f}]")

# -------------------------------------------------------
# 2. System parameters (customize here)
# -------------------------------------------------------
N_ANGLES       = 180         # Number of projection angles
ANGULAR_RANGE  = 180.0       # Angular span (degrees); 360 for full-scan
I0             = 1e5         # Incident photon count (noise level)
SIGMA_READOUT  = 2.0         # Readout noise (standard deviations in e⁻)
BH_COEFF       = 0.03        # Beam hardening coefficient (0=mono, 0.05=clinical)
ADD_COR_OFFSET = 0.0         # Center-of-rotation offset (pixels)
DETECTOR_PIXELS = 362        # Number of detector pixels

# -------------------------------------------------------
# 3. Forward projection (Radon transform)
# -------------------------------------------------------
angles = np.linspace(0.0, ANGULAR_RANGE, N_ANGLES, endpoint=False)
sino_ideal = radon(x_true, theta=angles, circle=False)
print(f"Sinogram shape: {sino_ideal.shape}  ({N_ANGLES} angles × {sino_ideal.shape[0]} detectors)")

# Apply CoR offset (shift sinogram)
if ADD_COR_OFFSET != 0:
    from scipy.ndimage import shift as nd_shift
    sino_ideal = nd_shift(sino_ideal, shift=(0, ADD_COR_OFFSET), mode='nearest')

# -------------------------------------------------------
# 4. Noise simulation
# -------------------------------------------------------
# Poisson noise (photon counting)
transmission = I0 * np.exp(-sino_ideal)
counts_noisy = np.random.poisson(np.clip(transmission, 0, None)).astype(np.float32)

# Log-transform back to attenuation
epsilon = 1.0  # minimum count to avoid log(0)
sino_noisy = -np.log((counts_noisy + epsilon) / I0)

# Add Gaussian readout noise
if SIGMA_READOUT > 0:
    sino_noisy += (SIGMA_READOUT / I0) * np.random.randn(*sino_noisy.shape).astype(np.float32)

# Beam hardening (polychromatic X-ray effect)
if BH_COEFF > 0:
    sino_noisy = sino_noisy + BH_COEFF * sino_noisy**2

y = sino_noisy.astype(np.float32)

# -------------------------------------------------------
# 5. Simple FBP reconstruction (validate forward model)
# -------------------------------------------------------
x_fbp = iradon(y, theta=angles, circle=False, filter_name='ramp').astype(np.float32)

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
ref_max = x_true.max()
psnr = peak_signal_noise_ratio(x_true, x_fbp, data_range=ref_max)
ssim = structural_similarity(x_true, x_fbp, data_range=ref_max)
print(f"\nFBP reconstruction: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")
print(f"(Expected: ~27 dB at I0=1e5, 180 angles)")

# -------------------------------------------------------
# 6. Visualize
# -------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Row 1: phantom and sinograms
axes[0, 0].imshow(x_true, cmap='gray'); axes[0, 0].set_title('Phantom x_true')
axes[0, 1].imshow(sino_ideal.T, cmap='gray', aspect='auto')
axes[0, 1].set_title(f'Ideal sinogram\n({N_ANGLES} angles)')
axes[0, 2].imshow(y.T, cmap='gray', aspect='auto')
axes[0, 2].set_title(f'Noisy sinogram\n(I₀={I0:.0e}, σ={SIGMA_READOUT})')

# Row 2: reconstruction and profiles
axes[1, 0].imshow(x_fbp, cmap='gray'); axes[1, 0].set_title(f'FBP reconstruction\nPSNR={psnr:.1f}dB')
axes[1, 1].plot(x_true[128, :], label='Phantom', linewidth=2)
axes[1, 1].plot(x_fbp[128, :], label='FBP', linewidth=1, alpha=0.8)
axes[1, 1].legend(); axes[1, 1].set_title('Horizontal profile (row 128)')
axes[1, 2].imshow(np.abs(x_fbp - x_true), cmap='hot'); axes[1, 2].set_title('|Error|')
axes[1, 2].colorbar = plt.colorbar(axes[1, 2].get_images()[0], ax=axes[1, 2])

plt.tight_layout()
plt.savefig('ct_simulation.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: ct_simulation.png")

# -------------------------------------------------------
# 7. Save simulated data for further use
# -------------------------------------------------------
np.save('ct_sinogram_simulated.npy', y)
np.save('ct_phantom_simulated.npy', x_true)
print("Saved: ct_sinogram_simulated.npy, ct_phantom_simulated.npy")
```

---

## Parameter Sweep

```python
# Sweep noise levels to show reconstruction quality vs. dose
import pandas as pd

results = []
for I0_val in [1e3, 1e4, 1e5, 1e6]:
    transmission = I0_val * np.exp(-sino_ideal)
    counts = np.random.poisson(np.clip(transmission, 0, None))
    sino_n = -np.log((counts + 1) / I0_val)
    x_r = iradon(sino_n, theta=angles, circle=False, filter_name='ramp')
    psnr_v = peak_signal_noise_ratio(x_true, x_r, data_range=x_true.max())
    results.append({'I0': I0_val, 'PSNR_FBP': psnr_v})

df = pd.DataFrame(results)
print(df.to_string(index=False))
# Expected output:
#      I0  PSNR_FBP
#    1000      22.3
#   10000      25.8
#  100000      27.1
# 1000000      28.0
```

---

## Analytical Validation

For the Shepp-Logan phantom, the ideal sinogram can be validated against known analytical
projections. The FBP reconstruction should converge to the phantom as I₀ → ∞ and
N_ANGLES → 1000.

---

## References

- **Radon transform**: Radon, Ber. Verh. Sächs. Akad. Wiss. 1917
- **Shepp-Logan phantom**: Shepp & Logan, IEEE TNS 1974
- **FBP**: Ramachandran & Lakshminarayanan, PNAS 1971
- **LoDoPaB-CT dataset**: Leuschner et al., Scientific Data 2021
