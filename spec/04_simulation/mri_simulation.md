# MRI Physics Simulation

> **Use Case 4: Scientific Simulation**
> Simulate MRI k-space acquisition with undersampling and noise

---

## Forward Model

```
y = F_u · x + η
```

where `F_u` = undersampled 2D Fourier transform (random Cartesian mask), `η ~ N(0, σ²)`.

---

## Simulation Code

```python
# ============================================================
# MRI Physics Simulation — PWM Run Button
# ============================================================
import numpy as np, matplotlib.pyplot as plt

# -------------------------------------------------------
# Parameters
# -------------------------------------------------------
H, W         = 128, 128      # Image size
ACCELERATION = 4             # Undersampling factor
CENTER_FRAC  = 0.08          # Fraction of center k-space always sampled
SNR_DB       = 35            # Signal-to-noise ratio
SEED         = 42

rng = np.random.default_rng(SEED)

# -------------------------------------------------------
# 1. Phantom
# -------------------------------------------------------
from skimage.data import shepp_logan_phantom
x_true = shepp_logan_phantom()[:H, :W].astype(np.float32)

# -------------------------------------------------------
# 2. Sampling mask (Cartesian, R=4, center always sampled)
# -------------------------------------------------------
mask = np.zeros((H, W), dtype=bool)
# Center fraction always sampled
nc = int(H * CENTER_FRAC)
mask[H//2 - nc//2 : H//2 + nc//2, :] = True
# Random remaining lines (total ~1/R fraction)
n_remaining = int(H / ACCELERATION) - nc
candidate_rows = [r for r in range(H) if r < H//2 - nc//2 or r >= H//2 + nc//2]
sampled_rows = rng.choice(candidate_rows, size=n_remaining, replace=False)
mask[sampled_rows, :] = True

print(f"Sampling rate: {mask.mean()*100:.1f}% (target: {100/ACCELERATION:.0f}%)")

# -------------------------------------------------------
# 3. Forward: full k-space → undersample → add noise
# -------------------------------------------------------
kspace_full = np.fft.fftshift(np.fft.fft2(x_true))
kspace_us   = kspace_full * mask

noise_sigma = np.abs(x_true).max() / (10**(SNR_DB/20))
kspace_us  += noise_sigma * (rng.standard_normal((H,W)) + 1j*rng.standard_normal((H,W)))

# -------------------------------------------------------
# 4. Zero-filled reconstruction (baseline)
# -------------------------------------------------------
x_zf = np.abs(np.fft.ifft2(np.fft.ifftshift(kspace_us))).astype(np.float32)

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
ref_max = x_true.max()
psnr_zf = peak_signal_noise_ratio(x_true, x_zf, data_range=ref_max)
ssim_zf = structural_similarity(x_true, x_zf, data_range=ref_max)
print(f"Zero-Filled: PSNR={psnr_zf:.2f}dB, SSIM={ssim_zf:.4f}")

# -------------------------------------------------------
# 5. Visualize
# -------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes[0,0].imshow(x_true, cmap='gray'); axes[0,0].set_title('Phantom')
axes[0,1].imshow(np.log1p(np.abs(kspace_full)), cmap='gray'); axes[0,1].set_title('Full k-space (log)')
axes[0,2].imshow(mask, cmap='gray'); axes[0,2].set_title(f'Sampling mask ({ACCELERATION}× accel.)')
axes[1,0].imshow(np.log1p(np.abs(kspace_us)), cmap='gray'); axes[1,0].set_title('Undersampled k-space')
axes[1,1].imshow(x_zf, cmap='gray'); axes[1,1].set_title(f'Zero-Filled\nPSNR={psnr_zf:.1f}dB')
axes[1,2].imshow(np.abs(x_zf - x_true), cmap='hot'); axes[1,2].set_title('|Error|')
plt.tight_layout(); plt.savefig('mri_simulation.png', dpi=150); plt.show()

# Save for reconstruction
np.save('mri_kspace_simulated.npy', kspace_us)
np.save('mri_mask_simulated.npy', mask.astype(np.float32))
np.save('mri_phantom_simulated.npy', x_true)
print("Saved: mri_kspace_simulated.npy, mri_mask_simulated.npy, mri_phantom_simulated.npy")
```

---

## References

- **MRI forward model**: Lauterbur, Nature 1973; Mansfield & Grannell, J. Phys. C 1973
- **Compressed sensing MRI**: Lustig et al., MRM 2007
