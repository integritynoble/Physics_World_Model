# Widefield Fluorescence Microscopy — Noise2Void (PnP-PGD DRUNet) + Gradient Mismatch Correction

**Device**: GPU  **Input**: fluorescence image (H × W, float32)
**Reference**: Krull et al. 2019, CVPR


## Mismatch

| Parameter | Range | Typical | Gradient Correction |
|-----------|-------|---------|---------------------|
| PSF sigma (px) | `[0.5, 3.0] px` | ±0.3 px | Sharpness-based grid search; gradient refines PSF width via Richardson-Lucy |

## Correction Parameters

```python
mismatch_cfg = {
    'psf_sigma': None,  # None = auto-calibrate
    'search_range': '[0.5, 3.0] px',
    'grad_steps': 50,
    'grad_lr': 0.01,
}
```

## Measurement

```python
# Option A — PWM benchmark (ground truth → PSNR/SSIM)
import h5py
with h5py.File('widefield_public.h5', 'r') as f:   # GCS: gs://pwm-benchmark-datasets/datasets/Benchmark/widefield/public/
    y, x_true = f['y'][0], f['x_true'][0]

# Option B — your data
import numpy as np
y      = np.load('your_measurement.npy').astype('float32')  # fluorescence image (H × W, float32)
x_true = None   # provide for PSNR/SSIM
```

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.widefield.solvers import run_solver
from pwm_core.mismatch.operators import fluorescence_calibrate_sigma
from pwm_core.utils.metrics import compute_psnr, compute_ssim

# 1. Baseline — reconstruct WITHOUT mismatch correction
x_wrong = run_solver('famous_dl', y, cfg={})

# 2. Calibrate mismatch parameter
psf_sigma = fluorescence_calibrate_sigma(y, sigma_range=[0.5, 3.0])

# 3. Reconstruct WITH correction
cfg = {"psf_sigma": float(psf_sigma)}
cfg.update({})
x_corrected = run_solver('famous_dl', y, cfg=cfg)

# 4. Evaluate
if x_true is not None:
    print(f"No correction  PSNR {compute_psnr(x_true, x_wrong):.2f} dB  "
          f"SSIM {compute_ssim(x_true, x_wrong):.4f}")
    print(f"Corrected      PSNR {compute_psnr(x_true, x_corrected):.2f} dB  "
          f"SSIM {compute_ssim(x_true, x_corrected):.4f}")

x = x_corrected
# Visualize
import matplotlib.pyplot as plt
ncols = 3 if x_true is not None else 2
fig, axes = plt.subplots(1, ncols, figsize=(4*ncols, 4))
y_show = y if y.ndim == 2 else y[0] if y.ndim == 3 else y
axes[0].imshow(y_show, cmap='gray'); axes[0].set_title('Measurement')
axes[1].imshow(x, cmap='gray'); axes[1].set_title('Reconstruction')
if x_true is not None:
    axes[2].imshow(x_true, cmap='gray'); axes[2].set_title('Ground Truth')
plt.tight_layout(); plt.savefig('widefield_famous_dl_corrected.png'); plt.show()
```

## Benchmark

`Widefield Fluorescence Microscopy — Noise2Void (PnP-PGD DRUNet) + gradient` corresponds to the leaderboard entry at
`https://pwm.platformai.org/benchmark/widefield` (mismatch correction tier).
