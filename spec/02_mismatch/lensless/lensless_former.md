# Lensless (Diffuser Camera) Imaging — LenslessFormer + Gradient Mismatch Correction

**Device**: GPU  **Input**: diffuser measurement (H × W, float32)
**Reference**: Cao H. et al., LenslessFormer: Lensless Image Restoration via Transformer, CVPR, 2024


## Mismatch

| Parameter | Range | Typical | Gradient Correction |
|-----------|-------|---------|---------------------|
| PSF shift (px) | `[-5, +5] px (x and y)` | ±1 px thermal drift | Gradient w.r.t. PSF shift; minimizes ||Hx - y|| with shifted PSF |

## Correction Parameters

```python
mismatch_cfg = {
    'psf_shift': None,  # None = auto-calibrate
    'search_range': '[-5, +5] px (x and y)',
    'grad_steps': 50,
    'grad_lr': 0.01,
}
```

## Measurement

```python
# Option A — PWM benchmark (ground truth → PSNR/SSIM)
import h5py
with h5py.File('lensless_public.h5', 'r') as f:   # GCS: gs://pwm-benchmark-datasets/datasets/Benchmark/lensless/public/
    y, x_true = f['y'][0], f['x_true'][0]

# Option B — your data
import numpy as np
y      = np.load('your_measurement.npy').astype('float32')  # diffuser measurement (H × W, float32)
x_true = None   # provide for PSNR/SSIM
```

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.lensless.solvers import run_solver
from pwm_core.mismatch.operators import lensless_calibrate_shift
from scipy.ndimage import gaussian_filter
from pwm_core.utils.metrics import compute_psnr, compute_ssim

# 1. Baseline — reconstruct WITHOUT mismatch correction
x_wrong = run_solver('lensless_former', y, cfg={})

# 2. Calibrate mismatch parameter
psf = gaussian_filter(psf_nominal, sigma=5); psf /= psf.sum()
shift = lensless_calibrate_shift(y, psf, shift_range=5)

# 3. Reconstruct WITH correction
cfg = {"psf_shift_y": float(shift[0]), "psf_shift_x": float(shift[1])}
cfg.update({})
x_corrected = run_solver('lensless_former', y, cfg=cfg)

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
plt.tight_layout(); plt.savefig('lensless_lensless_former_corrected.png'); plt.show()
```

## Benchmark

`Lensless (Diffuser Camera) Imaging — LenslessFormer + gradient` corresponds to the leaderboard entry at
`https://pwm.platformai.org/benchmark/lensless` (mismatch correction tier).
