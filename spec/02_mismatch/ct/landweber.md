# X-ray Computed Tomography (CT) — Landweber + Gradient Mismatch Correction

**Device**: CPU  **Input**: sinogram (angles × detectors, float32)
**Reference**: Landweber, Am J Math 1951


## Mismatch

| Parameter | Range | Typical | Gradient Correction |
|-----------|-------|---------|---------------------|
| CoR offset | `[-5, +5] px` | ±1 px | Differentiable Radon transform; gradient descent refines CoR offset |

## Correction Parameters

```python
mismatch_cfg = {
    'cor_offset': None,  # None = auto-calibrate
    'search_range': '[-5, +5] px',
    'grad_steps': 50,
    'grad_lr': 0.01,
}
```

## Measurement

```python
# Option A — PWM benchmark (ground truth → PSNR/SSIM)
import h5py
with h5py.File('ct_public.h5', 'r') as f:   # GCS: gs://pwm-benchmark-datasets/datasets/Benchmark/ct/public/
    y, x_true = f['y'][0], f['x_true'][0]

# Option B — your data
import numpy as np
y      = np.load('your_measurement.npy').astype('float32')  # sinogram (angles × detectors, float32)
x_true = None   # provide for PSNR/SSIM
```

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.ct.solvers import run_solver
from pwm_core.mismatch.operators import ct_calibrate_cor
from pwm_core.utils.metrics import compute_psnr, compute_ssim

# 1. Baseline — reconstruct WITHOUT mismatch correction
x_wrong = run_solver('landweber', y, cfg={'iters': 30, 'step': 0.005})

# 2. Calibrate mismatch parameter
cor_offset = ct_calibrate_cor(y, shift_range=5)

# 3. Reconstruct WITH correction
cfg = {"cor_offset": float(cor_offset)}
cfg.update({'iters': 30, 'step': 0.005})
x_corrected = run_solver('landweber', y, cfg=cfg)

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
plt.tight_layout(); plt.savefig('ct_landweber_corrected.png'); plt.show()
```

## Benchmark

`X-ray Computed Tomography (CT) — Landweber + gradient` corresponds to the leaderboard entry at
`https://pwm.platformai.org/benchmark/ct` (mismatch correction tier).
