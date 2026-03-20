# Confocal Live-Cell Microscopy — Tikhonov Regularization

**Device**: CPU  **Input**: time-lapse (T × H × W, float32)

*No reference metric in registry*

## System

Confocal Live-Cell Microscopy reconstruction problem.

## Algorithm Parameters

| Parameter | Default |
|-----------|---------|
| `iters` | `50` |
| `lam` | `0.01` |
| `step` | `0.5` |

## Measurement

```python
# Option A — PWM benchmark (ground truth available → PSNR/SSIM)
import h5py
with h5py.File('confocal_livecell_public.h5', 'r') as f:   # download from GCS below
    y, x_true = f['y'][0], f['x_true'][0]
# GCS: gs://pwm-benchmark-datasets/datasets/Benchmark/confocal_livecell/public/

# Option B — your data
import numpy as np
y      = np.load('your_measurement.npy').astype('float32')  # time-lapse (T × H × W, float32)
x_true = None   # provide your ground truth for PSNR/SSIM
```

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.confocal_livecell.solvers import run_solver
from pwm_core.utils.metrics import compute_psnr, compute_ssim

cfg = {'iters': 50, 'lam': 0.01, 'step': 0.5}
x   = run_solver('tikhonov', y, cfg=cfg)

if x_true is not None:
    print(f"PSNR {compute_psnr(x_true, x):.2f} dB  SSIM {compute_ssim(x_true, x):.4f}")

# Visualize (3D: orthogonal slices)
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
mid = [s // 2 for s in x.shape]
axes[0, 0].imshow(x[mid[0]], cmap='gray'); axes[0, 0].set_title('Recon axial')
axes[0, 1].imshow(x[:, mid[1], :], cmap='gray'); axes[0, 1].set_title('Recon coronal')
axes[0, 2].imshow(x[:, :, mid[2]], cmap='gray'); axes[0, 2].set_title('Recon sagittal')
if x_true is not None:
    axes[1, 0].imshow(x_true[mid[0]], cmap='gray'); axes[1, 0].set_title('GT axial')
    axes[1, 1].imshow(x_true[:, mid[1], :], cmap='gray'); axes[1, 1].set_title('GT coronal')
    axes[1, 2].imshow(x_true[:, :, mid[2]], cmap='gray'); axes[1, 2].set_title('GT sagittal')
plt.tight_layout(); plt.savefig('confocal_livecell_tikhonov.png'); plt.show()
```

## Reference

*Tikhonov, Soviet Math Doklady 1963*
