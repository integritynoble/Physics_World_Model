# Proton Therapy Imaging — PnP-FISTA (NLM)

**Device**: CPU  **Input**: dose distribution (H × W × D, float32)

*No reference metric in registry*

## System

Proton Therapy Imaging reconstruction problem.

## Algorithm Parameters

| Parameter | Default |
|-----------|---------|
| `iters` | `20` |
| `sigma` | `0.05` |
| `mu` | `0.5` |

## Measurement

```python
# Option A — PWM benchmark (ground truth available → PSNR/SSIM)
import h5py
with h5py.File('proton_therapy_img_public.h5', 'r') as f:   # download from GCS below
    y, x_true = f['y'][0], f['x_true'][0]
# GCS: gs://pwm-benchmark-datasets/datasets/Benchmark/proton_therapy_img/public/

# Option B — your data
import numpy as np
y      = np.load('your_measurement.npy').astype('float32')  # dose distribution (H × W × D, float32)
x_true = None   # provide your ground truth for PSNR/SSIM
```

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.proton_therapy_img.solvers import run_solver
from pwm_core.utils.metrics import compute_psnr, compute_ssim

cfg = {'iters': 20, 'sigma': 0.05, 'mu': 0.5}
x   = run_solver('pnp_fista_nlm', y, cfg=cfg)

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
plt.tight_layout(); plt.savefig('proton_therapy_img_pnp_fista_nlm.png'); plt.show()
```

## Reference

*Beck & Teboulle 2009 + PnP*
