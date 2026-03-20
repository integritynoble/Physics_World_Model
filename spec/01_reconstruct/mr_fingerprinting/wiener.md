# MR Fingerprinting (MRF) — Wiener Deconvolution

**Device**: CPU  **Input**: signal evolution (T × H × W, complex64)

*No reference metric in registry*

## System

MR Fingerprinting (MRF) reconstruction problem.

## Algorithm Parameters

| Parameter | Default |
|-----------|---------|
| `reg` | `0.01` |

## Measurement

```python
# Option A — PWM benchmark (ground truth available → PSNR/SSIM)
import h5py
with h5py.File('mr_fingerprinting_public.h5', 'r') as f:   # download from GCS below
    y, x_true = f['y'][0], f['x_true'][0]
# GCS: gs://pwm-benchmark-datasets/datasets/Benchmark/mr_fingerprinting/public/

# Option B — your data
import numpy as np
y      = np.load('your_measurement.npy').astype('float32')  # signal evolution (T × H × W, complex64)
x_true = None   # provide your ground truth for PSNR/SSIM
```

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.mr_fingerprinting.solvers import run_solver
from pwm_core.utils.metrics import compute_psnr, compute_ssim

cfg = {'reg': 0.01}
x   = run_solver('wiener', y, cfg=cfg)

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
plt.tight_layout(); plt.savefig('mr_fingerprinting_wiener.png'); plt.show()
```

## Reference

*Wiener, Extrapolation, Interpolation... 1949*
