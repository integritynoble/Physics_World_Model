# Coded Aperture Snapshot Spectral Imaging (CASSI) — BiSRNet

**Device**: GPU  **Input**: coded snapshot (H × W, float32)

*No reference metric in registry*

## System

Coded Aperture Snapshot Spectral Imaging (CASSI) reconstruction problem.

## Algorithm Parameters

| Parameter | Default |
|-----------|---------|
| `model_key` | `bisrnet` |

## Measurement

```python
# Option A — PWM benchmark (ground truth available → PSNR/SSIM)
import h5py
with h5py.File('cassi_public.h5', 'r') as f:   # download from GCS below
    y, x_true = f['y'][0], f['x_true'][0]
# GCS: gs://pwm-benchmark-datasets/datasets/Benchmark/cassi/public/

# Option B — your data
import numpy as np
y      = np.load('your_measurement.npy').astype('float32')  # coded snapshot (H × W, float32)
x_true = None   # provide your ground truth for PSNR/SSIM
```

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.cassi.solvers import run_solver
from pwm_core.utils.metrics import compute_psnr, compute_ssim

cfg = {'model_key': 'bisrnet'}
x   = run_solver('bisrnet', y, cfg=cfg)

if x_true is not None:
    print(f"PSNR {compute_psnr(x_true, x):.2f} dB  SSIM {compute_ssim(x_true, x):.4f}")

# Visualize (hyperspectral: spatial slice + spectral profile)
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
n_bands = x.shape[-1] if x.ndim == 3 else x.shape[0]
band = n_bands // 2
xb = x[..., band] if x.ndim == 3 else x[band]
axes[0].imshow(xb, cmap='gray'); axes[0].set_title(f'Recon band {band}')
axes[1].plot(x[x.shape[0]//2, x.shape[1]//2] if x.ndim==3 else x[:,x.shape[1]//2,x.shape[2]//2])
axes[1].set_title('Spectral profile (center pixel)')
if x_true is not None:
    xtb = x_true[..., band] if x_true.ndim == 3 else x_true[band]
    axes[2].imshow(xtb, cmap='gray'); axes[2].set_title('GT band')
plt.tight_layout(); plt.savefig('cassi_bisrnet.png'); plt.show()
```

## Reference

*BiSRNet, 2023*
