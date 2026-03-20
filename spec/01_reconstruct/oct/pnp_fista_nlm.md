# Optical Coherence Tomography (OCT) — PnP-FISTA (NLM)

**Device**: CPU  **Input**: spectrum (wavenumbers × A-scans, float32)

*No reference metric in registry*

## System

Optical Coherence Tomography (OCT) reconstruction problem.

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
with h5py.File('oct_public.h5', 'r') as f:   # download from GCS below
    y, x_true = f['y'][0], f['x_true'][0]
# GCS: gs://pwm-benchmark-datasets/datasets/Benchmark/oct/public/

# Option B — your data
import numpy as np
y      = np.load('your_measurement.npy').astype('float32')  # spectrum (wavenumbers × A-scans, float32)
x_true = None   # provide your ground truth for PSNR/SSIM
```

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.oct.solvers import run_solver
from pwm_core.utils.metrics import compute_psnr, compute_ssim

cfg = {'iters': 20, 'sigma': 0.05, 'mu': 0.5}
x   = run_solver('pnp_fista_nlm', y, cfg=cfg)

if x_true is not None:
    print(f"PSNR {compute_psnr(x_true, x):.2f} dB  SSIM {compute_ssim(x_true, x):.4f}")

# Visualize
import matplotlib.pyplot as plt
ncols = 3 if x_true is not None else 2
fig, axes = plt.subplots(1, ncols, figsize=(4*ncols, 4))
y_show = y if y.ndim == 2 else y[0] if y.ndim == 3 else y
axes[0].imshow(y_show, cmap='gray'); axes[0].set_title('Measurement')
axes[1].imshow(x, cmap='gray'); axes[1].set_title('Reconstruction')
if x_true is not None:
    axes[2].imshow(x_true, cmap='gray'); axes[2].set_title('Ground Truth')
plt.tight_layout(); plt.savefig('oct_pnp_fista_nlm.png'); plt.show()
```

## Reference

*Beck & Teboulle 2009 + PnP*
