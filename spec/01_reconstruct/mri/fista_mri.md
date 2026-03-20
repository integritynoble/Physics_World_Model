# Magnetic Resonance Imaging (MRI) — FISTA

**Device**: CPU  **Input**: k-space (H × W × 2: real+imag, float32)

*No reference metric in registry*

## System

Magnetic Resonance Imaging (MRI) reconstruction problem.

## Algorithm Parameters

*Uses default configuration.*

## Measurement

```python
# Option A — PWM benchmark (ground truth available → PSNR/SSIM)
import h5py
with h5py.File('mri_public.h5', 'r') as f:   # download from GCS below
    y, x_true = f['y'][0], f['x_true'][0]
# GCS: gs://pwm-benchmark-datasets/datasets/Benchmark/mri/public/

# Option B — your data
import numpy as np
y      = np.load('your_measurement.npy').astype('float32')  # k-space (H × W × 2: real+imag, float32)
x_true = None   # provide your ground truth for PSNR/SSIM
```

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.mri.solvers import run_solver
from pwm_core.utils.metrics import compute_psnr, compute_ssim

cfg = {}
x   = run_solver('fista_mri', y, cfg=cfg)

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
plt.tight_layout(); plt.savefig('mri_fista_mri.png'); plt.show()
```

## Reference

*Beck & Teboulle, SIAM J Imaging Sci 2009*
