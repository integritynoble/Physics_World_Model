# Generic Matrix Sensing — FISTA-L1 (high quality)

**Device**: CPU  **Input**: partial matrix (M × N, float32, NaN=missing)

*No reference metric in registry*

## System

Generic Matrix Sensing reconstruction problem.

## Algorithm Parameters

*Uses default configuration.*

## Measurement

```python
# Option A — PWM benchmark (ground truth available → PSNR/SSIM)
import h5py
with h5py.File('matrix_public.h5', 'r') as f:   # download from GCS below
    y, x_true = f['y'][0], f['x_true'][0]
# GCS: gs://pwm-benchmark-datasets/datasets/Benchmark/matrix/public/

# Option B — your data
import numpy as np
y      = np.load('your_measurement.npy').astype('float32')  # partial matrix (M × N, float32, NaN=missing)
x_true = None   # provide your ground truth for PSNR/SSIM
```

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.matrix.solvers import run_solver
from pwm_core.utils.metrics import compute_psnr, compute_ssim

cfg = {}
x   = run_solver('best_quality', y, cfg=cfg)

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
plt.tight_layout(); plt.savefig('matrix_best_quality.png'); plt.show()
```

## Reference

*Beck & Teboulle 2009*
