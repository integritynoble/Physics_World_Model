# Magnetic Resonance Imaging (MRI) — Tikhonov Regularization + Gradient Mismatch Correction

**Device**: CPU  **Input**: k-space (H × W × 2: real+imag, float32)
**Reference**: Tikhonov, Soviet Math Dokl 1963


## Mismatch

| Parameter | Range | Typical | Gradient Correction |
|-----------|-------|---------|---------------------|
| Coil sensitivity maps (B1 error) | `[0.9, 1.1] gain per coil` | ±5% per coil | ESPIRiT auto-calibration from ACS center lines |

## Correction Parameters

```python
mismatch_cfg = {
    'coil_sensitivity_maps': None,  # None = auto-calibrate
    'search_range': '[0.9, 1.1] gain per coil',
    'grad_steps': 50,
    'grad_lr': 0.01,
}
```

## Measurement

```python
# Option A — PWM benchmark (ground truth → PSNR/SSIM)
import h5py
with h5py.File('mri_public.h5', 'r') as f:   # GCS: gs://pwm-benchmark-datasets/datasets/Benchmark/mri/public/
    y, x_true = f['y'][0], f['x_true'][0]

# Option B — your data
import numpy as np
y      = np.load('your_measurement.npy').astype('float32')  # k-space (H × W × 2: real+imag, float32)
x_true = None   # provide for PSNR/SSIM
```

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.mri.solvers import run_solver
from pwm_core.mismatch.operators import mri_estimate_sensitivities_acs
from pwm_core.utils.metrics import compute_psnr, compute_ssim

# 1. Baseline — reconstruct WITHOUT mismatch correction
x_wrong = run_solver('tikhonov', y, cfg={})

# 2. Calibrate mismatch parameter
sens_maps = mri_estimate_sensitivities_acs(y, acs_lines=24)

# 3. Reconstruct WITH correction
cfg = {"sens_maps": sens_maps}
cfg.update({})
x_corrected = run_solver('tikhonov', y, cfg=cfg)

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
plt.tight_layout(); plt.savefig('mri_tikhonov_corrected.png'); plt.show()
```

## Benchmark

`Magnetic Resonance Imaging (MRI) — Tikhonov Regularization + gradient` corresponds to the leaderboard entry at
`https://pwm.platformai.org/benchmark/mri` (mismatch correction tier).
