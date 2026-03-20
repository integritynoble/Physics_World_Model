# Single-Pixel Camera (SPC) — OPINE-Net+ + Gradient Mismatch Correction

**Device**: GPU  **Input**: photon counts (T × H × W, uint16)
**Reference**: Zhang et al., IEEE TCSVT 2020


## Mismatch

| Parameter | Range | Typical | Gradient Correction |
|-----------|-------|---------|---------------------|
| Sensor gain and dark bias | `gain [0.8, 1.2], bias [0, 50]` | ±10% gain, ±5 counts bias | NLL-based gain/bias estimation; gradient refines via photon statistics |

## Correction Parameters

```python
mismatch_cfg = {
    'sensor_gain_and_dark_bias': None,  # None = auto-calibrate
    'search_range': 'gain [0.8, 1.2], bias [0, 50]',
    'grad_steps': 50,
    'grad_lr': 0.01,
}
```

## Measurement

```python
# Option A — PWM benchmark (ground truth → PSNR/SSIM)
import h5py
with h5py.File('spc_public.h5', 'r') as f:   # GCS: gs://pwm-benchmark-datasets/datasets/Benchmark/spc/public/
    y, x_true = f['y'][0], f['x_true'][0]

# Option B — your data
import numpy as np
y      = np.load('your_measurement.npy').astype('float32')  # photon counts (T × H × W, uint16)
x_true = None   # provide for PSNR/SSIM
```

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.spc.solvers import run_solver
from pwm_core.mismatch.operators import spc_calibrate_gain_bias
from pwm_core.utils.metrics import compute_psnr, compute_ssim

# 1. Baseline — reconstruct WITHOUT mismatch correction
x_wrong = run_solver('opine_net', y, cfg={})

# 2. Calibrate mismatch parameter
gain, bias = spc_calibrate_gain_bias(y)

# 3. Reconstruct WITH correction
cfg = {"gain": float(gain), "bias": float(bias)}
cfg.update({})
x_corrected = run_solver('opine_net', y, cfg=cfg)

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
plt.tight_layout(); plt.savefig('spc_opine_net_corrected.png'); plt.show()
```

## Benchmark

`Single-Pixel Camera (SPC) — OPINE-Net+ + gradient` corresponds to the leaderboard entry at
`https://pwm.platformai.org/benchmark/spc` (mismatch correction tier).
