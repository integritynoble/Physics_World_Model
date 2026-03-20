# Ultrasound B-mode Imaging — US-CNN (DnCNN denoise) + Gradient Mismatch Correction

**Device**: GPU  **Input**: RF data (elements × samples, float32)
**Reference**: Zhang et al. 2017, IEEE TIP


## Mismatch

| Parameter | Range | Typical | Gradient Correction |
|-----------|-------|---------|---------------------|
| Speed of sound (m/s) | `[1400, 1600] m/s` | ±30 m/s tissue variation | Coherence-based SoS estimation; gradient maximizes beamformed coherence |

## Correction Parameters

```python
mismatch_cfg = {
    'speed_of_sound': None,  # None = auto-calibrate
    'search_range': '[1400, 1600] m/s',
    'grad_steps': 50,
    'grad_lr': 0.01,
}
```

## Measurement

```python
# Option A — PWM benchmark (ground truth → PSNR/SSIM)
import h5py
with h5py.File('ultrasound_public.h5', 'r') as f:   # GCS: gs://pwm-benchmark-datasets/datasets/Benchmark/ultrasound/public/
    y, x_true = f['y'][0], f['x_true'][0]

# Option B — your data
import numpy as np
y      = np.load('your_measurement.npy').astype('float32')  # RF data (elements × samples, float32)
x_true = None   # provide for PSNR/SSIM
```

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.ultrasound.solvers import run_solver
from pwm_core.mismatch.operators import ultrasound_calibrate_sos
from pwm_core.utils.metrics import compute_psnr, compute_ssim

# 1. Baseline — reconstruct WITHOUT mismatch correction
x_wrong = run_solver('small_gpu', y, cfg={})

# 2. Calibrate mismatch parameter
sos = ultrasound_calibrate_sos(rf_data, sos_range=[1400, 1600])

# 3. Reconstruct WITH correction
cfg = {"c0": float(sos)}
cfg.update({})
x_corrected = run_solver('small_gpu', y, cfg=cfg)

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
plt.tight_layout(); plt.savefig('ultrasound_small_gpu_corrected.png'); plt.show()
```

## Benchmark

`Ultrasound B-mode Imaging — US-CNN (DnCNN denoise) + gradient` corresponds to the leaderboard entry at
`https://pwm.platformai.org/benchmark/ultrasound` (mismatch correction tier).
