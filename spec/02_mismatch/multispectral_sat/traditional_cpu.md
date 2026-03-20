# Multispectral Satellite Imaging — RDA [proxy] + Gradient Mismatch Correction

**Device**: CPU  **Input**: radiance (H × W × bands, float32)
**Reference**: Richardson 1972, JOSA


## Mismatch

| Parameter | Range | Typical | Gradient Correction |
|-----------|-------|---------|---------------------|
| operator model parameter | `modality-dependent` | calibration-dependent | Grid search on reconstruction quality metric; gradient refines optimal parameter |

## Correction Parameters

```python
mismatch_cfg = {
    'operator_model_parameter': None,  # None = auto-calibrate
    'search_range': 'modality-dependent',
    'grad_steps': 50,
    'grad_lr': 0.01,
}
```

## Measurement

```python
# Option A — PWM benchmark (ground truth → PSNR/SSIM)
import h5py
with h5py.File('multispectral_sat_public.h5', 'r') as f:   # GCS: gs://pwm-benchmark-datasets/datasets/Benchmark/multispectral_sat/public/
    y, x_true = f['y'][0], f['x_true'][0]

# Option B — your data
import numpy as np
y      = np.load('your_measurement.npy').astype('float32')  # radiance (H × W × bands, float32)
x_true = None   # provide for PSNR/SSIM
```

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.multispectral_sat.solvers import run_solver

from pwm_core.utils.metrics import compute_psnr, compute_ssim

# 1. Baseline — reconstruct WITHOUT mismatch correction
x_wrong = run_solver('traditional_cpu', y, cfg={})

# 2. Calibrate mismatch parameter
# Use grid search over mismatch parameter

# 3. Reconstruct WITH correction
cfg = {"mismatch_param": None}  # None = auto-estimate
cfg.update({})
x_corrected = run_solver('traditional_cpu', y, cfg=cfg)

# 4. Evaluate
if x_true is not None:
    print(f"No correction  PSNR {compute_psnr(x_true, x_wrong):.2f} dB  "
          f"SSIM {compute_ssim(x_true, x_wrong):.4f}")
    print(f"Corrected      PSNR {compute_psnr(x_true, x_corrected):.2f} dB  "
          f"SSIM {compute_ssim(x_true, x_corrected):.4f}")

x = x_corrected
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
plt.tight_layout(); plt.savefig('multispectral_sat_traditional_cpu_corrected.png'); plt.show()
```

## Benchmark

`Multispectral Satellite Imaging — RDA [proxy] + gradient` corresponds to the leaderboard entry at
`https://pwm.platformai.org/benchmark/multispectral_sat` (mismatch correction tier).
