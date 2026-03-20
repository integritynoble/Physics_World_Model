# Passive Microwave Radiometry — PnP-FISTA (NLM) + Gradient Mismatch Correction

**Device**: CPU  **Input**: brightness T (H × W × ch, float32)
**Reference**: Beck & Teboulle 2009 + PnP


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
with h5py.File('passive_microwave_public.h5', 'r') as f:   # GCS: gs://pwm-benchmark-datasets/datasets/Benchmark/passive_microwave/public/
    y, x_true = f['y'][0], f['x_true'][0]

# Option B — your data
import numpy as np
y      = np.load('your_measurement.npy').astype('float32')  # brightness T (H × W × ch, float32)
x_true = None   # provide for PSNR/SSIM
```

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.passive_microwave.solvers import run_solver

from pwm_core.utils.metrics import compute_psnr, compute_ssim

# 1. Baseline — reconstruct WITHOUT mismatch correction
x_wrong = run_solver('pnp_fista_nlm', y, cfg={'iters': 20, 'sigma': 0.05, 'mu': 0.5})

# 2. Calibrate mismatch parameter
# Use grid search over mismatch parameter

# 3. Reconstruct WITH correction
cfg = {"mismatch_param": None}  # None = auto-estimate
cfg.update({'iters': 20, 'sigma': 0.05, 'mu': 0.5})
x_corrected = run_solver('pnp_fista_nlm', y, cfg=cfg)

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
plt.tight_layout(); plt.savefig('passive_microwave_pnp_fista_nlm_corrected.png'); plt.show()
```

## Benchmark

`Passive Microwave Radiometry — PnP-FISTA (NLM) + gradient` corresponds to the leaderboard entry at
`https://pwm.platformai.org/benchmark/passive_microwave` (mismatch correction tier).
