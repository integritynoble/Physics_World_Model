# Coded Aperture Snapshot Spectral Imaging (CASSI) — CST-L-Plus + Gradient Mismatch Correction

**Device**: GPU  **Input**: coded snapshot (H × W, float32)
**Reference**: Cai et al., ECCV 2022 — 36.1 dB on KAIST
**PSNR**: ~36.1 dB

## Mismatch

| Parameter | Range | Typical | Gradient Correction |
|-----------|-------|---------|---------------------|
| Dispersion step (px) | `[1, 5] px` | ±0.5 px drift | Sparsity-based grid search; gradient refines sub-pixel step |

## Correction Parameters

```python
mismatch_cfg = {
    'dispersion_step': None,  # None = auto-calibrate
    'search_range': '[1, 5] px',
    'grad_steps': 50,
    'grad_lr': 0.01,
}
```

## Measurement

```python
# Option A — PWM benchmark (ground truth → PSNR/SSIM)
import h5py
with h5py.File('cassi_public.h5', 'r') as f:   # GCS: gs://pwm-benchmark-datasets/datasets/Benchmark/cassi/public/
    y, x_true = f['y'][0], f['x_true'][0]

# Option B — your data
import numpy as np
y      = np.load('your_measurement.npy').astype('float32')  # coded snapshot (H × W, float32)
x_true = None   # provide for PSNR/SSIM
```

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.cassi.solvers import run_solver
from pwm_core.mismatch.operators import cassi_calibrate_step
from pwm_core.utils.metrics import compute_psnr, compute_ssim

# 1. Baseline — reconstruct WITHOUT mismatch correction
x_wrong = run_solver('cst_l_plus', y, cfg={'model_key': 'cst_l_plus'})

# 2. Calibrate mismatch parameter
disp_step = cassi_calibrate_step(y, step_range=[1, 5], n_steps=20)

# 3. Reconstruct WITH correction
cfg = {"disp_step": int(round(disp_step))}
cfg.update({'model_key': 'cst_l_plus'})
x_corrected = run_solver('cst_l_plus', y, cfg=cfg)

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
plt.tight_layout(); plt.savefig('cassi_cst_l_plus_corrected.png'); plt.show()
```

## Benchmark

`Coded Aperture Snapshot Spectral Imaging (CASSI) — CST-L-Plus + gradient` corresponds to the leaderboard entry at
`https://pwm.platformai.org/benchmark/cassi` (mismatch correction tier).
